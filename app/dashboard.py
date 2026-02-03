import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import sys
import os
import logging

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [DASHBOARD] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Fix path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils import load_config

# Import new modules - use absolute imports for Streamlit
from app.constants import (
    HISTORY_WINDOW_SIZE, CHART_WINDOW_SIZE, DEFAULT_SPEED_MS,
    DEFAULT_START_IDX, DEFAULT_DURATION, TEST_DATA_PATH,
    RESAMPLE_WINDOW, FREQUENCY, SIGMA_MULTIPLIER
)
from app.api_client import AutoscalingAPIClient
from app.data_loader import load_data
from app.visualization import create_realtime_chart

# --- CONFIGURATION ---
st.set_page_config(page_title="Autoscaling Live Demo", layout="wide")
CONFIG = load_config()
API_URL = f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}"

# --- MAIN UI ---
st.title("⚡ Autoscaling Decision Engine - Live Simulation")
st.markdown("### Real-time Traffic & Resource Management System")

# Sidebar
st.sidebar.header("🕹️ Simulation Controls")
speed = st.sidebar.slider("Simulation Speed (ms)", 500, 5000, DEFAULT_SPEED_MS)
# Slice data for demo (e.g., specific event)
start_idx = st.sidebar.number_input("Start Minute", 0, 100000, DEFAULT_START_IDX) # Pick a busy time
duration = st.sidebar.number_input("Duration (Minutes)", 60, 1000, DEFAULT_DURATION)

# Initialize API Client
api_client = AutoscalingAPIClient(API_URL)

if st.sidebar.button("🚀 Start Simulation"):
    
    # Init Data - use config parameters
    df = load_data(TEST_DATA_PATH, RESAMPLE_WINDOW, FREQUENCY)
    # Simulate Forecast (Mocking it here or relying on API? 
    # The API has GET /forecast, but for simulator efficiency we can just pass noise here 
    # OR call the API for forecast too. User plan said: "Dùng tạm data từ tập train".
    # Let's verify API connectivity first.
    
    subset = df.iloc[start_idx : start_idx + duration]
    
    # Placeholders for Real-time UI
    col1, col2, col3, col4 = st.columns(4)
    metric_servers = col1.empty()
    metric_cost = col2.empty()
    metric_cv = col3.empty() # Changed from metric_dropped
    metric_status = col4.empty()
    
    chart_placeholder = st.empty()
    
    # State Lists
    time_history = []
    load_history = []
    capacity_history = []
    forecast_history = [] # New
    upper_history = []    # New (Confidence Interval)
    lower_history = []    # New
    dropped_history = []  # Fixed: Added missing list
    total_cost = 0.0
    total_requests = 0
    total_dropped = 0
    
    # --- SIMULATION LOOP ---
    # Prepare history buffer (simulate "system memory")
    # We need some initial history to start predicting.
    # Let's take the 10 minutes *before* the start_idx if possible, or pad with 0.
    history_window_size = HISTORY_WINDOW_SIZE # From config: 30
    history_buffer = [] 
    
    # Pre-fill history if possible
    if start_idx > history_window_size:
        history_buffer = df.iloc[start_idx-history_window_size : start_idx]['requests'].tolist()
        error_buffer = [0.0] * history_window_size # Assume 0 errors
        hour_sin_buffer = df.iloc[start_idx-history_window_size : start_idx]['hour_sin'].tolist()
        hour_cos_buffer = df.iloc[start_idx-history_window_size : start_idx]['hour_cos'].tolist()
        weekend_buffer = df.iloc[start_idx-history_window_size : start_idx]['is_weekend'].tolist()
    else:
         history_buffer = [0.0] * history_window_size
         error_buffer = [0.0] * history_window_size
         hour_sin_buffer = [0.0] * history_window_size
         hour_cos_buffer = [0.0] * history_window_size
         weekend_buffer = [0.0] * history_window_size

    for i, (idx, row) in enumerate(subset.iterrows()):
        
        # 1. Closed-Loop Forecast: Ask API "What is next?"
        # We send the recent history (including current moment? Or up to previous?)
        # Usuallly to predict T+1, we use data up to T.
        # But here 'row' is T. So let's append T to history, then ask for T+1?
        # Or ask for T using history T-1?
        # Let's assume we want to predict 'Future' to scale proactively.
        
        # Current thought:
        # At Time T, we observe Load T.
        # We want to know Forecast T+1 to Scale for T+1.
        
        # Append current load to history
        current_load = row['requests']
        history_buffer.append(float(current_load))
        error_buffer.append(0.0) # Placeholder for current step
        hour_sin_buffer.append(float(row['hour_sin']))
        hour_cos_buffer.append(float(row['hour_cos']))
        weekend_buffer.append(float(row['is_weekend']))
        
        if len(history_buffer) > history_window_size:
            history_buffer.pop(0)
            error_buffer.pop(0)
            hour_sin_buffer.pop(0)
            hour_cos_buffer.pop(0)
            weekend_buffer.pop(0)
            
        # Call Forecast Endpoint
        forecast_payload = {
            "timestamp_minute": int(row.name),
            "recent_history": history_buffer,
            "error_history": error_buffer,
            "hour_sin_history": hour_sin_buffer,
            "hour_cos_history": hour_cos_buffer,
            "is_weekend_history": weekend_buffer,
            "actual_load_current": float(current_load)
        }
        
        logger.info(f"📤 Sending Forecast Request for Step {row['step_index']} (Min {row['step_index']*5})")
        # Use api_client instead of direct requests
        data = api_client.get_forecast(forecast_payload)
        if data:
            forecast_val = data['forecast_load']
            sigma = data.get('sigma', 0.0) # Gen 2
            cv = data.get('cv', 0.0)       # Gen 2
            logger.info(f"   📥 Received Forecast: {forecast_val}, Sigma: {sigma:.2f}, CV: {cv:.3f}")
        else:
            # Fallback
            forecast_val = current_load 
            sigma, cv = 0.0, 0.0
        
        # 2. Call Decision API (Autoscaler) - use api_client
        scaling_payload = {
            "timestamp_minute": int(row.name),
            "current_load": float(row['requests']),
            "forecast_load": float(forecast_val),
            "hour_of_day": int(row['hour_of_day']),
            "sigma": float(sigma),
            "cv": float(cv)
        }
        decision = api_client.get_scaling_decision(scaling_payload)
        if not decision:
            break
        
        logger.info(f"   ⚖️  Scaling Decision: Servers={decision['servers']}, Action={decision['action']}")
            
        # 3. Update Metrics
        total_cost += decision['cost_infra'] + decision['cost_sla']
        total_requests += row['requests']
        dropped = decision['dropped']
        total_dropped += dropped
        
        # Update error history with actual observed error
        current_error_rate = min(1.0, dropped / (row['requests'] + 1e-9))
        error_buffer[-1] = current_error_rate
        
        # Track dropped history
        dropped_history.append(dropped if dropped > 0 else None)
        
        metric_servers.metric("Active Servers", f"{decision['servers']}", delta=decision['action'])
        metric_cost.metric("Total Cost ($)", f"{total_cost:.2f}")
        metric_cv.metric("AI Confidence (CV)", f"{cv:.3f}", delta_color="inverse")
        

        status_msg = "✅ Stable"
        if decision['is_ddos']:
            status_msg = "🛡️ DDoS Detected!"
        elif decision.get('is_warming_up', False):
            status_msg = "🔥 Warming Up..."
        elif decision['action'] == "HOLD (Hysteresis)":
            status_msg = "⚓ Hysteresis"
        else:
            status_msg = decision['action']
        metric_status.metric("System Status", status_msg)

        # 4. Update Plot Data
        time_history.append(row['step_index'] * 5)
        load_history.append(current_load)
        capacity_history.append(decision['capacity'])
        forecast_history.append(forecast_val)
        
        # Confidence Interval (Upper/Lower) - SIGMA_MULTIPLIER from config (2 Sigma ~95%)
        upper_bound = forecast_val + (SIGMA_MULTIPLIER * sigma)
        lower_bound = max(0, forecast_val - (SIGMA_MULTIPLIER * sigma))
        upper_history.append(upper_bound)
        lower_history.append(lower_bound)

        # Draw Real-time Chart - use visualization module
        with chart_placeholder.container():
            fig = create_realtime_chart(
                time_history, load_history, capacity_history,
                forecast_history, upper_history, lower_history,
                window=CHART_WINDOW_SIZE
            )
            
            # Streamlit 1.40+ compatibility
            st.plotly_chart(fig, use_container_width=True)
        
        # 5. Sleep
        time.sleep(speed / 1000)

    st.success("✅ Simulation Complete!")
    
    # --- PHẦN BỔ SUNG: BÁO CÁO CHI PHÍ & HIỆU SUẤT ---
    st.divider()
    st.subheader("📊 Full Session Executive Summary")
    
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Total Processed", f"{total_requests:,} reqs")
    s_col2.metric("Total Dropped", f"{total_dropped:,} reqs", delta=f"{(total_dropped/total_requests)*100:.2f}%" if total_requests > 0 else "0%", delta_color="inverse")
    s_col3.metric("Total Infrastructure Cost", f"${total_cost:.2f}")
    
    # Tính toán hiệu quả (Cost per 1k Requests)
    cost_per_k = (total_cost / total_requests * 1000) if total_requests > 0 else 0
    s_col4.metric("Efficiency", f"${cost_per_k:.4f} /1k req")

    # Hiển thị bảng dữ liệu chi tiết nếu cần
    with st.expander("📂 View Detailed Session Logs"):
        summary_df = pd.DataFrame({
            "Step": time_history,
            "Actual Load": load_history,
            "Capacity": capacity_history,
            "Dropped": [d if d is not None else 0 for d in dropped_history]
        })
        st.dataframe(summary_df, use_container_width=True)
