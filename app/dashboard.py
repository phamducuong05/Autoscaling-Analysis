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

# --- CONFIGURATION ---
from src.data_loader import load_and_process_logs, resample_traffic
from src.features import add_features

# --- CONFIGURATION ---
st.set_page_config(page_title="Autoscaling Live Demo", layout="wide")
CONFIG = load_config()
API_URL = f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}"
TEST_DATA_PATH = "data/raw/test.txt"

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    """
    Load raw test data, resample to 5-minute intervals, and add features.
    """
    if not os.path.exists(TEST_DATA_PATH):
        st.error(f"File not found: {TEST_DATA_PATH}")
        return pd.DataFrame()
        
    with st.spinner('Processing raw logs (Resampling to 5min)...'):
        # 1. Parse Raw Logs
        df_raw = load_and_process_logs([TEST_DATA_PATH])
        
        # 2. Resample to 5 Minutes
        df_resampled = resample_traffic(df_raw, window='5min')
        
        # 3. Add Features (Cyclic Time, Weekend, etc.)
        df_features = add_features(df_resampled, frequency='5m')
        
        # Reset index to make timestamp a column for iteration
        df_features = df_features.reset_index()
        
        # Create step index for loop
        df_features['step_index'] = range(len(df_features))
        
        return df_features

def call_api(row, forecast_val, sigma, cv):
    payload = {
        "timestamp_minute": int(row.name),
        "current_load": float(row['requests']),
        "forecast_load": float(forecast_val),
        "hour_of_day": int(row['hour_of_day']),
        "sigma": float(sigma),
        "cv": float(cv)
    }
    try:
        resp = requests.post(f"{API_URL}/recommend-scaling", json=payload)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: Is the API running? {e}")
        return None

# --- MAIN UI ---
st.title("⚡ Autoscaling Decision Engine - Live Simulation")
st.markdown("### Real-time Traffic & Resource Management System")

# Sidebar
st.sidebar.header("🕹️ Simulation Controls")
speed = st.sidebar.slider("Simulation Speed (ms)", 500, 5000, 1000)
# Slice data for demo (e.g., specific event)
start_idx = st.sidebar.number_input("Start Minute", 0, 100000, 42000) # Pick a busy time
duration = st.sidebar.number_input("Duration (Minutes)", 60, 1000, 200)

if st.sidebar.button("🚀 Start Simulation"):
    
    # Init Data
    df = load_data()
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
    history_window_size = 30 # Send last 30 mins to API
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
        
        try:
            logger.info(f"📤 Sending Forecast Request for Step {row['step_index']} (Min {row['step_index']*5})")
            resp_forecast = requests.post(f"{API_URL}/forecast", json=forecast_payload)
            if resp_forecast.status_code == 200:
                data = resp_forecast.json()
                forecast_val = data['forecast_load']
                sigma = data.get('sigma', 0.0) # Gen 2
                cv = data.get('cv', 0.0)       # Gen 2
                logger.info(f"   📥 Received Forecast: {forecast_val}, Sigma: {sigma:.2f}, CV: {cv:.3f}")
            else:
                st.warning(f"Forecast API Error: {resp_forecast.status_code}")
                forecast_val = current_load 
                sigma, cv = 0.0, 0.0
        except Exception as e:
            st.error(f"API Connection Error: {e}")
            break
        
        # 2. Call Decision API (Autoscaler)
        decision = call_api(row, forecast_val, sigma, cv)
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
        
        # Confidence Interval (Upper/Lower) - 2 Sigma (~95%)
        upper_bound = forecast_val + (2 * sigma)
        lower_bound = max(0, forecast_val - (2 * sigma))
        upper_history.append(upper_bound)
        lower_history.append(lower_bound)

        # Draw Real-time Chart
        with chart_placeholder.container():
            fig = go.Figure()
            
            # SLIDING WINDOW LOGIC (Optimize Performance)
            # Only show last 50 points
            window = 50
            if len(time_history) > window:
                plot_time = time_history[-window:]
                plot_load = load_history[-window:]
                plot_capacity = capacity_history[-window:]
                plot_forecast = forecast_history[-window:]
                plot_upper = upper_history[-window:]
                plot_lower = lower_history[-window:]
            else:
                plot_time = time_history
                plot_load = load_history
                plot_capacity = capacity_history
                plot_forecast = forecast_history
                plot_upper = upper_history
                plot_lower = lower_history

            # 1. Confidence Band (Shaded)
            # We create a closed shape by concatenating upper and reversed lower bound
            fig.add_trace(go.Scatter(
                x=plot_time + plot_time[::-1],
                y=plot_upper + plot_lower[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Confidence Interval (2σ)'
            ))
            
            # 2. System Capacity
            fig.add_trace(go.Scatter(
                x=plot_time, y=plot_capacity,
                mode='lines', name='System Capacity',
                line=dict(color='#2ca02c', width=2, dash='dash')
            ))
            
            # 3. AI Forecast
            fig.add_trace(go.Scatter(
                x=plot_time, y=plot_forecast,
                mode='lines', name='AI Forecast',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            # 4. Actual Load
            fig.add_trace(go.Scatter(
                x=plot_time, y=plot_load,
                mode='lines', name='Actual Load',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Real-time Autoscaling Monitor (Gen 2)",
                xaxis_title="Time (Minutes)",
                yaxis_title="Requests / Capacity",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Streamlit 1.40+ compatibility
            st.plotly_chart(fig, use_container_width=True)
        
        # 5. Sleep
        time.sleep(speed / 1000)

    st.success("Simulation Complete!")
    
    st.divider()
    st.subheader("📊 Full Session Summary")
    
    # --- FULL HISTORY CHART ---
    fig_full = go.Figure()
    
    fig_full.add_trace(go.Scatter(
        x=time_history, y=capacity_history,
        fill='tozeroy', mode='none',
        name='Capacity',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    fig_full.add_trace(go.Scatter(
        x=time_history, y=load_history,
        mode='lines', name='Actual Load',
        line=dict(color='blue')
    ))
    
    fig_full.add_trace(go.Scatter(
        x=time_history, y=dropped_history,
        mode='markers', name='Dropped',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig_full.update_layout(
        title="Comprehensive Load Analysis (Full Duration)",
        xaxis_title="Simulation Minute",
        yaxis_title="Requests / Minute",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_full, width="stretch")
