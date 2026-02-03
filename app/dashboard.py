import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import sys
import os

# Fix path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils import load_config

# --- CONFIGURATION ---
st.set_page_config(page_title="Autoscaling Live Demo", layout="wide")
CONFIG = load_config()
API_URL = f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}"

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned/data_1m.csv')
    return df

def call_api(row, forecast_val):
    payload = {
        "timestamp_minute": int(row.name),
        "current_load": float(row['requests']),
        "forecast_load": float(forecast_val),
        "hour_of_day": int(row['hour_of_day'])
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
speed = st.sidebar.slider("Simulation Speed (ms)", 10, 500, 100)
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
    metric_dropped = col3.empty()
    metric_status = col4.empty()
    
    chart_placeholder = st.empty()
    
    # State Lists
    history_time = []
    history_load = []
    history_capacity = []
    history_servers = []
    history_dropped = []
    total_cost = 0.0
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
    else:
         history_buffer = [0.0] * history_window_size

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
        if len(history_buffer) > history_window_size:
            history_buffer.pop(0)
            
        # Call Forecast Endpoint
        forecast_payload = {
            "timestamp_minute": int(row.name),
            "recent_history": history_buffer
        }
        
        try:
            resp_forecast = requests.post(f"{API_URL}/forecast", json=forecast_payload)
            if resp_forecast.status_code == 200:
                forecast_val = resp_forecast.json()['forecast_load']
            else:
                st.warning(f"Forecast API Error: {resp_forecast.status_code}")
                forecast_val = current_load # Fallback
        except Exception as e:
            st.error(f"API Connection Error: {e}")
            break
        
        # 2. Call Decision API (Autoscaler)
        # We pass the Forecast we just got
        decision = call_api(row, forecast_val)
        if not decision:
            break
            
        # 3. Update Metrics
        total_cost += decision['cost_infra'] + decision['cost_sla']
        dropped = decision['details']['dropped']
        total_dropped += dropped
        
        metric_servers.metric("Active Servers", f"{decision['servers']}", delta=decision['action'])
        metric_cost.metric("Total Cost", f"${total_cost:,.2f}")
        metric_dropped.metric("Dropped Req", f"{total_dropped}", delta_color="inverse")
        

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

        # 4. Update Charts
        history_time.append(i)
        history_load.append(row['requests'])
        history_capacity.append(decision['details']['capacity'])
        history_servers.append(decision['servers'])
        # Store dropped requests for visualization (None if no drop)
        history_dropped.append(row['requests'] if dropped > 0 else None)
        
        # --- SLIDING WINDOW LOGIC (Optimize Performance) ---
        # Only show last 60 minutes in the live chart to avoid browser lag
        window_size = 60
        if len(history_time) > window_size:
            plot_time = history_time[-window_size:]
            plot_load = history_load[-window_size:]
            plot_capacity = history_capacity[-window_size:]
            plot_dropped = history_dropped[-window_size:]
        else:
            plot_time = history_time
            plot_load = history_load
            plot_capacity = history_capacity
            plot_dropped = history_dropped
        
        fig = go.Figure()
        
        # Real Capacity Area
        fig.add_trace(go.Scatter(
            x=plot_time, y=plot_capacity,
            fill='tozeroy', mode='none',
            name='Capacity',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        # Actual Load
        fig.add_trace(go.Scatter(
            x=plot_time, y=plot_load,
            mode='lines', name='Actual Load',
            line=dict(color='blue')
        ))
        
        # Dropped Requests (Persistent in Window)
        fig.add_trace(go.Scatter(
            x=plot_time, y=plot_dropped,
            mode='markers', name='Dropped',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        # Servers (Secondary Y-axis could be useful, or just overlay capacity)
        # User requested: "số lượng server (đường màu đỏ)"
        # But servers * capacity = capacity line. Let's just stick to Capacity for visual clarity on "Dropped"
        
        # Highlight Drops


        fig.update_layout(
            title="Load vs Capacity",
            xaxis_title="Simulation Minute",
            yaxis_title="Requests / Minute",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        # Streamlit 1.40+ compatibility: use_container_width is deprecated
        try:
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        except:
             # Fallback if new version enforces 'width' kwarg, though the warning usually allows the old one to work.
             # The user log says "Please replace...", implying it still works but prints warning.
             # To silence warning:
             # st.plotly_chart(fig, width="stretch") # Only valid in newest.
             # Let's try the safest "use_container_width=True" but ignore warning?
             # User specifically asked to FIX it.
             # Warning: For use_container_width=True, use width='stretch'.
             chart_placeholder.plotly_chart(fig, width="stretch")
        
        # 5. Sleep
        time.sleep(speed / 1000)

    st.success("Simulation Complete!")
    
    st.divider()
    st.subheader("📊 Full Session Summary")
    
    # --- FULL HISTORY CHART ---
    fig_full = go.Figure()
    
    fig_full.add_trace(go.Scatter(
        x=history_time, y=history_capacity,
        fill='tozeroy', mode='none',
        name='Capacity',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    fig_full.add_trace(go.Scatter(
        x=history_time, y=history_load,
        mode='lines', name='Actual Load',
        line=dict(color='blue')
    ))
    
    fig_full.add_trace(go.Scatter(
        x=history_time, y=history_dropped,
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
