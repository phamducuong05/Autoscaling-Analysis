"""
Visualization Module
Extracted from dashboard.py (lines 250-314) - Chart creation logic
"""
import plotly.graph_objects as go
import streamlit as st

def create_realtime_chart(time_history, load_history, capacity_history, 
                         forecast_history, upper_history, lower_history, 
                         window=50):
    """
    Create real-time autoscaling chart with confidence interval
    
    This is EXACT copy from dashboard.py lines 250-314
    NO changes to logic
    
    Args:
        time_history: List of timestamps (minutes)
        load_history: List of actual load values
        capacity_history: List of capacity values
        forecast_history: List of forecast values
        upper_history: List of upper bound values
        lower_history: List of lower bound values
        window: Sliding window size (default 50)
        
    Returns:
        Plotly Figure object
    """
    # SLIDING WINDOW LOGIC (Optimize Performance)
    # Only show last 'window' points
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

    fig = go.Figure()
    
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
    
    return fig
