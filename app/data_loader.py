"""
Data Loading Module
Extracted from dashboard.py (lines 34-59) - EXACT copy
"""
import streamlit as st
import pandas as pd
import os
import sys

# Fix path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data_loader import load_and_process_logs, resample_traffic
from src.features import add_features

@st.cache_data
def load_data(test_data_path, resample_window='5min', frequency='5m'):
    """
    Load raw test data, resample to 5-minute intervals, and add features.
    
    This is EXACT copy from dashboard.py lines 34-59
    NO changes to logic
    
    Args:
        test_data_path: Path to test data file
        resample_window: Resampling window (default '5min')
        frequency: Frequency for features (default '5m')
        
    Returns:
        DataFrame with processed data
    """
    if not os.path.exists(test_data_path):
        st.error(f"File not found: {test_data_path}")
        return pd.DataFrame()
        
    with st.spinner('Processing raw logs (Resampling to 5min)...'):
        # 1. Parse Raw Logs
        df_raw = load_and_process_logs([test_data_path])
        
        # 2. Resample to 5 Minutes
        df_resampled = resample_traffic(df_raw, window=resample_window)
        
        # 3. Add Features (Cyclic Time, Weekend, etc.)
        df_features = add_features(df_resampled, frequency=frequency)
        
        # Reset index to make timestamp a column for iteration
        df_features = df_features.reset_index()
        
        # Create step index for loop
        df_features['step_index'] = range(len(df_features))
        
        return df_features
