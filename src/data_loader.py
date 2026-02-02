import pandas as pd
import os

def load_simulation_data(actual_data_path, forecast_data_path=None):
    """
    Loads and prepares data for the simulation.
    
    Args:
        actual_data_path (str): Path to the cleaned 1-minute ground truth data (data_1m.csv).
        forecast_data_path (str): Path to the forecast predictions csv. 
                                  If None, a naive SMA forecast is generated.
    
    Returns:
        pd.DataFrame: Merged dataframe with 'timestamp', 'hour_of_day', 'requests' (load), and 'forecast'.
    """
    if not os.path.exists(actual_data_path):
        raise FileNotFoundError(f"Actual data file not found: {actual_data_path}")

    # 1. Load Actual Data
    df_actual = pd.read_csv(actual_data_path)
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'requests', 'hour_of_day']
    for col in required_cols:
        if col not in df_actual.columns:
            raise ValueError(f"Actual data missing required column: {col}")

    # 2. Load or Generate Forecast
    if forecast_data_path and os.path.exists(forecast_data_path):
        print(f"Loading Forecast from model: {forecast_data_path}")
        df_forecast = pd.read_csv(forecast_data_path)
        
        # Expecting forecast file to have 'timestamp' and 'yhat' (or similar)
        # We assume standard timestamp matching via string or datetime conversion could be needed.
        # For simplicity in this competition context, we assume row-alignment or perfect timestamp match index.
        # Ideally: df_merged = pd.merge(df_actual, df_forecast, on='timestamp', how='left')
        
        # Placeholder for simple concantenation if indices align (common in competitions)
        # But let's try to be safe:
        if 'yhat' in df_forecast.columns:
            df_actual['forecast_load'] = df_forecast['yhat']
        elif 'forecast' in df_forecast.columns:
            df_actual['forecast_load'] = df_forecast['forecast']
        else:
            # Fallback if specific column name unknown, assume it's the second column or matching 'requests' length
            print("Warning: Unknown column name in forecast file. Generating naive forecast instead.")
            df_actual['forecast_load'] = df_actual['requests'].rolling(window=5).mean().shift(-1)
            
    else:
        # Generate Naive Forecast (SMA 5) if no external model provided yet
        print("No external forecast file provided. Generating Naive Baseline (SMA-5).")
        df_actual['forecast_load'] = df_actual['requests'].rolling(window=5).mean().shift(-1)
    
    # Fill NaN at edges
    df_actual['forecast_load'] = df_actual['forecast_load'].fillna(df_actual['requests'])
    
    # Select only columns needed for simulation
    simulation_df = df_actual[['timestamp', 'hour_of_day', 'requests', 'forecast_load']].copy()
    simulation_df.rename(columns={'requests': 'load'}, inplace=True)
    
    return simulation_df
