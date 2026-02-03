import pandas as pd
import os

def load_simulation_data(actual_data_path):
    """
    Loads ONLY the Ground Truth (Actual) traffic data.
    This serves as the 'Problem Generator' for the Dashboard Simulator.
    The 'Forecast' will be generated On-The-Fly by the API model.
    
    Args:
        actual_data_path (str): Path to CSV containing 'timestamp', 'requests', 'hour_of_day'.
    
    Returns:
        pd.DataFrame: DataFrame with ['timestamp', 'hour_of_day', 'load']
    """
    if not os.path.exists(actual_data_path):
        raise FileNotFoundError(f"Actual data file not found: {actual_data_path}")
    
    df = pd.read_csv(actual_data_path)
    
    # Validation
    if 'requests' not in df.columns:
        raise ValueError("Actual data must have a 'requests' column.")
        
    if 'load' not in df.columns:
        df['load'] = df['requests']

    # Select columns needed for the simulation "Problem"
    required_cols = ['timestamp', 'hour_of_day', 'load']
    
    # Return strict columns
    return df[required_cols]
