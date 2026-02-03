"""
Data preparation and preprocessing utilities
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series windowing
    
    Converts (X, y) into sequences of length sequence_length
    """
    
    def __init__(self, X, y, sequence_length=12):
        """
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target array of shape (n_samples,)
        sequence_length : int
            Length of sliding window
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, i):
        return (
            self.X[i : i + self.sequence_length],
            self.y[i + self.sequence_length]
        )


def select_features_lstm(df, n_estimators=50, max_depth=5):
    """
    Select important features for LSTM using Random Forest
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    n_estimators : int
        Number of trees in RandomForest
    max_depth : int
        Max depth of trees
        
    Returns:
    --------
    pd.DataFrame : DataFrame with selected features
    """
    potential_cols = ['error_rate', 'hour_sin', 'hour_cos', 'is_weekend']
    
    X_check = df[potential_cols]
    y_check = df['requests_target']
    
    # Train RF for feature importance
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_check, y_check)
    
    # Get importances
    importances = pd.Series(
        rf.feature_importances_,
        index=potential_cols
    ).sort_values(ascending=False)
    
    print("Feature Importances:")
    print(importances)
    
    # Plot
    plt.figure(figsize=(10, 4))
    importances.plot(kind='barh', title='Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # Select features for LSTM
    selected_features = ['requests_target', 'error_rate', 'hour_sin', 'hour_cos', 'is_weekend']
    print(f"\n→ Features selected for LSTM: {selected_features}")
    
    return df[selected_features]


def prepare_data_pipeline(df, sequence_length=12, batch_size=32, 
                         train_ratio=0.6, val_ratio=0.2):
    """
    Complete data preparation pipeline
    
    Steps:
    1. Split into train/validation/test
    2. Fit scalers on training data
    3. Scale all splits
    4. Create windowed datasets
    5. Create DataLoaders
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features (index should be datetime)
    sequence_length : int
        Window size for LSTM (default: 12)
    batch_size : int
        Batch size for DataLoader (default: 32)
    train_ratio : float
        Proportion for training (default: 0.6)
    val_ratio : float
        Proportion for validation (default: 0.2)
        
    Returns:
    --------
    tuple : (train_loader, val_loader, test_loader, scaler_target, scaler_features)
    """
    # Option 1: Time-based split (if you have datetime index)
    # Use cutoff dates if available
    try:
        # Try to split by date (e.g., 1995-08-16)
        train_cutoff = '1995-08-16 23:59:59'
        val_cutoff = '1995-08-22 23:59:59'
        
        train_df = df[df.index <= train_cutoff]
        val_df = df[(df.index > train_cutoff) & (df.index <= val_cutoff)]
        test_df = df[df.index > val_cutoff]
    except:
        # Fallback to random split
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
    
    print(f"Train size: {len(train_df)} | Validation size: {len(val_df)} | Test size: {len(test_df)}")
    
    # Initialize scalers
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    # Fit on training data only
    scaler_features.fit(train_df)
    scaler_target.fit(train_df[['requests_target']])
    
    # Transform all splits
    train_scaled = scaler_features.transform(train_df)
    val_scaled = scaler_features.transform(val_df)
    test_scaled = scaler_features.transform(test_df)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_scaled, train_scaled[:, 0], sequence_length)
    val_dataset = TimeSeriesDataset(val_scaled, val_scaled[:, 0], sequence_length)
    test_dataset = TimeSeriesDataset(test_scaled, test_scaled[:, 0], sequence_length)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler_target, scaler_features


def create_data_buffer(df, frequency='5m', window_size=12, required_cols=None):
    """
    Create sliding window buffer from DataFrame
    
    Used for real-time prediction with streaming data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features (datetime index)
    frequency : str
        Data frequency ('1m', '5m', '15m', etc)
    window_size : int
        Window size (default: 12 for 5m = 60 min)
    required_cols : list
        Required columns (auto-detected if None)
        
    Returns:
    --------
    dict : Buffer with data, features, metadata
    """
    if required_cols is None:
        required_cols = ['requests_target', 'error_rate', 'hour_sin', 'hour_cos', 'is_weekend']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    buffer = {
        'data': df[required_cols].tail(window_size).values,  # Shape: (window_size, n_features)
        'features': required_cols,
        'timestamp': df.index[-1],
        'frequency': frequency,
        'window_size': window_size
    }
    
    return buffer


def update_buffer(buffer, new_row_dict):
    """
    Update buffer with new data (sliding window)
    
    Removes oldest row, adds newest row
    
    Parameters:
    -----------
    buffer : dict
        Buffer from create_data_buffer
    new_row_dict : dict
        New row data as dict: {'requests_target': val, ...}
        
    Returns:
    --------
    dict : Updated buffer
    """
    from datetime import datetime
    
    features = buffer['features']
    new_values = np.array([new_row_dict[f] for f in features])
    
    # Shift: remove oldest, add newest
    buffer['data'] = np.vstack([buffer['data'][1:], new_values])
    buffer['timestamp'] = datetime.now()
    
    return buffer


def validate_data_shapes(df, expected_features=None):
    """
    Validate dataframe structure for LSTM
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    expected_features : list
        Expected columns
        
    Returns:
    --------
    tuple : (is_valid: bool, message: str)
    """
    if expected_features is None:
        expected_features = ['requests_target', 'error_rate', 'hour_sin', 'hour_cos', 'is_weekend']
    
    # Check columns
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    
    # Check for NaN
    if df[expected_features].isnull().any().any():
        null_cols = df[expected_features].columns[df[expected_features].isnull().any()].tolist()
        return False, f"NaN values found in: {null_cols}"
    
    # Check shape
    if len(df) < 12:
        return False, f"Not enough data: {len(df)} rows < 12 (minimum window size)"
    
    return True, "Validation passed"
