"""
Feature Engineering Module
Extracted from api.py (lines 53-101) - EXACT copy, no changes
"""
import numpy as np
import pandas as pd
import torch
from .constants import FEATURE_NAMES

def prepare_features(req, scaler_features, seq_len=12):
    """
    Combine separate feature lists into a single input tensor (1, 12, 5).
    Features: [requests_target, error_rate, hour_sin, hour_cos, is_weekend]
    
    This is EXACT copy from api.py lines 53-101
    NO changes to logic
    """
    # 1. Combine parallel lists into columns
    # We expect the dashboard to send lists of same length.
    # If not, we truncate to the shortest length.
    min_len = min(
        len(req.recent_history), 
        len(req.error_history) if req.error_history else len(req.recent_history),
        len(req.hour_sin_history) if req.hour_sin_history else len(req.recent_history)
    )
    
    # If histories are empty, return None
    if min_len == 0:
        return None

    # Slice to last 12 (Sequence Length)
    # seq_len = 12  # Now passed as parameter
    
    # Extract raw vectors
    # Note: Dashboard sends 'recent_history' (load) and buffers.
    # We iterate backwards or just slice the end.
    
    # Helper to safe-get last N
    def get_last_n(lst, n):
        if not lst: return [0.0] * n
        return lst[-n:] if len(lst) >= n else [0.0] * (n - len(lst)) + lst

    f1 = get_last_n(req.recent_history, seq_len) # Load
    f2 = get_last_n(req.error_history, seq_len)  # Error
    f3 = get_last_n(req.hour_sin_history, seq_len)
    f4 = get_last_n(req.hour_cos_history, seq_len)
    f5 = get_last_n(req.is_weekend_history, seq_len)
    
    # Stack columns: shape (12, 5)
    input_data = np.column_stack((f1, f2, f3, f4, f5))
    
    # 2. Scale
    if scaler_features:
        # Fix UserWarning: Use DataFrame with correct feature names
        input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)
        input_scaled = scaler_features.transform(input_df)
    else:
        # Fallback if scaler missing (should not happen if model loaded)
        return None
        
    # 3. Tensor conversion
    tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0) # (1, 12, 5)
    return tensor
