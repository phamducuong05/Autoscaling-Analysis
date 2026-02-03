"""
Model Loading Module
Extracted from api.py (lines 38-51) - EXACT copy, no changes
"""
import logging
import joblib
import torch
from src.lstm.models.lstm_model import LSTMModel

logger = logging.getLogger(__name__)

def load_lstm_model(model_dir, input_size, hidden_size, num_layers):
    """
    Load LSTM Model and Scalers
    
    This is EXACT copy of loading logic from api.py lines 38-51
    NO changes to logic
    
    Returns:
        tuple: (model, scaler_features, scaler_target) or (None, None, None) if failed
    """
    try:
        scaler_features = joblib.load(f"{model_dir}/scaler_features.pkl")
        scaler_target = joblib.load(f"{model_dir}/scaler_target.pkl")
        
        # Initialize Model with parameters from metadata.json
        # input_size=5, hidden_size=32, num_layers=1
        real_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        real_model.load_state_dict(torch.load(f"{model_dir}/model_weights.pth", map_location='cpu'))
        real_model.eval()
        logger.info("✅ LSTM Model Loaded Successfully")
        
        return real_model, scaler_features, scaler_target
        
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not load AI Model. Using heuristic fallback. Error: {e}")
        return None, None, None
