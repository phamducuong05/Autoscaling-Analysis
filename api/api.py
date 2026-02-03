import logging
import sys
import os
import joblib
import numpy as np  
import torch        
from fastapi import FastAPI
app = FastAPI()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [API] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.autoscaler import Autoscaler
from src.utils import load_config
from src.lstm.models.lstm_model import LSTMModel
from .schema import (
    ForecastRequest, PredictionRequest, ForecastResponse, 
    ScalingRequest, ScalingResponse
)

# --- GLOBAL STATE ---
scaler = Autoscaler()
residuals_buffer = [] # Store last 15 residuals
last_forecast_val = 0.0 # Store prediction from previous step
MAX_RESIDUALS = 15

logger.info("✅ Autoscaler Engine Initialized (Stateful)")

# --- LOAD AI MODEL ---
try:
    MODEL_DIR = "models_export"
    scaler_features = joblib.load(f"{MODEL_DIR}/scaler_features.pkl")
    scaler_target = joblib.load(f"{MODEL_DIR}/scaler_target.pkl")
    
    # Initialize Model with parameters from metadata.json
    # input_size=5, hidden_size=32, num_layers=1
    real_model = LSTMModel(input_size=5, hidden_size=32, num_layers=1)
    real_model.load_state_dict(torch.load(f"{MODEL_DIR}/model_weights.pth", map_location='cpu'))
    real_model.eval()
    logger.info("✅ LSTM Model Loaded Successfully")
except Exception as e:
    logger.warning(f"⚠️ Warning: Could not load AI Model. Using heuristic fallback. Error: {e}")
    real_model = None

def prepare_features(req: PredictionRequest):
    """
    Combine separate feature lists into a single input tensor (1, 12, 5).
    Features: [requests_target, error_rate, hour_sin, hour_cos, is_weekend]
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
    seq_len = 12
    
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
    if 'scaler_features' in globals() and scaler_features:
        input_scaled = scaler_features.transform(input_data)
    else:
        # Fallback if scaler missing (should not happen if model loaded)
        return None
        
    # 3. Tensor conversion
    tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0) # (1, 12, 5)
    return tensor

@app.post("/forecast", response_model=ForecastResponse)
def get_forecast_live(req: PredictionRequest):
    global last_forecast_val, residuals_buffer
    
    prediction = 0.0
    sigma = 0.0
    cv = 0.0
    
    logger.info(f"🔮 Forecast Request for Minute {req.timestamp_minute}")
    
    # --- PHASE 2: ERROR MEMORY (Online Learning) ---
    if req.actual_load_current is not None:
        # We have the ACTUAL value for the CURRENT step.
        # We compare it with what we predicted in the PREVIOUS step (last_forecast_val).
        # Residual = |Actual - Forecast|
        
        # Note: Handle startup case where last_forecast might be 0
        if last_forecast_val > 0:
            residual = abs(req.actual_load_current - last_forecast_val)
            residuals_buffer.append(residual)
            if len(residuals_buffer) > MAX_RESIDUALS:
                residuals_buffer.pop(0)
                
            # Calculate Statistics (Sigma & CV)
            if len(residuals_buffer) >= 3:
                sigma = np.std(residuals_buffer)
                avg_forecast = last_forecast_val # Approximation
                cv = sigma / (avg_forecast + 1e-9)
                
            logger.info(f"   📉 Error Tracking: Actual={req.actual_load_current}, Predicted={last_forecast_val}, Residual={residual:.2f}, CV={cv:.3f}")
    
    # Try AI Inference
    if real_model:
        try:
            input_tensor = prepare_features(req)
            if input_tensor is not None:
                logger.info(f"   Input Tensor Shape: {input_tensor.shape}")
                with torch.no_grad():
                    output_scaled = real_model(input_tensor)
                    output_scaled = output_scaled.cpu().numpy() # (1, 1)
                    
                # Inverse Scale Target
                output_val = scaler_target.inverse_transform(output_scaled)[0][0]
                prediction = float(output_val)
                logger.info(f"   ✅ Model Prediction: {prediction:.2f} (Scaled: {output_scaled[0][0]:.4f})")
            else:
                logger.warning("   ⚠️ Feature prep failed (empty history?), using fallback.")
                prediction = req.recent_history[-1] if req.recent_history else 0.0
                
        except Exception as e:
            logger.error(f"   ⚠️ Inference Error: {e}")
            prediction = req.recent_history[-1] if req.recent_history else 0.0
            
    else:
        # Heuristic Fallback (Mock)
        if req.recent_history:
            lag_1 = req.recent_history[-1]
            rolling_mean = sum(req.recent_history[-5:]) / 5 if len(req.recent_history) >= 5 else lag_1
            prediction = rolling_mean * 1.05 
            logger.info(f"   ℹ️  Mock Prediction: {prediction:.2f}")
        else:
            prediction = 0.0
            
    # Update state for next step
    last_forecast_val = prediction
        
    return {
        "timestamp_minute": req.timestamp_minute,
        "forecast_load": max(0.0, round(prediction, 2)),
        "sigma": float(sigma),
        "cv": float(cv)
    }

@app.post("/recommend-scaling", response_model=ScalingResponse)
def recommend_scaling(req: ScalingRequest):
    """
    The Core Autoscaling Logic.
    Receives Load & Forecast -> Returns Decision (Action/Servers).
    This endpoint is STATEFUL (updates stability window).
    """
    decision = scaler.decide_scale(
        timestamp_minute=req.timestamp_minute,
        current_load=req.current_load,
        forecast=req.forecast_load,
        hour=req.hour_of_day,
        sigma=req.sigma,
        cv=req.cv
    )
    
    return {
        "timestamp": decision['timestamp'],
        "servers": decision['servers'],
        "action": decision['action'],
        "cost_infra": decision['cost_infra'],
        "cost_sla": decision['cost_sla'],
        "is_ddos": decision['is_ddos'],
        "is_warming_up": decision['is_warming_up'],
        "dropped": decision.get('dropped', 0.0),
        "capacity": decision.get('capacity', 0.0), # Critical for Dashboard
        "raw_demand": decision.get('raw_demand', 0),
        "safety_factor": decision.get('safety_factor', 0.0),
        "details": decision # Pass full details for debugging
    }

if __name__ == "__main__":
    import uvicorn
    # Load host/port from config
    cfg = load_config()
    host = cfg.get('api', {}).get('host', '0.0.0.0')
    port = cfg.get('api', {}).get('port', 8000)
    
    uvicorn.run(app, host=host, port=port)
