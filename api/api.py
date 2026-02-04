import logging
import sys
import os
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
from .schema import (
    ForecastRequest, PredictionRequest, ForecastResponse, 
    ScalingRequest, ScalingResponse
)
from .constants import (
    MODEL_DIR, MODEL_INPUT_SIZE, MODEL_HIDDEN_SIZE, MODEL_NUM_LAYERS,
    MAX_RESIDUALS, SEQUENCE_LENGTH
)
from .model_loader import load_lstm_model
from .feature_engineering import prepare_features

# --- GLOBAL STATE ---
scaler = Autoscaler()
residuals_buffer = [] # Store last MAX_RESIDUALS residuals (from config)
last_forecast_val = 0.0 # Store prediction from previous step
# MAX_RESIDUALS is now imported from config

logger.info("✅ Autoscaler Engine Initialized (Stateful)")

# --- LOAD AI MODEL ---
# Extracted to model_loader.py - exact same logic
real_model, scaler_features, scaler_target = load_lstm_model(
    MODEL_DIR, MODEL_INPUT_SIZE, MODEL_HIDDEN_SIZE, MODEL_NUM_LAYERS
)

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
            input_tensor = prepare_features(req, scaler_features, SEQUENCE_LENGTH)
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