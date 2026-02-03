from fastapi import FastAPI, HTTPException
import random
import pandas as pd
from typing import Optional, Dict, Any
from src.optimization import Autoscaler
from src.utils import load_config
import joblib
import numpy as np 

MODEL_PATH = "models/my_forecast_model.pkl"
real_model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Autoscaling Analysis API",
    description="API for Autoscaling Decision Engine involved in Traffic Forecasting and Resource Optimization",
    version="1.0.0"
)

print("🚀 DEBUG: API RELOADED WITH WARMUP FIX 🚀")

# --- GLOBAL STATE ---
# We use a global instance to maintain the "Stability Window" (deque) across requests.
# In a real microservice, this might be stored in Redis or standard K8s HPA metrics.
scaler = Autoscaler()
print("✅ Autoscaler Engine Initialized (Stateful)")

# --- DATA MODELS ---
from .schema import (
    ForecastRequest, PredictionRequest, ForecastResponse, 
    ScalingRequest, ScalingResponse
)

def get_real_forecast(timestamp_minute, input_features):
    features = prepare_features(timestamp_minute) 
    prediction = real_model.predict(features)
    return prediction

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "system": "Autoscaling Decision Engine"}

@app.post("/forecast", response_model=ForecastResponse)
def get_forecast_live(req: PredictionRequest):
    if not req.recent_history:
        return {"timestamp_minute": req.timestamp_minute, "forecast_load": 0.0}
        
    # Extract features from history
    # Example: If model expects [lag_1, rolling_mean_5]
    lag_1 = req.recent_history[-1]
    rolling_mean = sum(req.recent_history[-5:]) / 5 if len(req.recent_history) >= 5 else lag_1
    
    # Prepare input vector for model
    # Note: Structure depends on your specific model.data
    # For now, we use a dummy logic or the real_model if loaded.
    
    try:
        # Check if real_model is actually a loaded object
        if 'real_model' in globals() and hasattr(real_model, 'predict'):
            # Example feature vector construction
            # features = [[lag_1, rolling_mean]] 
            # prediction = real_model.predict(features)[0]
            
            # Since we don't know the exact model signature yet, we use a placeholder:
            # prediction = real_model.predict([req.recent_history]) 
            pass 
            
        # --- PLACEHOLDER LOGIC FOR DEMO UNTIL MODEL SIGNATURE IS KNOWN ---
        # "Model performs calculation"
        # Let's say Model predicts a +5% trend
        prediction = rolling_mean * 1.05 
        
    except Exception as e:
        print(f"Model Inference Failed: {e}")
        prediction = lag_1 # Fallback
        
    return {
        "timestamp_minute": req.timestamp_minute,
        "forecast_load": round(prediction, 2)
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
        hour=req.hour_of_day
    )
    
    return {
        "timestamp": decision['timestamp'],
        "servers": decision['servers'],
        "action": decision['action'],
        "cost_infra": decision['cost_infra'],
        "cost_sla": decision['cost_sla'],
        "is_ddos": decision['is_ddos'],
        "is_warming_up": decision['is_warming_up'],
        "details": decision # Pass full details for debugging
    }

if __name__ == "__main__":
    import uvicorn
    # Load host/port from config
    cfg = load_config()
    host = cfg.get('api', {}).get('host', '0.0.0.0')
    port = cfg.get('api', {}).get('port', 8000)
    
    uvicorn.run(app, host=host, port=port)
