from pydantic import BaseModel
from typing import Dict, Any, Optional

class ForecastRequest(BaseModel):
    timestamp_minute: int

class PredictionRequest(BaseModel):
    timestamp_minute: int
    recent_history: list[float] # Last N minutes of load data
    error_history: list[float] = [] # Last N minutes of error rates (0.0 to 1.0)
    hour_sin_history: list[float] = [] 
    hour_cos_history: list[float] = []
    is_weekend_history: list[float] = []
    actual_load_current: Optional[float] = None # For feedback loop (Gen 2)

class ForecastResponse(BaseModel):
    timestamp_minute: int
    forecast_load: float
    sigma: float = 0.0 # Confidence Interval (Std Dev)
    cv: float = 0.0    # Coefficient of Variation

class ScalingRequest(BaseModel):
    timestamp_minute: int
    current_load: float
    forecast_load: float
    hour_of_day: int
    sigma: float = 0.0 # From Forecast
    cv: float = 0.0 # From Forecast

class ScalingResponse(BaseModel):
    timestamp: int
    servers: int
    action: str
    cost_infra: float
    cost_sla: float
    is_ddos: bool
    is_warming_up: bool  
    dropped: float = 0.0
    capacity: float = 0.0 # Restored for Dashboard plotting
    raw_demand: int = 0
    safety_factor: float = 0.0
    details: Dict[str, Any]
