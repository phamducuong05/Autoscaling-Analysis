from pydantic import BaseModel
from typing import Dict, Any

class ForecastRequest(BaseModel):
    timestamp_minute: int

class PredictionRequest(BaseModel):
    timestamp_minute: int
    recent_history: list[float] # Last N minutes of load data

class ForecastResponse(BaseModel):
    timestamp_minute: int
    forecast_load: float

class ScalingRequest(BaseModel):
    timestamp_minute: int
    current_load: float
    forecast_load: float
    hour_of_day: int

class ScalingResponse(BaseModel):
    timestamp: int
    servers: int
    action: str
    cost_infra: float
    cost_sla: float
    is_ddos: bool
    is_warming_up: bool  
    details: Dict[str, Any]
