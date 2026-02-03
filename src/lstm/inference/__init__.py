"""
Inference package initialization
"""
from .predictor import (
    validate_real_time_input,
    real_time_predict,
    batch_predict,
    PredictionBuffer
)

__all__ = [
    'validate_real_time_input',
    'real_time_predict',
    'batch_predict',
    'PredictionBuffer'
]
