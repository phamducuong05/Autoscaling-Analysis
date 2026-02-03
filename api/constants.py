"""
API Configuration Constants
Extracted from api.py - EXACT values, no changes
"""

# Model Configuration
MODEL_DIR = "models_export"
MODEL_INPUT_SIZE = 5
MODEL_HIDDEN_SIZE = 32
MODEL_NUM_LAYERS = 1

# Forecast Configuration
MAX_RESIDUALS = 15
SEQUENCE_LENGTH = 12
MIN_RESIDUALS_FOR_STATS = 3

# Feature Names (for documentation)
FEATURE_NAMES = [
    'requests_target',
    'error_rate', 
    'hour_sin',
    'hour_cos',
    'is_weekend'
]
