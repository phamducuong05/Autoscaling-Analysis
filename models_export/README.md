# LSTM Model Package

## Files:
- model_weights.pth: PyTorch model weights
- scaler_features.pkl: Features normalization scaler
- scaler_target.pkl: Target denormalization scaler
- model_metadata.json: Architecture & hyperparameters

## Usage:
```python
import torch
import pickle
import json
import numpy as np

# Load model
model = LSTMModel(...)
model.load_state_dict(torch.load('model_weights.pth'))

# Load scalers
with open('scaler_features.pkl', 'rb') as f:
    scaler_features = pickle.load(f)
with open('scaler_target.pkl', 'rb') as f:
    scaler_target = pickle.load(f)

# Prediction
input_dict = {
    'requests_target': [vals...],  # 12 values
    'error_rate': [vals...],
    ...
}
result = real_time_predict(model, scaler_features, scaler_target, input_dict)
```
