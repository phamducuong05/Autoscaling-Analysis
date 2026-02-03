import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_and_process_logs, resample_traffic
from src.features import add_features
from src.lstm.models.lstm_model import LSTMModel
import sys
import os

# Setup paths
MODEL_DIR = "models_export"
TEST_FILE = "data/raw/test.txt"

def main():
    print(f"🚀 Starting LSTM Visualization on {TEST_FILE}...")

    # 1. Load & Process Data
    print("1️⃣  Loading and Processing Log Data...")
    if not os.path.exists(TEST_FILE):
        print(f"❌ Error: File {TEST_FILE} not found.")
        return

    # Parse logs
    df_raw = load_and_process_logs([TEST_FILE])
    print(f"   Parsed {len(df_raw)} log lines.")

    # Resample to 5 minutes (matching model training)
    df_resampled = resample_traffic(df_raw, window='5min')
    print(f"   Resampled to {len(df_resampled)} data points (5min intervals).")

    # Feature Engineering
    df_features = add_features(df_resampled, frequency='5m')
    print("   ✅ Features Added (Lag, Rolling, Cyclic Time)")

    # 2. Load Model & Scalers
    print("\n2️⃣  Loading Model & Scalers...")
    try:
        scaler_features = joblib.load(f"{MODEL_DIR}/scaler_features.pkl")
        scaler_target = joblib.load(f"{MODEL_DIR}/scaler_target.pkl")
        
        model = LSTMModel(input_size=5, hidden_size=32, num_layers=1)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/model_weights.pth", map_location='cpu'))
        model.eval()
        print("   ✅ LSTM Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Prepare Data for Inference
    print("\n3️⃣  Running Inference...")
    
    # Select features in correct order
    feature_cols = ['requests_target', 'error_rate', 'hour_sin', 'hour_cos', 'is_weekend']
    # 'requests_target' is the target we want to predict (shifted in training), 
    # but for inference input at time T, we use 'requests' (lag 0) or shifted versions depending on training strategy.
    # The 'feature_cols' above looks like direct inputs. 
    # wait, 'requests_target' is usually the LABEL. 
    # Let's check model_metadata.json features list again.
    # "features": ["requests_target", "error_rate", "hour_sin", "hour_cos", "is_weekend"]
    # If using lag features, we usually need past data.
    # The LSTM takes a sequence of length 12.
    
    # We need to construct sequences
    sequence_length = 12
    data_matrix = df_features[feature_cols].values
    
    # Scale Data
    data_scaled = scaler_features.transform(data_matrix)
    
    predictions = []
    actuals = []
    timestamps = []

    # Iterate through data to create sequences
    # We can only start predicting after 'sequence_length' steps
    for i in range(sequence_length, len(data_scaled)):
        # Input sequence: [i-12 : i]
        seq = data_scaled[i-sequence_length : i] 
        # Target: data_scaled[i] (specifically the Load component)
        
        # Convert to Tensor (1, 12, 5)
        input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).item()
            
        # Inverse Scale
        pred_val = scaler_target.inverse_transform([[pred_scaled]])[0][0]
        
        predictions.append(pred_val)
        actuals.append(df_features['requests'].iloc[i]) # Use original 'requests' as truth
        timestamps.append(df_features.index[i])

    # 4. Visualization
    print("\n4️⃣  Generating Plot...")
    plt.figure(figsize=(15, 7))
    plt.plot(timestamps, actuals, label='Actual Traffic', color='#1f77b4', alpha=0.7)
    plt.plot(timestamps, predictions, label='LSTM Forecast (Next Step)', color='#ff7f0e', linewidth=2)
    
    plt.title('LSTM Model Performance on Test Data (5-min intervals)', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Requests / 5min')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = "lstm_test_result.png"
    plt.savefig(output_img)
    print(f"   ✅ Plot saved to: {output_img}")
    print(f"   Processed {len(predictions)} validation points.")

if __name__ == "__main__":
    main()
