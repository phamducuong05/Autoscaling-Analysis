"""
Example: Real-time prediction with LSTM model
Shows how to use the model for production inference
"""
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS_DIR, OUTPUT_DIR, DATA_CLEANED_DIR
from src.lstm.models import load_production_model, load_scalers
from src.lstm.data import create_data_buffer, update_buffer
from src.lstm.inference import real_time_predict, batch_predict, PredictionBuffer


def example_single_prediction():
    """
    Example 1: Make a single prediction
    """
    print("=" * 80)
    print("EXAMPLE 1: Single Real-Time Prediction")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and metadata
    print("\n1. Loading Model...")
    model, metadata = load_production_model(
        model_weights_path=MODELS_DIR / 'lstm_5m_best_model.pth',
        metadata_path=OUTPUT_DIR / 'model_metadata.json',
        device=device
    )
    
    # Load scalers
    print("\n2. Loading Scalers...")
    scaler_features, scaler_target = load_scalers(
        features_scaler_path=MODELS_DIR / 'scaler_features_5m.pkl',
        target_scaler_path=MODELS_DIR / 'scaler_target_5m.pkl'
    )
    
    # Create dummy input data (simulate real-time)
    print("\n3. Creating Input Data...")
    data_input = {
        'requests_target': [500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720],
        'error_rate': [0.01, 0.01, 0.02, 0.01, 0.015, 0.02, 0.015, 0.01, 0.02, 0.015, 0.01, 0.02],
        'hour_sin': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        'hour_cos': [0.99, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89],
        'is_weekend': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    print(f"   Input shape: 12 timesteps × 5 features")
    
    # Make prediction
    print("\n4. Making Prediction...")
    try:
        result = real_time_predict(
            model=model,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            data_dict=data_input,
            expected_features=metadata['features'],
            metadata=metadata,
            return_confidence=True,
            device=device
        )
        
        print(f"\n✓ Prediction successful!")
        print(f"   Predicted Requests: {result['prediction']:.2f}")
        print(f"   Predicted (Scaled): {result['prediction_scaled']:.4f}")
        
        if result['confidence_interval']:
            lower, upper = result['confidence_interval']
            print(f"   Confidence Interval: [{lower:.2f}, {upper:.2f}]")
            print(f"   Interval Width: {upper - lower:.2f}")
        
        print(f"   Timestamp: {result['timestamp']}")
        
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")


def example_streaming_buffer():
    """
    Example 2: Streaming data with sliding window buffer
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Streaming Buffer Management")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("\n1. Loading Model and Data...")
    model, metadata = load_production_model(
        model_weights_path=MODELS_DIR / 'lstm_5m_best_model.pth',
        metadata_path=OUTPUT_DIR / 'model_metadata.json',
        device=device
    )
    
    scaler_features, scaler_target = load_scalers(
        features_scaler_path=MODELS_DIR / 'scaler_features_5m.pkl',
        target_scaler_path=MODELS_DIR / 'scaler_target_5m.pkl'
    )
    
    # Load test data to simulate streaming
    data_5m_path = DATA_CLEANED_DIR / 'data_5m.csv'
    df = pd.read_csv(data_5m_path, index_col=0, parse_dates=True)
    
    # Create initial buffer
    print("\n2. Creating Initial Buffer...")
    df_sample = df.iloc[-100:]  # Last 100 rows for demo
    buffer = create_data_buffer(df_sample, frequency='5m', window_size=12)
    
    print(f"   Buffer size: {buffer['data'].shape}")
    print(f"   Latest timestamp: {buffer['timestamp']}")
    
    # Simulate streaming: make predictions as new data arrives
    print("\n3. Simulating Streaming Predictions...")
    
    predictions = []
    for i, (idx, row) in enumerate(df_sample.iloc[-20:].iterrows()):
        # Create input from buffer
        data_input = {
            feature: buffer['data'][:, j].tolist()
            for j, feature in enumerate(buffer['features'])
        }
        
        # Predict
        result = real_time_predict(
            model=model,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            data_dict=data_input,
            expected_features=metadata['features'],
            metadata=metadata,
            return_confidence=False,
            device=device
        )
        
        predictions.append({
            'timestamp': idx,
            'actual': row['requests_target'],
            'predicted': result['prediction']
        })
        
        # Update buffer with new data
        new_row = {
            feature: row[feature] for feature in buffer['features']
        }
        buffer = update_buffer(buffer, new_row)
        
        if (i + 1) % 5 == 0:
            print(f"   ✓ Processed {i + 1} updates")
    
    # Show results
    pred_df = pd.DataFrame(predictions)
    print(f"\n✓ Streaming predictions complete!")
    print(f"\nSample predictions:")
    print(pred_df.tail(5).to_string(index=False))


def example_batch_predictions():
    """
    Example 3: Batch predictions
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Batch Predictions")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("\n1. Loading Model...")
    model, metadata = load_production_model(
        model_weights_path=MODELS_DIR / 'lstm_5m_best_model.pth',
        metadata_path=OUTPUT_DIR / 'model_metadata.json',
        device=device
    )
    
    scaler_features, scaler_target = load_scalers(
        features_scaler_path=MODELS_DIR / 'scaler_features_5m.pkl',
        target_scaler_path=MODELS_DIR / 'scaler_target_5m.pkl'
    )
    
    # Create batch of inputs
    print("\n2. Creating Batch Inputs...")
    batch_data = []
    
    for batch_idx in range(5):
        data_input = {
            'requests_target': np.random.uniform(500, 800, 12).tolist(),
            'error_rate': np.random.uniform(0.01, 0.03, 12).tolist(),
            'hour_sin': np.random.uniform(0, 1, 12).tolist(),
            'hour_cos': np.random.uniform(0.8, 1, 12).tolist(),
            'is_weekend': [0.0] * 12
        }
        batch_data.append(data_input)
    
    print(f"   Batch size: {len(batch_data)}")
    
    # Make batch predictions
    print("\n3. Making Batch Predictions...")
    results = batch_predict(
        model=model,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        data_list=batch_data,
        expected_features=metadata['features'],
        metadata=metadata,
        device=device
    )
    
    # Show results
    print(f"\n✓ Batch predictions complete!")
    print(f"\nResults:")
    for i, result in enumerate(results):
        if result['status'] == 'success':
            print(f"   [{i}] Prediction: {result['prediction']:.2f}")
        else:
            print(f"   [{i}] Error: {result['error']}")


def example_prediction_monitoring():
    """
    Example 4: Monitor predictions over time
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Prediction Monitoring & History")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("\n1. Loading Model...")
    model, metadata = load_production_model(
        model_weights_path=MODELS_DIR / 'lstm_5m_best_model.pth',
        metadata_path=OUTPUT_DIR / 'model_metadata.json',
        device=device
    )
    
    scaler_features, scaler_target = load_scalers(
        features_scaler_path=MODELS_DIR / 'scaler_features_5m.pkl',
        target_scaler_path=MODELS_DIR / 'scaler_target_5m.pkl'
    )
    
    # Initialize prediction buffer
    print("\n2. Creating Prediction Buffer...")
    pred_buffer = PredictionBuffer(max_size=1000)
    
    # Simulate making predictions
    print("\n3. Simulating Predictions...")
    for i in range(50):
        data_input = {
            'requests_target': np.random.uniform(500, 800, 12).tolist(),
            'error_rate': np.random.uniform(0.01, 0.03, 12).tolist(),
            'hour_sin': np.random.uniform(0, 1, 12).tolist(),
            'hour_cos': np.random.uniform(0.8, 1, 12).tolist(),
            'is_weekend': [0.0] * 12
        }
        
        result = real_time_predict(
            model=model,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            data_dict=data_input,
            expected_features=metadata['features'],
            metadata=metadata,
            return_confidence=True,
            device=device
        )
        
        pred_buffer.add(result)
        
        if (i + 1) % 10 == 0:
            print(f"   ✓ {i + 1} predictions recorded")
    
    # Show statistics
    print(f"\n✓ Prediction monitoring complete!")
    print(f"\nBuffer Statistics:")
    print(f"   Total predictions: {len(pred_buffer.predictions)}")
    print(f"   Avg confidence width: {pred_buffer.get_avg_confidence(n=50):.2f}")
    
    recent = pred_buffer.get_recent(5)
    print(f"\nLast 5 predictions:")
    for i, pred in enumerate(recent):
        print(f"   [{i}] {pred['prediction']:.2f} ± {(pred['confidence_interval'][1] - pred['confidence_interval'][0])/2:.2f}")


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 80)
    print("LSTM INFERENCE EXAMPLES")
    print("=" * 80)
    
    example_single_prediction()
    example_streaming_buffer()
    example_batch_predictions()
    example_prediction_monitoring()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
