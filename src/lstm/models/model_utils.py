"""
Model persistence and utility functions
"""
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
import torch

from .lstm_model import LSTMModel


def save_model_weights(model, filepath):
    """
    Save model weights to disk
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to save
    filepath : str or Path
        Path to save weights
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"✓ Model weights saved to: {filepath}")


def load_model_weights(model, filepath, device='cpu'):
    """
    Load model weights from disk
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to load weights into
    filepath : str or Path
        Path to weights file
    device : str
        Device to load to ('cpu' or 'cuda')
        
    Returns:
    --------
    torch.nn.Module
        Model with loaded weights
    """
    filepath = Path(filepath)
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"✓ Model weights loaded from: {filepath}")
    return model


def save_scalers(scaler_features, scaler_target, output_dir=None):
    """
    Save feature and target scalers for normalization/denormalization
    
    Parameters:
    -----------
    scaler_features : sklearn.preprocessing.MinMaxScaler
        Scaler for input features
    scaler_target : sklearn.preprocessing.MinMaxScaler
        Scaler for target variable (requests_target)
    output_dir : str or Path
        Directory to save scalers (default: models/)
        
    Returns:
    --------
    dict : Paths to scaler files
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'models'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature scaler
    features_scaler_path = output_dir / 'scaler_features_5m.pkl'
    with open(features_scaler_path, 'wb') as f:
        pickle.dump(scaler_features, f)
    print(f"✓ Features scaler saved: {features_scaler_path}")
    
    # Save target scaler
    target_scaler_path = output_dir / 'scaler_target_5m.pkl'
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(scaler_target, f)
    print(f"✓ Target scaler saved: {target_scaler_path}")
    
    return {
        'features_scaler_path': features_scaler_path,
        'target_scaler_path': target_scaler_path
    }


def load_scalers(features_scaler_path, target_scaler_path):
    """
    Load scalers from disk
    
    Parameters:
    -----------
    features_scaler_path : str or Path
        Path to features scaler
    target_scaler_path : str or Path
        Path to target scaler
        
    Returns:
    --------
    tuple : (scaler_features, scaler_target)
    """
    with open(features_scaler_path, 'rb') as f:
        scaler_features = pickle.load(f)
    
    with open(target_scaler_path, 'rb') as f:
        scaler_target = pickle.load(f)
    
    print(f"✓ Scalers loaded successfully")
    return scaler_features, scaler_target


def save_model_metadata(model, scaler_features, scaler_target, eval_result, 
                       model_path=None, metadata_path=None):
    """
    Save model architecture, hyperparameters, and metrics to JSON
    
    Parameters:
    -----------
    model : LSTMModel
        Trained model
    scaler_features : MinMaxScaler
        Features scaler
    scaler_target : MinMaxScaler
        Target scaler
    eval_result : dict
        Evaluation results with 'metrics' key
    model_path : str or Path
        Path to model weights
    metadata_path : str or Path
        Path to save metadata (default: output/model_metadata.json)
        
    Returns:
    --------
    dict : Metadata dictionary
    """
    if metadata_path is None:
        metadata_path = Path(__file__).parent.parent.parent / 'output' / 'model_metadata.json'
    
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract model architecture
    model_arch = {
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'output_size': 1
    }
    
    # Extract scaler info
    scaler_info = {
        'features': {
            'feature_range': list(scaler_features.feature_range),
            'data_min': scaler_features.data_min_.tolist(),
            'data_max': scaler_features.data_max_.tolist()
        },
        'target': {
            'feature_range': list(scaler_target.feature_range),
            'data_min': float(scaler_target.data_min_[0]),
            'data_max': float(scaler_target.data_max_[0])
        }
    }
    
    # Build metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': 'lstm_5m_best_model',
        'frequency': '5m',
        'model_architecture': model_arch,
        'model_weights_path': str(model_path) if model_path else 'lstm_5m_best_model.pth',
        'training_config': {
            'sequence_length': 12,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10
        },
        'features': ['requests_target', 'error_rate', 'hour_sin', 'hour_cos', 'is_weekend'],
        'feature_count': 5,
        'scaler_info': scaler_info,
        'evaluation_metrics': {
            'rmse': float(eval_result['metrics']['rmse']),
            'mae': float(eval_result['metrics']['mae']),
            'mape': float(eval_result['metrics']['mape']),
            'mse': float(eval_result['metrics']['mse'])
        },
        'inference_config': {
            'sequence_length': 12,
            'expected_input_shape': [None, 12, 5],
            'output_shape': [None, 1],
            'device': 'cpu'
        }
    }
    
    # Save to JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Model metadata saved: {metadata_path}")
    return metadata


def load_production_model(model_weights_path, metadata_path=None, device='cpu'):
    """
    Load model and metadata for production inference
    
    Parameters:
    -----------
    model_weights_path : str or Path
        Path to model weights
    metadata_path : str or Path
        Path to model metadata JSON
    device : str
        Device to load on ('cpu' or 'cuda')
        
    Returns:
    --------
    tuple : (model, metadata)
    """
    model_weights_path = Path(model_weights_path)
    
    if metadata_path is None:
        metadata_path = Path(__file__).parent.parent.parent / 'output' / 'model_metadata.json'
    
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Reconstruct model
    arch = metadata['model_architecture']
    model = LSTMModel(
        input_size=arch['input_size'],
        hidden_size=arch['hidden_size'],
        num_layers=arch['num_layers'],
        dropout=0.2
    )
    
    # Load weights
    model = load_model_weights(model, model_weights_path, device=device)
    model.eval()
    
    print(f"✓ Production model loaded successfully")
    print(f"  Architecture: {arch['input_size']} → {arch['hidden_size']}×{arch['num_layers']} → {arch['output_size']}")
    print(f"  Metrics: MAPE={metadata['evaluation_metrics']['mape']:.2f}%, RMSE={metadata['evaluation_metrics']['rmse']:.2f}")
    
    return model, metadata


def export_model_and_scalers_for_api(model_path, scaler_features, scaler_target, metadata,
                                     export_dir='./models_export'):
    """
    Export complete model package for API service
    
    Creates:
    - model_weights.pth
    - scaler_features.pkl
    - scaler_target.pkl
    - model_metadata.json
    - requirements.txt
    - README.md
    
    Parameters:
    -----------
    model_path : str or Path
        Path to model weights
    scaler_features : MinMaxScaler
        Features scaler
    scaler_target : MinMaxScaler
        Target scaler
    metadata : dict
        Model metadata
    export_dir : str
        Export directory (default: ./models_export)
        
    Returns:
    --------
    dict : Package information
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model weights
    model_dest = export_path / 'model_weights.pth'
    shutil.copy(str(model_path), str(model_dest))
    print(f"✓ Model weights: {model_dest}")
    
    # Save scalers
    features_scaler_dest = export_path / 'scaler_features.pkl'
    target_scaler_dest = export_path / 'scaler_target.pkl'
    
    with open(features_scaler_dest, 'wb') as f:
        pickle.dump(scaler_features, f)
    print(f"✓ Features scaler: {features_scaler_dest}")
    
    with open(target_scaler_dest, 'wb') as f:
        pickle.dump(scaler_target, f)
    print(f"✓ Target scaler: {target_scaler_dest}")
    
    # Save metadata
    metadata_dest = export_path / 'model_metadata.json'
    with open(metadata_dest, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Metadata: {metadata_dest}")
    
    # Create requirements.txt
    requirements = """torch==2.0.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
flask==2.3.0
"""
    
    req_dest = export_path / 'requirements.txt'
    with open(req_dest, 'w') as f:
        f.write(requirements)
    print(f"✓ Requirements: {req_dest}")
    
    # Create README
    readme = """# LSTM Model Package

## Production-Ready LSTM Time-Series Forecasting Model

### Files:
- **model_weights.pth**: PyTorch model weights
- **scaler_features.pkl**: MinMaxScaler for input normalization
- **scaler_target.pkl**: MinMaxScaler for output denormalization
- **model_metadata.json**: Architecture configuration & performance metrics

### Installation:
```bash
pip install -r requirements.txt
```

### Usage:
```python
import torch
import pickle
import json
import numpy as np
from pathlib import Path

# Load model
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

arch = metadata['model_architecture']
model = LSTMModel(
    input_size=arch['input_size'],
    hidden_size=arch['hidden_size'],
    num_layers=arch['num_layers']
)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Load scalers
with open('scaler_features.pkl', 'rb') as f:
    scaler_features = pickle.load(f)
with open('scaler_target.pkl', 'rb') as f:
    scaler_target = pickle.load(f)

# Make prediction
input_data = {
    'requests_target': [...],  # 12 values
    'error_rate': [...],
    'hour_sin': [...],
    'hour_cos': [...],
    'is_weekend': [...]
}

result = real_time_predict(
    model, scaler_features, scaler_target, input_data
)
print(f"Predicted requests: {result['prediction']:.2f}")
```

### Performance:
- **MAPE**: {mape:.2f}%
- **RMSE**: {rmse:.2f}
- **MAE**: {mae:.2f}

### Model Architecture:
- Input: 12 timesteps × 5 features
- LSTM Layers: {num_layers} × {hidden_size} units
- Output: 1 (requests forecast)
- Frequency: 5 minutes
"""
    
    readme_dest = export_path / 'README.md'
    readme_formatted = readme.format(
        mape=metadata['evaluation_metrics']['mape'],
        rmse=metadata['evaluation_metrics']['rmse'],
        mae=metadata['evaluation_metrics']['mae'],
        num_layers=metadata['model_architecture']['num_layers'],
        hidden_size=metadata['model_architecture']['hidden_size']
    )
    
    with open(readme_dest, 'w') as f:
        f.write(readme_formatted)
    print(f"✓ README: {readme_dest}")
    
    package_info = {
        'export_dir': str(export_path),
        'files': {
            'model_weights': str(model_dest),
            'scaler_features': str(features_scaler_dest),
            'scaler_target': str(target_scaler_dest),
            'metadata': str(metadata_dest),
            'requirements': str(req_dest),
            'readme': str(readme_dest)
        }
    }
    
    print(f"\n✓ Model package exported to: {export_path}")
    return package_info
