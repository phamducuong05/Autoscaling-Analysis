"""
Real-time prediction engine
"""
import numpy as np
import torch
from datetime import datetime


def validate_real_time_input(data_dict, expected_features, window_size=12):
    """
    Validate real-time input data before prediction
    
    Parameters:
    -----------
    data_dict : dict
        Input data in format:
        {
            'requests_target': [val1, val2, ...],
            'error_rate': [val1, val2, ...],
            ...
        }
    expected_features : list
        List of expected feature names
    window_size : int
        Expected window size (default: 12)
        
    Returns:
    --------
    tuple : (is_valid: bool, error_message: str or None)
    """
    # Check all features present
    missing_features = [f for f in expected_features if f not in data_dict]
    if missing_features:
        return False, f"Missing features: {missing_features}"
    
    # Check length and values
    for feature, values in data_dict.items():
        if feature in expected_features:
            # Type check
            if not isinstance(values, (list, tuple, np.ndarray)):
                return False, f"Feature {feature} must be list/array, got {type(values)}"
            
            # Length check
            if len(values) != window_size:
                return False, f"Feature {feature} should have {window_size} values, got {len(values)}"
            
            # Value checks
            if np.any(np.isnan(values)):
                return False, f"Feature {feature} contains NaN values"
            if np.any(np.isinf(values)):
                return False, f"Feature {feature} contains Inf values"
    
    return True, None


def real_time_predict(model, scaler_features, scaler_target, data_dict,
                     expected_features, metadata=None, return_confidence=False,
                     device='cpu'):
    """
    Make prediction on real-time data
    
    Complete inference pipeline:
    1. Validate input
    2. Prepare input array
    3. Normalize using training scalers
    4. Convert to tensor
    5. Model inference
    6. Inverse transform output
    7. Return result with optional confidence interval
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained LSTM model (eval mode)
    scaler_features : MinMaxScaler
        Features scaler (fitted on training data)
    scaler_target : MinMaxScaler
        Target scaler
    data_dict : dict
        Real-time input data
        {
            'requests_target': [12 values],
            'error_rate': [12 values],
            'hour_sin': [12 values],
            'hour_cos': [12 values],
            'is_weekend': [12 values]
        }
    expected_features : list
        Feature names in order
    metadata : dict
        Model metadata (for confidence estimation)
    return_confidence : bool
        Return confidence interval (default: False)
    device : str
        Torch device ('cpu' or 'cuda')
        
    Returns:
    --------
    dict : {
        'prediction': float,           # Actual value
        'prediction_scaled': float,    # Normalized [0,1]
        'timestamp': str,
        'confidence_interval': tuple or None
    }
    """
    # 1. Validation
    is_valid, error_msg = validate_real_time_input(data_dict, expected_features)
    if not is_valid:
        raise ValueError(f"Input validation failed: {error_msg}")
    
    # 2. Prepare input array (window_size, n_features)
    input_array = np.array([
        data_dict[feature] for feature in expected_features
    ]).T
    
    # 3. Normalize
    input_scaled = scaler_features.transform(input_array)
    
    # 4. Convert to tensor (1, window_size, n_features) for batch
    input_tensor = torch.tensor(
        input_scaled,
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    # 5. Inference
    with torch.no_grad():
        output_scaled = model(input_tensor)
        output_scaled = output_scaled.squeeze().cpu().numpy()
    
    # 6. Inverse transform
    output_unscaled = scaler_target.inverse_transform(
        [[output_scaled]]
    )[0][0]
    
    # 7. Build result
    result = {
        'prediction': float(output_unscaled),
        'prediction_scaled': float(output_scaled),
        'timestamp': datetime.now().isoformat(),
        'confidence_interval': None
    }
    
    # 8. Optional: Confidence interval
    if return_confidence:
        if metadata:
            mape = metadata['evaluation_metrics']['mape']
        else:
            mape = 18.2  # Default from training
        
        margin = output_unscaled * (mape / 100.0)
        result['confidence_interval'] = (
            float(output_unscaled - margin),
            float(output_unscaled + margin)
        )
    
    return result


def batch_predict(model, scaler_features, scaler_target, data_list,
                 expected_features, metadata=None, device='cpu'):
    """
    Make predictions on batch of inputs
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    scaler_features : MinMaxScaler
        Features scaler
    scaler_target : MinMaxScaler
        Target scaler
    data_list : list of dict
        List of input dicts
    expected_features : list
        Feature names
    metadata : dict
        Model metadata
    device : str
        Torch device
        
    Returns:
    --------
    list : List of prediction results
    """
    results = []
    
    for i, data_dict in enumerate(data_list):
        try:
            result = real_time_predict(
                model=model,
                scaler_features=scaler_features,
                scaler_target=scaler_target,
                data_dict=data_dict,
                expected_features=expected_features,
                metadata=metadata,
                return_confidence=True,
                device=device
            )
            result['index'] = i
            result['status'] = 'success'
            results.append(result)
        except Exception as e:
            results.append({
                'index': i,
                'status': 'error',
                'error': str(e),
                'prediction': None
            })
    
    return results


class PredictionBuffer:
    """
    Maintains prediction history for monitoring
    """
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.predictions = []
    
    def add(self, prediction_result):
        """Add prediction to buffer"""
        self.predictions.append(prediction_result)
        
        # Keep buffer size manageable
        if len(self.predictions) > self.max_size:
            self.predictions.pop(0)
    
    def get_recent(self, n=10):
        """Get last n predictions"""
        return self.predictions[-n:]
    
    def get_avg_confidence(self, n=100):
        """Get average confidence interval width"""
        recent = self.predictions[-n:]
        widths = []
        
        for pred in recent:
            if pred.get('confidence_interval'):
                lower, upper = pred['confidence_interval']
                widths.append(upper - lower)
        
        return np.mean(widths) if widths else None
    
    def clear(self):
        """Clear buffer"""
        self.predictions = []
