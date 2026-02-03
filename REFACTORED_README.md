# LSTM Time-Series Forecasting - Refactored Codebase

## 📁 Project Structure

```
src/
├── models/                 # Model architecture and persistence
│   ├── __init__.py
│   ├── lstm_model.py      # LSTMModel class
│   └── model_utils.py     # Save/load model, scalers, metadata
│
├── data/                   # Data preparation and preprocessing
│   ├── __init__.py
│   └── data_preparation.py # Loading, scaling, windowing
│
├── training/              # Training and evaluation
│   ├── __init__.py
│   ├── training.py        # Training loop with early stopping
│   ├── evaluation.py      # Metrics and evaluation
│   └── tuning.py          # Hyperparameter tuning
│
└── inference/             # Real-time prediction
    ├── __init__.py
    └── predictor.py       # Inference engine and buffer

notebooks/experimental/
└── duccuong_lstm.ipynb    # Original Jupyter notebook

config/
└── settings.py            # Project settings and paths

main_training.py           # Complete training pipeline
example_inference.py       # Inference examples
```

## 🚀 Quick Start

### 1. Training a Model

```python
from main_training import main

# Run complete training pipeline
results = main()
```

This will:

- Load and prepare data
- Select important features
- Train LSTM model with early stopping
- Evaluate on test set
- Save all artifacts
- Export production package

### 2. Making Predictions

```python
from src.models import load_production_model, load_scalers
from src.inference import real_time_predict

# Load model and scalers
model, metadata = load_production_model(
    model_weights_path='models/lstm_5m_best_model.pth',
    metadata_path='output/model_metadata.json'
)

scaler_features, scaler_target = load_scalers(
    features_scaler_path='models/scaler_features_5m.pkl',
    target_scaler_path='models/scaler_target_5m.pkl'
)

# Prepare input
data_input = {
    'requests_target': [500, 520, ..., 720],  # 12 values
    'error_rate': [0.01, 0.01, ..., 0.02],
    'hour_sin': [0.1, 0.15, ..., 0.65],
    'hour_cos': [0.99, 0.99, ..., 0.89],
    'is_weekend': [0.0, 0.0, ..., 0.0]
}

# Make prediction
result = real_time_predict(
    model=model,
    scaler_features=scaler_features,
    scaler_target=scaler_target,
    data_dict=data_input,
    expected_features=metadata['features'],
    metadata=metadata,
    return_confidence=True
)

print(f"Prediction: {result['prediction']:.2f}")
print(f"Confidence: {result['confidence_interval']}")
```

## 📚 Module Documentation

### `src.models`

**Classes:**

- `LSTMModel`: LSTM architecture with configurable layers and dropout

**Functions:**

- `save_model_weights()`: Save PyTorch model weights
- `load_model_weights()`: Load PyTorch model weights
- `save_scalers()`: Save MinMaxScalers to pickle
- `load_scalers()`: Load MinMaxScalers from pickle
- `save_model_metadata()`: Save architecture and metrics to JSON
- `load_production_model()`: Load model with metadata
- `export_model_and_scalers_for_api()`: Create complete export package

### `src.data`

**Classes:**

- `TimeSeriesDataset`: PyTorch Dataset for windowed data

**Functions:**

- `select_features_lstm()`: Feature selection using Random Forest
- `prepare_data_pipeline()`: Complete data preprocessing
- `create_data_buffer()`: Initialize sliding window buffer
- `update_buffer()`: Update buffer with new data
- `validate_data_shapes()`: Validate dataframe structure

### `src.training`

**Training:**

- `train()`: Complete training loop with early stopping
- `train_epoch()`: Train for one epoch
- `validate_epoch()`: Validate on validation set

**Evaluation:**

- `evaluate()`: Evaluate on test set with metrics
- `calculate_metrics()`: Calculate RMSE, MAE, MAPE
- `calculate_mape()`: Calculate MAPE only
- `plot_metrics_comparison()`: Compare multiple models

**Tuning:**

- `run_hyperparameter_tuning()`: Hyperparameter grid search
- `load_tuning_config()`: Load tuning configuration from YAML
- `save_tuning_results()`: Save tuning results

### `src.inference`

**Prediction:**

- `real_time_predict()`: Single prediction on real-time data
- `batch_predict()`: Batch predictions
- `validate_real_time_input()`: Validate input before prediction

**Utilities:**

- `PredictionBuffer`: Track prediction history

## 📊 File Artifacts

After training, the following files are created:

```
models/
├── lstm_5m_best_model.pth       # Model weights (PyTorch)
├── scaler_features_5m.pkl       # Features normalization scaler
└── scaler_target_5m.pkl         # Target denormalization scaler

output/
├── model_metadata.json           # Architecture & configuration
└── lstm_5m_results.json          # Training metrics

models_export/                    # Complete production package
├── model_weights.pth
├── scaler_features.pkl
├── scaler_target.pkl
├── model_metadata.json
├── requirements.txt
└── README.md
```

## 🔄 Workflow

### Training Workflow

```
Data Loading (CSV)
    ↓
Feature Selection (Random Forest)
    ↓
Data Splitting (Train/Val/Test)
    ↓
Scaling (MinMaxScaler)
    ↓
Windowing (TimeSeriesDataset)
    ↓
Training Loop (Adam + MSE Loss)
    ↓
Early Stopping (Validation Loss)
    ↓
Model Evaluation (RMSE, MAE, MAPE)
    ↓
Artifact Saving (Model + Scalers + Metadata)
    ↓
Production Export (API Package)
```

### Inference Workflow

```
Input Data (12 timesteps × 5 features)
    ↓
Validation (Shape, NaN, Range)
    ↓
Normalization (Features Scaler)
    ↓
Model Inference (LSTM Forward Pass)
    ↓
Denormalization (Target Scaler)
    ↓
Confidence Interval (Optional)
    ↓
Output (Prediction + Metadata)
```

## 🎯 Usage Patterns

### Pattern 1: Training from Scratch

```python
from main_training import main

results = main()
```

### Pattern 2: Load Trained Model

```python
from src.models import load_production_model, load_scalers

model, metadata = load_production_model(
    model_weights_path='models/lstm_5m_best_model.pth',
    metadata_path='output/model_metadata.json'
)

scaler_features, scaler_target = load_scalers(
    features_scaler_path='models/scaler_features_5m.pkl',
    target_scaler_path='models/scaler_target_5m.pkl'
)
```

### Pattern 3: Streaming Predictions

```python
from src.data import create_data_buffer, update_buffer
from src.inference import real_time_predict

# Initialize buffer from historical data
buffer = create_data_buffer(df, frequency='5m', window_size=12)

# Make prediction
data_input = {
    feature: buffer['data'][:, i].tolist()
    for i, feature in enumerate(buffer['features'])
}
result = real_time_predict(model, scaler_features, scaler_target, data_input, ...)

# Update with new data
new_row = {'requests_target': 750, 'error_rate': 0.01, ...}
buffer = update_buffer(buffer, new_row)
```

### Pattern 4: Batch Processing

```python
from src.inference import batch_predict

batch_data = [data1, data2, data3, ...]  # List of dicts

results = batch_predict(
    model=model,
    scaler_features=scaler_features,
    scaler_target=scaler_target,
    data_list=batch_data,
    expected_features=metadata['features']
)
```

## 📈 Model Architecture

```
Input (batch_size, 12, 5)
    ↓
LSTM Layer 1 (64 units)
    ↓
LSTM Layer 2 (64 units)
    ↓
Last Time Step (batch_size, 64)
    ↓
Dense Layer (batch_size, 1)
    ↓
Output (batch_size,)
```

**Hyperparameters:**

- Input size: 5 (features)
- Hidden size: 64
- Number of layers: 2
- Dropout: 0.2
- Learning rate: 0.001
- Batch size: 32
- Sequence length: 12 (60 minutes at 5m frequency)

## 🔧 Configuration

Settings are defined in `config/settings.py`:

```python
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
```

Hyperparameter tuning configuration in `config/train_config.yaml`:

```yaml
model_configs:
  - name: "small_model"
    input_size: 5
    hidden_size: 32
    num_layers: 1
    dropout: 0.1

training_configs:
  - name: "standard"
    batch_size: 32
    learning_rate: 0.001
    epochs: 50
    patience: 5
```

## 📝 Examples

Run example scripts:

```bash
# Training example
python main_training.py

# Inference examples
python example_inference.py
```

## 🔗 Integration Points

For API integration:

1. **Model Loading** → `src.models.load_production_model()`
2. **Input Validation** → `src.inference.validate_real_time_input()`
3. **Prediction** → `src.inference.real_time_predict()`
4. **Batch Processing** → `src.inference.batch_predict()`

Example Flask app:

```python
from flask import Flask, request, jsonify
from src.models import load_production_model, load_scalers
from src.inference import real_time_predict

app = Flask(__name__)

# Load at startup
model, metadata = load_production_model(...)
scaler_features, scaler_target = load_scalers(...)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = real_time_predict(
        model, scaler_features, scaler_target,
        data, metadata['features'], metadata
    )
    return jsonify(result)
```

## 📊 Performance Metrics

Typical performance:

- **MAPE**: 18-20%
- **RMSE**: 100-150 requests
- **MAE**: 80-120 requests
- **Inference time**: <100ms per prediction

## 🐛 Troubleshooting

**Issue**: Model loading fails

- Solution: Check metadata.json matches model architecture

**Issue**: Prediction returns NaN

- Solution: Validate input has no NaN/Inf values, correct shape

**Issue**: Slow training

- Solution: Use CUDA if available, reduce model size, increase batch size

## 📚 References

- PyTorch LSTM: https://pytorch.org/docs/stable/nn.html#lstm
- Time-series forecasting: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- Early stopping: https://en.wikipedia.org/wiki/Early_stopping
