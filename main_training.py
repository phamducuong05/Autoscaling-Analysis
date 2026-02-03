"""
Main training pipeline
Complete workflow from data loading to model export
"""
import sys
from pathlib import Path
import torch
import pandas as pd
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules
from config.settings import DATA_CLEANED_DIR, MODELS_DIR, OUTPUT_DIR
from src.lstm.data import select_features_lstm, prepare_data_pipeline
from src.lstm.models import LSTMModel, save_scalers, save_model_metadata, export_model_and_scalers_for_api
from src.lstm.training import train, evaluate


def main():
    """
    Complete training pipeline
    """
    print("=" * 80)
    print("LSTM TIME-SERIES FORECASTING - COMPLETE PIPELINE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # ===========================
    # STEP 1: Load Data
    # ===========================
    print("STEP 1: Loading and Preparing Data")
    print("-" * 80)
    
    data_5m_path = DATA_CLEANED_DIR / 'data_5m.csv'
    df_5m = pd.read_csv(data_5m_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded data shape: {df_5m.shape}")
    
    # ===========================
    # STEP 2: Feature Selection
    # ===========================
    print("\nSTEP 2: Feature Selection")
    print("-" * 80)
    
    df_lstm_input = select_features_lstm(df_5m)
    print(f"✓ Selected features: {df_lstm_input.columns.tolist()}")
    
    # ===========================
    # STEP 3: Data Preparation
    # ===========================
    print("\nSTEP 3: Data Preparation (Train/Val/Test Split & Scaling)")
    print("-" * 80)
    
    train_loader, val_loader, test_loader, scaler_target, scaler_features = \
        prepare_data_pipeline(df_lstm_input, sequence_length=12, batch_size=32)
    
    print("✓ Data preparation complete")
    
    # ===========================
    # STEP 4: Model Training
    # ===========================
    print("\nSTEP 4: Model Training")
    print("-" * 80)
    
    model = LSTMModel(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model_save_path = MODELS_DIR / 'lstm_5m_best_model.pth'
    
    train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        patience=10,
        model_save_path=model_save_path,
        device=device
    )
    
    print(f"✓ Training complete (Best epoch: {train_result['best_epoch']})")
    
    # ===========================
    # STEP 5: Model Evaluation
    # ===========================
    print("\nSTEP 5: Model Evaluation")
    print("-" * 80)
    
    eval_result = evaluate(
        model=train_result['model'],
        test_loader=test_loader,
        scaler_target=scaler_target,
        device=device,
        plot=True
    )
    
    # ===========================
    # STEP 6: Save Artifacts
    # ===========================
    print("\nSTEP 6: Saving Artifacts")
    print("-" * 80)
    
    # Save scalers
    scalers_paths = save_scalers(
        scaler_features,
        scaler_target,
        output_dir=MODELS_DIR
    )
    
    # Save metadata
    metadata = save_model_metadata(
        model=train_result['model'],
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        eval_result=eval_result,
        model_path=model_save_path,
        metadata_path=OUTPUT_DIR / 'model_metadata.json'
    )
    
    # Save results summary
    results_summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_config': {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        },
        'training_config': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10,
            'sequence_length': 12
        },
        'training_results': {
            'best_epoch': train_result['best_epoch'],
            'best_val_loss': float(train_result['best_val_loss']),
            'num_epochs_trained': len(train_result['train_losses']),
            'training_time_seconds': train_result.get('training_time', None)
        },
        'test_metrics': {
            'rmse': float(eval_result['metrics']['rmse']),
            'mse': float(eval_result['metrics']['mse']),
            'mae': float(eval_result['metrics']['mae']),
            'mape': float(eval_result['metrics']['mape'])
        }
    }
    
    results_path = OUTPUT_DIR / 'lstm_5m_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"✓ Results saved to: {results_path}")
    
    # ===========================
    # STEP 7: Export for Production
    # ===========================
    print("\nSTEP 7: Export Package for Production API")
    print("-" * 80)
    
    package_info = export_model_and_scalers_for_api(
        model_path=model_save_path,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        metadata=metadata,
        export_dir=PROJECT_ROOT / 'models_export'
    )
    
    # ===========================
    # FINAL SUMMARY
    # ===========================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    print("\n📊 Performance Metrics:")
    print(f"   RMSE:  {eval_result['metrics']['rmse']:.2f}")
    print(f"   MAE:   {eval_result['metrics']['mae']:.2f}")
    print(f"   MAPE:  {eval_result['metrics']['mape']:.2f}%")
    
    print("\n📦 Artifacts Created:")
    print(f"   Model weights:     {model_save_path}")
    print(f"   Features scaler:   {scalers_paths['features_scaler_path']}")
    print(f"   Target scaler:     {scalers_paths['target_scaler_path']}")
    print(f"   Metadata:          {OUTPUT_DIR / 'model_metadata.json'}")
    print(f"   Results:           {results_path}")
    
    print(f"\n🚀 Production Package:")
    print(f"   Location:          {package_info['export_dir']}")
    print(f"   Files:             {len(package_info['files'])}")
    
    print("\n✅ Next steps:")
    print("   1. Use models_export/ directory for API deployment")
    print("   2. Load model with: from src.models import load_production_model")
    print("   3. Make predictions with: from src.inference import real_time_predict")
    
    return results_summary


if __name__ == '__main__':
    results = main()
