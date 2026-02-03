"""
Hyperparameter tuning utilities
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from .training import train
from .evaluation import evaluate


def load_tuning_config(config_path):
    """
    Load tuning configuration from YAML
    
    Parameters:
    -----------
    config_path : str or Path
        Path to train_config.yaml
        
    Returns:
    --------
    dict : Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded config from: {config_path}")
    return config


def get_model_config(config, model_name):
    """Get model configuration by name"""
    for model in config['model_configs']:
        if model['name'] == model_name:
            return model
    raise ValueError(f"Model '{model_name}' not found in config")


def get_training_config(config, training_name):
    """Get training configuration by name"""
    for training in config['training_configs']:
        if training['name'] == training_name:
            return training
    raise ValueError(f"Training '{training_name}' not found in config")


def get_data_config(config, data_name):
    """Get data configuration by name"""
    for data in config['data_configs']:
        if data['name'] == data_name:
            return data
    raise ValueError(f"Data '{data_name}' not found in config")


def run_hyperparameter_tuning(config, scaler_target, scaler_features, df_input,
                             LSTMModelClass, prepare_data_fn,
                             tuning_indices=None, models_dir=None, device='cpu'):
    """
    Run hyperparameter tuning over grid of configurations
    
    Parameters:
    -----------
    config : dict
        Tuning configuration
    scaler_target : MinMaxScaler
        Target scaler
    scaler_features : MinMaxScaler
        Features scaler
    df_input : pd.DataFrame
        Input data
    LSTMModelClass : class
        LSTM model class
    prepare_data_fn : callable
        Data preparation function
    tuning_indices : list
        Indices to run (default: all)
    models_dir : Path
        Directory to save models
    device : str
        Device to train on
        
    Returns:
    --------
    dict : {
        'results_df': DataFrame with results,
        'best_config': Best configuration,
        'best_idx': Index of best config
    }
    """
    if tuning_indices is None:
        tuning_indices = range(len(config['tuning_grid']))
    
    results = []
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING")
    print("=" * 80)
    
    for idx in tuning_indices:
        grid_item = config['tuning_grid'][idx]
        
        model_cfg = get_model_config(config, grid_item['model'])
        training_cfg = get_training_config(config, grid_item['training'])
        data_cfg = get_data_config(config, grid_item['data'])
        
        print(f"\n[{idx+1}/{len(tuning_indices)}] Testing: {grid_item['model']} | {grid_item['training']} | {grid_item['data']}")
        print(f"    Priority: {grid_item['priority']}")
        
        try:
            # Prepare data
            train_loader, val_loader, test_loader, _, _ = prepare_data_fn(
                df_input,
                sequence_length=data_cfg['sequence_length'],
                batch_size=training_cfg['batch_size']
            )
            
            # Initialize model
            model = LSTMModelClass(
                input_size=model_cfg['input_size'],
                hidden_size=model_cfg['hidden_size'],
                num_layers=model_cfg['num_layers'],
                dropout=model_cfg['dropout']
            ).to(device)
            
            # Train
            model_save_path = None
            if models_dir:
                model_save_path = Path(models_dir) / f"tuning_{grid_item['model']}_{grid_item['training']}.pth"
            
            train_result = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=training_cfg['epochs'],
                lr=training_cfg['learning_rate'],
                patience=training_cfg['patience'],
                model_save_path=model_save_path,
                device=device
            )
            
            # Evaluate
            eval_result = evaluate(
                model=train_result['model'],
                test_loader=test_loader,
                scaler_target=scaler_target,
                device=device,
                plot=False
            )
            
            # Store result
            result_item = {
                'index': idx,
                'model': grid_item['model'],
                'training': grid_item['training'],
                'data': grid_item['data'],
                'priority': grid_item['priority'],
                'best_epoch': train_result['best_epoch'],
                'best_val_loss': train_result['best_val_loss'],
                'test_rmse': eval_result['metrics']['rmse'],
                'test_mae': eval_result['metrics']['mae'],
                'test_mape': eval_result['metrics']['mape'],
                'model_path': str(model_save_path) if model_save_path else None,
                'training_time': train_result.get('training_time', None)
            }
            results.append(result_item)
            
            print(f"    ✓ Test MAPE: {eval_result['metrics']['mape']:.2f}% | RMSE: {eval_result['metrics']['rmse']:.2f}")
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("TUNING RESULTS SUMMARY")
    print("=" * 80)
    print(results_df[['model', 'training', 'data', 'test_rmse', 'test_mae', 'test_mape']].to_string(index=False))
    
    # Find best config
    best_idx = results_df['test_mape'].idxmin()
    best_config_item = results_df.loc[best_idx]
    
    print(f"\n🏆 Best Configuration:")
    print(f"   Model: {best_config_item['model']}")
    print(f"   Training: {best_config_item['training']}")
    print(f"   Data: {best_config_item['data']}")
    print(f"   Test MAPE: {best_config_item['test_mape']:.2f}%")
    print(f"   Test RMSE: {best_config_item['test_rmse']:.2f}")
    
    return {
        'results_df': results_df,
        'best_config': best_config_item,
        'best_idx': best_idx
    }


def save_tuning_results(results_df, output_path):
    """
    Save tuning results to CSV and JSON
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    output_path : str or Path
        Output directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_path / 'tuning_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    # Save JSON
    json_path = output_path / 'tuning_results.json'
    results_df.to_json(json_path, orient='records', indent=4)
    print(f"✓ Results saved to: {json_path}")
