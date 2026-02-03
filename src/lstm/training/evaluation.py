"""
Model evaluation metrics
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float : MAPE value
    """
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100


def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    dict : {
        'rmse': root mean squared error,
        'mse': mean squared error,
        'mae': mean absolute error,
        'mape': mean absolute percentage error
    }
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
        'mape': mape
    }


def evaluate(model, test_loader, scaler_target, device='cpu', plot=True):
    """
    Evaluate model on test set
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    test_loader : DataLoader
        Test data
    scaler_target : MinMaxScaler
        Target scaler for inverse transform
    device : str
        Device ('cpu' or 'cuda')
    plot : bool
        Whether to plot predictions vs actual
        
    Returns:
    --------
    dict : {
        'predictions': np.ndarray (unscaled),
        'actuals': np.ndarray (unscaled),
        'metrics': dict with rmse, mse, mae, mape
    }
    """
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # Convert to arrays
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # Inverse transform
    predictions = scaler_target.inverse_transform(predictions)
    actuals = scaler_target.inverse_transform(actuals)
    
    # Flatten for metrics
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    # Print metrics
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MSE:  {metrics['mse']:.2f}")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    if plot:
        # Plot zoomed
        plt.figure(figsize=(15, 6))
        plt.plot(actuals[:300], label='Actual Request', color='blue', linewidth=1.5)
        plt.plot(predictions[:300], label='LSTM Forecast', color='orange', linestyle='--', linewidth=1.5)
        plt.title('Predictions vs Actual (First 300 samples)')
        plt.xlabel('Time Steps')
        plt.ylabel('Requests')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot full
        plt.figure(figsize=(15, 6))
        plt.plot(actuals, label='Actual Request', color='blue', linewidth=0.8, alpha=0.8)
        plt.plot(predictions, label='LSTM Forecast', color='orange', linestyle='--', linewidth=0.8, alpha=0.8)
        plt.title('Full Predictions vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('Requests')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics
    }


def plot_metrics_comparison(results_list):
    """
    Plot comparison of multiple models
    
    Parameters:
    -----------
    results_list : list of dict
        List of results with 'model_name' and 'metrics'
    """
    models = [r['model_name'] for r in results_list]
    rmse_vals = [r['metrics']['rmse'] for r in results_list]
    mae_vals = [r['metrics']['mae'] for r in results_list]
    mape_vals = [r['metrics']['mape'] for r in results_list]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE
    axes[0].bar(models, rmse_vals, color='steelblue')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAE
    axes[1].bar(models, mae_vals, color='orange')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    # MAPE
    axes[2].bar(models, mape_vals, color='green')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('MAPE Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
