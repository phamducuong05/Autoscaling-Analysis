import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100



def evaluate(model, test_loader, scaler_target):
    """
    Returns:
    --------
    dict : Dictionary chứa:
        - 'predictions': array predictions (đơn vị thực tế)
        - 'actuals': array actual values (đơn vị thực tế)
        - 'metrics': dict chứa rmse, mse, mae, mape
    """
    print("\n--- EVALUATION ON TEST SET ---")
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            # Move về CPU và convert sang numpy
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
            
    # Inverse Transform về đơn vị request thực tế
    predictions = scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler_target.inverse_transform(np.array(actuals).reshape(-1, 1))
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = calculate_mape(actuals, predictions)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # ===========================
    # Plot Predictions vs Actuals
    # ===========================
    plt.figure(figsize=(15, 6))
    plt.plot(actuals[:300], label='Actual Request', color='blue', linewidth=1.5)
    plt.plot(predictions[:300], label='LSTM Forecast', color='orange', linestyle='--', linewidth=1.5)
    plt.title('Dự báo vs Thực tế (Zoom in 300 điểm dữ liệu đầu tập Test)')
    plt.xlabel('Time Steps')
    plt.ylabel('Requests')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # ===========================
    # Plot Full Predictions
    # ===========================
    plt.figure(figsize=(15, 6))
    plt.plot(actuals, label='Actual Request', color='blue', linewidth=0.8, alpha=0.8)
    plt.plot(predictions, label='LSTM Forecast', color='orange', linestyle='--', linewidth=0.8, alpha=0.8)
    plt.title('Full Predictions vs Actuals trên toàn bộ tập Test')
    plt.xlabel('Time Steps')
    plt.ylabel('Requests')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': {
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'mape': mape
        }
    }