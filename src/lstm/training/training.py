"""
Training loops and model optimization
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

from .evaluation import calculate_metrics


def train_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Train for one epoch
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to train
    train_loader : DataLoader
        Training data
    criterion : loss function
        Loss function
    optimizer : torch.optim
        Optimizer
    device : str
        Device to train on
        
    Returns:
    --------
    float : Average loss for epoch
    """
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device='cpu'):
    """
    Validate model
    
    Parameters:
    -----------
    model : torch.nn.Module
    val_loader : DataLoader
    criterion : loss function
    device : str
        
    Returns:
    --------
    float : Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(model, train_loader, val_loader, epochs=20, lr=0.001,
         patience=5, model_save_path=None, device='cpu'):
    """
    Complete training loop with early stopping
    
    Parameters:
    -----------
    model : torch.nn.Module
        LSTM model
    train_loader : DataLoader
        Training data
    val_loader : DataLoader
        Validation data for early stopping
    epochs : int
        Max epochs (default: 20)
    lr : float
        Learning rate (default: 0.001)
    patience : int
        Early stopping patience (default: 5)
    model_save_path : str or Path
        Path to save best model
    device : str
        Device to train on ('cpu' or 'cuda')
        
    Returns:
    --------
    dict : {
        'model': trained model,
        'train_losses': list of train losses,
        'val_losses': list of val losses,
        'best_epoch': epoch with best val loss,
        'best_val_loss': best validation loss,
        'training_time': total time in seconds
    }
    """
    start_time = datetime.now()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Patience: {patience}")
    if model_save_path:
        print(f"  Save Path: {model_save_path}")
    print()
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
                print(f'Epoch [{epoch+1:3d}/{epochs}] ✓ Best! Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                print(f'Epoch [{epoch+1:3d}/{epochs}] ✓ Best! Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1:3d}/{epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} (Patience: {patience_counter}/{patience})')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early Stopping! No improvement for {patience} epochs.")
            print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            break
    
    # Load best model
    if model_save_path:
        print(f"\nLoading best model from epoch {best_epoch}...")
        model.load_state_dict(torch.load(model_save_path))
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.axvline(best_epoch - 1, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Learning Curve with Early Stopping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate training time
    training_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'training_time': training_time
    }
