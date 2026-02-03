"""
LSTM Model Architecture
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based time-series forecasting model
    
    Architecture:
        Input (batch_size, sequence_length, input_size) 
        → LSTM layers 
        → Last time step 
        → Dense layer 
        → Output (batch_size, 1)
    
    Parameters:
    -----------
    input_size : int
        Number of input features (default: 5)
        Features: [requests_target, error_rate, hour_sin, hour_cos, is_weekend]
    hidden_size : int
        Number of hidden units in LSTM (default: 64)
    num_layers : int
        Number of LSTM layers (default: 2)
    dropout : float
        Dropout probability (default: 0.2)
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output of shape (batch_size, 1)
        """
        device = x.device
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Get last time step output
        out = out[:, -1, :]
        
        # Dense layer
        out = self.fc(out)
        
        return out
