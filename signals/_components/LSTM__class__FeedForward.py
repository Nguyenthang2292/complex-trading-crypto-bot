import logging
import sys
import torch
import torch.nn as nn

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None
from utilities._logger import setup_logging

# Initialize logger for LSTM Attention module
logger = setup_logging(module_name="LSTM__class__FeedForward", log_level=logging.DEBUG)

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network for transformer architectures.
    
    Applies two linear transformations with ReLU activation and dropout:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: Input and output dimension
        d_ff: Hidden dimension of the feed-forward layer
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))