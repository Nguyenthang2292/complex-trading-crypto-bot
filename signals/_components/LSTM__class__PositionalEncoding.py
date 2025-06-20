import logging
import numpy as np
import sys
import torch
import torch.nn as nn

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__class__PositionalEncoding", log_level=logging.DEBUG)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data in transformer-based models.
    
    Adds position information to the input embedding to help the model understand
    sequence order. Uses sine and cosine functions of different frequencies for
    position encoding as described in 'Attention Is All You Need' paper.
    
    Args:
        d_model: Dimension of the model embeddings
        max_seq_length: Maximum sequence length to pre-compute positions for
        
    Raises:
        ValueError: If d_model or max_seq_length is not positive
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:  # Handle odd d_model
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Store as buffer without extra dimensions
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_seq_length, d_model]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        
        Args:
            x: Input tensor with shape [batch, seq_len, d_model]
            
        Returns:
            torch.Tensor: Input with positional encoding added
            
        Raises:
            ValueError: If input is not a 3D tensor
        """
        if x.dim() != 3:
            raise ValueError(f"Input must be 3D tensor [batch, seq_len, d_model], got shape {x.shape}")
            
        seq_len = x.size(1)
        device = x.device
        
        # If sequence length exceeds stored positional encoding, compute extra positions
        if seq_len > self.pe.size(1):                                                                               # type: ignore
            extra_len = seq_len - self.pe.size(1)                                                                   # type: ignore
            extra_positions = torch.arange(self.pe.size(1), seq_len, dtype=torch.float, device=device).unsqueeze(1) # type: ignore
            d_model = self.pe.size(2)                                                                               # type: ignore
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model))
            extra_pe = torch.zeros(extra_len, d_model, device=device)
            extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
            if d_model % 2 == 1:
                extra_pe[:, 1::2] = torch.cos(extra_positions * div_term[:-1])
            else:
                extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
            full_pe = torch.cat([self.pe.squeeze(0).to(device), extra_pe], dim=0).unsqueeze(0)  # type: ignore
        else:
            full_pe = self.pe.to(device)
            
        return x + full_pe[:, :seq_len, :]                                                      # type: ignore