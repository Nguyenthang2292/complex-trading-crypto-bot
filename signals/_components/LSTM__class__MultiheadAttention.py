import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None
from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__class__MultiheadAttention", log_level=logging.DEBUG)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for LSTM outputs.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate for attention weights
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)))
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply multi-head attention to input sequences.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional attention mask [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Attention output [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k, seq_len_v = key.size(1), value.size(1)
        
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.attention(Q, K, V, mask, self.dropout)
        
        attention_output = (attention_output.transpose(1, 2)
                          .contiguous()
                          .view(batch_size, seq_len_q, self.d_model))
        
        return self.W_o(attention_output)
    
    def attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor], 
        dropout: nn.Dropout
    ) -> torch.Tensor:
        """Scaled dot-product attention mechanism.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len_q, d_k]
            key: Key tensor [batch_size, num_heads, seq_len_k, d_k]
            value: Value tensor [batch_size, num_heads, seq_len_v, d_k]
            mask: Optional attention mask
            dropout: Dropout layer
            
        Returns:
            Attention weighted values [batch_size, num_heads, seq_len_q, d_k]
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        
        return torch.matmul(attention_weights, value)