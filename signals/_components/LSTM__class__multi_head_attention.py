import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.dirname(current_dir)
signals_dir = os.path.dirname(components_dir)
if signals_dir not in sys.path:
    sys.path.insert(0, signals_dir)

from utilities._logger import setup_logging
# Initialize logger for LSTM Attention module
logger = setup_logging(module_name="_multi_head_attention", log_level=logging.DEBUG)

# Attention Mechanism Components
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for LSTM outputs
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Move scale to the same device as query
        if self.scale.device != query.device:
            self.scale = self.scale.to(query.device)
        
        # Linear transformations and split into num_heads
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention = self.attention(Q, K, V, mask, self.dropout)
        
        # Concatenate heads and put through final linear layer
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attention)
        
        return output
    
    def attention(self, query, key, value, mask, dropout):
        """Scaled dot-product attention"""
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        
        return torch.matmul(attention_weights, value)