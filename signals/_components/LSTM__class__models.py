import logging
import sys
from typing import List, Literal, Optional
import torch
import torch.nn as nn

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__class__models", log_level=logging.DEBUG)

from signals._components.LSTM__class__multi_head_attention import MultiHeadAttention  
from signals._components.LSTM__class__feed_foward import FeedForward
from signals._components.LSTM__class__positional_encoding import PositionalEncoding

class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for cryptocurrency price prediction.
    
    This model uses multiple LSTM layers with dropout for regularization
    and outputs probability distributions over 3 classes (SELL, NEUTRAL, BUY).
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension of LSTM layers (default: 64)
        num_layers: Number of LSTM layers (default: 3)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 3, 
        num_classes: int = 3, 
        dropout: float = 0.3
    ) -> None:
        super().__init__()
        
        if input_size <= 0:
            raise ValueError("input_size must be positive, got {0}".format(input_size))
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive, got {0}".format(hidden_size))
        if num_classes <= 0:
            raise ValueError("num_classes must be positive, got {0}".format(num_classes))
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0, got {0}".format(dropout))
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 32, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(32, 16, num_layers=1, batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output probabilities of shape (batch_size, num_classes)
        """
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        lstm_out3, _ = self.lstm3(lstm_out2)
        
        lstm_out = lstm_out3[:, -1, :]
        lstm_out = self.dropout3(lstm_out)
        
        out = self.relu(self.fc1(lstm_out))
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out
    

class LSTMAttentionModel(nn.Module):
    """
    PyTorch LSTM model with Multi-Head Attention for cryptocurrency price prediction.
    
    Architecture: Data → LSTM → Attention → Classification
    This model enhances LSTM with attention mechanism for better sequence modeling.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension of LSTM layers (default: 64)
        num_layers: Number of LSTM layers (default: 3)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
        num_heads: Number of attention heads (default: 8)
        use_positional_encoding: Whether to use positional encoding (default: True)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 3, 
        num_classes: int = 3,
        dropout: float = 0.3, 
        num_heads: int = 8, 
        use_positional_encoding: bool = True
    ) -> None:
        super().__init__()
        
        if input_size <= 0:
            raise ValueError("input_size must be positive, got {0}".format(input_size))
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive, got {0}".format(hidden_size))
        if num_classes <= 0:
            raise ValueError("num_classes must be positive, got {0}".format(num_classes))
        if num_heads <= 0:
            raise ValueError("num_heads must be positive, got {0}".format(num_heads))
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0, got {0}".format(dropout))
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 32, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(32, 16, num_layers=1, batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(0.2)
        
        self.attention_dim = 16
        
        if self.attention_dim % num_heads != 0:
            num_heads = min(num_heads, self.attention_dim)
            logger.warning("Adjusted num_heads to {0} to be compatible with attention_dim {1}".format(
                num_heads, self.attention_dim))
        
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(self.attention_dim)
        
        self.multihead_attention = MultiHeadAttention(
            d_model=self.attention_dim, 
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.feed_forward = FeedForward(
            d_model=self.attention_dim,
            d_ff=self.attention_dim * 2,
            dropout=dropout
        )
        
        self.layer_norm1 = nn.LayerNorm(self.attention_dim)
        self.layer_norm2 = nn.LayerNorm(self.attention_dim)
        
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes),
            nn.Softmax(dim=1)
        )
        
        logger.model("LSTM-Attention model initialized with {0} heads and {1}D attention".format(
            num_heads, self.attention_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM-Attention model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output probabilities of shape (batch_size, num_classes)
        """
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        lstm_out3, _ = self.lstm3(lstm_out2)
        
        if self.use_positional_encoding:
            lstm_out3 = self.pos_encoding(lstm_out3)
        
        attn_input = lstm_out3
        attn_output = self.multihead_attention(attn_input, attn_input, attn_input)
        attn_output = self.layer_norm1(attn_output + attn_input)
        
        ff_output = self.feed_forward(attn_output)
        ff_output = self.layer_norm2(ff_output + attn_output)
        
        attention_weights = self.attention_pooling(ff_output)
        pooled_output = torch.sum(ff_output * attention_weights, dim=1)
        
        output = self.classifier(pooled_output)
        
        return output


class CNN1DExtractor(nn.Module):
    """
    1D CNN feature extractor for time series data.
    
    Args:
        input_channels: Number of input channels
        cnn_features: Number of CNN features to extract (default: 64)
        kernel_sizes: List of kernel sizes for multi-scale convolution (default: [3, 5, 7])
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self, 
        input_channels: int, 
        cnn_features: int = 64, 
        kernel_sizes: List[int] = None, 
        dropout: float = 0.3
    ) -> None:
        super().__init__()
        
        if input_channels <= 0:
            raise ValueError("input_channels must be positive, got {0}".format(input_channels))
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
            
        self.input_channels = input_channels
        self.cnn_features = cnn_features
        
        num_scales = len(kernel_sizes)
        base = cnn_features // num_scales
        remainder = cnn_features % num_scales
        out_channels = []
        for i in range(num_scales):
            ch = base + (1 if i < remainder else 0)
            out_channels.append(ch)
        
        self.conv_layers = nn.ModuleList()
        for idx, kernel_size in enumerate(kernel_sizes):
            conv_block = nn.Sequential(
                nn.Conv1d(input_channels, out_channels[idx],
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels[idx]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_block)
        
        self.conv_refine = nn.Sequential(
            nn.Conv1d(cnn_features, cnn_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_features, cnn_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        logger.model("CNN1D Extractor initialized with {0} input channels, {1} features".format(
            input_channels, cnn_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN extractor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Extracted features of shape (batch_size, seq_len, cnn_features)
        """
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            conv_outputs.append(conv_out)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.conv_refine(x)
        x = x.transpose(1, 2)
        
        return x


class CNNLSTMAttentionModel(nn.Module):
    """
    Enhanced model: CNN feature extraction → LSTM sequence modeling → Attention → Classification/Regression.
    
    Pipeline:
    1. CNN 1D feature extraction
    2. LSTM processing  
    3. Multi-Head Attention
    4. Final Classification/Regression
    
    Args:
        input_size: Number of input features
        look_back: Sliding window size (default: 60)
        cnn_features: Number of CNN features (default: 64)
        lstm_hidden: LSTM hidden dimension (default: 32)
        num_layers: Number of LSTM layers (default: 2)
        num_classes: Number of output classes (default: 3)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.3)
        use_attention: Whether to use attention mechanism (default: True)
        use_positional_encoding: Whether to use positional encoding (default: True)
        output_mode: Output mode - 'classification' or 'regression' (default: 'classification')
    """
    
    def __init__(
        self, 
        input_size: int, 
        look_back: int = 60, 
        cnn_features: int = 64,
        lstm_hidden: int = 32, 
        num_layers: int = 2, 
        num_classes: int = 3,
        num_heads: int = 4, 
        dropout: float = 0.3,
        use_attention: bool = True, 
        use_positional_encoding: bool = True,
        output_mode: Literal['classification', 'regression'] = 'classification'
    ) -> None:
        super().__init__()
        
        if input_size <= 0:
            raise ValueError("input_size must be positive, got {0}".format(input_size))
        if look_back <= 0:
            raise ValueError("look_back must be positive, got {0}".format(look_back))
        if output_mode not in ['classification', 'regression']:
            raise ValueError("output_mode must be 'classification' or 'regression', got {0}".format(output_mode))
        
        self.input_size = input_size
        self.look_back = look_back
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.output_mode = output_mode
        
        self.cnn_extractor = CNN1DExtractor(
            input_channels=input_size,
            cnn_features=cnn_features,
            dropout=dropout
        )
        
        self.lstm1 = nn.LSTM(cnn_features, lstm_hidden, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden, lstm_hidden//2, num_layers=1, batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if use_attention:
            self.attention_dim = lstm_hidden // 2
            
            if self.attention_dim % num_heads != 0:
                num_heads = min(num_heads, self.attention_dim)
                logger.warning("Adjusted num_heads to {0} for compatibility".format(num_heads))
            
            if use_positional_encoding:
                self.pos_encoding = PositionalEncoding(self.attention_dim, max_seq_length=look_back)
            
            self.multihead_attention = MultiHeadAttention(
                d_model=self.attention_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
            self.feed_forward = FeedForward(
                d_model=self.attention_dim,
                d_ff=self.attention_dim * 2,
                dropout=dropout
            )
            
            self.layer_norm1 = nn.LayerNorm(self.attention_dim)
            self.layer_norm2 = nn.LayerNorm(self.attention_dim)
            
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.attention_dim, 1),
                nn.Softmax(dim=1)
            )
            
            final_features = self.attention_dim
        else:
            final_features = lstm_hidden // 2
        
        if output_mode == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(final_features, final_features//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(final_features//2, num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(final_features, final_features//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(final_features//2, 1),
                nn.Tanh()
            )
        
        logger.model("CNN-LSTM-Attention model initialized:")
        logger.model("  - Look back: {0}, CNN features: {1}".format(look_back, cnn_features))
        logger.model("  - LSTM hidden: {0}, Attention: {1}".format(lstm_hidden, use_attention))
        logger.model("  - Output mode: {0}".format(output_mode))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN-LSTM-Attention model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            For classification: probabilities of shape (batch_size, num_classes)
            For regression: predictions of shape (batch_size, 1)
        """
        cnn_features = self.cnn_extractor(x)
        
        lstm_out1, _ = self.lstm1(cnn_features)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        if self.use_attention:
            if hasattr(self, 'pos_encoding'):
                lstm_out2 = self.pos_encoding(lstm_out2)
            
            attn_input = lstm_out2
            attn_output = self.multihead_attention(attn_input, attn_input, attn_input)
            attn_output = self.layer_norm1(attn_output + attn_input)
            
            ff_output = self.feed_forward(attn_output)
            ff_output = self.layer_norm2(ff_output + attn_output)
            
            attention_weights = self.attention_pooling(ff_output)
            pooled_output = torch.sum(ff_output * attention_weights, dim=1)
        else:
            pooled_output = lstm_out2[:, -1, :]
        
        if self.output_mode == 'classification':
            output = self.classifier(pooled_output)
        else:
            output = self.regressor(pooled_output)
        
        return output