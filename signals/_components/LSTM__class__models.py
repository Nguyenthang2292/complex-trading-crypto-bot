import logging
import sys
from typing import List, Literal, Optional
import torch
import torch.nn as nn

from pathlib import Path;
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__class__Models", log_level=logging.DEBUG)

from signals._components.LSTM__class__MultiheadAttention import MultiHeadAttention  
from signals._components.LSTM__class__FeedForward import FeedForward
from signals._components.LSTM__class__PositionalEncoding import PositionalEncoding

class LSTMModel(nn.Module):
    """PyTorch LSTM model for cryptocurrency price prediction with multi-layer architecture."""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 3, 
        num_classes: int = 3, 
        dropout: float = 0.3
    ) -> None:
        super().__init__()
        
        # Validation
        for name, value in [("input_size", input_size), ("hidden_size", hidden_size), ("num_classes", num_classes)]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout}")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with progressive dimension reduction
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 32, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(32, 16, num_layers=1, batch_first=True)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(0.2)
        
        # Classification layers
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM layers with progressive dimension reduction."""
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out = lstm_out[:, -1, :] # Take last timestep
        lstm_out = self.dropout3(lstm_out)
        
        out = self.relu(self.fc1(lstm_out))
        return self.softmax(self.fc2(out))

class LSTMAttentionModel(nn.Module):
    """LSTM model enhanced with Multi-Head Attention mechanism."""
    
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
        
        # Validation
        for name, value in [("input_size", input_size), ("hidden_size", hidden_size), 
                           ("num_classes", num_classes), ("num_heads", num_heads)]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout}")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        self.attention_dim = 16
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 32, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(32, self.attention_dim, num_layers=1, batch_first=True)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(0.2)
        
        # Adjust num_heads for compatibility - must be done before creating MultiHeadAttention
        original_num_heads = num_heads
        if self.attention_dim % num_heads != 0:
            # Find the largest divisor of attention_dim that's <= original num_heads
            valid_heads = [i for i in range(1, min(num_heads, self.attention_dim) + 1) 
                          if self.attention_dim % i == 0]
            num_heads = max(valid_heads) if valid_heads else 1
            logger.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} to be compatible with attention_dim {self.attention_dim}")
        
        # Attention components
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
        
        # Attention pooling and classification
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
        
        logger.model(f"LSTM-Attention model initialized with {num_heads} heads and {self.attention_dim}D attention")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM-Attention architecture."""
        # LSTM processing
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        lstm_out, _ = self.lstm3(lstm_out)
        
        # Positional encoding
        if self.use_positional_encoding:
            lstm_out = self.pos_encoding(lstm_out)
        
        # Self-attention mechanism
        attn_output = self.multihead_attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.layer_norm1(attn_output + lstm_out)
        
        # Feed-forward network
        ff_output = self.feed_forward(attn_output)
        ff_output = self.layer_norm2(ff_output + attn_output)
        
        # Attention pooling and classification
        attention_weights = self.attention_pooling(ff_output)
        pooled_output = torch.sum(ff_output * attention_weights, dim=1)
        
        return self.classifier(pooled_output)

class CNN1DExtractor(nn.Module):
    """1D CNN feature extractor for multi-scale temporal pattern extraction."""
    
    def __init__(
        self, 
        input_channels: int, 
        cnn_features: int = 64, 
        kernel_sizes: Optional[List[int]] = None, 
        dropout: float = 0.3
    ) -> None:
        super().__init__()
        
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        
        kernel_sizes = kernel_sizes or [3, 5, 7]
        self.input_channels = input_channels
        self.cnn_features = cnn_features
        
        # Calculate channel distribution for multi-scale convolution
        num_scales = len(kernel_sizes)
        base_channels = cnn_features // num_scales
        remainder = cnn_features % num_scales
        out_channels = [base_channels + (1 if i < remainder else 0) for i in range(num_scales)]
        
        # Multi-scale convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, out_channels[idx], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels[idx]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for idx, kernel_size in enumerate(kernel_sizes)
        ])
        
        # Feature refinement layers
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
        
        logger.model(f"CNN1D Extractor: {input_channels} channels → {cnn_features} features")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features from temporal sequences."""
        x = x.transpose(1, 2) # (batch, seq, features) → (batch, features, seq)
        
        # Multi-scale feature extraction
        conv_outputs = [conv_layer(x) for conv_layer in self.conv_layers]
        x = torch.cat(conv_outputs, dim=1)
        
        # Feature refinement
        x = self.conv_refine(x)
        return x.transpose(1, 2) # Back to (batch, seq, features)

class CNNLSTMAttentionModel(nn.Module):
    """Hybrid CNN-LSTM-Attention model for advanced sequence modeling."""
    
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
        
        # Validation
        for name, value in [("input_size", input_size), ("look_back", look_back)]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if output_mode not in ['classification', 'regression']:
            raise ValueError(f"output_mode must be 'classification' or 'regression', got {output_mode}")
        
        self.input_size = input_size
        self.look_back = look_back
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.output_mode = output_mode
        
        # CNN feature extraction
        self.cnn_extractor = CNN1DExtractor(input_size, cnn_features, dropout=dropout)
        
        # LSTM sequence modeling
        self.lstm1 = nn.LSTM(cnn_features, lstm_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden, lstm_hidden//2, batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        final_features = lstm_hidden // 2
        
        # Attention mechanism setup
        if use_attention:
            self.attention_dim = final_features
            
            # Adjust num_heads for compatibility - must be done before creating MultiHeadAttention
            original_num_heads = num_heads
            if self.attention_dim % num_heads != 0:
                # Find the largest divisor of attention_dim that's <= original num_heads
                valid_heads = [i for i in range(1, min(num_heads, self.attention_dim) + 1) 
                              if self.attention_dim % i == 0]
                num_heads = max(valid_heads) if valid_heads else 1
                logger.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} for compatibility")
            
            if use_positional_encoding:
                self.pos_encoding = PositionalEncoding(self.attention_dim, look_back)
            
            self.multihead_attention = MultiHeadAttention(self.attention_dim, num_heads, dropout)
            self.feed_forward = FeedForward(self.attention_dim, self.attention_dim * 2, dropout)
            
            self.layer_norm1 = nn.LayerNorm(self.attention_dim)
            self.layer_norm2 = nn.LayerNorm(self.attention_dim)
            
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.attention_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Output layer configuration
        hidden_dim = final_features // 2
        if output_mode == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(final_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(final_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
        
        logger.model(f"CNN-LSTM-Attention: look_back={look_back}, cnn_features={cnn_features}, "
                    f"lstm_hidden={lstm_hidden}, attention={use_attention}, mode={output_mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid CNN-LSTM-Attention architecture."""
        # CNN feature extraction
        cnn_features = self.cnn_extractor(x)
        
        # LSTM sequence modeling
        lstm_out, _ = self.lstm1(cnn_features)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        # Attention processing or simple pooling
        if self.use_attention:
            # Apply positional encoding if enabled
            if hasattr(self, 'pos_encoding'):
                lstm_out = self.pos_encoding(lstm_out)
            
            # Self-attention mechanism
            attn_output = self.multihead_attention(lstm_out, lstm_out, lstm_out)
            attn_output = self.layer_norm1(attn_output + lstm_out)
            
            # Feed-forward network
            ff_output = self.feed_forward(attn_output)
            ff_output = self.layer_norm2(ff_output + attn_output)
            
            # Attention pooling
            attention_weights = self.attention_pooling(ff_output)
            pooled_output = torch.sum(ff_output * attention_weights, dim=1)
        else:
            pooled_output = lstm_out[:, -1, :] # Take last timestep
        
        # Final prediction
        return self.classifier(pooled_output) if self.output_mode == 'classification' else self.regressor(pooled_output)