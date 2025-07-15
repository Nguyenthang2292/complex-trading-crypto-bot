# ====================================================================
# TRANSFORMER MODEL FOR CRYPTO TRADING SIGNALS
# ====================================================================

"""
Transformer-based time series forecasting model for cryptocurrency trading signals.

This module implements a transformer architecture optimized for time series 
forecasting in cryptocurrency trading. The model uses self-attention mechanisms
to capture complex temporal patterns and generate trading signals.

Key Features:
- Self-attention mechanism for temporal pattern recognition
- Positional encodings for sequence order awareness  
- GPU acceleration with Tensor Core optimization
- Mixed precision training for faster convergence
- Comprehensive bias detection and threshold optimization
- Safe model saving/loading with PyTorch 2.6+ compatibility

Usage:
    # Train a transformer model
    model, path = train_and_save_transformer_model(df_market_data)
    
    # Load a trained model
    model, scaler, features, target_idx = load_transformer_model("path/to/model.pth")
    
    # Generate trading signal
    signal = get_latest_transformer_signal(df_new_data, model, scaler, features, target_idx)
"""

import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import autocast and GradScaler with version compatibility
try:
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
    except ImportError:
        print("Warning: AMP not available in this PyTorch version")
        autocast = None
        GradScaler = None

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities._logger import setup_logging
from utilities._gpu_resource_manager import get_gpu_resource_manager
logger = setup_logging(module_name="signals_transformer", log_level=logging.DEBUG)

from components._generate_indicator_features import generate_indicator_features
from components.config import (
    BUY_THRESHOLD, COL_CLOSE, CPU_MODEL_CONFIG, DEFAULT_EPOCHS,
    FUTURE_RETURN_SHIFT, GPU_MODEL_CONFIG, MIN_DATA_POINTS, 
    MODEL_FEATURES, MODELS_DIR, SELL_THRESHOLD, SIGNAL_LONG, 
    SIGNAL_NEUTRAL, SIGNAL_SHORT, TRANSFORMER_MODEL_FILENAME
)

# Check GPU availability
gpu_available: bool = get_gpu_resource_manager().is_cuda_available

def setup_safe_globals():
    """Setup safe globals for PyTorch serialization to handle numpy compatibility."""
    try:
        import numpy as np
        safe_globals_list = [np.ndarray, np.dtype]
        
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                torch.serialization.add_safe_globals(safe_globals_list)
                logger.debug("Safe globals configured for PyTorch serialization")
            except Exception as e:
                logger.warning(f"Error setting up safe globals: {e}")
        else:
            logger.debug("PyTorch version does not support add_safe_globals")
            
    except Exception as e:
        logger.debug(f"Could not setup safe globals: {e}")

# Initialize safe globals
setup_safe_globals()

def safe_memory_division(value: Union[int, str, float], divisor: int) -> float:
    """Safely divide memory values, handling different types."""
    if isinstance(value, (int, float)):
        return float(value) / divisor
    elif isinstance(value, str):
        try:
            return float(value) / divisor
        except (ValueError, TypeError):
            return 0.0
    return 0.0

def safe_memory_comparison(value: Union[int, str, float], threshold: int) -> bool:
    """Safely compare memory values, handling different types."""
    if isinstance(value, (int, float)):
        return float(value) > threshold
    elif isinstance(value, str):
        try:
            return float(value) > threshold
        except (ValueError, TypeError):
            return False
    return False

def safe_nan_to_num(features: np.ndarray) -> np.ndarray:
    """Safely handle nan_to_num with proper type checking."""
    try:
        if isinstance(features, np.ndarray):
            return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            features_array = np.array(features)
            return np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception as e:
        logger.warning(f"Error in nan_to_num: {e}, using fallback")
        return np.zeros_like(features) if hasattr(features, 'shape') else np.array([])

def safe_save_model(checkpoint: Dict[str, Any], model_path: str) -> bool:
    """Safely save PyTorch model with compatibility for PyTorch 2.6+."""
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, model_path, pickle_protocol=4)
        logger.debug(f"Model saved successfully to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        return False

def safe_load_model(model_path: str) -> Optional[Dict[str, Any]]:
    """Safely load PyTorch model with fallback strategies for PyTorch 2.6+ compatibility."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        logger.debug("Model loaded with weights_only=False")
        return checkpoint
    except Exception as e:
        logger.warning(f"Loading with weights_only=False failed: {e}")
    try:
        import numpy as np
        try:
            with torch.serialization.safe_globals([np.ndarray, np.dtype]):
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.debug("Model loaded with safe globals context manager")
                return checkpoint
        except (AttributeError, ImportError):
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([np.ndarray, np.dtype])
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.debug("Model loaded with add_safe_globals")
                return checkpoint
            else:
                raise ValueError("Neither safe_globals nor add_safe_globals available")
    except Exception as e:
        logger.warning(f"Loading with safe_globals approach failed: {e}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.debug("Model loaded with legacy method")
        return checkpoint
    except Exception as e:
        logger.error(f"All loading methods failed: {e}")
        return None

def select_and_scale_features(
    df: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, MinMaxScaler, List[str]]:
    """Select features and apply MinMax scaling for model training.
    
    Args:
        df: DataFrame with technical indicators
        feature_cols: Feature columns to use (defaults to MODEL_FEATURES + ma_20_slope)
    
    Returns:
        Tuple of (scaled_data, scaler, feature_columns_used)
    
    Raises:
        ValueError: If no valid feature columns found
    """
    if feature_cols is None:
        feature_cols = MODEL_FEATURES.copy() + ['ma_20_slope']

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.data(f"Missing columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
        
    if not feature_cols:
        raise ValueError("No valid feature columns found in DataFrame")

    logger.data(f"Using {len(feature_cols)} features")
    
    try:
        data = df[feature_cols].values
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        logger.data(f"Data scaled successfully, shape: {data_scaled.shape}")
        return data_scaled, scaler, feature_cols
    except Exception as e:
        logger.error(f"Error in scaling: {e}")
        raise
    
class CryptoDataset(Dataset):
    """Dataset for preprocessed time series sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize with preprocessed sequences and targets.
        
        Args:
            X: Preprocessed input sequences of shape (num_samples, seq_length, num_features)
            y: Preprocessed targets of shape (num_samples, prediction_length)
        """
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a preprocessed sequence-target pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_sequence, target_values)
        """
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting with self-attention mechanism.
    
    Implements a transformer-based architecture optimized for time series data,
    with positional encodings and configurable hyperparameters.
    """
    
    def __init__(
        self,
        feature_size: int = GPU_MODEL_CONFIG.feature_size,
        num_layers: int = GPU_MODEL_CONFIG.num_layers,
        d_model: int = GPU_MODEL_CONFIG.d_model,
        nhead: int = GPU_MODEL_CONFIG.nhead,
        dim_feedforward: int = GPU_MODEL_CONFIG.dim_feedforward,
        dropout: float = GPU_MODEL_CONFIG.dropout,
        seq_length: int = GPU_MODEL_CONFIG.seq_length,
        prediction_length: int = GPU_MODEL_CONFIG.prediction_length
    ) -> None:
        """Initialize transformer model with specified configuration.

        Args:
            feature_size: Number of input features
            num_layers: Number of transformer encoder layers
            d_model: Internal model dimension size
            nhead: Number of attention heads
            dim_feedforward: Feed-forward network hidden dimension
            dropout: Dropout rate for regularization
            seq_length: Input sequence length
            prediction_length: Number of future steps to predict
        """
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, prediction_length)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer model.
        
        Args:
            src: Input tensor of shape [batch_size, seq_length, feature_size]

        Returns:
            Predicted values tensor of shape [batch_size, prediction_length]
        """
        _, seq_len, _ = src.shape
        src = self.input_fc(src)
        src = src + self.pos_embedding[:, :seq_len, :]
        encoded = self.transformer_encoder(src)
        last_step = encoded[:, -1, :]
        return self.fc_out(last_step)

# ===================== PREPROCESSING PIPELINE =====================
def preprocess_transformer_data(
    df_input: pd.DataFrame, 
    seq_length: int = GPU_MODEL_CONFIG.seq_length,
    prediction_length: int = GPU_MODEL_CONFIG.prediction_length,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, List[str]]:
    """Preprocess data for transformer model with sliding window approach.
    
    Args:
        df_input: Input DataFrame with market data
        seq_length: Length of input sequences
        prediction_length: Number of future steps to predict
        feature_cols: Feature columns to use (defaults to MODEL_FEATURES + ma_20_slope)
        
    Returns:
        Tuple of (X_sequences, y_targets, scaler, feature_names)
    """
    logger.model(f"Starting transformer preprocessing: {df_input.shape} rows, seq_length={seq_length}")
    
    if df_input.empty or len(df_input) < seq_length + prediction_length + 10:
        logger.error(f"Insufficient data: {len(df_input)} rows, need at least {seq_length + prediction_length + 10}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    df = generate_indicator_features(df_input.copy())
    if df.empty:
        logger.error("Feature calculation returned empty DataFrame")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    if feature_cols is None:
        feature_cols = MODEL_FEATURES.copy() + ['ma_20_slope']
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.data(f"Missing columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
        
    if not feature_cols:
        logger.error("No valid feature columns found in DataFrame")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    initial_len = len(df)
    df.dropna(inplace=True)
    if len(df) < seq_length + prediction_length + 1:
        logger.error(f"Insufficient data after cleanup: {len(df)} rows (dropped {initial_len - len(df)} NaN)")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    features = df[feature_cols].values
    
    if np.isnan(features).any() or np.isinf(features).any():
        logger.warning("Cleaning invalid values in features")
        features = safe_nan_to_num(features.astype(np.float64))
    
    scaler = MinMaxScaler()
    try:
        scaled_features = scaler.fit_transform(features)
    except Exception as e:
        logger.error(f"Feature scaling failed: {e}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    # Create targets (future price changes)
    target_col_idx = feature_cols.index(COL_CLOSE)
    X_sequences, y_targets = [], []
    
    for i in range(seq_length, len(scaled_features) - prediction_length + 1):
        sequence = scaled_features[i-seq_length:i]
        target = scaled_features[i:i+prediction_length, target_col_idx]
        
        if (sequence.shape[0] == seq_length and 
            target.shape[0] == prediction_length and 
            not (np.isnan(sequence).any() or np.isinf(sequence).any() or 
                 np.isnan(target).any() or np.isinf(target).any())):
            X_sequences.append(sequence)
            y_targets.append(target)
    
    if not X_sequences:
        logger.error("No valid sequences created after filtering")
        return np.array([]), np.array([]), scaler, feature_cols
    
    X_sequences, y_targets = np.array(X_sequences), np.array(y_targets)
    
    logger.model(f"Preprocessing complete: {len(X_sequences)} sequences, shape {X_sequences.shape}")
    logger.model(f"Target range: [{np.min(y_targets):.4f}, {np.max(y_targets):.4f}]")
    
    return X_sequences, y_targets, scaler, feature_cols

def train_transformer_model(
    model: TimeSeriesTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    lr: float = 1e-3,
    epochs: int = DEFAULT_EPOCHS,
    device: str = 'cpu',
    use_early_stopping: bool = True,
    patience: int = 10,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
) -> Tuple[TimeSeriesTransformer, Dict[str, List[float]]]:
    """Train transformer model with optimizations and monitoring.
    
    Args:
        model: Transformer model instance to train
        train_loader: DataLoader for training batches 
        val_loader: Optional validation set DataLoader
        lr: Initial learning rate
        epochs: Maximum number of training epochs
        device: Target device ('cpu' or 'cuda')
        use_early_stopping: Enable multi-metric early stopping
        patience: Early stopping patience epochs
        use_mixed_precision: Enable automatic mixed precision
        gradient_accumulation_steps: Steps before optimizer update

    Returns:
        Tuple containing trained model and training history with metrics
    """
    gpu_manager = get_gpu_resource_manager()
    
    device_id = 0 if device == 'cuda' else 0
    with gpu_manager.gpu_scope(device_id=device_id) as gpu_device:
        actual_device = gpu_device if gpu_device is not None else torch.device('cpu')
        use_gpu = actual_device.type == 'cuda'
        
        use_amp = use_mixed_precision and use_gpu and autocast is not None and GradScaler is not None
        tensor_info = gpu_manager.get_tensor_core_info() if use_gpu else {'has_tensor_cores': False, 'generation': 'N/A'}
        has_tensor_cores = tensor_info['has_tensor_cores']
        
        if use_amp:
            if has_tensor_cores:
                logger.gpu("Using Automatic Mixed Precision (AMP) with Tensor Core acceleration")
            elif torch.cuda.get_device_capability(0)[0] >= 7:
                logger.gpu("Using Automatic Mixed Precision (AMP) for faster training")
            else:
                use_amp = False
                logger.warning("GPU doesn't support mixed precision, falling back to FP32")
        elif use_mixed_precision and use_gpu:
            logger.warning("AMP requested but not available in this PyTorch version, falling back to FP32")
            use_amp = False
        
        if use_gpu:
            memory_info = gpu_manager.get_memory_info()
            logger.gpu(f"Training on GPU: {torch.cuda.get_device_name()}")
            total_gb = safe_memory_division(memory_info['total'], 1024**3)
            allocated_gb = safe_memory_division(memory_info['allocated'], 1024**3)
            cached_gb = safe_memory_division(memory_info['cached'], 1024**3)
            logger.memory(f"GPU Memory - Total: {total_gb:.1f} GB, "
                         f"Allocated: {allocated_gb:.2f} GB, "
                         f"Cached: {cached_gb:.2f} GB")
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            if has_tensor_cores:
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                logger.gpu(f"Tensor Core optimizations enabled for {tensor_info['generation']} generation")
                
                try:
                    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                        torch.backends.cuda.enable_flash_sdp(True)
                        logger.gpu("Flash Attention enabled for Tensor Cores")
                except Exception as e:
                    logger.debug(f"Flash Attention not available: {e}")
            
            if safe_memory_comparison(memory_info['total'], 8 * 1024**3):
                torch.cuda.empty_cache()
                warmup_tensor_size = 64 if has_tensor_cores else 60
                dummy_tensor = torch.randn(1, warmup_tensor_size, 10, device=actual_device)
                del dummy_tensor
                torch.cuda.empty_cache()
        else:
            logger.config("Training on CPU")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, min_lr=1e-7
        )
        
        scaler = None
        if use_amp and GradScaler is not None:
            try:
                scaler = GradScaler()
            except Exception as e:
                logger.warning(f"Failed to initialize GradScaler: {e}")
                use_amp = False
                scaler = None
        
        model.to(actual_device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.model(f"Total parameters: {total_params:,}")
        logger.model(f"Trainable parameters: {trainable_params:,}")
        logger.config(f"Device: {actual_device}, Mixed Precision: {use_amp}, Gradient Accumulation: {gradient_accumulation_steps}")
        
        best_val_loss = float('inf')
        best_val_mae = float('inf')
        best_combined_score = float('-inf')
        patience_counter = 0
        best_model_state = None
        
        warmup_epochs = max(5, epochs // 10)
        initial_lr = lr
        
        def get_lr_for_epoch(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return initial_lr * ((current_epoch + 1) / warmup_epochs)
            return initial_lr
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'combined_scores': [],
            'learning_rates': []
        }
        
        logger.model(f"Starting training for {epochs} epochs (warmup: {warmup_epochs} epochs)...")
        
        for epoch in range(epochs):
            if use_gpu:
                torch.cuda.empty_cache()
                
            if epoch < warmup_epochs:
                new_lr = get_lr_for_epoch(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if epoch == 0:
                    logger.config(f"Learning rate warmup: {new_lr:.2e} -> {initial_lr:.2e} over {warmup_epochs} epochs")
                    
            model.train()
            train_losses = []
            optimizer.zero_grad()
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(actual_device, non_blocking=True)
                y_batch = y_batch.to(actual_device, non_blocking=True)

                if use_amp and scaler is not None and autocast is not None:
                    try:
                        with autocast(device_type='cuda'): # type: ignore
                            output = model(x_batch)
                            loss = criterion(output, y_batch)
                            loss = loss / gradient_accumulation_steps
                    except TypeError:
                        with autocast(device_type='cuda'): # type: ignore
                            output = model(x_batch)
                            loss = criterion(output, y_batch)
                            loss = loss / gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    accumulate_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                    is_last_batch = (batch_idx + 1) == len(train_loader)
                    
                    if accumulate_step or is_last_batch:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                    train_losses.append(loss.item() * gradient_accumulation_steps)
                else:
                    output = model(x_batch)
                    loss = criterion(output, y_batch)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    accumulate_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                    is_last_batch = (batch_idx + 1) == len(train_loader)
                    
                    if accumulate_step or is_last_batch:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    train_losses.append(loss.item() * gradient_accumulation_steps)
                
                if use_gpu and batch_idx % 20 == 0:
                    memory_info = gpu_manager.get_memory_info()
                    if batch_idx % 100 == 0:
                        allocated_gb = safe_memory_division(memory_info['allocated'], 1024**3)
                        cached_gb = safe_memory_division(memory_info['cached'], 1024**3)
                        logger.memory(f"Epoch {epoch+1}, Batch {batch_idx}: GPU Memory - "
                                    f"Allocated: {allocated_gb:.2f}GB, "
                                    f"Cached: {cached_gb:.2f}GB")

            mean_train_loss = np.mean(train_losses)
            training_history['train_loss'].append(mean_train_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            if val_loader is not None:
                model.eval()
                val_losses = []
                val_maes = []
                
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(actual_device, non_blocking=True)
                        y_val = y_val.to(actual_device, non_blocking=True)
                        
                        if use_amp and autocast is not None:
                            try:
                                with autocast(device_type='cuda'): # type: ignore
                                    output_val = model(x_val)   
                                    loss_val = criterion(output_val, y_val)
                            except TypeError:
                                with autocast(device_type='cuda'): # type: ignore
                                    output_val = model(x_val)
                                    loss_val = criterion(output_val, y_val)
                        else:
                            output_val = model(x_val)
                            loss_val = criterion(output_val, y_val)
                        
                        val_losses.append(loss_val.item())
                        
                        mae_val = torch.mean(torch.abs(output_val - y_val)).item()
                        val_maes.append(mae_val)
                
                mean_val_loss = np.mean(val_losses)
                mean_val_mae = np.mean(val_maes)
                
                training_history['val_loss'].append(mean_val_loss)
                training_history['val_mae'].append(mean_val_mae)
                
                normalized_loss = 1.0 / (1.0 + mean_val_loss)
                normalized_mae = 1.0 / (1.0 + mean_val_mae)
                combined_score = 0.6 * normalized_loss + 0.4 * normalized_mae
                training_history['combined_scores'].append(combined_score)
                
                if epoch >= warmup_epochs:
                    prev_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(mean_val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    if current_lr != prev_lr:
                        logger.model(f"Learning rate reduced from {prev_lr:.2e} to {current_lr:.2e} (ReduceLROnPlateau)")
                
                if use_early_stopping:
                    improved = False
                    improvement_reasons = []
                    
                    if mean_val_loss < best_val_loss - 1e-6:
                        best_val_loss = mean_val_loss
                        improved = True
                        improvement_reasons.append("loss")
                    
                    if mean_val_mae < best_val_mae - 1e-6:
                        best_val_mae = mean_val_mae
                        improved = True
                        improvement_reasons.append("mae")
                    
                    if combined_score > best_combined_score + 1e-6:
                        best_combined_score = combined_score
                        improved = True
                        improvement_reasons.append("combined")
                    
                    if improved:
                        patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        logger.performance(f"✅ New best model saved (improvements: {', '.join(improvement_reasons)}, "
                                         f"Combined Score: {combined_score:.6f})")
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.model(f"🛑 Early stopping triggered after {epoch+1} epochs (patience: {patience})")
                        if best_model_state is not None:
                            model.load_state_dict({k: v.to(actual_device) for k, v in best_model_state.items()})
                            logger.model("🔄 Restored best model state")
                        break
                
                memory_info_str = ""
                if use_gpu:
                    memory_info = gpu_manager.get_memory_info()
                    allocated_gb = safe_memory_division(memory_info['allocated'], 1024**3)
                    memory_info_str = f", GPU Mem: {allocated_gb:.2f}GB"
                
                current_lr = optimizer.param_groups[0]['lr']
                early_stop_info = f", Patience: {patience_counter}/{patience}" if use_early_stopping else ""
                
                logger.performance(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {mean_train_loss:.6f}, "
                                 f"Val Loss: {mean_val_loss:.6f}, Val MAE: {mean_val_mae:.6f}, "
                                 f"Combined: {combined_score:.6f}, LR: {current_lr:.2e}{early_stop_info}{memory_info_str}")
            else:
                memory_info_str = ""
                if use_gpu:
                    memory_info = gpu_manager.get_memory_info()
                    allocated_gb = safe_memory_division(memory_info['allocated'], 1024**3)
                    memory_info_str = f", GPU Mem: {allocated_gb:.2f}GB"
                
                current_lr = optimizer.param_groups[0]['lr']
                logger.performance(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {mean_train_loss:.6f}, "
                                 f"LR: {current_lr:.2e}{memory_info_str}")
        
        if training_history['val_loss']:
            final_val_loss = training_history['val_loss'][-1]
            best_val_loss_epoch = np.argmin(training_history['val_loss']) + 1
            best_combined_epoch = np.argmax(training_history['combined_scores']) + 1
            logger.success(f"📊 Training completed - Final Val Loss: {final_val_loss:.6f}, "
                          f"Best Val Loss: {min(training_history['val_loss']):.6f} (Epoch {best_val_loss_epoch})")
            logger.success(f"📊 Best Combined Score: {max(training_history['combined_scores']):.6f} (Epoch {best_combined_epoch})")
            
            if len(training_history['val_loss']) > 1:
                loss_improvement = (training_history['val_loss'][0] - min(training_history['val_loss'])) / training_history['val_loss'][0] * 100
                logger.analysis(f"📈 Loss improvement: {loss_improvement:.2f}%")
        else:
            final_train_loss = training_history['train_loss'][-1]
            logger.success(f"📊 Training completed - Final Train Loss: {final_train_loss:.6f}")

        return model, training_history

def train_and_save_transformer_model(
    df_input: pd.DataFrame, 
    model_filename: Optional[str] = None
) -> Tuple[Optional[TimeSeriesTransformer], str]:
    """Train and save a transformer model with bias detection and optimization.
    
    Args:
        df_input: Input DataFrame with market data
        model_filename: Optional filename for saving the model
        
    Returns:
        Tuple of (trained_model, model_path) or (None, "") if training fails
    """
    try:
        if df_input.empty:
            logger.error("Input DataFrame is empty")
            return None, ""
        logger.model(f"Training transformer with {len(df_input)} rows")
        if len(df_input) < MIN_DATA_POINTS:
            logger.error(f"Insufficient data for training: {len(df_input)} < {MIN_DATA_POINTS}")
            return None, ""
        gpu_manager = get_gpu_resource_manager()
        gpu_available = gpu_manager.is_cuda_available
        
        actual_device = get_optimal_device()
        has_tensor_cores = gpu_manager.get_tensor_core_info()['has_tensor_cores']
        
        logger.config("Starting Transformer model training...")
        logger.config(f"GPU available: {gpu_available}")
        logger.config(f"Using device: {actual_device}")

        # If using CPU, fall back to CPU-specific model architecture
        if actual_device.type == 'cpu':
            logger.warning("Falling back to CPU configuration for Transformer model")
            model_config = CPU_MODEL_CONFIG
        else:
            model_config = GPU_MODEL_CONFIG
        
        feature_cols = MODEL_FEATURES + ['ma_20_slope']
        X, y, scaler, feature_names = preprocess_transformer_data(
            df_input,
            seq_length=model_config.seq_length,
            prediction_length=model_config.prediction_length
        )
        if len(X) == 0:
            logger.error("Data preprocessing failed - no valid sequences created")
            return None, ""
        
        target_idx = feature_cols.index(COL_CLOSE)
        dataset = CryptoDataset(X, y)
        
        if len(dataset) < 100:
            logger.error(f"Insufficient samples for training: {len(dataset)}")
            return None, ""
        
        n = len(dataset)
        n_train, n_val = int(n * 0.8), int(n * 0.1)
        train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
        val_ds = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
        
        batch_size = 32
        if gpu_available:
            memory_info = gpu_manager.get_memory_info()
            total_memory_gb = safe_memory_division(memory_info['total'], 1024**3)
            if total_memory_gb >= 24:
                base_batch_size = 128
            elif total_memory_gb >= 16:
                base_batch_size = 96
            elif total_memory_gb >= 12:
                base_batch_size = 80
            elif total_memory_gb >= 8:
                base_batch_size = 64
            else:
                base_batch_size = 48
            if has_tensor_cores:
                batch_size = ((base_batch_size + 7) // 8) * 8
                logger.gpu(f"Using Tensor Core optimized batch size: {batch_size} (GPU Memory: {total_memory_gb:.1f}GB)")
            else:
                batch_size = base_batch_size
                logger.gpu(f"Using GPU-optimized batch size: {batch_size} (GPU Memory: {total_memory_gb:.1f}GB)")
        
        pin_memory = gpu_available
        cpu_count = os.cpu_count() or 2
        num_workers = 0 if gpu_available else min(4, cpu_count // 2)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                   pin_memory=pin_memory, num_workers=num_workers, 
                                   persistent_workers=True if num_workers > 0 else False)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                               pin_memory=pin_memory, num_workers=num_workers,
                               persistent_workers=True if num_workers > 0 else False)
        
        device = 'cuda' if gpu_available else 'cpu'
        logger.config(f"Training on device: {device}")
        
        model = TimeSeriesTransformer(
            feature_size=len(feature_cols),
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            nhead=model_config.nhead,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            seq_length=model_config.seq_length,
            prediction_length=model_config.prediction_length
        )
        
        model, history = train_transformer_model(
            model,
            train_loader,
            val_loader=val_loader,
            device=device
        )
        
        if not model_filename:
            model_filename = "transformer_model_global.pth"
        
        model_path = str(MODELS_DIR / model_filename)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'feature_cols': feature_cols,
            'target_col_idx': target_idx,
            'training_history': history,
            'model_config': {
                'feature_size': len(feature_cols),
                'num_layers': model_config.num_layers,
                'd_model': model_config.d_model,
                'nhead': model_config.nhead,
                'dim_feedforward': model_config.dim_feedforward,
                'dropout': model_config.dropout,
                'seq_length': model_config.seq_length,
                'prediction_length': model_config.prediction_length
            }
        }
        
        if safe_save_model(checkpoint, model_path):
            logger.success(f"Transformer model saved to {model_path}")
        else:
            logger.error(f"Failed to save Transformer model to {model_path}")
            return None, ""
        
        return model, model_path
        
    except Exception as e:
        import traceback
        logger.error(f"Error training transformer model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, ""

def get_latest_transformer_signal(
    df_market_data: pd.DataFrame, 
    model: TimeSeriesTransformer, 
    scaler: MinMaxScaler, 
    feature_cols: List[str], 
    target_col_idx: int, 
    device: str = 'cpu', 
    suggested_thresholds: Optional[Tuple[float, float]] = None
) -> str:
    """Generate trading signal using transformer model predictions.
    
    Args:
        df_market_data: Market data DataFrame
        model: Trained transformer model
        scaler: Fitted MinMaxScaler instance
        feature_cols: List of feature column names
        target_col_idx: Index of target column
        device: Device for inference
        suggested_thresholds: Optional custom buy/sell thresholds
        
    Returns:
        Trading signal: 'LONG', 'SHORT', or 'NEUTRAL'
    """
    try:
        if df_market_data is None or df_market_data.empty:
            logger.warning("Input DataFrame is empty")
            return SIGNAL_NEUTRAL
            
        if model is None:
            logger.error("Model is None")
            return SIGNAL_NEUTRAL
            
        if scaler is None:
            logger.error("Scaler is None")
            return SIGNAL_NEUTRAL
            
        if not feature_cols:
            logger.error("Feature columns list is empty")
            return SIGNAL_NEUTRAL
        
        symbol_name = 'UNKNOWN'
        if hasattr(df_market_data, 'symbol'):
            symbol_name = df_market_data.symbol
        elif 'symbol' in df_market_data.columns:
            symbol_name = df_market_data['symbol'].iloc[0]
        
        logger.signal(f"[{symbol_name}] Starting transformer signal generation...")
        
        df_with_features = generate_indicator_features(df_market_data.copy())
        if df_with_features.empty:
            logger.warning(f"[{symbol_name}] DataFrame became empty after feature calculation")
            return SIGNAL_NEUTRAL

        seq_length = model.pos_embedding.shape[1]
        if len(df_with_features) < seq_length:
            logger.warning(f"[{symbol_name}] Insufficient data for prediction: {len(df_with_features)} < {seq_length}")
            return SIGNAL_NEUTRAL

        available_features = [col for col in feature_cols if col in df_with_features.columns]
        if len(available_features) != len(feature_cols):
            missing_features = set(feature_cols) - set(available_features)
            logger.warning(f"[{symbol_name}] Missing features: {missing_features}")
            return SIGNAL_NEUTRAL

        latest_sequence = df_with_features[available_features].iloc[-seq_length:].values
        latest_sequence_scaled = scaler.transform(latest_sequence)
        
        input_tensor = torch.tensor(latest_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        current_price = df_with_features[COL_CLOSE].iloc[-1]
        
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(input_tensor).cpu().numpy()[0, 0]
        
        dummy_pred = np.zeros((1, len(feature_cols)))
        dummy_pred[:, target_col_idx] = prediction_scaled
        predicted_price = scaler.inverse_transform(dummy_pred)[0, target_col_idx]
        
        price_change = (predicted_price - current_price) / current_price
        buy_thr, sell_thr = suggested_thresholds if suggested_thresholds else (BUY_THRESHOLD, SELL_THRESHOLD)
        logger.signal(f"[{symbol_name}] Current: {current_price:.6f}, Predicted: {predicted_price:.6f}")
        logger.analysis(f"[{symbol_name}] Price change: {price_change:.6f} ({price_change:.2%})")
        logger.config(f"[{symbol_name}] Thresholds - BUY: {buy_thr:.6f}, SELL: {sell_thr:.6f}")

        if price_change > buy_thr:
            logger.signal(f"[{symbol_name}] LONG signal generated (change: {price_change:.4%} > {buy_thr:.4%})")
            return SIGNAL_LONG
        elif price_change < sell_thr:
            logger.signal(f"[{symbol_name}] SHORT signal generated (change: {price_change:.4%} < {sell_thr:.4%})")
            return SIGNAL_SHORT
        else:
            logger.signal(f"[{symbol_name}] NEUTRAL signal (change: {price_change:.4%} between {sell_thr:.4%} and {buy_thr:.4%})")
            return SIGNAL_NEUTRAL
            
    except Exception as e:
        logger.error(f"Error in transformer signal generation: {e}")
        return SIGNAL_NEUTRAL

def analyze_model_bias_and_adjust_thresholds(
    df_with_features: pd.DataFrame,
    model: TimeSeriesTransformer,
    scaler: MinMaxScaler,
    feature_cols: List[str],
    target_idx: int,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """Analyze model predictions for bias and suggest threshold adjustments.
    
    Args:
        df_with_features: DataFrame with technical indicators
        model: Trained transformer model
        scaler: Fitted MinMaxScaler instance
        feature_cols: List of feature column names
        target_idx: Target column index
        device: Device for inference
        
    Returns:
        Tuple of (suggested_buy_threshold, suggested_sell_threshold)
    """
    try:
        if model is None or scaler is None or not feature_cols:
            logger.error("Invalid model, scaler, or feature_cols provided")
            return BUY_THRESHOLD, SELL_THRESHOLD
            
        seq_length = model.pos_embedding.shape[1]
        if len(df_with_features) < seq_length + 100:
            logger.warning("Insufficient data for bias analysis")
            return BUY_THRESHOLD, SELL_THRESHOLD
        
        if target_idx >= len(feature_cols) or target_idx < 0:
            logger.error(f"Invalid target_idx: {target_idx}, feature_cols length: {len(feature_cols)}")
            return BUY_THRESHOLD, SELL_THRESHOLD
        
        test_samples = min(200, len(df_with_features) - seq_length)
        predictions, actual_returns = [], []
        
        model.eval()
        with torch.no_grad():
            for i in range(test_samples):
                start_idx = len(df_with_features) - test_samples - seq_length + i
                end_idx = start_idx + seq_length
                
                if end_idx >= len(df_with_features):
                    break
                    
                try:
                    sequence_data = df_with_features[feature_cols].iloc[start_idx:end_idx].values
                    if sequence_data.shape[0] != seq_length or sequence_data.shape[1] != len(feature_cols):
                        continue
                        
                    sequence_scaled = scaler.transform(sequence_data)
                    current_price = df_with_features[COL_CLOSE].iloc[end_idx - 1]
                    
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                        
                    if end_idx + 5 < len(df_with_features):
                        future_price = df_with_features[COL_CLOSE].iloc[end_idx + 4]
                        if pd.isna(future_price) or future_price <= 0:
                            continue
                        actual_return = (future_price - current_price) / current_price
                        actual_returns.append(actual_return)
                    else:
                        continue
                    
                    input_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                    prediction_scaled = model(input_tensor).cpu().numpy()[0, 0]
                    
                    if not np.isfinite(prediction_scaled):
                        continue
                    
                    dummy_pred = np.zeros((1, len(feature_cols)))
                    dummy_pred[:, target_idx] = prediction_scaled
                    predicted_price = scaler.inverse_transform(dummy_pred)[0, target_idx]
                    
                    if not np.isfinite(predicted_price) or predicted_price <= 0:
                        continue
                        
                    predicted_return = (predicted_price - current_price) / current_price
                    
                    if abs(predicted_return) < 10.0 and np.isfinite(predicted_return):
                        predictions.append(predicted_return)
                        
                except Exception:
                    continue
        
        if len(predictions) < 20:
            logger.warning(f"Insufficient valid predictions for bias analysis: {len(predictions)}")
            return BUY_THRESHOLD, SELL_THRESHOLD
        
        predictions = np.array(predictions)
        logger.analysis(f"Bias analysis: {len(predictions)} valid predictions generated")
        logger.analysis(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        long_signals = (predictions > BUY_THRESHOLD).sum()
        short_signals = (predictions < SELL_THRESHOLD).sum()
        
        long_pct = long_signals / len(predictions) * 100
        short_pct = short_signals / len(predictions) * 100
        
        suggested_buy_threshold, suggested_sell_threshold = BUY_THRESHOLD, SELL_THRESHOLD
        
        if short_pct < 15.0 or long_pct < 15.0:
            if short_pct < 15.0:
                sell_percentile = float(np.percentile(predictions, 25))
                suggested_sell_threshold = max(sell_percentile, SELL_THRESHOLD * 0.5)
                suggested_sell_threshold = max(suggested_sell_threshold, -0.05)
                
            if long_pct < 15.0:
                buy_percentile = float(np.percentile(predictions, 75))
                suggested_buy_threshold = min(buy_percentile, BUY_THRESHOLD * 0.5)
                suggested_buy_threshold = min(suggested_buy_threshold, 0.05)
        
        if not np.isfinite(suggested_buy_threshold) or not np.isfinite(suggested_sell_threshold):
            logger.warning("Generated invalid thresholds, using defaults")
            return float(BUY_THRESHOLD), float(SELL_THRESHOLD)
            
        return float(suggested_buy_threshold), float(suggested_sell_threshold)
        
    except Exception as e:
        logger.error(f"Error in bias analysis: {e}")
        return BUY_THRESHOLD, SELL_THRESHOLD

def evaluate_model(
    model: TimeSeriesTransformer, 
    test_loader: DataLoader, 
    scaler: MinMaxScaler,
    feature_cols: List[str], 
    target_col_idx: int, 
    device: str = 'cpu'
) -> Tuple[float, float]:
    """Evaluate model on test data and compute performance metrics.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        scaler: Scaler for inverse transforms
        feature_cols: Feature column names
        target_col_idx: Index of target column
        device: Device to use for evaluation
    
    Returns:
        Tuple of (mse, mae) metrics
    """
    model.eval()
    real_prices, predicted_prices = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            predictions = model(x_batch).cpu().numpy()
            y_batch = y_batch.cpu().numpy()

            for i in range(len(predictions)):
                dummy_pred = np.zeros((1, len(feature_cols)))
                dummy_pred[:, target_col_idx] = predictions[i]
                dummy_real = np.zeros((1, len(feature_cols)))
                dummy_real[:, target_col_idx] = y_batch[i]

                pred_inversed = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
                real_inversed = scaler.inverse_transform(dummy_real)[:, target_col_idx]

                predicted_prices.extend(pred_inversed)
                real_prices.extend(real_inversed)

    real_prices, predicted_prices = np.array(real_prices).flatten(), np.array(predicted_prices).flatten()
    mse = float(np.mean((real_prices - predicted_prices) ** 2))
    mae = float(np.mean(np.abs(real_prices - predicted_prices)))

    logger.performance(f"Model Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse, mae

def load_transformer_model(
    model_path: Optional[str] = None
) -> Tuple[Optional[TimeSeriesTransformer], Optional[MinMaxScaler], Optional[List[str]], Optional[int]]:
    """Load a trained transformer model with comprehensive error handling.
    
    Args:
        model_path: Path to the model file (defaults to TRANSFORMER_MODEL_FILENAME)
        
    Returns:
        Tuple of (model, scaler, feature_cols, target_idx) or (None, None, None, None) if loading fails
    """
    if model_path is None:
        model_path = str(MODELS_DIR / TRANSFORMER_MODEL_FILENAME)
    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        return None, None, None, None
    try:
        checkpoint = safe_load_model(model_path)
        if checkpoint is None:
            logger.error("Failed to load checkpoint")
            return None, None, None, None
        
        if 'data_info' in checkpoint:
            data_info = checkpoint['data_info']
            scaler = data_info.get('scaler')
            feature_cols = data_info.get('feature_cols')
            target_idx = data_info.get('target_idx')
        else:
            scaler = checkpoint.get('scaler')
            feature_cols = checkpoint.get('feature_cols')
            target_idx = checkpoint.get('target_idx')
        
        required_keys = ['model_state_dict', 'model_config']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            logger.error(f"Missing keys in checkpoint: {missing_keys}")
            return None, None, None, None
        
        if scaler is None or feature_cols is None or target_idx is None:
            logger.error("Missing scaler, feature_cols, or target_idx in checkpoint")
            return None, None, None, None
        
        config = checkpoint['model_config']
        model = TimeSeriesTransformer(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.success(f"Model loaded successfully from {model_path}")
        return model, scaler, feature_cols, target_idx
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None, None, None, None

