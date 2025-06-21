import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities._logger import setup_logging
from utilities._gpu_resource_manager import get_gpu_resource_manager, check_gpu_availability, get_tensor_core_info
logger = setup_logging(module_name="signals_transformer", log_level=logging.DEBUG)

from components._generate_indicator_features import generate_indicator_features
from components.config import (
    BUY_THRESHOLD,
    COL_CLOSE, CPU_MODEL_CONFIG,
    DEFAULT_EPOCHS,
    FUTURE_RETURN_SHIFT,
    GPU_MODEL_CONFIG,
    MIN_DATA_POINTS, MODEL_FEATURES, MODELS_DIR,
    SELL_THRESHOLD, SIGNAL_LONG, SIGNAL_NEUTRAL, SIGNAL_SHORT,
    TRANSFORMER_MODEL_FILENAME
)

# Check GPU availability
gpu_available: bool = check_gpu_availability()

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
    """Dataset for time series prediction with transformer models.
    
    Implements sliding window approach for sequence generation from time series data.
    Supports both single-step and multi-step prediction targets.

    Attributes:
        data: Input time series data
        seq_length: Length of input sequences
        pred_length: Number of future steps to predict
        feature_dim: Number of features in data
        target_column_idx: Index of target feature column
    """
    
    def __init__(
        self, 
        data: np.ndarray,
        seq_length: int = 60,
        prediction_length: int = 1,
        feature_dim: int = 4,
        target_column_idx: int = 3
    ) -> None:
        self.data = data
        self.seq_length = seq_length
        self.pred_length = prediction_length
        self.feature_dim = feature_dim
        self.target_column_idx = target_column_idx

    def __len__(self) -> int:
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length, self.target_column_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting with self-attention mechanism.
    
    Implements a transformer-based architecture optimized for time series data,
    with positional encodings and configurable hyperparameters.

    Attributes:
        input_fc: Linear layer for input feature projection
        pos_embedding: Learnable positional embeddings
        transformer_encoder: Main transformer encoder stack
        fc_out: Output projection layer
    """
    
    def __init__(
        self,
        feature_size: int = GPU_MODEL_CONFIG['feature_size'],
        num_layers: int = GPU_MODEL_CONFIG['num_layers'],
        d_model: int = GPU_MODEL_CONFIG['d_model'],
        nhead: int = GPU_MODEL_CONFIG['nhead'],
        dim_feedforward: int = GPU_MODEL_CONFIG['dim_feedforward'],
        dropout: float = GPU_MODEL_CONFIG['dropout'],
        seq_length: int = GPU_MODEL_CONFIG['seq_length'],
        prediction_length: int = GPU_MODEL_CONFIG['prediction_length']
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

        # Note: Tensor Core optimization is handled externally in train_and_save_transformer_model
        # to avoid double optimization and maintain consistency across training pipeline
        
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
        
        Applies linear projection, positional encoding, and self-attention
        to generate predictions from input sequences.

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

    Implements optimized training with GPU acceleration, mixed precision, gradient accumulation,
    early stopping with multiple metrics, and comprehensive logging.

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
        Tuple containing:
            - Trained model
            - Training history with metrics
    """
    gpu_manager = get_gpu_resource_manager()
    
    # Use GPU resource manager for better resource management
    with gpu_manager.gpu_scope(device_id=0 if device == 'cuda' else None) as gpu_device:
        # Determine actual device to use
        actual_device = gpu_device if gpu_device is not None else torch.device('cpu')
        use_gpu = actual_device.type == 'cuda'
        
        # GPU optimization setup with Tensor Core detection
        use_amp = use_mixed_precision and use_gpu
        gpu_manager = get_gpu_resource_manager()
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
        
        if use_gpu:
            # Get GPU memory info using resource manager
            memory_info = gpu_manager.get_memory_info()
            logger.gpu(f"Training on GPU: {torch.cuda.get_device_name()}")
            logger.memory(f"GPU Memory - Total: {memory_info['total'] / 1024**3:.1f} GB, "
                         f"Allocated: {memory_info['allocated'] / 1024**3:.2f} GB, "
                         f"Cached: {memory_info['cached'] / 1024**3:.2f} GB")
            
            # GPU memory optimization with Tensor Core support
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable optimizations for training
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Tensor Core specific optimizations
            if has_tensor_cores:
                # Enable Tensor Core usage for matrix operations
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                
                # Optimize for Tensor Core dimensions (multiples of 8 for FP16)
                logger.gpu(f"Tensor Core optimizations enabled for {tensor_info['generation']} generation")
                
                # Set optimal tensor shapes for Tensor Cores
                try:
                    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                        torch.backends.cuda.enable_flash_sdp(True)
                        logger.gpu("Flash Attention enabled for Tensor Cores")
                except Exception as e:
                    logger.debug(f"Flash Attention not available: {e}")
            
            # Pre-allocate GPU memory for better performance
            if memory_info['total'] > 8 * 1024**3:  # 8GB+
                torch.cuda.empty_cache()
                # Pre-warm GPU with Tensor Core optimized dimensions
                warmup_tensor_size = 64 if has_tensor_cores else 60
                dummy_tensor = torch.randn(1, warmup_tensor_size, 10, device=actual_device)
                del dummy_tensor
                torch.cuda.empty_cache()
        else:
            logger.config("Training on CPU")
        
        # Loss function and optimizer setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True, min_lr=1e-7
        )
        
        # Mixed precision scaler
        scaler = GradScaler() if use_amp else None
        
        model.to(actual_device)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.model(f"Total parameters: {total_params:,}")
        logger.model(f"Trainable parameters: {trainable_params:,}")
        logger.config(f"Device: {actual_device}, Mixed Precision: {use_amp}, Gradient Accumulation: {gradient_accumulation_steps}")
        
        # Enhanced early stopping with multiple metrics
        best_val_loss = float('inf')
        best_val_mae = float('inf')
        best_combined_score = float('-inf')
        patience_counter = 0
        best_model_state = None
        
        # Learning rate scheduling and warmup
        warmup_epochs = max(5, epochs // 10)
        initial_lr = lr
        
        # Apply learning rate warmup schedule
        def get_lr_for_epoch(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                # Linear warmup
                return initial_lr * ((current_epoch + 1) / warmup_epochs)
            return initial_lr  # After warmup, let scheduler take over
        
        # Training history
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'combined_scores': [],
            'learning_rates': []
        }
        
        logger.model(f"Starting training for {epochs} epochs (warmup: {warmup_epochs} epochs)...")
        
        for epoch in range(epochs):
            # Memory cleanup at epoch start
            if use_gpu:
                torch.cuda.empty_cache()
                
            # Apply learning rate warmup if in warmup phase
            if epoch < warmup_epochs:
                new_lr = get_lr_for_epoch(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if epoch == 0:
                    logger.config(f"Learning rate warmup: {new_lr:.2e} -> {initial_lr:.2e} over {warmup_epochs} epochs")
                    
            # Training phase
            model.train()
            train_losses = []
            optimizer.zero_grad()
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(actual_device, non_blocking=True)
                y_batch = y_batch.to(actual_device, non_blocking=True)

                # Forward pass with optional mixed precision
                if use_amp and scaler is not None:
                    with autocast(device_type='cuda'):
                        output = model(x_batch)
                        loss = criterion(output, y_batch)
                        loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation - handle last batch correctly
                    accumulate_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                    is_last_batch = (batch_idx + 1) == len(train_loader)
                    
                    if accumulate_step or is_last_batch:
                        # Gradient clipping before optimizer step
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                    train_losses.append(loss.item() * gradient_accumulation_steps)  # Restore original loss scale
                else:
                    output = model(x_batch)
                    loss = criterion(output, y_batch)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    # Gradient accumulation with consistent variables
                    accumulate_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                    is_last_batch = (batch_idx + 1) == len(train_loader)
                    
                    if accumulate_step or is_last_batch:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    train_losses.append(loss.item() * gradient_accumulation_steps)  # Restore original loss scale
                
                # GPU memory monitoring using resource manager
                if use_gpu and batch_idx % 20 == 0:
                    memory_info = gpu_manager.get_memory_info()
                    if batch_idx % 100 == 0:
                        logger.memory(f"Epoch {epoch+1}, Batch {batch_idx}: GPU Memory - "
                                    f"Allocated: {memory_info['allocated'] / 1024**3:.2f}GB, "
                                    f"Cached: {memory_info['cached'] / 1024**3:.2f}GB")

            mean_train_loss = np.mean(train_losses)
            training_history['train_loss'].append(mean_train_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # Validation phase with enhanced metrics
            if val_loader is not None:
                model.eval()
                val_losses = []
                val_maes = []
                
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(actual_device, non_blocking=True)
                        y_val = y_val.to(actual_device, non_blocking=True)
                        
                        if use_amp:
                            with autocast(device_type='cuda'):
                                output_val = model(x_val)
                                loss_val = criterion(output_val, y_val)
                        else:
                            output_val = model(x_val)
                            loss_val = criterion(output_val, y_val)
                        
                        val_losses.append(loss_val.item())
                        
                        # Calculate MAE for additional metric
                        mae_val = torch.mean(torch.abs(output_val - y_val)).item()
                        val_maes.append(mae_val)
                
                mean_val_loss = np.mean(val_losses)
                mean_val_mae = np.mean(val_maes)
                
                training_history['val_loss'].append(mean_val_loss)
                training_history['val_mae'].append(mean_val_mae)
                
                # Combined score for better early stopping
                normalized_loss = 1.0 / (1.0 + mean_val_loss)
                normalized_mae = 1.0 / (1.0 + mean_val_mae)
                combined_score = 0.6 * normalized_loss + 0.4 * normalized_mae
                training_history['combined_scores'].append(combined_score)
                
                # Learning rate scheduling (only after warmup)
                if epoch >= warmup_epochs:
                    scheduler.step(mean_val_loss)
                
                # Enhanced early stopping with multiple criteria
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
                
                # Progress logging with memory info
                memory_info_str = ""
                if use_gpu:
                    memory_info = gpu_manager.get_memory_info()
                    memory_info_str = f", GPU Mem: {memory_info['allocated'] / 1024**3:.2f}GB"
                
                current_lr = optimizer.param_groups[0]['lr']
                early_stop_info = f", Patience: {patience_counter}/{patience}" if use_early_stopping else ""
                
                logger.performance(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {mean_train_loss:.6f}, "
                                 f"Val Loss: {mean_val_loss:.6f}, Val MAE: {mean_val_mae:.6f}, "
                                 f"Combined: {combined_score:.6f}, LR: {current_lr:.2e}{early_stop_info}{memory_info_str}")
            else:
                # No validation data
                memory_info_str = ""
                if use_gpu:
                    memory_info = gpu_manager.get_memory_info()
                    memory_info_str = f", GPU Mem: {memory_info['allocated'] / 1024**3:.2f}GB"
                
                current_lr = optimizer.param_groups[0]['lr']
                logger.performance(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {mean_train_loss:.6f}, "
                                 f"LR: {current_lr:.2e}{memory_info_str}")
        
        # Training summary
        if training_history['val_loss']:
            final_val_loss = training_history['val_loss'][-1]
            best_val_loss_epoch = np.argmin(training_history['val_loss']) + 1
            best_combined_epoch = np.argmax(training_history['combined_scores']) + 1
            logger.success(f"📊 Training completed - Final Val Loss: {final_val_loss:.6f}, "
                          f"Best Val Loss: {min(training_history['val_loss']):.6f} (Epoch {best_val_loss_epoch})")
            logger.success(f"📊 Best Combined Score: {max(training_history['combined_scores']):.6f} (Epoch {best_combined_epoch})")
            
            # Performance improvement metrics
            if len(training_history['val_loss']) > 1:
                loss_improvement = (training_history['val_loss'][0] - min(training_history['val_loss'])) / training_history['val_loss'][0] * 100
                logger.analysis(f"📈 Loss improvement: {loss_improvement:.2f}%")
        else:
            final_train_loss = training_history['train_loss'][-1]
            logger.success(f"📊 Training completed - Final Train Loss: {final_train_loss:.6f}")

        # GPU resource manager will automatically cleanup when exiting context
        return model, training_history

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
        df_market_data: Market data with OHLCV
        model: Trained transformer model
        scaler: Fitted scaler for features
        feature_cols: Feature column names
        target_col_idx: Index of target column
        device: Device for inference
        suggested_thresholds: Custom (buy, sell) thresholds
    
    Returns:
        Trading signal (LONG, SHORT, or NEUTRAL)
    """
    try:
        if df_market_data.empty:
            logger.warning("Input DataFrame for signal generation is empty")
            return SIGNAL_NEUTRAL
        
        df_with_features = generate_indicator_features(df_market_data.copy())
        if df_with_features.empty:
            logger.warning("DataFrame became empty after feature calculation")
            return SIGNAL_NEUTRAL

        seq_length = model.pos_embedding.shape[1]
        if len(df_with_features) < seq_length:
            logger.warning(f"Insufficient data for prediction: {len(df_with_features)} < {seq_length}")
            return SIGNAL_NEUTRAL

        available_features = [col for col in feature_cols if col in df_with_features.columns]
        if len(available_features) != len(feature_cols):
            logger.warning(f"Missing features: {set(feature_cols) - set(available_features)}")
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
        
        # Get symbol name safely
        symbol_name = 'UNKNOWN'
        if hasattr(df_market_data, 'symbol'):
            symbol_name = df_market_data.symbol
        elif 'symbol' in df_market_data.columns:
            symbol_name = df_market_data['symbol'].iloc[0]
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
                    
                    input_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0).to(device)
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
                sell_percentile = np.percentile(predictions, 25)
                suggested_sell_threshold = max(sell_percentile, SELL_THRESHOLD * 0.5)
                suggested_sell_threshold = max(suggested_sell_threshold, -0.05)
                
            if long_pct < 15.0:
                buy_percentile = np.percentile(predictions, 75)
                suggested_buy_threshold = min(buy_percentile, BUY_THRESHOLD * 0.5)
                suggested_buy_threshold = min(suggested_buy_threshold, 0.05)
        
        if not np.isfinite(suggested_buy_threshold) or not np.isfinite(suggested_sell_threshold):
            logger.warning("Generated invalid thresholds, using defaults")
            return float(BUY_THRESHOLD), float(SELL_THRESHOLD)
            
        return float(suggested_buy_threshold), float(suggested_sell_threshold)
        
    except Exception as e:
        logger.error(f"Error in bias analysis: {e}")
        return BUY_THRESHOLD, SELL_THRESHOLD

def train_and_save_transformer_model(
    df_input: pd.DataFrame, 
    model_filename: Optional[str] = None
) -> Tuple[Optional[TimeSeriesTransformer], str]:
    """Train and save a transformer model with bias detection and optimization.
    
    Args:
        df_input: Input DataFrame with market data
        model_filename: Custom filename for model (auto-generated if None)
        
    Returns:
        Tuple of (trained_model, model_path) or (None, "") if failed
    """
    try:
        if df_input.empty:
            logger.error("Input DataFrame is empty")
            return None, ""
        
        logger.model(f"Training transformer with {len(df_input)} rows")
        
        if len(df_input) < MIN_DATA_POINTS:
            logger.error(f"Insufficient data for training: {len(df_input)} < {MIN_DATA_POINTS}")
            return None, ""
        
        # Check for Tensor Core support for model optimization
        gpu_manager = get_gpu_resource_manager()
        tensor_info = gpu_manager.get_tensor_core_info() if gpu_available else {'has_tensor_cores': False, 'generation': 'N/A'}
        has_tensor_cores = tensor_info['has_tensor_cores']
        
        if has_tensor_cores:
            logger.gpu(f"Detected {tensor_info['generation']} Tensor Cores - Optimization enabled")
        
        df_with_features = generate_indicator_features(df_input)
        if df_with_features.empty:
            logger.error("No data after feature calculation")
            return None, ""
        
        data_scaled, scaler, feature_cols = select_and_scale_features(df_with_features)
        target_idx = feature_cols.index(COL_CLOSE)
        
        seq_len, pred_len = GPU_MODEL_CONFIG['seq_length'], GPU_MODEL_CONFIG['prediction_length']
        dataset = CryptoDataset(data_scaled, seq_len, pred_len, len(feature_cols), target_idx)
        
        if len(dataset) < 100:
            logger.error(f"Insufficient samples for training: {len(dataset)}")
            return None, ""
        
        n = len(dataset)
        n_train, n_val = int(n * 0.8), int(n * 0.1)
        
        train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
        val_ds = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
        
        batch_size = 32
        if gpu_available:
            gpu_manager = get_gpu_resource_manager()
            memory_info = gpu_manager.get_memory_info()
            total_memory_gb = memory_info['total'] / 1024**3
            
            # Optimize batch size for both memory and Tensor Cores
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
            
            # Adjust for Tensor Core optimization (prefer multiples of 8)
            if has_tensor_cores:
                batch_size = ((base_batch_size + 7) // 8) * 8  # Round to nearest multiple of 8
                logger.gpu(f"Using Tensor Core optimized batch size: {batch_size} (GPU Memory: {total_memory_gb:.1f}GB)")
            else:
                batch_size = base_batch_size
                logger.gpu(f"Using GPU-optimized batch size: {batch_size} (GPU Memory: {total_memory_gb:.1f}GB)")
        
        pin_memory = gpu_available
        num_workers = 0 if gpu_available else min(4, os.cpu_count() // 2)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                 pin_memory=pin_memory, num_workers=num_workers, 
                                 persistent_workers=True if num_workers > 0 else False)
        
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                               pin_memory=pin_memory, num_workers=num_workers,
                               persistent_workers=True if num_workers > 0 else False)
        
        device = 'cuda' if gpu_available else 'cpu'
        logger.config(f"Training on device: {device}")
        
        # Fix feature size mismatch and optimize for Tensor Cores
        model_config = GPU_MODEL_CONFIG.copy() if gpu_available else CPU_MODEL_CONFIG.copy()
        model_config['feature_size'] = len(feature_cols)
        
        # Optimize model dimensions for Tensor Cores if available
        if gpu_available and has_tensor_cores:
            # Ensure d_model and feedforward dimensions are multiples of 8 for optimal Tensor Core usage
            original_d_model = model_config['d_model']
            original_feedforward = model_config['dim_feedforward']
            
            # Round up to nearest multiple of 8
            model_config['d_model'] = ((original_d_model + 7) // 8) * 8
            model_config['dim_feedforward'] = ((original_feedforward + 7) // 8) * 8
            
            if model_config['d_model'] != original_d_model or model_config['dim_feedforward'] != original_feedforward:
                logger.gpu(f"Optimized model dimensions for Tensor Cores: "
                          f"d_model {original_d_model} -> {model_config['d_model']}, "
                          f"feedforward {original_feedforward} -> {model_config['dim_feedforward']}")
        
        model = TimeSeriesTransformer(**model_config)
        
        trained_model, training_history = train_transformer_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device, 
            epochs=DEFAULT_EPOCHS,
            use_early_stopping=True,
            patience=15,
            use_mixed_precision=gpu_available,
            gradient_accumulation_steps=2 if gpu_available else 1,
            lr=3e-4 if gpu_available else 1e-3
        )
        
        logger.analysis("="*60)
        logger.analysis("TRAINING DATA ANALYSIS FOR BIAS DETECTION")
        logger.analysis("="*60)
        
        sample_size = min(1000, len(df_with_features) - FUTURE_RETURN_SHIFT)
        if sample_size > 0:
            sample_data = df_with_features.tail(sample_size)
            future_returns = sample_data[COL_CLOSE].shift(FUTURE_RETURN_SHIFT) / sample_data[COL_CLOSE] - 1
            future_returns = future_returns.dropna()
            
            if len(future_returns) > 0:
                positive_returns = (future_returns > BUY_THRESHOLD).sum()
                negative_returns = (future_returns < SELL_THRESHOLD).sum()
                neutral_returns = len(future_returns) - positive_returns - negative_returns
                
                logger.analysis(f"Historical price movement distribution:")
                logger.analysis(f"  • Positive moves (>{BUY_THRESHOLD:.4f}): {positive_returns} ({positive_returns/len(future_returns):.1%})")
                logger.analysis(f"  • Negative moves (<{SELL_THRESHOLD:.4f}): {negative_returns} ({negative_returns/len(future_returns):.1%})")
                logger.analysis(f"  • Neutral moves: {neutral_returns} ({neutral_returns/len(future_returns):.1%})")
                logger.analysis(f"  • Mean return: {future_returns.mean():.6f}")
                logger.analysis(f"  • Std return: {future_returns.std():.6f}")
                
                if negative_returns / len(future_returns) < 0.2:
                    logger.warning("⚠️  TRAINING DATA BIAS DETECTED:")
                    logger.warning("   Historical data shows fewer downward movements")
                    logger.warning("   This may cause model to be biased toward bullish predictions")
        
        suggested_buy_threshold, suggested_sell_threshold = analyze_model_bias_and_adjust_thresholds(
            df_with_features, trained_model, scaler, feature_cols, target_idx, device
        )
        
        logger.analysis("="*60)
        
        if model_filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_filename = f"{TRANSFORMER_MODEL_FILENAME.split('.')[0]}_{timestamp}.pth"
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / model_filename
        
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': {
                'feature_size': len(feature_cols),
                'seq_length': model_config['seq_length'],
                'prediction_length': model_config['prediction_length'],
                'num_layers': model_config['num_layers'],
                'd_model': model_config['d_model'],
                'nhead': model_config['nhead'],
                'dim_feedforward': model_config['dim_feedforward'],
                'dropout': model_config['dropout']
            },
            'scaler': scaler,
            'feature_cols': feature_cols,
            'target_idx': target_idx,
            'training_stats': {
                'total_samples': len(df_with_features),
                'buy_threshold': BUY_THRESHOLD,
                'sell_threshold': SELL_THRESHOLD,
                'suggested_buy_threshold': suggested_buy_threshold,
                'suggested_sell_threshold': suggested_sell_threshold
            }
        }, model_path)
        
        logger.success(f"Model saved to {model_path}")
        return trained_model, str(model_path)
        
    except Exception as e:
        logger.error(f"Error training transformer model: {e}")
        return None, ""

def load_transformer_model(
    model_path: Optional[str] = None
) -> Tuple[Optional[TimeSeriesTransformer], Optional[MinMaxScaler], Optional[List[str]], Optional[int]]:
    """Load a trained transformer model with comprehensive error handling.
    
    Args:
        model_path: Path to model file (uses default if None)
        
    Returns:
        Tuple containing:
            - Loaded TimeSeriesTransformer model or None if failed
            - MinMaxScaler instance or None
            - List of feature column names or None
            - Target column index or None
    """
    if model_path is None:
        model_path = str(MODELS_DIR / TRANSFORMER_MODEL_FILENAME)

    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        return None, None, None, None

    try:
        checkpoint = None
        
        # PyTorch 2.6+ safe loading strategy with multiple fallbacks
        try:
            # Attempt 1: Try with weights_only=False (default safe mode)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except (TypeError, AttributeError):
            try:
                # Attempt 2: Try older PyTorch loading method
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None, None, None, None
        
        if checkpoint is None:
            logger.error("Failed to load checkpoint")
            return None, None, None, None

        required_keys = ['model_state_dict', 'model_config', 'scaler', 'feature_cols', 'target_idx']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            logger.error(f"Missing keys in checkpoint: {missing_keys}")
            return None, None, None, None

        config = checkpoint['model_config']
        model = TimeSeriesTransformer(**config)
        model.load_state_dict(checkpoint['model_state_dict'])

        scaler = checkpoint['scaler']
        feature_cols = checkpoint['feature_cols']
        target_idx = checkpoint['target_idx']

        logger.success(f"Model loaded successfully from {model_path}")
        return model, scaler, feature_cols, target_idx
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None, None, None, None