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
from typing import List, Optional, Tuple

# Add the parent directory to sys.path to allow importing from config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities._logger import setup_logging
from utilities._gpu_resource_manager import get_gpu_resource_manager
logger = setup_logging(module_name="signals_transformer", log_level=logging.DEBUG)

from signals._components._gpu_check_availability import check_gpu_availability
from signals._components._generate_indicator_features import generate_indicator_features

from livetrade.config import (
    BUY_THRESHOLD,
    COL_CLOSE, CPU_MODEL_CONFIG,
    DEFAULT_EPOCHS,
    FUTURE_RETURN_SHIFT,
    GPU_MODEL_CONFIG,
    MIN_DATA_POINTS, MODEL_FEATURES, MODELS_DIR,
    SELL_THRESHOLD, SIGNAL_LONG, SIGNAL_NEUTRAL, SIGNAL_SHORT,
    TRANSFORMER_MODEL_FILENAME
)

# Check GPU availability at module level
gpu_available = check_gpu_availability()

def select_and_scale_features(df: pd.DataFrame, 
                              feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, MinMaxScaler, List[str]]:
    """
    Selects features and applies MinMax scaling to prepare data for model.
    
    Args:
        df (pd.DataFrame): Input DataFrame with technical indicators
        feature_cols (Optional[List[str]]): Optional list of features to use (defaults to MODEL_FEATURES + ma_20_slope)
    
    Returns:
        Tuple[np.ndarray, MinMaxScaler, List[str]]: (scaled_data, scaler, feature_columns_used)
    """
    if feature_cols is None:
        feature_cols = MODEL_FEATURES.copy() + ['ma_20_slope']

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.data(f"Missing columns in DataFrame: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
        
    if not feature_cols:
        raise ValueError("No valid feature columns found in DataFrame")

    logger.data(f"Using features: {feature_cols}")
    
    try:
        data = df[feature_cols].values
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        logger.data(f"Data scaled successfully, shape: {data_scaled.shape}")
        return data_scaled, scaler, feature_cols
    except Exception as e:
        logger.error(f"Error in select_and_scale_features: {e}")
        raise

class CryptoDataset(Dataset):
    def __init__(self, data, seq_length=60, prediction_length=1, feature_dim=4, target_column_idx=3):
        """
        Dataset for time series prediction with transformer models.
        
        Args:
            data (np.ndarray): Scaled feature data
            seq_length (int): Length of input sequences
            prediction_length (int): Number of future steps to predict
            feature_dim (int): Number of features in data
            target_column_idx (int): Index of target feature column
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_length = prediction_length
        self.feature_dim = feature_dim
        self.target_column_idx = target_column_idx

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length, self.target_column_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size=GPU_MODEL_CONFIG['feature_size'],
        num_layers=GPU_MODEL_CONFIG['num_layers'],
        d_model=GPU_MODEL_CONFIG['d_model'],
        nhead=GPU_MODEL_CONFIG['nhead'],
        dim_feedforward=GPU_MODEL_CONFIG['dim_feedforward'],
        dropout=GPU_MODEL_CONFIG['dropout'],
        seq_length=GPU_MODEL_CONFIG['seq_length'],
        prediction_length=GPU_MODEL_CONFIG['prediction_length']
    ):
        """
        Transformer model for time series forecasting.
        
        Args:
            feature_size (int): Number of input features
            num_layers (int): Number of transformer encoder layers
            d_model (int): Model dimension size
            nhead (int): Number of attention heads
            dim_feedforward (int): Feed-forward layer dimension
            dropout (float): Dropout rate
            seq_length (int): Input sequence length
            prediction_length (int): Number of steps to predict
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

    def forward(self, src):
        """
        Forward pass through the model.
        
        Args:
            src (torch.Tensor): Input tensor [batch_size, seq_length, feature_size]
            
        Returns:
            torch.Tensor: Prediction [batch_size, prediction_length]
        """
        _, seq_len, _ = src.shape
        src = self.input_fc(src)
        src = src + self.pos_embedding[:, :seq_len, :]
        encoded = self.transformer_encoder(src)
        last_step = encoded[:, -1, :]
        out = self.fc_out(last_step)
        return out

def train_transformer_model(model: TimeSeriesTransformer, 
                            train_loader: DataLoader, 
                            val_loader: Optional[DataLoader] = None, 
                            lr: float = 1e-3, 
                            epochs: int = DEFAULT_EPOCHS, 
                            device: str = 'cpu',
                            use_early_stopping: bool = True, 
                            patience: int = 10,
                            use_mixed_precision: bool = True, 
                            gradient_accumulation_steps: int = 1) -> TimeSeriesTransformer:
    """
    Trains the transformer model with advanced optimizations including early stopping, 
    mixed precision training, and GPU optimizations.
    
    Args:
        model (TimeSeriesTransformer): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader, optional): Validation data loader
        lr (float): Learning rate
        epochs (int): Number of training epochs
        device (str): Device to train on ('cpu' or 'cuda')
        use_early_stopping (bool): Enable early stopping
        patience (int): Early stopping patience
        use_mixed_precision (bool): Use automatic mixed precision (AMP)
        gradient_accumulation_steps (int): Steps to accumulate gradients
        
    Returns:
        TimeSeriesTransformer: Trained model
    """
    gpu_manager = get_gpu_resource_manager()
    
    # Use GPU resource manager for better resource management
    with gpu_manager.gpu_scope(device_id=0 if device == 'cuda' else None) as gpu_device:
        # Determine actual device to use
        actual_device = gpu_device if gpu_device is not None else torch.device('cpu')
        use_gpu = actual_device.type == 'cuda'
        
        # GPU optimization setup
        use_amp = use_mixed_precision and use_gpu
        if use_amp:
            # Check if GPU supports mixed precision (Compute Capability >= 7.0)
            if torch.cuda.get_device_capability(0)[0] >= 7:
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
            
            # GPU memory optimization
            torch.backends.cudnn.benchmark = True  # Optimize cuDNN for consistent input sizes
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
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        logger.model(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
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
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping before optimizer step
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
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

            # Validation phase
            if val_loader is not None:
                model.eval()
                val_losses = []
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
                
                mean_val_loss = np.mean(val_losses)
                training_history['val_loss'].append(mean_val_loss)
                
                # Learning rate scheduling
                scheduler.step(mean_val_loss)
                
                # Early stopping logic
                if use_early_stopping:
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                        logger.performance(f"✅ New best validation loss: {best_val_loss:.6f}")
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.model(f"🛑 Early stopping triggered after {epoch+1} epochs (patience: {patience})")
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
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
                                 f"Val Loss: {mean_val_loss:.6f}, LR: {current_lr:.2e}{early_stop_info}{memory_info_str}")
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
            logger.success(f"📊 Training completed - Final Val Loss: {final_val_loss:.6f}, "
                          f"Best Val Loss: {min(training_history['val_loss']):.6f} (Epoch {best_val_loss_epoch})")
        else:
            final_train_loss = training_history['train_loss'][-1]
            logger.success(f"📊 Training completed - Final Train Loss: {final_train_loss:.6f}")

        # GPU resource manager will automatically cleanup when exiting context
        return model

def evaluate_model(model: TimeSeriesTransformer, test_loader: DataLoader, scaler: MinMaxScaler, 
                   feature_cols: List[str], target_col_idx: int, device: str = 'cpu') -> Tuple[float, float]:
    """
    Evaluates model on test data and computes performance metrics.
    
    Args:
        model (TimeSeriesTransformer): Model to evaluate
        test_loader (DataLoader): Test data loader
        scaler (MinMaxScaler): Scaler for inverse transforms
        feature_cols (list): Feature column names
        target_col_idx (int): Index of target column
        device (str): Device to use for evaluation
    
    Returns:
        Tuple[float, float]: (mse, mae) metrics
    """
    model.eval()
    real_prices = []
    predicted_prices = []

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

    real_prices = np.array(real_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()

    mse = float(np.mean((real_prices - predicted_prices) ** 2))
    mae = float(np.mean(np.abs(real_prices - predicted_prices)))

    logger.performance(f"Model Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse, mae

def get_latest_transformer_signal(df_market_data: pd.DataFrame, model: TimeSeriesTransformer, 
                                  scaler: MinMaxScaler, feature_cols: List[str], 
                                  target_col_idx: int, device: str = 'cpu', 
                                  suggested_thresholds: Optional[Tuple] = None) -> str:
    """
    Generates trading signal using transformer model predictions.
    
    Args:
        df_market_data (pd.DataFrame): Market data with OHLC
        model (TimeSeriesTransformer): Trained model
        scaler (MinMaxScaler): Fitted scaler for features
        feature_cols (list): Feature column names
        target_col_idx (int): Index of target column
        device (str): Device for inference
        suggested_thresholds (tuple, optional): Custom (buy, sell) thresholds
    
    Returns:
        str: Trading signal (LONG, SHORT, or NEUTRAL)
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
        
        symbol_name = df_market_data.get('symbol', 'UNKNOWN') if hasattr(df_market_data, 'symbol') else 'UNKNOWN'
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

def analyze_model_bias_and_adjust_thresholds(df_with_features: pd.DataFrame, 
                                            model: TimeSeriesTransformer, 
                                            scaler: MinMaxScaler, 
                                            feature_cols: List[str], 
                                            target_idx: int, 
                                            device: str = 'cpu') -> Tuple[float, float]:
    """
    Analyzes model predictions for bias and suggests threshold adjustments.
    
    Args:
        df_with_features (pd.DataFrame): DataFrame with technical indicators
        model (TimeSeriesTransformer): Trained model
        scaler (MinMaxScaler): Fitted scaler
        feature_cols (List[str]): Feature column names
        target_idx (int): Target column index
        device (str): Device for inference
        
    Returns:
        Tuple[float, float]: (suggested_buy_threshold, suggested_sell_threshold)
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
        predictions = []
        actual_returns = []
        
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
                        logger.debug(f"Invalid sequence shape: {sequence_data.shape}")
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
                        logger.debug(f"Invalid prediction: {prediction_scaled}")
                        continue
                    
                    dummy_pred = np.zeros((1, len(feature_cols)))
                    dummy_pred[:, target_idx] = prediction_scaled
                    
                    predicted_price = scaler.inverse_transform(dummy_pred)[0, target_idx]
                    
                    if not np.isfinite(predicted_price) or predicted_price <= 0:
                        logger.debug(f"Invalid predicted price: {predicted_price}")
                        continue
                        
                    predicted_return = (predicted_price - current_price) / current_price
                    
                    if abs(predicted_return) < 10.0 and np.isfinite(predicted_return):
                        predictions.append(predicted_return)
                    else:
                        logger.debug(f"Skipping unrealistic prediction: {predicted_return}")
                        
                except Exception as e:
                    logger.debug(f"Error in prediction calculation for sample {i}: {e}")
                    continue
        
        if len(predictions) < 20:
            logger.warning(f"Insufficient valid predictions for bias analysis: {len(predictions)}")
            return BUY_THRESHOLD, SELL_THRESHOLD
        
        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns[:len(predictions)])
        
        logger.analysis(f"Bias analysis: {len(predictions)} valid predictions generated")
        logger.analysis(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        long_signals = (predictions > BUY_THRESHOLD).sum()
        short_signals = (predictions < SELL_THRESHOLD).sum()
        
        long_pct = long_signals / len(predictions) * 100
        short_pct = short_signals / len(predictions) * 100
        
        suggested_buy_threshold = BUY_THRESHOLD
        suggested_sell_threshold = SELL_THRESHOLD
        
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

def train_and_save_transformer_model(df_input: pd.DataFrame, model_filename: Optional[str] = None) -> Tuple[Optional[object], str]:
    """
    Trains and saves a transformer model with bias detection.
    
    Args:
        df_input (pd.DataFrame): Input DataFrame with market data
        model_filename (str, optional): Custom filename for model
        
    Returns:
        Tuple[object, str]: (trained_model, model_path) or (None, "") if failed
    """
    try:
        if df_input.empty:
            logger.error("Input DataFrame is empty")
            return None, ""
        
        logger.model(f"Training transformer with {len(df_input)} rows")
        
        if len(df_input) < MIN_DATA_POINTS:
            logger.error(f"Insufficient data for training: {len(df_input)} < {MIN_DATA_POINTS}")
            return None, ""
        
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
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
        val_ds = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
        
        batch_size = 32
        if gpu_available:
            # Increase batch size for GPU training
            batch_size = 64
            logger.gpu(f"Using GPU-optimized batch size: {batch_size}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                 pin_memory=gpu_available, num_workers=0 if gpu_available else 2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               pin_memory=gpu_available, num_workers=0 if gpu_available else 2)
        
        device = 'cuda' if gpu_available else 'cpu'
        logger.config(f"Training on device: {device}")
        
        # Fix feature size mismatch
        model_config = GPU_MODEL_CONFIG.copy() if gpu_available else CPU_MODEL_CONFIG.copy()
        model_config['feature_size'] = len(feature_cols)
        model = TimeSeriesTransformer(**model_config)
        
        # Training with enhanced parameters
        trained_model = train_transformer_model(
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

def load_transformer_model(model_path: Optional[str] = None) -> Tuple[Optional[TimeSeriesTransformer], Optional[MinMaxScaler], Optional[List[str]], Optional[int]]:
    """
    Loads a trained transformer model with error handling.
    
    Args:
        model_path (str, optional): Path to model file (uses default if None)
        
    Returns:
        Tuple[Optional[TimeSeriesTransformer], Optional[MinMaxScaler], Optional[List[str]], Optional[int]]: 
            (model, scaler, feature_cols, target_idx) or (None, None, None, None) if failed
    """
    if model_path is None:
        model_path = str(MODELS_DIR / TRANSFORMER_MODEL_FILENAME)

    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        return None, None, None, None

    try:
        checkpoint = None
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                logger.error(f"Failed to load model with both methods: {e}")
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