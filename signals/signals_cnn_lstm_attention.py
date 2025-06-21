import logging
import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Literal, Optional, Tuple, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities._logger import setup_logging
logger = setup_logging(module_name="signals_cnn_lstm_attention", log_level=logging.DEBUG)

# Environment setup for PyTorch
os.environ.update({
    'KMP_DUPLICATE_LIB_OK': 'True',
    'OMP_NUM_THREADS': '1',
    'CUDA_LAUNCH_BLOCKING': '1',
    'TORCH_USE_CUDA_DSA': '1'
})

# PyTorch imports with error handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler 
    logger.success(f"PyTorch {torch.__version__} loaded successfully")
    
    if torch.cuda.is_available():
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                torch.ones(1).cuda()
            
            cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else "Unknown"          # type: ignore
            logger.gpu(f"CUDA {cuda_version} available with {torch.cuda.device_count()} device(s)")
        except Exception as cuda_error:
            logger.warning(f"CUDA available but not functional: {cuda_error}")
            logger.warning("Falling back to CPU mode...")
            torch.cuda.is_available = lambda: False  
    else:
        logger.info("CUDA not available, using CPU mode")
        
except ImportError as e:
    logger.error(f"Failed to import PyTorch: {e}")
    logger.error("Please reinstall PyTorch: pip install torch torchvision torchaudio")
    sys.exit(1)

from livetrade.config import (
    COL_CLOSE, DEFAULT_EPOCHS, GPU_MODEL_CONFIG, MODEL_FEATURES, MODELS_DIR,
    NEUTRAL_ZONE_LSTM, TARGET_THRESHOLD_LSTM, TRAIN_TEST_SPLIT, VALIDATION_SPLIT,
    WINDOW_SIZE_LSTM, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL
)

from signals._components.LSTM__class__FocalLoss import FocalLoss
from signals._components.LSTM__class__GridSearchThresholdOptimizer import GridSearchThresholdOptimizer
from signals._components.LSTM__class__Models import CNNLSTMAttentionModel, LSTMModel, LSTMAttentionModel
from signals._components._generate_indicator_features import generate_indicator_features
from signals._components.LSTM__function__create_balanced_target import create_balanced_target
from signals._components.LSTM__function__get_optimal_batch_size import get_optimal_batch_size
from utilities._gpu_resource_manager import get_gpu_resource_manager

def preprocess_cnn_lstm_data(df_input: pd.DataFrame, look_back: int = WINDOW_SIZE_LSTM, output_mode: str = 'classification', scaler_type: str = 'minmax') -> Tuple[np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler], list[str]]:
    """
    Preprocess data for CNN-LSTM model with sliding window approach.
    
    Args:
        df_input: Input DataFrame containing price data
        look_back: Number of time steps to look back for sequence creation
        output_mode: 'classification' for signal prediction or 'regression' for return prediction
        scaler_type: Scaling method ('minmax' or 'standard')
        
    Returns:
        X_sequences: Feature sequences array
        y_targets: Target values array
        fitted_scaler: Fitted scaler for feature normalization
        feature_names: List of features used in model
    """
    logger.model(f"Starting CNN-LSTM preprocessing: {df_input.shape} rows, lookback={look_back}, mode={output_mode}")
    
    if df_input.empty or len(df_input) < look_back + 10:
        logger.error(f"Insufficient data: {len(df_input)} rows, need at least {look_back + 10}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    # Calculate technical features
    df = generate_indicator_features(df_input.copy())
    if df.empty:
        logger.error("Feature calculation returned empty DataFrame")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    # Create targets based on mode
    if output_mode == 'classification':
        df = create_balanced_target(df, threshold=TARGET_THRESHOLD_LSTM, neutral_zone=NEUTRAL_ZONE_LSTM)
        if 'Target' not in df.columns:
            logger.error("Classification target creation failed")
            return np.array([]), np.array([]), MinMaxScaler(), []
    else:
        df['Target'] = df['close'].pct_change().shift(-1)
    
    # Clean data and validate
    initial_len = len(df)
    df.dropna(inplace=True)
    if len(df) < look_back + 1:
        logger.error(f"Insufficient data after cleanup: {len(df)} rows (dropped {initial_len - len(df)} NaN)")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    # Prepare feature matrix
    available_features = [col for col in MODEL_FEATURES if col in df.columns]
    if not available_features:
        logger.error(f"No valid features found from {MODEL_FEATURES}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    features = df[available_features].values
    
    # Handle invalid values
    if np.isnan(features).any() or np.isinf(features).any():
        logger.warning("Cleaning invalid values in features")
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Scale features
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features)
    except Exception as e:
        logger.error(f"Feature scaling failed: {e}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    # Create sliding window sequences
    target_values = df['Target'].values
    X_sequences, y_targets = [], []
    
    for i in range(look_back, min(len(scaled_features), len(target_values))):
        sequence = scaled_features[i-look_back:i]
        
        if sequence.shape[0] == look_back and not (np.isnan(sequence).any() or np.isinf(sequence).any()):
            X_sequences.append(sequence)
            y_targets.append(target_values[i])
    
    if not X_sequences:
        logger.error("No valid sequences created after filtering")
        return np.array([]), np.array([]), scaler, available_features
    
    X_sequences, y_targets = np.array(X_sequences), np.array(y_targets)
    
    # Log preprocessing results
    logger.model(f"Preprocessing complete: {len(X_sequences)} sequences, shape {X_sequences.shape}")
    if output_mode == 'classification':
        unique, counts = np.unique(y_targets, return_counts=True)
        logger.model(f"Target distribution: {dict(zip(unique, counts))}")
    else:
        logger.model(f"Target range: [{np.min(y_targets):.4f}, {np.max(y_targets):.4f}]")
    
    return X_sequences, y_targets, scaler, available_features

def split_train_test_data(X: np.ndarray, y: np.ndarray, train_ratio: float = TRAIN_TEST_SPLIT, validation_ratio: float = VALIDATION_SPLIT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets with data validation.
    
    Args:
        X: Input sequences array 
        y: Target array
        train_ratio: Ratio for training set
        validation_ratio: Ratio for validation set
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays")
    
    n_samples = len(X)
    if n_samples != len(y):
        raise ValueError(f"X and y length mismatch: X={n_samples}, y={len(y)}")
    
    if n_samples < 10:
        raise ValueError(f"Insufficient data: {n_samples} samples, need at least 10")
    
    if not (0 < train_ratio < 1) or not (0 < validation_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1")
    
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Sum of ratios must be less than 1")
    
    train_end = max(int(n_samples * train_ratio), 3)
    val_end = max(int(n_samples * (train_ratio + validation_ratio)), train_end + 2)
    val_end = min(val_end, n_samples - 1)
    
    logger.model(f"Data split - Train: {train_end}, Val: {val_end - train_end}, Test: {n_samples - val_end}")
    
    return (X[:train_end], X[train_end:val_end], X[val_end:],
            y[:train_end], y[train_end:val_end], y[val_end:])

def create_cnn_lstm_attention_model(input_size: int, use_attention: bool = True, 
                                    use_cnn: bool = False, look_back: int = WINDOW_SIZE_LSTM, 
                                    output_mode: str = 'classification', **kwargs) -> Union[LSTMModel, LSTMAttentionModel, CNNLSTMAttentionModel]:
    """
    Create CNN-LSTM-Attention model based on configuration.
    
    Args:
        input_size: Number of input features
        use_attention: Whether to use attention mechanism
        use_cnn: Whether to use CNN layers
        look_back: Sequence length for time series
        output_mode: 'classification' or 'regression'
        **kwargs: Additional model parameters
        
    Returns:
        Neural network model configured according to parameters
    """
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"input_size must be positive integer, got {input_size}")
    if not isinstance(look_back, int) or look_back <= 0:
        raise ValueError(f"look_back must be positive integer, got {look_back}")
    if output_mode not in ['classification', 'regression']:
        raise ValueError(f"output_mode must be 'classification' or 'regression', got {output_mode}")
    
    if use_cnn:
        logger.model(f"Creating CNN-LSTM-Attention model with {output_mode} mode")
        validated_output_mode: Literal['classification', 'regression'] = 'classification' if output_mode == 'classification' else 'regression'
        
        model = CNNLSTMAttentionModel(
            input_size=input_size,
            look_back=look_back,
            output_mode=validated_output_mode,
            use_attention=use_attention,
            **kwargs
        )
        logger.model("Created CNN-LSTM-Attention model with {0} mode".format(output_mode))
        return model
    
    if use_attention:
        logger.model("Creating LSTM model with Multi-Head Attention")
        attention_params = {k: v for k, v in kwargs.items() 
                          if k in ['num_heads', 'dropout', 'hidden_size', 'num_layers']}
        return LSTMAttentionModel(input_size=input_size, **attention_params)
    
    logger.model("Creating standard LSTM model")
    lstm_params = {k: v for k, v in kwargs.items() 
                  if k in ['dropout', 'hidden_size', 'num_layers']}
    return LSTMModel(input_size=input_size, **lstm_params)

def train_cnn_lstm_attention_model(df_input: pd.DataFrame,
                                    save_model: bool = True,
                                    epochs: int = DEFAULT_EPOCHS,
                                    use_early_stopping: bool = True,
                                    early_stopping_patience: int = 10,
                                    early_stopping_min_delta: float = 1e-4,
                                    use_attention: bool = True,
                                    use_cnn: bool = True,
                                    look_back: int = WINDOW_SIZE_LSTM,
                                    output_mode: str = 'classification',
                                    attention_heads: int = GPU_MODEL_CONFIG['nhead']) -> Tuple[Optional[nn.Module], GridSearchThresholdOptimizer]:
    """
    Train CNN-LSTM model with advanced attention mechanism and GPU optimization.
    
    This function implements state-of-the-art training with:
    • Advanced early stopping with multiple criteria (loss + accuracy)
    • Dynamic GPU memory optimization and batch size scaling
    • Mixed precision training for faster GPU computation
    • Adaptive learning rate scheduling with plateau detection
    • Comprehensive GPU resource management and monitoring
    • Enhanced training history tracking and performance logging
    
    Args:
        df_input: Input DataFrame with price data
        save_model: Whether to save the trained model with comprehensive metadata
        epochs: Maximum number of training epochs
        use_early_stopping: Enable multi-criteria early stopping system
        early_stopping_patience: Number of epochs to wait before stopping if no improvement
        early_stopping_min_delta: Minimum change in validation metrics to qualify as improvement
        use_attention: Use multi-head attention mechanism for temporal dependencies
        use_cnn: Use CNN layers for local feature extraction
        look_back: Sequence length for time series input
        output_mode: 'classification' for signal prediction or 'regression' for return prediction
        attention_heads: Number of attention heads for multi-head attention
        
    Returns:
        Tuple of (trained_model, threshold_optimizer) with comprehensive training metadata
        
    GPU Optimizations:
        • Dynamic batch size based on available GPU memory
        • Mixed precision training (FP16) for compatible GPUs
        • Advanced CuDNN optimizations and TF32 acceleration
        • Persistent data workers and prefetching for faster data loading
        • Real-time GPU memory monitoring and automatic cleanup
        
    Early Stopping Features:
        • Multi-criteria monitoring (validation loss + accuracy/metric)
        • Adaptive learning rate reduction on plateau detection
        • Best model state preservation and restoration
        • Comprehensive training progress tracking and logging
    """
    model, best_model_state, scaler_amp = None, None, None
    threshold_optimizer = GridSearchThresholdOptimizer()
    epoch = 0
    
    # GPU setup with resource manager
    gpu_manager = get_gpu_resource_manager()
    
    with gpu_manager.gpu_scope() as device:
        if device is None:
            device = torch.device('cpu')
            logger.info("Using CPU for CNN-LSTM training")
            use_mixed_precision = False
        else:
            logger.gpu(f"Using GPU for CNN-LSTM training: {device}")
            # Enable mixed precision for compatible GPUs
            use_mixed_precision = (device.type == 'cuda' and 
                                 torch.cuda.get_device_capability(device)[0] >= 7)
            
            if use_mixed_precision:
                scaler_amp = GradScaler()
                logger.gpu("Mixed precision training enabled for faster GPU training")
            
            # Log GPU memory info
            memory_info = gpu_manager.get_memory_info()
            logger.gpu(f"GPU Memory - Total: {memory_info['total'] // 1024**3}GB, "
                      f"Allocated: {memory_info['allocated'] // 1024**2}MB")
    
        # Data validation
        if df_input.empty or len(df_input) < look_back + 50:
            raise ValueError(f"Insufficient data: {len(df_input)} rows, need at least {look_back + 50}")
        
        logger.model(f"Starting CNN-LSTM pipeline - Look back: {look_back}, Mode: {output_mode}, CNN: {use_cnn}")
        
        # Preprocess data
        X, y, scaler, feature_names = preprocess_cnn_lstm_data(
            df_input, look_back=look_back, output_mode=output_mode, scaler_type='minmax'
        )
    
    if len(X) == 0:
        logger.error(f"Preprocessing failed - Input shape: {df_input.shape}")
        basic_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_input.columns]
        logger.error(f"Available columns: {list(df_input.columns)}")
        logger.error(f"Required features: {MODEL_FEATURES}")
        logger.error(f"Feature names returned: {feature_names}")
        
        if not basic_cols:
            logger.error("No basic OHLC columns found in input data")
            # Try to continue with minimal features if possible
            if 'close' in df_input.columns or COL_CLOSE in df_input.columns:
                logger.warning("Attempting to create minimal feature set...")
                minimal_df = df_input.copy()
                close_col = 'close' if 'close' in df_input.columns else COL_CLOSE
                
                # Create basic features from close price only
                minimal_df['returns'] = minimal_df[close_col].pct_change()
                minimal_df['sma_5'] = minimal_df[close_col].rolling(5).mean()
                minimal_df['sma_20'] = minimal_df[close_col].rolling(20).mean()
                minimal_df['volatility'] = minimal_df['returns'].rolling(10).std()
                
                # Retry preprocessing with minimal features
                logger.warning("Retrying preprocessing with minimal feature set...")
                X, y, scaler, feature_names = preprocess_cnn_lstm_data(
                    minimal_df, look_back=max(5, look_back//2), output_mode=output_mode, scaler_type='minmax'
                )
                
                if len(X) > 0:
                    logger.warning(f"Successfully created {len(X)} sequences with minimal features")
                else:
                    raise ValueError("Failed to create sequences even with minimal features - data may be corrupted or insufficient")
            else:
                raise ValueError("No price data columns found in input data")
        else:
            raise ValueError(f"Data preprocessing failed - no valid sequences created. Available basic columns: {basic_cols}")
    
    # Split data and prepare tensors
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y, 0.7, 0.15)
    
    if len(X_train) == 0:
        raise ValueError("Insufficient data after train/test split")
    
    X_train, X_val, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test)
    
    if output_mode == 'classification':
        y_train, y_val, y_test = torch.LongTensor(y_train + 1), torch.LongTensor(y_val + 1), torch.LongTensor(y_test + 1)
        num_classes = 3
    else:
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        num_classes = 1
    
    # Create model
    input_size = len(feature_names)
    try:
        model = create_cnn_lstm_attention_model(
            input_size=input_size,
            use_attention=use_attention,
            use_cnn=use_cnn,
            look_back=look_back,
            output_mode=output_mode,
            num_heads=attention_heads,
            cnn_features=64,
            lstm_hidden=32,
            num_classes=num_classes,
            dropout=0.3
        ).to(device)
        
        model_type = "CNN-LSTM-Attention" if use_cnn else ("LSTM-Attention" if use_attention else "LSTM")
        logger.model(f"{model_type} model created - Input: {input_size}, Look back: {look_back}, Classes: {num_classes}")
        
    except Exception as model_error:
        logger.error(f"Model creation failed: {model_error}")
        raise ValueError(f"Cannot create CNN-LSTM model: {model_error}")
    
    # Training setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if output_mode == 'classification' else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Enhanced batch size optimization with GPU memory considerations
    gpu_enabled = device.type == 'cuda'
    
    if gpu_enabled:
        # Get GPU memory info for dynamic batch size optimization
        memory_info = gpu_manager.get_memory_info()
        available_memory = memory_info['total'] - memory_info['allocated']
        memory_gb = available_memory / (1024**3)
        
        # Dynamic batch size based on available GPU memory and model complexity
        base_batch_size = get_optimal_batch_size(device, input_size, look_back)
        
        if use_cnn:
            # CNN requires more memory, be more conservative
            memory_multiplier = min(2.0, memory_gb / 4.0)  # Scale based on available memory
            optimal_batch_size = max(8, int(base_batch_size * memory_multiplier * 0.6))
        else:
            # LSTM-only can use larger batches
            memory_multiplier = min(3.0, memory_gb / 2.0)
            optimal_batch_size = max(16, int(base_batch_size * memory_multiplier))
        
        # Ensure batch size is power of 2 for better GPU utilization
        optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
        optimal_batch_size = min(optimal_batch_size, 512)  # Cap at reasonable maximum
        
        logger.gpu(f"GPU-optimized batch size: {optimal_batch_size} (Memory: {memory_gb:.1f}GB available)")
        
        # Enable advanced GPU features
        pin_memory = True
        num_workers = min(4, torch.cuda.device_count() * 2)  # More workers for faster data loading
        persistent_workers = True
        prefetch_factor = 4  # Prefetch more batches
        
        # Enable GPU optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True
        
        logger.gpu("Advanced GPU optimizations enabled: CuDNN benchmark, TF32, persistent workers")
    else:
        optimal_batch_size = max(8, get_optimal_batch_size(device, input_size, look_back))
        pin_memory = False
        num_workers = 1
        persistent_workers = False
        prefetch_factor = 2
    
    # Data loaders with advanced GPU optimization
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=optimal_batch_size, 
        shuffle=True, 
        pin_memory=pin_memory, 
        num_workers=num_workers, 
        drop_last=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=optimal_batch_size,
        shuffle=False, 
        pin_memory=pin_memory, 
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2
    )
    
    # Advanced early stopping configuration with multiple criteria
    best_val_loss = float('inf')
    best_val_metric = -float('inf')  # For accuracy or other metrics
    best_epoch = 0
    patience_counter = 0
    patience = early_stopping_patience
    min_delta = early_stopping_min_delta
    
    # Enhanced early stopping with learning rate decay monitoring
    lr_patience = max(5, patience // 3)  # Reduce LR if no improvement for 1/3 of patience
    lr_reduction_counter = 0
    min_lr = 1e-7
    
    # Performance tracking for advanced early stopping
    val_loss_history = []
    val_metric_history = []
    training_history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'learning_rates': [], 'gpu_memory': []
    }
    
    # GPU performance monitoring
    if gpu_enabled:
        initial_memory = gpu_manager.get_memory_info()
        logger.gpu(f"Training started - GPU Memory: {initial_memory['allocated'] // 1024**2}MB allocated, "
                  f"{initial_memory['total'] // 1024**3}GB total")
    
    logger.model(f"Training {model_type} for {epochs} epochs - Batch size: {optimal_batch_size}, Mixed precision: {use_mixed_precision}")
    
    # Training loop with enhanced early stopping and GPU optimization
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = train_total = 0
            
            # GPU memory monitoring
            if gpu_enabled and epoch == 0:
                torch.cuda.reset_peak_memory_stats(device)
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                if use_mixed_precision and scaler_amp is not None:
                    with autocast('cuda'):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                
                if output_mode == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
                
                # GPU memory cleanup for large batches
                if gpu_enabled and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                    
                    if use_mixed_precision:
                        with autocast('cuda'):
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    
                    if output_mode == 'classification':
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics and update scheduler
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step()
            
            # Logging with enhanced metrics and store training history
            if output_mode == 'classification':
                train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
                val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
                training_history['train_acc'].append(train_acc)
                training_history['val_acc'].append(val_acc)
                logger.performance(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                                 f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
                current_metric = val_acc  # Use accuracy as monitoring metric for classification
            else:
                logger.performance(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                                 f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                current_metric = -val_loss  # Use negative loss as monitoring metric for regression
            
            # Store training history for analysis
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if gpu_enabled:
                current_memory = gpu_manager.get_memory_info()
                training_history['gpu_memory'].append(current_memory['allocated'] // 1024**2)
            
            # Enhanced early stopping with multiple criteria and adaptive learning rate
            if use_early_stopping:
                improvement = False
                significant_improvement = False
                
                # Check validation loss improvement
                if val_loss < (best_val_loss - min_delta):
                    improvement_amount = best_val_loss - val_loss
                    best_val_loss = val_loss
                    improvement = True
                    if improvement_amount > min_delta * 5:  # Significant improvement
                        significant_improvement = True
                
                # For classification, also check accuracy improvement
                if output_mode == 'classification' and current_metric > (best_val_metric + min_delta):
                    improvement_amount = current_metric - best_val_metric
                    best_val_metric = current_metric
                    improvement = True
                    if improvement_amount > min_delta * 10:  # Significant improvement for accuracy
                        significant_improvement = True
                
                # For regression, check negative loss improvement
                if output_mode == 'regression' and current_metric > (best_val_metric + min_delta):
                    improvement_amount = current_metric - best_val_metric
                    best_val_metric = current_metric
                    improvement = True
                    if improvement_amount > min_delta * 5:  # Significant improvement
                        significant_improvement = True
                
                # Update best model and counters
                if improvement:
                    patience_counter = 0
                    lr_reduction_counter = 0
                    best_epoch = epoch
                    best_model_state = model.state_dict().copy()
                    
                    if significant_improvement:
                        logger.performance(f"🚀 Significant improvement - Val Loss: {val_loss:.4f}, Metric: {current_metric:.4f}")
                    else:
                        logger.debug(f"Model improved - Val Loss: {val_loss:.4f}, Metric: {current_metric:.4f}")
                else:
                    patience_counter += 1
                    lr_reduction_counter += 1
                
                # Track validation history for plateau detection
                val_loss_history.append(val_loss)
                val_metric_history.append(current_metric)
                
                # Keep only recent history for analysis
                if len(val_loss_history) > patience:
                    val_loss_history.pop(0)
                    val_metric_history.pop(0)
                
                # Advanced learning rate scheduling based on plateau detection
                if lr_reduction_counter >= lr_patience and optimizer.param_groups[0]['lr'] > min_lr:
                    # Check if we're in a plateau (loss not improving significantly)
                    if len(val_loss_history) >= lr_patience:
                        recent_losses = val_loss_history[-lr_patience:]
                        loss_improvement = max(recent_losses) - min(recent_losses)
                        
                        if loss_improvement < min_delta * 2:  # Very little improvement
                            old_lr = optimizer.param_groups[0]['lr']
                            new_lr = max(old_lr * 0.5, min_lr)
                            
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            
                            lr_reduction_counter = 0
                            logger.performance(f"📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e} (plateau detected)")
                
                # Log patience status with enhanced information
                if patience_counter > 0:
                    progress_pct = (patience_counter / patience) * 100
                    epochs_since_best = epoch - best_epoch
                    logger.debug(f"No improvement for {patience_counter}/{patience} epochs ({progress_pct:.1f}%) - "
                               f"Best was {epochs_since_best} epochs ago")
                
                # Early stopping with confirmation
                if patience_counter >= patience:
                    logger.model(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    logger.model(f"   • Best epoch: {best_epoch + 1} (Val Loss: {best_val_loss:.4f})")
                    logger.model(f"   • Patience exhausted: {patience_counter}/{patience}")
                    
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        logger.model("   • Restored best model state")
                    break
            
            # Enhanced GPU memory monitoring and optimization
            epoch_time = time.time() - epoch_start_time
            
            if gpu_enabled:
                # More frequent memory cleanup for stability
                if (epoch + 1) % 3 == 0:
                    torch.cuda.empty_cache()
                
                # Detailed memory logging every 5 epochs
                if (epoch + 1) % 5 == 0:
                    memory_info = gpu_manager.get_memory_info()
                    peak_memory = torch.cuda.max_memory_allocated(device) // 1024**2
                    
                    logger.gpu(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.1f}s, "
                              f"Memory: {memory_info['allocated'] // 1024**2}MB current, "
                              f"{peak_memory}MB peak")
                    
                    # Reset peak memory stats periodically
                    torch.cuda.reset_peak_memory_stats(device)
            else:
                if (epoch + 1) % 10 == 0:
                    logger.performance(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.1f}s (CPU mode)")
    
    except Exception as training_error:
        logger.error(f"🚨 Training failed at epoch {epoch + 1}: {training_error}")
        logger.error(f"   • Training history length: {len(training_history.get('train_loss', []))}")
        
        if gpu_enabled:
            # Emergency GPU cleanup
            torch.cuda.empty_cache()
            memory_info = gpu_manager.get_memory_info()
            logger.gpu(f"   • GPU memory after error: {memory_info['allocated'] // 1024**2}MB")
        
        # Try to save partial progress if possible
        if best_model_state is not None:
            logger.warning("Attempting to restore best model state from training...")
            model.load_state_dict(best_model_state)
        
        import traceback
        logger.error(f"   • Full traceback: {traceback.format_exc()}")
    
    # Final training summary
    total_epochs_trained = epoch + 1 if 'epoch' in locals() else 0
    logger.model(f"🏁 Training completed - Total epochs: {total_epochs_trained}")
    
    if 'best_epoch' in locals():
        logger.model(f"   • Best epoch: {best_epoch + 1} (Val Loss: {best_val_loss:.4f})")
        logger.model(f"   • Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    if gpu_enabled:
        final_memory = gpu_manager.get_memory_info()
        peak_memory = torch.cuda.max_memory_allocated(device) // 1024**2
        logger.gpu(f"   • Final GPU usage: {final_memory['allocated'] // 1024**2}MB current, {peak_memory}MB peak")
    
    # Model evaluation and threshold optimization
    logger.model("Starting model evaluation and threshold optimization...")
    
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        test_predictions = model(X_test).cpu().numpy()
    
    # Clean up GPU memory after evaluation
    if gpu_enabled:
        torch.cuda.empty_cache()
        final_memory = gpu_manager.get_memory_info()
        logger.gpu(f"Post-evaluation GPU Memory: {final_memory['allocated'] // 1024**2}MB allocated")
    
    if output_mode == 'regression':
        test_returns = y_test.squeeze().cpu().numpy()
        close_col = 'close' if 'close' in df_input.columns else COL_CLOSE
        prices = df_input[close_col].values[-len(test_returns):]
        best_threshold, best_sharpe = threshold_optimizer.optimize_regression_threshold(
            test_predictions.flatten(), test_returns, prices
        )
        logger.model(f"Regression optimization - Threshold: {best_threshold or 0.02:.4f}, "
                    f"Sharpe: {best_sharpe if best_sharpe is not None else 0.0:.4f}")
    else:
        test_returns = y_test.squeeze().cpu().numpy() - 1
        best_confidence, best_sharpe = threshold_optimizer.optimize_classification_threshold(
            test_predictions, test_returns
        )
        logger.model(f"Classification optimization - Confidence: {best_confidence or 0.7:.2f}, "
                    f"Sharpe: {best_sharpe if best_sharpe is not None else 0.0:.4f}")
    
    # Enhanced model saving with comprehensive metadata
    if save_model and model is not None:
            try:
                model_filename = f"cnn_lstm_attention_{output_mode}_model.pth"
                model_path = MODELS_DIR / model_filename
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                
                # Comprehensive save dictionary with training insights
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'input_size': input_size,
                        'look_back': look_back,
                        'output_mode': output_mode,
                        'use_cnn': use_cnn,
                        'use_attention': use_attention,
                        'attention_heads': attention_heads,
                        'num_classes': num_classes,
                        'model_type': model_type
                    },
                    'training_info': {
                        'epochs_trained': total_epochs_trained,
                        'best_epoch': best_epoch + 1 if 'best_epoch' in locals() else 1,
                        'best_val_loss': best_val_loss,
                        'final_lr': optimizer.param_groups[0]['lr'],
                        'initial_lr': 0.001,  # Default initial LR
                        'early_stopping_patience': early_stopping_patience,
                        'min_delta': early_stopping_min_delta,
                        'used_mixed_precision': use_mixed_precision,
                        'batch_size': optimal_batch_size,
                        'training_device': str(device),
                        'training_time': datetime.now().isoformat()
                    },
                    'data_info': {
                        'scaler': scaler,
                        'feature_names': feature_names,
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'test_samples': len(X_test),
                        'input_shape': list(X_train.shape),
                        'scaler_type': 'MinMaxScaler'
                    },
                    'gpu_info': {
                        'gpu_used': gpu_enabled,
                        'device_name': torch.cuda.get_device_name(device) if gpu_enabled else 'CPU',
                        'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
                        'peak_memory_mb': torch.cuda.max_memory_allocated(device) // 1024**2 if gpu_enabled else 0
                    } if gpu_enabled else {'gpu_used': False},
                    'training_history': training_history,
                    'optimization_results': {
                        'optimal_threshold': threshold_optimizer.best_threshold,
                        'best_sharpe': threshold_optimizer.best_sharpe
                    },
                    'training_history': training_history
                }
                
                torch.save(save_dict, model_path)
                logger.success(f"CNN-LSTM model saved to {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
        
    return model, threshold_optimizer

def train_and_save_global_cnn_lstm_attention_model(combined_df: pd.DataFrame, 
                                                model_filename: Optional[str] = None,
                                                use_attention: bool = True,
                                                use_cnn: bool = True,
                                                look_back: int = WINDOW_SIZE_LSTM,
                                                output_mode: str = 'classification',
                                                attention_heads: int = GPU_MODEL_CONFIG['nhead'],
                                                early_stopping_patience: int = 20,
                                                early_stopping_min_delta: float = 1e-4) -> Tuple[Optional[nn.Module], str]:
    """
    Train and save CNN-LSTM-Attention model for global price data.
    
    Args:
        combined_df: Combined DataFrame with price data for all trading pairs
        model_filename: Optional custom model filename
        use_attention: Enable attention mechanism
        use_cnn: Enable CNN layers
        look_back: Time steps for sequence modeling
        output_mode: 'classification' or 'regression'
        attention_heads: Number of attention heads
        early_stopping_patience: Number of epochs to wait before stopping if no improvement
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement
        
    Returns:
        Tuple of (trained_model, model_path_string)
    """
    start_time = time.time()

    try:
        # Use GPU resource manager for better GPU utilization
        gpu_manager = get_gpu_resource_manager()
        
        with gpu_manager.gpu_scope() as device:
            if device is None:
                logger.info("Using CPU for CNN-LSTM model training")
            else:
                logger.gpu(f"Using GPU for CNN-LSTM model training: {device}")
                memory_info = gpu_manager.get_memory_info()
                logger.gpu(f"GPU Memory - Total: {memory_info['total'] // 1024**3}GB, "
                          f"Available: {(memory_info['total'] - memory_info['allocated']) // 1024**2}MB")

            if not model_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                model_type = "cnn_lstm_attention" if use_cnn else ("lstm_attention" if use_attention else "lstm")
                model_filename = f"{model_type}_{output_mode}_model_{timestamp}.pth"

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / model_filename

            logger.model(f"Training CNN-LSTM{'-Attention' if use_attention else ''} model...")
            
            model, threshold_optimizer = train_cnn_lstm_attention_model(
                combined_df, 
                save_model=False,
                use_attention=use_attention,
                use_cnn=use_cnn,
                look_back=look_back,
                output_mode=output_mode,
                attention_heads=attention_heads,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta
            )

        if not model:
            logger.error("Model training failed - returned None")
            return None, ""

        logger.model(f"Saving model to: {model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': len(MODEL_FEATURES),
                'look_back': look_back,
                'output_mode': output_mode,
                'use_cnn': use_cnn,
                'use_attention': use_attention,
                'attention_heads': attention_heads
            },
            'optimization_results': {
                'optimal_threshold': threshold_optimizer.best_threshold,
                'best_sharpe': threshold_optimizer.best_sharpe
            }
        }, model_path)

        elapsed_time = time.time() - start_time
        logger.performance(f"CNN-LSTM model trained and saved in {elapsed_time:.2f}s: {model_path}")

        return model, str(model_path)

    except Exception as e:
        logger.exception(f"Error during CNN-LSTM model training: {e}")
        return None, ""

# NEW METHODS
def load_cnn_lstm_attention_model(model_path: Union[str, Path]) -> Optional[Tuple[nn.Module, Dict, Dict, Dict]]:
    """
    Load a trained CNN-LSTM-Attention model from a file.
    
    Args:
        model_path: Path to the model file (.pth)
        
    Returns:
        A tuple containing (model, model_config, data_info, optimization_results) or None if loading fails.
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        model_config = checkpoint.get('model_config', {})
        data_info = checkpoint.get('data_info', {})
        optimization_results = checkpoint.get('optimization_results', {})
        
        if not model_config or not data_info:
            logger.error(f"Model checkpoint at {model_path} is missing required 'model_config' or 'data_info' keys.")
            return None

        # Recreate model from config using CNNLSTMAttentionModel since that's what create_cnn_lstm_attention_model returns
        model = CNNLSTMAttentionModel(
            input_size=model_config['input_size'],
            look_back=model_config['look_back'],
            output_mode=model_config['output_mode'],
            use_attention=model_config['use_attention'],
            cnn_features=model_config.get('cnn_features', 64),
            lstm_hidden=model_config.get('lstm_hidden', 32),
            num_classes=model_config.get('num_classes', 3 if model_config['output_mode'] == 'classification' else 1),
            num_heads=model_config.get('attention_heads', 4),
            dropout=model_config.get('dropout', 0.3)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.success(f"Successfully loaded CNN-LSTM model from {model_path}")
        return model, model_config, data_info, optimization_results

    except Exception as e:
        logger.error(f"Failed to load CNN-LSTM model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_latest_cnn_lstm_attention_signal(
    df_input: pd.DataFrame, 
    model: nn.Module, 
    model_config: Dict, 
    data_info: Dict,
    optimization_results: Dict
) -> str:
    """
    Get the latest signal from a loaded CNN-LSTM-Attention model.
    
    Args:
        df_input: DataFrame with the latest market data. Must contain enough rows for look_back.
        model: The loaded PyTorch model.
        model_config: Dictionary with model configuration.
        data_info: Dictionary with data processing info (scaler, feature_names).
        optimization_results: Dictionary with optimization results (e.g., optimal_threshold).
        
    Returns:
        Signal string ('LONG', 'SHORT', 'NEUTRAL').
    """
    try:
        device = next(model.parameters()).device
        look_back = model_config['look_back']
        scaler = data_info['scaler']
        feature_names = data_info['feature_names']

        if len(df_input) < look_back:
            logger.warning(f"Not enough data for prediction: got {len(df_input)} rows, need {look_back}")
            return SIGNAL_NEUTRAL

        # 1. Generate features
        df = generate_indicator_features(df_input.copy())
        
        # 2. Select and scale features for the last sequence
        latest_data = df.tail(look_back)
        if len(latest_data) < look_back:
            logger.warning(f"Not enough data after feature generation: got {len(latest_data)} rows, need {look_back}")
            return SIGNAL_NEUTRAL
            
        available_features = [f for f in feature_names if f in latest_data.columns]
        if len(available_features) != len(feature_names):
             logger.warning(f"Mismatched features. Model needs {len(feature_names)} features, data has {len(available_features)}.")
             return SIGNAL_NEUTRAL
        
        features = latest_data[feature_names].values
        features = np.nan_to_num(features, nan=0.0)
        scaled_features = scaler.transform(features)
        
        # 3. Create tensor
        sequence_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)

        # 4. Predict
        with torch.no_grad():
            output = model(sequence_tensor)
            
        # 5. Interpret output
        if model_config['output_mode'] == 'classification':
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence_threshold = optimization_results.get('optimal_threshold', 0.5)

            if confidence.item() < confidence_threshold:
                logger.signal(f"CNN-LSTM Prediction: NEUTRAL (Confidence {confidence.item():.2f} < Threshold {confidence_threshold:.2f})")
                return SIGNAL_NEUTRAL

            if predicted_class == 2: # LONG (mapped from 1)
                signal = SIGNAL_LONG
            elif predicted_class == 0: # SHORT (mapped from -1)  
                signal = SIGNAL_SHORT
            else: # NEUTRAL (class 1, mapped from 0)
                signal = SIGNAL_NEUTRAL
            
            logger.signal(f"CNN-LSTM Prediction: {signal} (Confidence: {confidence.item():.2f})")
            return signal
        else:
            logger.warning("Regression output mode not implemented for signal generation.")
            return SIGNAL_NEUTRAL

    except Exception as e:
        logger.error(f"Error during CNN-LSTM signal prediction: {e}")
        import traceback
        traceback.print_exc()
        return SIGNAL_NEUTRAL