import logging
import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Literal, Optional, Tuple, Union

# Add the parent directory to sys.path to allow importing modules from sibling directories
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
            
            cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else "Unknown" # type: ignore
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
    WINDOW_SIZE_LSTM,
)

from signals._components._gpu_check_availability import (check_gpu_availability, configure_gpu_memory)
from signals._components.LSTM__class__focal_loss import FocalLoss
from signals._components.LSTM__class__grid_search_threshold_optimizer import GridSearchThresholdOptimizer
from signals._components.LSTM__class__models import (CNNLSTMAttentionModel, LSTMModel, LSTMAttentionModel)
from signals._components._generate_indicator_features import _generate_indicator_features
from signals._components.LSTM__function__create_balanced_target import create_balanced_target
from signals._components.LSTM__function__get_optimal_batch_size import get_optimal_batch_size

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
    df = _generate_indicator_features(df_input.copy())
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
                                    use_attention: bool = True,
                                    use_cnn: bool = True,
                                    look_back: int = WINDOW_SIZE_LSTM,
                                    output_mode: str = 'classification',
                                    attention_heads: int = GPU_MODEL_CONFIG['nhead']) -> Tuple[Optional[nn.Module], GridSearchThresholdOptimizer]:
    """
    Train CNN-LSTM model with attention mechanism.
    
    Args:
        df_input: Input DataFrame with price data
        save_model: Whether to save the trained model
        epochs: Number of training epochs
        use_early_stopping: Enable early stopping
        use_attention: Use attention mechanism
        use_cnn: Use CNN layers
        look_back: Sequence length for time series
        output_mode: 'classification' or 'regression'
        attention_heads: Number of attention heads
        
    Returns:
        Tuple of (trained_model, threshold_optimizer)
    """
    model, best_model_state, scaler_amp = None, None, None
    threshold_optimizer = GridSearchThresholdOptimizer()
    epoch = 0
    
    # GPU setup and mixed precision configuration
    gpu_available = check_gpu_availability()
    device = torch.device('cuda:0' if gpu_available and configure_gpu_memory() else 'cpu')
    use_mixed_precision = gpu_available and device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 7
    
    if use_mixed_precision:
        scaler_amp = GradScaler()
        logger.gpu("Using mixed precision training for CNN-LSTM")
    
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
    
    # Batch size optimization
    optimal_batch_size = get_optimal_batch_size(device, input_size, look_back)
    if use_cnn:
        optimal_batch_size = max(4, optimal_batch_size // 8)
        logger.gpu(f"CNN optimized batch size: {optimal_batch_size}")
    
    # Data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=optimal_batch_size, 
        shuffle=True, pin_memory=gpu_available, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=optimal_batch_size,
        shuffle=False, pin_memory=gpu_available, num_workers=0
    )
    
    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    training_history = {'train_loss': [], 'val_loss': []}
    
    logger.model(f"Training {model_type} for {epochs} epochs - Batch size: {optimal_batch_size}, Mixed precision: {use_mixed_precision}")
    
    # Training loop
    try:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                optimizer.zero_grad()
                
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
            
            # Calculate metrics and update
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            scheduler.step()
            
            # Logging
            if output_mode == 'classification':
                train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
                val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
                logger.performance(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                                 f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            else:
                logger.performance(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                                 f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    logger.debug(f"New best validation loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.model(f"Early stopping triggered after {epoch+1} epochs (patience: {patience})")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        logger.model("Restored best model state")
                    break
    
    except Exception as training_error:
        logger.error(f"Training failed at epoch {epoch + 1}: {training_error}")
    
    # Model evaluation and threshold optimization
    logger.model("Starting model evaluation...")
    
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        test_predictions = model(X_test).cpu().numpy()
    
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
    
    # Model saving
    if save_model and model is not None:
        try:
            model_filename = f"cnn_lstm_attention_{output_mode}_model.pth"
            model_path = MODELS_DIR / model_filename
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            save_dict = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': input_size,
                    'look_back': look_back,
                    'output_mode': output_mode,
                    'use_cnn': use_cnn,
                    'use_attention': use_attention,
                    'attention_heads': attention_heads,
                    'num_classes': num_classes
                },
                'training_info': {
                    'epochs_trained': epoch + 1,
                    'best_val_loss': best_val_loss,
                    'final_lr': optimizer.param_groups[0]['lr']
                },
                'data_info': {
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test)
                },
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
                                                attention_heads: int = GPU_MODEL_CONFIG['nhead']) -> Tuple[Optional[nn.Module], str]:
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
        
    Returns:
        Tuple of (trained_model, model_path_string)
    """
    start_time = time.time()

    try:
        gpu_available = check_gpu_availability()
        if gpu_available:
            configure_gpu_memory()
            logger.gpu("Using GPU for CNN-LSTM model training")
        else:
            logger.info("Using CPU for CNN-LSTM model training")

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
            attention_heads=attention_heads
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