"""
This module provides a unified API for training, loading, and predicting signals
from four different LSTM-based model variants for cryptocurrency trading.

The supported variants are:
- Standard LSTM
- LSTM with Attention
- CNN-LSTM
- CNN-LSTM with Attention

The module includes functions for training individual models, batch training all
variants, and a consistent interface for loading and generating predictions from
any of the trained models. It also handles GPU/CPU optimization, mixed-precision
training, and includes safety checks for PyTorch version compatibility.

Pylint C0302: This module is too long and should be refactored.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.config import (
    DEFAULT_EPOCHS, MODEL_FEATURES,
    SIGNAL_NEUTRAL,
    WINDOW_SIZE_LSTM
)
from components._generate_indicator_features import generate_indicator_features
from signals._components.LSTM__class__Models import (
    CNNLSTMAttentionModel, LSTMAttentionModel, LSTMModel
)

from utilities._logger import setup_logging
from utilities._gpu_resource_manager import get_gpu_resource_manager
from signals._components.LSTM__function__preprocessor import (
    create_target_variable,
    scale_data,
    create_sequences,
)
from signals._components.LSTM__class__ModelTrainer import ModelTrainer
from signals.signals_cnn_lstm_attention_helpers import (
    _create_training_config,
    _create_data_loaders,
    _evaluate_model,
    _save_model_checkpoint,
    _validate_prediction_inputs,
    _prepare_prediction_data,
    _process_classification_output,
    _process_regression_output,
)

os.environ.update({
    'KMP_DUPLICATE_LIB_OK': 'True',
    'OMP_NUM_THREADS': '1',
    'CUDA_LAUNCH_BLOCKING': '1',
    'TORCH_USE_CUDA_DSA': '1'
})


# Setup logging
logger = setup_logging(
    module_name="signals_cnn_lstm_attention",
    log_level=logging.DEBUG
)

# Initialize GPU Resource Manager
gpu_manager = get_gpu_resource_manager()
gpu_manager.setup_environment()

# PyTorch availability check with GPU Resource Manager
try:
    logger.success(f"PyTorch {torch.__version__} loaded successfully")
    
    if gpu_manager.is_cuda_available:
        try:
            # Initialize GPU with resource manager
            if gpu_manager.initialize(device_id=0):
                cuda_version = "Unknown"
                if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):    # type: ignore
                    cuda_version = torch.version.cuda                               # type: ignore
                logger.gpu(
                    f"CUDA {cuda_version} available with "
                    f"{torch.cuda.device_count()} device(s) - "
                    f"Using device: {gpu_manager.current_device}"
                )
                
                # Log GPU memory info
                memory_info = gpu_manager.get_memory_info()
                if isinstance(memory_info['total'], int) and isinstance(memory_info['allocated'], int):
                    logger.memory(f"GPU Memory - Total: {memory_info['total']/1024**3:.1f}GB, "
                               f"Allocated: {memory_info['allocated']/1024**3:.1f}GB")
                else:
                    logger.memory(f"GPU Memory - Device: {memory_info['device']}")
                
                # Log Tensor Core information
                tensor_core_info = gpu_manager.get_tensor_core_info()
                if tensor_core_info['has_tensor_cores']:
                    logger.gpu(f"Tensor Cores available: {tensor_core_info['generation']} "
                               f"(Compute Capability: {tensor_core_info['compute_capability']})")
            else:
                logger.warning("GPU initialization failed, falling back to CPU")
        except Exception as cuda_error:
            logger.warning(f"CUDA available but not functional: {cuda_error}")
            def is_cuda_available_false():
                return False
            torch.cuda.is_available = is_cuda_available_false
    else:
        logger.info("CUDA not available, using CPU mode")
        
except ImportError as e:
    logger.error(f"Failed to import PyTorch: {e}")
    sys.exit(1)

# ====================================================================
# PYTORCH 2.6+ COMPATIBILITY NOTES
# ====================================================================
# IMPORTANT: PyTorch 2.6+ changed the default value of `weights_only`
# from False to True. This can cause loading errors with models saved
# using older versions or with numpy arrays.
# ====================================================================

def safe_save_model(checkpoint: Dict, model_path: str) -> bool:
    """
    Safely save PyTorch model with compatibility for PyTorch 2.6+.
    
    Args:
        checkpoint: Model checkpoint dictionary
        model_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with protocol version for better compatibility
        torch.save(checkpoint, model_path, pickle_protocol=4)
        logger.debug(f"Model saved successfully to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        return False

def safe_load_model(model_path: str) -> Optional[Dict]:
    """
    Safely load PyTorch model with fallback strategies for PyTorch 2.6+ compatibility.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded checkpoint dictionary or None if failed
    """
    try:
        # Attempt 1: Try with weights_only=False (safe for trusted sources)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        logger.debug("Model loaded with weights_only=False")
        return checkpoint
    except Exception as e:
        logger.warning(f"Loading with weights_only=False failed: {e}")
        
    try:
        # Attempt 2: Try with safe globals for numpy compatibility
        import numpy as np
        try:
            # First check if safe_globals context manager exists (PyTorch 2.1+)
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):  # type: ignore
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.debug("Model loaded with safe globals context manager")
                return checkpoint
        except (AttributeError, ImportError):
            # For older PyTorch versions that don't have safe_globals
            logger.debug("torch.serialization.safe_globals not available, using add_safe_globals")
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])  # type: ignore
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.debug("Model loaded with add_safe_globals")
                return checkpoint
            logger.debug("add_safe_globals not available, trying legacy method")
            # Fall through to the next attempt
            raise ValueError("Neither safe_globals nor add_safe_globals available")
    except Exception as e:
        logger.warning(f"Loading with safe_globals approach failed: {e}")
        
    try:
        # Attempt 3: Fallback to older PyTorch loading method (no weights_only parameter)
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.debug("Model loaded with legacy method")
        return checkpoint
    except Exception as e:
        logger.error(f"All loading methods failed: {e}")
        return None

def setup_safe_globals():
    """
    Setup safe globals for PyTorch serialization to handle numpy compatibility.
    Call this once at module initialization.
    """
    try:
        import numpy as np
        # Add commonly needed numpy functions to safe globals
        safe_globals_list = [
            np.ndarray,
            np.dtype,
        ]
        
        # Check if PyTorch version supports add_safe_globals
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

# ====================================================================
# GPU RESOURCE MANAGEMENT UTILITIES
# ====================================================================

def with_gpu_context(func):
    """
    Decorator to automatically manage GPU resources for functions.
    
    Args:
        func: Function to wrap with GPU context management
        
    Returns:
        Wrapped function with automatic GPU cleanup
    """
    def wrapper(*args, **kwargs):
        gpu_manager = get_gpu_resource_manager()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if gpu_manager.is_cuda_available:
                gpu_manager.cleanup()
    return wrapper

def get_optimal_device() -> torch.device:
    """
    Get the optimal device for PyTorch operations using GPU resource manager.
    
    Returns:
        Optimal PyTorch device (GPU if available, otherwise CPU)
    """
    gpu_manager = get_gpu_resource_manager()
    if gpu_manager.is_cuda_available:
        return gpu_manager.current_device or torch.device("cuda:0")
    return torch.device("cpu")

def check_gpu_compatibility() -> Dict[str, Union[bool, str, int]]:
    """
    Check GPU compatibility and return detailed information.
    
    Returns:
        Dictionary with GPU compatibility information
    """
    gpu_manager = get_gpu_resource_manager()
    
    compatibility_info = {
        'cuda_available': gpu_manager.is_cuda_available,
        'device_count': 0,
        'current_device': 'cpu',
        'memory_total_gb': 0,
        'memory_allocated_gb': 0,
        'tensor_cores_available': False,
        'compute_capability': 'N/A'
    }
    
    if gpu_manager.is_cuda_available:
        try:
            compatibility_info['device_count'] = torch.cuda.device_count()
            compatibility_info['current_device'] = str(gpu_manager.current_device)
            
            # Get memory information
            memory_info = gpu_manager.get_memory_info()
            if isinstance(memory_info['total'], int):
                compatibility_info['memory_total_gb'] = memory_info['total'] / 1024**3
            if isinstance(memory_info['allocated'], int):
                compatibility_info['memory_allocated_gb'] = memory_info['allocated'] / 1024**3
            
            # Get Tensor Core information
            tensor_core_info = gpu_manager.get_tensor_core_info()
            compatibility_info['tensor_cores_available'] = tensor_core_info['has_tensor_cores']
            compatibility_info['compute_capability'] = tensor_core_info['compute_capability'] or 'N/A'
            
        except Exception as e:
            logger.warning(f"Error getting GPU compatibility info: {e}")
    
    return compatibility_info

def monitor_gpu_memory() -> Dict[str, Union[str, float]]:
    """
    Monitor current GPU memory usage.
    
    Returns:
        Dictionary with current memory usage information
    """
    gpu_manager = get_gpu_resource_manager()
    
    if not gpu_manager.is_cuda_available:
        return {'status': 'GPU not available', 'device': 'cpu'}
    
    try:
        memory_info = gpu_manager.get_memory_info()
        if isinstance(memory_info['total'], int) and isinstance(memory_info['allocated'], int):
            total_gb = memory_info['total'] / 1024**3
            allocated_gb = memory_info['allocated'] / 1024**3
            usage_percent = (allocated_gb / total_gb) * 100 if total_gb > 0 else 0
            
            return {
                'device': str(memory_info['device']),
                'total_gb': round(total_gb, 2),
                'allocated_gb': round(allocated_gb, 2),
                'usage_percent': round(usage_percent, 1),
                'status': 'GPU memory monitored successfully'
            }
        else:
            return {
                'device': str(memory_info['device']),
                'status': 'Memory info not available'
            }
    except Exception as e:
        return {
            'status': f'Error monitoring GPU memory: {e}',
            'device': 'unknown'
        }

# ====================================================================
# USAGE EXAMPLES - TRAINING AND USING ALL 4 MODEL VARIANTS
# ====================================================================

def train_all_model_variants(df_input: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Example function demonstrating how to train all 4 model variants.
    
    Args:
        df_input: Input DataFrame with market data
        
    Returns:
        Dictionary mapping model names to their (model_path, variant_info) tuples.
    """
    results = {}
    
    # Check GPU compatibility before training
    gpu_info = check_gpu_compatibility()
    logger.gpu(f"GPU Compatibility: {gpu_info}")
    
    # Monitor initial GPU memory
    initial_memory = monitor_gpu_memory()
    logger.memory(f"Initial GPU Memory: {initial_memory}")
    
    # 1. Standard LSTM
    logger.info("Training Standard LSTM...")
    model, model_path = train_cnn_lstm_attention_model(
        df_input, 
        use_cnn=False, 
        use_attention=False, 
        model_filename="lstm_standard.pth"
    )
    if model:
        results["LSTM"] = (model_path, "Standard LSTM with 3 sequential layers")
    
    # 2. LSTM with Attention
    logger.info("Training LSTM-Attention...")
    model, model_path = train_cnn_lstm_attention_model(
        df_input, 
        use_cnn=False, 
        use_attention=True, 
        model_filename="lstm_attention.pth"
    )
    if model:
        results["LSTM-Attention"] = (model_path, "LSTM with Multi-Head Attention mechanism")
    
    # 3. CNN-LSTM
    logger.info("Training CNN-LSTM...")
    model, model_path = train_cnn_lstm_attention_model(
        df_input, 
        use_cnn=True, 
        use_attention=False, 
        model_filename="cnn_lstm.pth"
    )
    if model:
        results["CNN-LSTM"] = (model_path, "CNN feature extraction + LSTM sequence modeling")
    
    # 4. CNN-LSTM-Attention (Full Model)
    logger.info("Training CNN-LSTM-Attention...")
    model, model_path = train_cnn_lstm_attention_model(
        df_input, 
        use_cnn=True, 
        use_attention=True, 
        model_filename="cnn_lstm_attention_full.pth"
    )
    if model:
        results["CNN-LSTM-Attention"] = (
            model_path, 
            "Full hybrid architecture with CNN + LSTM + Attention"
        )
    
    # Monitor final GPU memory
    final_memory = monitor_gpu_memory()
    logger.memory(f"Final GPU Memory: {final_memory}")
    
    return results

def load_and_predict_all_variants(
    df_input: pd.DataFrame, 
    model_paths: Dict[str, str]
) -> Dict[str, str]:
    """
    Example function demonstrating how to load and get predictions from all variants.
    
    Args:
        df_input: Market data for prediction
        model_paths: Dictionary mapping model names to their file paths
        
    Returns:
        Dictionary mapping model names to their signal predictions
    """
    predictions = {}
    
    # Check GPU compatibility before prediction
    gpu_info = check_gpu_compatibility()
    logger.gpu(f"GPU Compatibility for Prediction: {gpu_info}")
    
    for model_name, model_path in model_paths.items():
        try:
            # Monitor GPU memory before loading each model
            memory_before = monitor_gpu_memory()
            logger.memory(f"Memory before loading {model_name}: {memory_before}")
            
            # Load model
            loaded_data = load_cnn_lstm_attention_model(model_path)
            if loaded_data:
                model, model_config, data_info, optimization_results = loaded_data
                
                # Get prediction
                signal = get_latest_cnn_lstm_attention_signal(
                    df_input, model, model_config, data_info, optimization_results
                )
                
                predictions[model_name] = signal
                logger.info(f"{model_name}: {signal}")
                
                # Monitor GPU memory after prediction
                memory_after = monitor_gpu_memory()
                logger.memory(f"Memory after {model_name} prediction: {memory_after}")
            else:
                logger.error(f"Failed to load {model_name} from {model_path}")
                predictions[model_name] = SIGNAL_NEUTRAL
                
        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
            predictions[model_name] = SIGNAL_NEUTRAL
    
    return predictions

def preprocess_cnn_lstm_data(
    df_input: pd.DataFrame,
    look_back: int = WINDOW_SIZE_LSTM,
    output_mode: str = 'classification',
    scaler_type: str = 'minmax'
) -> Tuple[np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler], List[str]]:
    """
    Preprocesses the data for the CNN-LSTM model by generating features,
    creating a target variable, scaling the data, and creating sequences.

    Args:
        df_input: The input DataFrame.
        look_back: The look-back period.
        output_mode: The output mode for the target variable.
        scaler_type: The type of scaler to use.

    Returns:
        A tuple containing the sequences of features (X), the target (y),
        the scaler used, and the list of features.
    """
    if 'Date' in df_input.columns:
        df_input = df_input.set_index('Date')

    if not all(feature in df_input.columns for feature in MODEL_FEATURES):
        df_with_features = generate_indicator_features(df_input.copy())
    else:
        df_with_features = df_input.copy()

    df_with_target = create_target_variable(
        df_with_features, output_mode, look_back
    )

    features_to_scale = [
        f for f in MODEL_FEATURES if f in df_with_target.columns
    ]

    df_scaled, scaler = scale_data(
        df_with_target, features_to_scale, scaler_type
    )

    X, y = create_sequences(df_scaled, features_to_scale, look_back)

    return X, y, scaler, features_to_scale

def _split_train_test_data(
    X: np.ndarray, 
    y: np.ndarray, 
    train_ratio: Optional[float] = None, 
    validation_ratio: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets with data validation.
    
    Args:
        X: Input features array
        y: Target values array
        train_ratio: Training set ratio
        validation_ratio: Validation set ratio
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        
    Raises:
        ValueError: If data is invalid or insufficient
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays")
    
    n_samples = len(X)
    if n_samples != len(y):
        raise ValueError(f"X and y length mismatch: X={n_samples}, y={len(y)}")
    
    if n_samples < 10:
        raise ValueError(f"Insufficient data: {n_samples} samples, need at least 10")
    
    # Use safe defaults if ratios are None or invalid
    if train_ratio is None:
        train_ratio = 0.7  # Safe default
    if validation_ratio is None:
        validation_ratio = 0.2  # Safe default
    
    # Safety check: if config values are invalid, use safe defaults
    if (not (0 < train_ratio < 1) or not (0 < validation_ratio < 1) or 
            (train_ratio + validation_ratio >= 1)):
        logger.warning(
            f"Invalid ratios detected: train={train_ratio}, val={validation_ratio}. "
            "Using safe defaults."
        )
        train_ratio = 0.7
        validation_ratio = 0.2
    
    # Ensure minimum test set size (at least 1 sample)
    test_ratio = 1 - train_ratio - validation_ratio
    min_test_samples = max(1, int(n_samples * 0.05))  # At least 5% or 1 sample for test
    
    # Adjust ratios if needed to ensure minimum test set
    if test_ratio * n_samples < min_test_samples:
        # Recalculate ratios to ensure minimum test set
        remaining_ratio = 1 - (min_test_samples / n_samples)
        adjusted_train_ratio = train_ratio * remaining_ratio / (train_ratio + validation_ratio)
        adjusted_val_ratio = validation_ratio * remaining_ratio / (train_ratio + validation_ratio)
        
        train_end = max(int(n_samples * adjusted_train_ratio), 3)
        val_end = max(train_end + int(n_samples * adjusted_val_ratio), train_end + 2)
        val_end = min(val_end, n_samples - min_test_samples)
        
        logger.model(
            f"Adjusted ratios for minimum test set - Train: {train_end}, "
            f"Val: {val_end - train_end}, Test: {n_samples - val_end}"
        )
    else:
        # Original logic
        train_end = max(int(n_samples * train_ratio), 3)
        val_end = max(int(n_samples * (train_ratio + validation_ratio)), train_end + 2)
        val_end = min(val_end, n_samples - 1)
        
        logger.model(
            f"Data split - Train: {train_end}, Val: {val_end - train_end}, "
            f"Test: {n_samples - val_end}"
        )
    
    return (X[:train_end], X[train_end:val_end], X[val_end:],
            y[:train_end], y[train_end:val_end], y[val_end:])

# ====================================================================
# CORE API: Primary Functions for Model Training, Loading & Prediction
# ====================================================================

def create_cnn_lstm_attention_model(
    input_size: int, 
    use_attention: bool = True, 
    use_cnn: bool = True, 
    look_back: int = WINDOW_SIZE_LSTM, 
    output_mode: str = 'classification', 
    **kwargs
) -> Union[LSTMModel, LSTMAttentionModel, CNNLSTMAttentionModel]:
    """
    Universal model factory for creating all 4 LSTM variants.
    
    Creates models based on feature flags:
    - LSTM: use_cnn=False, use_attention=False
    - LSTM-Attention: use_cnn=False, use_attention=True  
    - CNN-LSTM: use_cnn=True, use_attention=False
    - CNN-LSTM-Attention: use_cnn=True, use_attention=True
    
    Args:
        input_size: Number of input features
        use_attention: Enable multi-head attention mechanism
        use_cnn: Enable CNN feature extraction layers
        look_back: Sequence length for time series
        output_mode: 'classification' or 'regression'
        **kwargs: Additional model parameters (num_heads, dropout, etc.)
        
    Returns:
        Configured neural network model
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"input_size must be positive integer, got {input_size}")
    if not isinstance(look_back, int) or look_back <= 0:
        raise ValueError(f"look_back must be positive integer, got {look_back}")
    if output_mode not in ['classification', 'regression', 'classification_advanced']:
        raise ValueError(
            f"output_mode must be 'classification', 'regression', or "
            f"'classification_advanced', got {output_mode}"
        )
    
    # Model selection logic
    if use_cnn:
        logger.model(
            f"Creating CNN-LSTM{'-Attention' if use_attention else ''} "
            f"model with {output_mode} mode"
        )
        # Validate output_mode for CNNLSTMAttentionModel
        if output_mode not in ['classification', 'regression', 'classification_advanced']:
            raise ValueError(f"Invalid output_mode: {output_mode}")
        
        # Map classification_advanced to classification for model creation
        model_output_mode = 'classification' if output_mode == 'classification_advanced' else output_mode
        validated_output_mode = model_output_mode  # type: ignore
        
        model = CNNLSTMAttentionModel(
            input_size=input_size,
            look_back=look_back,
            output_mode=validated_output_mode,  # type: ignore
            use_attention=use_attention,
            **kwargs
        )
        model_type = f"CNN-LSTM{'-Attention' if use_attention else ''}"
        logger.model(f"Created {model_type} model with {output_mode} mode")
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

def train_cnn_lstm_attention_model(
    df_input: pd.DataFrame,
    save_model: bool = True,
    epochs: int = DEFAULT_EPOCHS,
    use_early_stopping: bool = True,
    use_attention: bool = True,
    use_cnn: bool = True,
    look_back: int = WINDOW_SIZE_LSTM,
    output_mode: str = 'classification',
    model_filename: Optional[str] = None
) -> Tuple[Optional[nn.Module], str]:
    """
    Train a CNN-LSTM-Attention model with comprehensive error handling and optimization.
    
    Args:
        df_input: Input DataFrame with market data
        save_model: Whether to save the trained model
        epochs: Number of training epochs
        use_early_stopping: Whether to use early stopping
        use_attention: Whether to use attention mechanism
        use_cnn: Whether to use CNN layers
        look_back: Sequence length for time series
        output_mode: Output mode ('classification' or 'regression')
        model_filename: Custom filename for model saving
        
    Returns:
        Tuple of (trained_model, model_path)
    """
    # Use GPU resource manager for device management
    gpu_manager = get_gpu_resource_manager()
    
    # Initialize GPU if available, otherwise use CPU
    if gpu_manager.is_cuda_available:
        device = gpu_manager.current_device or torch.device("cuda:0")
        use_amp = True
        logger.info(f"Training on GPU: {device}")
    else:
        device = torch.device("cpu")
        use_amp = False
        logger.info("Training on CPU")
    
    # Get appropriate config based on device
    config = _create_training_config(device)

    # Preprocess data
    X, y, scaler, features = preprocess_cnn_lstm_data(
        df_input, look_back, output_mode
    )

    if X.size == 0:
        logger.error("Preprocessing returned no data.")
        return None, ""

    # Split data
    (X_train, y_train, X_val, y_val, X_test, y_test) = _split_train_test_data(X, y)

    # Create data loaders
    train_loader, val_loader, test_loader = _create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        device, use_cnn, use_attention, look_back
    )

    # Create model
    model = create_cnn_lstm_attention_model(
        input_size=X_train.shape[2],
        use_attention=use_attention,
        use_cnn=use_cnn,
        look_back=look_back,
        output_mode=output_mode,
        **config
    ).to(device)

    # Loss and optimizer
    if output_mode == 'classification':
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training using ModelTrainer
    trainer = ModelTrainer(model, optimizer, criterion, device, use_amp)
    trainer.train(train_loader, val_loader, epochs, use_early_stopping)

    # Final evaluation
    accuracy = _evaluate_model(model, test_loader, device, use_amp, output_mode)

    # Save model
    model_path = ""
    if save_model:
        if model_filename is None:
            model_name = "cnn_lstm_attention_model.pth"
            if use_cnn and not use_attention:
                model_name = "cnn_lstm_model.pth"
            elif not use_cnn and use_attention:
                model_name = "lstm_attention_model.pth"
            elif not use_cnn and not use_attention:
                model_name = "lstm_model.pth"
            model_filename = model_name

        model_type = 'cnn_lstm' if use_cnn else 'lstm_attention' if use_attention else 'lstm'
        model_path = _save_model_checkpoint(
            model, config, features, look_back, output_mode, scaler,
            accuracy, model_type, model_filename, use_cnn, use_attention
        )

    # Cleanup GPU resources
    if gpu_manager.is_cuda_available:
        gpu_manager.cleanup()

    return model, model_path

def load_cnn_lstm_attention_model(
    model_path: Union[str, Path]
) -> Optional[Tuple[nn.Module, Dict, Dict, Dict]]:
    """
    Load a trained CNN-LSTM-Attention model from a file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (model, model_config, data_info, optimization_results) or None if failed
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        # Use GPU resource manager for device management
        gpu_manager = get_gpu_resource_manager()
        if gpu_manager.is_cuda_available:
            device = gpu_manager.current_device or torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        
        checkpoint = safe_load_model(str(model_path))
        
        if checkpoint is None:
            logger.error("Failed to load checkpoint")
            return None

        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'model_config', 'data_info']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            logger.error(f"Missing keys in checkpoint: {missing_keys}")
            return None
        
        model_config = checkpoint.get('model_config', {})
        data_info = checkpoint.get('data_info', {})
        optimization_results = checkpoint.get('optimization_results', {})
        
        if not model_config or not data_info:
            logger.error(
                f"Model checkpoint at {model_path} is missing required "
                "'model_config' or 'data_info' keys."
            )
            return None

        # Recreate model from config - fix model selection logic
        use_cnn = model_config.get('use_cnn', True)
        use_attention = model_config.get('use_attention', True)
        
        # More robustly get hidden size, compatible with different saved configs.
        # This handles cases where the config might have 'hidden_size' or 'lstm_hidden'.
        hidden_size = model_config.get('hidden_size') or model_config.get('lstm_hidden')

        if use_cnn:
            # Use CNNLSTMAttentionModel for both CNN-LSTM and CNN-LSTM-Attention
            model = CNNLSTMAttentionModel(
                input_size=model_config['input_size'],
                look_back=model_config['look_back'],
                output_mode=model_config['output_mode'],
                use_attention=use_attention,
                cnn_features=model_config.get('cnn_features', 64),
                lstm_hidden=model_config.get('lstm_hidden', 32),
                num_classes=model_config.get(
                    'num_classes', 
                    3 if model_config['output_mode'] in ['classification', 'classification_advanced'] else 1
                ),
                num_heads=model_config.get('attention_heads', 4),
                dropout=model_config.get('dropout', 0.3)
            )
        elif use_attention:
            # Use LSTMAttentionModel for LSTM-Attention
            model = LSTMAttentionModel(
                input_size=model_config['input_size'],
                num_heads=model_config.get('attention_heads', 4),
                dropout=model_config.get('dropout', 0.3),
                hidden_size=hidden_size or 32,
                num_layers=model_config.get('num_layers', 3)
            )
        else:
            # Use LSTMModel for basic LSTM
            model = LSTMModel(
                input_size=model_config['input_size'],
                dropout=model_config.get('dropout', 0.3),
                hidden_size=hidden_size or 32,
                num_layers=model_config.get('num_layers', 3)
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.success(f"Successfully loaded model from {model_path} on {device}")
        return model, model_config, data_info, optimization_results

    except Exception as e:
        logger.error(f"Failed to load CNN-LSTM model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

@with_gpu_context
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
        df_input: Input DataFrame with market data
        model: Loaded trained model
        model_config: Model configuration
        data_info: Data information including scaler and features
        optimization_results: Optimization results including thresholds
        
    Returns:
        Trading signal (LONG, SHORT, or NEUTRAL)
    """
    try:
        # Validate inputs
        is_valid, error_msg, device, look_back, scaler, feature_names = _validate_prediction_inputs(
            df_input, model, model_config, data_info
        )
        if not is_valid:
            logger.warning(error_msg)
            return SIGNAL_NEUTRAL

        # Additional null check for scaler
        if scaler is None:
            logger.error("Scaler is None after validation")
            return SIGNAL_NEUTRAL

        # Prepare data for prediction
        success, error_msg, sequence_tensor = _prepare_prediction_data(
            df_input, look_back, feature_names, scaler
        )
        if not success:
            logger.error(error_msg)
            return SIGNAL_NEUTRAL

        # Additional null check for sequence_tensor
        if sequence_tensor is None:
            logger.error("Sequence tensor is None after preparation")
            return SIGNAL_NEUTRAL

        # Move tensor to device
        sequence_tensor = sequence_tensor.to(device)

        # Predict
        with torch.no_grad():
            try:
                output = model(sequence_tensor)
            except Exception as model_error:
                logger.error(f"Model prediction failed: {model_error}")
                return SIGNAL_NEUTRAL
            
        # Interpret output
        output_mode = model_config.get('output_mode', 'classification')
        if output_mode in ['classification', 'classification_advanced']:
            return _process_classification_output(output, output_mode, optimization_results)
        elif output_mode == 'regression':
            return _process_regression_output(output)
        else:
            logger.warning(f"Unsupported output_mode: {output_mode}")
            return SIGNAL_NEUTRAL

    except Exception as e:
        logger.error(f"Error during CNN-LSTM signal prediction: {e}")
        logger.debug(f"Error type: {type(e).__name__}")
        logger.debug(f"Error args: {e.args}")
        return SIGNAL_NEUTRAL

def demonstrate_gpu_resource_manager_features():
    """
    Demonstrate all GPU resource manager features.
    
    This function shows how to use the GPU resource manager for:
    - GPU compatibility checking
    - Memory monitoring
    - Device management
    - Resource cleanup
    """
    logger.gpu("="*60)
    logger.gpu("GPU RESOURCE MANAGER DEMONSTRATION")
    logger.gpu("="*60)
    
    # 1. Get GPU resource manager instance
    gpu_manager = get_gpu_resource_manager()
    
    # 2. Check GPU compatibility
    compatibility = check_gpu_compatibility()
    logger.gpu(f"GPU Compatibility Check:")
    for key, value in compatibility.items():
        logger.gpu(f"  {key}: {value}")
    
    # 3. Monitor GPU memory
    memory_info = monitor_gpu_memory()
    logger.memory(f"GPU Memory Monitor:")
    for key, value in memory_info.items():
        logger.memory(f"  {key}: {value}")
    
    # 4. Get optimal device
    optimal_device = get_optimal_device()
    logger.gpu(f"Optimal Device: {optimal_device}")
    
    # 5. Use GPU context manager
    with gpu_manager.gpu_scope(device_id=0) as device:
        if device:
            logger.gpu(f"Using GPU device: {device}")
            # Perform some GPU operations here
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.sum(test_tensor)
            logger.memory(f"GPU test operation result: {result.item():.4f}")
        else:
            logger.info("GPU not available, using CPU")
    
    # 6. Check Tensor Core information
    tensor_core_info = gpu_manager.get_tensor_core_info()
    logger.gpu(f"Tensor Core Information:")
    for key, value in tensor_core_info.items():
        logger.gpu(f"  {key}: {value}")
    
    logger.gpu("="*60)
    logger.gpu("GPU RESOURCE MANAGER DEMONSTRATION COMPLETE")
    logger.gpu("="*60)

# ====================================================================
# GPU RESOURCE MANAGER INTEGRATION DOCUMENTATION
# ====================================================================
"""
GPU RESOURCE MANAGER INTEGRATION GUIDE

This module now includes comprehensive GPU resource management through the
_gpu_resource_manager.py utility. Here's how to use the new features:

1. AUTOMATIC GPU INITIALIZATION
   - The module automatically initializes GPU resources on import
   - GPU compatibility is checked and logged
   - Memory information is displayed
   - Tensor Core capabilities are detected

2. DEVICE MANAGEMENT
   - Use get_optimal_device() to get the best available device
   - GPU resource manager handles device selection automatically
   - Fallback to CPU if GPU is not available

3. MEMORY MONITORING
   - monitor_gpu_memory() provides real-time memory usage
   - check_gpu_compatibility() gives detailed GPU information
   - Automatic memory cleanup after operations

4. CONTEXT MANAGEMENT
   - Use @with_gpu_context decorator for automatic cleanup
   - Use gpu_manager.gpu_scope() for manual context management
   - Resources are automatically cleaned up on function exit

5. TRAINING WITH GPU RESOURCE MANAGER
   - train_cnn_lstm_attention_model() now uses GPU resource manager
   - Automatic device selection and memory management
   - Proper cleanup after training

6. PREDICTION WITH GPU RESOURCE MANAGER
   - get_latest_cnn_lstm_attention_signal() uses GPU context
   - Automatic memory cleanup after predictions
   - Device management handled automatically

7. COMPATIBILITY FEATURES
   - PyTorch 2.6+ compatibility with safe loading
   - CUDA DLL loading and environment setup
   - Automatic fallback to CPU if GPU issues occur

EXAMPLE USAGE:

# Check GPU compatibility
gpu_info = check_gpu_compatibility()
print(f"GPU Available: {gpu_info['cuda_available']}")

# Monitor memory
memory = monitor_gpu_memory()
print(f"Memory Usage: {memory['usage_percent']}%")

# Train with automatic GPU management
model, path = train_cnn_lstm_attention_model(df_data)

# Predict with automatic cleanup
signal = get_latest_cnn_lstm_attention_signal(df_data, model, config, info, results)

# Demonstrate all features
demonstrate_gpu_resource_manager_features()

ADVANTAGES:

1. AUTOMATIC RESOURCE MANAGEMENT
   - No manual GPU memory cleanup required
   - Automatic device selection
   - Proper error handling and fallbacks

2. ENHANCED COMPATIBILITY
   - Works with various CUDA versions
   - Handles PyTorch version differences
   - Robust error recovery

3. BETTER PERFORMANCE
   - Optimized memory usage
   - Tensor Core detection and utilization
   - Efficient batch size calculation

4. IMPROVED MONITORING
   - Real-time memory monitoring
   - GPU capability detection
   - Performance optimization suggestions

5. SAFETY FEATURES
   - Automatic cleanup prevents memory leaks
   - Graceful fallback to CPU
   - Comprehensive error handling
"""