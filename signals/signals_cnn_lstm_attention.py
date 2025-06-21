# ====================================================================
# COMPLETE API USAGE GUIDE FOR ALL 4 MODEL VARIANTS
# ====================================================================

"""
COMPREHENSIVE GUIDE: Training and Using All 4 LSTM Model Variants

This guide demonstrates how to use the unified API to train, load, and get signals 
from all 4 model variants using only the 3 core functions:

1. train_cnn_lstm_attention_model()  - Universal training function
2. load_cnn_lstm_attention_model()   - Universal loading function  
3. get_latest_cnn_lstm_attention_signal() - Universal prediction function

MODEL VARIANTS SUPPORTED:
┌─────────────────────┬─────────────┬──────────────┬─────────────────┐
│ Model Type          │ use_cnn     │ use_attention │ Description     │
├─────────────────────┼─────────────┼──────────────┼─────────────────┤
│ LSTM                │ False       │ False        │ Basic LSTM      │
│ LSTM-Attention      │ False       │ True         │ LSTM + Attention│
│ CNN-LSTM            │ True        │ False        │ CNN + LSTM      │
│ CNN-LSTM-Attention  │ True        │ True         │ Full Hybrid     │
└─────────────────────┴─────────────┴──────────────┴─────────────────┘

USAGE EXAMPLES:

# 1. Train a standard LSTM model
model, path = train_cnn_lstm_attention_model(
    df_data, 
    use_cnn=False, 
    use_attention=False,
    model_filename="lstm_basic.pth"
)

# 2. Train LSTM with Attention
model, path = train_cnn_lstm_attention_model(
    df_data, 
    use_cnn=False, 
    use_attention=True,
    model_filename="lstm_attention.pth"
)

# 3. Train CNN-LSTM hybrid
model, path = train_cnn_lstm_attention_model(
    df_data, 
    use_cnn=True, 
    use_attention=False,
    model_filename="cnn_lstm.pth"
)

# 4. Train full CNN-LSTM-Attention model
model, path = train_cnn_lstm_attention_model(
    df_data, 
    use_cnn=True, 
    use_attention=True,
    model_filename="cnn_lstm_attention.pth"
)

# Load any trained model (works for all variants)
loaded_data = load_cnn_lstm_attention_model("path/to/model.pth")
if loaded_data:
    model, config, data_info, results = loaded_data

# Get prediction from any loaded model (works for all variants)
signal = get_latest_cnn_lstm_attention_signal(
    df_market_data, model, config, data_info, results
)

AUTOMATED BATCH TRAINING:

# Train all 4 variants automatically
all_models = train_all_model_variants(df_data)
print(all_models)
# Output: {
#   'LSTM': ('path/lstm_standard.pth', 'Standard LSTM...'),
#   'LSTM-Attention': ('path/lstm_attention.pth', 'LSTM with Attention...'),
#   'CNN-LSTM': ('path/cnn_lstm.pth', 'CNN + LSTM...'),
#   'CNN-LSTM-Attention': ('path/cnn_lstm_attention_full.pth', 'Full hybrid...')
# }

# Get predictions from all variants
model_paths = {name: path for name, (path, _) in all_models.items()}
all_signals = load_and_predict_all_variants(df_new_data, model_paths)
print(all_signals)
# Output: {
#   'LSTM': 'LONG',
#   'LSTM-Attention': 'SHORT', 
#   'CNN-LSTM': 'NEUTRAL',
#   'CNN-LSTM-Attention': 'LONG'
# }

ADVANCED CONFIGURATION:

# Custom model parameters for each variant
lstm_model, _ = train_cnn_lstm_attention_model(
    df_data,
    use_cnn=False,
    use_attention=False,
    hidden_size=128,        # LSTM-specific parameter
    num_layers=4,           # LSTM-specific parameter
    dropout=0.2
)

attention_model, _ = train_cnn_lstm_attention_model(
    df_data,
    use_cnn=False,
    use_attention=True,
    num_heads=8,            # Attention-specific parameter
    hidden_size=64,
    dropout=0.3
)

cnn_lstm_model, _ = train_cnn_lstm_attention_model(
    df_data,
    use_cnn=True,
    use_attention=True,
    cnn_features=128,       # CNN-specific parameter
    lstm_hidden=64,         # CNN-LSTM-specific parameter
    num_heads=4,            # Attention-specific parameter
    dropout=0.25
)

BENEFITS OF UNIFIED API:
✓ Single set of functions for all model types
✓ Consistent interface across variants
✓ Automatic model architecture detection
✓ Unified data preprocessing pipeline
✓ Compatible model saving/loading format
✓ Consistent signal generation interface
✓ GPU/CPU automatic optimization for all variants
✓ Mixed precision training support
✓ Advanced early stopping and scheduling
"""

import logging
import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.update({
    'KMP_DUPLICATE_LIB_OK': 'True',
    'OMP_NUM_THREADS': '1',
    'CUDA_LAUNCH_BLOCKING': '1',
    'TORCH_USE_CUDA_DSA': '1'
})

from components.config import (
    CPU_MODEL_CONFIG, 
    DEFAULT_EPOCHS,
    GPU_MODEL_CONFIG, 
    MODEL_FEATURES, MODELS_DIR,
    NEUTRAL_ZONE_LSTM, 
    SIGNAL_LONG, SIGNAL_NEUTRAL, SIGNAL_SHORT,
    TARGET_THRESHOLD_LSTM, TRAIN_TEST_SPLIT, 
    VALIDATION_SPLIT, 
    WINDOW_SIZE_LSTM
)

from components._generate_indicator_features import generate_indicator_features
from signals._components.LSTM__class__GridSearchThresholdOptimizer import GridSearchThresholdOptimizer
from signals._components.LSTM__class__Models import CNNLSTMAttentionModel, LSTMModel, LSTMAttentionModel
from signals._components.LSTM__function__create_balanced_target import create_balanced_target
from signals._components.LSTM__function__get_optimal_batch_size import get_optimal_batch_size
from utilities._gpu_resource_manager import get_gpu_resource_manager, get_tensor_core_info

from utilities._logger import setup_logging
logger = setup_logging(module_name="signals_cnn_lstm_attention", log_level=logging.DEBUG)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader, TensorDataset
    logger.success(f"PyTorch {torch.__version__} loaded successfully")

    if torch.cuda.is_available():
        try:
            torch.ones(1).cuda()
            cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else "Unknown"
            logger.gpu(f"CUDA {cuda_version} available with {torch.cuda.device_count()} device(s)")
        except Exception as cuda_error:
            logger.warning(f"CUDA available but not functional: {cuda_error}")
            torch.cuda.is_available = lambda: False
    else:
        logger.info("CUDA not available, using CPU mode")
except ImportError as e:
    logger.error(f"Failed to import PyTorch: {e}")
    sys.exit(1)

# ====================================================================
# PYTORCH 2.6+ COMPATIBILITY NOTES
# ====================================================================
"""
IMPORTANT: PyTorch 2.6+ changed the default value of `weights_only` from False to True.
This can cause loading errors with models saved using older versions or with numpy arrays.

Common error: "WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray._reconstruct"

SOLUTIONS:
1. Use safe_load_model() function (recommended)
2. Add safe globals before loading
3. Use weights_only=False for trusted sources
4. Retrain models with current PyTorch version

USAGE EXAMPLE:
    # Load model safely
    checkpoint = safe_load_model("path/to/model.pth")
    if checkpoint is None:
        logger.error("Failed to load model")
        return None
"""

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
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.debug("Model loaded with safe globals context manager")
                return checkpoint
        except (AttributeError, ImportError):
            # For older PyTorch versions that don't have safe_globals
            logger.debug("torch.serialization.safe_globals not available, using add_safe_globals")
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.debug("Model loaded with add_safe_globals")
                return checkpoint
            else:
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
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.core.multiarray.scalar,
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
# USAGE EXAMPLES - TRAINING AND USING ALL 4 MODEL VARIANTS
# ====================================================================

def train_all_model_variants(df_input: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Example function demonstrating how to train all 4 model variants.
    
    Returns a dictionary mapping model names to their (model_path, variant_info) tuples.
    """
    results = {}
    
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
        results["CNN-LSTM-Attention"] = (model_path, "Full hybrid architecture with CNN + LSTM + Attention")
    
    return results

def load_and_predict_all_variants(df_input: pd.DataFrame, model_paths: Dict[str, str]) -> Dict[str, str]:
    """
    Example function demonstrating how to load and get predictions from all variants.
    
    Args:
        df_input: Market data for prediction
        model_paths: Dictionary mapping model names to their file paths
        
    Returns:
        Dictionary mapping model names to their signal predictions
    """
    predictions = {}
    
    for model_name, model_path in model_paths.items():
        try:
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
            else:
                logger.error(f"Failed to load {model_name} from {model_path}")
                predictions[model_name] = SIGNAL_NEUTRAL
                
        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
            predictions[model_name] = SIGNAL_NEUTRAL
    
    return predictions

def preprocess_cnn_lstm_data(df_input: pd.DataFrame, 
                             look_back: int = WINDOW_SIZE_LSTM, 
                             output_mode: str = 'classification', 
                             scaler_type: str = 'minmax') -> Tuple[np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler], List[str]]:
    """Preprocess data for CNN-LSTM model with sliding window approach."""
    logger.model(f"Starting CNN-LSTM preprocessing: {df_input.shape} rows, lookback={look_back}, mode={output_mode}")
    
    if df_input.empty or len(df_input) < look_back + 10:
        logger.error(f"Insufficient data: {len(df_input)} rows, need at least {look_back + 10}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
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
    
    initial_len = len(df)
    df.dropna(inplace=True)
    if len(df) < look_back + 1:
        logger.error(f"Insufficient data after cleanup: {len(df)} rows (dropped {initial_len - len(df)} NaN)")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    available_features = [col for col in MODEL_FEATURES if col in df.columns]
    if not available_features:
        logger.error(f"No valid features found from {MODEL_FEATURES}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
    features = df[available_features].values
    
    if np.isnan(features).any() or np.isinf(features).any():
        logger.warning("Cleaning invalid values in features")
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features)
    except Exception as e:
        logger.error(f"Feature scaling failed: {e}")
        return np.array([]), np.array([]), MinMaxScaler(), []
    
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
    
    logger.model(f"Preprocessing complete: {len(X_sequences)} sequences, shape {X_sequences.shape}")
    if output_mode == 'classification':
        unique, counts = np.unique(y_targets, return_counts=True)
        logger.model(f"Target distribution: {dict(zip(unique, counts))}")
    else:
        logger.model(f"Target range: [{np.min(y_targets):.4f}, {np.max(y_targets):.4f}]")
    
    return X_sequences, y_targets, scaler, available_features

def _get_safe_split_ratios() -> Tuple[float, float]:
    """
    Get safe train/validation split ratios from config with fallback to defaults.
    
    Returns:
        Tuple of (train_ratio, validation_ratio) that are guaranteed to be valid
    """
    try:
        # Try to get values from config
        train_ratio = float(TRAIN_TEST_SPLIT) if hasattr(TRAIN_TEST_SPLIT, '__float__') else 0.7
        val_ratio = float(VALIDATION_SPLIT) if hasattr(VALIDATION_SPLIT, '__float__') else 0.2
        
        # Validate ratios
        if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or (train_ratio + val_ratio >= 1):
            logger.warning(f"Invalid config ratios: train={train_ratio}, val={val_ratio}. Using safe defaults.")
            return 0.7, 0.2
        
        return train_ratio, val_ratio
        
    except Exception as e:
        logger.warning(f"Error reading config ratios: {e}. Using safe defaults.")
        return 0.7, 0.2

def _split_train_test_data(X: np.ndarray, y: np.ndarray, train_ratio: float = None, validation_ratio: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/validation/test sets with data validation."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays")
    
    n_samples = len(X)
    if n_samples != len(y):
        raise ValueError(f"X and y length mismatch: X={n_samples}, y={len(y)}")
    
    if n_samples < 10:
        raise ValueError(f"Insufficient data: {n_samples} samples, need at least 10")
    
    # Use safe defaults if ratios are None or invalid
    if train_ratio is None:
        train_ratio = getattr(globals().get('TRAIN_TEST_SPLIT'), '__float__', lambda: 0.7)()
    if validation_ratio is None:
        validation_ratio = getattr(globals().get('VALIDATION_SPLIT'), '__float__', lambda: 0.2)()
    
    # Safety check: if config values are invalid, use safe defaults
    if not (0 < train_ratio < 1) or not (0 < validation_ratio < 1) or (train_ratio + validation_ratio >= 1):
        logger.warning(f"Invalid ratios detected: train={train_ratio}, val={validation_ratio}. Using safe defaults.")
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
        
        logger.model(f"Adjusted ratios for minimum test set - Train: {train_end}, Val: {val_end - train_end}, Test: {n_samples - val_end}")
    else:
        # Original logic
        train_end = max(int(n_samples * train_ratio), 3)
        val_end = max(int(n_samples * (train_ratio + validation_ratio)), train_end + 2)
        val_end = min(val_end, n_samples - 1)
        
        logger.model(f"Data split - Train: {train_end}, Val: {val_end - train_end}, Test: {n_samples - val_end}")
    
    return (X[:train_end], X[train_end:val_end], X[val_end:],
            y[:train_end], y[train_end:val_end], y[val_end:])

# ====================================================================
# CORE API: Primary Functions for Model Training, Loading & Prediction
# ====================================================================

def create_cnn_lstm_attention_model(input_size: int, 
                                    use_attention: bool = True, 
                                    use_cnn: bool = True, 
                                    look_back: int = WINDOW_SIZE_LSTM, 
                                    output_mode: str = 'classification', 
                                    **kwargs) -> Union[LSTMModel, LSTMAttentionModel, CNNLSTMAttentionModel]:
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
    if output_mode not in ['classification', 'regression']:
        raise ValueError(f"output_mode must be 'classification' or 'regression', got {output_mode}")
    
    # Model selection logic
    if use_cnn:
        logger.model(f"Creating CNN-LSTM{'-Attention' if use_attention else ''} model with {output_mode} mode")
        # Validate output_mode for CNNLSTMAttentionModel
        if output_mode not in ['classification', 'regression']:
            raise ValueError(f"Invalid output_mode: {output_mode}")
        validated_output_mode: Literal['classification', 'regression'] = output_mode  # type: ignore
        
        model = CNNLSTMAttentionModel(
            input_size=input_size,
            look_back=look_back,
            output_mode=validated_output_mode,
            use_attention=use_attention,
            **kwargs
        )
        model_type = f"CNN-LSTM{'-Attention' if use_attention else ''}"
        logger.model(f"Created {model_type} model with {output_mode} mode")
        return model
    
    elif use_attention:
        logger.model("Creating LSTM model with Multi-Head Attention")
        attention_params = {k: v for k, v in kwargs.items() 
                          if k in ['num_heads', 'dropout', 'hidden_size', 'num_layers']}
        return LSTMAttentionModel(input_size=input_size, **attention_params)
    
    else:
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
                                    attention_heads: int = GPU_MODEL_CONFIG['nhead'],
                                    model_filename: Optional[str] = None,
                                    gradient_accumulation_steps: int = 1) -> Tuple[Optional[nn.Module], str]:
    """Train CNN-LSTM model with advanced optimizations and GPU efficiency."""
    start_time = time.time()
    model, best_model_state, scaler_amp = None, None, None
    threshold_optimizer = GridSearchThresholdOptimizer()
    
    gpu_manager = get_gpu_resource_manager()
    
    device = None
    has_tensor_cores = False  
    tensor_info = {'generation': 'N/A'}  
    try:
        with gpu_manager.gpu_scope() as device:
            use_mixed_precision = False
            if device is None:
                device = torch.device('cpu')
                logger.info("Using CPU for CNN-LSTM training")
            else:
                logger.gpu(f"Using GPU for CNN-LSTM training: {device}")
                use_mixed_precision = (device.type == 'cuda' and 
                                     torch.cuda.get_device_capability(device)[0] >= 7)
                
                if use_mixed_precision:
                    scaler_amp = torch.amp.GradScaler()
                    logger.gpu("Mixed precision training enabled for faster GPU training")
                    
                # Get Tensor Core info early for mixed precision optimization
                try:
                    gpu_manager_for_tensor = get_gpu_resource_manager()
                    tensor_info = gpu_manager_for_tensor.get_tensor_core_info()
                    has_tensor_cores = tensor_info['has_tensor_cores']
                    
                    if has_tensor_cores and use_mixed_precision:
                        logger.gpu(f"Enhanced mixed precision with {tensor_info['generation']} Tensor Cores")
                except Exception as tc_error:
                    logger.warning(f"Error detecting Tensor Cores: {tc_error}")
                    # Fallback to safe defaults
                    has_tensor_cores = False
                    tensor_info = {'generation': 'N/A', 'has_tensor_cores': False}
                
                memory_info = gpu_manager.get_memory_info()
                logger.gpu(f"GPU Memory - Total: {memory_info['total'] // 1024**3}GB, "
                          f"Available: {(memory_info['total'] - memory_info['allocated']) // 1024**2}MB")

            # Enable optimizations for training
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                if has_tensor_cores:
                    logger.gpu(f"Detected {tensor_info['generation']} Tensor Cores - Enhanced performance available")
                    
                    # Enable Tensor Core optimizations
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                    
                    # Enable Flash Attention if available
                    try:
                        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                            torch.backends.cuda.enable_flash_sdp(True)
                            logger.gpu("Flash Attention enabled for Tensor Cores")
                    except Exception as e:
                        logger.debug(f"Flash Attention not available: {e}")
                        
                    logger.gpu(f"Tensor Core optimizations enabled for {tensor_info['generation']} generation")
                else:
                    logger.gpu("No Tensor Cores detected - using standard GPU optimizations")
                    # Use standard optimizations
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # GPU memory pre-warming with Tensor Core optimization
                memory_info = gpu_manager.get_memory_info()
                if memory_info['total'] > 8 * 1024**3:  
                    torch.cuda.empty_cache()
                    
                    # Pre-warm GPU with Tensor Core optimized dimensions
                    warmup_tensor_size = 64 if has_tensor_cores else 60
                    dummy_tensor = torch.randn(1, warmup_tensor_size, 10, device=device)
                    del dummy_tensor
                    torch.cuda.empty_cache()
                    logger.debug(f"GPU pre-warming completed with tensor size: {warmup_tensor_size}")

            if not model_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                model_type = "cnn_lstm_attention" if use_cnn and use_attention else "cnn_lstm" if use_cnn else "lstm_attention" if use_attention else "lstm"
                model_filename = f"{model_type}_{output_mode}_model_{timestamp}.pth"

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / model_filename

            logger.model("Starting data preprocessing...")
            X, y, scaler, feature_names = preprocess_cnn_lstm_data(
                df_input, 
                look_back=look_back, 
                output_mode=output_mode
            )
            
            if len(X) == 0:
                logger.error("Data preprocessing failed - no valid sequences created")
                return None, ""

            # Debug: Check current train/validation split ratios
            safe_train_ratio, safe_val_ratio = _get_safe_split_ratios()
            logger.debug(f"Using safe split ratios - Train: {safe_train_ratio}, Validation: {safe_val_ratio}, Test: {1 - safe_train_ratio - safe_val_ratio}")
            
            X_train, X_val, X_test, y_train, y_val, y_test = _split_train_test_data(X, y, safe_train_ratio, safe_val_ratio)
            
            # Use pinned memory for faster GPU transfers
            X_train_tensor = torch.FloatTensor(X_train).to(device, non_blocking=True)
            X_val_tensor = torch.FloatTensor(X_val).to(device, non_blocking=True)
            X_test_tensor = torch.FloatTensor(X_test).to(device, non_blocking=True)
            
            if output_mode == 'classification':
                y_train_tensor = torch.LongTensor(y_train + 1).to(device, non_blocking=True)
                y_val_tensor = torch.LongTensor(y_val + 1).to(device, non_blocking=True)
                y_test_tensor = torch.LongTensor(y_test + 1).to(device, non_blocking=True)
            else:
                y_train_tensor = torch.FloatTensor(y_train).to(device, non_blocking=True)
                y_val_tensor = torch.FloatTensor(y_val).to(device, non_blocking=True)
                y_test_tensor = torch.FloatTensor(y_test).to(device, non_blocking=True)

            logger.model(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Create model with Tensor Core optimization
            model_kwargs = {
                'input_size': len(feature_names),
                'use_attention': use_attention,
                'use_cnn': use_cnn,
                'look_back': look_back,
                'output_mode': output_mode,
                'num_heads': attention_heads,
                'dropout': GPU_MODEL_CONFIG['dropout'] if device is not None and device.type == 'cuda' else CPU_MODEL_CONFIG['dropout']
            }
            
            # Optimize model dimensions for Tensor Cores if available
            if device is not None and device.type == 'cuda' and has_tensor_cores:
                # For CNN-LSTM models, optimize hidden dimensions
                if use_cnn:
                    cnn_features = model_kwargs.get('cnn_features', 64)
                    lstm_hidden = model_kwargs.get('lstm_hidden', 32)
                    
                    # Round up to nearest multiple of 8 for optimal Tensor Core performance
                    optimized_cnn_features = ((cnn_features + 7) // 8) * 8
                    optimized_lstm_hidden = ((lstm_hidden + 7) // 8) * 8
                    
                    if optimized_cnn_features != cnn_features or optimized_lstm_hidden != lstm_hidden:
                        logger.gpu(f"Optimized CNN-LSTM dimensions for Tensor Cores: "
                                  f"CNN features {cnn_features} -> {optimized_cnn_features}, "
                                  f"LSTM hidden {lstm_hidden} -> {optimized_lstm_hidden}")
                    
                    model_kwargs.update({
                        'cnn_features': optimized_cnn_features,
                        'lstm_hidden': optimized_lstm_hidden
                    })
                else:
                    # For LSTM/LSTM-Attention models, optimize hidden size
                    hidden_size = model_kwargs.get('hidden_size', 32)
                    optimized_hidden_size = ((hidden_size + 7) // 8) * 8
                    
                    if optimized_hidden_size != hidden_size:
                        logger.gpu(f"Optimized LSTM hidden size for Tensor Cores: {hidden_size} -> {optimized_hidden_size}")
                    
                    model_kwargs['hidden_size'] = optimized_hidden_size

            model = create_cnn_lstm_attention_model(**model_kwargs).to(device)

            logger.model(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

            if output_mode == 'classification':
                unique_classes = np.unique(y_train + 1)
                class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train + 1)
                class_weights_tensor = torch.FloatTensor(class_weights).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            else:
                criterion = nn.MSELoss()

            # Enhanced optimizer with improved settings
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=0.001, 
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
            # Advanced scheduler combination
            warmup_steps = min(100, epochs * 10)
            total_steps = epochs * (len(X_train) // get_optimal_batch_size(device, len(feature_names), look_back, 'cnn_lstm' if use_cnn else 'lstm_attention' if use_attention else 'lstm') + 1)
            
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
            scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-7)

            optimal_batch_size = get_optimal_batch_size(
                device, 
                len(feature_names), 
                look_back,
                'cnn_lstm' if use_cnn else 'lstm_attention' if use_attention else 'lstm'
            )

            # Tensor Core optimization for batch size
            if device is not None and device.type == 'cuda' and has_tensor_cores:
                # Optimize batch size for Tensor Cores (prefer multiples of 8)
                original_batch_size = optimal_batch_size
                optimal_batch_size = ((optimal_batch_size + 7) // 8) * 8
                if optimal_batch_size != original_batch_size:
                    logger.gpu(f"Optimized batch size for Tensor Cores: {original_batch_size} -> {optimal_batch_size}")
            
            logger.model(f"Using batch size: {optimal_batch_size} (Tensor Core optimized: {device is not None and device.type == 'cuda' and has_tensor_cores})")

            # Optimized data loaders with better worker settings
            num_workers_train = min(4, os.cpu_count() // 2) if device is not None and hasattr(device, 'type') and device.type == 'cuda' else 0
            num_workers_val = min(2, os.cpu_count() // 4) if device is not None and hasattr(device, 'type') and device.type == 'cuda' else 0
            
            train_loader = DataLoader(
                TensorDataset(X_train_tensor, y_train_tensor),
                batch_size=optimal_batch_size,
                shuffle=True,
                pin_memory=(device is not None and hasattr(device, 'type') and device.type == 'cuda'),
                num_workers=num_workers_train,
                persistent_workers=True if num_workers_train > 0 else False,
                prefetch_factor=2 if num_workers_train > 0 else None
            )
            
            val_loader = DataLoader(
                TensorDataset(X_val_tensor, y_val_tensor),
                batch_size=optimal_batch_size * 2,  
                shuffle=False,
                pin_memory=(device.type == 'cuda'),
                num_workers=num_workers_val,
                persistent_workers=True if num_workers_val > 0 else False,
                prefetch_factor=2 if num_workers_val > 0 else None
            )

            # Enhanced early stopping with multiple metrics
            best_val_loss = float('inf')
            best_val_metric = -float('inf') if output_mode == 'regression' else 0.0
            best_combined_score = float('-inf')
            patience_counter = 0
            warmup_counter = 0
            current_lr = optimizer.param_groups[0]['lr']  
            training_history = {
                'train_loss': [], 'val_loss': [], 'val_metric': [], 
                'learning_rates': [], 'combined_scores': []
            }
            
            # Advanced gradient management
            max_grad_norm = 1.0
            gradient_accumulation_steps = max(1, gradient_accumulation_steps)

            logger.model(f"Starting training for {epochs} epochs with batch size {optimal_batch_size}")
            logger.model(f"Gradient accumulation steps: {gradient_accumulation_steps}")
            if device is not None and device.type == 'cuda' and has_tensor_cores and tensor_info and 'generation' in tensor_info:
                logger.model(f"🚀 Tensor Core acceleration enabled ({tensor_info['generation']}) for optimal performance")

            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                # Memory and performance optimization
                if device is not None and hasattr(device, 'type') and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                model.train()
                train_loss = 0.0
                train_samples = 0
                accumulation_loss = 0.0

                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    
                    # Gradient accumulation for larger effective batch sizes
                    if use_mixed_precision and scaler_amp is not None:
                        with autocast('cuda'):
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y) / gradient_accumulation_steps
                        scaler_amp.scale(loss).backward()
                        accumulation_loss += loss.item()
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y) / gradient_accumulation_steps
                        loss.backward()
                        accumulation_loss += loss.item()
                    
                    # Update weights after accumulation or at last batch
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        if use_mixed_precision and scaler_amp is not None:
                            scaler_amp.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            scaler_amp.step(optimizer)
                            scaler_amp.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()
                        
                        optimizer.zero_grad()
                        
                        # Advanced learning rate scheduling
                        if warmup_counter < warmup_steps:
                            warmup_lr = 0.001 * (warmup_counter + 1) / warmup_steps
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = warmup_lr
                            warmup_counter += 1
                        elif warmup_counter >= warmup_steps:
                            scheduler_cosine.step()
                        
                        train_loss += accumulation_loss * gradient_accumulation_steps
                        accumulation_loss = 0.0
                    
                    train_samples += batch_X.size(0)

                # Enhanced validation with comprehensive metrics
                model.eval()
                val_loss = 0.0
                val_samples = 0
                val_correct = 0
                all_val_outputs = []
                all_val_targets = []

                try:
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            if use_mixed_precision:
                                with autocast('cuda'):
                                    outputs = model(batch_X)
                                    loss = criterion(outputs, batch_y)
                            else:
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                            
                            val_loss += loss.item() * batch_X.size(0)
                            val_samples += batch_X.size(0)
                            
                            all_val_outputs.append(outputs.cpu())
                            all_val_targets.append(batch_y.cpu())
                            
                            if output_mode == 'classification':
                                _, predicted = torch.max(outputs, 1)
                                val_correct += (predicted == batch_y).sum().item()
                except Exception as val_error:
                    logger.error(f"Error during validation: {val_error}")
                    # In case of validation error, use training loss as fallback
                    val_loss = train_loss
                    val_samples = max(1, train_samples)
                    val_correct = 0
                    if not all_val_outputs:
                        # Create empty tensors for fallback
                        all_val_outputs = [torch.tensor([])]
                        all_val_targets = [torch.tensor([])]

                avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
                avg_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
                
                if output_mode == 'classification':
                    val_accuracy = 100.0 * val_correct / val_samples if val_samples > 0 else 0
                    val_metric = val_accuracy
                    metric_name = "Accuracy"
                else:
                    val_metric = -avg_val_loss
                    metric_name = "Neg Loss"

                # Update schedulers
                prev_lr = current_lr
                scheduler_plateau.step(avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log learning rate changes manually
                if current_lr != prev_lr:
                    logger.model(f"Learning rate reduced from {prev_lr:.2e} to {current_lr:.2e} (ReduceLROnPlateau)")
                
                # Enhanced combined scoring
                normalized_loss = 1.0 / (1.0 + avg_val_loss)
                if output_mode == 'classification':
                    normalized_metric = val_metric / 100.0
                else:
                    normalized_metric = max(0, min(1, (val_metric + 10) / 20))  
                
                combined_score = 0.6 * normalized_loss + 0.4 * normalized_metric
                
                # Enhanced logging
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_loss'].append(avg_val_loss)
                training_history['val_metric'].append(val_metric)
                training_history['learning_rates'].append(current_lr)
                training_history['combined_scores'].append(combined_score)

                epoch_time = time.time() - epoch_start_time
                logger.performance(
                    f'Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) - '
                    f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                    f'Val {metric_name}: {val_metric:.2f}{"%" if output_mode == "classification" else ""}, '
                    f'LR: {current_lr:.2e}, Combined: {combined_score:.4f}'
                )

                # Advanced early stopping with multiple criteria
                if use_early_stopping:
                    improved = False
                    improvement_reasons = []
                    
                    if avg_val_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss = avg_val_loss
                        improved = True
                        improvement_reasons.append("loss")
                    
                    if val_metric > best_val_metric + early_stopping_min_delta:
                        best_val_metric = val_metric
                        improved = True
                        improvement_reasons.append("metric")
                    
                    if combined_score > best_combined_score + early_stopping_min_delta:
                        best_combined_score = combined_score
                        improved = True
                        improvement_reasons.append("combined")
                    
                    if improved:
                        patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        logger.debug(f"New best model saved at epoch {epoch+1} "
                                   f"(improvements: {', '.join(improvement_reasons)}, "
                                   f"Combined Score: {combined_score:.4f})")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.model(f"Early stopping triggered at epoch {epoch+1} "
                                   f"(no improvement for {patience_counter} epochs)")
                        if best_model_state:
                            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                        break

            # Comprehensive final evaluation
            if len(X_test) > 0:
                model.eval()
                try:
                    with torch.no_grad():
                        if use_mixed_precision:
                            with autocast('cuda'):
                                test_outputs = model(X_test_tensor)
                        else:
                            test_outputs = model(X_test_tensor)
                        
                        if output_mode == 'classification':
                            test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
                            test_predictions = np.argmax(test_probs, axis=1) - 1
                            test_actual = y_test
                            
                            # Enhanced threshold optimization
                            if hasattr(df_input, 'close') and 'close' in df_input.columns:
                                try:
                                    returns = df_input['close'].pct_change().dropna().values
                                    if len(returns) >= len(test_actual):
                                        aligned_returns = returns[-len(test_actual):]
                                        if len(aligned_returns) == len(test_probs):
                                            threshold_optimizer.optimize_classification_threshold(test_probs, aligned_returns)
                                except Exception as thresh_error:
                                    logger.warning(f"Error during threshold optimization: {thresh_error}")
                                    # This is non-critical, we can continue with default threshold
                            
                            test_accuracy = accuracy_score(test_actual, test_predictions)
                            logger.analysis(f"Final Test Accuracy: {test_accuracy:.3f}")
                            
                        else:
                            test_predictions = test_outputs.cpu().numpy().flatten()
                            test_actual = y_test
                            
                            test_mse = np.mean((test_predictions - test_actual) ** 2)
                            test_mae = np.mean(np.abs(test_predictions - test_actual))
                            logger.analysis(f"Final Test MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")
                except Exception as test_error:
                    logger.error(f"Error during final test evaluation: {test_error}")
                    # This is non-critical, so we can continue

            # Enhanced model saving with comprehensive metadata
            if save_model and model is not None:
                model_params = {}
                if use_cnn:
                    model_params.update({
                        'cnn_features': getattr(model, 'cnn_features', 64),
                        'lstm_hidden': getattr(model, 'lstm_hidden', 32),
                    })
                else:
                    model_params.update({
                        'hidden_size': getattr(model, 'hidden_size', 32),
                        'num_layers': getattr(model, 'num_layers', 3),
                    })
                
                model_save_dict = {
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'input_size': len(feature_names),
                        'look_back': look_back,
                        'output_mode': output_mode,
                        'use_cnn': use_cnn,
                        'use_attention': use_attention,
                        'attention_heads': attention_heads,
                        'num_classes': 3 if output_mode == 'classification' else 1,
                        'dropout': GPU_MODEL_CONFIG['dropout'] if device is not None and hasattr(device, 'type') and device.type == 'cuda' else CPU_MODEL_CONFIG['dropout'],
                        **model_params
                    },
                    'data_info': {
                        'scaler': scaler,
                        'feature_names': feature_names,
                        'sequence_length': look_back
                    },
                    'optimization_results': {
                        'optimal_threshold': threshold_optimizer.best_threshold,
                        'best_sharpe': threshold_optimizer.best_sharpe,
                        'best_val_loss': best_val_loss,
                        'best_val_metric': best_val_metric,
                        'best_combined_score': best_combined_score
                    },
                    'training_history': training_history,
                    'training_metadata': {
                        'epochs_trained': epoch + 1,
                        'early_stopped': patience_counter >= early_stopping_patience,
                        'final_lr': optimizer.param_groups[0]['lr'],
                        'device_used': str(device),
                        'mixed_precision': use_mixed_precision,
                        'tensor_cores_used': has_tensor_cores,
                        'tensor_core_generation': tensor_info.get('generation', 'N/A')
                    }
                }
                
                success = safe_save_model(model_save_dict, str(model_path))
                if success:
                    logger.success(f"Model saved to: {model_path}")
                else:
                    logger.error(f"Failed to save model to: {model_path}")

            elapsed_time = time.time() - start_time
            logger.performance(f"Training completed in {elapsed_time:.2f}s")

            return model, str(model_path)

    except Exception as e:
        logger.exception(f"Error during model training: {e}")
        return None, ""
    
    finally:
        if device is not None and hasattr(device, 'type') and device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def load_cnn_lstm_attention_model(model_path: Union[str, Path]) -> Optional[Tuple[nn.Module, Dict, Dict, Dict]]:
    """Load a trained CNN-LSTM-Attention model from a file."""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
            logger.error(f"Model checkpoint at {model_path} is missing required 'model_config' or 'data_info' keys.")
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
                num_classes=model_config.get('num_classes', 3 if model_config['output_mode'] == 'classification' else 1),
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
        
        logger.success(f"Successfully loaded model from {model_path}")
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
    """Get the latest signal from a loaded CNN-LSTM-Attention model."""
    try:
        # Validate inputs
        if df_input is None or df_input.empty:
            logger.warning("Input DataFrame is empty")
            return SIGNAL_NEUTRAL
            
        if model is None:
            logger.error("Model is None")
            return SIGNAL_NEUTRAL
            
        if not model_config or not data_info:
            logger.error("Model config or data info is missing")
            return SIGNAL_NEUTRAL
            
        device = next(model.parameters()).device
        look_back = model_config.get('look_back', WINDOW_SIZE_LSTM)
        scaler = data_info.get('scaler')
        feature_names = data_info.get('feature_names', [])
        
        if scaler is None:
            logger.error("Scaler is missing from data_info")
            return SIGNAL_NEUTRAL
            
        if not feature_names:
            logger.error("Feature names are missing from data_info")
            return SIGNAL_NEUTRAL

        if len(df_input) < look_back:
            logger.warning(f"Not enough data for prediction: got {len(df_input)} rows, need {look_back}")
            return SIGNAL_NEUTRAL

        # Generate features
        df = generate_indicator_features(df_input.copy())
        if df is None or df.empty:
            logger.error("Feature generation failed")
            return SIGNAL_NEUTRAL
        
        # Select and scale features for the last sequence
        try:
            latest_data = df.tail(look_back)
            if len(latest_data) < look_back:
                logger.warning(f"Not enough data after feature generation: got {len(latest_data)} rows, need {look_back}")
                return SIGNAL_NEUTRAL
                
            available_features = [f for f in feature_names if f in latest_data.columns]
            if len(available_features) != len(feature_names):
                logger.warning(f"Mismatched features. Model needs {len(feature_names)} features, but data has {len(available_features)}.")
                logger.debug(f"Missing features: {set(feature_names) - set(available_features)}")
                return SIGNAL_NEUTRAL
            
            features = latest_data[feature_names].values
            if features is None or features.size == 0:
                logger.error("Feature extraction resulted in empty array")
                return SIGNAL_NEUTRAL
                
            # Handle invalid values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        except Exception as seq_error:
            logger.error(f"Error processing sequence data: {seq_error}")
            return SIGNAL_NEUTRAL
        
        try:
            scaled_features = scaler.transform(features)
        except Exception as scale_error:
            logger.error(f"Feature scaling failed: {scale_error}")
            return SIGNAL_NEUTRAL
        
        # Create tensor
        try:
            sequence_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)
        except Exception as tensor_error:
            logger.error(f"Tensor creation failed: {tensor_error}")
            return SIGNAL_NEUTRAL

        # Predict
        with torch.no_grad():
            try:
                output = model(sequence_tensor)
            except Exception as model_error:
                logger.error(f"Model prediction failed: {model_error}")
                return SIGNAL_NEUTRAL
            
        # Interpret output
        output_mode = model_config.get('output_mode', 'classification')
        if output_mode == 'classification':
            try:
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_class = predicted_class.item()
                confidence_value = confidence.item()
                
                # Safe threshold handling with multiple fallbacks
                confidence_threshold = None
                if optimization_results:
                    confidence_threshold = optimization_results.get('optimal_threshold')
                
                # Validate threshold
                if confidence_threshold is None or not isinstance(confidence_threshold, (int, float)) or not (0 <= confidence_threshold <= 1):
                    confidence_threshold = 0.5  # Default fallback threshold
                    logger.debug(f"Using default confidence threshold: {confidence_threshold}")
                
                # Ensure confidence_value is valid
                if not isinstance(confidence_value, (int, float)) or not (0 <= confidence_value <= 1):
                    logger.warning(f"Invalid confidence value: {confidence_value}, using NEUTRAL")
                    return SIGNAL_NEUTRAL
                
                if confidence_value < confidence_threshold:
                    logger.signal(f"CNN-LSTM Prediction: NEUTRAL (Confidence {confidence_value:.2f} < Threshold {confidence_threshold:.2f})")
                    return SIGNAL_NEUTRAL

                # Map prediction to signal
                if predicted_class == 2:
                    signal = SIGNAL_LONG
                elif predicted_class == 0:
                    signal = SIGNAL_SHORT
                else:
                    signal = SIGNAL_NEUTRAL
                
                logger.signal(f"CNN-LSTM Prediction: {signal} (Confidence: {confidence_value:.2f})")
                return signal
                
            except Exception as classification_error:
                logger.error(f"Classification output processing failed: {classification_error}")
                return SIGNAL_NEUTRAL
        else:
            logger.warning("Regression output mode not implemented for signal generation.")
            return SIGNAL_NEUTRAL

    except Exception as e:
        logger.error(f"Error during CNN-LSTM signal prediction: {e}")
        logger.debug(f"Error type: {type(e).__name__}")
        logger.debug(f"Error args: {e.args}")
        return SIGNAL_NEUTRAL


