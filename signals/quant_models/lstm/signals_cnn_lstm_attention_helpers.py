import numpy as np
import pandas as pd
import torch
import sys
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components._generate_indicator_features import generate_indicator_features
from components.config import (
    CPU_MODEL_CONFIG,
    GPU_MODEL_CONFIG,
    MODELS_DIR,
    SIGNAL_LONG,
    SIGNAL_NEUTRAL,
    SIGNAL_SHORT,
    WINDOW_SIZE_LSTM,
)
from signals._components.LSTM__function__get_optimal_batch_size import get_optimal_batch_size
from utilities._logger import setup_logging

logger = setup_logging(
    module_name="signals_cnn_lstm_attention_helpers",
    log_level=logging.DEBUG
)

def _create_training_config(device: torch.device) -> Dict:
    """
    Create training configuration based on device type.
    
    Args:
        device: PyTorch device (CPU or CUDA)
        
    Returns:
        Configuration dictionary for training
    """
    if device.type == 'cuda':
        return {
            'learning_rate': 0.001,
            'hidden_size': GPU_MODEL_CONFIG.d_model,
            'num_layers': GPU_MODEL_CONFIG.num_layers,
            'dropout': GPU_MODEL_CONFIG.dropout,
            'num_heads': GPU_MODEL_CONFIG.nhead,
            'cnn_features': 64,
            'lstm_hidden': 32
        }
    
    return {
        'learning_rate': 0.001,
        'hidden_size': CPU_MODEL_CONFIG.d_model,
        'num_layers': CPU_MODEL_CONFIG.num_layers,
        'dropout': CPU_MODEL_CONFIG.dropout,
        'num_heads': CPU_MODEL_CONFIG.nhead,
        'cnn_features': 32,
        'lstm_hidden': 16
    }

def _create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    use_cnn: bool,
    use_attention: bool,
    look_back: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for training, validation, and testing.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        device: PyTorch device
        use_cnn: Whether using CNN layers
        use_attention: Whether using attention mechanism
        look_back: Sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    # Create data loaders with proper batch size calculation
    model_type = 'cnn_lstm' if use_cnn else 'lstm_attention' if use_attention else 'lstm'
    batch_size = get_optimal_batch_size(
        device=device,
        input_size=X_train.shape[2],
        sequence_length=look_back,
        model_type=model_type
    )
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(X_test_tensor, torch.from_numpy(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def _evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    output_mode: str
) -> float:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: PyTorch device
        use_amp: Whether to use mixed precision
        output_mode: Output mode for prediction processing
        
    Returns:
        Test accuracy
    """
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch_x)
            test_predictions.extend(outputs.cpu().numpy())

    test_predictions = np.array(test_predictions)
    if output_mode == 'classification':
        test_predictions = np.argmax(test_predictions, axis=1)

    # Get test targets for accuracy calculation
    test_targets = []
    for _, batch_y in test_loader:
        test_targets.extend(batch_y.numpy())
    test_targets = np.array(test_targets)

    accuracy = accuracy_score(test_targets, test_predictions)
    logger.success(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy

def _save_model_checkpoint(
    model: nn.Module,
    config: Dict,
    features: List[str],
    look_back: int,
    output_mode: str,
    scaler: Union[MinMaxScaler, StandardScaler],
    accuracy: float,
    model_type: str,
    model_filename: str,
    use_cnn: bool,
    use_attention: bool
) -> str:
    """
    Save model checkpoint with all necessary information.
    
    Args:
        model: Trained model
        config: Model configuration
        features: Feature names
        look_back: Sequence length
        output_mode: Output mode
        scaler: Data scaler
        accuracy: Test accuracy
        model_type: Type of model
        model_filename: Filename for saving
        use_cnn: Whether using CNN
        use_attention: Whether using attention
        
    Returns:
        Path to saved model
    """
    from signals.signals_cnn_lstm_attention import safe_save_model
    
    model_path = Path(MODELS_DIR) / model_filename
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': config.get('hidden_size', 64),
            'look_back': look_back,
            'output_mode': output_mode,
            'use_cnn': use_cnn,
            'use_attention': use_attention,
            **config
        },
        'data_info': {
            'features': features, 
            'look_back': look_back, 
            'output_mode': output_mode,
            'feature_names': features,
            'scaler': scaler
        },
        'optimization_results': {
            'test_accuracy': accuracy,
            'model_type': model_type
        }
    }
    safe_save_model(checkpoint, str(model_path))
    return str(model_path)

def _validate_prediction_inputs(
    df_input: pd.DataFrame,
    model: nn.Module,
    model_config: Dict,
    data_info: Dict
) -> Tuple[bool, str, Optional[torch.device], int, Optional[Union[MinMaxScaler, StandardScaler]], List[str]]:
    """
    Validate inputs for prediction and extract necessary parameters.
    
    Args:
        df_input: Input DataFrame
        model: Loaded model
        model_config: Model configuration
        data_info: Data information
        
    Returns:
        Tuple of (is_valid, error_message, device, look_back, scaler, feature_names)
    """
    # Validate inputs
    if df_input is None or df_input.empty:
        return False, "Input DataFrame is empty", None, 0, None, []
        
    if model is None:
        return False, "Model is None", None, 0, None, []
        
    if not model_config or not data_info:
        return False, "Model config or data info is missing", None, 0, None, []
        
    device = next(model.parameters()).device
    look_back = model_config.get('look_back', WINDOW_SIZE_LSTM)
    scaler = data_info.get('scaler')
    feature_names = data_info.get('feature_names', [])
    
    if scaler is None:
        return False, "Scaler is missing from data_info", None, 0, None, []
        
    if not feature_names:
        return False, "Feature names are missing from data_info", None, 0, None, []
        
    return True, "", device, look_back, scaler, feature_names

def _prepare_prediction_data(
    df_input: pd.DataFrame,
    look_back: int,
    feature_names: List[str],
    scaler: Union[MinMaxScaler, StandardScaler]
) -> Tuple[bool, str, Optional[torch.Tensor]]:
    """
    Prepare data for prediction by generating features and creating tensor.
    
    Args:
        df_input: Input DataFrame
        look_back: Sequence length
        feature_names: Required feature names
        scaler: Data scaler
        
    Returns:
        Tuple of (success, error_message, sequence_tensor)
    """
    if len(df_input) < look_back:
        return False, f"Not enough data for prediction: got {len(df_input)} rows, need {look_back}", None

    # Generate features
    df = generate_indicator_features(df_input.copy())
    if df is None or df.empty:
        return False, "Feature generation failed", None
    
    # Select and scale features for the last sequence
    try:
        latest_data = df.tail(look_back)
        if len(latest_data) < look_back:
            return False, f"Not enough data after feature generation: got {len(latest_data)} rows, need {look_back}", None
            
        available_features = [f for f in feature_names if f in latest_data.columns]
        if len(available_features) != len(feature_names):
            missing_features = set(feature_names) - set(available_features)
            return False, f"Mismatched features. Model needs {len(feature_names)} features, but data has {len(available_features)}. Missing: {missing_features}", None
        
        features = latest_data[feature_names].values
        if features is None or features.size == 0:
            return False, "Feature extraction resulted in empty array", None
            
        # Handle invalid values
        features_array = np.array(features, dtype=np.float64)
        features = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception as seq_error:
        return False, f"Error processing sequence data: {seq_error}", None
    
    try:
        scaled_features = scaler.transform(features)
    except Exception as scale_error:
        return False, f"Feature scaling failed: {scale_error}", None
    
    # Create tensor
    try:
        sequence_tensor = torch.FloatTensor(scaled_features).unsqueeze(0)
    except Exception as tensor_error:
        return False, f"Tensor creation failed: {tensor_error}", None
        
    return True, "", sequence_tensor

def _process_classification_output(
    output: torch.Tensor,
    output_mode: str,
    optimization_results: Dict
) -> str:
    """
    Process classification model output to generate trading signal.
    
    Args:
        output: Model output tensor
        output_mode: Output mode ('classification' or 'classification_advanced')
        optimization_results: Optimization results containing thresholds
        
    Returns:
        Trading signal (LONG, SHORT, or NEUTRAL)
    """
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
        if (confidence_threshold is None or 
                not isinstance(confidence_threshold, (int, float)) or 
                not (0 <= confidence_threshold <= 1)):
            confidence_threshold = 0.5  # Default fallback threshold
            logger.debug(f"Using default confidence threshold: {confidence_threshold}")
        
        # Ensure confidence_value is valid
        if (not isinstance(confidence_value, (int, float)) or 
                not (0 <= confidence_value <= 1)):
            logger.warning(f"Invalid confidence value: {confidence_value}, using NEUTRAL")
            return SIGNAL_NEUTRAL
        
        if confidence_value < confidence_threshold:
            logger.signal(
                f"CNN-LSTM Prediction: NEUTRAL (Confidence {confidence_value:.2f} < "
                f"Threshold {confidence_threshold:.2f})"
            )
            return SIGNAL_NEUTRAL

        # Map prediction to signal based on mode
        if output_mode == 'classification':
            # Standard classification: 0=SHORT, 1=NEUTRAL, 2=LONG
            if predicted_class == 2:
                signal = SIGNAL_LONG
            elif predicted_class == 0:
                signal = SIGNAL_SHORT
            else:
                signal = SIGNAL_NEUTRAL
        else:  # classification_advanced
            # Advanced classification: 0=SHORT, 1=NEUTRAL, 2=LONG (mapped from -1, 0, 1)
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

def _process_regression_output(output: torch.Tensor) -> str:
    """
    Process regression model output to generate trading signal.
    
    Args:
        output: Model output tensor
        
    Returns:
        Trading signal (LONG, SHORT, or NEUTRAL)
    """
    try:
        # Regression mode: output is continuous value
        predicted_return = output.item()
        
        # Convert regression output to signal based on thresholds
        if predicted_return > 0.01:  # 1% positive return threshold
            signal = SIGNAL_LONG
        elif predicted_return < -0.01:  # 1% negative return threshold
            signal = SIGNAL_SHORT
        else:
            signal = SIGNAL_NEUTRAL
        
        logger.signal(f"CNN-LSTM Regression Prediction: {signal} (Return: {predicted_return:.4f})")
        return signal
        
    except Exception as regression_error:
        logger.error(f"Regression output processing failed: {regression_error}")
        return SIGNAL_NEUTRAL
