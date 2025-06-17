import logging
import numpy as np
import pandas as pd
import sys
import time

from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from utilities._logger import setup_logging
logger = setup_logging('lstm_attention_model', log_level=logging.DEBUG)

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent)) if str(current_dir.parent.parent) not in sys.path else None

from livetrade.config import (
    COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, CONFIDENCE_THRESHOLD, CPU_BATCH_SIZE, CPU_MODEL_CONFIG,
    DEFAULT_EPOCHS, GPU_MODEL_CONFIG, MIN_DATA_POINTS, MODEL_FEATURES, MODELS_DIR, NEUTRAL_ZONE_LSTM,
    SIGNAL_LONG, SIGNAL_NEUTRAL, SIGNAL_SHORT, TARGET_THRESHOLD_LSTM, TRAIN_TEST_SPLIT, VALIDATION_SPLIT,
    WINDOW_SIZE_LSTM
)
from signals._components._gpu_check_availability import (
    check_gpu_availability, configure_gpu_memory
)
from signals._components.LSTM__class__focal_loss import FocalLoss
from signals._components.LSTM__class__models import LSTMAttentionModel, LSTMModel
from signals._components._generate_indicator_features import _generate_indicator_features
from signals._components.LSTM__function__create_balanced_target import create_balanced_target
from signals._components.LSTM__function__evaluate_models import (evaluate_model_in_batches, evaluate_model_with_confidence)
from signals._components.LSTM__function__get_optimal_batch_size import get_optimal_batch_size
from signals.signals_cnn_lstm_attention import create_cnn_lstm_attention_model

def load_lstm_attention_model(model_path: Optional[Path] = None, use_attention: bool = True) -> Optional[nn.Module]:
    """
    Load LSTM model with or without attention.
    
    Args:
        model_path: Path to the model file
        use_attention: Whether to load attention model
        
    Returns:
        Loaded model or None if failed
    """
    if model_path is None:
        model_filename = "lstm_attention_model.pth" if use_attention else "lstm_model.pth"
        model_path = MODELS_DIR / model_filename
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        input_size = checkpoint['input_size']
        
        # Check if model uses attention
        model_has_attention = checkpoint.get('use_attention', False)
        attention_heads = checkpoint.get('attention_heads', 8)
        
        if model_has_attention:
            model = LSTMAttentionModel(
                input_size=input_size,
                num_heads=attention_heads
            )
            logger.model("Loading LSTM-Attention model with {0} heads".format(attention_heads))
        else:
            model = LSTMModel(input_size=input_size)
            logger.model("Loading standard LSTM model")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.model("Successfully loaded model from {0}".format(model_path))
        return model
        
    except Exception as e:
        logger.error("Error loading model: {0}".format(e))
        return None

def get_latest_lstm_attention_signal(df_market_data: pd.DataFrame, model: nn.Module) -> str:
    """
    Generate trading signal from LSTM model using latest market data.
    
    Args:
        df_market_data: DataFrame with OHLC market data
        model: Trained LSTM/LSTM-Attention model
        
    Returns:
        Trading signal: SIGNAL_LONG, SIGNAL_SHORT, or SIGNAL_NEUTRAL
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if df_market_data.empty:
        logger.warning("Empty input DataFrame for signal generation")
        return SIGNAL_NEUTRAL
    
    required_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]
    if not all(col in df_market_data.columns for col in required_cols):
        logger.warning(f"Missing OHLC columns. Available: {list(df_market_data.columns)}, Required: {required_cols}")
        return SIGNAL_NEUTRAL

    df_features = _generate_indicator_features(df_market_data.copy())
    if df_features.empty:
        logger.warning("Empty DataFrame after feature calculation")
        return SIGNAL_NEUTRAL
        
    if len(df_features) < WINDOW_SIZE_LSTM:
        logger.warning(f"Insufficient data: {len(df_features)} < {WINDOW_SIZE_LSTM}")
        return SIGNAL_NEUTRAL
    
    available_features = [col for col in MODEL_FEATURES if col in df_features.columns]
    if not available_features:
        logger.error(f"No valid features found from {MODEL_FEATURES}")
        return SIGNAL_NEUTRAL
    
    if len(available_features) < len(MODEL_FEATURES):
        logger.warning(f"Using {len(available_features)}/{len(MODEL_FEATURES)} features")
        
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_features[available_features])
    input_window = torch.FloatTensor([scaled_features[-WINDOW_SIZE_LSTM:]]).to(device)
    
    try:
        model.eval()
        with torch.no_grad():
            prediction_probs = model(input_window)[0].cpu().numpy()
        
        predicted_class = np.argmax(prediction_probs) - 1
        confidence = np.max(prediction_probs)
        
        model_type = "LSTM-Attention" if isinstance(model, LSTMAttentionModel) else "LSTM"
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        logger.signal(f"{model_type} ({device_info}) - Class: {predicted_class}, Confidence: {confidence:.3f}")
        
        if confidence >= CONFIDENCE_THRESHOLD:
            if predicted_class == 1:
                logger.signal(f"HIGH CONFIDENCE BUY signal ({confidence:.1%})")
                return SIGNAL_LONG
            elif predicted_class == -1:
                logger.signal(f"HIGH CONFIDENCE SELL signal ({confidence:.1%})")
                return SIGNAL_SHORT
            else:
                logger.signal(f"HIGH CONFIDENCE NEUTRAL signal ({confidence:.1%})")
                return SIGNAL_NEUTRAL
        else:
            logger.signal(f"LOW CONFIDENCE - Returning NEUTRAL ({confidence:.1%})")
            return SIGNAL_NEUTRAL
            
    except Exception as e:
        logger.error(f"Error generating LSTM signal: {e}")
        return SIGNAL_NEUTRAL

def train_lstm_attention_model(df_input: pd.DataFrame, save_model: bool = True, epochs: int = DEFAULT_EPOCHS, 
                    use_early_stopping: bool = True, use_attention: bool = True, attention_heads: int = GPU_MODEL_CONFIG['nhead']) -> Tuple[Optional[nn.Module], pd.DataFrame]:   
    """
    Trains a PyTorch LSTM model with optional attention mechanism.
    
    Args:
        df_input: Input DataFrame with OHLC price data
        save_model: Whether to save the trained model to disk
        epochs: Number of training epochs
        use_early_stopping: Whether to use early stopping during training
        use_attention: Whether to use multi-head attention mechanism
        attention_heads: Number of attention heads for attention mechanism
        
    Returns:
        Tuple of (trained_model, evaluation_results_dataframe)
        
    Raises:
        ValueError: If input data is insufficient or invalid
        RuntimeError: If CUDA/GPU operations fail during training
    """
    model, best_model_state, scaler_amp, results = None, None, None, pd.DataFrame()
    gpu_available, use_mixed_precision = False, False
    device = torch.device('cpu')
    
    # GPU setup and device configuration
    try:
        gpu_available = check_gpu_availability()
        if gpu_available and configure_gpu_memory():
            device = torch.device('cuda:0')
            if torch.cuda.get_device_capability(0)[0] >= 7:
                use_mixed_precision = True
                scaler_amp = GradScaler('cuda')
                logger.gpu("Using mixed precision training for faster performance")
            
            # Test GPU functionality
            try:
                torch.ones(1, device=device)
                logger.success("GPU device allocation test passed")
            except Exception as device_test_error:
                logger.error(f"GPU device test failed: {device_test_error}")
                logger.warning("Falling back to CPU due to GPU issues")
                device, gpu_available, use_mixed_precision = torch.device('cpu'), False, False
        else:
            logger.info("Using CPU for LSTM training")
    except Exception as gpu_error:
        logger.error(f"GPU setup failed: {gpu_error}")
        device, gpu_available, use_mixed_precision = torch.device('cpu'), False, False

    # Input validation and data preprocessing
    if df_input.empty:
        raise ValueError("Input DataFrame is empty. Cannot train model.")
    
    df = _generate_indicator_features(df_input)
    df = create_balanced_target(df, threshold=TARGET_THRESHOLD_LSTM, neutral_zone=NEUTRAL_ZONE_LSTM)
    df.dropna(inplace=True)
    
    if len(df) < MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data after preprocessing: {len(df)} rows. Need at least {MIN_DATA_POINTS} rows.")
    
    # Feature preparation and validation
    available_features = [col for col in MODEL_FEATURES if col in df.columns]
    if not available_features:
        logger.error(f"No valid features found from {MODEL_FEATURES} in dataframe columns: {df.columns.tolist()}")
        raise ValueError("No valid features available for model training")
    
    if len(available_features) < len(MODEL_FEATURES):
        missing_features = [col for col in MODEL_FEATURES if col not in df.columns]
        logger.warning(f"Missing features: {missing_features}. Using {len(available_features)} of {len(MODEL_FEATURES)} features")

    # Feature scaling and sequence creation
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[available_features].values)
    
    X, y = [], []
    for i in range(WINDOW_SIZE_LSTM, len(scaled_features)):
        X.append(scaled_features[i-WINDOW_SIZE_LSTM:i])
        y.append(df['Target'].iloc[i])
    
    X_array, y_array = np.array(X), np.array(y)
    
    if len(X_array) == 0:
        raise ValueError("No sequences created. Data too short for LSTM window.")
    
    # Data splitting and tensor conversion
    split = int(TRAIN_TEST_SPLIT * len(X_array))
    X_train, X_test = X_array[:split], X_array[split:]
    y_train, y_test = y_array[:split], y_array[split:]
    
    # Convert to PyTorch tensors (shift labels from -1,0,1 to 0,1,2)
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train + 1)
    y_test_tensor = torch.LongTensor(y_test + 1)
    
    # Batch size optimization and validation split
    optimal_batch_size = get_optimal_batch_size(device, len(MODEL_FEATURES), WINDOW_SIZE_LSTM)
    if use_attention and gpu_available:
        optimal_batch_size = max(16, optimal_batch_size // 2)
        logger.gpu(f"Adjusted batch size for attention model: {optimal_batch_size}")
    
    val_split = int(VALIDATION_SPLIT * len(X_train_tensor))
    X_val, y_val = X_train_tensor[-val_split:], y_train_tensor[-val_split:]
    X_train_tensor, y_train_tensor = X_train_tensor[:-val_split], y_train_tensor[:-val_split]
    
    # DataLoader creation with performance optimization
    num_workers = 2 if gpu_available else 0
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), 
        batch_size=optimal_batch_size, shuffle=True,
        pin_memory=gpu_available, num_workers=num_workers, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=optimal_batch_size, shuffle=False,
        pin_memory=gpu_available, num_workers=num_workers, persistent_workers=num_workers > 0
    )
    
    X_test_tensor, y_test_tensor = X_test_tensor.to(device, non_blocking=True), y_test_tensor.to(device, non_blocking=True)
    
    # Model creation with error handling
    input_size = len(MODEL_FEATURES)
    model_type = "LSTM-Attention" if use_attention else "LSTM" 
    try:
        model = create_cnn_lstm_attention_model(
            input_size=input_size,
            use_attention=use_attention,
            num_heads=attention_heads,
            dropout=GPU_MODEL_CONFIG['dropout'] if gpu_available else CPU_MODEL_CONFIG['dropout']
        ).to(device)
        
        logger.model(f"{model_type} model successfully created and moved to {device}")
        
    except Exception as model_error:
        if any(keyword in str(model_error).lower() for keyword in ["cudnn", "cuda"]):
            logger.error(f"CUDA/cuDNN error creating model: {model_error}")
            logger.info("Attempting CPU fallback...")
            
            device, use_mixed_precision = torch.device('cpu'), False
            try:
                model = create_cnn_lstm_attention_model(
                    input_size=input_size, use_attention=use_attention, num_heads=attention_heads,
                    dropout=CPU_MODEL_CONFIG['dropout']
                ).to(device)
                gpu_available = False
                logger.success("Model successfully created on CPU fallback")
            except Exception as cpu_model_error:
                raise ValueError(f"Cannot create LSTM model: {cpu_model_error}")
        else:
            raise ValueError(f"Cannot create LSTM model: {model_error}")
    
    if model is None:
        raise ValueError("Model creation failed - model is None")
    
    # Training setup: loss function, optimizer, scheduler
    y_train_cpu = y_train_tensor.cpu().numpy()
    unique_classes = np.unique(y_train_cpu)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_cpu)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.model(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Training loop with early stopping
    best_val_loss, patience_counter, patience, actual_epochs = float('inf'), 0, 5, 0
    logger.model(f"Training {model_type} model on {device} with {len(X_train_tensor)} samples for max {epochs} epochs...")
    logger.model(f"Batch size: {optimal_batch_size}, Mixed precision: {use_mixed_precision}")
    
    if use_attention:
        logger.model(f"Attention heads: {attention_heads}, Positional encoding: enabled")
    
    epoch_times = []
    
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            actual_epochs = epoch + 1
            
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            try:
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    
                    if use_mixed_precision and scaler_amp is not None:
                        with autocast('cuda'):
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                        scaler_amp.scale(loss).backward()
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += int((predicted == batch_y).sum().item())
                    
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        logger.performance(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
            except RuntimeError as runtime_error:
                error_msg = str(runtime_error).lower()
                if any(keyword in error_msg for keyword in ["cuda", "gpu", "cudnn"]):
                    logger.error(f"GPU error during training: {runtime_error}")
                    logger.info("Switching to CPU training...")
                    
                    # CPU fallback
                    device = torch.device('cpu')
                    model = model.cpu()
                    X_test_tensor, y_test_tensor = X_test_tensor.cpu(), y_test_tensor.cpu()
                    use_mixed_precision, scaler_amp, gpu_available = False, None, False
                    
                    # Recreate data loaders
                    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=CPU_BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=CPU_BATCH_SIZE, shuffle=False)
                    
                    logger.success("Successfully switched to CPU training")
                    continue
                else:
                    raise runtime_error
            
            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
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
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += int((predicted == batch_y).sum().item())
        
            # Calculate metrics and update scheduler
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            scheduler.step(val_loss)
            
            logger.performance(f'Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) - '
                             f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                             f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping logic
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.model(f"Early stopping triggered after {epoch+1} epochs")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
    
    except Exception as training_error:
        logger.error(f"Training failed: {training_error}")
        if model is None:
            raise ValueError(f"Training failed and model is None: {training_error}")
        logger.warning("Returning partially trained model due to training failure")
    
    # Performance summary and memory cleanup
    if epoch_times:
        avg_epoch_time, total_training_time = float(np.mean(epoch_times)), sum(epoch_times)
        logger.performance(f"Training completed - Avg epoch time: {avg_epoch_time:.1f}s, Total time: {total_training_time:.1f}s")
    
    if gpu_available and device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared after training")
        except:
            pass
    
    logger.model(f"Model training completed after {actual_epochs} epochs on {device}")
    
    # Model evaluation and results generation
    if len(X_test_tensor) > 0 and model is not None:
        try:
            model.eval()
            evaluation_batch_size = 32 if gpu_available else 64
            logger.info(f"Starting model evaluation with batch size: {evaluation_batch_size}")
            
            if gpu_available and device.type == 'cuda':
                torch.cuda.empty_cache()
            
            try:
                y_pred_prob = evaluate_model_in_batches(model, X_test_tensor, device, evaluation_batch_size)
                y_pred_classes = np.argmax(y_pred_prob, axis=1) - 1  # Convert back to -1,0,1
                y_test_cpu = y_test_tensor.cpu().numpy() - 1
                logger.success("Model evaluation completed successfully")
                
            except RuntimeError as eval_error:
                if "out of memory" in str(eval_error).lower():
                    logger.warning("GPU OOM during evaluation, falling back to CPU")
                    model = model.cpu()
                    X_test_cpu = X_test_tensor.cpu()
                    torch.cuda.empty_cache()
                    
                    y_pred_prob = evaluate_model_in_batches(model, X_test_cpu, torch.device('cpu'), 64)
                    y_pred_classes = np.argmax(y_pred_prob, axis=1) - 1
                    y_test_cpu = y_test_tensor.cpu().numpy() - 1
                    logger.success("Model evaluation completed on CPU")
                else:
                    raise eval_error
            
            # Print evaluation metrics
            logger.analysis("Classification Report:")
            logger.analysis(classification_report(y_test_cpu, y_pred_classes, zero_division=0))
            logger.analysis("Confusion Matrix:")
            logger.analysis(confusion_matrix(y_test_cpu, y_pred_classes))

            # Confidence evaluation with memory management
            logger.analysis("\n" + "="*60)
            logger.analysis("CONFIDENCE THRESHOLD EVALUATION")
            logger.analysis("="*60)
            
            if device.type == 'cuda':
                model_cpu = model.cpu()
                X_test_cpu = X_test_tensor.cpu() if X_test_tensor.device.type == 'cuda' else X_test_tensor
                evaluate_model_with_confidence(model_cpu, X_test_cpu, y_test_cpu, torch.device('cpu'))
                if gpu_available:
                    model = model.to(device)
            else:
                evaluate_model_with_confidence(model, X_test_tensor, y_test_cpu, device)
            
            # Create results DataFrame
            test_start_idx, test_end_idx = split + WINDOW_SIZE_LSTM, split + WINDOW_SIZE_LSTM + len(X_test_tensor)
            
            if test_end_idx <= len(df):
                test_data = df.iloc[test_start_idx:test_end_idx].reset_index()
                confidence_values = np.max(y_pred_prob, axis=1)
                signal_list = ['BUY' if i == 1 else 'SELL' if i == -1 else 'NEUTRAL' for i in y_pred_classes]
                signal_strength_list = ['Strong' if conf > 0.8 else 'Moderate' if conf > 0.65 else 'Weak' 
                                      for conf in confidence_values]
                
                results = pd.DataFrame({
                    'Date': test_data['time'] if 'time' in test_data.columns else test_data.index,
                    'close': test_data[COL_CLOSE],
                    'Actual_Next_Direction': y_test_cpu,
                    'Predicted_Direction': y_pred_classes,
                    'Confidence': confidence_values,
                    'Signal': signal_list,
                    'Signal_Strength': signal_strength_list
                })
            else:
                logger.warning("Not enough test data to create results DataFrame")

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            if gpu_available and device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cleared after evaluation error")
                except:
                    pass

    # Model saving
    if save_model and model is not None:
        try:
            model_filename = "lstm_attention_model.pth" if use_attention else "lstm_model.pth"
            model_path = MODELS_DIR / model_filename
            save_dict = {
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'scaler': scaler,
                'use_attention': use_attention,
                'attention_heads': attention_heads if use_attention else None
            }
            torch.save(save_dict, model_path)
            logger.success(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    elif model is None:
        logger.error("Cannot save model: model is None")
    
    return model, results

def train_and_save_global_lstm_attention_model(combined_df: pd.DataFrame, model_filename: Optional[str] = None,
                                     use_attention: bool = True, attention_heads: int = GPU_MODEL_CONFIG['nhead']) -> Tuple[Optional[nn.Module], str]:
    """
    Train and save PyTorch LSTM model with optional attention mechanism.

    Args:
        combined_df: Preprocessed DataFrame with multi-pair/timeframe data
        model_filename: Custom model filename, auto-generated if None
        use_attention: Enable attention mechanism
        attention_heads: Number of attention heads

    Returns:
        Tuple of (trained_model, saved_path) or (None, "") if training fails
    """
    start_time = time.time()
    
    try:
        # Setup GPU/CPU configuration
        gpu_available = check_gpu_availability()
        if gpu_available:
            configure_gpu_memory()
            logger.gpu("Using GPU for model training")
        else:
            logger.info("Using CPU for model training")

        # Generate model filename if not provided
        if not model_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_type = "attention" if use_attention else "standard"
            model_filename = f"lstm_{model_type}_model_{timestamp}.pth"

        # Ensure models directory exists and set save path
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / model_filename

        # Train the model
        logger.model(f"Training LSTM{'-Attention' if use_attention else ''} model...")
        model, _ = train_lstm_attention_model(
            combined_df, 
            save_model=False,
            use_attention=use_attention,
            attention_heads=attention_heads
        )

        if not model:
            logger.error("Model training failed - returned None")
            return None, ""

        # Save trained model with metadata
        logger.model(f"Saving model at: {model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': len(MODEL_FEATURES),
            'use_attention': use_attention,
            'attention_heads': attention_heads if use_attention else None
        }, model_path)

        elapsed_time = time.time() - start_time
        logger.performance(f"Model trained and saved in {elapsed_time:.2f}s: {model_path}")

        return model, str(model_path)

    except Exception as e:
        logger.exception(f"Error during LSTM model training: {e}")
        return None, ""
    
