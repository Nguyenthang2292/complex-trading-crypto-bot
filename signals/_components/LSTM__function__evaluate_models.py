import logging
import numpy as np
import os
import sys
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)

current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.dirname(current_dir)
signals_dir = os.path.dirname(components_dir)
if signals_dir not in sys.path:
    sys.path.insert(0, signals_dir)

from livetrade.config import (
    CONFIDENCE_THRESHOLD, 
    CONFIDENCE_THRESHOLDS, 
)

from utilities._logger import setup_logging

# Initialize logger for LSTM Attention module
logger = setup_logging(module_name="_evaluate_model", log_level=logging.DEBUG)

def apply_confidence_threshold(y_proba, threshold):
    """
    Apply confidence threshold to model predictions to improve reliability.
    
    This function converts probability predictions to class labels based on a confidence
    threshold. If the maximum probability is below the threshold, the prediction defaults
    to neutral (0).
    
    Args:
        y_proba (np.ndarray): Array of prediction probabilities for each class
        threshold (float): Confidence threshold to apply (0.0-1.0)
        
    Returns:
        np.ndarray: Array of class predictions (-1, 0, 1) after applying threshold
    """
    predictions = []
    for proba_row in y_proba:
        max_confidence = np.max(proba_row)
        if max_confidence >= threshold:
            predicted_class = np.argmax(proba_row) - 1  # Convert back to -1,0,1
        else:
            predicted_class = 0  # NEUTRAL
        predictions.append(predicted_class)
    return np.array(predictions)


def evaluate_model_in_batches(model, X_test, device, batch_size=32):
    """
    Evaluate model in batches to avoid CUDA out of memory errors.
    
    Args:
        model: PyTorch model to evaluate
        X_test: Test data tensor
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        numpy array: Prediction probabilities
    """
    model.eval()
    all_predictions = []
    
    # Process in smaller batches to avoid OOM
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            batch_X = X_test[i:batch_end].to(device)
            
            try:
                batch_pred = model(batch_X).cpu()
                all_predictions.append(batch_pred)
                
                # Clear intermediate tensors
                del batch_X, batch_pred
                
                # Clear GPU cache periodically
                if device.type == 'cuda' and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM during batch {0}-{1}, reducing batch size".format(i, batch_end))  # Fix: Use consistent formatting
                    # Try with smaller batch size
                    if batch_size > 1:
                        smaller_batch = max(1, batch_size // 2)
                        return evaluate_model_in_batches(model, X_test, device, smaller_batch)
                    else:
                        logger.error("Cannot reduce batch size further - falling back to CPU")
                        raise e
                else:
                    raise e
    
    # Concatenate all predictions
    return torch.cat(all_predictions, dim=0).numpy()

def evaluate_model_with_confidence(model, X_test, y_test, device):
    """
    Evaluate LSTM model with multiple confidence thresholds for trading signals.
    
    Performs comprehensive evaluation using various confidence thresholds to assess
    trading signal reliability and determine optimal settings for live trading.
    
    Args:
        model (torch.nn.Module): Trained PyTorch LSTM model
        X_test (torch.Tensor): Test features (n_samples, sequence_length, n_features)
        y_test (np.ndarray): Test labels {-1: SELL, 0: NEUTRAL, 1: BUY}
        device (torch.device): Evaluation device (CPU/CUDA)
        
    Returns:
        None: Logs evaluation results and threshold recommendations
        
    Note:
        Uses batch evaluation to handle GPU memory constraints.
        Evaluates multiple thresholds from config.CONFIDENCE_THRESHOLDS.
    """
    # Use batch evaluation to avoid memory issues
    evaluation_batch_size = 16 if device.type == 'cuda' else 32  # Very small batch for GPU
    
    logger.analysis("Evaluating model with {0} test samples in batches of {1}...".format(
        len(X_test), evaluation_batch_size))
    
    try:
        y_pred_prob = evaluate_model_in_batches(model, X_test, device, evaluation_batch_size)
    except Exception as e:
        logger.error("Failed to evaluate model in batches: {0}".format(e))
        return
    
    logger.analysis("Test set class distribution: {0}".format(np.bincount(y_test + 1)))  # Shift for counting
    
    # Use config confidence thresholds
    for threshold in CONFIDENCE_THRESHOLDS:
        logger.analysis("\n" + "="*50)
        logger.analysis("CONFIDENCE THRESHOLD: {0:.1%}".format(threshold))
        logger.analysis("="*50)
        
        y_pred = apply_confidence_threshold(y_pred_prob, threshold)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        logger.analysis("Accuracy: {0:.3f}".format(accuracy))
        logger.analysis("Precision: {0:.3f}".format(precision))
        logger.analysis("Recall: {0:.3f}".format(recall))
        logger.analysis("F1-Score: {0:.3f}".format(f1))
        
        # Signal distribution
        unique_preds = np.unique(y_pred)
        signal_counts = np.zeros(3)  # For -1, 0, 1
        
        for pred in unique_preds:
            count = np.sum(y_pred == pred)
            if pred == -1:
                signal_counts[0] = count
            elif pred == 0:
                signal_counts[1] = count
            elif pred == 1:
                signal_counts[2] = count
        
        total = len(y_pred)
        logger.analysis("Signal Distribution:")
        logger.analysis("  SELL (-1): {0:.0f} ({1:.1f}%)".format(signal_counts[0], signal_counts[0]/total*100))
        logger.analysis("  NEUTRAL (0): {0:.0f} ({1:.1f}%)".format(signal_counts[1], signal_counts[1]/total*100))
        logger.analysis("  BUY (1): {0:.0f} ({1:.1f}%)".format(signal_counts[2], signal_counts[2]/total*100))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        logger.analysis("Confusion Matrix:")
        logger.analysis("     SELL  NEUTRAL  BUY")
        for i, label in enumerate(['SELL', 'NEUTRAL', 'BUY']):
            logger.analysis("{0:>8}: {1}".format(label, cm[i]))
        
        # Trading-specific metrics
        buy_precision = precision_score(y_test, y_pred, labels=[1], average='macro', zero_division=0)
        sell_precision = precision_score(y_test, y_pred, labels=[-1], average='macro', zero_division=0)
        
        logger.analysis("Trading Metrics:")
        logger.analysis("  BUY Signal Precision: {0:.3f}".format(buy_precision))
        logger.analysis("  SELL Signal Precision: {0:.3f}".format(sell_precision))
        
        # Calculate trading score
        buy_freq = signal_counts[2] / total
        sell_freq = signal_counts[0] / total
        trading_score = (buy_precision * buy_freq + sell_precision * sell_freq) / (buy_freq + sell_freq + 1e-6)
        logger.analysis("  Trading Score: {0:.3f}".format(trading_score))
    
    # Recommend optimal threshold
    logger.analysis("RECOMMENDATION:")
    logger.analysis("For conservative trading: Use threshold 0.75+ (higher precision, fewer signals)")
    logger.analysis("For active trading: Use threshold 0.60-0.65 (balanced precision/frequency)")
    logger.analysis("Current default: {0} (used in get_latest_LSTM_signal)".format(CONFIDENCE_THRESHOLD))