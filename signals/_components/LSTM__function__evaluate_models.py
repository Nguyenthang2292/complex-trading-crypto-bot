import logging
import numpy as np
import sys
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from livetrade.config import (CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLDS)
from utilities._logger import setup_logging

logger = setup_logging(module_name="LSTM__function__evaluate_models", log_level=logging.DEBUG)

def apply_confidence_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply confidence threshold to model predictions for improved reliability.
    
    Args:
        y_proba: Prediction probabilities for each class
        threshold: Confidence threshold (0.0-1.0)
        
    Returns:
        Array of class predictions (-1, 0, 1) after applying threshold
    """
    max_confidences = np.max(y_proba, axis=1)
    predictions = np.where(
        max_confidences >= threshold,
        np.argmax(y_proba, axis=1) - 1,
        0
    )
    return predictions

def evaluate_model_in_batches(
    model: torch.nn.Module, 
    X_test: torch.Tensor, 
    device: torch.device, 
    batch_size: int = 32
) -> np.ndarray:
    """
    Evaluate model in batches to avoid CUDA out of memory errors.
    
    Args:
        model: PyTorch model to evaluate
        X_test: Test data tensor
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Prediction probabilities as numpy array
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            batch_X = X_test[i:batch_end].to(device)
            
            try:
                batch_pred = model(batch_X).cpu()
                all_predictions.append(batch_pred)
                
                del batch_X, batch_pred
                
                if device.type == 'cuda' and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM during batch {0}-{1}, reducing batch size".format(i, batch_end))
                    if batch_size > 1:
                        smaller_batch = max(1, batch_size // 2)
                        return evaluate_model_in_batches(model, X_test, device, smaller_batch)
                    else:
                        logger.error("Cannot reduce batch size further - falling back to CPU")
                        raise e
                else:
                    raise e
    
    return torch.cat(all_predictions, dim=0).numpy()

def evaluate_model_with_confidence(
    model: torch.nn.Module, 
    X_test: torch.Tensor, 
    y_test: np.ndarray, 
    device: torch.device
) -> None:
    """
    Evaluate LSTM model with multiple confidence thresholds for trading signals.
    
    Args:
        model: Trained PyTorch LSTM model
        X_test: Test features (n_samples, sequence_length, n_features)
        y_test: Test labels {-1: SELL, 0: NEUTRAL, 1: BUY}
        device: Evaluation device (CPU/CUDA)
    """
    evaluation_batch_size = 16 if device.type == 'cuda' else 32
    
    logger.analysis("Evaluating model with {0} test samples in batches of {1}...".format(
        len(X_test), evaluation_batch_size))
    
    try:
        y_pred_prob = evaluate_model_in_batches(model, X_test, device, evaluation_batch_size)
    except Exception as e:
        logger.error("Failed to evaluate model in batches: {0}".format(e))
        return
    
    logger.analysis("Test set class distribution: {0}".format(np.bincount(y_test + 1)))
    
    for threshold in CONFIDENCE_THRESHOLDS:
        logger.analysis("\n" + "="*50)
        logger.analysis("CONFIDENCE THRESHOLD: {0:.1%}".format(threshold))
        logger.analysis("="*50)
        
        y_pred = apply_confidence_threshold(y_pred_prob, threshold)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
        
        for metric_name, metric_value in metrics.items():
            logger.analysis("{0}: {1:.3f}".format(metric_name.capitalize(), metric_value))
        
        signal_counts = np.array([np.sum(y_pred == i) for i in [-1, 0, 1]])
        total = len(y_pred)
        signal_labels = ['SELL (-1)', 'NEUTRAL (0)', 'BUY (1)']
        
        logger.analysis("Signal Distribution:")
        for i, (label, count) in enumerate(zip(signal_labels, signal_counts)):
            logger.analysis("  {0}: {1:.0f} ({2:.1f}%)".format(label, count, count/total*100))
        
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        logger.analysis("Confusion Matrix:")
        logger.analysis("     SELL  NEUTRAL  BUY")
        cm_labels = ['SELL', 'NEUTRAL', 'BUY']
        for i, label in enumerate(cm_labels):
            logger.analysis("{0:>8}: {1}".format(label, cm[i]))
        
        buy_precision = precision_score(y_test, y_pred, labels=[1], average='macro', zero_division=0)
        sell_precision = precision_score(y_test, y_pred, labels=[-1], average='macro', zero_division=0)
        
        logger.analysis("Trading Metrics:")
        logger.analysis("  BUY Signal Precision: {0:.3f}".format(buy_precision))
        logger.analysis("  SELL Signal Precision: {0:.3f}".format(sell_precision))
        
        buy_freq, sell_freq = signal_counts[2] / total, signal_counts[0] / total
        trading_score = (buy_precision * buy_freq + sell_precision * sell_freq) / (buy_freq + sell_freq + 1e-6)
        logger.analysis("  Trading Score: {0:.3f}".format(trading_score))
    
    logger.analysis("RECOMMENDATION:")
    logger.analysis("For conservative trading: Use threshold 0.75+ (higher precision, fewer signals)")
    logger.analysis("For active trading: Use threshold 0.60-0.65 (balanced precision/frequency)")
    logger.analysis("Current default: {0} (used in get_latest_LSTM_signal)".format(CONFIDENCE_THRESHOLD))