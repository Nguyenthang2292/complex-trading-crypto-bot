import logging
import sys
from typing import Optional, Tuple
import numpy as np

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None
from utilities._logger import setup_logging

logger = setup_logging(module_name="LSTM__class__grid_search_threshold_optimizer", log_level=logging.DEBUG)

class GridSearchThresholdOptimizer:
    """
    Grid search optimizer for signal thresholds to maximize Sharpe ratio.
    
    Optimizes thresholds for both regression and classification models by evaluating
    different threshold values and selecting the one that yields the highest Sharpe ratio.
    
    Args:
        threshold_range: Array of threshold values to test (default: 0.01 to 0.15 with 0.01 step)
    """
    
    def __init__(self, threshold_range: np.ndarray = np.arange(0.01, 0.15, 0.01)) -> None:
        self.threshold_range = threshold_range
        self.best_threshold: Optional[float] = None
        self.best_sharpe: float = -np.inf
    
    def optimize_regression_threshold(
        self, 
        predictions: np.ndarray, 
        returns: np.ndarray, 
        prices: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Optimize threshold for regression model to maximize Sharpe ratio.
        
        Args:
            predictions: Model predictions (returns) of shape (n_samples,)
            returns: Actual returns of shape (n_samples,)
            prices: Price series for backtesting of shape (n_samples,)
            
        Returns:
            Tuple of (best_threshold, best_sharpe_ratio)
        """
        best_threshold = None
        best_sharpe = -np.inf
        
        for threshold in self.threshold_range:
            signals = np.where(predictions > threshold, 1,  
                             np.where(predictions < -threshold, -1, 0))
            
            strategy_returns = signals[:-1] * returns[1:]
            
            if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_threshold = threshold
        
        self.best_threshold = best_threshold
        self.best_sharpe = best_sharpe
        
        logger.model("Optimal threshold: {0:.4f}, Best Sharpe: {1:.4f}".format(
            best_threshold or 0, best_sharpe))
        
        return best_threshold, best_sharpe
    
    def optimize_classification_threshold(
        self, 
        probabilities: np.ndarray, 
        returns: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Optimize confidence threshold for classification model.
        
        Args:
            probabilities: Softmax probabilities of shape (n_samples, n_classes)
            returns: Actual returns of shape (n_samples,)
            
        Returns:
            Tuple of (best_confidence_threshold, best_sharpe_ratio)
        """
        confidence_thresholds = np.arange(0.5, 0.95, 0.05)
        best_confidence = None
        best_sharpe = -np.inf
        
        for conf_threshold in confidence_thresholds:
            max_probs = np.max(probabilities, axis=1)
            predicted_classes = np.argmax(probabilities, axis=1) - 1
            
            signals = np.where(max_probs >= conf_threshold, predicted_classes, 0)
            strategy_returns = signals[:-1] * returns[1:]
            
            if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_confidence = conf_threshold
        
        logger.model("Optimal confidence: {0:.2f}, Best Sharpe: {1:.4f}".format(
            best_confidence or 0.6, best_sharpe))
        
        return (float(best_confidence) if best_confidence is not None else None, float(best_sharpe))