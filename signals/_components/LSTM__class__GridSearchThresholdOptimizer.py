import logging
import numpy as np
import sys
from typing import Optional, Tuple

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__class__GridSearchThresholdOptimizer", log_level=logging.DEBUG)

class GridSearchThresholdOptimizer:
    """Grid search optimizer for signal thresholds to maximize Sharpe ratio."""
    
    def __init__(self, threshold_range: np.ndarray = np.arange(0.01, 0.15, 0.01)) -> None:
        self.threshold_range = threshold_range
        self.best_threshold: Optional[float] = None
        self.best_sharpe: float = -np.inf
    
    def optimize_regression_threshold(self, predictions: np.ndarray, returns: np.ndarray, prices: np.ndarray) -> Tuple[Optional[float], float]:
        """Optimize threshold for regression model to maximize Sharpe ratio."""
        if len(predictions) != len(returns) or len(returns) != len(prices):
            raise ValueError(f"Input arrays must have same length. Got predictions: {len(predictions)}, "
                           f"returns: {len(returns)}, prices: {len(prices)}")
        
        if len(predictions) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        best_threshold, best_sharpe = None, -np.inf
        
        for threshold in self.threshold_range:
            signals = np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))
            
            portfolio_values = [prices[0]]
            for i in range(len(signals) - 1):
                if signals[i] != 0:
                    portfolio_values.append(portfolio_values[-1] * (1 + signals[i] * returns[i + 1]))
                else:
                    portfolio_values.append(portfolio_values[-1])
            
            portfolio_values = np.array(portfolio_values)
            strategy_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_threshold = threshold
        
        self.best_threshold, self.best_sharpe = best_threshold, best_sharpe
        logger.model(f"Optimal threshold: {best_threshold or 0:.4f}, Best Sharpe: {best_sharpe:.4f}")
        
        return best_threshold, best_sharpe
    
    def optimize_classification_threshold(self, probabilities: np.ndarray, returns: np.ndarray) -> Tuple[Optional[float], float]:
        """Optimize confidence threshold for classification model."""
        if len(probabilities) != len(returns):
            raise ValueError(f"Input arrays must have same length. Got probabilities: {len(probabilities)}, "
                           f"returns: {len(returns)}")
        
        if len(probabilities) == 0:
            self.best_threshold = None
            self.best_sharpe = -np.inf
            return None, -np.inf
        
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