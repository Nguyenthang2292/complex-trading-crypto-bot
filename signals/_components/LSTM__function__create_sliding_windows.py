import logging
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.dirname(current_dir)
signals_dir = os.path.dirname(components_dir)
if signals_dir not in sys.path:
    sys.path.insert(0, signals_dir)

from utilities._logger import setup_logging
logger = setup_logging(module_name="_create_sliding_windows", log_level=logging.DEBUG)

def create_sliding_windows(data, look_back=60, target_col='target', feature_cols=None):
    """
    Create sliding windows for CNN-LSTM model
    
    Args:
        data: DataFrame with features and target
        look_back: Window size (default 60)
        target_col: Target column name
        feature_cols: List of feature columns (if None, use all except target)
        
    Returns:
        X: Input sequences [n_samples, look_back, n_features]
        y: Targets [n_samples] or [n_samples, 1]
        feature_names: List of feature column names
    """
    if data.empty:
        logger.warning("Input data is empty")
        return np.array([]), np.array([]), []
    
    if target_col not in data.columns:
        logger.error("Target column '{0}' not found in data".format(target_col))
        return np.array([]), np.array([]), []
    
    # Select feature columns
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]
    
    # Ensure we have enough data
    if len(data) < look_back + 1:
        logger.warning("Insufficient data: need at least {0} rows, got {1}".format(
            look_back + 1, len(data)))
        return np.array([]), np.array([]), []
    
    features = data[feature_cols].values
    targets = data[target_col].values
    
    X, y = [], []
    for i in range(look_back, len(data)):
        # X: window of features [look_back, n_features]
        X.append(features[i-look_back:i])
        # y: target at time i
        y.append(targets[i])
    
    X = np.array(X)  # [n_samples, look_back, n_features]
    y = np.array(y)  # [n_samples]
    
    logger.model("Created {0} sliding windows with shape X: {1}, y: {2}".format(
        len(X), X.shape, y.shape))
    
    return X, y, feature_cols


