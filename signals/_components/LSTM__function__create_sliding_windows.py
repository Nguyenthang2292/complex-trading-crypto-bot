import logging
import numpy as np
import pandas as pd
import sys
from typing import List, Optional, Tuple

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging

logger = setup_logging(module_name="LSTM__function__create_sliding_windows", log_level=logging.DEBUG)

def create_sliding_windows(
    data: pd.DataFrame, 
    look_back: int = 60, 
    target_col: str = 'target', 
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sliding windows for LSTM model training and prediction.
    
    Transforms time series data into sequences of fixed length windows for supervised learning.
    Each window contains 'look_back' time steps of features with corresponding target value.
    
    Args:
        data: DataFrame containing time series features and target column
        look_back: Number of time steps to look back for each sequence window (default: 60)
        target_col: Name of the target column in DataFrame (default: 'target')
        feature_cols: List of feature column names. If None, uses all columns except target
        
    Returns:
        Tuple containing:
        - X: Input sequences array with shape [n_samples, look_back, n_features]
        - y: Target values array with shape [n_samples]
        - feature_cols: List of feature column names used in sequences
    """
    empty_result: Tuple[np.ndarray, np.ndarray, List[str]] = (np.array([]), np.array([]), [])
    
    if data.empty:
        logger.warning("Input data is empty")
        return empty_result
    
    if target_col not in data.columns:
        logger.error("Target column '{0}' not found in data".format(target_col))
        return empty_result
    
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]
    
    data_length: int = len(data)
    if data_length < look_back + 1:
        logger.warning("Insufficient data: need at least {0} rows, got {1}".format(
            look_back + 1, data_length))
        return empty_result
    
    features: np.ndarray = data[feature_cols].values
    targets: np.ndarray = data[target_col].values
    n_samples: int = data_length - look_back
    n_features: int = len(feature_cols)
    
    X: np.ndarray = np.zeros((n_samples, look_back, n_features))
    y: np.ndarray = np.zeros(n_samples)
    
    for i in range(n_samples):
        X[i] = features[i:i + look_back]
        y[i] = targets[i + look_back]
    
    logger.model("Created {0} sliding windows with shape X: {1}, y: {2}".format(
        n_samples, X.shape, y.shape))
    
    return X, y, feature_cols


