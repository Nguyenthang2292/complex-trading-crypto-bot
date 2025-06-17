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
# Initialize logger for LSTM Attention module
logger = setup_logging(module_name="_create_regression_targets", log_level=logging.DEBUG)

def create_regression_targets(df, target_col='close', future_shift=-1):
    """
    Create regression targets (future returns)
    
    Args:
        df: Input DataFrame
        target_col: Column to calculate returns from
        future_shift: Periods to shift for future returns (negative for future)
        
    Returns:
        DataFrame with 'return_target' column
    """
    df = df.copy()
    
    if target_col not in df.columns:
        logger.error("Target column '{0}' not found".format(target_col))
        return df
    
    # Calculate future returns: (price_t+1 - price_t) / price_t
    future_prices = df[target_col].shift(future_shift)
    current_prices = df[target_col]
    
    df['return_target'] = (future_prices - current_prices) / current_prices
    
    # Remove outliers (returns > 10%)
    df['return_target'] = np.clip(df['return_target'], -0.1, 0.1)
    
    logger.model("Created regression targets, mean return: {0:.4f}, std: {1:.4f}".format(
        df['return_target'].mean(), df['return_target'].std()))
    
    return df