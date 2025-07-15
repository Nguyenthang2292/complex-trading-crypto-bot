import logging
import numpy as np
import sys
import pandas as pd

from pathlib import Path; 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__function__create_regression_targets", log_level=logging.DEBUG)

def create_regression_targets(
    df: pd.DataFrame, 
    target_col: str = 'close', 
    future_shift: int = -1
) -> pd.DataFrame:
    """
    Create regression targets (future returns) for time series prediction.
    
    This function calculates future returns based on price movements over a specified
    time shift. The returns are computed as percentage changes and clipped to remove
    extreme outliers that could destabilize model training.
    
    Formula: return = (future_price - current_price) / current_price
    
    Args:
        df: Input DataFrame containing price data
        target_col: Column name to calculate returns from (typically 'close')
        future_shift: Number of periods to shift for future returns 
                     (negative values look forward, positive look backward)
        
    Returns:
        DataFrame with added 'return_target' column containing clipped percentage returns
        
    Raises:
        Error: If target column is not found in DataFrame
    """
    df = df.copy()
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return df
    
    # Calculate future returns: (price_t+shift - price_t) / price_t
    current_prices = df[target_col]
    future_prices = current_prices.shift(future_shift)
    
    df['return_target'] = (future_prices - current_prices) / current_prices
    
    # Remove outliers by clipping returns to [-10%, +10%] range
    df['return_target'] = np.clip(df['return_target'], -0.1, 0.1)
    
    # Log statistics for monitoring
    mean_return = df['return_target'].mean()
    std_return = df['return_target'].std()
    logger.debug(f"Created regression targets, mean return: {mean_return:.4f}, std: {std_return:.4f}")
    
    return df