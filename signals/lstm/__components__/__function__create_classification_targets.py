import logging
import numpy as np
import sys
import pandas as pd

from pathlib import Path; 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__function__create_classification_targets", log_level=logging.DEBUG)

from signals._components.LSTM__function__create_regression_targets import create_regression_targets

def create_classification_targets(
    df: pd.DataFrame, 
    target_col: str = 'close', 
    future_shift: int = -1, 
    threshold: float = 0.01, 
    neutral_zone: float = 0.005
) -> pd.DataFrame:
    """
    Create classification targets (up/down/neutral) from price data.
    
    This function converts continuous returns into discrete classification labels
    based on price movement thresholds. It first creates regression targets using
    the specified column and shift, then classifies them into three categories:
    - Strong UP (1): Returns above threshold
    - Strong DOWN (-1): Returns below negative threshold  
    - Neutral (0): Returns within neutral zone around zero
    
    Args:
        df: Input DataFrame with price data
        target_col: Column name to calculate returns from
        future_shift: Number of periods to shift for future returns (negative for forward-looking)
        threshold: Minimum absolute return for strong movement classification
        neutral_zone: Maximum absolute return for neutral classification
        
    Returns:
        DataFrame with added 'class_target' column containing classification labels (-1, 0, 1)
        
    Raises:
        Error: If regression target creation fails
    """
    df = df.copy()
    
    # Create regression targets first
    df = create_regression_targets(df, target_col, future_shift)
    
    if 'return_target' not in df.columns:
        logger.error("Failed to create return targets")
        return df
    
    returns = df['return_target']
    
    # Create classification labels based on return thresholds
    conditions = [
        returns > threshold,           # Strong UP (1)
        returns < -threshold,          # Strong DOWN (-1)
        abs(returns) <= neutral_zone   # Neutral (0)
    ]
    choices = [1, -1, 0]
    
    df['class_target'] = np.select(conditions, choices, default=0)
    
    # Log class distribution for monitoring
    class_counts = df['class_target'].value_counts().sort_index()
    logger.debug(f"Classification targets distribution: {dict(class_counts)}")
    
    return df