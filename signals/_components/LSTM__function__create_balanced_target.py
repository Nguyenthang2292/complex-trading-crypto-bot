import logging
import numpy as np
import sys
import pandas as pd

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__function__create_balanced_target", log_level=logging.DEBUG)

from livetrade.config import (COL_CLOSE,TARGET_THRESHOLD_LSTM, NEUTRAL_ZONE_LSTM,FUTURE_RETURN_SHIFT)

def create_balanced_target(
    df: pd.DataFrame, 
    threshold: float = TARGET_THRESHOLD_LSTM, 
    neutral_zone: float = NEUTRAL_ZONE_LSTM
) -> pd.DataFrame:
    """
    Create balanced target labels (-1, 0, 1) for LSTM training based on future returns.
    
    This function analyzes future price movements and classifies them into three categories:
    - Strong Buy (1): Future return exceeds the threshold
    - Strong Sell (-1): Future return is below negative threshold
    - Neutral (0): Future return is within the neutral zone
    
    Intermediate cases between neutral_zone and threshold are assigned probabilistically
    to maintain class balance with a bias toward their movement direction.
    
    Args:
        df: DataFrame with price data containing at minimum 'close' column
        threshold: Strong movement threshold percentage for buy/sell signals
        neutral_zone: Threshold percentage for neutral classification
        
    Returns:
        DataFrame with added 'Target' column containing class labels (1, -1, 0)
        
    Raises:
        Warning: If DataFrame is empty or doesn't have enough data points
    """
    try:
        # Validate input data
        if df.empty or COL_CLOSE not in df.columns:
            logger.warning("Input DataFrame is empty or missing close column")
            return df
        
        if len(df) < abs(FUTURE_RETURN_SHIFT) + 1:
            logger.warning("Not enough data points for future return calculation")
            return df
        
        # Calculate future price movement as percentage
        future_return = df[COL_CLOSE].shift(-abs(FUTURE_RETURN_SHIFT)) / df[COL_CLOSE] - 1
        
        # Create target labels for strong and neutral movements
        conditions = [
            future_return > threshold,             # Strong upward movement
            future_return < -threshold,            # Strong downward movement
            abs(future_return) <= neutral_zone     # Minimal movement (neutral)
        ]
        choices = [1, -1, 0]  # Buy, Sell, Hold
        
        df['Target'] = np.select(conditions, choices, default=np.nan)
        
        # Handle intermediate cases with probabilistic assignment for balance
        np.random.seed(42)  # For reproducibility
        intermediate_mask = (abs(future_return) > neutral_zone) & (abs(future_return) <= threshold)
        intermediate_indices = future_return[intermediate_mask].index
        
        # Assign intermediate cases with bias toward their direction but some randomness
        for idx in intermediate_indices:
            direction = future_return[idx] > 0
            df.loc[idx, 'Target'] = 1 if direction and np.random.random() > 0.3 else (-1 if not direction and np.random.random() > 0.3 else 0)
        
        # Remove rows with undefined targets
        df = df.dropna(subset=['Target'])
        
        # Log class distribution for monitoring
        if not df.empty:
            target_counts = df['Target'].value_counts().sort_index()
            logger.debug(f"Target distribution: {dict(target_counts)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in create_balanced_target: {e}")
        return df