import logging
import numpy as np
import os
import sys

# Setup module paths
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.dirname(current_dir)
signals_dir = os.path.dirname(components_dir)
if signals_dir not in sys.path:
    sys.path.insert(0, signals_dir)

from utilities._logger import setup_logging
# Initialize logger for target creation
logger = setup_logging(module_name="_create_balanced_target", log_level=logging.DEBUG)

from livetrade.config import (
    COL_CLOSE,
    TARGET_THRESHOLD_LSTM,
    NEUTRAL_ZONE_LSTM,
    FUTURE_RETURN_SHIFT,
)

def create_balanced_target(df, threshold=TARGET_THRESHOLD_LSTM, neutral_zone=NEUTRAL_ZONE_LSTM):
    """
    Create balanced target labels (-1, 0, 1) for LSTM training based on future returns.
    
    Args:
        df: DataFrame with price data (lowercase column names)
        threshold: Strong movement threshold for buy/sell signals
        neutral_zone: Threshold for neutral classification
        
    Returns:
        DataFrame with 'Target' column containing class labels:
        1 = Strong Buy, -1 = Strong Sell, 0 = Neutral
    """
    # Create uppercase column versions for compatibility
    df['Close'] = df['close']
    df['Open'] = df['open']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Volume'] = df['volume']
    
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
        
        # Create clear target labels for strong movements
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
        intermediate_returns = future_return[intermediate_mask]
        
        # Assign intermediate cases with bias toward their direction but some randomness
        for idx in intermediate_returns.index:
            if future_return[idx] > 0:
                df.loc[idx, 'Target'] = 1 if np.random.random() > 0.3 else 0  # 70% chance of Buy
            else:
                df.loc[idx, 'Target'] = -1 if np.random.random() > 0.3 else 0  # 70% chance of Sell
        
        # Remove rows with undefined targets
        df = df.dropna(subset=['Target'])
        
        # Log class distribution for monitoring
        if 'Target' in df.columns and not df.empty:
            target_counts = df['Target'].value_counts().sort_index()
            logger.debug(f"Target distribution: {dict(target_counts)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in create_balanced_target: {e}")
        return df