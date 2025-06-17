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
logger = setup_logging(module_name="_create_classification_targets", log_level=logging.DEBUG)

from signals._quant_models.LSTM__function__create_regression_targets import create_regression_targets

def create_classification_targets(df, target_col='close', future_shift=-1, 
                                threshold=0.01, neutral_zone=0.005):
    """
    Create classification targets (up/down/neutral)
    
    Args:
        df: Input DataFrame
        target_col: Column to calculate returns from
        future_shift: Periods to shift for future returns
        threshold: Threshold for strong movements
        neutral_zone: Neutral zone around zero
        
    Returns:
        DataFrame with 'class_target' column (-1, 0, 1)
    """
    df = df.copy()
    
    # First create regression targets
    df = create_regression_targets(df, target_col, future_shift)
    
    if 'return_target' not in df.columns:
        logger.error("Failed to create return targets")
        return df
    
    returns = df['return_target']
    
    # Create classification labels
    conditions = [
        returns > threshold,           # Strong UP (1)
        returns < -threshold,          # Strong DOWN (-1)
        abs(returns) <= neutral_zone   # Neutral (0)
    ]
    choices = [1, -1, 0]
    
    df['class_target'] = np.select(conditions, choices, default=0)
    
    # Log class distribution
    class_counts = df['class_target'].value_counts().sort_index()
    logger.model("Classification targets distribution: {0}".format(dict(class_counts)))
    
    return df