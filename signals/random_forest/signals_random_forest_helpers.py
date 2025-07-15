"""Helper functions for Random Forest model training and data preprocessing.

This module provides utility functions for preparing training data, applying SMOTE
for class balancing, and creating Random Forest models with proper class weights.
"""

import gc
import logging
import os
import sys
import numpy as np
import pandas as pd
import psutil

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from typing import Optional, Tuple, cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    BUY_THRESHOLD,
    COL_CLOSE,
    LARGE_DATASET_THRESHOLD_FOR_SMOTE,
    MIN_MEMORY_GB,
    MIN_TRAINING_SAMPLES,
    MODEL_FEATURES,
    MODEL_RANDOM_STATE,
    SELL_THRESHOLD,
)
from components.generate_indicator_features import generate_indicator_features
from utilities.logger import setup_logging

logger = setup_logging(module_name="signals_random_forest_helpers", log_level=logging.INFO)

SMOTE_RANDOM_STATE = MODEL_RANDOM_STATE

def _prepare_training_data(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Prepare feature data and target variable for model training.

    Args:
        df (pd.DataFrame): Input OHLCV data.

    Returns:
        Optional[Tuple[pd.DataFrame, pd.Series]]: Tuple (features, target) or None if error.
    """
    logger.info("Calculating features for training data...")
    df_with_features = generate_indicator_features(df)
    if df_with_features.empty:
        logger.error("Feature calculation resulted in an empty DataFrame.")
        return None
    
    logger.info("Creating target variable 'target'...")
    future_return = (
        df_with_features[COL_CLOSE].shift(-5) / df_with_features[COL_CLOSE] - 1
    )
    df_with_features['target'] = np.select(
        [future_return > BUY_THRESHOLD, future_return < SELL_THRESHOLD],
        [1, -1],
        default=0
    )
    
    # Ensure target is numeric and convert to int
    df_with_features['target'] = pd.to_numeric(df_with_features['target'], errors='coerce').astype(int)
    
    df_with_features.dropna(subset=['target'] + MODEL_FEATURES, inplace=True)
    
    if len(df_with_features) < MIN_TRAINING_SAMPLES:
        logger.warning(
            "Insufficient training samples after feature creation: "
            f"{len(df_with_features)} < {MIN_TRAINING_SAMPLES}"
        )
        return None
    
    features = df_with_features[MODEL_FEATURES]
    target = df_with_features['target']
    logger.info(
        f"Training data prepared. Features shape: {features.shape}, "
        f"target shape: {target.shape}"
    )
    logger.debug(
        f"Class distribution before resampling: {target.value_counts().to_dict()}"
    )
    return cast(Tuple[pd.DataFrame, pd.Series], (features, target))

def _apply_smote(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance training data using SMOTE.

    Args:
        features (pd.DataFrame): Input features.
        target (pd.Series): Target labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Balanced data or original data if error.
    """
    logger.info("Applying SMOTE for class balancing...")
    try:
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < MIN_MEMORY_GB:
            logger.warning(
                f"Low memory ({available_gb:.2f}GB), skipping SMOTE to avoid crashing."
            )
            return features, target
        
        smote_kwargs = {'random_state': SMOTE_RANDOM_STATE, 'sampling_strategy': 'auto'}
        if len(features) > LARGE_DATASET_THRESHOLD_FOR_SMOTE:
            logger.info("Dataset is large, using reduced k_neighbors for SMOTE.")
            smote_kwargs['k_neighbors'] = 3
        
        smote = SMOTE(**smote_kwargs)
        result = smote.fit_resample(features, target)
        
        if isinstance(result, tuple) and len(result) == 2:
            features_resampled, target_resampled = result
        else:
            logger.error(
                "SMOTE did not return a tuple of (features, target). "
                "Skipping resampling."
            )
            return features, target
        
        logger.info(f"SMOTE applied. New data shape: {features_resampled.shape}")
        logger.debug(
            f"Class distribution after resampling: "
            f"{pd.Series(target_resampled).value_counts().to_dict()}"
        )
        gc.collect()
        
        if not isinstance(features_resampled, pd.DataFrame):
            features_resampled = pd.DataFrame(
                features_resampled, columns=features.columns
            )
        if not isinstance(target_resampled, pd.Series):
            target_resampled = pd.Series(target_resampled, name='target')
        
        # Ensure target remains numeric
        target_resampled = pd.to_numeric(target_resampled, errors='coerce').astype(int)
        
        return features_resampled, target_resampled
    except (ValueError, RuntimeError, MemoryError) as e:
        logger.error(
            "SMOTE failed: %s. Training will continue with original data.",
            e,
            exc_info=True
        )
        return features, target

def _create_model_and_weights(target_resampled: pd.Series) -> RandomForestClassifier:
    """Create Random Forest model and compute class weights.

    Args:
        target_resampled (pd.Series): Resampled target variable.

    Returns:
        RandomForestClassifier: Model with computed class weights.
    """
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(target_resampled),
        y=target_resampled
    )
    weight_dict = {
        int(cls): weight
        for cls, weight in zip(np.unique(target_resampled), class_weights)
    }
    logger.debug(f"Computed class weights: {weight_dict}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight=weight_dict,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=5
    )
    return model
