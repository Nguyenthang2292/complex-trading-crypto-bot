"""Random Forest model for trading signal prediction.

This module provides functionality for training and using Random Forest models
to predict trading signals based on technical indicators. It includes data
preprocessing, feature engineering, model training, and signal generation.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, cast
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components._generate_indicator_features import generate_indicator_features
from components.config import (
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLDS,
    MAX_TRAINING_ROWS,
    MODEL_FEATURES,
    MODEL_RANDOM_STATE,
    MODEL_TEST_SIZE,
    MODELS_DIR,
    RANDOM_FOREST_MODEL_FILENAME,
    SIGNAL_LONG,
    SIGNAL_NEUTRAL,
    SIGNAL_SHORT
)
from signals.signals_random_forest_helpers import (
    _apply_smote,
    _create_model_and_weights,
    _prepare_training_data
)
from utilities._logger import setup_logging

logger = setup_logging(module_name="signals_random_forest", log_level=logging.INFO)

SMOTE_RANDOM_STATE = MODEL_RANDOM_STATE

def load_random_forest_model(model_path: Optional[Path] = None) -> Optional[RandomForestClassifier]:
    """Load a trained Random Forest model from a file.

    Args:
        model_path (Optional[Path]): Path to the saved model. If None, uses default path.

    Returns:
        Optional[RandomForestClassifier]: Loaded model or None if not found or error.
    """
    if model_path is None:
        model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
    if not model_path.exists():
        logger.error(f"Model file not found at: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from: {model_path}")
        return model
    except (OSError, IOError, ValueError) as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None

def train_random_forest_model(df_input: pd.DataFrame, save_model: bool = True) -> Optional[RandomForestClassifier]:
    """Train and save Random Forest model for trading signal prediction.

    Args:
        df_input (pd.DataFrame): Input OHLCV data.
        save_model (bool): If True, save model after training.

    Returns:
        Optional[RandomForestClassifier]: Trained model or None if error.
    """
    if df_input is None or df_input.empty:
        logger.error("Input DataFrame for training is None or empty.")
        return None
    df_processed = df_input.copy()
    if len(df_processed) > MAX_TRAINING_ROWS:
        logger.warning(
            f"Dataset too large ({len(df_processed)} rows), "
            f"sampling down to {MAX_TRAINING_ROWS}."
        )
        df_processed = df_processed.sample(
            n=MAX_TRAINING_ROWS,
            random_state=MODEL_RANDOM_STATE
        )
    prepared_data = _prepare_training_data(df_processed)
    if prepared_data is None:
        return None
    features, target = prepared_data
    if len(target.value_counts()) < 2:
        logger.error("Cannot train model with only one class present in the target variable.")
        return None
    features_resampled, target_resampled = _apply_smote(features, target)
    features_train, features_test, target_train, target_test = train_test_split(
        features_resampled, target_resampled, test_size=MODEL_TEST_SIZE, random_state=MODEL_RANDOM_STATE
    )
    if not isinstance(features_test, pd.DataFrame):
        features_test = pd.DataFrame(features_test, columns=pd.Index(MODEL_FEATURES))
    if not isinstance(target_test, pd.Series):
        target_test = pd.Series(target_test, name='target')
    logger.info(
        f"Data split into training ({len(features_train)}) and "
        f"testing ({len(features_test)}) sets."
    )
    logger.info("Computing class weights for model training...")
    model = _create_model_and_weights(target_resampled)
    logger.info("Training the Random Forest model...")
    try:
        model.fit(features_train, target_train)
        logger.info("Model training completed successfully.")
    except (ValueError, RuntimeError) as e:
        logger.error(
            "An error occurred during model.fit: %s", e, exc_info=True
        )
        return None
    evaluate_model_with_confidence(model, features_test, target_test)
    if save_model:
        try:
            model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
            joblib.dump(model, model_path)
            logger.info(f"Model successfully saved to: {model_path}")
        except (OSError, IOError) as e:
            logger.error(f"Error saving model to {model_path}: {e}", exc_info=True)
    return model

def evaluate_model_with_confidence(
    model: RandomForestClassifier, features_test: pd.DataFrame, target_test: pd.Series
) -> None:
    """Evaluate the model's performance at various confidence thresholds."""
    logger.info("Evaluating model performance with different confidence thresholds...")
    y_proba = model.predict_proba(features_test)
    # Ensure y_proba is numpy.ndarray and model.classes_ is numpy.ndarray
    y_proba = np.asarray(y_proba)
    classes = np.asarray(model.classes_)
    for threshold in CONFIDENCE_THRESHOLDS:
        y_pred = apply_confidence_threshold(y_proba, threshold, classes)
        calculate_and_display_metrics(target_test, y_pred, threshold)

def apply_confidence_threshold(
    y_proba: np.ndarray, threshold: float, classes: np.ndarray
) -> np.ndarray:
    """Apply a confidence threshold to prediction probabilities.

    If the highest probability for a prediction is below the threshold, the
    prediction is set to neutral (0).

    Args:
        y_proba: The prediction probabilities from the model.
        threshold: The minimum confidence required to make a non-neutral prediction.
        classes: The class labels from the classifier.

    Returns:
        An array of predictions adjusted for the confidence threshold.
    """
    y_pred_confident = np.full(y_proba.shape[0], SIGNAL_NEUTRAL)
    max_proba = y_proba.max(axis=1)
    high_confidence_mask = max_proba >= threshold
    if np.any(high_confidence_mask):
        pred_indices = np.argmax(y_proba[high_confidence_mask], axis=1)
        y_pred_confident[high_confidence_mask] = classes[pred_indices]
    return y_pred_confident

def calculate_and_display_metrics(
    y_true: pd.Series, y_pred: np.ndarray, threshold: float
) -> None:
    """Calculate and logs performance metrics for a given confidence threshold.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        threshold: The confidence threshold used for the predictions.
    """
    labels = np.unique(np.concatenate((y_true.to_numpy(), y_pred)))
    precision = precision_score(
        y_true, y_pred, average='weighted', labels=labels, zero_division="warn"
    )
    recall = recall_score(
        y_true, y_pred, average='weighted', labels=labels, zero_division="warn"
    )
    f1 = f1_score(
        y_true, y_pred, average='weighted', labels=labels, zero_division="warn"
    )
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(
        f"Metrics @ {threshold:.2f} threshold | "
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )

def get_latest_random_forest_signal(
    df_market_data: pd.DataFrame, model: RandomForestClassifier
) -> Tuple[str, float]:
    """Generate a trading signal for the most recent data point.

    Args:
        df_market_data: A DataFrame containing the latest market data.
        model: The trained RandomForestClassifier model.

    Returns:
        A tuple containing the signal string ('LONG', 'SHORT', 'NEUTRAL')
        and the confidence of the prediction.
    """
    if df_market_data.empty:
        logger.warning("Market data for signal generation is empty.")
        return SIGNAL_NEUTRAL, 0.0
    logger.info("Calculating features for the latest data point...")
    df_with_features = generate_indicator_features(df_market_data)
    if df_with_features.empty or not all(f in df_with_features.columns for f in MODEL_FEATURES):
        logger.warning(
            "Could not generate features for the latest data. "
            "Returning NEUTRAL."
        )
        return SIGNAL_NEUTRAL, 0.0
    latest_features = df_with_features[MODEL_FEATURES].iloc[-1:]
    try:
        prediction_proba = model.predict_proba(latest_features)[0]
        confidence = max(prediction_proba)
        predicted_class = model.classes_[np.argmax(prediction_proba)]
        if confidence < CONFIDENCE_THRESHOLD:
            signal = SIGNAL_NEUTRAL
        elif predicted_class == 1:
            signal = SIGNAL_LONG
        elif predicted_class == -1:
            signal = SIGNAL_SHORT
        else:
            signal = SIGNAL_NEUTRAL
        logger.info(
            f"Latest signal: {signal} with confidence {confidence:.4f} "
            f"(Class: {predicted_class})"
        )
        return signal, confidence
    except (ValueError, RuntimeError) as e:
        logger.error(
            f"Error during signal generation: {e}", exc_info=True
        )
        return SIGNAL_NEUTRAL, 0.0

def train_and_save_global_rf_model(
    combined_df: pd.DataFrame, model_filename: Optional[str] = None
) -> Tuple[Optional[RandomForestClassifier], str]:
    """Train a global Random Forest model on a combined dataset from multiple symbols.

    Args:
        combined_df: A DataFrame containing data from multiple trading symbols.
        model_filename: An optional filename for the saved model.

    Returns:
        A tuple containing the trained model and the path where it was saved.
        Returns (None, "") on failure.
    """
    if combined_df.empty:
        logger.error(
            "The combined DataFrame for global model training is empty."
        )
        return None, ""
    logger.info("Starting training for the global Random Forest model...")
    model = train_random_forest_model(combined_df, save_model=False)
    if model is None:
        logger.error(
            "Global model training failed."
        )
        return None, ""
    filename = (
        model_filename or f"rf_model_global_{datetime.now():%Y%m%d_%H%M}.joblib"
    )
    model_path = MODELS_DIR / filename
    try:
        joblib.dump(model, model_path)
        logger.info(
            f"Global Random Forest model successfully saved to: {model_path}"
        )
        return model, str(model_path)
    except (OSError, IOError) as e:
        logger.error(
            f"Error saving global model to {model_path}: {e}", exc_info=True
        )
        return None, ""
