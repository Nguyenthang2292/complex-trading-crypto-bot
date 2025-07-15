# -*- coding: utf-8 -*-
# ====================================================================
# ARIMAX-GARCH MODEL FOR CRYPTO TRADING SIGNALS
# ====================================================================

"""
ARIMAX-GARCH model for cryptocurrency trading signal generation.

This module provides a complete pipeline for training an ARIMAX-GARCH model
and using it to generate trading signals from cryptocurrency market data.
It integrates feature engineering, model fitting, and signal interpretation.

Key Features:
- ARIMAX model for price forecasting with technical indicators as exogenous variables
- GARCH model for volatility forecasting
- Integration with existing technical indicator pipeline
- Comprehensive signal generation with confidence scoring
- Safe model saving/loading with PyTorch 2.6+ compatibility

Usage:
    # Train ARIMAX-GARCH model
    model, path = train_and_save_arimax_garch_model(df_market_data)
    
    # Load trained model
    model_data = load_arimax_garch_model("path/to/model.pth")
    
    # Generate trading signal
    signal = get_latest_arimax_garch_signal(df_new_data, model_data)
"""

import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from .arimax_garch import ArimaxGarchModel
from utilities._logger import setup_logging
from components.config import (
    COL_CLOSE, GARCH_P, GARCH_Q, MIN_DATA_POINTS, MODELS_DIR,
    RISK_FREE_RATE, SIGNAL_CONFIDENCE_THRESHOLD, SIGNAL_LONG,
    SIGNAL_NEUTRAL, SIGNAL_SHORT, STRONG_SIGNAL_THRESHOLD,
    WEAK_SIGNAL_THRESHOLD
)

logger = setup_logging(module_name="signals_arimax_garch", log_level=logging.DEBUG)

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def safe_save_model(checkpoint: Dict[str, Any], model_path: str) -> bool:
    """Saves a model checkpoint to a file using joblib.

    Args:
        checkpoint (Dict[str, Any]): The model checkpoint dictionary to save.
        model_path (str): The file path where the model will be saved.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(checkpoint, model_path)
        logger.debug(f"Model saved successfully to {model_path}")
        return True
    except (IOError, pickle.PickleError) as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        return False

def safe_load_model(model_path: str) -> Optional[Dict[str, Any]]:
    """Loads a model checkpoint from a file using joblib.

    Args:
        model_path (str): The file path from which to load the model.

    Returns:
        Optional[Dict[str, Any]]: The loaded model checkpoint, or None if loading fails.
    """
    try:
        checkpoint = joblib.load(model_path)
        logger.debug(f"Model loaded successfully from {model_path}")
        return checkpoint
    except (IOError, pickle.PickleError) as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

# ====================================================================
# SIGNAL GENERATION FUNCTIONS
# ====================================================================

def generate_arimax_garch_signals(
    forecast_df: pd.DataFrame,
    current_price: float,
) -> pd.DataFrame:
    """Generates trading signals based on ARIMAX-GARCH forecast results.

    This function calculates expected return, risk (from volatility), and the
    Sharpe ratio to determine the signal's strength and direction.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing price and volatility forecasts.
        current_price (float): The current price to calculate returns against.

    Returns:
        pd.DataFrame: A DataFrame containing detailed trading signals for each
                      forecast step. Returns an empty DataFrame on failure.
    """
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        logger.warning("Forecast DataFrame is empty or invalid.")
        return pd.DataFrame()
    if not isinstance(current_price, (int, float)) or current_price <= 0:
        logger.warning(f"Invalid current_price: {current_price}. Must be a positive number.")
        return pd.DataFrame()

    signals = []
    for i, (_, row) in enumerate(forecast_df.iterrows()):
        forecast_price = row['Price_Forecast']
        volatility = row['Volatility_Forecast']

        expected_return = (forecast_price - current_price) / current_price
        risk = volatility / current_price
        sharpe = (expected_return - RISK_FREE_RATE) / risk if risk > 0 else 0

        if expected_return > STRONG_SIGNAL_THRESHOLD and sharpe > 0.5:
            signal, confidence = 'STRONG_BUY', min(float(abs(sharpe)) * 20, 100)
        elif expected_return > WEAK_SIGNAL_THRESHOLD and sharpe > 0.2:
            signal, confidence = 'BUY', min(float(abs(sharpe)) * 15, 80)
        elif expected_return < -STRONG_SIGNAL_THRESHOLD and sharpe < -0.5:
            signal, confidence = 'STRONG_SELL', min(float(abs(sharpe)) * 20, 100)
        elif expected_return < -WEAK_SIGNAL_THRESHOLD and sharpe < -0.2:
            signal, confidence = 'SELL', min(float(abs(sharpe)) * 15, 80)
        else:
            signal, confidence = 'HOLD', 50

        signals.append({
            'Step': i + 1,
            'Current_Price': current_price,
            'Forecast_Price': forecast_price,
            'Expected_Return': expected_return * 100,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Signal': signal,
            'Confidence': confidence
        })

    return pd.DataFrame(signals)

# ====================================================================
# MAIN TRAINING AND PREDICTION FUNCTIONS
# ====================================================================

def train_and_save_arimax_garch_model(
    df_input: pd.DataFrame,
    model_filename: Optional[str] = None
) -> Tuple[Optional[ArimaxGarchModel], str]:
    """Trains an ARIMAX-GARCH model and saves it to a file.

    Args:
        df_input (pd.DataFrame): The input DataFrame containing market data.
        model_filename (Optional[str]): The filename for the saved model.
            If None, a timestamped name is generated.

    Returns:
        Tuple[Optional[ArimaxGarchModel], str]: A tuple containing the trained
            model instance and the path to the saved model file. Returns
            (None, "") on failure.
    """
    start_time = time.time()
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        logger.error("Input DataFrame is empty or invalid.")
        return None, ""

    if len(df_input) < MIN_DATA_POINTS:
        logger.error(f"Insufficient data: {len(df_input)} rows, requires at least {MIN_DATA_POINTS}.")
        return None, ""

    logger.model(f"Training ARIMAX-GARCH model with {len(df_input)} rows of data.")
    
    model = ArimaxGarchModel()
    df_prepared = model.prepare_data(df_input)
    if df_prepared.empty:
        logger.error("Data preparation failed, stopping training.")
        return None, ""

    if not model.fit_arimax_model(df_prepared):
        logger.error("ARIMAX model fitting failed.")
        return None, ""

    if not model.fit_garch_model(use_arimax_residuals=True):
        logger.error("GARCH model fitting failed.")
        return None, ""

    if model_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_filename = f"arimax_garch_model_{timestamp}.joblib"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(MODELS_DIR / model_filename)

    checkpoint = {
        'arimax_results': model.arimax_results,
        'garch_results': model.garch_results,
        'exog_cols': model.exog_cols,
        'model_config': {
            'arimax_order': model.arimax_results.model.order if model.arimax_results else None,
            'garch_order': (GARCH_P, GARCH_Q),
            'training_samples': len(df_prepared),
        },
        'training_metadata': {
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': time.time() - start_time,
            'data_shape': df_prepared.shape
        }
    }

    if safe_save_model(checkpoint, model_path):
        elapsed_time = time.time() - start_time
        logger.success(f"ARIMAX-GARCH model trained and saved in {elapsed_time:.2f}s: {model_path}")
        return model, model_path
    
    logger.error("Failed to save the trained model.")
    return None, ""

def load_arimax_garch_model(model_path: str) -> Optional[Dict[str, Any]]:
    """Loads a trained ARIMAX-GARCH model from the specified path.

    Args:
        model_path (str): The full path to the saved model file (.joblib).

    Returns:
        Optional[Dict[str, Any]]: The loaded model data (checkpoint),
            or None if the file doesn't exist or loading fails.
    """
    if not model_path or not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        return None

    logger.debug(f"Attempting to load model from: {model_path}")
    return safe_load_model(model_path)

def get_latest_arimax_garch_signal(
    df_market_data: pd.DataFrame,
    model_data: Dict[str, Any],
    forecast_steps: int = 5
) -> str:
    """Generates a trading signal using a pre-trained ARIMAX-GARCH model.

    Args:
        df_market_data (pd.DataFrame): DataFrame with recent market data for prediction.
        model_data (Dict[str, Any]): The loaded model checkpoint dictionary.
        forecast_steps (int): The number of future steps to forecast.

    Returns:
        str: The final trading signal ('LONG', 'SHORT', 'NEUTRAL').
    """
    if not isinstance(df_market_data, pd.DataFrame) or df_market_data.empty:
        logger.warning("Market data for prediction is empty or invalid.")
        return SIGNAL_NEUTRAL
    
    if not isinstance(model_data, dict):
        logger.error("Model data is invalid or not a dictionary.")
        return SIGNAL_NEUTRAL

    arimax_results = model_data.get('arimax_results')
    garch_results = model_data.get('garch_results')
    
    if arimax_results is None or garch_results is None:
        logger.error("Model data is missing ARIMAX or GARCH results.")
        return SIGNAL_NEUTRAL

    # Use a mock model object for forecasting to avoid re-training.
    mock_model = ArimaxGarchModel()
    mock_model.arimax_results = arimax_results
    mock_model.garch_results = garch_results
    mock_model.exog_cols = model_data.get('exog_cols', [])

    df_prepared = mock_model.prepare_data(df_market_data)
    if df_prepared.empty:
        logger.warning("Data preparation for signal generation failed.")
        return SIGNAL_NEUTRAL

    forecast_df = mock_model.forecast(steps=forecast_steps)
    if forecast_df.empty:
        logger.warning("Forecast generation failed.")
        return SIGNAL_NEUTRAL

    current_price = df_prepared[COL_CLOSE].iloc[-1]
    signals_df = generate_arimax_garch_signals(forecast_df, current_price)

    if signals_df.empty:
        logger.warning("Signal interpretation failed.")
        return SIGNAL_NEUTRAL

    # Use the signal from the first forecast step.
    first_signal = signals_df.iloc[0]
    signal_type = first_signal['Signal']
    confidence = first_signal['Confidence']
    
    if confidence < SIGNAL_CONFIDENCE_THRESHOLD:
        final_signal = SIGNAL_NEUTRAL
    elif signal_type in ['STRONG_BUY', 'BUY']:
        final_signal = SIGNAL_LONG
    elif signal_type in ['STRONG_SELL', 'SELL']:
        final_signal = SIGNAL_SHORT
    else:
        final_signal = SIGNAL_NEUTRAL

    logger.signal(
        f"ARIMAX-GARCH Signal: {final_signal} (Type: {signal_type}, "
        f"Confidence: {confidence:.1f}%, Threshold: {SIGNAL_CONFIDENCE_THRESHOLD}%)"
    )
    return final_signal 