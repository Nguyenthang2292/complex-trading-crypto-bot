import gc
import joblib
import logging
import numpy as np
import os
import pandas as pd
import pandas_ta as ta
import psutil
import sys
import time
from datetime import datetime
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities._logger import setup_logging
logger = setup_logging(module_name="signals_random_forest", log_level=logging.INFO)

from components.config import (
    BB_STD_MULTIPLIER, BUY_THRESHOLD, 
    COL_BB_LOWER, COL_BB_UPPER, COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN,
    CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLDS, DEFAULT_WINDOW_SIZE,
    LARGE_DATASET_THRESHOLD_FOR_SMOTE, MACD_FAST_PERIOD, MACD_SIGNAL_PERIOD,
    MACD_SLOW_PERIOD, MAX_TRAINING_ROWS, MIN_DATA_POINTS, MIN_MEMORY_GB,
    MIN_TRAINING_SAMPLES, MODEL_FEATURES, MODEL_RANDOM_STATE, MODEL_TEST_SIZE,
    MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME, RSI_PERIOD, SELL_THRESHOLD,
    SIGNAL_LONG, SIGNAL_NEUTRAL, SIGNAL_SHORT, SMA_PERIOD
)

SMOTE_RANDOM_STATE = MODEL_RANDOM_STATE

def load_random_forest_model(model_path: Optional[Path] = None) -> Optional[RandomForestClassifier]:
    """
    Load a Random Forest model from disk using joblib.
    
    Args:
        model_path: Path to model file. If None, uses default path.
        
    Returns:
        Loaded RandomForestClassifier model or None if loading fails.
    """
    model_path = model_path or MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
    
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        logger.model(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def _calculate_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for trading signals.
    
    Args:
        df_input: DataFrame with OHLC data
        
    Returns:
        DataFrame with calculated technical indicators
    """
    if df_input.empty or COL_CLOSE not in df_input.columns:
        logger.warning("Input DataFrame is empty or missing close column")
        return pd.DataFrame()
    
    df = df_input.copy()
    
    try:
        # RSI calculation
        df['rsi'] = ta.rsi(df[COL_CLOSE], length=RSI_PERIOD)
        
        # MACD calculation
        macd_output = ta.macd(df[COL_CLOSE], fast=MACD_FAST_PERIOD, slow=MACD_SLOW_PERIOD, signal=MACD_SIGNAL_PERIOD)
        if macd_output is not None and len(macd_output.columns) >= 2:
            df['macd'], df['macd_signal'] = macd_output.iloc[:, 0], macd_output.iloc[:, 1]
        else:
            df['macd'] = df['macd_signal'] = np.nan

        # Bollinger Bands calculation
        close_series = df[COL_CLOSE]
        sma = close_series.rolling(window=DEFAULT_WINDOW_SIZE, min_periods=1).mean()
        std = close_series.rolling(window=DEFAULT_WINDOW_SIZE, min_periods=1).std()
        df[COL_BB_UPPER] = sma + (std * BB_STD_MULTIPLIER)
        df[COL_BB_LOWER] = sma - (std * BB_STD_MULTIPLIER)

        # SMA calculation
        df['ma_20'] = ta.sma(df[COL_CLOSE], length=SMA_PERIOD)

        return df.dropna()
        
    except Exception as e:
        logger.error(f"Error in feature calculation: {e}")
        return pd.DataFrame()

def train_random_forest_model(df_input: pd.DataFrame, save_model: bool = True) -> Optional[RandomForestClassifier]:
    """
    Train a Random Forest model for trading signal prediction.
    
    Args:
        df_input: Input DataFrame with OHLC data
        save_model: Whether to save the trained model to disk
        
    Returns:
        Trained RandomForestClassifier model or None if training fails
    """
    if df_input is None or df_input.empty:
        logger.error("Input DataFrame is None or empty")
        return None

    # Memory check
    available_gb = psutil.virtual_memory().available / (1024**3)
    logger.memory(f"Available memory: {available_gb:.2f}GB")
    
    if available_gb < MIN_MEMORY_GB:
        logger.error(f"Insufficient memory: only {available_gb:.2f}GB available")
        return None

    df_full = df_input.copy()
    logger.data(f"Input dataset shape: {df_full.shape}")
    
    # Limit dataset size to prevent memory issues
    if len(df_full) > MAX_TRAINING_ROWS:
        logger.warning(f"Dataset too large ({len(df_full)} rows), sampling {MAX_TRAINING_ROWS} rows")
        df_full = df_full.sample(n=MAX_TRAINING_ROWS, random_state=MODEL_RANDOM_STATE)
        gc.collect()

    # Standardize column names
    column_mapping = {'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low'}
    for old_col, new_col in column_mapping.items():
        if old_col in df_full.columns:
            df_full[new_col] = df_full[old_col]

    required_ohlc_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]
    if not all(col in df_full.columns for col in required_ohlc_cols):
        logger.error(f"Input DataFrame must contain OHLC columns: {required_ohlc_cols}")
        return None

    logger.data("Cleaning data...")
    df_full.dropna(subset=required_ohlc_cols, inplace=True)
    
    logger.analysis("Calculating features...")
    df_with_features = _calculate_features(df_full)

    if df_with_features.empty:
        logger.error("DataFrame is empty after feature calculation")
        return None

    logger.analysis(f"Features calculated. Dataset shape: {df_with_features.shape}")

    # Create target variable based on future price change
    logger.analysis("Creating target variables...")
    try:
        future_return = df_with_features[COL_CLOSE].shift(-5) / df_with_features[COL_CLOSE] - 1
        df_with_features['target'] = np.where(
            future_return > BUY_THRESHOLD, 1,
            np.where(future_return < SELL_THRESHOLD, -1, 0)
        )
    except Exception as e:
        logger.error(f"Error creating target variable: {e}")
        return None
    
    df_with_features.dropna(subset=['target'] + MODEL_FEATURES, inplace=True)
    if len(df_with_features) < MIN_TRAINING_SAMPLES:
        logger.error(f"Insufficient training samples: {len(df_with_features)} < {MIN_TRAINING_SAMPLES}")
        return None

    X, y = df_with_features[MODEL_FEATURES], df_with_features['target']
    logger.model(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.analysis(f"Class distribution: {y.value_counts().sort_index().to_dict()}")
    
    # Check for class imbalance
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        logger.error("Only one class present in target variable")
        return None
    
    # Memory check before SMOTE
    available_gb = psutil.virtual_memory().available / (1024**3)
    logger.memory(f"Memory before SMOTE: {available_gb:.2f}GB")
    
    # Apply SMOTE for class balancing with memory safeguards
    X_resampled, y_resampled = X, y
    try:
        logger.data("Applying SMOTE resampling...")
        smote_kwargs = {'random_state': SMOTE_RANDOM_STATE, 'sampling_strategy': 'auto'}
        if len(X) > LARGE_DATASET_THRESHOLD_FOR_SMOTE:
            logger.warning("Dataset large for SMOTE, using k_neighbors=3 for large dataset")
            smote_kwargs['k_neighbors'] = 3
        
        smote = SMOTE(**smote_kwargs)
        smote_result = smote.fit_resample(X, y)
        X_resampled, y_resampled = smote_result[:2]
        logger.data(f"SMOTE completed. New shape: X={X_resampled.shape}, y={y_resampled.shape}")
        gc.collect()
        
    except Exception as e:
        logger.error(f"SMOTE failed: {e}. Using original data without resampling.")
        X_resampled, y_resampled = X, y
    
    # Train/test split
    logger.data("Splitting data...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=MODEL_TEST_SIZE, 
            random_state=MODEL_RANDOM_STATE, shuffle=True
        )
        logger.data(f"Train/test split: train={X_train.shape[0]}, test={X_test.shape[0]}")
    except Exception as e:
        logger.error(f"Error during train/test split: {e}")
        return None
        
    # Compute class weights for balanced training
    logger.analysis("Computing class weights...")
    weight_dict = None
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        weight_dict = {int(cls): weight for cls, weight in zip(np.unique(y_resampled), class_weights)}
        logger.analysis(f"Class weights: {weight_dict}")
    except Exception as e:
        logger.error(f"Error computing class weights: {e}")
    
    # Train Random Forest model
    logger.model("Training Random Forest model...")
    try:
        model = RandomForestClassifier(
            n_estimators=MIN_DATA_POINTS, 
            random_state=MODEL_RANDOM_STATE,
            class_weight=weight_dict,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.success("Model training completed")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None

    # Evaluate model performance
    logger.performance("Evaluating model...")
    try:
        evaluate_model_with_confidence(model, X_test, y_test)
    except Exception as e:
        logger.warning(f"Error during model evaluation: {e}")

    # Save model if requested
    if save_model:
        model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
        try:
            joblib.dump(model, model_path)
            logger.model(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    # Memory cleanup
    try:
        del X_resampled, y_resampled, X_train, X_test, y_train, y_test
        gc.collect()
    except:
        pass
    
    logger.success("Model training process completed successfully")
    return model

def evaluate_model_with_confidence(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate model performance across different confidence thresholds.
    
    Args:
        model: Trained RandomForestClassifier model
        X_test: Test features DataFrame
        y_test: Test targets Series
        
    Returns:
        None
    """
    y_proba: np.ndarray = model.predict_proba(X_test)
    
    logger.performance("CONFIDENCE THRESHOLD EVALUATION")
    for threshold in CONFIDENCE_THRESHOLDS:
        y_pred_confident: np.ndarray = apply_confidence_threshold(y_proba, threshold, model.classes_)
        calculate_and_display_metrics(y_test.to_numpy(), y_pred_confident, threshold)

def apply_confidence_threshold(y_proba: np.ndarray, threshold: float, classes: np.ndarray) -> np.ndarray:
    """
    Apply confidence threshold to model predictions.
    
    Args:
        y_proba: Prediction probabilities array of shape (n_samples, n_classes)
        threshold: Confidence threshold (0.0 to 1.0)
        classes: Model classes array
        
    Returns:
        Array of predictions with confidence filtering applied
    """
    predictions = []
    for proba_row in y_proba:
        max_confidence = np.max(proba_row)
        predicted_class = classes[np.argmax(proba_row)] if max_confidence >= threshold else 0
        predictions.append(predicted_class)
    return np.array(predictions)

def calculate_and_display_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
    """
    Calculate and display trading performance metrics.
    
    Args:
        y_true: True labels array
        y_pred: Predicted labels array
        threshold: Confidence threshold used
        
    Returns:
        None
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    signal_counts = pd.Series(y_pred).value_counts().sort_index()
    total_signals = len(y_pred)
    logger.performance(f"Threshold {threshold:.1%}: Acc={accuracy:.3f}, P={precision_macro:.3f}, R={recall_macro:.3f}, F1={f1_macro:.3f}")
    
    signal_names = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}
    for signal, count in signal_counts.items():
        try:
            # Check if signal can be converted to int
            if isinstance(signal, (int, float)) or (isinstance(signal, str) and signal.isdigit()):
                signal_key = int(signal)
                signal_name = signal_names.get(signal_key, str(signal))
            else:
                signal_name = str(signal)
        except (ValueError, TypeError):
            signal_name = str(signal)
        percentage = (count / total_signals) * 100
        logger.performance(f"  {signal_name}: {count} ({percentage:.1f}%)")

def get_latest_random_forest_signal(df_market_data: pd.DataFrame, model: RandomForestClassifier) -> str:
    """
    Generate trading signal from latest market data using trained model.
    
    Args:
        df_market_data: DataFrame containing market OHLC data
        model: Trained RandomForestClassifier model
        
    Returns:
        Trading signal: SIGNAL_LONG, SIGNAL_SHORT, or SIGNAL_NEUTRAL
    """
    if df_market_data.empty:
        logger.warning("Input DataFrame for signal generation is empty")
        return SIGNAL_NEUTRAL
    
    required_input_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]
    if not all(col in df_market_data.columns for col in required_input_cols):
        logger.warning("Input DataFrame is missing OHLC columns")
        return SIGNAL_NEUTRAL

    df_with_features = _calculate_features(df_market_data.copy()) 
    if df_with_features.empty:
        logger.warning("DataFrame became empty after feature calculation")
        return SIGNAL_NEUTRAL

    # Get latest features for prediction
    latest_features_row = df_with_features[MODEL_FEATURES].iloc[-1:]
    if latest_features_row.isnull().values.any() or latest_features_row.empty:
        logger.warning("Latest features contain NaNs or empty")
        return SIGNAL_NEUTRAL

    # Generate prediction with confidence check
    try:
        prediction_proba = model.predict_proba(latest_features_row)[0]
        max_confidence = np.max(prediction_proba)
        predicted_class = model.classes_[np.argmax(prediction_proba)]
        
        logger.analysis(f"Prediction confidence: {max_confidence:.3f}, threshold: {CONFIDENCE_THRESHOLD:.3f}")
        
        # Return signal only when confidence exceeds threshold
        if max_confidence >= CONFIDENCE_THRESHOLD:
            signal_map = {1: SIGNAL_LONG, -1: SIGNAL_SHORT, 0: SIGNAL_NEUTRAL}
            signal = signal_map.get(predicted_class, SIGNAL_NEUTRAL)
            logger.signal(f"HIGH CONFIDENCE {signal} signal ({max_confidence:.1%})")
            return signal
        else:
            logger.signal(f"LOW CONFIDENCE - Returning NEUTRAL (confidence: {max_confidence:.1%})")
            return SIGNAL_NEUTRAL
            
    except Exception as e:
        logger.error(f"Error in signal prediction: {e}")
        return SIGNAL_NEUTRAL

def train_and_save_global_rf_model(combined_df: pd.DataFrame, model_filename: Optional[str] = None) -> Tuple[Optional[RandomForestClassifier], str]:
    """
    Train and save a Random Forest model from combined market data.
    
    Args:
        combined_df: Combined DataFrame with market data from multiple sources
        model_filename: Optional filename for saved model
        
    Returns:
        Tuple of (trained model, model file path)
    """
    start_time = time.time()

    try:
        # Generate timestamped filename if not provided
        if model_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M") 
            model_filename = f"rf_model_{timestamp}.joblib"

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / model_filename

        # Train model without automatic saving
        logger.model("Training Random Forest model...")
        model = train_random_forest_model(combined_df, save_model=False)

        if model is None:
            logger.error("Model training failed")
            return None, ""

        # Save trained model
        joblib.dump(model, model_path)
        elapsed_time = time.time() - start_time
        logger.success(f"Model trained and saved in {elapsed_time:.2f}s: {model_path}")

        return model, str(model_path)

    except Exception as e:
        logger.error(f"Error during Random Forest model training: {e}")
        return None, ""
