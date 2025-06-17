import joblib
import logging
import os
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Union, Dict, Tuple

# Ensure the project root (the parent of 'livetrade') is in sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
signals_dir = os.path.dirname(current_dir)            
project_root = os.path.dirname(signals_dir)           
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities._logger import setup_logging
logger = setup_logging(module_name="process_signals_random_forest", log_level=logging.INFO)

from livetrade._components._combine_all_dataframes import combine_all_dataframes
from livetrade.config import (
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NEUTRAL
)
from signals.signals_random_forest import (
    load_random_forest_model,
    get_latest_random_forest_signal,
    train_and_save_global_rf_model,
)

DATAFRAME_COLUMNS = ['Pair', 'SignalTimeframe', 'SignalType']

def load_latest_rf_model(models_dir: str) -> Tuple[Optional[RandomForestClassifier], Optional[str]]:
    """
    Find the latest Random Forest model matching rf_model_*.joblib in the models_dir directory.
    This function will NOT create a new model if none is found.

    Args:
        models_dir: Path to the models directory.

    Returns:
        Tuple (model, model_path): The loaded model and the path to the model file.
                                Returns (None, None) if not found or failed to load.
    """
    models_path = Path(models_dir)
    if not models_path.is_dir():
        logger.data(f"Models directory does not exist: {models_dir}")
        return None, None

    model_files = list(models_path.glob("rf_model_*.joblib"))
    
    default_link_path = models_path / "random_forest_model.joblib"
    potential_model_files = [f for f in model_files if f.resolve() != default_link_path.resolve()]

    model = None
    model_path_to_load = None
    
    if potential_model_files:
        # Get the newest file by creation time (ctime) among files with timestamp
        latest_timestamped_model_file = max(potential_model_files, key=lambda p: p.stat().st_ctime)
        model_path_to_load = latest_timestamped_model_file
        logger.model(f"Found latest model based on timestamp: {latest_timestamped_model_file.name}")
    elif default_link_path.exists() and default_link_path.is_file():
        
        # If no timestamped file, try loading from the default link
        model_path_to_load = default_link_path
        logger.model(f"No timestamped model found, trying to load from default link: {default_link_path.name}")
    else:
        logger.model(f"No Random Forest model found in directory: {models_dir}")
        return None, None
        
    if model_path_to_load:
        try:
            model = joblib.load(model_path_to_load)
            logger.success(f"Successfully loaded model from: {model_path_to_load}")
            return model, str(model_path_to_load)
        except Exception as e:
            logger.error(f"Error loading model from {model_path_to_load}: {e}")
            return None, None

    # This case should not occur if the above logic is correct, but for safety:
    logger.data(f"Could not determine model to load in directory: {models_dir}")
    return None, None

def process_signals_random_forest(
    preloaded_data: Dict[str, Dict[str, pd.DataFrame]],
    timeframes_to_scan: Optional[List[str]] = None,
    trained_model: Optional[RandomForestClassifier] = None,
    model_path: Optional[Union[str, Path]] = None,
    auto_train_if_missing: bool = False,
    include_long_signals: bool = True,
    include_short_signals: bool = False
) -> pd.DataFrame:
    """
    Processes trading signals for cryptocurrency symbols using a Random Forest model.
    Returns symbols with LONG and/or SHORT signals based on parameters.
    
    Args:
        preloaded_data (Dict[str, Dict[str, pd.DataFrame]]): Pre-loaded symbol data in format:
                                                        {symbol: {timeframe: dataframe}}
        timeframes_to_scan (Optional[List[str]]): Specific timeframes to scan, in order of priority. 
                                                Defaults to ['1h', '4h', '1d'].
        trained_model (Optional[RandomForestClassifier]): Pre-trained Random Forest model. If None,
                                                        it will be loaded from `model_path` or a new model 
                                                        will be trained if `auto_train_if_missing` is True.
        model_path (Optional[Union[str, Path]]): Path to the saved model file. If None and `trained_model` is None,
                                                a default location will be used.
        auto_train_if_missing (bool): If True and the model cannot be loaded, a new model will be trained automatically.
        include_long_signals (bool): If True, include LONG signals in the results. Default True.
        include_short_signals (bool): If True, include SHORT signals in the results. Default False.    Returns:
        pd.DataFrame: DataFrame containing symbols with signals, with columns ['Pair', 'SignalTimeframe', 'SignalType'].
    """
    logger.model("===============================================")
    logger.model("STARTING RANDOM FOREST SIGNAL ANALYSIS (CRYPTO)")
    logger.model("===============================================")

    # Validate input data
    if not preloaded_data or not isinstance(preloaded_data, dict):
        logger.error("No preloaded_data provided or invalid format. Expected Dict[str, Dict[str, pd.DataFrame]]")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
    
    # Validate signal type parameters
    if not include_long_signals and not include_short_signals:
        logger.error("At least one of include_long_signals or include_short_signals must be True")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)

    signal_types = []
    if include_long_signals:
        signal_types.append("LONG")
    if include_short_signals:
        signal_types.append("SHORT")
    
    logger.signal(f"Signal types to analyze: {', '.join(signal_types)}")

    # Initialize parameters with defaults
    actual_timeframes_to_scan = timeframes_to_scan if timeframes_to_scan is not None else ['1h', '4h', '1d']
    logger.config(f"Using {'default' if timeframes_to_scan is None else 'specified'} timeframes for RF scan (priority order): {actual_timeframes_to_scan}")
    
    if not actual_timeframes_to_scan:
        logger.warning("No timeframes specified or defaulted for RF scan. Cannot proceed.")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)

    symbols_to_analyze = list(preloaded_data.keys())
    logger.analysis(f"Analyzing {len(symbols_to_analyze)} crypto symbols from preloaded data.")

    # Prepare data for model training if needed
    combined_df = combine_all_dataframes(preloaded_data)
    
    # --- Model Loading/Training ---
    model = trained_model
    if model is None:
        if model_path is not None:
            logger.model(f"Attempting to load model from specified path: {model_path}")
            model = load_random_forest_model(Path(model_path) if isinstance(model_path, str) else model_path)
        else:
            logger.model("No model path specified, attempting to load from default location.")
            model = load_random_forest_model()
        
        if model is None and auto_train_if_missing:
            logger.model("No model found. auto_train_if_missing=True, training a new model...")
            if not combined_df.empty: 
                # Ensure there's data to train on
                trained_model_data, trained_model_path_str = train_and_save_global_rf_model(combined_df)
                if trained_model_data:
                    model = trained_model_data
                    logger.success(f"Successfully trained and saved new model to {trained_model_path_str}")
                else: 
                    logger.error("Model training initiated by auto_train_if_missing failed to produce a model.")
            else:
                logger.error("Cannot train new model: combined_df is empty.")
    
    if model is None: 
        logger.error("No Random Forest model available. Cannot generate any signals.")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
    
    logger.model("Using Random Forest model to generate signals.")
    
    # --- Signal Processing per Symbol ---
    signals_list = []

    for symbol in symbols_to_analyze: 
        logger.debug(f"Processing symbol: {symbol}")
        
        symbol_data_for_tfs = preloaded_data.get(symbol)
        if not symbol_data_for_tfs or not isinstance(symbol_data_for_tfs, dict):
            logger.data(f"  No data or incorrect data format for {symbol}. Skipping.")
            continue
        
        # Find signals on prioritized timeframes
        signal_found = False
        for tf_scan in actual_timeframes_to_scan:
            if signal_found:
                break  # Found a signal, no need to check other timeframes
                
            df_current_tf = symbol_data_for_tfs.get(tf_scan)

            if df_current_tf is None or df_current_tf.empty or len(df_current_tf) < 50:
                logger.data(f"  Insufficient data for {symbol} on {tf_scan} (need at least 50 candles)")
                continue

            try:
                signal = get_latest_random_forest_signal(df_current_tf, model)
                
                # Process LONG signals
                if signal == SIGNAL_LONG and include_long_signals:
                    signals_list.append({
                        'Pair': symbol,
                        'SignalTimeframe': tf_scan,
                        'SignalType': 'LONG'
                    })
                    logger.signal(f"  LONG signal found for {symbol} on {tf_scan}")
                    signal_found = True
                
                # Process SHORT signals
                elif signal == SIGNAL_SHORT and include_short_signals:
                    signals_list.append({
                        'Pair': symbol,
                        'SignalTimeframe': tf_scan,
                        'SignalType': 'SHORT'
                    })
                    logger.signal(f"  SHORT signal found for {symbol} on {tf_scan}")
                    signal_found = True
                
                # Log other signals for debugging
                else:
                    signal_type = "NEUTRAL" if signal == SIGNAL_NEUTRAL else str(signal)
                    logger.debug(f"  {signal_type} signal for {symbol} on {tf_scan} - not included based on parameters")
                    
            except Exception as e:
                logger.error(f"    Error getting RF signal for {symbol} on {tf_scan}: {e}")

    # Count signals by type
    long_count = sum(1 for s in signals_list if s['SignalType'] == 'LONG')
    short_count = sum(1 for s in signals_list if s['SignalType'] == 'SHORT')
    
    logger.model("===============================================")
    logger.success(f"COMPLETED RANDOM FOREST SIGNAL ANALYSIS")
    logger.signal(f"Found {long_count} LONG signals and {short_count} SHORT signals")
    logger.analysis(f"Total signals: {len(signals_list)}")
    logger.model("===============================================")

    return pd.DataFrame(signals_list) if signals_list else pd.DataFrame([], columns=DATAFRAME_COLUMNS)