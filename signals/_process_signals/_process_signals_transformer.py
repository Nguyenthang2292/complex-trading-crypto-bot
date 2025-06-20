import logging
import pandas as pd
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import numpy as np

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from livetrade.config import (
    MODELS_DIR, 
    TRANSFORMER_MODEL_FILENAME,
    DEFAULT_TIMEFRAMES
)
    
from signals.signals_transformer import (
    get_latest_transformer_signal,
    load_transformer_model,
    train_and_save_transformer_model,
    SIGNAL_LONG,
    SIGNAL_NEUTRAL,
    SIGNAL_SHORT
)
from livetrade._components._combine_all_dataframes import combine_all_dataframes
from utilities._gpu_resource_manager import get_gpu_resource_manager
from utilities._logger import setup_logging

logger = setup_logging(module_name="_process_signals_transformer", log_level=logging.DEBUG)

DATAFRAME_COLUMNS = ['Symbol', 'SignalTimeframe', 'SignalType']

def load_latest_transformer_model(models_dir: str) -> Tuple[Optional[Tuple], Optional[str]]:
    """
    Find the latest Transformer model matching transformer_model_*.pth in the models_dir directory.
    This function will NOT create a new model if none is found.

    Args:
        models_dir: Path to the models directory.

    Returns:
        Tuple (model_data, model_path): The loaded model data tuple (model, scaler, feature_cols, target_idx) 
                                      and the path to the model file.
                                      Returns (None, None) if not found or failed to load.
    """
    try:
        models_path = Path(models_dir)
        if not models_path.is_dir():
            logger.warning(f"Models directory does not exist: {models_dir}")
            return None, None

        # Look for both timestamped models and default model
        timestamped_files = list(models_path.glob("transformer_model_*.pth"))
        default_file = models_path / TRANSFORMER_MODEL_FILENAME
        
        # Filter out default file from timestamped files to avoid duplicates
        model_files = [f for f in timestamped_files if f.name != TRANSFORMER_MODEL_FILENAME]
        
        model_path_to_load = None
        
        if model_files:
            # Get the newest timestamped file
            try:
                latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
            except (AttributeError, OSError) as e:
                logger.warning(f"Error accessing file timestamps: {e}")
                latest_model_file = model_files[0]  # Use first available
            model_path_to_load = latest_model_file
            logger.model(f"Found latest timestamped transformer model: {latest_model_file.name}")
        elif default_file.exists():
            model_path_to_load = default_file
            logger.model(f"Using default transformer model: {default_file.name}")
        else:
            logger.model(f"No Transformer model found in directory: {models_dir}")
            return None, None
        
        # Load the model using the improved load function
        model_data = load_transformer_model(str(model_path_to_load))
        
        # Check if all components are loaded successfully
        if model_data and len(model_data) == 4 and model_data[0] is not None:
            logger.success(f"Successfully loaded transformer model from: {model_path_to_load}")
            return model_data, str(model_path_to_load)
        else:
            logger.error(f"Failed to load transformer model from: {model_path_to_load}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in load_latest_transformer_model: {e}")
        return None, None

def process_signals_transformer(
    preloaded_data: Dict[str, Dict[str, pd.DataFrame]],
    timeframes_to_scan: Optional[List[str]] = None,
    trained_model_data: Optional[Tuple] = None,
    model_path: Optional[Union[str, Path]] = None,
    auto_train_if_missing: bool = False,
    include_long_signals: bool = True,
    include_short_signals: bool = False
) -> pd.DataFrame:
    """
    Processes trading signals for cryptocurrency symbols using a Transformer model with GPU resource management.
    Returns symbols with LONG and/or SHORT signals based on parameters.
    
    Args:
        preloaded_data: Pre-loaded symbol data in format {symbol: {timeframe: dataframe}}
        timeframes_to_scan: Specific timeframes to scan, in order of priority. 
                          Defaults to ['1h', '4h', '1d'].
        trained_model_data: Pre-trained Transformer model data tuple 
                          (model, scaler, feature_cols, target_idx). If None,
                          it will be loaded from `model_path` or a new model 
                          will be trained if `auto_train_if_missing` is True.
        model_path: Path to the saved model file. If None and 
                   `trained_model_data` is None, a default location will be used.
        auto_train_if_missing: If True and the model cannot be loaded, a new model will be trained automatically.
        include_long_signals: If True, include LONG signals in the results. Default True.
        include_short_signals: If True, include SHORT signals in the results. Default False.
        
    Returns:
        DataFrame containing symbols with signals, with columns ['Symbol', 'SignalTimeframe', 'SignalType'].
    """
    gpu_manager = get_gpu_resource_manager()
    
    with gpu_manager.gpu_scope() as device:
        device_str = str(device) if device else 'cpu'
        logger.gpu(f"Using device for Transformer processing: {device_str}")
        
        if device:
            memory_info = gpu_manager.get_memory_info()
            logger.memory(f"GPU Memory - Total: {memory_info['total']//1024**2}MB, "
                         f"Allocated: {memory_info['allocated']//1024**2}MB, "
                         f"Cached: {memory_info['cached']//1024**2}MB")
    logger.analysis("===============================================")
    logger.analysis("STARTING TRANSFORMER SIGNAL ANALYSIS (CRYPTO)")
    logger.analysis("===============================================")

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
    actual_timeframes_to_scan = timeframes_to_scan if timeframes_to_scan is not None else DEFAULT_TIMEFRAMES[:3]  # ['1h', '4h', '1d']
    logger.config(f"Using {'default' if timeframes_to_scan is None else 'specified'} timeframes for Transformer scan (priority order): {actual_timeframes_to_scan}")
    
    if not actual_timeframes_to_scan:
        logger.warning("No timeframes specified or defaulted for Transformer scan. Cannot proceed.")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)

    symbols_to_analyze = list(preloaded_data.keys())
    logger.data(f"Analyzing {len(symbols_to_analyze)} crypto symbols from preloaded data.")

    # Prepare data for model training if needed
    combined_df = combine_all_dataframes(preloaded_data)        
    
    # --- Model Loading/Training with GPU resource management ---
    model_data = trained_model_data
        
    if model_data is None:
        if model_path is not None:
            logger.model(f"Attempting to load transformer model from specified path: {model_path}")
            try:
                model_data = load_transformer_model(str(model_path) if isinstance(model_path, Path) else model_path)
                if model_data and model_data[0] is None:
                    logger.warning("Model loaded but appears to be invalid")
                    model_data = None
            except Exception as e:
                logger.error(f"Error loading model from specified path: {e}")
                model_data = None
        else:
            logger.model("No model path specified, attempting to load from default location.")
            try:
                model_data, _ = load_latest_transformer_model(str(MODELS_DIR))
            except Exception as e:
                logger.error(f"Exception when loading model: {e}")
                model_data = None
        
        if (model_data is None or (model_data and model_data[0] is None)) and auto_train_if_missing:
            logger.model("No transformer model found. auto_train_if_missing=True, training a new model...")
            if not combined_df.empty: 
                try:
                    # Training with GPU resource management
                    trained_model, trained_model_path = train_and_save_transformer_model(combined_df)
                    if trained_model is not None and trained_model_path:
                        model_data = load_transformer_model(trained_model_path)
                        if model_data and model_data[0] is not None:
                            logger.success(f"Successfully trained and saved new transformer model to {trained_model_path}")
                        else:
                            logger.error("Failed to load newly trained model")
                            model_data = None
                    else: 
                        logger.error("Transformer model training failed to produce a model.")
                        model_data = None
                except Exception as e:
                    logger.error(f"Error during model training: {e}")
                    model_data = None
            else:
                logger.error("Cannot train new transformer model: combined_df is empty.")
    
    if model_data is None or (model_data and model_data[0] is None): 
        logger.error("No Transformer model available. Cannot generate any signals.")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
    
    # Unpack model data and move to appropriate device
    model, scaler, feature_cols, target_idx = model_data
    
    # Move model to GPU if available, with error handling
    try:
        if device:
            model.to(device)
            logger.gpu(f"Model successfully moved to {device}")
        else:
            model.to('cpu')
            logger.gpu("Model running on CPU")
    except Exception as e:
        logger.error(f"Error moving model to device: {e}")
        model.to('cpu')
        device_str = 'cpu'
        logger.gpu("Fallback to CPU due to GPU error")
    logger.model("Using Transformer model to generate signals.")
    
    # Add debugging for threshold values
    from livetrade.config import BUY_THRESHOLD, SELL_THRESHOLD
    logger.config(f"🔍 Signal generation thresholds:")
    logger.config(f"  • BUY_THRESHOLD: {BUY_THRESHOLD:.6f} ({BUY_THRESHOLD:.2%})")
    logger.config(f"  • SELL_THRESHOLD: {SELL_THRESHOLD:.6f} ({SELL_THRESHOLD:.2%})")
    
    # Run bias analysis once for the combined dataset to get suggested thresholds
    suggested_thresholds = None
    if combined_df is not None and not combined_df.empty:
        logger.analysis("Running one-time bias analysis on combined dataset...")
        try:
            from signals.signals_transformer import add_technical_indicators, analyze_model_bias_and_adjust_thresholds
            
            df_with_features = add_technical_indicators(combined_df)
            if not df_with_features.empty and len(df_with_features) > 100:
                suggested_buy, suggested_sell = analyze_model_bias_and_adjust_thresholds(
                    df_with_features, model, scaler, feature_cols, target_idx, device_str
                )
                
                # Validate suggested thresholds
                if (np.isfinite(suggested_buy) and np.isfinite(suggested_sell) and 
                    -0.1 <= suggested_sell <= 0.1 and -0.1 <= suggested_buy <= 0.1):
                    suggested_thresholds = (suggested_buy, suggested_sell)
                    logger.success(f"✅ Using suggested thresholds: BUY={suggested_buy:.6f}, SELL={suggested_sell:.6f}")
                else:
                    logger.warning(f"❌ Invalid suggested thresholds: BUY={suggested_buy:.6f}, SELL={suggested_sell:.6f}")
                    logger.warning("   Using default thresholds instead")
            else:
                logger.warning("Insufficient data for bias analysis, using default thresholds")
        except Exception as e:
            logger.error(f"Error in bias analysis: {e}")
            logger.model("Continuing with default thresholds")
    
    # --- Signal Processing per Symbol ---
    signals_list = []
    
    # Track signal statistics for debugging
    signal_stats = {
        'processed': 0,
        'long_generated': 0,
        'short_generated': 0,
        'neutral_generated': 0,
        'errors': 0,
        'insufficient_data': 0
    }

    for symbol in symbols_to_analyze: 
        logger.debug(f"Processing symbol: {symbol}")
        signal_stats['processed'] += 1
        
        symbol_data_for_tfs = preloaded_data.get(symbol)
        if not symbol_data_for_tfs or not isinstance(symbol_data_for_tfs, dict):
            logger.warning(f"  No data or incorrect data format for {symbol}. Skipping.")
            signal_stats['errors'] += 1
            continue
        
        # Find signals on prioritized timeframes
        signal_found = False
        for tf_scan in actual_timeframes_to_scan:
            if signal_found:
                break  # Found a signal, no need to check other timeframes
                
            df_current_tf = symbol_data_for_tfs.get(tf_scan)

            if df_current_tf is None or df_current_tf.empty or len(df_current_tf) < 50:
                logger.debug(f"  Insufficient data for {symbol} on {tf_scan} (need at least 50 candles)")
                signal_stats['insufficient_data'] += 1
                continue

            try:
                # Add symbol information to DataFrame for debugging
                df_with_symbol = df_current_tf.copy()
                if hasattr(df_with_symbol, 'symbol'):
                    df_with_symbol.symbol = symbol                    
                    signal = get_latest_transformer_signal(
                        df_with_symbol, model, scaler, feature_cols, target_idx, device_str, suggested_thresholds
                    )
                
                # Validate signal
                if signal not in [SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL]:
                    logger.warning(f"  Invalid signal returned for {symbol} on {tf_scan}: {signal}")
                    signal = SIGNAL_NEUTRAL
                
                # Track statistics
                if signal == SIGNAL_LONG:
                    signal_stats['long_generated'] += 1
                elif signal == SIGNAL_SHORT:
                    signal_stats['short_generated'] += 1
                else:
                    signal_stats['neutral_generated'] += 1
                
                # Process LONG signals
                if signal == SIGNAL_LONG and include_long_signals:
                    signals_list.append({
                        'Symbol': symbol,
                        'SignalTimeframe': tf_scan,
                        'SignalType': 'LONG'
                    })
                    logger.signal(f"  ✅ LONG signal found for {symbol} on {tf_scan}")
                    signal_found = True
                
                # Process SHORT signals
                elif signal == SIGNAL_SHORT and include_short_signals:
                    signals_list.append({
                        'Symbol': symbol,
                        'SignalTimeframe': tf_scan,
                        'SignalType': 'SHORT'
                    })
                    logger.signal(f"  ✅ SHORT signal found for {symbol} on {tf_scan}")
                    signal_found = True
                
                # Log other signals for debugging
                else:
                    signal_type = "NEUTRAL" if signal == SIGNAL_NEUTRAL else str(signal)
                    logger.debug(f"  ⚪ {signal_type} signal for {symbol} on {tf_scan} - not included based on parameters")
                    
            except Exception as e:
                logger.error(f"    ❌ Error getting Transformer signal for {symbol} on {tf_scan}: {e}")
                signal_stats['errors'] += 1
                # Continue to next timeframe instead of skipping symbol entirely
                continue

    # Enhanced logging with additional statistics
    logger.analysis("="*60)
    logger.analysis("TRANSFORMER SIGNAL GENERATION STATISTICS")
    logger.analysis("="*60)
    logger.performance(f"📊 Processing Statistics:")
    logger.performance(f"  • Symbols processed: {signal_stats['processed']}")
    logger.performance(f"  • Processing errors: {signal_stats['errors']}")
    logger.performance(f"  • Insufficient data cases: {signal_stats['insufficient_data']}")
    logger.performance(f"")
    logger.signal(f"🎯 Signal Generation Statistics:")
    logger.signal(f"  • LONG signals generated: {signal_stats['long_generated']}")
    logger.signal(f"  • SHORT signals generated: {signal_stats['short_generated']}")
    logger.signal(f"  • NEUTRAL signals generated: {signal_stats['neutral_generated']}")
    logger.signal(f"  • Total signals evaluated: {signal_stats['long_generated'] + signal_stats['short_generated'] + signal_stats['neutral_generated']}")
    
    if signal_stats['processed'] > 0:
        total_evaluated = signal_stats['long_generated'] + signal_stats['short_generated'] + signal_stats['neutral_generated']
        if total_evaluated > 0:
            long_rate = (signal_stats['long_generated'] / total_evaluated) * 100
            short_rate = (signal_stats['short_generated'] / total_evaluated) * 100
            neutral_rate = (signal_stats['neutral_generated'] / total_evaluated) * 100
            
            logger.analysis(f"")
            logger.analysis(f"📈 Signal Distribution:")
            logger.analysis(f"  • LONG rate: {long_rate:.1f}%")
            logger.analysis(f"  • SHORT rate: {short_rate:.1f}%")
            logger.analysis(f"  • NEUTRAL rate: {neutral_rate:.1f}%")
              # Check for potential bias with improved threshold suggestions
            bias_detected = False
            
            # Check for SHORT bias (model too bullish)
            if short_rate < 5.0 and include_short_signals:
                logger.warning(f"⚠️  POTENTIAL MODEL BIAS DETECTED (TOO BULLISH):")
                logger.warning(f"   Very low SHORT signal rate ({short_rate:.1f}%) might indicate:")
                logger.warning(f"   - Model bias toward bullish predictions")
                if suggested_thresholds:
                    logger.warning(f"   - Already using adjusted thresholds: SELL={suggested_thresholds[1]:.6f}")
                else:
                    logger.warning(f"   - SELL_THRESHOLD too restrictive ({SELL_THRESHOLD:.6f})")
                logger.warning(f"   - Training data bias toward upward price movements")
                logger.warning(f"   Consider adjusting thresholds or retraining with more balanced data")
                bias_detected = True
            
            # Check for LONG bias (model too bearish)
            if long_rate < 5.0 and include_long_signals:
                logger.warning(f"⚠️  POTENTIAL MODEL BIAS DETECTED (TOO BEARISH):")
                logger.warning(f"   Very low LONG signal rate ({long_rate:.1f}%) might indicate:")
                logger.warning(f"   - Model bias toward bearish predictions")
                if suggested_thresholds:
                    logger.warning(f"   - Already using adjusted thresholds: BUY={suggested_thresholds[0]:.6f}")
                else:
                    logger.warning(f"   - BUY_THRESHOLD too restrictive ({BUY_THRESHOLD:.6f})")
                logger.warning(f"   - Training data bias toward downward price movements")
                logger.warning(f"   Consider adjusting thresholds or retraining with more balanced data")
                bias_detected = True
            
            # Additional analysis for extreme bias cases
            if bias_detected and total_evaluated > 50:
                logger.analysis(f"🔍 BIAS ANALYSIS RECOMMENDATIONS:")
                if long_rate < 5.0 and short_rate < 5.0:
                    logger.analysis(f"   - Both LONG and SHORT rates are very low - model may be too conservative")
                    logger.analysis(f"   - Consider lowering both BUY_THRESHOLD and SELL_THRESHOLD")
                elif abs(long_rate - short_rate) > 50.0:
                    logger.analysis(f"   - Severe imbalance detected (difference: {abs(long_rate - short_rate):.1f}%)")
                    logger.analysis(f"   - Strong recommendation to retrain model with balanced dataset")
        else:
            logger.warning("No signals were successfully evaluated")

    # Count signals by type
    long_count = sum(1 for s in signals_list if s['SignalType'] == 'LONG')
    short_count = sum(1 for s in signals_list if s['SignalType'] == 'SHORT')
    
    logger.analysis("="*60)
    logger.success(f"COMPLETED TRANSFORMER SIGNAL ANALYSIS")
    logger.signal(f"Found {long_count} LONG signals and {short_count} SHORT signals")
    logger.analysis(f"Total signals: {len(signals_list)}")
    
    # Add success rate information
    if signal_stats['processed'] > 0:
        success_rate = ((signal_stats['processed'] - signal_stats['errors']) / signal_stats['processed']) * 100
        logger.performance(f"Processing success rate: {success_rate:.1f}%")        
        # Final GPU memory report
        if device:
            final_memory_info = gpu_manager.get_memory_info()
            logger.memory(f"Final GPU Memory - Allocated: {final_memory_info['allocated']//1024**2}MB, "
                         f"Cached: {final_memory_info['cached']//1024**2}MB")
        
        logger.analysis("="*60)

        return pd.DataFrame(signals_list) if signals_list else pd.DataFrame([], columns=DATAFRAME_COLUMNS)
