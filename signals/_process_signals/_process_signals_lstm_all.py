import logging
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the project root is in sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
signals_dir = os.path.dirname(current_dir)            
project_root = os.path.dirname(signals_dir)           
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities._logger import setup_logging
logger = setup_logging(module_name="process_signals_lstm_all", log_level=logging.INFO)

from livetrade._components._combine_all_dataframes import combine_all_dataframes
from livetrade._components._load_all_pairs_data import load_all_pairs_data
from livetrade._components._tick_processor import tick_processor
from livetrade.config import (
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NEUTRAL,
    DEFAULT_CRYPTO_SYMBOLS,
)
from signals.signals_cnn_lstm_attention import (
    train_and_save_global_LSTM_model,
    train_and_save_global_cnn_lstm_model_attention,
    get_latest_LSTM_signal,
    load_LSTM_model
)
from utilities._gpu_resource_manager import get_gpu_resource_manager

DATAFRAME_COLUMNS = ['Pair', 'SignalTimeframe', 'SignalType', 'Confidence', 'ModelScores']

class ModelConfiguration:
    """Configuration class for different LSTM model types"""
    def __init__(self, name: str, use_cnn: bool, use_attention: bool, 
                 attention_heads: int = 8, look_back: int = 50, output_mode: str = 'classification',
                 weight: float = 1.0):
        self.name = name
        self.use_cnn = use_cnn
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.look_back = look_back
        self.output_mode = output_mode
        self.weight = weight  # Weight for ensemble scoring

def get_lstm_model_configurations() -> List[ModelConfiguration]:
    """Define all 4 LSTM model configurations with weights"""
    return [
        ModelConfiguration("LSTM", use_cnn=False, use_attention=False, weight=1.0),
        ModelConfiguration("LSTM-Attention", use_cnn=False, use_attention=True, 
                         attention_heads=8, weight=1.2),
        ModelConfiguration("CNN-LSTM", use_cnn=True, use_attention=False, 
                         look_back=50, weight=1.1),
        ModelConfiguration("CNN-LSTM-Attention", use_cnn=True, use_attention=True, 
                         attention_heads=8, look_back=50, weight=1.3)
    ]

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """Validate DataFrame before processing"""
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if df is None or df.empty:
        logger.error("Empty or None DataFrame provided")
        return False
    
    # Check for both lowercase and uppercase column variants
    available_columns = set(df.columns.str.lower())
    missing_cols = []
    
    for col in required_columns:
        if col.lower() not in available_columns:
            missing_cols.append(col)
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {list(df.columns)}")
        return False
    
    return True

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame columns to lowercase format"""
    df_normalized = df.copy()
    
    # Create mapping for common column variations
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close', 'volume']:
            if col_lower not in df_normalized.columns:
                column_mapping[col] = col_lower
    
    # Apply mapping
    for old_col, new_col in column_mapping.items():
        df_normalized[new_col] = df_normalized[old_col]
    
    return df_normalized

def load_or_train_lstm_model(config: ModelConfiguration, combined_df: pd.DataFrame, 
                           models_dir: str, gpu_manager) -> Tuple[Optional[nn.Module], str]:
    """
    Load existing LSTM model or train new one if not found
    
    Args:
        config: Model configuration
        combined_df: Combined training data
        models_dir: Directory to save/load models
        gpu_manager: GPU resource manager
        
    Returns:
        Tuple of (model, model_path)
    """
    # Generate model filename based on configuration
    model_filename = f"lstm_model_{config.name.lower().replace('-', '_')}.pth"
    model_path = os.path.join(models_dir, model_filename)
    
    # Try to load existing model
    try:
        if os.path.exists(model_path):
            model = load_LSTM_model(Path(model_path))
            if model is not None:
                logger.info(f"Loaded existing {config.name} model from: {model_path}")
                return model, model_path
    except Exception as e:
        logger.warning(f"Failed to load existing {config.name} model: {e}")
    
    # Train new model if loading failed or model doesn't exist
    logger.info(f"Training new {config.name} model...")
    
    try:
        with gpu_manager.gpu_scope():
            if config.use_cnn:
                # CNN-LSTM or CNN-LSTM-Attention
                model, model_path = train_and_save_global_cnn_lstm_model_attention(
                    combined_df=combined_df,
                    model_filename=model_filename,
                    use_attention=config.use_attention,
                    use_cnn=config.use_cnn,
                    look_back=config.look_back,
                    output_mode=config.output_mode,
                    attention_heads=config.attention_heads
                )
            else:
                # LSTM or LSTM-Attention
                model, model_path = train_and_save_global_LSTM_model(
                    combined_df=combined_df,
                    model_filename=model_filename,
                    use_attention=config.use_attention,
                    attention_heads=config.attention_heads
                )
        
        if model is not None:
            logger.info(f"Successfully trained {config.name} model: {model_path}")
        else:
            logger.error(f"Failed to train {config.name} model")
        
        return model, model_path
        
    except Exception as e:
        logger.error(f"Error training {config.name} model: {e}")
        return None, ""

def get_model_signal_with_confidence(df: pd.DataFrame, model: nn.Module, 
                                   config: ModelConfiguration) -> Tuple[str, float]:
    """Get signal from a single model with confidence score"""
    try:
        df_normalized = normalize_dataframe_columns(df)
        
        if not validate_dataframe(df_normalized):
            return SIGNAL_NEUTRAL, 0.0
        
        signal = get_latest_LSTM_signal(df_normalized, model)
        
        if signal == SIGNAL_LONG:
            confidence = 0.8 * config.weight
        elif signal == SIGNAL_SHORT:
            confidence = 0.8 * config.weight
        else:
            confidence = 0.1
        
        return signal, confidence
        
    except Exception as e:
        logger.error(f"Error getting signal from {config.name} model: {e}")
        return SIGNAL_NEUTRAL, 0.0

def combine_model_signals(model_results: Dict[str, Tuple[str, float]], 
                        include_long_signals: bool, include_short_signals: bool) -> Tuple[str, float, Dict]:
    """Combine signals from multiple models with weighted scoring"""
    if not model_results:
        return SIGNAL_NEUTRAL, 0.0, {}
    
    long_score = 0.0
    short_score = 0.0
    neutral_score = 0.0
    total_weight = 0.0
    
    scores_breakdown = {}
    
    for model_name, (signal, confidence) in model_results.items():
        if signal == SIGNAL_LONG:
            long_score += confidence
        elif signal == SIGNAL_SHORT:
            short_score += confidence
        else:
            neutral_score += confidence
        
        total_weight += confidence if signal != SIGNAL_NEUTRAL else 0.1
        scores_breakdown[model_name] = {'signal': signal, 'confidence': confidence}
    
    # Normalize scores
    if total_weight > 0:
        long_score /= total_weight
        short_score /= total_weight
        neutral_score /= total_weight
    
    # Determine final signal
    final_signal = SIGNAL_NEUTRAL
    final_confidence = 0.0
    
    if long_score > short_score and long_score > neutral_score and include_long_signals:
        final_signal = SIGNAL_LONG
        final_confidence = long_score
    elif short_score > long_score and short_score > neutral_score and include_short_signals:
        final_signal = SIGNAL_SHORT
        final_confidence = short_score
    else:
        final_signal = SIGNAL_NEUTRAL
        final_confidence = neutral_score
    
    scores_breakdown['final_scores'] = {
        'long': long_score,
        'short': short_score,
        'neutral': neutral_score
    }
    
    return final_signal, final_confidence, scores_breakdown

def process_symbol_lstm_signals(symbol: str, symbol_data: Dict[str, pd.DataFrame],
                               timeframes_to_scan: List[str], models: Dict[str, nn.Module],
                               model_configs: List[ModelConfiguration],
                               include_long_signals: bool, include_short_signals: bool) -> List[Dict]:
    """Process LSTM signals for a single symbol across multiple timeframes and models"""
    signals_list = []
    
    for tf_scan in timeframes_to_scan:
        df_current_tf = symbol_data.get(tf_scan)
        
        if df_current_tf is None or df_current_tf.empty or len(df_current_tf) < 50:
            logger.debug(f"Insufficient data for {symbol} on {tf_scan}")
            continue
        
        # Get signals from all models for this timeframe
        model_results = {}
        
        for config in model_configs:
            model = models.get(config.name)
            if model is not None:
                signal, confidence = get_model_signal_with_confidence(
                    df_current_tf, model, config
                )
                model_results[config.name] = (signal, confidence)
        
        if not model_results:
            logger.warning(f"No model results for {symbol} on {tf_scan}")
            continue
        
        # Combine signals from all models
        final_signal, final_confidence, scores_breakdown = combine_model_signals(
            model_results, include_long_signals, include_short_signals
        )
        
        # Only add signal if it matches requested types
        if ((final_signal == SIGNAL_LONG and include_long_signals) or 
            (final_signal == SIGNAL_SHORT and include_short_signals)):
            
            signals_list.append({
                'Pair': symbol,
                'SignalTimeframe': tf_scan,
                'SignalType': 'LONG' if final_signal == SIGNAL_LONG else 'SHORT',
                'Confidence': final_confidence,
                'ModelScores': scores_breakdown
            })
            
            logger.info(f"LSTM ensemble signal for {symbol} on {tf_scan}: "
                        f"{final_signal} (confidence: {final_confidence:.3f})")
            break  # Found signal for this symbol, move to next symbol
    
    return signals_list

def process_signals_lstm_all(
    preloaded_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    timeframes_to_scan: Optional[List[str]] = None,
    include_long_signals: bool = True,
    include_short_signals: bool = False,
    processor: Optional[object] = None,
    symbols: Optional[List[str]] = None,
    auto_train_if_missing: bool = True,
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Process trading signals using all 4 LSTM model configurations with ensemble scoring.
    
    Args:
        preloaded_data: Pre-loaded symbol data in format {symbol: {timeframe: dataframe}}
        timeframes_to_scan: Specific timeframes to scan, defaults to ['1h', '4h', '1d']
        include_long_signals: Include LONG signals in results
        include_short_signals: Include SHORT signals in results
        processor: Tick processor for loading data if preloaded_data is None
        symbols: List of symbols to analyze if loading new data
        auto_train_if_missing: Train new models if not found
        max_workers: Maximum number of threads for parallel processing
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Pair', 'SignalTimeframe', 'SignalType', 'Confidence', 'ModelScores']
    """
    logger.info("="*80)
    logger.info("STARTING COMPREHENSIVE LSTM SIGNAL ANALYSIS")
    logger.info("="*80)
    
    # Initialize GPU resource manager
    gpu_manager = get_gpu_resource_manager()
    
    # Validate signal type parameters
    if not include_long_signals and not include_short_signals:
        logger.error("At least one of include_long_signals or include_short_signals must be True")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
    
    # Initialize parameters
    actual_timeframes_to_scan = timeframes_to_scan or ['1h', '4h', '1d']
    actual_symbols = symbols or DEFAULT_CRYPTO_SYMBOLS[:10]  # Limit for performance
    
    logger.info(f"Timeframes to scan: {actual_timeframes_to_scan}")
    logger.info(f"Signal types: LONG={include_long_signals}, SHORT={include_short_signals}")
    
    try:
        # Step 1: Data preparation
        if preloaded_data:
            logger.info("Using provided preloaded_data")
            combined_df = combine_all_dataframes(preloaded_data)
            symbols_to_analyze = list(preloaded_data.keys())
        else:
            logger.info("Loading new data using processor")
            if processor is None:
                logger.info("Creating new processor instance")
                processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
            
            # Load all pairs data
            loaded_data = load_all_pairs_data(
                processor=processor,
                symbols=actual_symbols,
                load_multi_timeframes=True,
                timeframes=actual_timeframes_to_scan
            )
            
            if not loaded_data:
                logger.error("Failed to load data")
                return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
            
            # Filter and validate the loaded data to match expected type
            preloaded_data = {}
            for symbol, data in loaded_data.items():
                if isinstance(data, dict):
                    # Only include symbols with valid dict[str, DataFrame] structure
                    valid_timeframes = {}
                    for tf, df in data.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            valid_timeframes[tf] = df
                    if valid_timeframes:
                        preloaded_data[symbol] = valid_timeframes
            
            if not preloaded_data:
                logger.error("No valid data found after filtering")
                return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
            
            combined_df = combine_all_dataframes(preloaded_data)
            symbols_to_analyze = list(preloaded_data.keys())
        
        if combined_df.empty:
            logger.error("Combined DataFrame is empty")
            return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
        
        logger.info(f"Analyzing {len(symbols_to_analyze)} symbols with {len(combined_df)} total data points")
        
        # Step 2: Model loading/training
        logger.info("="*60)
        logger.info("LOADING/TRAINING LSTM MODELS")
        logger.info("="*60)
        
        model_configs = get_lstm_model_configurations()
        models = {}
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        for config in model_configs:
            logger.info(f"Processing {config.name} model...")
            
            if auto_train_if_missing:
                model, model_path = load_or_train_lstm_model(
                    config, combined_df, models_dir, gpu_manager
                )
                if model is not None:
                    models[config.name] = model
                    logger.info(f"✅ {config.name} model ready")
                else:
                    logger.error(f"❌ Failed to load/train {config.name} model")
            else:
                # Only try to load existing models
                model_filename = f"lstm_model_{config.name.lower().replace('-', '_')}.pth"
                model_path = os.path.join(models_dir, model_filename)
                if os.path.exists(model_path):
                    model = load_LSTM_model(Path(model_path))
                    if model is not None:
                        models[config.name] = model
                        logger.info(f"✅ Loaded existing {config.name} model")
                    else:
                        logger.warning(f"⚠️ Failed to load {config.name} model")
                else:
                    logger.warning(f"⚠️ {config.name} model not found")
        
        if not models:
            logger.error("No LSTM models available. Cannot generate signals.")
            return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
        
        logger.info(f"Successfully loaded {len(models)}/{len(model_configs)} models")
        
        # Step 3: Signal generation
        logger.info("="*60)
        logger.info("GENERATING ENSEMBLE LSTM SIGNALS")
        logger.info("="*60)
        
        all_signals = []
        
        # Process symbols with potential parallel execution or sequential processing
        if max_workers > 1 and len(symbols_to_analyze) > 1:
            logger.info(f"Processing {len(symbols_to_analyze)} symbols using {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {}
                
                for symbol in symbols_to_analyze:
                    symbol_data = preloaded_data.get(symbol, {})
                    if symbol_data:
                        future = executor.submit(
                            process_symbol_lstm_signals,
                            symbol, symbol_data, actual_timeframes_to_scan,
                            models, model_configs, include_long_signals, include_short_signals
                        )
                        future_to_symbol[future] = symbol
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        symbol_signals = future.result()
                        all_signals.extend(symbol_signals)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
        else:
            # Sequential processing
            logger.info(f"Processing {len(symbols_to_analyze)} symbols sequentially")
            
            for symbol in symbols_to_analyze:
                symbol_data = preloaded_data.get(symbol, {})
                if symbol_data:
                    try:
                        symbol_signals = process_symbol_lstm_signals(
                            symbol, symbol_data, actual_timeframes_to_scan,
                            models, model_configs, include_long_signals, include_short_signals
                        )
                        all_signals.extend(symbol_signals)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
        
        # Count results
        long_count = sum(1 for s in all_signals if s['SignalType'] == 'LONG')
        short_count = sum(1 for s in all_signals if s['SignalType'] == 'SHORT')
        
        logger.info("="*80)
        logger.info("COMPLETED COMPREHENSIVE LSTM SIGNAL ANALYSIS")
        logger.info(f"Found {long_count} LONG signals and {short_count} SHORT signals")
        logger.info(f"Total ensemble signals: {len(all_signals)}")
        logger.info(f"Used {len(models)} LSTM model configurations")
        logger.info("="*80)
        
        return pd.DataFrame(all_signals) if all_signals else pd.DataFrame([], columns=DATAFRAME_COLUMNS)
        
    except Exception as e:
        logger.error(f"Error in LSTM signal processing: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
    
    finally:
        # Cleanup GPU resources
        try:
            torch.cuda.empty_cache()
        except:
            pass
