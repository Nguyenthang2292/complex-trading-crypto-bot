import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import traceback

# Add parent directory to sys.path for importing from sibling directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from livetrade._components._combine_all_dataframes import combine_all_dataframes
from livetrade._components._load_all_symbols_data import load_all_symbols_data
from livetrade._components._tick_processor import tick_processor
from livetrade.config import (DEFAULT_CRYPTO_SYMBOLS, DEFAULT_TIMEFRAMES, DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME)
from signals._quant_models.lstm_attention_model import (
    get_latest_lstm_attention_signal,
    train_and_save_global_lstm_attention_model
)
from signals.signals_cnn_lstm_attention import train_and_save_global_cnn_lstm_attention_model
from utilities._gpu_resource_manager import get_gpu_resource_manager
from utilities._logger import setup_logging

# Constants
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

# Initialize logger
logger = setup_logging(module_name="signals_cnn_lstm_attention__main__", log_level=logging.DEBUG)

class ModelConfiguration:
    """
    Configuration class for different model types
    
    Attributes:
        name (str): Model name identifier
        use_cnn (bool): Whether to use CNN layers
        use_attention (bool): Whether to use attention mechanism
        attention_heads (int): Number of attention heads if attention is used
        look_back (int): Number of time steps to look back for sequence models
        output_mode (str): Model output type ('classification' or 'regression')
    """
    def __init__(self, name: str, use_cnn: bool, use_attention: bool, 
                 attention_heads: int = 8, look_back: int = 50, output_mode: str = 'classification'):
        self.name = name
        self.use_cnn = use_cnn
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.look_back = look_back
        self.output_mode = output_mode

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = REQUIRED_COLUMNS) -> bool:
    """
    Validate DataFrame before processing
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        bool: True if DataFrame is valid, False otherwise
    """
    if df is None or df.empty:
        logger.error("Empty or None DataFrame provided")
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    return True

def validate_model_input(
    df: pd.DataFrame, 
    config: ModelConfiguration,
    required_columns: List[str] = REQUIRED_COLUMNS
) -> Tuple[bool, str]:
    """
    Validate input data for model training
    
    Args:
        df: DataFrame to validate
        config: Model configuration
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not validate_dataframe(df, required_columns):
        return False, "Invalid DataFrame"
    
    if config.use_cnn and len(df) < config.look_back + 100:
        return False, f"Insufficient data for CNN model: {len(df)} rows, need at least {config.look_back + 100}"
    
    if df.isnull().values.any():
        return False, "DataFrame contains NaN values"
    
    return True, ""

def log_model_error(logger, model_name: str, error: Exception, config: ModelConfiguration):
    """
    Unified error logging for model training
    
    Args:
        logger: Logger instance
        model_name: Name of the model that experienced an error
        error: Exception that occurred
        config: Model configuration
    """
    logger.error(f"Error during {model_name} model training: {error}")
    logger.warning("This might be due to:")
    logger.warning("  - Insufficient data after feature calculation")
    logger.warning("  - Missing required OHLC columns") 
    logger.warning("  - Too many NaN values in the data")
    logger.debug(f"Configuration: {config.__dict__}")

def safe_execute_with_gpu(function, *args, gpu_manager=None, **kwargs):
    """
    Execute function with proper GPU resource management
    
    Args:
        function: Function to execute
        gpu_manager: GPU resource manager instance
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Return value from the function
        
    Raises:
        Exception: If function execution fails
    """
    if gpu_manager is None:
        gpu_manager = get_gpu_resource_manager()
    
    try:
        with gpu_manager.gpu_scope():
            return function(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {function.__name__}: {e}")
        raise

def cleanup_resources(processor=None, model=None):
    """
    Cleanup resources after model training
    
    Args:
        processor: Data processor instance
        model: Model instance to clean up
    """
    if processor is not None and hasattr(processor, 'stop'):
        try:
            processor.stop()
            logger.info("Data processor stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping processor: {e}")
    
    if model is not None:
        try:
            del model
            torch.cuda.empty_cache()
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up model resources: {e}")

def get_model_configurations() -> List[ModelConfiguration]:
    """
    Define all 4 model configurations
    
    Returns:
        List of ModelConfiguration objects
    """
    return [
        ModelConfiguration("LSTM", use_cnn=False, use_attention=False),
        ModelConfiguration("LSTM-Attention", use_cnn=False, use_attention=True, attention_heads=8),
        ModelConfiguration("CNN-LSTM", use_cnn=True, use_attention=False, look_back=50),
        ModelConfiguration("CNN-LSTM-Attention", use_cnn=True, use_attention=True, 
                         attention_heads=8, look_back=50)
    ]

def load_and_prepare_data(processor, symbols: List[str], timeframes: List[str]) -> Optional[pd.DataFrame]:
    """
    Load all pairs data and combine into single DataFrame
    
    Args:
        processor: Tick processor instance
        symbols: List of crypto symbols
        timeframes: List of timeframes to load
        
    Returns:
        Combined DataFrame or None if failed
    """
    try:
        logger.info("="*80)
        logger.info("STEP 1: LOADING ALL PAIRS DATA")
        logger.info("="*80)
        
        logger.info(f"Loading data for {len(symbols)} symbols across {len(timeframes)} timeframes...")
        logger.debug(f"Symbols: {symbols}")
        logger.debug(f"Timeframes: {timeframes}")
        
        all_symbols_data = load_all_symbols_data(
            processor=processor,
            symbols=symbols,
            load_multi_timeframes=True,
            timeframes=timeframes
        )
        
        if not all_symbols_data:
            logger.error("Failed to load any pairs data")
            return None
        
        total_pairs = len(all_symbols_data)
        successful_pairs = sum(1 for v in all_symbols_data.values() if v is not None)
        logger.info(f"Data loading results: {successful_pairs}/{total_pairs} pairs loaded successfully")
        
        logger.info("="*80)
        logger.info("STEP 2: COMBINING ALL DATAFRAMES")
        logger.info("="*80)
        
        filtered_data = {k: v for k, v in all_symbols_data.items() if isinstance(v, dict)}
        combined_df = combine_all_dataframes(filtered_data)
        
        if combined_df.empty:
            logger.error("Combined DataFrame is empty")
            return None
        
        logger.info(f"Successfully combined data: {len(combined_df)} total rows")
        logger.debug(f"Combined DataFrame shape: {combined_df.shape}")
        logger.debug(f"Columns: {list(combined_df.columns)}")
        
        if 'pair' in combined_df.columns:
            logger.info(f"Unique pairs in combined data: {combined_df['pair'].nunique()}")
        
        if 'timeframe' in combined_df.columns:
            logger.info(f"Unique timeframes in combined data: {combined_df['timeframe'].nunique()}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error in data loading and preparation: {e}")
        return None

def train_model_configuration(
    config: ModelConfiguration, 
    combined_df: pd.DataFrame,
    gpu_manager=None
) -> Tuple[Optional[object], str]:
    """
    Train a specific model configuration with proper resource management
    
    Args:
        config: Model configuration
        combined_df: Combined training data
        gpu_manager: Optional GPU resource manager
        
    Returns:
        Tuple of (model, model_path)
    """
    if gpu_manager is None:
        gpu_manager = get_gpu_resource_manager()
    
    try:
        logger.info("="*80)
        logger.info(f"TRAINING {config.name.upper()} MODEL")
        logger.info("="*80)
        
        logger.info(f"Configuration: {config.name}")
        logger.debug(f"  - Use CNN: {config.use_cnn}")
        logger.debug(f"  - Use Attention: {config.use_attention}")
        if config.use_attention:
            logger.debug(f"  - Attention Heads: {config.attention_heads}")
        if config.use_cnn:
            logger.debug(f"  - Look Back: {config.look_back}")
            logger.debug(f"  - Output Mode: {config.output_mode}")
        
        is_valid, error = validate_model_input(combined_df, config)
        if not is_valid:
            logger.error(f"Invalid input data for {config.name}: {error}")
            return None, ""
        
        logger.debug(f"Training data shape: {combined_df.shape}")
        logger.debug(f"Training data columns: {list(combined_df.columns)}")
        
        start_time = time.time()
        
        with gpu_manager.gpu_scope():
            if config.use_cnn:
                try:
                    model, model_path = safe_execute_with_gpu(
                        train_and_save_global_cnn_lstm_attention_model,
                        combined_df=combined_df,
                        model_filename=None,
                        use_attention=config.use_attention,
                        use_cnn=config.use_cnn,
                        look_back=config.look_back,
                        output_mode=config.output_mode,
                        attention_heads=config.attention_heads
                    )
                except ValueError as ve:
                    if "no valid sequences created" in str(ve):
                        log_model_error(logger, config.name, ve, config)
                        logger.warning("Attempting fallback to LSTM-only model...")
                        
                        try:
                            model, model_path = safe_execute_with_gpu(
                                train_and_save_global_lstm_attention_model,
                                combined_df=combined_df,
                                model_filename=None,
                                use_attention=config.use_attention,
                                attention_heads=config.attention_heads
                            )
                            logger.warning(f"Successfully trained fallback LSTM model for {config.name}")
                        except Exception as fallback_error:
                            logger.error(f"Fallback LSTM training also failed for {config.name}: {fallback_error}")
                            return None, ""
                    else:
                        raise ve
            else:
                model, model_path = train_and_save_global_lstm_attention_model(
                    combined_df=combined_df,
                    model_filename=None,
                    use_attention=config.use_attention,
                    attention_heads=config.attention_heads
                )
        
        training_time = time.time() - start_time
        
        if model is not None:
            logger.info(f"{config.name} model trained successfully in {training_time:.2f}s")
            logger.info(f"Model saved to: {model_path}")
        else:
            logger.error(f"{config.name} model training failed")
        
        return model, model_path
        
    except Exception as e:
        logger.error(f"Error training {config.name} model: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None, ""

def test_signal_generation(config: ModelConfiguration, model: object, test_symbol: str, 
                         test_timeframe: str, all_pairs_data: Dict) -> str:
    """
    Test signal generation for a specific model
    
    Args:
        config: Model configuration
        model: Trained model
        test_symbol: Symbol to test (e.g., 'BTCUSDT')
        test_timeframe: Timeframe to test (e.g., '1h')
        all_pairs_data: All loaded pairs data
        
    Returns:
        Generated signal string
    """
    try:
        logger.info(f"Testing {config.name} signal generation for {test_symbol} {test_timeframe}...")
        
        # Get test data
        test_df = None
        if (all_pairs_data and 
            test_symbol in all_pairs_data and 
            all_pairs_data[test_symbol] and
            isinstance(all_pairs_data[test_symbol], dict) and
            test_timeframe in all_pairs_data[test_symbol]):
            
            test_df = all_pairs_data[test_symbol][test_timeframe].copy()
            logger.debug(f"Using real data: {len(test_df)} rows")
            logger.debug(f"Current columns: {list(test_df.columns)}")
            
            # Create lowercase versions of columns if needed
            column_mapping = {}
            for col in test_df.columns:
                col_lower = col.lower()
                if col_lower in REQUIRED_COLUMNS:
                    column_mapping[col] = col_lower
            
            if column_mapping:
                for old_col, new_col in column_mapping.items():
                    if new_col not in test_df.columns:
                        test_df[new_col] = test_df[old_col]
                        logger.debug(f"Added column mapping: {old_col} -> {new_col}")
            
            # Check for missing required columns
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in test_df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                # Try uppercase versions
                for missing_col in missing_columns:
                    upper_col = missing_col.title()  # 'close' -> 'Close'
                    if upper_col in test_df.columns:
                        test_df[missing_col] = test_df[upper_col]
                        logger.debug(f"Mapped {upper_col} to {missing_col}")
                    else:
                        logger.error(f"Cannot find column for {missing_col}")
                        
        else:
            # Create sample data
            logger.warning("Real data not available, creating sample data")
            np.random.seed(42)
            
            sample_data = []
            base_price = 45000
            for i in range(100):
                open_price = base_price + np.random.normal(0, 1000)
                high_price = open_price + np.random.uniform(0, 2000)
                low_price = open_price - np.random.uniform(0, 2000)
                close_price = np.random.uniform(low_price, high_price)
                volume = np.random.uniform(50, 500)
                
                sample_data.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            test_df = pd.DataFrame(sample_data)
            
        # Final verification of required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in test_df.columns]
        
        if missing_columns:
            logger.error(f"Still missing required columns after mapping: {missing_columns}")
            logger.debug(f"Available columns: {list(test_df.columns)}")
            return "ERROR - Missing columns"
            
        # Generate signal
        from typing import cast
        signal = get_latest_lstm_attention_signal(test_df, cast(nn.Module, model))
        logger.info(f"{config.name} signal for {test_symbol} {test_timeframe}: {signal}")
        
        return signal
        
    except Exception as e:
        logger.error(f"Error testing {config.name} signal generation: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return "ERROR"

def main():
    """
    Main function that loads data, trains all model configurations,
    and tests signal generation on specified symbol and timeframe.
    
    Returns:
        Dict containing training results, signal results, and metrics,
        or None if execution fails
    """
    total_start_time = time.time()
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL TRAINING AND TESTING")
    logger.info("="*80)
    logger.info("Testing 4 model configurations:")
    logger.info("  1. LSTM")
    logger.info("  2. LSTM + Attention")
    logger.info("  3. CNN + LSTM")
    logger.info("  4. CNN + LSTM + Attention")
    logger.info("="*80)    
    
    gpu_manager = get_gpu_resource_manager()
    processor = None
    
    try:
        logger.info("Initializing data processor...")
        processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
        logger.info("Data processor initialized successfully")
        
        # Get symbols and timeframes (limited subset for testing)
        symbols = DEFAULT_CRYPTO_SYMBOLS[:5]
        timeframes = DEFAULT_TIMEFRAMES[:3]
        
        logger.debug(f"Selected symbols: {symbols}")
        logger.debug(f"Selected timeframes: {timeframes}")
        
        # Load and prepare training data
        combined_df = load_and_prepare_data(processor, symbols, timeframes)
        if combined_df is None:
            logger.error("Failed to load and prepare data")
            return None
        
        # Load test data for signal generation
        logger.info(f"Loading test data for {DEFAULT_TEST_SYMBOL} {DEFAULT_TEST_TIMEFRAME}...")
        test_data = load_all_symbols_data(
            processor=processor,
            symbols=[DEFAULT_TEST_SYMBOL],
            load_multi_timeframes=True,
            timeframes=[DEFAULT_TEST_TIMEFRAME]
        )
        
        # Get model configurations and prepare result storage
        model_configs = get_model_configurations()
        training_results = {}
        signal_results = {}
        
        # Train and test each model configuration
        for i, config in enumerate(model_configs, 1):
            logger.info(f"\n{'='*20} MODEL {i}/4: {config.name.upper()} {'='*20}")
            
            model, model_path = train_model_configuration(config, combined_df, gpu_manager)
            
            if model is not None:
                training_results[config.name] = {
                    'model': model,
                    'model_path': model_path,
                    'success': True
                }
                
                signal = test_signal_generation(config, model, DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME, test_data)
                signal_results[config.name] = signal
            else:
                training_results[config.name] = {
                    'model': None,
                    'model_path': "",
                    'success': False
                }
                signal_results[config.name] = "FAILED"
        
        # Generate final results summary
        total_time = time.time() - total_start_time
        
        logger.info("="*80)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*80)
        
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Data processed: {len(combined_df)} rows from {len(symbols)} pairs")
        
        # Training results summary
        logger.info("\nTRAINING RESULTS:")
        successful_models = 0
        for config_name, result in training_results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            logger.info(f"  {config_name:20} : {status}")
            if result['success']:
                successful_models += 1
                logger.debug(f"    Model saved: {result['model_path']}")
        
        # Signal generation results
        logger.info(f"\nSIGNAL GENERATION RESULTS FOR {DEFAULT_TEST_SYMBOL} {DEFAULT_TEST_TIMEFRAME}:")
        for config_name, signal in signal_results.items():
            if signal == "FAILED":
                logger.error(f"  {config_name:20} : ❌ FAILED")
            elif signal == "ERROR":
                logger.warning(f"  {config_name:20} : ⚠️  ERROR")
            else:
                logger.info(f"  {config_name:20} : 📊 {signal}")
        
        # Performance summary
        logger.info("\nPERFORMANCE SUMMARY:")
        logger.info(f"  Models trained successfully: {successful_models}/4")
        logger.info(f"  Success rate: {successful_models/4*100:.1f}%")
        logger.info(f"  Average time per model: {total_time/4:.2f}s")
        
        # Detailed configuration comparison
        logger.info("\nCONFIGURATION COMPARISON:")
        for config in model_configs:
            logger.debug(f"  {config.name}:")
            logger.debug(f"    - CNN: {config.use_cnn}")
            logger.debug(f"    - Attention: {config.use_attention}")
            if config.use_attention:
                logger.debug(f"    - Attention Heads: {config.attention_heads}")
            if config.use_cnn:
                logger.debug(f"    - Look Back: {config.look_back}")
            logger.info(f"    - Training: {'✅' if training_results[config.name]['success'] else '❌'}")
            logger.info(f"    - Signal: {signal_results[config.name]}")        
        logger.info("Comprehensive model testing completed!")
        
        return {
            'training_results': training_results,
            'signal_results': signal_results,
            'total_time': total_time,
            'success_rate': successful_models/4
        }
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None
    
    finally:
        cleanup_resources(processor)

if __name__ == "__main__":
    logger.info("Starting comprehensive model training and testing...")
    logger.info("This will test all 4 model configurations:")
    logger.info("  1. LSTM")
    logger.info("  2. LSTM + Attention") 
    logger.info("  3. CNN + LSTM")
    logger.info("  4. CNN + LSTM + Attention")
    logger.info("")
    
    result = main()
    
    if result:
        logger.info("All tests completed successfully!")
        logger.info(f"Overall success rate: {result['success_rate']*100:.1f}%")
    else:
        logger.error("Tests failed!")