import argparse
import logging
import os
import pandas as pd
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from components._combine_all_dataframes import combine_all_dataframes
from components._load_all_symbols_data import load_all_symbols_data
from components.tick_processor import TickProcessor
from components.config import (
    DATA_PROCESSING_WAIT_TIME_IN_SECONDS,
    DEFAULT_CRYPTO_SYMBOLS,
    DEFAULT_TEST_SYMBOL,
    DEFAULT_TEST_TIMEFRAME,
    DEFAULT_TIMEFRAMES,
)
from signals.arimax_garch.signals_arimax_garch import (
    get_latest_arimax_garch_signal,
    train_and_save_arimax_garch_model,
    load_arimax_garch_model,
)

from utilities._logger import setup_logging
logger = setup_logging(module_name="signals_arimax_garch__main__", log_level=logging.INFO)

def main():
    """
    Main function to train ARIMAX-GARCH model and test signal generation.
    """
    start_time = time.time()
    
    logger.info("Starting ARIMAX-GARCH model training for crypto signals")
    
    parser = argparse.ArgumentParser(description='Crypto pair signal analysis using ARIMAX-GARCH')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')
    parser.add_argument('--multi-timeframe', action='store_true', default=True,
                        help='Use multiple timeframes for analysis')
    parser.add_argument('--pairs', type=str, help='Comma-separated list of crypto pairs to analyze')
    parser.add_argument('--top-symbols', type=int, default=10, 
                        help='Number of top symbols by volume to analyze (0 for all symbols, default 10)')
    args = parser.parse_args()

    logger._logger.setLevel(getattr(logging, args.log_level))
    
    try:
        processor = TickProcessor(trade_open_callback=None, trade_close_callback=None)
        logger.network("Successfully initialized Crypto TickProcessor")
        if processor is None:
            raise ConnectionError("Failed to initialize tick processor instance")
    except Exception as e:
        logger.error(f"Failed to initialize tick processor: {e}")
        return
    
    crypto_pairs = DEFAULT_CRYPTO_SYMBOLS.copy()
    if args.pairs:
        crypto_pairs = [pair.strip().upper() for pair in args.pairs.split(',')]
        logger.config(f"Using custom pairs: {crypto_pairs}")
    else:
        try:
            all_usdt_pairs = processor.get_symbols_list_by_quote_usdt()
            top_symbols_to_use = args.top_symbols if args.top_symbols > 0 else len(all_usdt_pairs)
            crypto_pairs = all_usdt_pairs[:top_symbols_to_use]
            logger.info(f"Using top {len(crypto_pairs)} USDT pairs by volume.")
        except Exception as e:
            logger.error(f"Error getting symbol list: {e}. Using default pairs.")
            
    logger.config(f"Final pairs list ({len(crypto_pairs)} pairs): {crypto_pairs[:5]}{'...' if len(crypto_pairs) > 5 else ''}")
    
    timeframes = DEFAULT_TIMEFRAMES if args.multi_timeframe else ['1h']
    
    start_time_load = time.time()
    all_symbols_data = load_all_symbols_data(processor, crypto_pairs, load_multi_timeframes=args.multi_timeframe, timeframes=timeframes)
    logger.data(f"Data loading completed in {time.time() - start_time_load:.2f} seconds")

    if not all_symbols_data:
        logger.warning("No data loaded for ARIMAX-GARCH signal analysis")
        return

    valid_symbols_data = {k: v for k, v in all_symbols_data.items() if isinstance(v, dict) and all(isinstance(df, pd.DataFrame) for df in v.values())}
    combined_df = combine_all_dataframes(valid_symbols_data)
    
    model, model_path = None, ""
    if not combined_df.empty:
        try:
            model, model_path = train_and_save_arimax_garch_model(combined_df)
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
    else:
        logger.warning("Combined DataFrame is empty, skipping model training.")

    if model is not None and model_path:
        logger.success(f"Model trained successfully! Saved at: {model_path}")
        loaded_model_data = load_arimax_garch_model(model_path)
        
        if loaded_model_data:
            logger.info("=" * 80)
            logger.model("ARIMAX-GARCH MODEL DETAILS".center(80))
            logger.info("=" * 80)
            
            config = loaded_model_data.get('model_config', {})
            metadata = loaded_model_data.get('training_metadata', {})
            
            logger.model(f"\n🔍 GENERAL INFORMATION:")
            logger.model(f"- Model type: ARIMAX-GARCH")
            logger.model(f"- ARIMAX order: {config.get('arimax_order')}")
            logger.model(f"- GARCH order: {config.get('garch_order')}")
            logger.model(f"- Training samples: {config.get('training_samples')}")
            logger.model(f"- Training time: {metadata.get('training_time', 0):.2f}s")
            
            exog_cols = loaded_model_data.get('exog_cols', [])
            if exog_cols:
                logger.analysis(f"\n📊 EXOGENOUS VARIABLES ({len(exog_cols)}):")
                for col in exog_cols:
                    logger.analysis(f"- {col}")

            logger.info("=" * 80)
    else:
        logger.error("Model training failed!")

    processor.get_historic_data_by_symbol(DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME)
    time.sleep(DATA_PROCESSING_WAIT_TIME_IN_SECONDS)
    df_test = processor.df_cache.get((DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME))
    
    if df_test is not None and model_path:
        loaded_model_data = load_arimax_garch_model(model_path)
        if loaded_model_data:
            signal = get_latest_arimax_garch_signal(df_test, loaded_model_data)
            logger.signal(f"Latest signal for {DEFAULT_TEST_SYMBOL} on {DEFAULT_TEST_TIMEFRAME}: {signal}")
        else:
            logger.error("Could not load model for signal generation test.")
    elif df_test is None:
        logger.error(f"Failed to retrieve data for {DEFAULT_TEST_SYMBOL} {DEFAULT_TEST_TIMEFRAME}")
    else:
        logger.error("Model was not trained, cannot generate signal.")

    try:
        processor.stop()
        logger.success("Tick processor stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping tick processor: {e}")
    
    elapsed_time = time.time() - start_time
    logger.performance(f"Process completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 