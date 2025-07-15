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
    DEFAULT_TOP_SYMBOLS,
    MODEL_FEATURES,
)
from signals.random_forest.signals_random_forest import (
    get_latest_random_forest_signal,
    train_and_save_global_rf_model,
)

from utilities._logger import setup_logging
logger = setup_logging(module_name="signals_random_forest__main__", log_level=logging.INFO)

def main():
    """
    Main function to train Random Forest model and test signal generation on crypto data.
    
    This function:
    1. Initializes Binance tick processor
    2. Loads market data for specified crypto pairs
    3. Trains Random Forest model on combined data
    4. Tests signal generation on a sample pair
    """
    DATAFRAME_COLUMNS = ['Pair', 'FinalSignal', 'SignalTimeframe']
    start_time = time.time()
    
    logger.info("Starting Random Forest model training for crypto signals")
    
    parser = argparse.ArgumentParser(description='Crypto pair signal analysis using Random Forest')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')
    parser.add_argument('--multi-timeframe', action='store_true', default=True,
                        help='Use multiple timeframes for analysis')
    parser.add_argument('--pairs', type=str, help='Comma-separated list of crypto pairs to analyze')
    parser.add_argument('--top-symbols', type=int, default=0, 
                        help='Number of top symbols by volume to analyze (0 for all symbols)')
    args = parser.parse_args()

    logger._logger.setLevel(getattr(logging, args.log_level))
    
    # Initialize Binance tick processor
    try:
        processor = TickProcessor(trade_open_callback=None, trade_close_callback=None)
        logger.network(f"Successfully initialized Crypto TickProcessor with Binance API")
        if processor is None:
            logger.error("Failed to initialize tick processor instance")
            return
    except Exception as e:
        logger.error(f"Failed to initialize Binance tick processor: {e}")
        return
    
    # Determine crypto pairs to analyze
    crypto_pairs = DEFAULT_CRYPTO_SYMBOLS.copy()
    
    if args.pairs:
        custom_pairs = [pair.strip() for pair in args.pairs.split(',')]
        if custom_pairs:
            crypto_pairs = custom_pairs
            logger.config(f"Using custom pairs: {crypto_pairs}")
    else:
        # Get all available USDT trading pairs from Binance
        try:
            all_usdt_pairs = processor.get_symbols_list_by_quote_asset('USDT')
            logger.network(f"Found {len(all_usdt_pairs)} USDT trading pairs on Binance")
            
            top_symbols_to_use = args.top_symbols if args.top_symbols > 0 else DEFAULT_TOP_SYMBOLS
            
            if top_symbols_to_use < len(all_usdt_pairs):
                crypto_pairs = all_usdt_pairs[:top_symbols_to_use]
                logger.memory(f"Using top {top_symbols_to_use} USDT pairs (limited for memory management)")
            else:
                crypto_pairs = all_usdt_pairs
                logger.info(f"Using all {len(crypto_pairs)} USDT pairs")
                
        except Exception as e:
            logger.error(f"Error getting symbol list: {e}. Using default pairs")
            
    logger.config(f"Final pairs list ({len(crypto_pairs)} pairs): {crypto_pairs[:5]}{'...' if len(crypto_pairs) > 5 else ''}")
    
    timeframes = DEFAULT_TIMEFRAMES if args.multi_timeframe else ['1h']
    
    # Load market data for all pairs
    start_time_load = time.time()
    all_symbols_data = load_all_symbols_data(processor, crypto_pairs, load_multi_timeframes=args.multi_timeframe, timeframes=timeframes)
    logger.data(f"Data loading completed in {time.time() - start_time_load:.2f} seconds")

    if not all_symbols_data:
        logger.warning("No data loaded for Random Forest signal analysis")
        return pd.DataFrame(columns=pd.Index(DATAFRAME_COLUMNS))
    
    # Filter and combine valid data
    valid_symbols_data = {k: v for k, v in all_symbols_data.items() 
                         if isinstance(v, dict) and all(isinstance(df, pd.DataFrame) for df in v.values())}
    combined_df = combine_all_dataframes(valid_symbols_data)
    
    # Train Random Forest model
    model, model_path = None, ""
    try:
        model, model_path = train_and_save_global_rf_model(combined_df)
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
    
    # Display model information if training successful
    if model is not None and model_path:
        logger.success(f"Model trained successfully! Saved at: {model_path}")
        
        logger.info("=" * 80)
        logger.model("RANDOM FOREST MODEL DETAILS".center(80))
        logger.info("=" * 80)
        
        # Model general information
        model_params = model.get_params()
        logger.model(f"\n🔍 GENERAL INFORMATION:")
        logger.model(f"- Model type: {type(model).__name__}")
        logger.model(f"- Trees (n_estimators): {model_params['n_estimators']}")
        logger.model(f"- Max depth: {'unlimited' if model_params['max_depth'] is None else model_params['max_depth']}")
        
        # Check if model has been fitted before accessing n_features_in_
        if hasattr(model, 'n_features_in_'):
            logger.model(f"- Number of features: {model.n_features_in_}")  # type: ignore
        else:
            logger.model(f"- Number of features: Not available (model not fitted)")
            
        logger.model(f"- Random state: {model_params['random_state']}")
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            logger.analysis(f"\n📊 FEATURE IMPORTANCE:")
            feature_importance = pd.DataFrame({
                'Feature': MODEL_FEATURES,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for _, row in feature_importance.iterrows():
                logger.analysis(f"- {row['Feature']}: {row['Importance']:.4f}")
        
        # First tree analysis
        if hasattr(model, 'estimators_') and model.estimators_:
            first_tree = model.estimators_[0]
            logger.model(f"\n🌲 FIRST TREE INFORMATION:")
            logger.model(f"- Nodes: {first_tree.tree_.node_count}")
            logger.model(f"- Depth: {first_tree.get_depth()}")
            logger.model(f"- Leaves: {first_tree.get_n_leaves()}")
        
        # Additional parameters
        logger.model(f"\n⚙️ OTHER PARAMETERS:")
        if hasattr(model, 'classes_'):
            logger.model(f"- Classes: {model.classes_}")
        else:
            logger.model(f"- Classes: Not available (model not fitted)")
        logger.model(f"- OOB score enabled: {model_params.get('oob_score', False)}")
        
        logger.info("=" * 80)
        logger.model("END OF MODEL INFORMATION".center(80))
        logger.info("=" * 80)
    else:
        logger.error("Model training failed!")
    
    # Test signal generation on sample pair
    processor.get_historic_data_by_symbol(DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME)
    time.sleep(DATA_PROCESSING_WAIT_TIME_IN_SECONDS)

    df = processor.df_cache.get((DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME))
    
    if df is not None and model is not None:
        signal = get_latest_random_forest_signal(df, model)
        logger.signal(f"Latest signal for {DEFAULT_TEST_SYMBOL} on {DEFAULT_TEST_TIMEFRAME}: {signal}")
    elif df is None:
        logger.error(f"Failed to retrieve data for {DEFAULT_TEST_SYMBOL} {DEFAULT_TEST_TIMEFRAME}")
    elif model is None:
        logger.error("Model is not available, cannot generate signal")

    # Cleanup and shutdown
    try:
        processor.stop()
        logger.success("Tick processor stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping tick processor: {e}")
    
    elapsed_time = time.time() - start_time
    logger.performance(f"Process completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()