import logging
import os
from pathlib import Path
import sys
import warnings

# Fix KMeans memory leak on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent)) if str(current_dir.parent.parent) not in sys.path else None

from signals._components.HMM__class__OptimizingParameters import OptimizingParameters
from signals._quant_models.hmm_kama import hmm_kama
from utilities._logger import setup_logging
logger = setup_logging('hmm_kama__main__', log_level=logging.DEBUG)

if __name__ == "__main__":
    import time
    
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    livetrade_dir = os.path.join(main_dir, 'components')
    components_dir = os.path.join(livetrade_dir, '_components')
    
    for path in [main_dir, livetrade_dir, components_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    try:
        from components._components._tick_processor import tick_processor
    except ImportError:
        logger.error("Failed to import tick_processor. Check the path structure.")
        sys.exit(1)
    
    config_path = os.path.join(livetrade_dir, 'config.py')
    if os.path.exists(config_path):
        try:
            from components.config import (
                DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME,
                DATA_PROCESSING_WAIT_TIME_IN_SECONDS
            )
        except ImportError:
            logger.config("Could not import config values. Using defaults.")
            DEFAULT_TEST_SYMBOL = 'BTCUSDT'
            DEFAULT_TEST_TIMEFRAME = '4h'
            DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 5
    else:
        logger.config(f"Config file not found at {config_path}. Using default values.")
        DEFAULT_TEST_SYMBOL = 'BTCUSDT'
        DEFAULT_TEST_TIMEFRAME = '4h'
        DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 5
    
    # Initialize tick processor for crypto data
    try:
        processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
        logger.network(f"Successfully initialized Crypto TickProcessor with Binance API")
        
        # Request historic data
        symbol = DEFAULT_TEST_SYMBOL
        timeframe = DEFAULT_TEST_TIMEFRAME
        logger.network(f"Requesting historic data for {symbol} / {timeframe}")
        
        processor.get_historic_data_by_symbol(symbol, timeframe)
        
        # Wait for data to be processed
        time.sleep(DATA_PROCESSING_WAIT_TIME_IN_SECONDS)
        
        # Retrieve cached data
        df = processor.df_cache.get((symbol, timeframe))
        
        if df is not None and not df.empty:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in required_cols):
                
                logger.data(f"Loaded {len(df)} rows of {symbol} {timeframe} data for HMM-KAMA analysis")
                logger.data(f"Data range: {df.index[0]} to {df.index[-1]}")
                
                # Run HMM-KAMA analysis
                hmm_state = hmm_kama(df, OptimizingParameters())
                
                logger.analysis(f"HMM-KAMA Results for {symbol}:")
                for field, value in vars(hmm_state).items():
                    logger.info(f"  {field}: {value}")
                
                # Print readable summary
                state_descriptions = {
                    0: "Bearish Weak",
                    1: "Bullish Weak",
                    2: "Bearish Strong",
                    3: "Bullish Strong"
                }
                
                logger.analysis(f"\n{'='*50}")
                logger.analysis(f"HMM-KAMA Model Prediction for {symbol} ({timeframe})")
                logger.analysis(f"{'='*50}")
                logger.signal(f"Next State: {state_descriptions.get(hmm_state.next_state_with_hmm_kama, 'Unknown')}")
                logger.info(f"Current state using std: {'Preparing to switch' if hmm_state.current_state_of_state_using_std == 1 else 'Just entered'}")
                logger.info(f"Current state using HMM: {'Preparing to switch' if hmm_state.current_state_of_state_using_hmm == 1 else 'Just entered'}")
                logger.info(f"State probabilities (apriori): {state_descriptions.get(hmm_state.state_high_probabilities_using_arm_apriori, 'Unknown')}")
                logger.info(f"State probabilities (fpgrowth): {state_descriptions.get(hmm_state.state_high_probabilities_using_arm_fpgrowth, 'Unknown')}")
                logger.info(f"Current state using KMeans: {'Preparing to switch' if hmm_state.current_state_of_state_using_kmeans == 1 else 'Just entered'}")
                logger.analysis(f"{'='*50}")
                
            else:
                missing_cols = [col for col in required_cols if col not in df.columns]
                logger.error(f"Missing required columns in DataFrame: {missing_cols}")
                logger.error(f"Available columns: {list(df.columns)}")
        else:
            logger.error(f"Failed to retrieve data for {symbol} {timeframe}")
            
        # Stop the tick processor
        try:
            processor.stop()
            logger.success("Tick processor stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping tick processor: {e}")
            
    except Exception as e:
        logger.error(f"Failed to initialize or use tick processor: {e}")
        logger.error("Please ensure the tick processor and related components are properly configured")
