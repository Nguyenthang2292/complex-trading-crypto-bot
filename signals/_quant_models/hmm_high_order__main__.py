import logging
import os
import sys
from pathlib import Path
from colorama import init
import pandas as pd
init(autoreset=True)

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent)) if str(current_dir.parent.parent) not in sys.path else None

from data_class.__class__OptimizingParametersHMM import OptimizingParametersHMM
from signals._quant_models.hmm_high_order import hmm_high_order
from utilities._logger import setup_logging
logger = setup_logging('hmm_high_order__main__', log_level=logging.DEBUG)

BULLISH, NEUTRAL, BEARISH = 1, 0, -1

if __name__ == "__main__":
    import time
    
    # Add parent directories to sys.path using pathlib
    main_dir = str(current_dir.parent.parent.parent)
    livetrade_dir = str(Path(main_dir) / 'livetrade')
    components_dir = str(Path(livetrade_dir) / '_components')
    
    # Update sys.path
    for path in [main_dir, livetrade_dir, components_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Import tick processor
    try:
        from livetrade._components._tick_processor import tick_processor
    except ImportError:
        logger.error("Failed to import tick_processor. Check the path structure.")
        sys.exit(1)
    
    # Import config constants
    try:
        if os.path.exists(os.path.join(livetrade_dir, 'config.py')):
            from livetrade.config import DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME, DATA_PROCESSING_WAIT_TIME_IN_SECONDS
        else:
            DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME, DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 'BTCUSDT', '4h', 5
    except ImportError:
        DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME, DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 'BTCUSDT', '4h', 5
    
    # Initialize tick processor
    try:
        processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
        
        # Request historic data
        symbol, timeframe = DEFAULT_TEST_SYMBOL, DEFAULT_TEST_TIMEFRAME
        logger.info(f"Requesting historic data for {symbol} / {timeframe}")
        processor.get_historic_data_by_symbol(symbol, timeframe)
        time.sleep(DATA_PROCESSING_WAIT_TIME_IN_SECONDS)
        
        # Retrieve cached data
        df = processor.df_cache.get((symbol, timeframe))
        
        if df is not None and not df.empty:
            
            # Log DataFrame info for debugging
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"DataFrame index type: {type(df.index)}")
            if len(df) > 0:
                logger.info(f"First few index values: {df.index[:3].tolist()}")
            
            # Convert open_time to datetime and set as index if it exists
            if 'open_time' in df.columns:
                try:
                    # Convert open_time to datetime
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    # Set open_time as index
                    df = df.set_index('open_time')
                    logger.info(f"Successfully converted open_time to DatetimeIndex")
                    logger.info(f"New index type: {type(df.index)}")
                    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
                except Exception as e:
                    logger.warning(f"Failed to convert open_time to datetime: {e}")
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in required_cols):
                
                # Run HMM analysis
                high_order_hmm = hmm_high_order(df, train_ratio=0.8, eval_mode=True, optimizing_params=OptimizingParametersHMM())
                
                # Map state to readable format
                state_description = {
                    BULLISH: "BULLISH (1)",
                    NEUTRAL: "NEUTRAL (0)", 
                    BEARISH: "BEARISH (-1)"
                }
                
                # Display results
                logger.info(f"\n{'='*50}")
                logger.info(f"HMM Model Prediction for {symbol} ({timeframe})")
                logger.info(f"{'='*50}")
                logger.info(f"Market State: {state_description[high_order_hmm.next_state_with_high_order_hmm]}")
                logger.info(f"Expected Duration: {high_order_hmm.next_state_duration} {timeframe} candles")
                logger.info(f"Confidence: {high_order_hmm.next_state_probability:.2f}")
                logger.info(f"{'='*50}")
            else:
                missing_cols = [col for col in required_cols if col not in df.columns]
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")

        else:
            logger.error(f"Failed to retrieve data for {symbol} {timeframe}")
            
        # Stop the tick processor
        processor.stop()
            
    except Exception as e:
        logger.error(f"Failed to initialize or use tick processor: {e}")
        logger.error("Please ensure the tick processor and related components are properly configured")

