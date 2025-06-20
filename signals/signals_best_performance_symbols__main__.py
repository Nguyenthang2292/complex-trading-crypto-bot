import logging
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from livetrade._components._tick_processor import tick_processor
from livetrade._components._load_all_symbols_data import load_symbol_data
from livetrade.config import DEFAULT_TIMEFRAMES
from signals.signals_best_performance_symbols import (signal_best_performance_symbols, logging_performance_summary)
from utilities._logger import setup_logging

logger = setup_logging('signals_best_performance_symbols__main__', log_level=logging.DEBUG)

def main():
    """Run signal analysis for best performing pairs"""
    try:
        # Initialize processor with mock callbacks
        def mock_open_callback(data):
            print(f"Mock open: {data}")
        
        def mock_close_callback(data):
            print(f"Mock close: {data}")
        
        processor = tick_processor(mock_open_callback, mock_close_callback)
        
        # Get symbols and timeframes from config
        symbols = processor.get_symbols_list_by_quote_usdt()
        timeframes = [tf for tf in DEFAULT_TIMEFRAMES if tf in ['1h', '4h', '1d']]
        
        # Pre-load data with progress tracking
        logger.info(f"Loading data for {len(symbols)} symbols across {timeframes} timeframes...")
        symbol_data = {}
        processed_count = 0
        error_count = 0
        
        with tqdm(total=len(symbols), desc="Loading symbols", unit="symbol", ncols=100, 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for symbol in symbols:
                try:
                    data = load_symbol_data(
                        processor=processor,
                        symbol=symbol,
                        timeframes=timeframes,
                        load_multi_timeframes=True,
                    )
                    
                    if isinstance(data, dict) and data:
                        symbol_data[symbol] = data
                        processed_count += 1
                        pbar.set_postfix({"success": processed_count, "errors": error_count})
                    
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error loading data for {symbol}: {e}")
                finally:
                    pbar.update(1)
        
        logger.success(f"Successfully loaded {processed_count}/{len(symbols)} symbols ({error_count} errors)")
        
        # Run performance analysis
        result = signal_best_performance_symbols(
            processor=processor,
            symbols=symbols,
            timeframes=timeframes,
            performance_period=24,
            top_percentage=0.3,  
            worst_percentage=0.3,
            min_volume_usdt=50000,
            include_short_signals=True,
            preloaded_data=symbol_data
        )
        
        logging_performance_summary(result)
        return result
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return {} 

if __name__ == "__main__":
    main()
        

