import logging
import os
import pandas as pd
import sys
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from livetrade.config import (
    DEFAULT_TIMEFRAMES
)
    
from livetrade._components._tick_processor import tick_processor
from livetrade._components._load_all_pairs_data import load_all_pairs_data
from signals.signals_best_performance_symbols import signal_best_performance_symbols
from signals._components._process_signals_transformer import process_signals_transformer

from utilities._logger import setup_logging
# Initialize logger with improved format
logger = setup_logging(module_name="main_process_signals_transformer", log_level=logging.INFO)

DATAFRAME_COLUMNS = ['Pair', 'SignalTimeframe', 'SignalType']

def main():
    """
    Example of how to use signal_best_performance_pairs_long_short results as input for process_signals_transformer
    """
    try:
        from datetime import datetime
        
        # Initialize processor with dummy callbacks
        def mock_open_callback(data):
            logger.debug(f"Mock open: {data}")
        
        def mock_close_callback(data):
            logger.debug(f"Mock close: {data}")
        
        processor = tick_processor(mock_open_callback, mock_close_callback)
        
        # Define timeframes for analysis
        timeframes = DEFAULT_TIMEFRAMES[2:5]  # ['1h', '4h', '1d']
        
        # Step 1: Get all available symbols
        logger.data("Getting available symbols...")
        all_symbols = processor.get_symbols_list_by_quote_usdt()
        logger.data(f"Found {len(all_symbols)} available symbols")
        
        # Step 2: Pre-load data for all symbols (or a subset for testing)
        logger.data(f"Pre-loading data for {len(all_symbols)} symbols...")
        
        preloaded_data = load_all_pairs_data(
            processor=processor,
            pairs=all_symbols,
            load_multi_timeframes=True,
            timeframes=timeframes
        )
        
        if not preloaded_data:
            logger.error("Failed to load data")
            return
        
        logger.success(f"Successfully loaded data for {len(preloaded_data)} symbols")
        
        # Step 3: Use signal_best_performance_pairs_long_short to analyze and get top/worst performers
        logger.performance("Analyzing performance with signal_best_performance_pairs_long_short...")
        performance_result = signal_best_performance_symbols(
            processor=processor,
            symbols=list(preloaded_data.keys()),  # Only analyze symbols we have data for
            timeframes=timeframes,
            performance_period=48,  # Updated to match your workflow
            top_percentage=0.3,     # Top 30% performers for LONG
            worst_percentage=0.3,   # Bottom 30% performers for SHORT
            min_volume_usdt=50000,  # Updated to match your workflow
            exclude_stable_coins=True,
            include_short_signals=True,  # Enable SHORT signal analysis
            preloaded_data=preloaded_data  # Use our pre-loaded data
        )
        
        if not performance_result or 'best_performers' not in performance_result:
            logger.error("Failed to get performance analysis results")
            return
        
        # Step 4: Extract top and worst performing symbols and their data
        best_performers = performance_result['best_performers']
        worst_performers = performance_result.get('worst_performers', [])
        
        logger.performance(f"Found {len(best_performers)} top performing symbols for LONG signals")
        logger.performance(f"Found {len(worst_performers)} worst performing symbols for SHORT signals")
        
        # Create filtered data dictionaries
        top_performers_data = {}
        worst_performers_data = {}
        
        # Process top performers for LONG signals
        logger.process("Processing top performers for LONG signals...")
        for performer in tqdm(best_performers, desc="📈 Processing top performers", unit="symbol"):
            symbol = performer['symbol']
            if symbol in preloaded_data:
                top_performers_data[symbol] = preloaded_data[symbol]
                score = performer.get('composite_score', 'N/A')
                logger.debug(f"📈 LONG candidate {symbol}: Score {score}")
        
        # Process worst performers for SHORT signals
        logger.process("Processing worst performers for SHORT signals...")
        for performer in tqdm(worst_performers, desc="📉 Processing worst performers", unit="symbol"):
            symbol = performer['symbol']
            if symbol in preloaded_data:
                worst_performers_data[symbol] = preloaded_data[symbol]
                score = performer.get('composite_score', 'N/A')
                logger.debug(f"📉 SHORT candidate {symbol}: Score {score}")
        
        logger.data(f"Prepared data for {len(top_performers_data)} top performers and {len(worst_performers_data)} worst performers")
        
        # Step 5: Use the performers data as input for Transformer analysis
        logger.analysis("="*70)
        logger.analysis("RUNNING TRANSFORMER ANALYSIS")
        logger.analysis("="*70)
        
        # Analyze both LONG and SHORT signals from combined dataset
        logger.model(f"3. Analyzing both LONG and SHORT signals from combined dataset...")
    
        transformer_both_result = process_signals_transformer(
            preloaded_data=preloaded_data,
            timeframes_to_scan=timeframes,
            auto_train_if_missing=True,
            include_long_signals=True,
            include_short_signals=True
        )
        
        # Step 6: Display comprehensive results
        logger.analysis("="*80)
        logger.analysis("FINAL RESULTS - COMPREHENSIVE TRANSFORMER SIGNAL ANALYSIS")
        logger.analysis("="*80)
        
        # Display combined analysis summary
        logger.signal(f"🔄 COMBINED TRANSFORMER ANALYSIS SUMMARY ({len(transformer_both_result)} total signals):")
        logger.analysis("-" * 60)
        if not transformer_both_result.empty:
            long_count = len(transformer_both_result[transformer_both_result['SignalType'] == 'LONG'])
            short_count = len(transformer_both_result[transformer_both_result['SignalType'] == 'SHORT'])
            
            logger.signal(f"  📈 LONG signals in combined analysis: {long_count}")
            logger.signal(f"  📉 SHORT signals in combined analysis: {short_count}")
            logger.signal(f"  📊 Total signals: {len(transformer_both_result)}")
            
            # Show distribution by timeframe
            timeframe_dist = transformer_both_result['SignalTimeframe'].value_counts()
            logger.analysis(f"  📅 Signals by timeframe:")
            for tf, count in timeframe_dist.items():
                logger.analysis(f"     {tf}: {count} signals")
        else:
            logger.signal("  No signals found in combined analysis")
        
        # Extract LONG and SHORT signals from the combined results
        transformer_long_result = transformer_both_result[transformer_both_result['SignalType'] == 'LONG'] if not transformer_both_result.empty else pd.DataFrame([], columns=DATAFRAME_COLUMNS)
        transformer_short_result = transformer_both_result[transformer_both_result['SignalType'] == 'SHORT'] if not transformer_both_result.empty else pd.DataFrame([], columns=DATAFRAME_COLUMNS)
        
        logger.signal(f"\n📈 LONG signals extracted: {len(transformer_long_result)}")
        logger.signal(f"📉 SHORT signals extracted: {len(transformer_short_result)}")
        
        # Filter transformer_long_result to only include pairs from best_performers
        best_performer_symbols = [p['symbol'] for p in best_performers] if best_performers else []
        transformer_long_result_filtered = transformer_long_result[transformer_long_result['Pair'].isin(best_performer_symbols)] if not transformer_long_result.empty else transformer_long_result
        
        # Filter transformer_short_result to only include pairs from worst_performers
        worst_performer_symbols = [p['symbol'] for p in worst_performers] if worst_performers else []
        transformer_short_result_filtered = transformer_short_result[transformer_short_result['Pair'].isin(worst_performer_symbols)] if not transformer_short_result.empty else transformer_short_result
        
        # Log the filtering results
        logger.process(f"\n🔍 FILTERED SIGNALS BASED ON PERFORMANCE:")
        logger.signal(f"  • LONG signals from best performers: {len(transformer_long_result_filtered)} / {len(transformer_long_result)}")
        logger.signal(f"  • SHORT signals from worst performers: {len(transformer_short_result_filtered)} / {len(transformer_short_result)}")
        
        # Combine the filtered results into final signals dataframe
        final_signals = pd.concat([transformer_long_result_filtered, transformer_short_result_filtered])
        logger.signal(f"  • FINAL combined signals after performance filtering: {len(final_signals)}")
        
        # Step 7: Show comprehensive summary statistics
        logger.analysis("="*80)
        logger.analysis("COMPREHENSIVE TRANSFORMER SUMMARY STATISTICS")
        logger.analysis("="*80)
        
        logger.analysis(f"📊 Analysis Overview:")
        logger.data(f"  • Total symbols analyzed: {len(all_symbols)}")
        logger.data(f"  • Symbols with data loaded: {len(preloaded_data)}")
        logger.config(f"  • Performance analysis timeframes: {timeframes}")
        
        logger.performance(f"\n🎯 Performance Selection:")
        logger.performance(f"  • Top performers identified: {len(best_performers)}")
        logger.performance(f"  • Worst performers identified: {len(worst_performers)}")
        
        logger.success(f"✅ Transformer analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'performance_analysis': performance_result,
            'long_signals': transformer_long_result,
            'short_signals': transformer_short_result,
            'long_signals_filtered': transformer_long_result_filtered,
            'short_signals_filtered': transformer_short_result_filtered,
            'final_signals': final_signals,
            'combined_signals': transformer_both_result,
            'preloaded_data': preloaded_data,
            'top_performers_data': top_performers_data,
            'worst_performers_data': worst_performers_data,
            'summary_stats': {
                'total_symbols': len(all_symbols),
                'loaded_symbols': len(preloaded_data),
                'top_performers': len(best_performers),
                'worst_performers': len(worst_performers),
                'long_signals': len(transformer_long_result),
                'short_signals': len(transformer_short_result),
                'long_signals_filtered': len(transformer_long_result_filtered),
                'short_signals_filtered': len(transformer_short_result_filtered),
                'final_signals': len(final_signals),
                'combined_signals': len(transformer_both_result)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    result = main()
    
    if result:
        logger.success(f"🎉 Transformer example completed successfully!")
        logger.signal(f"Generated {result['summary_stats']['long_signals']} LONG signals and {result['summary_stats']['short_signals']} SHORT signals")
        logger.signal(f"Final filtered signals: {result['summary_stats']['final_signals']} ({result['summary_stats']['long_signals_filtered']} LONG, {result['summary_stats']['short_signals_filtered']} SHORT)")
    else:
        logger.error("❌ Transformer example failed - check logs for details")
