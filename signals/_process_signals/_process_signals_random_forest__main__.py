import logging
import os
import sys

# Ensure the project root (the parent of 'livetrade') is in sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
signals_dir = os.path.dirname(current_dir)            
project_root = os.path.dirname(signals_dir)           
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from livetrade._components._tick_processor import tick_processor
from livetrade._components._load_all_pairs_data import load_all_pairs_data
from signals.signals_best_performance_symbols import signal_best_performance_pairs
from signals._components._process_signals_random_forest import process_signals_random_forest

from utilities._logger import setup_logging
logger = setup_logging(module_name="main_process_signals_random_forest", log_level=logging.DEBUG)

DATAFRAME_COLUMNS = ['Pair', 'SignalTimeframe', 'SignalType']

def main():
    """
    Example of how to use signal_best_performance_pairs_long_short results as input for process_signals_random_forest
    """
    try:
        
        # Initialize processor with dummy callbacks
        def mock_open_callback(data):
            logger.trade(f"Mock open: {data}")
        
        def mock_close_callback(data):
            logger.trade(f"Mock close: {data}")
        
        processor = tick_processor(mock_open_callback, mock_close_callback)
        
        # Define timeframes for analysis
        timeframes = ['1h', '4h', '1d']
        
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
        logger.analysis("Analyzing performance with signal_best_performance_pairs_long_short...")
        performance_result = signal_best_performance_pairs(
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
        
        logger.performance(f"Found {len(best_performers)} top performing symbols for LONG signals:")
        logger.performance(f"Found {len(worst_performers)} worst performing symbols for SHORT signals:")
        
        # Create filtered data dictionaries
        top_performers_data = {}
        worst_performers_data = {}
        
        # Process top performers for LONG signals
        for performer in best_performers:
            symbol = performer['symbol']
            if symbol in preloaded_data:
                top_performers_data[symbol] = preloaded_data[symbol]
                score = performer.get('composite_score', 'N/A')
                logger.analysis(f"  📈 LONG candidate {symbol}: Score {score}")
        
        # Process worst performers for SHORT signals
        for performer in worst_performers:
            symbol = performer['symbol']
            if symbol in preloaded_data:
                worst_performers_data[symbol] = preloaded_data[symbol]
                score = performer.get('composite_score', 'N/A')
                logger.analysis(f"  📉 SHORT candidate {symbol}: Score {score}")
        
        logger.data(f"Prepared data for {len(top_performers_data)} top performers and {len(worst_performers_data)} worst performers")
        
        # Step 5: Use the performers data as input for Random Forest analysis
        logger.model("\n" + "="*70)
        logger.model("RUNNING RANDOM FOREST ANALYSIS")
        logger.model("="*70)
        
        # Analyze LONG signals from top performers
        logger.model(f"\n1. Analyzing LONG signals from {len(top_performers_data)} top performers...")
        rf_long_result = process_signals_random_forest(
            preloaded_data=top_performers_data,
            timeframes_to_scan=timeframes,
            auto_train_if_missing=True,  # Allow model training if needed
            include_long_signals=True,
            include_short_signals=False
        )
        
        # Analyze SHORT signals from worst performers
        logger.model(f"\n2. Analyzing SHORT signals from {len(worst_performers_data)} worst performers...")
        rf_short_result = process_signals_random_forest(
            preloaded_data=worst_performers_data,
            timeframes_to_scan=timeframes,
            auto_train_if_missing=True,
            include_long_signals=False,
            include_short_signals=True
        )
        
        # Analyze both LONG and SHORT signals from combined dataset
        logger.model(f"\n3. Analyzing both LONG and SHORT signals from combined dataset...")
        
        # Combine top and worst performers for comprehensive analysis
        combined_data = {**top_performers_data, **worst_performers_data}
        logger.analysis(f"Analyzing {len(combined_data)} symbols for both signal types...")
        
        rf_both_result = process_signals_random_forest(
            preloaded_data=combined_data,
            timeframes_to_scan=timeframes,
            auto_train_if_missing=True,
            include_long_signals=True,
            include_short_signals=True
        )
        
        # Step 6: Display comprehensive results
        logger.performance("\n" + "="*80)
        logger.performance("FINAL RESULTS - COMPREHENSIVE SIGNAL ANALYSIS")
        logger.performance("="*80)
        
        # Display LONG signals
        logger.signal(f"\n🟢 LONG SIGNALS FROM TOP PERFORMERS ({len(rf_long_result)} found):")
        logger.performance("-" * 60)
        if not rf_long_result.empty:
            for idx, row in rf_long_result.iterrows():
                performer_info = next((p for p in best_performers if p['symbol'] == row['Pair']), {})
                score = performer_info.get('composite_score', 'N/A')
                tf_scores = performer_info.get('timeframe_scores', {})
                consistency = performer_info.get('score_consistency', 'N/A')
                
                # Get volume info from timeframe analysis
                volume = 'N/A'
                timeframe_used = row['SignalTimeframe']
                
                if 'timeframe_analysis' in performance_result:
                    tf_analysis = performance_result['timeframe_analysis'].get(timeframe_used, {})
                    symbol_metrics = tf_analysis.get('symbol_metrics', [])
                    symbol_data = next((m for m in symbol_metrics if m['symbol'] == row['Pair']), {})
                    if symbol_data:
                        volume = symbol_data.get('avg_volume_usdt', 'N/A')
                        if isinstance(volume, (int, float)):
                            volume = f"{volume:,.0f} USDT"
                
                logger.signal(f"  📈 {row['Pair']}: {row['SignalType']} on {row['SignalTimeframe']}")
                logger.performance(f"     Performance Score: {score}")
                logger.data(f"     Avg Volume (24h): {volume}")
                logger.analysis(f"     Score Consistency: {consistency}")
                logger.analysis(f"     Timeframe Scores: {tf_scores}")
        else:
            logger.signal("  No LONG signals found from top performers")
        
        # Display SHORT signals
        logger.signal(f"\n🔴 SHORT SIGNALS FROM WORST PERFORMERS ({len(rf_short_result)} found):")
        logger.performance("-" * 60)
        if not rf_short_result.empty:
            for idx, row in rf_short_result.iterrows():
                performer_info = next((p for p in worst_performers if p['symbol'] == row['Pair']), {})
                score = performer_info.get('composite_score', 'N/A')
                tf_scores = performer_info.get('timeframe_scores', {})
                consistency = performer_info.get('score_consistency', 'N/A')
                
                # Get volume info from timeframe analysis
                volume = 'N/A'
                timeframe_used = row['SignalTimeframe']
                
                if 'timeframe_analysis' in performance_result:
                    tf_analysis = performance_result['timeframe_analysis'].get(timeframe_used, {})
                    symbol_metrics = tf_analysis.get('symbol_metrics', [])
                    symbol_data = next((m for m in symbol_metrics if m['symbol'] == row['Pair']), {})
                    if symbol_data:
                        volume = symbol_data.get('avg_volume_usdt', 'N/A')
                        if isinstance(volume, (int, float)):
                            volume = f"{volume:,.0f} USDT"
                
                logger.signal(f"  📉 {row['Pair']}: {row['SignalType']} on {row['SignalTimeframe']}")
                logger.performance(f"     Performance Score: {score}")
                logger.data(f"     Avg Volume (24h): {volume}")
                logger.analysis(f"     Score Consistency: {consistency}")
                logger.analysis(f"     Timeframe Scores: {tf_scores}")
        else:
            logger.signal("  No SHORT signals found from worst performers")
        
        # Display combined analysis summary
        logger.analysis(f"\n🔄 COMBINED ANALYSIS SUMMARY ({len(rf_both_result)} total signals):")
        logger.performance("-" * 60)
        if not rf_both_result.empty:
            long_count = len(rf_both_result[rf_both_result['SignalType'] == 'LONG'])
            short_count = len(rf_both_result[rf_both_result['SignalType'] == 'SHORT'])
            
            logger.signal(f"  📈 LONG signals in combined analysis: {long_count}")
            logger.signal(f"  📉 SHORT signals in combined analysis: {short_count}")
            logger.performance(f"  📊 Total signals: {len(rf_both_result)}")
            
            # Show distribution by timeframe
            timeframe_dist = rf_both_result['SignalTimeframe'].value_counts()
            logger.analysis(f"  📅 Signals by timeframe:")
            for tf, count in timeframe_dist.items():
                logger.analysis(f"     {tf}: {count} signals")
                
            # Show distribution by signal type and timeframe
            logger.analysis(f"  📊 Detailed breakdown:")
            for signal_type in ['LONG', 'SHORT']:
                type_data = rf_both_result[rf_both_result['SignalType'] == signal_type]
                if not type_data.empty:
                    tf_dist = type_data['SignalTimeframe'].value_counts()
                    logger.analysis(f"     {signal_type}: {tf_dist.to_dict()}")
        else:
            logger.signal("  No signals found in combined analysis")
        
        # Step 7: Show comprehensive summary statistics
        logger.performance("\n" + "="*80)
        logger.performance("COMPREHENSIVE SUMMARY STATISTICS")
        logger.performance("="*80)
        
        logger.data(f"📊 Analysis Overview:")
        logger.data(f"  • Total symbols analyzed: {len(all_symbols)}")
        logger.data(f"  • Symbols with data loaded: {len(preloaded_data)}")
        logger.config(f"  • Performance analysis timeframes: {timeframes}")
        logger.config(f"  • Performance period: {performance_result.get('summary', {}).get('performance_period', 48)} periods")
        logger.config(f"  • Minimum volume filter: {performance_result.get('summary', {}).get('min_volume_usdt', 50000):,} USDT")
        
        logger.analysis(f"\n🎯 Performance Selection:")
        logger.analysis(f"  • Top performers identified: {len(best_performers)}")
        logger.analysis(f"  • Worst performers identified: {len(worst_performers)}")
        logger.config(f"  • Top percentage threshold: {performance_result.get('summary', {}).get('top_percentage', 0.3)*100:.1f}%")
        logger.config(f"  • Bottom percentage threshold: {performance_result.get('summary', {}).get('short_percentage', 0.3)*100:.1f}%")
        
        logger.model(f"\n🤖 Random Forest Results:")
        logger.signal(f"  • LONG signals from top performers: {len(rf_long_result)}")
        logger.signal(f"  • SHORT signals from worst performers: {len(rf_short_result)}")
        logger.signal(f"  • Combined signals total: {len(rf_both_result)}")
        
        logger.performance(f"\n📈 Signal Efficiency:")
        if len(top_performers_data) > 0:
            long_signal_rate = (len(rf_long_result) / len(top_performers_data)) * 100
            logger.performance(f"  • LONG signal rate from top performers: {long_signal_rate:.1f}%")
        
        if len(worst_performers_data) > 0:
            short_signal_rate = (len(rf_short_result) / len(worst_performers_data)) * 100
            logger.performance(f"  • SHORT signal rate from worst performers: {short_signal_rate:.1f}%")
        
        if len(combined_data) > 0:
            combined_signal_rate = (len(rf_both_result) / len(combined_data)) * 100
            logger.performance(f"  • Combined signal rate: {combined_signal_rate:.1f}%")
        
        logger.success(f"\n✅ Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'performance_analysis': performance_result,
            'long_signals': rf_long_result,
            'short_signals': rf_short_result,
            'combined_signals': rf_both_result,
            'preloaded_data': preloaded_data,
            'top_performers_data': top_performers_data,
            'worst_performers_data': worst_performers_data,
            'summary_stats': {
                'total_symbols': len(all_symbols),
                'loaded_symbols': len(preloaded_data),
                'top_performers': len(best_performers),
                'worst_performers': len(worst_performers),
                'long_signals': len(rf_long_result),
                'short_signals': len(rf_short_result),
                'combined_signals': len(rf_both_result)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Import datetime for timestamp
    from datetime import datetime
    
    # Run example
    result = main()
    
    if result:
        logger.success(f"\n🎉 Example completed successfully!")
        logger.performance(f"Generated {result['summary_stats']['long_signals']} LONG signals and {result['summary_stats']['short_signals']} SHORT signals")
    else:
        logger.error("❌ Example failed - check logs for details")