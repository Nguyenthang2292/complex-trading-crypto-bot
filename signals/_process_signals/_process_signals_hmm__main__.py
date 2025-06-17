import logging
import os
import sys
from typing import Dict

# Try to import psutil for CPU information, fallback to os if not available
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    import os
    _HAS_PSUTIL = False

# Ensure the project root (the parent of 'livetrade') is in sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
signals_dir = os.path.dirname(current_dir)            
project_root = os.path.dirname(signals_dir)           
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from livetrade._components._load_all_pairs_data import load_all_pairs_data
from signals._components._process_signals_hmm import (reload_timeframes_for_symbols, process_signals_hmm)

from utilities._logger import setup_logging
logger = setup_logging(module_name="main_process_signals_hmm", log_level=logging.DEBUG)

DATAFRAME_COLUMNS = ['Pair', 'SignalTimeframe', 'SignalType']

def main():
    """
    Example of how to use signal_best_performance_pairs_long_short results as input for process_signals_hmm
    """
    try:
        from livetrade._components._tick_processor import tick_processor
        from signals.signals_best_performance_pairs import signal_best_performance_pairs
        from datetime import datetime
        
        # Initialize processor with dummy callbacks
        def mock_open_callback(data):
            logger.trade(f"Mock open: {data}")
        
        def mock_close_callback(data):
            logger.trade(f"Mock close: {data}")
        
        processor = tick_processor(mock_open_callback, mock_close_callback)
        
        # Define timeframes for analysis
        timeframes = ['1h', '4h', '1d']
        hmm_timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        
        # Step 1: Get all available symbols
        logger.network("Getting available symbols...")
        all_symbols = processor.get_symbols_list_by_quote_usdt()
        logger.success(f"Found {len(all_symbols)} available symbols")
        
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
        
        # Step 3: Use signal_best_performance_pairs_long_short to analyze and get both top/worst performers
        logger.analysis("Analyzing performance with signal_best_performance_pairs_long_short...")
        performance_result = signal_best_performance_pairs(
            processor=processor,
            symbols=list(preloaded_data.keys()),  # Only analyze symbols we have data for
            timeframes=timeframes,
            performance_period=48,  # Updated to match main workflow
            top_percentage=0.3,     # Top 30% performers for LONG
            worst_percentage=0.3,   # Bottom 30% performers for SHORT
            min_volume_usdt=50000,  # Updated to match main workflow
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
        
        logger.success(f"Found {len(best_performers)} top performing symbols for LONG signals:")
        logger.success(f"Found {len(worst_performers)} worst performing symbols for SHORT signals:")
        
        # Extract symbol lists
        best_performer_symbols = [performer['symbol'] for performer in best_performers]
        worst_performer_symbols = [performer['symbol'] for performer in worst_performers]
        
        # Step 4.1: Reload specific timeframes for both top and worst performers for better HMM analysis
        all_candidate_symbols = list(set(best_performer_symbols + worst_performer_symbols))
        logger.data(f"Reloading HMM analysis timeframes for {len(all_candidate_symbols)} candidates...")
        
        # Reload data with specific timeframes optimized for HMM analysis
        candidates_hmm_data = reload_timeframes_for_symbols(processor, all_candidate_symbols, hmm_timeframes)
        
        # Separate data for top and worst performers
        top_performers_data = {symbol: candidates_hmm_data[symbol] for symbol in best_performer_symbols if symbol in candidates_hmm_data}
        worst_performers_data = {symbol: candidates_hmm_data[symbol] for symbol in worst_performer_symbols if symbol in candidates_hmm_data}
        
        # Log the results for top performers
        logger.analysis("\n📈 TOP PERFORMERS FOR LONG SIGNALS:")
        for performer in best_performers:
            symbol = performer['symbol']
            score = performer.get('composite_score', 'N/A')
            tf_scores = performer.get('timeframe_scores', {})
            consistency = performer.get('score_consistency', 'N/A')
            
            if symbol in top_performers_data:
                logger.success(f"  ✅ {symbol}: Score {score:.4f} - HMM data loaded")
                logger.info(f"     Timeframe Scores: {tf_scores}")
                logger.info(f"     Score Consistency: {consistency:.4f}")
            else:
                logger.error(f"  ❌ {symbol}: Score {score:.4f} - HMM data failed to load")
        
        # Log the results for worst performers
        logger.analysis("\n📉 WORST PERFORMERS FOR SHORT SIGNALS:")
        for performer in worst_performers:
            symbol = performer['symbol']
            score = performer.get('composite_score', 'N/A')
            tf_scores = performer.get('timeframe_scores', {})
            consistency = performer.get('score_consistency', 'N/A')
            
            if symbol in worst_performers_data:
                logger.success(f"  ✅ {symbol}: Score {score:.4f} - HMM data loaded")
                logger.info(f"     Timeframe Scores: {tf_scores}")
                logger.info(f"     Score Consistency: {consistency:.4f}")
            else:
                logger.error(f"  ❌ {symbol}: Score {score:.4f} - HMM data failed to load")
        
        logger.success(f"\nPrepared HMM data for {len(top_performers_data)} top performers and {len(worst_performers_data)} worst performers")
        
        # Step 5: Use the performers data as input for comprehensive HMM analysis
        logger.model("\n" + "="*80)
        logger.model("RUNNING COMPREHENSIVE HMM ANALYSIS")
        logger.model("="*80)
        
        # Test both strict and non-strict modes with different signal combinations
        for strict_mode in [False, True]:
            mode_name = "STRICT" if strict_mode else "NON-STRICT"
            logger.config(f"\n{'='*60}")
            logger.config(f"{mode_name} HMM MODE")
            logger.config('='*60)
            
            # 1. LONG signals only from top performers
            logger.signal(f"\n--- LONG SIGNALS FROM TOP PERFORMERS ({mode_name}) ---")
            long_result = process_signals_hmm(
                preloaded_data=top_performers_data,
                timeframes_to_scan=hmm_timeframes,
                strict_mode=strict_mode,
                include_long_signals=True,
                include_short_signals=False,
                max_workers=4
            )
            logger.signal(f"Found {len(long_result)} LONG signals from top performers")
            
            # 2. SHORT signals only from worst performers
            logger.signal(f"\n--- SHORT SIGNALS FROM WORST PERFORMERS ({mode_name}) ---")
            short_result = process_signals_hmm(
                preloaded_data=worst_performers_data,
                timeframes_to_scan=hmm_timeframes,
                strict_mode=strict_mode,
                include_long_signals=False,
                include_short_signals=True,
                max_workers=4
            )
            logger.signal(f"Found {len(short_result)} SHORT signals from worst performers")
            
            # 3. Both LONG and SHORT signals from combined dataset
            logger.analysis(f"\n--- COMBINED ANALYSIS: BOTH LONG & SHORT SIGNALS ({mode_name}) ---")
            combined_data = {**top_performers_data, **worst_performers_data}
            both_result = process_signals_hmm(
                preloaded_data=combined_data,
                timeframes_to_scan=hmm_timeframes,
                strict_mode=strict_mode,
                include_long_signals=True,
                include_short_signals=True,
                max_workers=4
            )
            
            # Analyze combined results
            if not both_result.empty:
                long_count = len(both_result[both_result['SignalType'] == 'LONG'])
                short_count = len(both_result[both_result['SignalType'] == 'SHORT'])
                logger.success(f"Found {len(both_result)} total signals: {long_count} LONG, {short_count} SHORT")
                
                logger.analysis(f"\n📊 DETAILED SIGNAL BREAKDOWN ({mode_name} MODE):")
                logger.info("-" * 80)
                
                # Display LONG signals with performance context
                long_signals = both_result[both_result['SignalType'] == 'LONG']
                if not long_signals.empty:
                    logger.signal(f"\n🟢 LONG SIGNALS ({len(long_signals)} found):")
                    for idx, row in long_signals.iterrows():
                        performer_info = next((p for p in best_performers if p['symbol'] == row['Pair']), {})
                        score = performer_info.get('composite_score', 'N/A')
                        tf_scores = performer_info.get('timeframe_scores', {})
                        consistency = performer_info.get('score_consistency', 'N/A')
                        
                        # Get volume from timeframe analysis
                        volume = _get_volume_info(performance_result, row['SignalTimeframe'], row['Pair'])
                        
                        logger.trade(f"  📈 {row['Pair']}: {row['SignalType']} on {row['SignalTimeframe']}")
                        logger.performance(f"     Performance Score: {score}")
                        logger.data(f"     Avg Volume (24h): {volume}")
                        logger.info(f"     Score Consistency: {consistency}")
                        logger.info(f"     Timeframe Scores: {tf_scores}")
                        logger.info("")
                
                # Display SHORT signals with performance context
                short_signals = both_result[both_result['SignalType'] == 'SHORT']
                if not short_signals.empty:
                    logger.signal(f"\n🔴 SHORT SIGNALS ({len(short_signals)} found):")
                    for idx, row in short_signals.iterrows():
                        performer_info = next((p for p in worst_performers if p['symbol'] == row['Pair']), {})
                        score = performer_info.get('composite_score', 'N/A')
                        tf_scores = performer_info.get('timeframe_scores', {})
                        consistency = performer_info.get('score_consistency', 'N/A')
                        
                        # Get volume from timeframe analysis
                        volume = _get_volume_info(performance_result, row['SignalTimeframe'], row['Pair'])
                        
                        logger.trade(f"  📉 {row['Pair']}: {row['SignalType']} on {row['SignalTimeframe']}")
                        logger.performance(f"     Performance Score: {score}")
                        logger.data(f"     Avg Volume (24h): {volume}")
                        logger.info(f"     Score Consistency: {consistency}")
                        logger.info(f"     Timeframe Scores: {tf_scores}")
                        logger.info("")
                
                # Show distribution by timeframe
                logger.analysis(f"📅 SIGNAL DISTRIBUTION BY TIMEFRAME:")
                timeframe_dist = both_result['SignalTimeframe'].value_counts()
                for tf, count in timeframe_dist.items():
                    tf_long = len(both_result[(both_result['SignalTimeframe'] == tf) & (both_result['SignalType'] == 'LONG')])
                    tf_short = len(both_result[(both_result['SignalTimeframe'] == tf) & (both_result['SignalType'] == 'SHORT')])
                    logger.signal(f"  {tf}: {count} total ({tf_long} LONG, {tf_short} SHORT)")
            else:
                logger.warning(f"  No signals found in {mode_name} mode")
        
        # Step 6: Show comprehensive summary statistics
        logger.performance("\n" + "="*80)
        logger.performance("COMPREHENSIVE SUMMARY STATISTICS")
        logger.performance("="*80)
        
        logger.analysis(f"📊 Analysis Overview:")
        logger.info(f"  • Total symbols analyzed: {len(all_symbols)}")
        logger.info(f"  • Symbols with performance data: {len(preloaded_data)}")
        logger.config(f"  • Performance analysis timeframes: {timeframes}")
        logger.config(f"  • HMM analysis timeframes: {hmm_timeframes}")
        logger.config(f"  • Performance period: {performance_result.get('summary', {}).get('performance_period', 48)} periods")
        logger.config(f"  • Minimum volume filter: {performance_result.get('summary', {}).get('min_volume_usdt', 50000):,} USDT")
        
        logger.analysis(f"\n🎯 Performance Selection:")
        logger.info(f"  • Top performers identified: {len(best_performers)}")
        logger.info(f"  • Worst performers identified: {len(worst_performers)}")
        logger.info(f"  • Top percentage threshold: {performance_result.get('summary', {}).get('top_percentage', 0.3)*100:.1f}%")
        logger.info(f"  • Bottom percentage threshold: {performance_result.get('summary', {}).get('short_percentage', 0.3)*100:.1f}%")
        
        logger.data(f"\n🔄 HMM Data Preparation:")
        logger.info(f"  • Symbols with HMM data loaded: {len(candidates_hmm_data)}")
        logger.info(f"  • Top performers with HMM data: {len(top_performers_data)}")
        logger.info(f"  • Worst performers with HMM data: {len(worst_performers_data)}")
        
        logger.model(f"\n🤖 HMM Analysis Capabilities Demonstrated:")
        logger.success(f"  ✓ Support for both LONG and SHORT signals")
        logger.success(f"  ✓ Configurable signal type inclusion via parameters")
        logger.success(f"  ✓ Enhanced logging with signal type breakdown")
        logger.success(f"  ✓ Backward compatibility maintained")
        logger.success(f"  ✓ Flexible parameter-based signal filtering")
        logger.success(f"  ✓ Performance-based candidate pre-filtering")
        logger.success(f"  ✓ Multi-timeframe HMM analysis")
        
        logger.success(f"\n✅ Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'performance_analysis': performance_result,
            'preloaded_data': preloaded_data,
            'candidates_hmm_data': candidates_hmm_data,
            'top_performers_data': top_performers_data,
            'worst_performers_data': worst_performers_data,
            'summary_stats': {
                'total_symbols': len(all_symbols),
                'loaded_symbols': len(preloaded_data),
                'top_performers': len(best_performers),
                'worst_performers': len(worst_performers),
                'hmm_candidates': len(candidates_hmm_data),
                'top_performers_with_hmm': len(top_performers_data),
                'worst_performers_with_hmm': len(worst_performers_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return None

def _get_volume_info(performance_result: Dict, timeframe: str, symbol: str) -> str:
    """Helper function to extract volume information from performance analysis"""
    try:
        if 'timeframe_analysis' in performance_result:
            tf_analysis = performance_result['timeframe_analysis'].get(timeframe, {})
            symbol_metrics = tf_analysis.get('symbol_metrics', [])
            symbol_data = next((m for m in symbol_metrics if m['symbol'] == symbol), {})
            if symbol_data:
                volume = symbol_data.get('avg_volume_usdt', 'N/A')
                if isinstance(volume, (int, float)):
                    return f"{volume:,.0f} USDT"
        return 'N/A'
    except:
        return 'N/A'

if __name__ == "__main__":
    # Run example
    result = main()
    
    if result:
        logger.success(f"\n🎉 HMM Analysis Example completed successfully!")
        logger.performance(f"Summary: {result['summary_stats']}")
    else:
        logger.error("❌ Example failed - check logs for details")