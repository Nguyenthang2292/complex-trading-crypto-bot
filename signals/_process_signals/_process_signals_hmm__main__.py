import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    import os
    _HAS_PSUTIL = False

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from livetrade._components._load_all_symbols_data import load_all_symbols_data
from signals._process_signals._process_signals_hmm import process_signals_hmm, reload_timeframes_for_symbols

from utilities._logger import setup_logging
logger = setup_logging(module_name="_process_signals_hmm__main__", log_level=logging.DEBUG)

DATAFRAME_COLUMNS: List[str] = ['Symbol', 'SignalTimeframe', 'SignalType']

def _get_volume_info(performance_result: Dict[str, Any], timeframe: str, symbol: str) -> str:
    """Extract volume information from performance analysis results."""
    try:
        tf_analysis = performance_result.get('timeframe_analysis', {}).get(timeframe, {})
        symbol_metrics = tf_analysis.get('symbol_metrics', [])
        symbol_data = next((m for m in symbol_metrics if m['symbol'] == symbol), {})
        volume = symbol_data.get('avg_volume_usdt', 'N/A')
        return f"{volume:,.0f} USDT" if isinstance(volume, (int, float)) else 'N/A'
    except Exception:
        return 'N/A'

def main() -> Optional[Dict[str, Any]]:
    """
    Main function demonstrating HMM signal analysis with performance-based filtering.
    
    Workflow:
    1. Load all available symbols and their market data
    2. Analyze performance to identify top/worst performers
    3. Reload HMM-optimized timeframes for candidates
    4. Run HMM analysis in both strict and non-strict modes
    5. Generate comprehensive signal reports
    
    Returns:
        Dictionary containing analysis results and summary statistics, None if failed
    """
    try:
        from livetrade._components._tick_processor import tick_processor
        from signals.signals_best_performance_symbols import signal_best_performance_symbols
        
        def mock_open_callback(data: Any) -> None:
            logger.trade(f"Mock open: {data}")
        
        def mock_close_callback(data: Any) -> None:
            logger.trade(f"Mock close: {data}")
        
        processor = tick_processor(mock_open_callback, mock_close_callback)
        
        timeframes: List[str] = ['1h', '4h', '1d']
        hmm_timeframes: List[str] = ['5m', '15m', '30m', '1h', '4h', '1d']
        
        logger.network("Getting available symbols...")
        all_symbols: List[str] = processor.get_symbols_list_by_quote_usdt()
        logger.success(f"Found {len(all_symbols)} available symbols")
        
        logger.data(f"Pre-loading data for {len(all_symbols)} symbols...")
        preloaded_data: Dict[str, Dict[str, Any]] = load_all_symbols_data(
            processor=processor,
            symbols=all_symbols,
            load_multi_timeframes=True,
            timeframes=timeframes
        )
        
        if not preloaded_data:
            logger.error("Failed to load data")
            return None
        
        logger.success(f"Successfully loaded data for {len(preloaded_data)} symbols")
        
        logger.analysis("Analyzing performance with signal_best_performance_symbols...")
        performance_result: Dict[str, Any] = signal_best_performance_symbols(
            processor=processor,
            symbols=list(preloaded_data.keys()),
            timeframes=timeframes,
            performance_period=48,
            top_percentage=0.1,
            worst_percentage=0.1,
            min_volume_usdt=50000,
            exclude_stable_coins=True,
            include_short_signals=True,
            preloaded_data=preloaded_data
        )
        
        if not performance_result or 'best_performers' not in performance_result:
            logger.error("Failed to get performance analysis results")
            return None
        
        best_performers: List[Dict[str, Any]] = performance_result['best_performers']
        worst_performers: List[Dict[str, Any]] = performance_result.get('worst_performers', [])
        
        logger.success(f"Found {len(best_performers)} top performing symbols for LONG signals:")
        logger.success(f"Found {len(worst_performers)} worst performing symbols for SHORT signals:")
        
        best_performer_symbols: List[str] = [performer['symbol'] for performer in best_performers]
        worst_performer_symbols: List[str] = [performer['symbol'] for performer in worst_performers]
        
        all_candidate_symbols: List[str] = list(set(best_performer_symbols + worst_performer_symbols))
        logger.data(f"Reloading HMM analysis timeframes for {len(all_candidate_symbols)} candidates...")
        
        candidates_hmm_data: Dict[str, Dict[str, Any]] = reload_timeframes_for_symbols(
            processor, all_candidate_symbols, hmm_timeframes
        )
        
        top_performers_data: Dict[str, Dict[str, Any]] = {
            symbol: candidates_hmm_data[symbol] 
            for symbol in best_performer_symbols 
            if symbol in candidates_hmm_data
        }
        worst_performers_data: Dict[str, Dict[str, Any]] = {
            symbol: candidates_hmm_data[symbol] 
            for symbol in worst_performer_symbols 
            if symbol in candidates_hmm_data
        }
        
        logger.analysis("\n📈 TOP PERFORMERS FOR LONG SIGNALS:")
        for performer in best_performers:
            symbol: str = performer['symbol']
            score: float = performer.get('composite_score', 0.0)
            tf_scores: Dict[str, float] = performer.get('timeframe_scores', {})
            consistency: float = performer.get('score_consistency', 0.0)
            
            status = "✅" if symbol in top_performers_data else "❌"
            data_status = "HMM data loaded" if symbol in top_performers_data else "HMM data failed to load"
            logger.success(f"  {status} {symbol}: Score {score:.4f} - {data_status}")
            logger.info(f"     Timeframe Scores: {tf_scores}")
            logger.info(f"     Score Consistency: {consistency:.4f}")
        
        logger.analysis("\n📉 WORST PERFORMERS FOR SHORT SIGNALS:")
        for performer in worst_performers:
            symbol: str = performer['symbol']
            score: float = performer.get('composite_score', 0.0)
            tf_scores: Dict[str, float] = performer.get('timeframe_scores', {})
            consistency: float = performer.get('score_consistency', 0.0)
            
            status = "✅" if symbol in worst_performers_data else "❌"
            data_status = "HMM data loaded" if symbol in worst_performers_data else "HMM data failed to load"
            logger.success(f"  {status} {symbol}: Score {score:.4f} - {data_status}")
            logger.info(f"     Timeframe Scores: {tf_scores}")
            logger.info(f"     Score Consistency: {consistency:.4f}")
        
        logger.success(f"\nPrepared HMM data for {len(top_performers_data)} top performers and {len(worst_performers_data)} worst performers")
        
        logger.model("\n" + "="*80)
        logger.model("RUNNING COMPREHENSIVE HMM ANALYSIS")
        logger.model("="*80)
        
        for strict_mode in [False, True]:
            mode_name: str = "STRICT" if strict_mode else "NON-STRICT"
            logger.config(f"\n{'='*60}")
            logger.config(f"{mode_name} HMM MODE")
            logger.config('='*60)
            
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
            
            logger.analysis(f"\n--- COMBINED ANALYSIS: BOTH LONG & SHORT SIGNALS ({mode_name}) ---")
            combined_data: Dict[str, Dict[str, Any]] = {**top_performers_data, **worst_performers_data}
            both_result = process_signals_hmm(
                preloaded_data=combined_data,
                timeframes_to_scan=hmm_timeframes,
                strict_mode=strict_mode,
                include_long_signals=True,
                include_short_signals=True,
                max_workers=4
            )
            
            if not both_result.empty:
                long_count: int = len(both_result[both_result['SignalType'] == 'LONG'])
                short_count: int = len(both_result[both_result['SignalType'] == 'SHORT'])
                logger.success(f"Found {len(both_result)} total signals: {long_count} LONG, {short_count} SHORT")
                
                logger.analysis(f"\n📊 DETAILED SIGNAL BREAKDOWN ({mode_name} MODE):")
                logger.info("-" * 80)
                
                long_signals = both_result[both_result['SignalType'] == 'LONG']
                if not long_signals.empty:
                    logger.signal(f"\n🟢 LONG SIGNALS ({len(long_signals)} found):")
                    for idx, row in long_signals.iterrows():
                        performer_info: Dict[str, Any] = next(
                            (p for p in best_performers if p['symbol'] == row['Symbol']), {}
                        )
                        score: Any = performer_info.get('composite_score', 'N/A')
                        tf_scores: Dict[str, float] = performer_info.get('timeframe_scores', {})
                        consistency: Any = performer_info.get('score_consistency', 'N/A')
                        volume: str = _get_volume_info(performance_result, row['SignalTimeframe'], row['Symbol'])
                        
                        logger.trade(f"  📈 {row['Symbol']}: {row['SignalType']} on {row['SignalTimeframe']}")
                        logger.performance(f"     Performance Score: {score}")
                        logger.data(f"     Avg Volume (24h): {volume}")
                        logger.info(f"     Score Consistency: {consistency}")
                        logger.info(f"     Timeframe Scores: {tf_scores}")
                        logger.info("")
                
                short_signals = both_result[both_result['SignalType'] == 'SHORT']
                if not short_signals.empty:
                    logger.signal(f"\n🔴 SHORT SIGNALS ({len(short_signals)} found):")
                    for idx, row in short_signals.iterrows():
                        performer_info: Dict[str, Any] = next(
                            (p for p in worst_performers if p['symbol'] == row['Symbol']), {}
                        )
                        score: Any = performer_info.get('composite_score', 'N/A')
                        tf_scores: Dict[str, float] = performer_info.get('timeframe_scores', {})
                        consistency: Any = performer_info.get('score_consistency', 'N/A')
                        volume: str = _get_volume_info(performance_result, row['SignalTimeframe'], row['Symbol'])
                        
                        logger.trade(f"  📉 {row['Symbol']}: {row['SignalType']} on {row['SignalTimeframe']}")
                        logger.performance(f"     Performance Score: {score}")
                        logger.data(f"     Avg Volume (24h): {volume}")
                        logger.info(f"     Score Consistency: {consistency}")
                        logger.info(f"     Timeframe Scores: {tf_scores}")
                        logger.info("")
                
                logger.analysis(f"📅 SIGNAL DISTRIBUTION BY TIMEFRAME:")
                timeframe_dist = both_result['SignalTimeframe'].value_counts()
                for tf, count in timeframe_dist.items():
                    tf_long: int = len(both_result[(both_result['SignalTimeframe'] == tf) & (both_result['SignalType'] == 'LONG')])
                    tf_short: int = len(both_result[(both_result['SignalTimeframe'] == tf) & (both_result['SignalType'] == 'SHORT')])
                    logger.signal(f"  {tf}: {count} total ({tf_long} LONG, {tf_short} SHORT)")
            else:
                logger.warning(f"  No signals found in {mode_name} mode")
        
        logger.performance("\n" + "="*80)
        logger.performance("COMPREHENSIVE SUMMARY STATISTICS")
        logger.performance("="*80)
        
        summary_dict: Dict[str, Any] = performance_result.get('summary', {})
        
        logger.analysis(f"📊 Analysis Overview:")
        logger.info(f"  • Total symbols analyzed: {len(all_symbols)}")
        logger.info(f"  • Symbols with performance data: {len(preloaded_data)}")
        logger.config(f"  • Performance analysis timeframes: {timeframes}")
        logger.config(f"  • HMM analysis timeframes: {hmm_timeframes}")
        logger.config(f"  • Performance period: {summary_dict.get('performance_period', 48)} periods")
        logger.config(f"  • Minimum volume filter: {summary_dict.get('min_volume_usdt', 50000):,} USDT")
        
        logger.analysis(f"\n🎯 Performance Selection:")
        logger.info(f"  • Top performers identified: {len(best_performers)}")
        logger.info(f"  • Worst performers identified: {len(worst_performers)}")
        logger.info(f"  • Top percentage threshold: {summary_dict.get('top_percentage', 0.3)*100:.1f}%")
        logger.info(f"  • Bottom percentage threshold: {summary_dict.get('short_percentage', 0.3)*100:.1f}%")
        
        logger.data(f"\n🔄 HMM Data Preparation:")
        logger.info(f"  • Symbols with HMM data loaded: {len(candidates_hmm_data)}")
        logger.info(f"  • Top performers with HMM data: {len(top_performers_data)}")
        logger.info(f"  • Worst performers with HMM data: {len(worst_performers_data)}")
        
        logger.model(f"\n🤖 HMM Analysis Capabilities Demonstrated:")
        capabilities: List[str] = [
            "Support for both LONG and SHORT signals",
            "Configurable signal type inclusion via parameters",
            "Enhanced logging with signal type breakdown",
            "Backward compatibility maintained",
            "Flexible parameter-based signal filtering",
            "Performance-based candidate pre-filtering",
            "Multi-timeframe HMM analysis"
        ]
        for capability in capabilities:
            logger.success(f"  ✓ {capability}")
        
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

if __name__ == "__main__":
    result: Optional[Dict[str, Any]] = main()
    
    if result:
        logger.success(f"\n🎉 HMM Analysis Example completed successfully!")
        logger.performance(f"Summary: {result['summary_stats']}")
    else:
        logger.error("❌ Example failed - check logs for details")
        