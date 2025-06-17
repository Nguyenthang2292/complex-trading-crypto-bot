import logging
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# Add the parent directory to sys.path to allow importing modules from sibling directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from livetrade._components._tick_processor import tick_processor
from livetrade.config import (DEFAULT_TIMEFRAMES,DEFAULT_TOP_SYMBOLS,MIN_DATA_POINTS)
from signals._quant_models.best_performance_pairs__class__performance_analyzer import PerformanceAnalyzer
from utilities._logger import setup_logging

# Setup logging with new format
logger = setup_logging(module_name="signals_best_performance_pairs", log_level=logging.DEBUG)

def signal_best_performance_pairs(
    processor: tick_processor,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    performance_period: int = 24,
    top_percentage: float = 0.3,
    include_short_signals: bool = True,
    worst_percentage: float = 0.3,
    min_volume_usdt: float = 1000000,
    exclude_stable_coins: bool = True,
    preloaded_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None) -> Dict:
    """
    Analyze cryptocurrency symbols across multiple timeframes to identify top-performing pairs.
    
    Args:
        processor: tick_processor instance with Binance client
        symbols: List of symbols to analyze (if None, uses all USDT pairs)
        timeframes: List of timeframes to analyze (default: ['1h', '4h', '1d'])
        performance_period: Number of periods to calculate performance over
        top_percentage: Percentage of top performers to return for LONG signals
        include_short_signals: Whether to include SHORT signal analysis
        worst_percentage: Percentage of worst performers to return for SHORT signals
        min_volume_usdt: Minimum 24h volume in USDT to consider
        exclude_stable_coins: Whether to exclude stablecoins from analysis
        preloaded_data: Pre-loaded data dictionary {symbol: {timeframe: dataframe}}
        
    Returns:
        Dictionary containing analysis results for both LONG and SHORT signals
    """
    try:
        logger.analysis("Starting best performance pairs analysis...")
        
        analyzer = PerformanceAnalyzer()
        timeframes = timeframes or [tf for tf in DEFAULT_TIMEFRAMES if tf in ['1h', '4h', '1d']]
        symbols = _prepare_symbols(processor, symbols, exclude_stable_coins)
        
        if not symbols or not preloaded_data:
            logger.error("No symbols found or preloaded data missing")
            return {}
        
        logger.analysis(f"Analyzing {len(symbols)} symbols across timeframes: {timeframes}")
        if include_short_signals:
            logger.info(f"Finding top {top_percentage*100:.0f}% for LONG and bottom {worst_percentage*100:.0f}% for SHORT")
        
        timeframe_results = {}
        symbol_scores = {}
        
        for tf in timeframes:
            logger.analysis(f"Analyzing timeframe: {tf}")
            timeframe_results[tf] = _analyze_timeframe_performance(
                analyzer, preloaded_data, tf, performance_period, min_volume_usdt, include_short_signals
            )
            
            for symbol_metrics in timeframe_results[tf]['symbol_metrics']:
                symbol = symbol_metrics['symbol']
                if symbol not in symbol_scores:
                    symbol_scores[symbol] = {}
                symbol_scores[symbol][tf] = symbol_metrics['composite_score']
        
        logger.analysis("Calculating composite scores across all timeframes...")
        overall_scores = analyzer.calculate_overall_scores(symbol_scores, timeframes)
        
        best_performers, worst_performers = _select_performers(
            overall_scores, top_percentage, worst_percentage, include_short_signals
        )
        
        return _create_result_dict(
            best_performers, worst_performers, timeframe_results, symbols, timeframes,
            top_percentage, worst_percentage, performance_period, min_volume_usdt, include_short_signals
        )
        
    except Exception as e:
        logger.error(f"Error in signal_best_performance_pairs: {e}")
        return {}

def _prepare_symbols(processor: Optional[tick_processor], symbols: Optional[List[str]] = None, exclude_stable_coins: bool = True) -> List[str]:
    """
    Prepare and filter symbol list, optionally excluding stablecoins.
    
    Args:
        processor: tick_processor instance
        symbols: List of symbols to analyze (if None, gets all USDT pairs)
        exclude_stable_coins: Whether to exclude stablecoins from analysis
        
    Returns:
        Filtered list of symbols
    """
    if symbols is None:
        if processor is not None:
            symbols = processor.get_symbols_list_by_quote_usdt()
        else:
            logger.error("Processor is None; cannot retrieve symbols.")
            return []
    
    if exclude_stable_coins and symbols:
        stable_coins = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAXG', 'USDP'}
        initial_count = len(symbols)
        
        symbols = [
            symbol for symbol in symbols 
            if (symbol[:-4] if symbol.endswith('USDT') else symbol) not in stable_coins
        ]
        
        logger.success(f"Filtered out stablecoins: {initial_count} -> {len(symbols)} symbols")
    
    return symbols

def _analyze_timeframe_performance(
    analyzer: PerformanceAnalyzer,
    symbol_data: Dict[str, Dict[str, pd.DataFrame]],
    timeframe: str, performance_period: int,
    min_volume_usdt: float, analyze_for_short: bool = True) -> Dict:
    """
    Analyze performance metrics for symbols in a specific timeframe.
    
    Args:
        analyzer: PerformanceAnalyzer instance for metrics calculation
        symbol_data: Dictionary mapping symbols to their timeframe data
        timeframe: Timeframe to analyze ('1h', '4h', '1d', etc.)
        performance_period: Number of periods for performance calculation
        min_volume_usdt: Minimum volume threshold in USDT
        analyze_for_short: Whether to analyze for short signals
        
    Returns:
        Dictionary containing timeframe analysis results and statistics
    """
    try:
        logger.analysis(f"Analyzing {len(symbol_data)} symbols for {timeframe} timeframe...")
        
        symbol_metrics = []
        stats = {'processed': 0, 'no_timeframe': 0, 'insufficient_data': 0, 'low_volume': 0}
        total_symbols = len(symbol_data)
        
        with tqdm(total=total_symbols, desc=f"Analyzing {timeframe}", unit="symbol", ncols=100,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for symbol, timeframe_data in symbol_data.items():
                try:
                    if timeframe not in timeframe_data:
                        stats['no_timeframe'] += 1
                        continue
                    
                    df = timeframe_data[timeframe]
                    min_required_data = max(performance_period, MIN_DATA_POINTS)
                    
                    if df is None or df.empty or len(df) < min_required_data:
                        stats['insufficient_data'] += 1
                        continue
                    
                    metrics = analyzer.calculate_performance_metrics(df, symbol, timeframe, performance_period)
                    
                    if metrics['avg_volume_usdt'] < min_volume_usdt:
                        stats['low_volume'] += 1
                        continue
                    
                    symbol_metrics.append(metrics)
                    stats['processed'] += 1
                    
                    pbar.set_postfix({
                        "processed": stats['processed'], 
                        "no_tf": stats['no_timeframe'],
                        "low_vol": stats['low_volume']
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for {timeframe}: {e}")
                finally:
                    pbar.update(1)
        
        _log_timeframe_stats(timeframe, stats)
        
        symbol_metrics.sort(key=lambda x: x['composite_score'], reverse=True)
        timeframe_stats = _calculate_timeframe_statistics(symbol_metrics, analyze_for_short)
        
        return {
            'timeframe': timeframe,
            'symbol_metrics': symbol_metrics,
            'statistics': timeframe_stats
        }
        
    except Exception as e:
        logger.error(f"Error analyzing timeframe {timeframe}: {e}")
        return {
            'timeframe': timeframe,
            'symbol_metrics': [],
            'statistics': {'symbols_processed': 0}
        }

def _log_timeframe_stats(timeframe: str, stats: Dict):
    """Log timeframe processing statistics"""
    logger.performance(f"Timeframe {timeframe} results:")
    logger.performance(f"  - Processed successfully: {stats['processed']}")
    logger.performance(f"  - Skipped (no timeframe data): {stats['no_timeframe']}")
    logger.performance(f"  - Skipped (insufficient data): {stats['insufficient_data']}")
    logger.performance(f"  - Skipped (low volume): {stats['low_volume']}")

def _calculate_timeframe_statistics(symbol_metrics: List[Dict], analyze_for_short: bool) -> Dict:
    """Calculate comprehensive statistics for a timeframe"""
    if not symbol_metrics:
        return {
            'symbols_processed': 0,
            'avg_score': 0,
            'median_score': 0,
            'avg_return': 0,
            'avg_volatility': 0,
            'avg_short_score': 0,
            'top_performer': None,
            'worst_performer': None
        }
    
    scores = [m['composite_score'] for m in symbol_metrics]
    returns = [m['total_return'] for m in symbol_metrics]
    volatilities = [m['volatility'] for m in symbol_metrics]
    short_scores = [m.get('short_composite_score', 0) for m in symbol_metrics]
    
    return {
        'symbols_processed': len(symbol_metrics),
        'avg_score': np.mean(scores),
        'median_score': np.median(scores),
        'avg_return': np.mean(returns),
        'avg_volatility': np.mean(volatilities),
        'avg_short_score': np.mean(short_scores),
        'top_performer': symbol_metrics[0],
        'worst_performer': symbol_metrics[-1]
    }

def _select_performers(
    overall_scores: List[Dict],
    top_percentage: float,
    worst_percentage: float,
    include_short_signals: bool
) -> Tuple[List[Dict], List[Dict]]:
    """Select top and worst performers based on specified percentages"""
    num_top_performers = max(1, int(len(overall_scores) * top_percentage))
    best_performers = overall_scores[:num_top_performers]
    
    worst_performers = []
    if include_short_signals:
        num_worst_performers = max(1, int(len(overall_scores) * worst_percentage))
        worst_performers = overall_scores[-num_worst_performers:]
        worst_performers.reverse()
        logger.signal(f"Selected {len(worst_performers)} worst performers for SHORT signals")
    
    return best_performers, worst_performers

def _create_result_dict(
    best_performers: List[Dict],
    worst_performers: List[Dict],
    timeframe_results: Dict,
    symbols: List[str],
    timeframes: List[str],
    top_percentage: float,
    worst_percentage: float,
    performance_period: int,
    min_volume_usdt: float,
    include_short_signals: bool
) -> Dict:
    """Create comprehensive result dictionary with analysis summary"""
    result = {
        'best_performers': best_performers,
        'worst_performers': worst_performers,
        'timeframe_analysis': timeframe_results,
        'summary': {
            'total_symbols_analyzed': len(symbols),
            'timeframes_analyzed': timeframes,
            'top_performers_count': len(best_performers),
            'worst_performers_count': len(worst_performers),
            'top_percentage': top_percentage,
            'short_percentage': worst_percentage,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'performance_period': performance_period,
            'min_volume_usdt': min_volume_usdt,
            'include_short_signals': include_short_signals
        },
    }
    
    _log_analysis_results(best_performers, worst_performers, include_short_signals)
    
    return result

def _log_analysis_results(best_performers: List[Dict], worst_performers: List[Dict], include_short_signals: bool):
    """Log final analysis results with top performers"""
    logger.success("Analysis complete!")
    logger.signal(f"Top {len(best_performers)} performers for LONG signals:")
    for i, performer in enumerate(best_performers[:5], 1):
        logger.signal(f"  {i}. {performer['symbol']}: Score {performer['composite_score']:.4f}")
    
    if include_short_signals and worst_performers:
        logger.signal(f"Bottom {len(worst_performers)} performers for SHORT signals:")
        for i, performer in enumerate(worst_performers[:5], 1):
            logger.signal(f"  {i}. {performer['symbol']}: Score {performer['composite_score']:.4f}")

def get_top_performers_by_timeframe(analysis_result: Dict, timeframe: str, top_n: int = 10) -> List[Dict]:
    """Extract top performers for a specific timeframe from analysis results."""
    try:
        timeframe_data = analysis_result.get('timeframe_analysis', {}).get(timeframe)
        if not timeframe_data:
            return []
        
        symbol_metrics = timeframe_data['symbol_metrics']
        return symbol_metrics[:top_n]
    
    except Exception as e:
        logger.error(f"Error extracting top performers for {timeframe}: {e}")
        return []

def get_worst_performers_by_timeframe(analysis_result: Dict, timeframe: str, top_n: int = 10) -> List[Dict]:
    """Extract worst performers for a specific timeframe (for SHORT signals)."""
    try:
        timeframe_data = analysis_result.get('timeframe_analysis', {}).get(timeframe)
        if not timeframe_data:
            return []
        
        symbol_metrics = timeframe_data['symbol_metrics']
        return symbol_metrics[-top_n:][::-1]
    
    except Exception as e:
        logger.error(f"Error extracting worst performers for {timeframe}: {e}")
        return []

def get_short_signal_candidates(analysis_result: Dict, min_short_score: float = 0.6) -> List[Dict]:
    """Get symbols that are good candidates for SHORT signals based on SHORT composite score."""
    try:
        candidates = []
        worst_performers = analysis_result.get('worst_performers', [])
        
        for performer in worst_performers:
            timeframe_scores = performer.get('timeframe_scores', {})
            short_scores = [
                tf_data.get('short_composite_score', 0)
                for tf_data in timeframe_scores.values()
                if isinstance(tf_data, dict) and 'short_composite_score' in tf_data
            ]
            
            if short_scores and np.mean(short_scores) >= min_short_score:
                performer_copy = performer.copy()
                performer_copy.update({
                    'avg_short_score': np.mean(short_scores),
                    'short_score_consistency': np.std(short_scores)
                })
                candidates.append(performer_copy)
        
        candidates.sort(key=lambda x: x.get('avg_short_score', 0), reverse=True)
        return candidates
    
    except Exception as e:
        logger.error(f"Error getting SHORT signal candidates: {e}")
        return []

def print_performance_summary(analysis_result: Dict):
    """Print a formatted summary of the performance analysis including SHORT signals."""
    try:
        if not analysis_result:
            logger.warning("No analysis results to display")
            return
        
        summary = analysis_result.get('summary', {})
        best_performers = analysis_result.get('best_performers', [])
        worst_performers = analysis_result.get('worst_performers', [])
        
        print("\n" + "="*80)
        print("CRYPTO PERFORMANCE ANALYSIS SUMMARY (LONG & SHORT)")
        print("="*80)
        
        print(f"Analysis Date: {summary.get('analysis_timestamp', 'Unknown')}")
        print(f"Total Symbols Analyzed: {summary.get('total_symbols_analyzed', 0)}")
        print(f"Timeframes: {', '.join(summary.get('timeframes_analyzed', []))}")
        print(f"Top Performers (LONG): {summary.get('top_performers_count', 0)}")
        print(f"Worst Performers (SHORT): {summary.get('worst_performers_count', 0)}")
        
        _print_performers_section("🟢 TOP", best_performers, "LONG SIGNALS")
        
        if worst_performers:
            _print_performers_section("🔴 BOTTOM", worst_performers, "SHORT SIGNALS")
        
        print("\n" + "="*80)
    
    except Exception as e:
        logger.error(f"Error printing performance summary: {e}")

def _print_performers_section(prefix: str, performers: List[Dict], signal_type: str):
    """Print a section of performers with their scores and timeframe breakdowns."""
    print(f"\n{prefix} {len(performers)} PERFORMERS ({signal_type}):")
    print("-"*60)
    
    display_limit = min(DEFAULT_TOP_SYMBOLS, len(performers))
    for i, performer in enumerate(performers[:display_limit], 1):
        symbol = performer['symbol']
        score = performer['composite_score']
        tf_scores = performer.get('timeframe_scores', {})
        
        print(f"{i:2d}. {symbol:12s} | Score: {score:.4f}")
        if tf_scores:
            tf_str = " | ".join([f"{tf}: {score:.3f}" for tf, score in tf_scores.items()])
            print(f"     Timeframe Scores: {tf_str}")

