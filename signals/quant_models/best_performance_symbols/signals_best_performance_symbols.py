"""
Best Performance Symbols Analysis Module

This module provides comprehensive analysis of cryptocurrency symbols across multiple
timeframes to identify top-performing pairs for both LONG and SHORT trading signals.

Key Features:
- Multi-timeframe performance analysis (1h, 4h, 1d)
- Composite scoring system for symbol ranking
- Support for both LONG and SHORT signal identification
- Volume filtering and stablecoin exclusion
- Comprehensive logging and progress tracking
- Statistical analysis and performance metrics

Main Functions:
- signal_best_performance_symbols: Main analysis function
- get_top_performers_by_timeframe: Extract top performers for specific timeframe
- get_worst_performers_by_timeframe: Extract worst performers for SHORT signals
- get_short_signal_candidates: Get SHORT signal candidates based on scores
- logging_performance_summary: Display formatted analysis results

Dependencies:
- PerformanceAnalyzer: For calculating performance metrics
- tick_processor: For data access and symbol management
- pandas, numpy: For data manipulation and calculations
"""

import logging
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
from typing import Any, List, Dict, Optional, Tuple, Union

# Add the parent directory to sys.path to allow importing modules from sibling directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from components.tick_processor import TickProcessor
from config.config import (DEFAULT_TIMEFRAMES, DEFAULT_TOP_SYMBOLS, MIN_DATA_POINTS)
from signals.quant_models.best_performance_symbols.__class__PerformanceAnalyzer import PerformanceAnalyzer
from utilities._logger import setup_logging

# Setup logging with new format
logger = setup_logging(module_name="signals_best_performance_symbols", log_level=logging.DEBUG)

def _validate_inputs(
    symbols: Optional[List[str]],
    timeframes: Optional[List[str]],
    performance_period: int,
    top_percentage: float,
    worst_percentage: float
) -> bool:
    """Validates the input parameters for the performance analysis.

    Args:
        symbols: An optional list of trading symbols.
        timeframes: An optional list of timeframes to analyze.
        performance_period: The number of periods for performance calculation.
        top_percentage: The percentage of top performers to select (0 to 1).
        worst_percentage: The percentage of worst performers to select (0 to 1).

    Returns:
        True if all inputs are valid, False otherwise.
    """
    if performance_period <= 0:
        logger.error(f"Invalid performance_period: {performance_period}. Must be > 0.")
        return False
    if not 0 < top_percentage <= 1:
        logger.error(f"Invalid top_percentage: {top_percentage}. Must be between 0 and 1.")
        return False
    if not 0 < worst_percentage <= 1:
        logger.error(f"Invalid worst_percentage: {worst_percentage}. Must be between 0 and 1.")
        return False
    if symbols is not None and not symbols:
        logger.error("Input 'symbols' list cannot be empty.")
        return False
    if timeframes is not None and not timeframes:
        logger.error("Input 'timeframes' list cannot be empty.")
        return False
    return True

def _prepare_symbols(
    processor: TickProcessor,
    symbols: Optional[List[str]] = None,
    exclude_stable_coins: bool = True
) -> List[str]:
    """Prepares and filters the list of symbols for analysis.

    If no symbols are provided, it fetches all USDT pairs. It can also
    filter out stablecoins.

    Args:
        processor: An instance of the TickProcessor to fetch symbol data.
        symbols: An optional list of symbols to process.
        exclude_stable_coins: If True, stablecoins are removed from the list.

    Returns:
        A filtered list of symbols.
    """
    if symbols is None:
        logger.info("No symbols provided, fetching all USDT pairs...")
        symbols = processor.get_symbols_list_by_quote_asset('USDT')
    
    if not symbols:
        logger.warning("No symbols were provided or could be retrieved.")
        return []
    
    if exclude_stable_coins:
        stable_coins = {'USDC', 'BUSD', 'DAI', 'TUSD', 'PAXG', 'USDP'}
        initial_count = len(symbols)
        
        # Assumes symbols are in 'BASEQUOTE' format, e.g., 'BTCUSDT'
        symbols = [s for s in symbols if not any(stable in s for stable in stable_coins)]
        
        logger.info(f"Filtered out stablecoins: {initial_count} -> {len(symbols)} symbols.")
    
    return symbols

def _analyze_timeframe_performance(
    analyzer: PerformanceAnalyzer,
    symbol_data: Dict[str, Dict[str, pd.DataFrame]],
    timeframe: str,
    performance_period: int,
    min_volume_usdt: float
) -> Dict[str, Any]:
    """Analyzes performance metrics for all symbols within a single timeframe.

    Args:
        analyzer: An instance of the PerformanceAnalyzer.
        symbol_data: A dictionary mapping symbols to their timeframe data.
        timeframe: The timeframe to analyze (e.g., '1h', '4h').
        performance_period: The number of periods for performance calculation.
        min_volume_usdt: The minimum average daily volume in USDT to consider a symbol.

    Returns:
        A dictionary containing the analysis results for the timeframe.
    """
    logger.info(f"Analyzing {len(symbol_data)} symbols for timeframe: {timeframe}...")
    symbol_metrics = []
    stats = {'processed': 0, 'no_data': 0, 'insufficient_data': 0, 'low_volume': 0}
    min_required_data = max(performance_period, MIN_DATA_POINTS)
    
    with tqdm(total=len(symbol_data), desc=f"Analyzing {timeframe}", unit="symbol") as pbar:
        for symbol, tf_data in symbol_data.items():
            df = tf_data.get(timeframe)
            if df is None or df.empty:
                stats['no_data'] += 1
            elif len(df) < min_required_data:
                stats['insufficient_data'] += 1
            else:
                metrics = analyzer.calculate_performance_metrics(df, symbol, timeframe, performance_period)
                if metrics.get('avg_volume_usdt', 0) < min_volume_usdt:
                    stats['low_volume'] += 1
                else:
                    symbol_metrics.append(metrics)
                    stats['processed'] += 1
            pbar.update(1)
            
    _log_timeframe_stats(timeframe, stats)
    
    symbol_metrics.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    timeframe_stats = _calculate_timeframe_statistics(symbol_metrics)
    
    return {
        'timeframe': timeframe,
        'symbol_metrics': symbol_metrics,
        'statistics': timeframe_stats
    }

def _log_timeframe_stats(timeframe: str, stats: Dict[str, int]) -> None:
    """Logs the summary statistics for a timeframe's analysis."""
    logger.info(f"Timeframe '{timeframe}' analysis summary:")
    logger.info(f"  - Symbols processed successfully: {stats['processed']}")
    logger.info(f"  - Skipped (no data for timeframe): {stats['no_data']}")
    logger.info(f"  - Skipped (insufficient data points): {stats['insufficient_data']}")
    logger.info(f"  - Skipped (volume too low): {stats['low_volume']}")

def _calculate_timeframe_statistics(symbol_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates summary statistics for a list of symbol metrics.

    Args:
        symbol_metrics: A list of dictionaries, each containing performance metrics
                        for a single symbol.

    Returns:
        A dictionary containing the aggregated statistical summary.
    """
    if not symbol_metrics:
        return {'symbols_processed': 0}

    df_metrics = pd.DataFrame(symbol_metrics)
    
    stats = {
        'symbols_processed': len(df_metrics),
        'avg_score': df_metrics['composite_score'].mean(),
        'median_score': df_metrics['composite_score'].median(),
        'avg_return': df_metrics['return_over_period'].mean(),
        'avg_volatility': df_metrics['volatility'].mean(),
        'avg_sharpe_ratio': df_metrics['sharpe_ratio'].mean(),
        'top_performer': df_metrics.iloc[0].to_dict() if not df_metrics.empty else None,
        'worst_performer': df_metrics.iloc[-1].to_dict() if not df_metrics.empty else None
    }
    return stats

def _select_performers(
    overall_scores: List[Dict[str, Any]],
    top_percentage: float,
    worst_percentage: float,
    include_short_signals: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Selects the top and worst performing symbols based on scores.

    Args:
        overall_scores: A list of symbols with their aggregated performance scores.
        top_percentage: The percentage of top performers to select.
        worst_percentage: The percentage of worst performers to select for shorting.
        include_short_signals: Flag to indicate if short candidates should be selected.

    Returns:
        A tuple containing two lists: the best performers for long signals and the
        worst performers for short signals.
    """
    sorted_by_long = sorted(overall_scores, key=lambda x: x['long_score'], reverse=True)
    num_top = int(len(sorted_by_long) * top_percentage)
    best_performers = sorted_by_long[:num_top]

    worst_performers = []
    if include_short_signals:
        sorted_by_short = sorted(overall_scores, key=lambda x: x['short_score'], reverse=True)
        num_worst = int(len(sorted_by_short) * worst_percentage)
        worst_performers = sorted_by_short[:num_worst]
    
    return best_performers, worst_performers

def _create_result_dict(
    best_performers: List[Dict[str, Any]],
    worst_performers: List[Dict[str, Any]],
    timeframe_results: Dict[str, Any],
    symbols: List[str],
    timeframes: List[str]
) -> Dict[str, Any]:
    """Creates the final dictionary for the analysis results.

    Args:
        best_performers: A list of the top-performing symbols.
        worst_performers: A list of the worst-performing symbols.
        timeframe_results: The detailed results for each analyzed timeframe.
        symbols: The list of symbols that were analyzed.
        timeframes: The list of timeframes that were analyzed.

    Returns:
        A dictionary containing the complete analysis results.
    """
    return {
        "best_performers_long": best_performers,
        "worst_performers_short": worst_performers,
        "timeframe_analysis": timeframe_results,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols_analyzed_count": len(symbols),
            "timeframes_analyzed": timeframes,
        }
    }

def signal_best_performance_symbols(
    processor: TickProcessor,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    performance_period: int = 24,
    top_percentage: float = 0.3,
    include_short_signals: bool = True,
    worst_percentage: float = 0.3,
    min_volume_usdt: float = 1_000_000,
    exclude_stable_coins: bool = True,
    preloaded_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
) -> Dict[str, Any]:
    """
    Analyzes symbol performance across timeframes to find top candidates.

    This is the main function that orchestrates the entire analysis pipeline,
    from data fetching and validation to performance calculation and result aggregation.

    Args:
        processor: The TickProcessor instance for data fetching.
        symbols: An optional list of symbols to analyze.
        timeframes: An optional list of timeframes.
        performance_period: The look-back period for performance calculation.
        top_percentage: The percentage of top symbols to return for long signals.
        include_short_signals: Whether to analyze for short signal candidates.
        worst_percentage: The percentage of top symbols for short signals.
        min_volume_usdt: The minimum required average volume in USDT.
        exclude_stable_coins: Whether to filter out stablecoins.
        preloaded_data: Optional preloaded market data to bypass fetching.

    Returns:
        A dictionary containing the detailed analysis results, including top
        long and short candidates and per-timeframe breakdowns.
    """
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    if not _validate_inputs(symbols, timeframes, performance_period, top_percentage, worst_percentage):
        return {}

    symbols = _prepare_symbols(processor, symbols, exclude_stable_coins)
    if not symbols:
        return {}

    analyzer = PerformanceAnalyzer()
    timeframe_results = {}
    
    if preloaded_data is None:
        logger.info(f"No preloaded data. Fetching data for {len(symbols)} symbols...")
        # This part should be replaced with a call to a data loading function
        # For now, we assume data is loaded elsewhere or passed in.
        # Example: preloaded_data = load_all_symbols_data(symbols, timeframes)
        pass  # Placeholder for data loading logic
    
    if not preloaded_data:
        logger.error("No data available for analysis, cannot proceed.")
        return {}
        
    for tf in timeframes:
        timeframe_results[tf] = _analyze_timeframe_performance(
            analyzer, preloaded_data, tf, performance_period, min_volume_usdt
        )
            
    overall_scores = analyzer.calculate_overall_scores(timeframe_results)
    best, worst = _select_performers(overall_scores, top_percentage, worst_percentage, include_short_signals)

    return _create_result_dict(best, worst, timeframe_results, symbols, timeframes)

def get_top_performers_by_timeframe(
    analysis_result: Dict[str, Any], timeframe: str, top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Extracts the top N performing symbols for a specific timeframe.

    Args:
        analysis_result: The result dictionary from the main analysis function.
        timeframe: The timeframe to get performers from.
        top_n: The number of top performers to return.

    Returns:
        A list of the top N performing symbols for the given timeframe.
    """
    if timeframe not in analysis_result.get("timeframe_analysis", {}):
        logger.warning(f"Timeframe '{timeframe}' not found in analysis results.")
        return []
        
    metrics = analysis_result["timeframe_analysis"][timeframe].get("symbol_metrics", [])
    return metrics[:top_n]

def get_worst_performers_by_timeframe(
    analysis_result: Dict[str, Any], timeframe: str, top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Extracts the worst N performing symbols for a specific timeframe.

    These are potential candidates for short signals.

    Args:
        analysis_result: The result dictionary from the main analysis function.
        timeframe: The timeframe to get performers from.
        top_n: The number of worst performers to return.

    Returns:
        A list of the worst N performing symbols for the given timeframe.
    """
    if timeframe not in analysis_result.get("timeframe_analysis", {}):
        logger.warning(f"Timeframe '{timeframe}' not found in analysis results.")
        return []

    metrics = analysis_result["timeframe_analysis"][timeframe].get("symbol_metrics", [])
    # Assumes metrics are sorted from best to worst, so worst are at the end
    return metrics[-top_n:]

def get_short_signal_candidates(
    analysis_result: Dict[str, Any], min_short_score: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Filters and returns symbols that are strong candidates for short signals.

    Args:
        analysis_result: The result dictionary from the main analysis function.
        min_short_score: The minimum short score required to be a candidate.

    Returns:
        A list of symbols that are strong candidates for shorting.
    """
    worst_performers = analysis_result.get("worst_performers_short", [])
    if not worst_performers:
        logger.info("No worst performers found to analyze for short signals.")
        return []

    candidates = [
        p for p in worst_performers if p.get("short_score", 0) >= min_short_score
    ]
    
    logger.info(f"Found {len(candidates)} candidates for short signals with score >= {min_short_score}.")
    return candidates

def logging_performance_summary(analysis_result: Dict[str, Any]) -> None:
    """
    Logs a formatted summary of the performance analysis results.

    Args:
        analysis_result: The result dictionary from the main analysis function.
    """
    if not analysis_result:
        logger.warning("Analysis result is empty, cannot generate summary.")
        return

    logger.info("\n" + "="*80)
    logger.info("           BEST PERFORMANCE SYMBOLS - ANALYSIS SUMMARY")
    logger.info("="*80)
    
    metadata = analysis_result.get("metadata", {})
    logger.info(
        f"Analysis completed at: {metadata.get('timestamp')} | "
        f"Symbols Analyzed: {metadata.get('symbols_analyzed_count')} | "
        f"Timeframes: {metadata.get('timeframes_analyzed')}"
    )

    best_performers = analysis_result.get("best_performers_long", [])
    if best_performers:
        logger.info("\n--- Top Performers (LONG Candidates) ---")
        df_best = pd.DataFrame(best_performers).set_index('symbol')
        logger.info("\n" + df_best[['long_score', 'short_score']].to_string())
    
    worst_performers = analysis_result.get("worst_performers_short", [])
    if worst_performers:
        logger.info("\n--- Worst Performers (SHORT Candidates) ---")
        df_worst = pd.DataFrame(worst_performers).set_index('symbol')
        logger.info("\n" + df_worst[['short_score', 'long_score']].to_string())

    timeframe_analysis = analysis_result.get("timeframe_analysis", {})
    if timeframe_analysis:
        logger.info("\n" + "-"*30 + " TIMEFRAME DETAILS " + "-"*30)
        for tf, data in timeframe_analysis.items():
            stats = data.get("statistics", {})
            logger.info(
                f"\nTimeframe: {tf} | Symbols Processed: {stats.get('symbols_processed')}"
            )
            logger.info(
                f"  Avg Score: {stats.get('avg_score', 0):.2f} | "
                f"Avg Return: {stats.get('avg_return', 0):.2%} | "
                f"Avg Volatility: {stats.get('avg_volatility', 0):.4f}"
            )
            top_performer = stats.get('top_performer', {})
            if top_performer:
                logger.info(
                    f"  Top Performer: {top_performer.get('symbol')} "
                    f"(Score: {top_performer.get('composite_score', 0):.2f})"
                )
    logger.info("\n" + "="*80)

