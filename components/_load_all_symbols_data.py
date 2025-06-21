import logging
import os
import pandas as pd
import sys
import threading
import time
from colorama import Fore, Style
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from tqdm import tqdm
from typing import Dict, List, Optional, Union

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
from utilities._logger import setup_logging
from components.config import (DEFAULT_TIMEFRAMES,
                               MAX_CPU_MEMORY_FRACTION,
                               DATA_PROCESSING_WAIT_TIME_IN_SECONDS,
)

# Configuration constants
MAX_CPU_MEMORY_FRACTION = MAX_CPU_MEMORY_FRACTION*1.5
MAX_WAIT_TIME = DATA_PROCESSING_WAIT_TIME_IN_SECONDS * 5
SLEEP_INTERVAL = 0.5
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.5

# Progress bar styling
TQDM_FORMAT = "{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Style.RESET_ALL)
TQDM_COLORS = {
    'main': '#FF8C00',
    'timeframes': '#FF8C00',
    'waiting': '#FFB347',
    'retry': '#FF6347',
    'error_retry': '#FF4500',
    'memory': '#DDA0DD'
}

logger = setup_logging(module_name="_load_all_symbols_data", log_level=logging.DEBUG)
_CACHE_LOCK = threading.Lock()

def _create_progress_bar(total: int, desc: str, color: str, unit: str = 'it', leave: bool = True) -> tqdm:
    """Create standardized progress bar with consistent styling."""
    return tqdm(
        total=total, desc=desc, bar_format=TQDM_FORMAT, colour=color, unit=unit, leave=leave
    )

def _wait_for_data_with_progress(processor, cache_key: tuple, symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """Wait for data processing with visual progress feedback."""
    wait_pbar = _create_progress_bar(
        total=MAX_WAIT_TIME, desc=f"Waiting for {symbol} ({tf})", 
        color=TQDM_COLORS['waiting'], unit='s', leave=False
    )
    
    wait_time = 0
    while wait_time < MAX_WAIT_TIME:
        with _CACHE_LOCK:
            if cache_key in processor.df_cache:
                cached_data = processor.df_cache[cache_key]
                if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                    wait_pbar.close()
                    return cached_data.copy()
        
        sleep(SLEEP_INTERVAL)
        wait_time += SLEEP_INTERVAL
        wait_pbar.update(SLEEP_INTERVAL)
    
    wait_pbar.close()
    return None

def load_symbol_data(processor, symbol: str = "", timeframes: Optional[List[str]] = None, load_multi_timeframes: bool = False) -> Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]:
    """
    Load and preprocess trading data for a single symbol across specified timeframes.
    
    Args:
        processor: Data processor with df_cache and get_historic_data_by_symbol method
        symbol: Trading symbol (e.g., 'BTCUSDT', 'EURUSD')
        timeframes: List of timeframes ['1m', '5m', '1h'] (None for defaults)
        load_multi_timeframes: Return dict of timeframes vs single DataFrame
        
    Returns:
        Dict[str, pd.DataFrame] if load_multi_timeframes=True, else pd.DataFrame
        None if loading fails for all timeframes
    """
    # Validate inputs
    if not symbol or not isinstance(symbol, str) or not symbol.strip() or not hasattr(processor, 'df_cache') or not hasattr(processor, 'get_historic_data_by_symbol'):
        logger.error(f"Invalid symbol: {symbol}" if symbol else "Invalid processor: missing required attributes")
        return None
    
    # Handle timeframes validation - check for empty list before setting defaults
    if timeframes is not None and (not isinstance(timeframes, list) or len(timeframes) == 0):
        logger.error(f"Invalid timeframes: {timeframes}")
        return None
    
    timeframes = timeframes or (DEFAULT_TIMEFRAMES if load_multi_timeframes else DEFAULT_TIMEFRAMES[:1])
    logger.data(f"Loading data for {symbol} with timeframes: {timeframes}")
    result_dict = {}
    
    # Create timeframe progress bar if needed
    timeframe_pbar = None
    if load_multi_timeframes and len(timeframes) > 1:
        timeframe_pbar = _create_progress_bar(
            total=len(timeframes), desc=f"Loading {symbol} timeframes",
            color=TQDM_COLORS['timeframes'], leave=False
        )
    
    for tf in timeframes:
        if not isinstance(tf, str):
            logger.warning(f"Skipping invalid timeframe: {tf}")
            if timeframe_pbar: timeframe_pbar.update(1)
            continue
            
        try:
            logger.process(f"Processing {tf} data for {symbol}")
            cache_key = (symbol, tf)
            data = None
            
            # Check cache first
            with _CACHE_LOCK:
                if cache_key in processor.df_cache:
                    cached_data = processor.df_cache[cache_key]
                    if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                        data = cached_data.copy()
                        logger.memory(f"Found cached data for {symbol} ({tf})")
            
            # Fetch data if not in cache
            if data is None:
                logger.network(f"Requesting data for {symbol} ({tf})")
                try:
                    processor.get_historic_data_by_symbol(symbol, tf)
                    data = _wait_for_data_with_progress(processor, cache_key, symbol, tf)
                except Exception as e:
                    logger.error(f"Error requesting data for {symbol} ({tf}): {e}")
                    if timeframe_pbar: timeframe_pbar.update(1)
                    continue
                
                if data is None:
                    logger.warning(f"Timeout waiting for data: {symbol} ({tf})")
                    if timeframe_pbar: timeframe_pbar.update(1)
                    continue

            logger.success(f"Successfully processed {tf} data for {symbol}, shape: {data.shape}")
            
            # Handle result based on mode
            if load_multi_timeframes:
                result_dict[tf] = data
                if timeframe_pbar:
                    timeframe_pbar.set_postfix({'Status': f'✓ {tf}', 'Rows': data.shape[0]})
                    timeframe_pbar.update(1)
            else:
                if timeframe_pbar: timeframe_pbar.close()
                return data
                
        except Exception as e:
            logger.error(f"Unexpected error loading {tf} data for {symbol}: {e}")
            if timeframe_pbar: timeframe_pbar.update(1)
            if not load_multi_timeframes: return None
    
    if timeframe_pbar: timeframe_pbar.close()
    
    if load_multi_timeframes:
        if not result_dict:
            logger.error(f"Failed to load any timeframe data for {symbol}")
            return None
        logger.success(f"Loaded {len(result_dict)} timeframes for {symbol}")
        return result_dict
    
    logger.warning(f"No data loaded for {symbol}")
    return None

def _retry_with_backoff(symbol: str, wait_time: float, attempt_type: str) -> None:
    """Execute retry wait with visual progress feedback."""
    retry_pbar = _create_progress_bar(
        total=int(wait_time * 10),
        desc=f"{attempt_type} {symbol}",
        color=TQDM_COLORS['retry'] if 'retry' in attempt_type.lower() else TQDM_COLORS['error_retry'],
        unit='0.1s', leave=False
    )
    
    for _ in range(int(wait_time * 10)):
        time.sleep(0.1)
        retry_pbar.update(1)
    retry_pbar.close()

def _calculate_memory_usage(dfs: Dict[str, Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]]) -> float:
    """Calculate total memory usage of loaded data with progress tracking."""
    memory_pbar = _create_progress_bar(
        total=len(dfs), desc="Calculating memory usage", 
        color=TQDM_COLORS['memory'], leave=False
    )
    
    total_memory_mb = 0
    for symbol, df in dfs.items():
        if df is not None:
            if isinstance(df, dict):
                for tf_df in df.values():
                    if isinstance(tf_df, pd.DataFrame):
                        total_memory_mb += tf_df.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(df, pd.DataFrame):
                total_memory_mb += df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        memory_pbar.set_postfix({'Symbol': symbol, 'Total MB': f"{total_memory_mb:.1f}"})
        memory_pbar.update(1)
    
    memory_pbar.close()
    return total_memory_mb

def load_all_symbols_data(processor, symbols: List[str], 
                        load_multi_timeframes: bool = True, 
                        timeframes: Optional[List[str]] = None,
                        max_retries: int = DEFAULT_MAX_RETRIES,
                        max_workers: Optional[int] = None) -> Dict[str, Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]]:
    """
    Load trading data for multiple symbols with parallel processing and comprehensive error handling.
    
    Args:
        processor: Data processor instance with caching capabilities
        symbols: List of trading symbols to load ['BTCUSDT', 'ETHUSDT', ...]
        load_multi_timeframes: Load multiple timeframes per symbol
        timeframes: Specific timeframes to load (None for defaults)
        max_retries: Maximum retry attempts per symbol (default: 3)
        max_workers: Max parallel workers (None for auto-detection)
        
    Returns:
        Dict mapping symbols to their loaded data:
        - Key: symbol string
        - Value: DataFrame (single timeframe) or Dict[str, DataFrame] (multi-timeframe) or None (failed)
    """
    # Validate inputs
    if not symbols or not isinstance(symbols, list) or not hasattr(processor, 'df_cache') or not hasattr(processor, 'get_historic_data_by_symbol'):
        logger.error(f"{'Invalid symbols list' if symbols else 'Invalid processor'}")
        return {}
    
    # Filter valid symbols
    valid_symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
    if len(valid_symbols) != len(symbols):
        logger.warning(f"Filtered out {len(symbols) - len(valid_symbols)} invalid symbols")
    
    if not valid_symbols:
        logger.error("No valid symbols to process")
        return {}
    
    start_time = time.time()
    
    # Define retry function
    def _load_with_retry(symbol: str, retries_left: int = max_retries, backoff_factor: float = DEFAULT_BACKOFF_FACTOR) -> Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]:
        try:
            data = load_symbol_data(processor, symbol, timeframes, load_multi_timeframes)
            
            if data is None and retries_left > 0:
                wait_time = backoff_factor * (max_retries - retries_left + 1)
                logger.warning(f"Failed to load {symbol}, retrying in {wait_time:.1f}s... ({retries_left} attempts left)")
                _retry_with_backoff(symbol, wait_time, "Retrying")
                return _load_with_retry(symbol, retries_left - 1, backoff_factor)
                
            return data
        
        except Exception as e:
            if retries_left > 0:
                wait_time = backoff_factor * (max_retries - retries_left + 1)
                logger.warning(f"Error loading {symbol}: {e}, retrying in {wait_time:.1f}s... ({retries_left} attempts left)")
                _retry_with_backoff(symbol, wait_time, "Error retry")
                return _load_with_retry(symbol, retries_left - 1, backoff_factor)
            else:
                logger.error(f"Failed to load {symbol} after {max_retries} attempts: {e}")
                return None
    
    # Configure workers
    available_cpus = os.cpu_count() or 4
    max_cpu_workers = max(1, int(available_cpus * MAX_CPU_MEMORY_FRACTION))
    max_workers = max(1, min(max_workers or max_cpu_workers, len(valid_symbols), max_cpu_workers))
    
    logger.config(f"Using {max_workers} parallel workers (max CPU fraction: {MAX_CPU_MEMORY_FRACTION})")
    logger.performance(f"Loading data for {len(valid_symbols)} symbols using {max_workers} parallel workers")
    
    # Create main progress bar
    main_pbar = _create_progress_bar(
        total=len(valid_symbols), desc="Loading symbol data",
        color=TQDM_COLORS['main'], unit='symbol'
    )
    
    # Execute parallel loading
    dfs = {}
    successful_loads = failed_loads = 0
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_with_retry, symbol): symbol for symbol in valid_symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    dfs[symbol] = result
                    
                    # Update counters and progress bar
                    is_success = result is not None
                    status_symbol = symbol if is_success else f"❌ {symbol}"
                    successful_loads += int(is_success)
                    failed_loads += int(not is_success)
                    
                    total_completed = successful_loads + failed_loads
                    success_rate = successful_loads / total_completed * 100 if total_completed else 0
                    
                    main_pbar.set_postfix({
                        'Success': successful_loads,
                        'Failed': failed_loads,
                        'Current': status_symbol,
                        'Rate': f"{success_rate:.1f}%"
                    })
                    main_pbar.update(1)
                    
                    # Log progress periodically
                    if total_completed % 5 == 0 or total_completed == len(valid_symbols):
                        elapsed = time.time() - start_time
                        logger.performance(f"Progress: {total_completed}/{len(valid_symbols)} symbols processed "
                                        f"({success_rate:.1f}% success) in {elapsed:.1f}s")
                        
                except Exception as e:
                    failed_loads += 1
                    logger.error(f"Unexpected error handling {symbol}: {str(e)}")
                    dfs[symbol] = None
                    main_pbar.set_postfix({
                        'Success': successful_loads,
                        'Failed': failed_loads,
                        'Current': f"💥 {symbol}",
                        'Rate': f"{successful_loads/(successful_loads + failed_loads)*100:.1f}%"
                    })
                    main_pbar.update(1)
                    
    except Exception as e:
        logger.error(f"Error in thread pool execution: {e}")
        main_pbar.close()
        return dfs
    
    main_pbar.close()
    
    # Log final statistics
    elapsed_time = time.time() - start_time
    success_rate = successful_loads / len(valid_symbols) * 100 if valid_symbols else 0
    
    logger.performance(f"Data loading complete: {successful_loads}/{len(valid_symbols)} symbols successfully loaded "
                    f"({success_rate:.1f}% success rate) in {elapsed_time:.2f}s")
    
    # Calculate memory usage
    total_memory_mb = _calculate_memory_usage(dfs)
    logger.memory(f"Total data memory usage: {total_memory_mb:.2f} MB")
    
    # Detailed logging if in debug mode
    if logger.level <= logging.DEBUG:
        for symbol, df in dfs.items():
            if df is not None:
                if load_multi_timeframes and isinstance(df, dict):
                    timeframe_info = []
                    for tf, timeframe_df in df.items():
                        if isinstance(timeframe_df, pd.DataFrame) and not timeframe_df.empty:
                            # Handle different index types safely
                            try:
                                if hasattr(timeframe_df.index, 'strftime'):
                                    time_range = f"{timeframe_df.index.min().strftime('%Y-%m-%d %H:%M')} to {timeframe_df.index.max().strftime('%Y-%m-%d %H:%M')}"
                                elif pd.api.types.is_datetime64_any_dtype(timeframe_df.index):
                                    time_range = f"{pd.to_datetime(timeframe_df.index.min()).strftime('%Y-%m-%d %H:%M')} to {pd.to_datetime(timeframe_df.index.max()).strftime('%Y-%m-%d %H:%M')}"
                                else:
                                    time_range = f"rows {timeframe_df.index.min()} to {timeframe_df.index.max()}"
                            except (AttributeError, ValueError):
                                time_range = f"rows {timeframe_df.index.min()} to {timeframe_df.index.max()}"
                            
                            timeframe_info.append(f"{tf} ({timeframe_df.shape[0]} bars, {time_range})")
                    
                    if timeframe_info:
                        logger.data(f"{symbol} data loaded: " + "; ".join(timeframe_info))
                elif isinstance(df, pd.DataFrame) and not df.empty:
                    # Handle different index types safely
                    try:
                        if hasattr(df.index, 'strftime'):
                            time_range = f"{df.index.min().strftime('%Y-%m-%d %H:%M')} to {df.index.max().strftime('%Y-%m-%d %H:%M')}"
                        elif pd.api.types.is_datetime64_any_dtype(df.index):
                            time_range = f"{pd.to_datetime(df.index.min()).strftime('%Y-%m-%d %H:%M')} to {pd.to_datetime(df.index.max()).strftime('%Y-%m-%d %H:%M')}"
                        else:
                            time_range = f"rows {df.index.min()} to {df.index.max()}"
                    except (AttributeError, ValueError):
                        time_range = f"rows {df.index.min()} to {df.index.max()}"
                    
                    logger.data(f"{symbol} data: {df.shape[0]} bars, {time_range}")

    return dfs
