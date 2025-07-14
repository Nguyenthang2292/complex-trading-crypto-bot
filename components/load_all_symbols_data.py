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
from typing import Dict, List, Optional, Union, Any, cast

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
from utilities._logger import setup_logging
from config.config import (DEFAULT_TIMEFRAMES,
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
    """Creates a standardized tqdm progress bar.

    Args:
        total: The total number of iterations.
        desc: The description to display.
        color: The color of the progress bar.
        unit: The unit for the iteration.
        leave: Whether to leave the progress bar after completion.

    Returns:
        A tqdm progress bar instance.
    """
    return tqdm(
        total=total, desc=desc, bar_format=TQDM_FORMAT, colour=color, unit=unit, leave=leave
    )

def _wait_for_data_with_progress(processor: Any, cache_key: tuple, symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """Waits for data to appear in the processor's cache with a progress bar.

    Args:
        processor: The data processor instance.
        cache_key: The tuple key for the cache.
        symbol: The trading symbol.
        tf: The timeframe.

    Returns:
        The DataFrame from the cache, or None if it times out.
    """
    wait_pbar = _create_progress_bar(
        total=int(MAX_WAIT_TIME / SLEEP_INTERVAL), desc=f"Waiting for {symbol} ({tf})", 
        color=TQDM_COLORS['waiting'], unit='s', leave=False
    )
    
    start_time = time.time()
    while time.time() - start_time < MAX_WAIT_TIME:
        with _CACHE_LOCK:
            if cache_key in processor.df_cache:
                cached_data = processor.df_cache[cache_key]
                if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                    wait_pbar.close()
                    return cached_data.copy()
        
        sleep(SLEEP_INTERVAL)
        wait_pbar.update(1)
    
    wait_pbar.close()
    return None

def load_symbol_data(
    processor: Any,
    symbol: str,
    timeframes: Optional[List[str]] = None,
    load_multi_timeframes: bool = False
) -> Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]:
    """Loads and preprocesses trading data for a single symbol.

    This function handles fetching data for one or more timeframes,
    utilizing a shared cache and a retry mechanism.

    Args:
        processor: An instance of a data processor class that has `df_cache`
            and `get_historic_data_by_symbol` methods.
        symbol: The trading symbol to load (e.g., 'BTCUSDT').
        timeframes: A list of timeframes to load (e.g., ['1h', '4h']). If None,
            defaults are used.
        load_multi_timeframes: If True, returns a dictionary of DataFrames keyed
            by timeframe. If False, returns a single DataFrame for the first
            valid timeframe.

    Returns:
        A dictionary of DataFrames if `load_multi_timeframes` is True, a single
        DataFrame otherwise, or None if loading fails.
    """
    if not (symbol and isinstance(symbol, str) and hasattr(processor, 'df_cache')):
        logger.error(f"Invalid input: symbol='{symbol}' or processor is invalid.")
        return None
    
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES if load_multi_timeframes else [DEFAULT_TIMEFRAMES[0]]
    
    logger.data(f"Loading data for {symbol} with timeframes: {timeframes}")
    result_dict: Dict[str, pd.DataFrame] = {}
    
    pbar_desc = f"Loading {symbol}"
    timeframe_pbar = _create_progress_bar(
        total=len(timeframes), desc=pbar_desc,
        color=TQDM_COLORS['timeframes'], leave=False
    ) if len(timeframes) > 1 else None
    
    for tf in timeframes:
        try:
            cache_key = (symbol, tf)
            data = None
            
            with _CACHE_LOCK:
                if cache_key in processor.df_cache:
                    cached_data = processor.df_cache.get(cache_key)
                    if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                        data = cached_data.copy()
                        logger.memory(f"Cache hit for {symbol} ({tf})")

            if data is None:
                logger.network(f"Cache miss. Requesting data for {symbol} ({tf})")
                processor.get_historic_data_by_symbol(symbol, tf)
                data = _wait_for_data_with_progress(processor, cache_key, symbol, tf)
                
                if data is None:
                    logger.warning(f"Timeout waiting for data: {symbol} ({tf})")
                    if timeframe_pbar: timeframe_pbar.update(1)
                    continue

            logger.success(f"Successfully processed {tf} data for {symbol}, shape: {data.shape}")
            
            if load_multi_timeframes:
                result_dict[tf] = data
                if timeframe_pbar:
                    timeframe_pbar.update(1)
            else:
                if timeframe_pbar: timeframe_pbar.close()
                return data
                
        except Exception as e:
            logger.error(f"Failed to load {tf} for {symbol}: {e}")
            if timeframe_pbar: timeframe_pbar.update(1)
            if not load_multi_timeframes: return None
    
    if timeframe_pbar: timeframe_pbar.close()
    
    if load_multi_timeframes:
        return result_dict if result_dict else None
    
    return None

def _retry_with_backoff(symbol: str, wait_time: float, attempt_type: str) -> None:
    """Executes a retry wait with a visual progress bar.

    Args:
        symbol: The symbol being retried.
        wait_time: The duration to wait in seconds.
        attempt_type: The type of attempt (e.g., "Retrying", "Error Retry").
    """
    pbar_color = TQDM_COLORS['retry'] if 'retry' in attempt_type.lower() else TQDM_COLORS['error_retry']
    retry_pbar = _create_progress_bar(
        total=int(wait_time * 10),
        desc=f"{attempt_type} {symbol}",
        color=pbar_color,
        unit='s',
        leave=False
    )
    
    for _ in range(int(wait_time * 10)):
        time.sleep(0.1)
        retry_pbar.update(1)
    retry_pbar.close()

def _calculate_memory_usage(dfs: Dict[str, Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]]) -> float:
    """Calculates the total memory usage of loaded dataframes.

    Args:
        dfs: A dictionary mapping symbols to their loaded data.

    Returns:
        The total memory usage in megabytes.
    """
    memory_pbar = _create_progress_bar(
        total=len(dfs), desc="Calculating memory usage", 
        color=TQDM_COLORS['memory'], leave=False
    )
    
    total_memory_mb = 0.0
    for symbol, data in dfs.items():
        if isinstance(data, dict):
            for tf_df in data.values():
                if isinstance(tf_df, pd.DataFrame):
                    total_memory_mb += tf_df.memory_usage(deep=True).sum()
        elif isinstance(data, pd.DataFrame):
            total_memory_mb += data.memory_usage(deep=True).sum()
        
        memory_pbar.update(1)
    
    memory_pbar.close()
    return total_memory_mb / (1024 * 1024)

def _safe_time_range(index: pd.Index) -> str:
    """Safely formats the min/max of a pandas Index as a time range string.

    Args:
        index: The pandas Index to format.

    Returns:
        A string representing the time range, or 'N/A' if not applicable.
    """
    try:
        if not isinstance(index, (pd.Index, pd.DatetimeIndex)) or index.empty:
            return "N/A"

        # The min/max of an empty index can raise a ValueError
        if index.empty:
            return "N/A"
            
        idx_min, idx_max = index.min(), index.max()

        # Check if the index is of a datetime type
        if pd.api.types.is_datetime64_any_dtype(index.dtype):
            min_val = cast(pd.Timestamp, idx_min)
            max_val = cast(pd.Timestamp, idx_max)
            
            if pd.isna(min_val) or pd.isna(max_val):
                return "N/A"
            
            return f"{min_val.strftime('%Y-%m-%d')} to {max_val.strftime('%Y-%m-%d')}"

        return f"Index {idx_min} to {idx_max}"
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not format time range for index: {e}")
        return "Invalid time range"

def _load_symbol_with_retry(
    processor: Any,
    symbol: str,
    timeframes: Optional[List[str]],
    load_multi_timeframes: bool,
    max_retries: int,
    attempt: int = 1
) -> Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]:
    """
    Wrapper to load symbol data with retries and exponential backoff.

    Args:
        processor: The data processor instance.
        symbol: The trading symbol.
        timeframes: A list of timeframes.
        load_multi_timeframes: Flag to load multiple timeframes.
        max_retries: Maximum number of retries.
        attempt: The current retry attempt number.

    Returns:
        The loaded data for the symbol, or None if all retries fail.
    """
    try:
        data = load_symbol_data(processor, symbol, timeframes, load_multi_timeframes)
        # A non-empty DataFrame or a non-empty Dict of DataFrames is a success.
        if (isinstance(data, pd.DataFrame) and not data.empty) or (isinstance(data, dict) and data):
            return data
        logger.warning(f"Initial data load for {symbol} returned empty. Retrying...")
    except Exception as e:
        logger.error(f"Exception on initial load for {symbol}: {e}. Retrying...")

    if attempt > max_retries:
        logger.error(f"All {max_retries} retries failed for {symbol}.")
        return None

    wait_time = DEFAULT_BACKOFF_FACTOR ** (attempt - 1)
    _retry_with_backoff(symbol, wait_time, f"Retry {attempt}/{max_retries}")
    return _load_symbol_with_retry(
        processor, symbol, timeframes, load_multi_timeframes, max_retries, attempt + 1
    )

def load_all_symbols_data(
    processor: Any,
    symbols: List[str],
    timeframes: Optional[List[str]] = None,
    load_multi_timeframes: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_workers: Optional[int] = None
) -> Dict[str, Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]]:
    """Loads trading data for multiple symbols using parallel processing.

    This function employs a thread pool to fetch data for multiple symbols
    concurrently. It includes a retry mechanism with exponential backoff for
    handling transient failures.

    Args:
        processor: An instance of a data processor with caching capabilities.
        symbols: A list of trading symbols to load.
        timeframes: A list of timeframes to load for each symbol.
        load_multi_timeframes: If True, loads multiple timeframes per symbol.
        max_retries: The maximum number of retry attempts for a failed symbol.
        max_workers: The maximum number of parallel worker threads. Defaults to
            the number of available CPU cores.

    Returns:
        A dictionary mapping each symbol to its loaded data. The value can be
        a DataFrame, a dictionary of DataFrames, or None if loading failed.
    """
    if not (symbols and isinstance(symbols, list) and hasattr(processor, 'df_cache')):
        logger.error("Invalid input: 'symbols' must be a non-empty list and 'processor' must be valid.")
        return {}

    all_data: Dict[str, Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]] = {}
    failed_symbols: List[str] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                _load_symbol_with_retry,
                processor,
                symbol,
                timeframes,
                load_multi_timeframes,
                max_retries
            ): symbol for symbol in symbols
        }
        
        pbar = _create_progress_bar(
            total=len(symbols), desc="Loading all symbols data", color=TQDM_COLORS['main']
        )
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                all_data[symbol] = data
                
                is_success = (isinstance(data, pd.DataFrame) and not data.empty) or \
                             (isinstance(data, dict) and data)

                status_icon = '✅' if is_success else '❌'
                if not is_success:
                    failed_symbols.append(symbol)
                    
                pbar.set_postfix_str(f"{symbol} {status_icon}", refresh=True)

            except Exception as exc:
                logger.critical(f"{symbol} generated a critical exception: {exc}", exc_info=True)
                all_data[symbol] = None
                failed_symbols.append(symbol)
            pbar.update(1)
        pbar.close()

    successful_count = len(symbols) - len(failed_symbols)
    logger.success(
        f"Data loading complete. "
        f"Successful: {successful_count}/{len(symbols)}. "
        f"Failed: {len(failed_symbols)}."
    )
    
    if failed_symbols:
        logger.warning(f"Failed symbols: {', '.join(sorted(failed_symbols))}")

    total_memory = _calculate_memory_usage(all_data)
    logger.memory(f"Total memory usage of loaded data: {total_memory:.2f} MB")
    
    return all_data
