import logging
import os
import pandas as pd
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict
from tqdm import tqdm 

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
from signals._components._generate_signals_hmm import generate_signal_hmm

from utilities._logger import setup_logging
logger = setup_logging(module_name="process_signals_hmm", log_level=logging.INFO)

DATAFRAME_COLUMNS = ['Pair', 'SignalTimeframe', 'SignalType']

# Thread-safe lock for logging
_LOG_LOCK = threading.Lock()

def _get_cpu_count() -> int:
    """Get CPU count using available methods."""
    try:
        if _HAS_PSUTIL:
            import psutil
            return psutil.cpu_count() or 4
        else:
            return os.cpu_count() or 4
    except Exception:
        return 4  # Safe fallback

def _process_symbol_worker(symbol: str, symbol_data_for_tfs: Optional[Dict[str, pd.DataFrame]], 
                        actual_timeframes_to_scan: List[str], strict_mode: bool,
                        include_long_signals: bool = True, include_short_signals: bool = True) -> Optional[Dict[str, str]]:
    """
    Worker function to process a single symbol for HMM signals.
    Thread-safe function that can be used in parallel processing.
    
    Args:
        symbol: Symbol name to process
        symbol_data_for_tfs: Dictionary of timeframe data for this symbol
        actual_timeframes_to_scan: List of timeframes to scan in priority order
        strict_mode: Whether to use strict HMM mode
        include_long_signals: Whether to include LONG signals (signal == 1)
        include_short_signals: Whether to include SHORT signals (signal == -1)
        
    Returns:
        Dictionary with signal data if signal found, None otherwise
    """
    try:
        if not symbol_data_for_tfs or not isinstance(symbol_data_for_tfs, dict):
            with _LOG_LOCK:
                logger.warning(f"  No data or incorrect data format for {symbol}. Skipping.")
            return None
        
        # Find signals on prioritized timeframes
        for tf_scan in actual_timeframes_to_scan:
            df_current_tf = symbol_data_for_tfs.get(tf_scan)

            if df_current_tf is None or df_current_tf.empty or len(df_current_tf) < 50:
                with _LOG_LOCK:
                    logger.data(f"  Insufficient data for {symbol} on {tf_scan} (need at least 50 candles)")
                continue

            try:
                # Use generate_signal_hmm to get HMM signal
                result = generate_signal_hmm(symbol, df_current_tf, strict_mode)
                if result is None or len(result) != 3:  # Changed from 5 to 3
                    with _LOG_LOCK:
                        logger.debug(f"  Invalid result for {symbol} on {tf_scan}")
                    continue
                
                # Unpack the 3-element tuple correctly
                _, signal, _ = result  # (pair, signal, current_close)
                
                # Process based on signal value and user preferences
                if signal == 1 and include_long_signals:
                    # LONG signal
                    with _LOG_LOCK:
                        logger.signal(f"  LONG signal found for {symbol} on {tf_scan}")
                    return {
                        'Pair': symbol,
                        'SignalTimeframe': tf_scan,
                        'SignalType': 'LONG'
                    }
                elif signal == -1 and include_short_signals:
                    # SHORT signal
                    with _LOG_LOCK:
                        logger.signal(f"  SHORT signal found for {symbol} on {tf_scan}")
                    return {
                        'Pair': symbol,
                        'SignalTimeframe': tf_scan,
                        'SignalType': 'SHORT'
                    }
                else:
                    with _LOG_LOCK:
                        signal_type = "LONG" if signal == 1 else "SHORT" if signal == -1 else "HOLD"
                        if signal == 0 or signal is None:
                            logger.debug(f"  HOLD signal for {symbol} on {tf_scan} - no action")
                        else:
                            logger.debug(f"  {signal_type} signal for {symbol} on {tf_scan} - skipping (not enabled)")
                    
            except Exception as e:
                with _LOG_LOCK:
                    logger.error(f"    Error getting HMM signal for {symbol} on {tf_scan}: {e}")
                    
        return None
        
    except Exception as e:
        with _LOG_LOCK:
            logger.error(f"Error processing symbol {symbol}: {e}")
        return None

def process_signals_hmm(
    preloaded_data: Dict[str, Dict[str, pd.DataFrame]],
    timeframes_to_scan: List[str],
    strict_mode: bool = False,
    include_long_signals: bool = True,
    include_short_signals: bool = True,
    max_workers: int = 4  
):
    """
    Processes trading signals for cryptocurrency symbols using HMM models.
    Returns symbols with LONG and/or SHORT signals based on parameters.
    
    Args:
        preloaded_data (Dict[str, Dict[str, pd.DataFrame]]): Pre-loaded symbol data in format:
                                                        {symbol: {timeframe: dataframe}}
        timeframes_to_scan (List[str]): Specific timeframes to scan, in order of priority. 
                                                Defaults to ['1h', '4h', '1d'].
        strict_mode (bool): Whether to use STRICT-HMM mode for HMM models.
        include_long_signals (bool): Whether to include LONG signals (signal == 1).
        include_short_signals (bool): Whether to include SHORT signals (signal == -1).
        max_workers (int): Maximum number of worker threads for parallel processing.
    
    Returns:
        pd.DataFrame: DataFrame containing symbols with signals, columns ['Pair', 'SignalTimeframe', 'SignalType'].
    """
    # Validate signal type parameters
    if not include_long_signals and not include_short_signals:
        logger.warning("Both include_long_signals and include_short_signals are False. No signals will be processed.")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)
    
    signal_types = []
    if include_long_signals:
        signal_types.append("LONG")
    if include_short_signals:
        signal_types.append("SHORT")
    
    mode_name = "STRICT-HMM" if strict_mode else "NON-STRICT-HMM"
    logger.model("===============================================")
    logger.model(f"STARTING HMM SIGNAL ANALYSIS (CRYPTO) - {mode_name} MODE")
    logger.config(f"Signal types enabled: {', '.join(signal_types)}")
    logger.model("===============================================")

    # Validate input data
    if not preloaded_data or not isinstance(preloaded_data, dict):
        logger.error("No preloaded_data provided or invalid format. Expected Dict[str, Dict[str, pd.DataFrame]]")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)    
    
    # Initialize parameters with defaults
    actual_timeframes_to_scan = timeframes_to_scan if timeframes_to_scan is not None else ['1h', '4h', '1d']
    logger.config(f"Using {'default' if timeframes_to_scan is None else 'specified'} timeframes for HMM scan (priority order): {actual_timeframes_to_scan}")
    
    if not actual_timeframes_to_scan:
        logger.warning("No timeframes specified or defaulted for HMM scan. Cannot proceed.")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)

    symbols_to_analyze = list(preloaded_data.keys())
    logger.analysis(f"Analyzing {len(symbols_to_analyze)} crypto symbols from preloaded data.")    
    
    # Use provided max_workers parameter or calculate optimal number (80% of CPU cores)
    if max_workers is None or max_workers <= 0:
        cpu_count = _get_cpu_count()
        optimal_workers = max(1, int(cpu_count * 0.8))
        max_workers = optimal_workers
        logger.process(f"Using {max_workers} worker threads for parallel processing (80% of {cpu_count} CPU cores)")
    else:
        logger.process(f"Using {max_workers} worker threads for parallel processing (user-specified)")

    # --- Parallel Signal Processing per Symbol ---
    signals_list = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all symbol processing tasks
        future_to_symbol = {}
        for symbol in symbols_to_analyze:
            symbol_data = preloaded_data.get(symbol)
            if symbol_data is not None:  # Only submit if data exists
                future = executor.submit(
                    _process_symbol_worker, 
                    symbol, 
                    symbol_data, 
                    actual_timeframes_to_scan, 
                    strict_mode,
                    include_long_signals,
                    include_short_signals
                )
                future_to_symbol[future] = symbol
        
        # Process completed tasks and collect results
        total_symbols_submitted = len(future_to_symbol)  # Use submitted count instead
        
        # Modern progress bar with black box style
        for future in tqdm(
            as_completed(future_to_symbol), 
            total=total_symbols_submitted, 
            desc=f"⚡ HMM Analysis ({mode_name})", 
            unit="", 
            ascii=False,  # Enable Unicode for better visuals
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            colour='green',
            ncols=100,
            leave=True,
            dynamic_ncols=True,
            miniters=1,  # Update frequency
            maxinterval=0.1  # Max update interval
        ):
            symbol = future_to_symbol[future]
            
            try:
                result = future.result()
                if result is not None:
                    signals_list.append(result)
                    
            except Exception as e:
                # Log error but continue processing other symbols
                logger.error(f"Error processing symbol {symbol}: {e}")
                continue

    # Count signal types for logging
    long_count = sum(1 for signal in signals_list if signal.get('SignalType') == 'LONG')
    short_count = sum(1 for signal in signals_list if signal.get('SignalType') == 'SHORT')
    
    logger.model("===============================================")
    logger.success(f"COMPLETED HMM SIGNAL ANALYSIS - Found {len(signals_list)} total signals")
    logger.signal(f"LONG signals: {long_count}, SHORT signals: {short_count}")
    logger.model("===============================================")

    return pd.DataFrame(signals_list) if signals_list else pd.DataFrame([], columns=DATAFRAME_COLUMNS)

def reload_timeframes_for_symbols(processor, symbols: List[str], timeframes: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Reload data for specific symbols with specified timeframes for optimized HMM analysis.
    
    Args:
        processor: tick_processor instance with data loading capabilities
        symbols: List of symbol names to reload data for
        timeframes: List of timeframes to reload (e.g., ['5m', '15m', '30m', '1h', '4h', '1d'])
        
    Returns:
        Dictionary with symbol as key and nested dictionary of {timeframe: DataFrame} as value
    """
    logger.data(f"Reloading data for {len(symbols)} symbols with timeframes: {timeframes}")
    
    # Import the load_pair_data function
    from livetrade._components._load_all_pairs_data import load_symbol_data
    
    reloaded_data = {}
    
    for symbol in symbols:
        try:
            logger.network(f"Reloading timeframes for {symbol}...")
            
            # Load multi-timeframe data for this symbol
            symbol_data = load_symbol_data(
                processor=processor,
                symbol=symbol,
                timeframes=timeframes,
                load_multi_timeframes=True
            )
            
            if symbol_data is not None and isinstance(symbol_data, dict):
                # Verify we have data for all required timeframes
                valid_timeframes = {}
                for tf in timeframes:
                    if tf in symbol_data and symbol_data[tf] is not None and not symbol_data[tf].empty:
                        valid_timeframes[tf] = symbol_data[tf]
                        logger.data(f"  ✓ {tf}: {symbol_data[tf].shape[0]} rows")
                    else:
                        logger.warning(f"  ❌ {tf}: No data available")
                
                if valid_timeframes:
                    reloaded_data[symbol] = valid_timeframes
                    logger.success(f"  Successfully reloaded {len(valid_timeframes)} timeframes for {symbol}")
                else:
                    logger.warning(f"  No valid timeframes found for {symbol}")
            else:
                logger.error(f"  Failed to reload data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error reloading data for {symbol}: {e}")
    
    logger.success(f"Successfully reloaded data for {len(reloaded_data)}/{len(symbols)} symbols")
    return reloaded_data
