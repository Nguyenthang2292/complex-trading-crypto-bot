import logging
import os
import pandas as pd
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from signals.signals_hmm import hmm_signals
from signals._components.HMM__class__OptimizingParameters import OptimizingParameters
from utilities._logger import setup_logging

logger = setup_logging(module_name="_process_signals_hmm", log_level=logging.DEBUG)

DATAFRAME_COLUMNS: List[str] = ['Symbol', 'SignalTimeframe', 'SignalType']
_LOG_LOCK: threading.Lock = threading.Lock()


def _get_cpu_count() -> int:
    """Get optimal CPU count using available system information."""
    try:
        return psutil.cpu_count() if _HAS_PSUTIL else (os.cpu_count() or 4)
    except Exception:
        return 4

def _generate_signal_hmm(pair: str, 
                        df: pd.DataFrame, 
                        strict_mode: bool = False) -> Tuple[str, Optional[int], Optional[float]]:
    """
    Generate HMM-based trading signals for a cryptocurrency pair.
    
    Combines RSI and HMM analysis to generate trading signals. Returns LONG signal (1)
    when RSI > 60 and HMM indicates bullish, SHORT signal (-1) when RSI < 40 and HMM 
    indicates bearish, otherwise no signal (None).
    
    Args:
        pair: Trading pair symbol identifier
        df: OHLC price DataFrame with required columns ['High', 'Low', 'close']
        strict_mode: Enable strict HMM parameter mode for conservative signals
        
    Returns:
        Tuple containing (pair_name, signal_value, current_price) where:
        - signal_value: 1 for LONG, -1 for SHORT, None for no signal
        - current_price: Latest close price or None if error
    """
    try:
        if 'close' not in df.columns:
            logger.data(f"{pair}: Missing required 'close' column")
            return pair, None, None
            
        import pandas_ta as ta
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        if df['rsi'].isna().all() or len(df['rsi'].dropna()) < 14:
            logger.data(f"{pair}: RSI calculation failed or insufficient data")
            return pair, None, None
            
        latest_rsi: float = df['rsi'].iloc[-1]
        rsi_signal: Optional[int] = (1 if latest_rsi > 60 else -1 if latest_rsi < 40 else None)
        
        logger.analysis(f"{pair}: RSI = {latest_rsi:.2f}, Signal: {rsi_signal}")
        
        if len(df) < 50:
            logger.data(f"{pair}: Insufficient data for HMM analysis ({len(df)} points)")
            return pair, None, None
            
        current_close: float = df['close'].iloc[-1]
        params = OptimizingParameters()
        params.strict_mode = strict_mode
        
        high_order_signal, hmm_kama_signal = hmm_signals(df, optimizing_params=params)
        
        if high_order_signal is None or hmm_kama_signal is None:
            logger.model(f"{pair}: Invalid HMM signals")
            return pair, None, current_close
            
        hmm_signal: int = (1 if (high_order_signal == 1 or hmm_kama_signal == 1) 
                          else -1 if (high_order_signal == -1 or hmm_kama_signal == -1) 
                          else 0)
        
        logger.model(f"{pair}: HMM signals - High Order: {high_order_signal}, KAMA: {hmm_kama_signal}, Combined: {hmm_signal}")
        
        score: int = (2 if hmm_signal == 1 else -2 if hmm_signal == -1 else 0) + (1 if rsi_signal == 1 else -1 if rsi_signal == -1 else 0)
        final_signal: Optional[int] = 1 if score >= 2 else -1 if score <= -2 else None
        
        return pair, final_signal, current_close
        
    except Exception as e:
        logger.error(f"Error analyzing {pair}: {str(e)}")
        return pair, None, None

def _process_symbol_worker(symbol: str, 
                           symbol_data_for_tfs: Optional[Dict[str, pd.DataFrame]], 
                          actual_timeframes_to_scan: List[str], 
                          strict_mode: bool,
                          include_long_signals: bool = True, 
                          include_short_signals: bool = True) -> Optional[Dict[str, str]]:
    """
    Worker function to process a single symbol for HMM signals.
    
    Thread-safe function that processes a symbol across multiple timeframes to find
    trading signals using HMM analysis. Returns the first valid signal found.
    
    Args:
        symbol: Symbol name to process
        symbol_data_for_tfs: Dictionary mapping timeframes to DataFrames for this symbol
        actual_timeframes_to_scan: List of timeframes to scan in priority order
        strict_mode: Whether to use strict HMM mode for conservative signals
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
        
        for tf_scan in actual_timeframes_to_scan:
            df_current_tf = symbol_data_for_tfs.get(tf_scan)

            if df_current_tf is None or df_current_tf.empty or len(df_current_tf) < 50:
                with _LOG_LOCK:
                    logger.data(f"  Insufficient data for {symbol} on {tf_scan} (need at least 50 candles)")
                continue

            try:
                result = _generate_signal_hmm(symbol, df_current_tf, strict_mode)
                if result is None or len(result) != 3:
                    with _LOG_LOCK:
                        logger.debug(f"  Invalid result for {symbol} on {tf_scan}")
                    continue
                
                _, signal, _ = result
                
                if signal == 1 and include_long_signals:
                    with _LOG_LOCK:
                        logger.signal(f"  LONG signal found for {symbol} on {tf_scan}")
                    return {
                        'Symbol': symbol,
                        'SignalTimeframe': tf_scan,
                        'SignalType': 'LONG'
                    }
                elif signal == -1 and include_short_signals:
                    with _LOG_LOCK:
                        logger.signal(f"  SHORT signal found for {symbol} on {tf_scan}")
                    return {
                        'Symbol': symbol,
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
) -> pd.DataFrame:
    """
    Process trading signals for cryptocurrency symbols using HMM models.
    
    Analyzes symbols across multiple timeframes using Hidden Markov Models combined with
    RSI indicators to generate LONG/SHORT trading signals with parallel processing.
    
    Args:
        preloaded_data: Pre-loaded symbol data in format {symbol: {timeframe: dataframe}}
        timeframes_to_scan: Specific timeframes to scan, in priority order
        strict_mode: Whether to use STRICT-HMM mode for conservative signals
        include_long_signals: Whether to include LONG signals (signal == 1)
        include_short_signals: Whether to include SHORT signals (signal == -1)
        max_workers: Maximum number of worker threads for parallel processing
    
    Returns:
        DataFrame containing symbols with signals, columns ['Symbol', 'SignalTimeframe', 'SignalType']
    """
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

    if not preloaded_data or not isinstance(preloaded_data, dict):
        logger.error("No preloaded_data provided or invalid format. Expected Dict[str, Dict[str, pd.DataFrame]]")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)    
    
    actual_timeframes_to_scan = timeframes_to_scan if timeframes_to_scan is not None else ['1h', '4h', '1d']
    logger.config(f"Using {'default' if timeframes_to_scan is None else 'specified'} timeframes for HMM scan (priority order): {actual_timeframes_to_scan}")
    
    if not actual_timeframes_to_scan:
        logger.warning("No timeframes specified or defaulted for HMM scan. Cannot proceed.")
        return pd.DataFrame([], columns=DATAFRAME_COLUMNS)

    symbols_to_analyze = list(preloaded_data.keys())
    logger.analysis(f"Analyzing {len(symbols_to_analyze)} crypto symbols from preloaded data.")    
    
    if max_workers is None or max_workers <= 0:
        cpu_count = _get_cpu_count()
        max_workers = max(1, int(cpu_count * 0.8))
        logger.process(f"Using {max_workers} worker threads for parallel processing (80% of {cpu_count} CPU cores)")
    else:
        logger.process(f"Using {max_workers} worker threads for parallel processing (user-specified)")

    signals_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {}
        for symbol in symbols_to_analyze:
            symbol_data = preloaded_data.get(symbol)
            if symbol_data is not None:
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
        
        total_symbols_submitted = len(future_to_symbol)
        
        for future in tqdm(
            as_completed(future_to_symbol), 
            total=total_symbols_submitted, 
            desc=f"⚡ HMM Analysis ({mode_name})", 
            unit="", 
            ascii=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            colour='green',
            ncols=100,
            leave=True,
            dynamic_ncols=True,
            miniters=1,
            maxinterval=0.1
        ):
            symbol = future_to_symbol[future]
            
            try:
                result = future.result()
                if result is not None:
                    signals_list.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                continue

    long_count = sum(1 for signal in signals_list if signal.get('SignalType') == 'LONG')
    short_count = sum(1 for signal in signals_list if signal.get('SignalType') == 'SHORT')
    
    logger.model("===============================================")
    logger.success(f"COMPLETED HMM SIGNAL ANALYSIS - Found {len(signals_list)} total signals")
    logger.signal(f"LONG signals: {long_count}, SHORT signals: {short_count}")
    logger.model("===============================================")

    return pd.DataFrame(signals_list) if signals_list else pd.DataFrame([], columns=DATAFRAME_COLUMNS)

def reload_timeframes_for_symbols(processor: Any, 
                                  symbols: List[str], 
                                  timeframes: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
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
    
    from livetrade._components._load_all_symbols_data import load_symbol_data
    
    reloaded_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    for symbol in symbols:
        try:
            logger.network(f"Reloading timeframes for {symbol}...")
            
            symbol_data = load_symbol_data(
                processor=processor,
                symbol=symbol,
                timeframes=timeframes,
                load_multi_timeframes=True
            )
            
            if symbol_data is not None and isinstance(symbol_data, dict):
                valid_timeframes: Dict[str, pd.DataFrame] = {}
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
