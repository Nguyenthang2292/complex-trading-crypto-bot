import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from utilities._logger import setup_logging

# Initialize logger
logger = setup_logging(module_name="_combine_all_dataframes", log_level=logging.DEBUG)

def _process_single_dataframe(
    symbol_key: str, tf_key: str, df_original: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Processes and validates a single symbol's DataFrame.

    Args:
        symbol_key: The symbol name (e.g., 'BTCUSDT').
        tf_key: The timeframe string (e.g., '1h').
        df_original: The input DataFrame to process.

    Returns:
        A tuple containing the processed DataFrame and a status string 
        ('processed', 'skipped', 'error').
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if not isinstance(df_original, pd.DataFrame) or df_original.empty:
        logger.debug(f"Skipping {symbol_key}/{tf_key}: DataFrame is invalid or empty.")
        return None, 'skipped'

    df_processed = df_original.copy()
    df_processed['symbol'] = symbol_key
    df_processed['timeframe'] = tf_key

    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        logger.warning(f"Skipping {symbol_key}/{tf_key}: Missing required columns {missing_cols}.")
        return None, 'skipped'

    try:
        numeric_cols = required_cols
        df_processed[numeric_cols] = df_processed[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        if not df_processed.empty:
            nan_count = df_processed[numeric_cols].isna().sum().sum()
            total_elements = len(df_processed) * len(numeric_cols)
            if total_elements > 0:
                nan_ratio = nan_count / total_elements
                if nan_ratio > 0.5:
                    logger.warning(
                        f"High NaN ratio ({nan_ratio:.2%}) in {symbol_key}/{tf_key}. "
                        "Data quality may be poor."
                    )
        
        logger.debug(f"Successfully processed {symbol_key}/{tf_key}: {len(df_processed)} rows.")
        return df_processed, 'processed'
        
    except Exception as e:
        logger.error(f"Error processing data types for {symbol_key}/{tf_key}: {e}", exc_info=True)
        return None, 'error'

def combine_all_dataframes(
    all_symbols_data: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Normalizes, validates, and combines multiple DataFrames for different symbols
    and timeframes into a single, unified DataFrame.

    The function iterates through a nested dictionary structure, processes each
    individual DataFrame to ensure it meets data quality standards, and then
    concatenates them into one large DataFrame.

    Args:
        all_symbols_data: A dictionary structured as {symbol: {timeframe: DataFrame}}.
            Each DataFrame should contain OHLCV data.

    Returns:
        A single pandas DataFrame containing the combined, normalized data from all
        valid input DataFrames, with additional 'symbol' and 'timeframe' columns.
        Returns an empty DataFrame if no valid data can be processed.

    Raises:
        TypeError: If the input `all_symbols_data` is not a dictionary.
    """
    if not isinstance(all_symbols_data, dict):
        raise TypeError(f"Expected a dictionary for all_symbols_data, but got {type(all_symbols_data)}.")
    
    if not all_symbols_data:
        logger.warning("Input 'all_symbols_data' is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    dataframes_to_combine: List[pd.DataFrame] = []
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    logger.info("Starting DataFrame normalization and combination process...")
    
    for symbol, timeframes in all_symbols_data.items():
        if not isinstance(timeframes, dict) or not timeframes:
            logger.warning(f"Invalid or empty timeframe data for symbol '{symbol}'. Skipping.")
            stats['skipped'] += 1
            continue
            
        for timeframe, df in timeframes.items():
            try:
                processed_df, status = _process_single_dataframe(symbol, timeframe, df)
                if status in stats:
                    stats[status] += 1
                else:
                    logger.warning(f"Received unknown status '{status}' from processor for {symbol}/{timeframe}.")
                    stats['errors'] += 1
                
                if status == 'processed' and processed_df is not None:
                    dataframes_to_combine.append(processed_df)
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}/{timeframe}: {e}", exc_info=True)
                stats['errors'] += 1

    if not dataframes_to_combine:
        logger.warning(
            "No valid DataFrames were available to combine. "
            f"Processing stats: {stats['processed']} processed, "
            f"{stats['skipped']} skipped, {stats['errors']} errors."
        )
        return pd.DataFrame()
    
    try:
        combined_df = pd.concat(dataframes_to_combine, ignore_index=True, sort=False)
        if combined_df.empty:
            logger.warning("Combined DataFrame is empty after concatenation.")
            return pd.DataFrame()
        
        combined_df = combined_df.sort_values(['symbol', 'timeframe']).reset_index(drop=True)
        
        logger.info(
            f"Successfully combined dataset: {len(combined_df)} total rows from "
            f"{len(dataframes_to_combine)} sources."
        )
        logger.info(
            f"Processing stats: {stats['processed']} processed, "
            f"{stats['skipped']} skipped, {stats['errors']} errors."
        )
        return combined_df
        
    except Exception as e:
        logger.error(f"Fatal error during final DataFrame concatenation: {e}", exc_info=True)
        return pd.DataFrame()