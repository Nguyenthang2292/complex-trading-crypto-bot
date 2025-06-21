import logging
from typing import Dict
import pandas as pd
from utilities._logger import setup_logging

# Initialize logger
logger = setup_logging(module_name="_combine_all_dataframes", log_level=logging.DEBUG)

def combine_all_dataframes(all_symbols_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Normalize and combine trading pair DataFrames into a single dataset.

    Args:
        all_symbols_data: {symbol_name: {timeframe: DataFrame}} structure with OHLCV data
        
    Returns:
        Combined DataFrame with normalized columns + 'symbol' + 'timeframe' metadata
        
    Raises:
        TypeError: If input is not a dictionary
    """
    # Input validation
    if not isinstance(all_symbols_data, dict):
        raise TypeError(f"Expected dict for all_symbols_data, got {type(all_symbols_data)}")
    
    if not all_symbols_data:
        logger.warning("Empty all_symbols_data provided")
        return pd.DataFrame()

    dataframes_to_combine = []
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    processing_stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    logger.data("Starting DataFrame normalization and combination process...")
    
    for symbol_key, timeframes_data in all_symbols_data.items():
        if not isinstance(timeframes_data, dict) or not timeframes_data:
            logger.warning(f"Invalid or empty data format for {symbol_key}. Skipping.")
            processing_stats['skipped'] += 1
            continue
            
        for tf_key, df_original in timeframes_data.items():
            try:
                # Skip if DataFrame is empty or not a DataFrame
                if df_original is None or not isinstance(df_original, pd.DataFrame) or df_original.empty:
                    logger.debug(f"Invalid DataFrame for {symbol_key}/{tf_key}. Skipping.")
                    processing_stats['skipped'] += 1
                    continue
                
                # Create processed copy with metadata
                df_processed = df_original.copy()
                df_processed['symbol'] = str(symbol_key)
                df_processed['timeframe'] = str(tf_key)
                
                # Verify required columns exist
                missing_cols = [col for col in required_cols if col not in df_processed.columns]
                if missing_cols:
                    logger.warning(f"Skipping {symbol_key}/{tf_key}: missing required columns {missing_cols}")
                    processing_stats['skipped'] += 1
                    continue
                
                # Validate data types and handle potential issues
                try:
                    # Ensure numeric columns are properly typed
                    numeric_cols = [col for col in required_cols if col in df_processed.columns]
                    for col in numeric_cols:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    
                    # Check for excessive NaN values
                    nan_ratio = df_processed[numeric_cols].isna().sum().sum() / (len(df_processed) * len(numeric_cols))
                    if nan_ratio > 0.5:  # More than 50% NaN values
                        logger.warning(f"High NaN ratio ({nan_ratio:.2%}) in {symbol_key}/{tf_key}. Still including but consider data quality.")
                    
                    dataframes_to_combine.append(df_processed)
                    processing_stats['processed'] += 1
                    logger.debug(f"Successfully processed {symbol_key}/{tf_key}: {len(df_processed)} rows")
                    
                except Exception as e:
                    logger.error(f"Error processing data types for {symbol_key}/{tf_key}: {e}")
                    processing_stats['errors'] += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol_key}/{tf_key}: {e}")
                processing_stats['errors'] += 1

    # Combine DataFrames if any valid data exists
    if not dataframes_to_combine:
        logger.warning(f"No valid DataFrames to combine. Processing stats - "
                    f"Processed: {processing_stats['processed']}, "
                    f"Skipped: {processing_stats['skipped']}, "
                    f"Errors: {processing_stats['errors']}")
        return pd.DataFrame()
    
    try:
        combined_df = pd.concat(dataframes_to_combine, ignore_index=True, sort=False)
        
        # Final validation and cleanup
        if combined_df.empty:
            logger.warning("Combined DataFrame is empty after concatenation")
            return pd.DataFrame()
        
        # Sort by symbol and timeframe for consistency
        if 'symbol' in combined_df.columns and 'timeframe' in combined_df.columns:
            combined_df = combined_df.sort_values(['symbol', 'timeframe']).reset_index(drop=True)
        
        # Log success statistics
        logger.data(f"Successfully combined dataset: {len(combined_df)} total rows from {len(dataframes_to_combine)} sources")
        logger.info(f"Processing stats - Processed: {processing_stats['processed']}, "
                f"Skipped: {processing_stats['skipped']}, Errors: {processing_stats['errors']}")
        
        # Log DataFrame info for debugging
        unique_symbols = combined_df['symbol'].nunique() if 'symbol' in combined_df.columns else 0
        unique_timeframes = combined_df['timeframe'].nunique() if 'timeframe' in combined_df.columns else 0
        logger.debug(f"Combined DataFrame contains {unique_symbols} unique symbols and {unique_timeframes} unique timeframes")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error during DataFrame concatenation: {e}")
        return pd.DataFrame()