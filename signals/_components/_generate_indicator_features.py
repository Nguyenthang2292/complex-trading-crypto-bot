import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging

logger = setup_logging(module_name="_generate_indicator_features", log_level=logging.DEBUG)

from livetrade.config import (
    BB_STD_MULTIPLIER, 
    BB_WINDOW, 
    COL_BB_LOWER, 
    COL_BB_UPPER, 
    COL_CLOSE,
    MACD_FAST_PERIOD, 
    MACD_SIGNAL_PERIOD, 
    MACD_SLOW_PERIOD, 
    RSI_PERIOD, 
    SMA_PERIOD
)

def _calculate_rsi_vectorized(close_values: np.ndarray, period: int) -> np.ndarray:
    """Calculate RSI using vectorized NumPy operations for maximum performance."""
    deltas: np.ndarray = np.diff(close_values, prepend=close_values[0])
    gains: np.ndarray = np.where(deltas > 0, deltas, 0)
    losses: np.ndarray = np.where(deltas < 0, -deltas, 0)
    
    # Use pandas for rolling mean (optimized internally)
    avg_gains: np.ndarray = pd.Series(gains).rolling(window=period, min_periods=1).mean().values
    avg_losses: np.ndarray = pd.Series(losses).rolling(window=period, min_periods=1).mean().values
    
    # Vectorized RSI calculation
    rs: np.ndarray = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
    rsi: np.ndarray = 100 - (100 / (1 + rs))
    
    return rsi

def _calculate_macd_vectorized(close_values: np.ndarray, fast_period: int, slow_period: int, signal_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate MACD using vectorized operations with exponential smoothing."""
    # Vectorized EMA calculation using pandas (optimized)
    close_series: pd.Series = pd.Series(close_values)
    ema_fast: np.ndarray = close_series.ewm(span=fast_period, adjust=False).mean().values
    ema_slow: np.ndarray = close_series.ewm(span=slow_period, adjust=False).mean().values
    
    # Vectorized MACD line calculation
    macd_line: np.ndarray = ema_fast - ema_slow
    
    # Vectorized signal line calculation
    macd_signal: np.ndarray = pd.Series(macd_line).ewm(span=signal_period, adjust=False).mean().values
    
    return macd_line, macd_signal

def _calculate_bollinger_bands_vectorized(close_values: np.ndarray, window: int, std_multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands using vectorized rolling operations."""
    close_series: pd.Series = pd.Series(close_values)
    
    # Vectorized rolling calculations
    sma: np.ndarray = close_series.rolling(window=window, min_periods=1).mean().values
    rolling_std: np.ndarray = close_series.rolling(window=window, min_periods=1).std().values
    
    # Vectorized band calculations
    bb_upper: np.ndarray = sma + (rolling_std * std_multiplier)
    bb_lower: np.ndarray = sma - (rolling_std * std_multiplier)
    
    return bb_upper, bb_lower

def _calculate_sma_vectorized(close_values: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average using vectorized operations."""
    return pd.Series(close_values).rolling(window=period, min_periods=1).mean().values

def _generate_indicator_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators using vectorized NumPy operations for optimal performance.
    
    Computes RSI, MACD, Bollinger Bands, and SMA indicators using pandas_ta with NumPy
    vectorized fallback calculations for maximum speed and reliability.
    
    Args:
        df_input: DataFrame with OHLCV price data containing lowercase column names
        
    Returns:
        DataFrame with added technical indicators (rsi, macd, macd_signal, bb_upper, 
        bb_lower, ma_20, ma_20_slope) or empty DataFrame on critical error
    """
    try:
        if df_input.empty or COL_CLOSE not in df_input.columns:
            logger.warning("Input DataFrame is empty or missing close column")
            return pd.DataFrame()
        
        df: pd.DataFrame = df_input.copy()
        close_values: np.ndarray = df[COL_CLOSE].values
        
        # Vectorized RSI calculation
        try:
            df['rsi'] = ta.rsi(df[COL_CLOSE], length=RSI_PERIOD)
            if df['rsi'].isna().all():
                df['rsi'] = _calculate_rsi_vectorized(close_values, RSI_PERIOD)
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}. Using vectorized calculation.")
            df['rsi'] = _calculate_rsi_vectorized(close_values, RSI_PERIOD)
            
        # Vectorized MACD calculation
        try:
            macd_output: Optional[pd.DataFrame] = ta.macd(
                df[COL_CLOSE], fast=MACD_FAST_PERIOD, slow=MACD_SLOW_PERIOD, signal=MACD_SIGNAL_PERIOD
            )
            if macd_output is not None and len(macd_output.columns) >= 2:
                df['macd'] = macd_output.iloc[:, 0]
                df['macd_signal'] = macd_output.iloc[:, 1]
            else:
                macd_line, macd_signal = _calculate_macd_vectorized(
                    close_values, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD
                )
                df['macd'] = macd_line
                df['macd_signal'] = macd_signal
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}. Using vectorized calculation.")
            macd_line, macd_signal = _calculate_macd_vectorized(
                close_values, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD
            )
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            
        # Vectorized Bollinger Bands calculation
        try:
            bb: Optional[pd.DataFrame] = ta.bbands(df[COL_CLOSE], length=BB_WINDOW, std=BB_STD_MULTIPLIER)
            if bb is not None and not bb.empty:
                bb_cols: List[str] = bb.columns.tolist()
                upper_col: List[str] = [col for col in bb_cols if 'BBU' in col]
                lower_col: List[str] = [col for col in bb_cols if 'BBL' in col]
                
                if upper_col and lower_col:
                    df[COL_BB_UPPER] = bb[upper_col[0]]
                    df[COL_BB_LOWER] = bb[lower_col[0]]
                else:
                    bb_upper, bb_lower = _calculate_bollinger_bands_vectorized(
                        close_values, BB_WINDOW, BB_STD_MULTIPLIER
                    )
                    df[COL_BB_UPPER] = bb_upper
                    df[COL_BB_LOWER] = bb_lower
            else:
                bb_upper, bb_lower = _calculate_bollinger_bands_vectorized(
                    close_values, BB_WINDOW, BB_STD_MULTIPLIER
                )
                df[COL_BB_UPPER] = bb_upper
                df[COL_BB_LOWER] = bb_lower
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}. Using vectorized calculation.")
            bb_upper, bb_lower = _calculate_bollinger_bands_vectorized(
                close_values, BB_WINDOW, BB_STD_MULTIPLIER
            )
            df[COL_BB_UPPER] = bb_upper
            df[COL_BB_LOWER] = bb_lower
            
        # Vectorized SMA calculation
        try:
            df['ma_20'] = ta.sma(df[COL_CLOSE], length=SMA_PERIOD)
            if df['ma_20'].isna().all():
                df['ma_20'] = _calculate_sma_vectorized(close_values, SMA_PERIOD)
        except Exception as e:
            logger.warning(f"SMA calculation failed: {e}. Using vectorized calculation.")
            df['ma_20'] = _calculate_sma_vectorized(close_values, SMA_PERIOD)

        # Vectorized slope calculation
        df['ma_20_slope'] = np.gradient(df['ma_20'].fillna(method='ffill').values)

        # Optimized NaN handling
        numeric_columns: List[str] = ['rsi', 'macd', 'macd_signal', COL_BB_UPPER, COL_BB_LOWER, 'ma_20', 'ma_20_slope']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        result_df: pd.DataFrame = df.dropna()
        if result_df.empty:
            logger.warning("All features resulted in NaN - insufficient data for technical indicators")
        else:
            logger.info(f"Technical indicators calculated successfully for {len(result_df)} rows")
    
        return result_df
        
    except Exception as e:
        logger.error(f"Error in feature calculation: {e}")
        return pd.DataFrame()

