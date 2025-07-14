import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import sys
from typing import List, Optional, Tuple, Union, cast, Callable, Any

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities.logger import setup_logging

logger = setup_logging(module_name="generate_indicator_features", log_level=logging.DEBUG)

from config.config import (
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

def _to_numpy_array(data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """Converts pandas objects to NumPy arrays safely.

    Args:
        data: The pandas Series, DataFrame, or NumPy array to convert.

    Returns:
        The data as a NumPy array.
    """
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, 'values'):
        return cast(np.ndarray, data.values)
    return np.array(data)

def _calculate_rsi_vectorized(close_values: np.ndarray, period: int) -> np.ndarray:
    """Calculates RSI using vectorized NumPy operations.

    Args:
        close_values: A NumPy array of closing prices.
        period: The time period for RSI calculation.

    Returns:
        A NumPy array containing the RSI values.
    """
    deltas = np.diff(close_values, prepend=close_values[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = _to_numpy_array(pd.Series(gains).rolling(window=period, min_periods=1).mean())
    avg_losses = _to_numpy_array(pd.Series(losses).rolling(window=period, min_periods=1).mean())
    
    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_macd_vectorized(
    close_values: np.ndarray, fast_period: int, slow_period: int, signal_period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates MACD using vectorized operations.

    Args:
        close_values: A NumPy array of closing prices.
        fast_period: The fast period for MACD EMA.
        slow_period: The slow period for MACD EMA.
        signal_period: The signal line period for MACD.

    Returns:
        A tuple containing the MACD line and MACD signal line as NumPy arrays.
    """
    close_series = pd.Series(close_values)
    ema_fast = _to_numpy_array(close_series.ewm(span=fast_period, adjust=False).mean())
    ema_slow = _to_numpy_array(close_series.ewm(span=slow_period, adjust=False).mean())
    
    macd_line = ema_fast - ema_slow
    macd_signal = _to_numpy_array(pd.Series(macd_line).ewm(span=signal_period, adjust=False).mean())
    return macd_line, macd_signal

def _calculate_bollinger_bands_vectorized(
    close_values: np.ndarray, window: int, std_multiplier: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates Bollinger Bands using vectorized operations.

    Args:
        close_values: A NumPy array of closing prices.
        window: The moving average window.
        std_multiplier: The standard deviation multiplier.

    Returns:
        A tuple containing the upper and lower Bollinger Bands as NumPy arrays.
    """
    close_series = pd.Series(close_values)
    
    sma = _to_numpy_array(close_series.rolling(window=window, min_periods=1).mean())
    rolling_std = _to_numpy_array(close_series.rolling(window=window, min_periods=1).std())
    
    bb_upper = sma + (rolling_std * std_multiplier)
    bb_lower = sma - (rolling_std * std_multiplier)
    return bb_upper, bb_lower

def _calculate_sma_vectorized(close_values: np.ndarray, period: int) -> np.ndarray:
    """Calculates Simple Moving Average using vectorized operations.

    Args:
        close_values: A NumPy array of closing prices.
        period: The time period for the moving average.

    Returns:
        A NumPy array of the SMA values.
    """
    return _to_numpy_array(pd.Series(close_values).rolling(window=period, min_periods=1).mean())

def _apply_indicator(
    df: pd.DataFrame,
    indicator_name: str,
    ta_function: Callable[..., Optional[pd.DataFrame]],
    fallback_function: Callable[..., Any],
    close_values: np.ndarray,
    params: dict,
    output_cols: List[str]
) -> None:
    """
    Applies a technical indicator calculation with a fallback mechanism.

    Args:
        df: The DataFrame to add the indicator to.
        indicator_name: The name of the indicator for logging.
        ta_function: The primary function from `pandas_ta` to call.
        fallback_function: The vectorized numpy function to use as a fallback.
        close_values: The numpy array of close prices.
        params: A dictionary of parameters for the functions.
        output_cols: A list of column names for the output.
    """
    try:
        output = ta_function(df[COL_CLOSE], **params)
        if output is not None and not output.empty:
            if isinstance(output, pd.DataFrame) and len(output.columns) >= len(output_cols):
                for i, col in enumerate(output_cols):
                    # Find the correct column from pandas_ta output
                    ta_col_name = next((c for c in output.columns if col.split('_')[0].upper() in c), None)
                    if ta_col_name:
                        df[col] = output[ta_col_name]
                    else:
                        raise ValueError(f"Could not find required column for {col} in {indicator_name} output.")
            elif isinstance(output, pd.Series):
                df[output_cols[0]] = output
            else:
                raise ValueError(f"Unexpected output type from {indicator_name}: {type(output)}")
                
            # If all values are NaN, try the fallback
            if all(df[col].isna().all() for col in output_cols):
                raise ValueError(f"{indicator_name} calculation resulted in all NaNs.")
        else:
            raise ValueError(f"{indicator_name} calculation returned None or empty.")
    except Exception as e:
        logger.warning(
            f"{indicator_name} calculation failed with {type(e).__name__}: {e}. "
            f"Using vectorized fallback."
        )
        try:
            fallback_output = fallback_function(close_values, **params)
            if isinstance(fallback_output, tuple):
                for i, col in enumerate(output_cols):
                    df[col] = fallback_output[i]
            else:
                df[output_cols[0]] = fallback_output
        except Exception as fallback_e:
            logger.error(
                f"Vectorized fallback for {indicator_name} also failed: {fallback_e}",
                exc_info=True
            )
            for col in output_cols:
                df[col] = np.nan

def generate_indicator_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators and features for a given OHLCV DataFrame.

    This function enriches the input DataFrame with several technical indicators,
    including RSI, MACD, Bollinger Bands, and SMA. It uses the `pandas_ta` library
    with a robust vectorized NumPy fallback for performance and reliability. It also
    calculates the slope of the moving average and handles any resulting NaN values.

    Args:
        df_input: A pandas DataFrame containing at least a 'close' column with
            OHLCV price data.

    Returns:
        A pandas DataFrame with the added technical indicator columns. Returns an
        empty DataFrame if the input is invalid or an unrecoverable error occurs.
    """
    try:
        if df_input.empty or COL_CLOSE not in df_input.columns:
            logger.warning("Input DataFrame is empty or missing 'close' column.")
            return pd.DataFrame()

        df = df_input.copy()
        close_values = _to_numpy_array(df[COL_CLOSE])

        # RSI
        _apply_indicator(
            df, 'RSI', ta.rsi, _calculate_rsi_vectorized, close_values,
            {'length': RSI_PERIOD}, ['rsi']
        )

        # MACD
        _apply_indicator(
            df, 'MACD', ta.macd, _calculate_macd_vectorized, close_values,
            {'fast': MACD_FAST_PERIOD, 'slow': MACD_SLOW_PERIOD, 'signal': MACD_SIGNAL_PERIOD},
            ['macd', 'macd_signal']
        )

        # Bollinger Bands
        _apply_indicator(
            df, 'Bollinger Bands', ta.bbands, _calculate_bollinger_bands_vectorized, close_values,
            {'length': BB_WINDOW, 'std': BB_STD_MULTIPLIER},
            [COL_BB_UPPER, COL_BB_LOWER]
        )

        # SMA
        _apply_indicator(
            df, 'SMA', ta.sma, _calculate_sma_vectorized, close_values,
            {'length': SMA_PERIOD}, ['ma_20']
        )

        # Calculate slope for the 'ma_20'
        if 'ma_20' in df.columns:
            ma_20_values = _to_numpy_array(df['ma_20'].ffill())
            df['ma_20_slope'] = np.gradient(ma_20_values)

        # Fill NaN values and drop rows if any NaNs still exist
        numeric_cols = [
            'rsi', 'macd', 'macd_signal', COL_BB_UPPER, COL_BB_LOWER, 
            'ma_20', 'ma_20_slope'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].bfill().ffill()

        result_df = df.dropna()
        if result_df.empty and not df.empty:
            logger.warning("DataFrame became empty after dropping NaNs from feature generation.")
        else:
            logger.success(f"Generated features for {len(result_df)} rows.")
            
        return result_df

    except Exception as e:
        logger.error(f"Critical error in generate_indicator_features: {e}", exc_info=True)
        return pd.DataFrame()

