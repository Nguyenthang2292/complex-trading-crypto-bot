import logging
import numpy as np
import os
import pandas as pd
import pandas_ta as ta
import sys

# Setup module paths
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.dirname(current_dir)
signals_dir = os.path.dirname(components_dir)
if signals_dir not in sys.path:
    sys.path.insert(0, signals_dir)

from utilities._logger import setup_logging
# Initialize logger for feature calculation
logger = setup_logging(module_name="_generate_indicator_features", log_level=logging.DEBUG)

from livetrade.config import (
    BB_STD_MULTIPLIER, BB_WINDOW, 
    COL_BB_LOWER, COL_BB_UPPER, COL_CLOSE,
    MACD_FAST_PERIOD, MACD_SIGNAL_PERIOD, MACD_SLOW_PERIOD, 
    RSI_PERIOD, SMA_PERIOD,
)

def _generate_indicator_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for market data analysis with fallback calculations.
    
    Args:
        df_input: DataFrame with OHLCV price data (lowercase column names)
        
    Returns:
        DataFrame with added technical indicators or empty DataFrame on error
    """
    try:
        # Validate input data
        if df_input.empty or COL_CLOSE not in df_input.columns:
            logger.warning("Input DataFrame is empty or missing close column")
            return pd.DataFrame()
        
        df = df_input.copy()
        
        # Calculate RSI (Relative Strength Index) with fallback
        try:
            df['rsi'] = ta.rsi(df[COL_CLOSE], length=RSI_PERIOD)
            if df['rsi'].isna().all():
                # Manual RSI calculation fallback
                delta = df[COL_CLOSE].diff()
                delta = pd.to_numeric(delta, errors='coerce')
                gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}. Using manual calculation.")
            delta = df[COL_CLOSE].diff()
            delta = pd.to_numeric(delta, errors='coerce')
            gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
        # Calculate MACD (Moving Average Convergence Divergence) with fallback
        try:
            macd_output = ta.macd(df[COL_CLOSE], fast=MACD_FAST_PERIOD, slow=MACD_SLOW_PERIOD, signal=MACD_SIGNAL_PERIOD)
            if macd_output is not None and len(macd_output.columns) >= 2:
                df['macd'] = macd_output.iloc[:, 0]  # MACD line
                df['macd_signal'] = macd_output.iloc[:, 1]  # Signal line
            else:
                # Manual MACD calculation fallback
                ema_fast = df[COL_CLOSE].ewm(span=MACD_FAST_PERIOD).mean()
                ema_slow = df[COL_CLOSE].ewm(span=MACD_SLOW_PERIOD).mean()
                df['macd'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL_PERIOD).mean()
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}. Using manual calculation.")
            ema_fast = df[COL_CLOSE].ewm(span=MACD_FAST_PERIOD).mean()
            ema_slow = df[COL_CLOSE].ewm(span=MACD_SLOW_PERIOD).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL_PERIOD).mean()
            
        # Calculate Bollinger Bands with fallback
        try:
            bb = ta.bbands(df[COL_CLOSE], length=BB_WINDOW, std=BB_STD_MULTIPLIER)
            if bb is not None and not bb.empty:
                bb_cols = bb.columns.tolist()
                upper_col = [col for col in bb_cols if 'BBU' in col]
                lower_col = [col for col in bb_cols if 'BBL' in col]
                
                if upper_col and lower_col:
                    df[COL_BB_UPPER] = bb[upper_col[0]]
                    df[COL_BB_LOWER] = bb[lower_col[0]]
                else:
                    # Manual Bollinger Bands calculation
                    sma = df[COL_CLOSE].rolling(window=BB_WINDOW).mean()
                    std = df[COL_CLOSE].rolling(window=BB_WINDOW).std()
                    df[COL_BB_UPPER] = sma + (std * BB_STD_MULTIPLIER)
                    df[COL_BB_LOWER] = sma - (std * BB_STD_MULTIPLIER)
            else:
                # Manual Bollinger Bands calculation fallback
                sma = df[COL_CLOSE].rolling(window=BB_WINDOW).mean()
                std = df[COL_CLOSE].rolling(window=BB_WINDOW).std()
                df[COL_BB_UPPER] = sma + (std * BB_STD_MULTIPLIER)
                df[COL_BB_LOWER] = sma - (std * BB_STD_MULTIPLIER)
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}. Using manual calculation.")
            sma = df[COL_CLOSE].rolling(window=BB_WINDOW, min_periods=1).mean()
            std = df[COL_CLOSE].rolling(window=BB_WINDOW, min_periods=1).std()
            df[COL_BB_UPPER] = sma + (std * BB_STD_MULTIPLIER)
            df[COL_BB_LOWER] = sma - (std * BB_STD_MULTIPLIER)
            
        # Calculate Simple Moving Average with fallback
        try:
            df['ma_20'] = ta.sma(df[COL_CLOSE], length=SMA_PERIOD)
            if df['ma_20'].isna().all():
                df['ma_20'] = df[COL_CLOSE].rolling(window=SMA_PERIOD).mean()
        except Exception as e:
            logger.warning(f"SMA calculation failed: {e}. Using manual calculation.")
            df['ma_20'] = df[COL_CLOSE].rolling(window=SMA_PERIOD).mean()

        # Calculate MA slope
        df['ma_20_slope'] = df['ma_20'].diff()

        # Fill NaNs from indicator calculations
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        
        # Remove rows with remaining NaN values
        result_df = df.dropna()
        if result_df.empty:
            logger.warning("All features resulted in NaN - insufficient data for technical indicators")
        else:
            logger.info(f"Technical indicators calculated successfully for {len(result_df)} rows")
    
        return result_df
        
    except Exception as e:
        logger.error(f"Error in feature calculation: {e}")
        return pd.DataFrame()