import logging
import os
import sys
from typing import Optional, Tuple 
import pandas as pd
import pandas_ta as ta

# Ensure the project root (the parent of 'livetrade') is in sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
signals_dir = os.path.dirname(current_dir)            
project_root = os.path.dirname(signals_dir)          
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from signals.signals_hmm import hmm_signals
from data_class.__class__OptimizingParametersHMM import OptimizingParametersHMM

from utilities._logger import setup_logging
logger = setup_logging(module_name="generate_signals_hmm", log_level=logging.INFO)
    
def generate_signal_hmm(pair: str, df: pd.DataFrame, strict_mode: bool = False) -> Tuple[str, Optional[int], Optional[float]]:
    """
    Generates trading signals, TP, and SL for a pair using RSI and HMM.

    Combines RSI and HMM signals. If both agree (LONG/SHORT), calculates
    TP/SL using ATR and swing points.

    Args:
        pair (str): Trading pair identifier.
        df (pd.DataFrame): Price data ('High', 'Low', 'Close').
        strict_mode (bool): Controls HMM parameters. Defaults to False.

    Returns:
        Tuple[str, Optional[int], Optional[float], Optional[float], Optional[float]]:
        (pair, signal, entry_price, take_profit, stop_loss).
        Signal: 1=LONG, -1=SHORT, None=No signal.
        Returns (pair, None, None) on errors.
    """
    
    try:
        # Create OptimizingParameters with specified strict_mode
        params = OptimizingParametersHMM()
        params.strict_mode = strict_mode
        
        # Calculate RSI using pandas_ta with default 14 period
        if 'Close' not in df.columns:
            logger.data(f"{pair}: Missing required 'Close' column for RSI calculation")
            return pair, None, None
            
        # Calculate RSI with appropriate error handling
        try:
            df['rsi'] = ta.rsi(df['Close'], length=14)
            
            # Verify RSI calculation succeeded
            if df['rsi'].isna().all() or len(df['rsi'].dropna()) < 14:
                logger.data(f"{pair}: RSI calculation failed or insufficient data")
                return pair, None, None
                
            # Get the latest RSI value
            latest_rsi = df['rsi'].iloc[-1]
            
            # Determine RSI signal with clear thresholds
            rsi_signal = None
            if latest_rsi > 60:  
                rsi_signal = 1  # LONG signal when RSI > 60
            elif latest_rsi < 40: 
                rsi_signal = -1  # SHORT signal when RSI < 40
                
            # Log RSI value for debugging
            logger.analysis(f"{pair}: RSI = {latest_rsi:.2f}, Signal: {rsi_signal}")
            
        except Exception as e:
            logger.error(f"{pair}: Error calculating RSI: {str(e)}")
            return pair, None, None 
        
        # Use 100% of the data for HMM analysis
        data_length = len(df)
        if data_length < 50:  # Minimum required data points
            logger.data(f"{pair}: Insufficient data for HMM analysis ({data_length} points)")
            return pair, None, None 
            
        # Use the complete dataframe instead of just the last 60%
        df_last_portion = df
        current_close = df['Close'].iloc[-1]

        # Get signals from both HMM models using only the last 60% of data
        try:
            high_order_signal, hmm_kama_signal = hmm_signals(df_last_portion, optimizing_params=params)
            
            # Check if HMM signals are valid
            if high_order_signal is None or hmm_kama_signal is None:
                logger.model(f"{pair}: Invalid HMM signals")
                return pair, None, current_close
                
            # Determine combined HMM signal (EITHER model gives a signal)
            hmm_signal = 0  # Neutral by default
            if high_order_signal == 1 or hmm_kama_signal == 1:
                hmm_signal = 1  # LONG signal
            elif high_order_signal == -1 or hmm_kama_signal == -1:
                hmm_signal = -1  # SHORT signal
                
            logger.model(f"{pair}: HMM signals - High Order: {high_order_signal}, KAMA: {hmm_kama_signal}, Combined: {hmm_signal}")
            
        except Exception as e:
            logger.error(f"{pair}: Error in HMM analysis: {str(e)}")
            return pair, None, current_close
        
        # Generate a trading signal
        final_signal = None
        score = 0
        if hmm_signal == 1: score += 2
        elif hmm_signal == -1: score -= 2
        if rsi_signal == 1: score += 1  
        elif rsi_signal == -1: score -= 1
        
        if score >= 2:
            final_signal = 1
        elif score <= -2:
            final_signal = -1
        
        return pair, final_signal, current_close, 
        
    except Exception as e:
        logger.error(f"Error analyzing {pair}: {str(e)}")
        return pair, None, None
