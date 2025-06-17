import logging
import os
import argparse
import time
import sys
from typing import Literal, Optional, Dict
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_class.__class__OptimizingParametersHMM import OptimizingParametersHMM
from livetrade._components._tick_processor import tick_processor
from livetrade.config import (
    SIGNAL_LONG_HMM as LONG,
    SIGNAL_HOLD_HMM as HOLD,
    SIGNAL_SHORT_HMM as SHORT,
    DATA_PROCESSING_WAIT_TIME_IN_SECONDS
)
from signals.signals_hmm import hmm_signals
from utilities._logger import setup_logging

logger = setup_logging(module_name="signals_hmm__main__", log_level=logging.INFO)
Signal = Literal[-1, 0, 1]

def main() -> None:
    """
    Test HMM signals on crypto data.
    
    Process flow:
    1. Initialize Binance tick processor
    2. Load historical market data
    3. Generate HMM signals from high-order HMM and HMM-KAMA models
    4. Display results and trading recommendations
    """
    start_time: float = time.time()
    logger.info("Starting HMM signal testing for BTC-USDT")
    
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='HMM signal analysis for crypto trading')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Crypto symbol to analyze (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe for analysis (default: 1h)')
    args: argparse.Namespace = parser.parse_args()

    logger._logger.setLevel(getattr(logging, args.log_level))
    
    try:
        processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
        logger.network(f"Successfully initialized Crypto TickProcessor with Binance API")
        if processor is None:
            logger.error("Failed to initialize tick processor instance")
            return
    except Exception as e:
        logger.error(f"Failed to initialize Binance tick processor: {e}")
        return
    
    symbol: str = args.symbol
    timeframe: str = args.timeframe
    logger.info(f"Loading historical data for {symbol} on {timeframe} timeframe...")
    
    try:
        processor.get_historic_data_by_symbol(symbol, timeframe)
        time.sleep(DATA_PROCESSING_WAIT_TIME_IN_SECONDS)
        
        df: Optional[pd.DataFrame] = processor.df_cache.get((symbol, timeframe))
        
        if df is None:
            logger.error(f"Failed to retrieve data for {symbol} {timeframe}")
            return
            
        logger.data(f"Successfully loaded {len(df)} candles for {symbol} {timeframe}")
        logger.data(f"Data range: {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return
    
    optimizing_params: OptimizingParametersHMM = OptimizingParametersHMM()
    
    logger.info("=" * 80)
    logger.model("HMM SIGNAL GENERATION".center(80))
    logger.info("=" * 80)
    
    try:
        signal_start_time: float = time.time()
        high_order_signal, hmm_kama_signal = hmm_signals(df, optimizing_params)
        signal_elapsed: float = time.time() - signal_start_time
        
        signal_map: Dict[Signal, str] = {LONG: "LONG", HOLD: "HOLD", SHORT: "SHORT"}
        
        logger.info("\n📊 HMM SIGNAL RESULTS:")
        logger.signal(f"- High-Order HMM Signal: {signal_map[high_order_signal]}")
        logger.signal(f"- HMM-KAMA Signal: {signal_map[hmm_kama_signal]}")
        logger.performance(f"- Signal generation time: {signal_elapsed:.3f} seconds")
        
        latest_candle: pd.Series = df.iloc[-1]
        logger.info(f"\n📈 LATEST MARKET DATA ({symbol} {timeframe}):")
        logger.data(f"- Timestamp: {df.index[-1]}")
        logger.data(f"- Open: ${latest_candle['open']:.2f}")
        logger.data(f"- High: ${latest_candle['high']:.2f}")
        logger.data(f"- Low: ${latest_candle['low']:.2f}")
        logger.data(f"- Close: ${latest_candle['close']:.2f}")
        logger.data(f"- Volume: {latest_candle['volume']:,.0f}")
        
        logger.info(f"\n🎯 TRADING RECOMMENDATION:")
        if high_order_signal == hmm_kama_signal:
            if high_order_signal == LONG:
                logger.success("✅ STRONG BUY - Both models agree on LONG signal")
            elif high_order_signal == SHORT:
                logger.warning("❌ STRONG SELL - Both models agree on SHORT signal")
            else:
                logger.info("⏸️ HOLD - Both models suggest no action")
        else:
            logger.info("⚠️ MIXED SIGNALS - Models disagree, consider waiting for confirmation")
            logger.info(f"   High-Order HMM: {signal_map[high_order_signal]}")
            logger.info(f"   HMM-KAMA: {signal_map[hmm_kama_signal]}")
        
    except Exception as e:
        logger.error(f"Error generating HMM signals: {e}", exc_info=True)
    
    logger.info("=" * 80)
    logger.model("END OF HMM SIGNAL ANALYSIS".center(80))
    logger.info("=" * 80)
    
    try:
        processor.stop()
        logger.success("Tick processor stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping tick processor: {e}")
    
    elapsed_time: float = time.time() - start_time
    logger.performance(f"HMM signal analysis completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()



