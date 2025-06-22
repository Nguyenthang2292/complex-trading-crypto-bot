#!/usr/bin/env python3
"""
Trading Signal Analyzer (Refactored)
Phân tích tín hiệu giao dịch từ nhiều model ML và đưa ra khuyến nghị.
Sử dụng global models được train từ tất cả symbols và timeframes.

Usage:
    python trading_signal_analyzer.py
"""

import logging
import os
import pandas as pd
import shutil
import sys
from typing import Dict, List, Optional, Tuple, Any, Union

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from components._load_all_symbols_data import load_all_symbols_data
from components.tick_processor import tick_processor
from components._combine_all_dataframes import combine_all_dataframes
from components.config import (
    DEFAULT_TIMEFRAMES, DEFAULT_CRYPTO_SYMBOLS, MODELS_DIR, 
    SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL
)
from signals.signals_best_performance_symbols import signal_best_performance_symbols, get_short_signal_candidates
from signals.signals_random_forest import get_latest_random_forest_signal, load_random_forest_model, train_and_save_global_rf_model
from signals.signals_hmm import hmm_signals
from signals.signals_transformer import get_latest_transformer_signal, load_transformer_model, train_and_save_transformer_model
from signals.signals_cnn_lstm_attention import (
    train_cnn_lstm_attention_model, 
    load_cnn_lstm_attention_model, 
    get_latest_cnn_lstm_attention_signal
)

from utilities._logger import setup_logging
logger = setup_logging(module_name="trading_signal_analyzer", log_level=logging.INFO)

class TradingSignalAnalyzer:
    """Analyzes trading signals from multiple ML models for a given symbol."""
    
    def __init__(self) -> None:
        """Initializes the analyzer, fetching the list of valid symbols."""
        self.valid_timeframes: List[str] = DEFAULT_TIMEFRAMES
        self.processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
        
        try:
            self.valid_symbols: List[str] = self.processor.get_symbols_list_by_quote_usdt()
            logger.config(f"Initialized with {len(self.valid_symbols)} USDT pair symbols from the exchange.")
        except Exception as e:
            logger.warning(f"Could not fetch symbols from exchange: {e}. Using default list.")
            self.valid_symbols = DEFAULT_CRYPTO_SYMBOLS
        
    def validate_symbol(self, symbol: str) -> bool:
        """Checks if a symbol is valid according to the fetched list.

        Args:
            symbol: The symbol to validate (e.g., 'BTC-USDT' or 'BTCUSDT').

        Returns:
            True if the symbol is valid, False otherwise.
        """
        if '-' in symbol:
            base, quote = symbol.split('-')
            symbol = f"{base}{quote}"
        return symbol.upper() in [s.upper() for s in self.valid_symbols]
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalizes a symbol name to the 'BTCUSDT' format.

        Args:
            symbol: The symbol to normalize.

        Returns:
            The normalized symbol string.
        """
        if '-' in symbol:
            base, quote = symbol.split('-')
            return f"{base}{quote}".upper()
        return symbol.upper()
    
    def clear_models_directory(self) -> None:
        """Removes and recreates the models directory."""
        try:
            if MODELS_DIR.exists():
                logger.process(f"Deleting all models in {MODELS_DIR}")
                shutil.rmtree(MODELS_DIR)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            logger.success("Models directory has been cleared and recreated.")
        except IOError as e:
            logger.error(f"Error clearing models directory: {e}")
            raise
    
    def reload_all_models(self) -> None:
        """Fetches all data and retrains all global models."""
        try:
            logger.process("Starting to reload all models...")
            self.clear_models_directory()
            
            logger.data(
                f"Loading data for {len(self.valid_symbols)} symbols and "
                f"{len(DEFAULT_TIMEFRAMES)} timeframes...")
            
            all_symbols_data = load_all_symbols_data(
                processor=self.processor,
                symbols=self.valid_symbols,
                timeframes=DEFAULT_TIMEFRAMES
            )
            
            if not all_symbols_data:
                raise RuntimeError("Failed to load any symbol data.")
            
            logger.data("Combining all dataframes...")
            # Filter out None values and ensure correct data structure before combining
            filtered_data: Dict[str, Dict[str, pd.DataFrame]] = {}
            for symbol, data in all_symbols_data.items():
                if data is not None:
                    if isinstance(data, dict):
                        # Filter out None DataFrames within the inner dictionary
                        filtered_data[symbol] = {
                            tf: df for tf, df in data.items() if isinstance(df, pd.DataFrame)
                        }
                    elif isinstance(data, pd.DataFrame):
                        # If data is a DataFrame, wrap it in a dict with a default timeframe key
                        filtered_data[symbol] = {'default': data}

            combined_df = combine_all_dataframes(filtered_data)

            if combined_df.empty:
                raise RuntimeError("Combined dataframe is empty.")
            
            logger.success(
                f"Successfully combined dataframe with {len(combined_df)} rows."
            )
            self._train_all_models_from_combined_data(combined_df)
            logger.success("Finished reloading all models.")
            
        except Exception as e:
            logger.error(f"Error during model reload: {e}")
            raise
    
    def _train_all_models_from_combined_data(self, combined_df: pd.DataFrame) -> None:
        """Trains all models using a single combined dataframe.

        Args:
            combined_df: A pandas DataFrame containing data from all symbols
                and timeframes.
        """
        logger.model("Training all models from the combined DataFrame...")
        if combined_df.empty:
            logger.error("Combined DataFrame is empty, cannot train models.")
            return

        unique_symbols = combined_df['symbol'].nunique() if 'symbol' in combined_df.columns else 0
        unique_timeframes = combined_df['timeframe'].nunique() if 'timeframe' in combined_df.columns else 0
        logger.data(f"Training data contains {unique_symbols} unique symbols and {unique_timeframes} unique timeframes")

        # --- Train Non-LSTM based models ---
        non_lstm_configs = [
            ("Random Forest", lambda: train_and_save_global_rf_model(combined_df, model_filename="rf_model_global.joblib")),
            ("Transformer", lambda: train_and_save_transformer_model(combined_df, model_filename="transformer_model_global.pth"))
        ]
        for model_name, train_func in non_lstm_configs:
            logger.model(f"Training {model_name} model...")
            try:
                model, model_path = train_func()
                if model:
                    logger.success(f"{model_name} model saved: {model_path}")
                else:
                    logger.warning(f"{model_name} training failed.")
            except Exception as e:
                logger.error(f"Error training {model_name} model: {e}")

        # --- Train all 12 LSTM model variants ---
        lstm_base_configs = [
            ("LSTM", False, False),
            ("LSTM-Attention", False, True),
            ("CNN-LSTM", True, False),
            ("CNN-LSTM-Attention", True, True)
        ]
        output_modes = ['classification', 'regression', 'classification_advanced']

        logger.model("Starting training for 12 LSTM configurations (4 types × 3 output modes)...")
        for base_name, use_cnn, use_attention in lstm_base_configs:
            for output_mode in output_modes:
                model_name = f"{base_name}-{output_mode.replace('_', '-')}"
                filename = f"{base_name.lower().replace('-', '_')}_{output_mode.replace('_', '-')}_model_global.pth"
                logger.model(f"Training {model_name} model...")
                try:
                    model, model_path = train_cnn_lstm_attention_model(
                        combined_df,
                        model_filename=filename,
                        use_cnn=use_cnn,
                        use_attention=use_attention,
                        output_mode=output_mode
                    )
                    if model:
                        logger.success(f"{model_name} model saved: {model_path}")
                    else:
                        logger.warning(f"{model_name} training failed.")
                except Exception as e:
                    logger.error(f"Error training {model_name} model: {e}")
        
        logger.success("Completed training all models from combined data.")
    
    def analyze_best_performance_signals(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Runs the best/worst performance analysis to get candidate symbols.

        Returns:
            A tuple containing a list of LONG candidates and SHORT candidates.
        """
        logger.analysis("Analyzing best/worst performing symbols...")
        symbol_data = load_all_symbols_data(
            processor=self.processor,
            symbols=self.valid_symbols,
            timeframes=DEFAULT_TIMEFRAMES
        )
        if not symbol_data:
            logger.error("No data available to analyze performance.")
            return [], []

        # Filter symbol_data to match expected type structure for preloaded_data
        filtered_symbol_data_perf: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol, data in symbol_data.items():
            if isinstance(data, dict):
                filtered_symbol_data_perf[symbol] = {
                    tf: df for tf, df in data.items() if isinstance(df, pd.DataFrame)
                }

        analysis_result = signal_best_performance_symbols(
            processor=self.processor,
            symbols=self.valid_symbols,
            timeframes=DEFAULT_TIMEFRAMES,
            performance_period=24,
            top_percentage=0.2,
            include_short_signals=True,
            worst_percentage=0.2,
            min_volume_usdt=1000000,
            exclude_stable_coins=True,
            preloaded_data=filtered_symbol_data_perf
        )
        if not analysis_result:
            logger.error("Performance analysis returned no results.")
            return [], []

        long_candidates = analysis_result.get('best_performers', [])
        short_candidates = get_short_signal_candidates(analysis_result, min_short_score=0.6)
        logger.analysis(f"Found {len(long_candidates)} LONG candidates and {len(short_candidates)} SHORT candidates.")
        return long_candidates, short_candidates
    
    def check_symbol_in_performance_list(
        self,
        symbol: str,
        timeframe: str,
        signal: str,
        long_candidates: List[Dict[str, Any]],
        short_candidates: List[Dict[str, Any]]
    ) -> int:
        """Checks if a symbol is in the top/worst performers list.

        Args:
            symbol: The symbol to check.
            timeframe: The timeframe to check.
            signal: The target signal ('LONG' or 'SHORT').
            long_candidates: List of best-performing symbols.
            short_candidates: List of worst-performing symbols.

        Returns:
            1: If the symbol matches the signal type's performance list.
           -1: If the signal is LONG but the symbol is in the SHORT list (conflict).
            0: If there is no match or conflict.
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            if signal.upper() == SIGNAL_LONG:
                for candidate in long_candidates:
                    if (candidate.get('symbol', '').upper() == normalized_symbol and
                        timeframe in candidate.get('timeframe_scores', {})):
                        logger.analysis(f"{symbol} {timeframe} is a top performer for LONG: +1")
                        return 1
                for candidate in short_candidates:
                    if (candidate.get('symbol', '').upper() == normalized_symbol and
                        timeframe in candidate.get('timeframe_scores', {})):
                        logger.analysis(f"{symbol} {timeframe} is a worst performer (conflict for LONG): -1")
                        return -1

            elif signal.upper() == SIGNAL_SHORT:
                for candidate in short_candidates:
                    if (candidate.get('symbol', '').upper() == normalized_symbol and
                        timeframe in candidate.get('timeframe_scores', {})):
                        logger.analysis(f"{symbol} {timeframe} is a worst performer for SHORT: +1")
                        return 1
            
            return 0
            
        except Exception as e:
            logger.error(f"Error checking performance list for {symbol}: {e}")
            return 0
    
    def _get_market_data_for_symbol(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetches market data for a specific symbol and timeframe.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe for the data.

        Returns:
            A pandas DataFrame with the market data, or None if not found.
        """
        try:
            symbol_data = load_all_symbols_data(
                processor=self.processor,
                symbols=[symbol],
                timeframes=[timeframe]
            )
            
            if not symbol_data or symbol not in symbol_data:
                logger.warning(f"No market data found for {symbol} on {timeframe}.")
                return None
            
            symbol_data_item = symbol_data.get(symbol)
            if symbol_data_item is None:
                return None

            if isinstance(symbol_data_item, pd.DataFrame):
                return symbol_data_item

            if isinstance(symbol_data_item, dict):
                timeframe_data = symbol_data_item.get(timeframe)
                if isinstance(timeframe_data, pd.DataFrame):
                    return timeframe_data
            
            return None # Return None if data is not in the expected format
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} {timeframe}: {e}")
            return None
    
    def get_random_forest_signal_score(self, market_data: pd.DataFrame, signal: str) -> int:
        """Gets the signal score from the global Random Forest model."""
        try:
            model_path = MODELS_DIR / "rf_model_global.joblib"
            if not model_path.exists():
                return 0
            model = load_random_forest_model(model_path)
            if not model:
                return 0
            predicted_signal = get_latest_random_forest_signal(market_data, model)
            return self._calculate_signal_match_score(predicted_signal, signal, "Random Forest")
        except Exception as e:
            logger.error(f"Error getting Random Forest signal: {e}")
            return 0
    
    def get_hmm_signal_score(self, market_data: pd.DataFrame, signal: str) -> int:
        """Gets the signal score from the HMM model (strict and non-strict)."""
        try:
            hmm_signals_result = hmm_signals(market_data)
            if len(hmm_signals_result) < 2:
                return 0
            strict_sig, non_strict_sig = hmm_signals_result[:2]
            signal_map = {1: SIGNAL_LONG, -1: SIGNAL_SHORT, 0: SIGNAL_NEUTRAL}
            strict_str = signal_map.get(strict_sig, SIGNAL_NEUTRAL)
            non_strict_str = signal_map.get(non_strict_sig, SIGNAL_NEUTRAL)
            return (self._calculate_signal_match_score(strict_str, signal, "HMM Strict") +
                    self._calculate_signal_match_score(non_strict_str, signal, "HMM Non-strict"))
        except Exception as e:
            logger.error(f"Error getting HMM signal: {e}")
            return 0
    
    def get_transformer_signal_score(self, market_data: pd.DataFrame, signal: str) -> int:
        """Gets the signal score from the global Transformer model."""
        try:
            model_path = MODELS_DIR / "transformer_model_global.pth"
            if not model_path.exists():
                return 0
            model_data = load_transformer_model(str(model_path))
            if not model_data or not model_data[0]:
                return 0
            
            model, scaler, feature_cols, target_idx = model_data
            if model is None or scaler is None or feature_cols is None or target_idx is None:
                return 0
            
            predicted_signal = get_latest_transformer_signal(market_data, model, scaler, feature_cols, target_idx)
            return self._calculate_signal_match_score(predicted_signal, signal, "Transformer")
        except Exception as e:
            logger.error(f"Error getting Transformer signal: {e}")
            return 0
    
    def get_lstm_signal_score(self, market_data: pd.DataFrame, signal: str) -> int:
        """Gets the combined signal score from all 12 LSTM model variants."""
        total_score = 0
        successful_predictions = 0
        base_configs = [
            ("LSTM", False, False), ("LSTM-Attention", False, True),
            ("CNN-LSTM", True, False), ("CNN-LSTM-Attention", True, True)
        ]
        output_modes = ['classification', 'regression', 'classification_advanced']
        
        for base_name, use_cnn, use_attention in base_configs:
            for output_mode in output_modes:
                variant_name = f"{base_name}-{output_mode.replace('_', '-')}"
                filename = f"{base_name.lower().replace('-', '_')}_{output_mode.replace('_', '-')}_model_global.pth"
                model_path = MODELS_DIR / filename
                
                try:
                    if not model_path.exists():
                        continue
                    loaded_data = load_cnn_lstm_attention_model(model_path)
                    if not loaded_data:
                        continue
                    
                    model, config, info, opt_res = loaded_data
                    predicted = get_latest_cnn_lstm_attention_signal(
                        df_input=market_data, model=model, model_config=config,
                        data_info=info, optimization_results=opt_res
                    )
                    total_score += self._calculate_signal_match_score(predicted, signal, variant_name)
                    successful_predictions += 1
                except Exception as e:
                    logger.debug(f"Error with LSTM variant {variant_name}: {e}")
                    continue
        
        if successful_predictions > 0:
            logger.signal(f"LSTM Combined ({successful_predictions}/12 variants) total score: {total_score}")
        
        return total_score
    
    def _calculate_signal_match_score(self, predicted_signal: str, target_signal: str, model_name: str) -> int:
        """Calculates a score based on a signal match.

        Args:
            predicted_signal: The signal predicted by the model.
            target_signal: The signal we are analyzing for ('LONG' or 'SHORT').
            model_name: The name of the model for logging.

        Returns:
            1 if signals match, -1 if they conflict, 0 if neutral or no signal.
        """
        if not predicted_signal or predicted_signal == SIGNAL_NEUTRAL:
            return 0
        if predicted_signal.upper() == target_signal.upper():
            return 1
        return -1
    
    def calculate_final_threshold(self, total_score: int, max_possible_score: int) -> float:
        """Calculates the final confidence threshold from the total score.

        The score is normalized to a range of [0, 1], where 0 represents the
        maximum possible negative score and 1 represents the maximum possible
        positive score.

        Args:
            total_score: The sum of scores from all models.
            max_possible_score: The maximum possible positive score.

        Returns:
            A normalized confidence score between 0.0 and 1.0.
        """
        if max_possible_score == 0:
            return 0.0
        
        # Normalize score from [-max, +max] to [0, 1]
        normalized_score = (total_score + max_possible_score) / (2 * max_possible_score)
        return max(0.0, min(1.0, normalized_score))
    
    def _calculate_max_possible_score(self) -> int:
        """Calculates the maximum possible positive score across all models.

        Returns:
            The total maximum possible score.
        """
        # Defines the maximum positive score each model component can contribute.
        model_max_scores = {
            'performance': 1,      # Best/worst performance analysis: +1
            'random_forest': 1,    # Random Forest: +1
            'hmm': 2,              # HMM: +2 (strict +1, non-strict +1)
            'transformer': 1,      # Transformer: +1
            'lstm': 12             # LSTM: +12 (12 variants × +1 each)
        }
        return sum(model_max_scores.values())
    
    def analyze_symbol_signal(
        self,
        symbol: str,
        timeframe: str,
        signal: str,
        long_candidates: List[Dict[str, Any]],
        short_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Union[str, int, float, Dict[str, int]]]:
        """Performs a comprehensive signal analysis for a single symbol.

        This method aggregates scores from all available models to generate a
        final score and a trading recommendation.

        Args:
            symbol: The trading symbol.
            timeframe: The data timeframe.
            signal: The target signal ('LONG' or 'SHORT').
            long_candidates: List of best-performing symbols.
            short_candidates: List of worst-performing symbols.

        Returns:
            A dictionary containing the detailed analysis results.
        """
        logger.analysis(f"Starting comprehensive analysis for {symbol} {timeframe} {signal}...")
        
        scores: Dict[str, int] = {}
        
        # 1. Performance list check
        scores['performance'] = self.check_symbol_in_performance_list(
            symbol, timeframe, signal, long_candidates, short_candidates
        )
        
        # 2. Model-based checks
        market_data = self._get_market_data_for_symbol(symbol, timeframe)
        if market_data is None or market_data.empty:
            logger.warning(f"No market data for {symbol} {timeframe}, skipping model checks.")
            scores.update({'random_forest': 0, 'hmm': 0, 'transformer': 0, 'lstm': 0})
        else:
            scores['random_forest'] = self.get_random_forest_signal_score(market_data, signal)
            scores['hmm'] = self.get_hmm_signal_score(market_data, signal)
            scores['transformer'] = self.get_transformer_signal_score(market_data, signal)
            scores['lstm'] = self.get_lstm_signal_score(market_data, signal)

        total_score = sum(scores.values())
        max_possible_score = self._calculate_max_possible_score()
        threshold = self.calculate_final_threshold(total_score, max_possible_score)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal,
            'scores': scores,
            'total_score': total_score,
            'max_possible_score': max_possible_score,
            'threshold': threshold,
            'recommendation': 'ENTER' if threshold >= 0.7 else 'WAIT'
        }
    
    def run_analysis(
        self,
        reload_model: bool,
        symbol: str,
        timeframe: str,
        signal: str
    ) -> Dict[str, Any]:
        """Main entry point to run a full analysis for a given request.

        Args:
            reload_model: If True, retrains all models before analysis.
            symbol: The trading symbol to analyze.
            timeframe: The timeframe to analyze.
            signal: The target signal to check for ('LONG' or 'SHORT').

        Returns:
            A dictionary containing the analysis results or an error message.
        """
        try:
            if not self.validate_symbol(symbol):
                return {'error': f"Invalid symbol: '{symbol}'"}
            if timeframe not in self.valid_timeframes:
                return {'error': f"Invalid timeframe: '{timeframe}'"}
            if signal.upper() not in [SIGNAL_LONG, SIGNAL_SHORT]:
                return {'error': f"Invalid signal: '{signal}'"}

            if reload_model:
                logger.process("Model reload requested, starting the process...")
                self.reload_all_models()
            
            long_candidates, short_candidates = self.analyze_best_performance_signals()
            
            result = self.analyze_symbol_signal(
                symbol, timeframe, signal, long_candidates, short_candidates
            )
            
            return result
            
        except Exception as e:
            logger.error(f"A critical error occurred during analysis: {e}")
            return {'error': str(e)}


def print_analysis_result(result: Dict[str, Any]) -> None:
    """Prints the analysis result in a formatted, user-friendly way.

    Args:
        result: The result dictionary from the `run_analysis` method.
    """
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return

    print("\n" + "="*60)
    print("🔍 TRADING SIGNAL ANALYSIS RESULT")
    print("="*60)
    
    print(f"📊 Symbol:    {result['symbol']}")
    print(f"⏰ Timeframe: {result['timeframe']}")
    print(f"📈 Signal:    {result['signal']}")
    
    print("\n📋 SCORE DETAILS:")
    scores = result['scores']
    print(f"  • Performance List:  {scores['performance']:+3d}")
    print(f"  • Random Forest:     {scores['random_forest']:+3d}")
    print(f"  • HMM (Combined):    {scores['hmm']:+3d}")
    print(f"  • Transformer:       {scores['transformer']:+3d}")
    print(f"  • LSTM (12 models):  {scores['lstm']:+3d}")
    
    print("\n🎯 FINAL ASSESSMENT:")
    print(
        f"  • Total Score:         {result['total_score']:+3d} / {result['max_possible_score']}"
    )
    print(f"  • Confidence Score:    {result['threshold']:.3f}")
    
    recommendation = result['recommendation']
    if recommendation == 'ENTER':
        print(f"  • Recommendation:      ✅ {recommendation} (Confidence ≥ 0.7)")
    else:
        print(f"  • Recommendation:      ⏳ {recommendation} (Confidence < 0.7)")
    
    print("="*60)


def main() -> None:
    """Main function to run the interactive command-line interface."""
    analyzer = TradingSignalAnalyzer()

    print("="*60)
    print("Trading Signal Analyzer (Global Models)")
    print("="*60)

    while True:
        reload_input = input("❓ Do you want to reload all models? (yes/no): ").lower()
        if reload_input in ['yes', 'y', 'no', 'n']:
            reload_model = reload_input in ['yes', 'y']
            break
        print("   ❌ Invalid input. Please enter 'yes' or 'no'.")

    while True:
        symbol_input = input("❓ Enter symbol to check (e.g., BTC-USDT): ").upper()
        if analyzer.validate_symbol(symbol_input):
            break
        print(f"   ❌ Invalid symbol '{symbol_input}'. Please try again.")

    while True:
        timeframe_input = input(f"❓ Enter timeframe ({', '.join(analyzer.valid_timeframes)}): ").lower()
        if timeframe_input in analyzer.valid_timeframes:
            break
        print("   ❌ Invalid timeframe. Please choose from the list.")

    while True:
        signal_input = input("❓ Enter signal to check (LONG/SHORT): ").upper()
        if signal_input in [SIGNAL_LONG, SIGNAL_SHORT]:
            break
        print("   ❌ Invalid input. Please enter 'LONG' or 'SHORT'.")

    print("="*60)
    logger.config("Starting analysis with parameters:")
    logger.config(f"  - Reload Models: {reload_model}")
    logger.config(f"  - Symbol:        {symbol_input}")
    logger.config(f"  - Timeframe:     {timeframe_input}")
    logger.config(f"  - Signal:        {signal_input}")

    result = analyzer.run_analysis(
        reload_model=reload_model,
        symbol=symbol_input,
        timeframe=timeframe_input,
        signal=signal_input
    )

    print_analysis_result(result)

if __name__ == "__main__":
    main()