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
import shutil
import sys
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd

from components._load_all_symbols_data import load_all_symbols_data
from components.tick_processor import TickProcessor
from components._combine_all_dataframes import combine_all_dataframes
from components.config import (
    DEFAULT_TIMEFRAMES, 
    DEFAULT_CRYPTO_SYMBOLS, 
    MODELS_DIR,
    SIGNAL_LONG, 
    SIGNAL_SHORT, 
    SIGNAL_NEUTRAL
)
from signals.best_performance_pairs.signals_best_performance_symbols import (
    signal_best_performance_symbols, 
    get_short_signal_candidates
)
from signals.quant_models.random_forest.signals_random_forest import (
    get_latest_random_forest_signal, 
    load_random_forest_model,
    train_and_save_global_rf_model
)
from signals.quant_models.hmm.signals_hmm import hmm_signals
from signals.signals_transformer import (
    get_latest_transformer_signal, 
    load_transformer_model,
    train_and_save_transformer_model
)
from signals.signals_cnn_lstm_attention import (
    train_cnn_lstm_attention_model,
    load_cnn_lstm_attention_model,
    get_latest_cnn_lstm_attention_signal
)
from utilities._logger import setup_logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

logger = setup_logging(module_name="trading_signal_analyzer", log_level=logging.INFO)

class TradingSignalAnalyzer:
    """Analyzes trading signals from multiple ML models for a given symbol."""

    def __init__(self) -> None:
        """Initializes the analyzer, fetching the list of valid symbols."""
        self.valid_timeframes: List[str] = DEFAULT_TIMEFRAMES
        self.processor = TickProcessor(trade_open_callback=None, trade_close_callback=None)

        try:
            self.valid_symbols: List[str] = self.processor.get_symbols_list_by_quote_usdt()
            logger.config(
                f"Initialized with {len(self.valid_symbols)} USDT "
                "pair symbols from the exchange."
            )
        except (ConnectionError, ValueError, KeyError) as e:
            logger.warning(
                f"Could not fetch symbols from exchange: {e}. Using default list."
            )
            self.valid_symbols = DEFAULT_CRYPTO_SYMBOLS

    def validate_symbol(self, symbol: str) -> bool:
        """Checks if a symbol is valid according to the fetched list.

        Args:
            symbol: The symbol to validate (e.g., 'BTC-USDT' or 'BTCUSDT').

        Returns:
            True if the symbol is valid, False otherwise.
        """
        if '-' in symbol:
            parts = symbol.split('-')
            if len(parts) != 2:
                logger.warning(f"Invalid symbol format received: {symbol}")
                return False
            base, quote = parts
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
        except OSError as e:
            logger.error(f"Error clearing models directory: {e}")
            raise

    def reload_all_models(self) -> None:
        """Fetches all data, retrains, and saves all global models."""
        try:
            logger.process("Starting to reload all models...")
            self.clear_models_directory()

            logger.data(
                f"Loading data for {len(self.valid_symbols)} symbols and "
                f"{len(DEFAULT_TIMEFRAMES)} timeframes..."
            )
            # Load data for all symbols and timeframes
            all_symbols_data = load_all_symbols_data(
                processor=self.processor,
                symbols=self.valid_symbols,
                timeframes=DEFAULT_TIMEFRAMES,
            )
            if not all_symbols_data:
                raise RuntimeError("Failed to load any symbol data.")

            logger.data("Combining all dataframes...")
            combined_df = self._prepare_combined_dataframe(all_symbols_data)
            if combined_df.empty:
                raise RuntimeError("Combined dataframe is empty, no data to train.")

            logger.success(
                f"Successfully combined dataframe with {len(combined_df)} rows."
            )
            self._train_all_models_from_combined_data(combined_df)
            logger.success("Finished reloading all models.")

        except (RuntimeError, IOError, OSError) as e:
            logger.error(f"Error during model reload: {e}")
            raise

    def _prepare_combined_dataframe(self, all_symbols_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare combined dataframe from symbol data.

        Args:
            all_symbols_data: Raw symbol data dictionary.

        Returns:
            Combined pandas DataFrame.
        """
        filtered_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol, data in all_symbols_data.items():
            if data is not None:
                if isinstance(data, dict):
                    filtered_data[symbol] = {
                        tf: df for tf, df in data.items() if isinstance(df, pd.DataFrame)
                    }
                elif isinstance(data, pd.DataFrame):
                    filtered_data[symbol] = {'default': data}

        return combine_all_dataframes(filtered_data)

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
        logger.data(
            f"Training data contains {unique_symbols} unique symbols and "
            f"{unique_timeframes} unique timeframes"
        )

        self._train_non_lstm_models(combined_df)
        self._train_lstm_models(combined_df)

        logger.success("Completed training all models from combined data.")

    def _train_non_lstm_models(self, combined_df: pd.DataFrame) -> None:
        """Train non-LSTM based models.

        Args:
            combined_df: Combined training data.
        """
        non_lstm_configs = [
            ("Random Forest", lambda: train_and_save_global_rf_model(
                combined_df, model_filename="rf_model_global.joblib"
            )),
            ("Transformer", lambda: train_and_save_transformer_model(
                combined_df, model_filename="transformer_model_global.pth"
            )),
        ]
        for model_name, train_func in non_lstm_configs:
            logger.model(f"Training {model_name} model...")
            try:
                model, model_path = train_func()
                if model and model_path:
                    logger.success(f"{model_name} model saved: {model_path}")
                else:
                    logger.warning(f"{model_name} training failed or model not saved.")
            except (ValueError, TypeError, IOError) as e:
                logger.error(f"Error training {model_name} model: {e}")

    def _train_lstm_models(self, combined_df: pd.DataFrame) -> None:
        """Train all 12 LSTM model variants based on the combined data.

        Args:
            combined_df: Combined training data.
        """
        lstm_base_configs = [
            ("LSTM", False, False),
            ("LSTM-Attention", False, True),
            ("CNN-LSTM", True, False),
            ("CNN-LSTM-Attention", True, True),
        ]
        output_modes = ['classification', 'regression', 'classification_advanced']

        logger.model(
            "Starting training for 12 LSTM configurations "
            "(4 types × 3 output modes)..."
        )
        for base_name, use_cnn, use_attention in lstm_base_configs:
            for output_mode in output_modes:
                self._train_single_lstm_model(
                    combined_df, base_name, use_cnn, use_attention, output_mode
                )

    def _train_single_lstm_model(
        self,
        combined_df: pd.DataFrame,
        base_name: str,
        use_cnn: bool,
        use_attention: bool,
        output_mode: str,
    ) -> None:
        """Train a single LSTM model variant.

        Args:
            combined_df: Combined training data.
            base_name: Base model name.
            use_cnn: Whether to use CNN.
            use_attention: Whether to use attention.
            output_mode: Output mode for the model.
        """
        model_name = f"{base_name}-{output_mode.replace('_', '-')}"
        filename = (
            f"{base_name.lower().replace('-', '_')}_{output_mode}_model_global.pth"
        )
        logger.model(f"Training {model_name} model...")
        try:
            model, model_path = train_cnn_lstm_attention_model(
                combined_df,
                model_filename=filename,
                use_cnn=use_cnn,
                use_attention=use_attention,
                output_mode=output_mode,
            )
            if model:
                logger.success(f"{model_name} model saved: {model_path}")
            else:
                logger.warning(f"{model_name} training failed.")
        except (ValueError, TypeError, IOError) as e:
            logger.error(f"Error training {model_name} model: {e}")

    def analyze_best_performance_signals(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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

        filtered_symbol_data_perf = self._filter_symbol_data(symbol_data)

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
        logger.analysis(
            f"Found {len(long_candidates)} LONG candidates and "
            f"{len(short_candidates)} SHORT candidates."
        )
        return long_candidates, short_candidates

    def _filter_symbol_data(self, symbol_data: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Filter symbol data to match expected type structure.

        Args:
            symbol_data: Raw symbol data.

        Returns:
            Filtered symbol data.
        """
        filtered_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol, data in symbol_data.items():
            if isinstance(data, dict):
                filtered_data[symbol] = {
                    tf: df for tf, df in data.items() if isinstance(df, pd.DataFrame)
                }
        return filtered_data

    def check_symbol_in_performance_list(
        self, symbol: str, timeframe: str, signal: str,
        performance_candidates: Dict[str, List[Dict[str, Any]]]
    ) -> int:
        """Checks if a symbol/timeframe is in the top/bottom performers.

        Args:
            symbol: The symbol to check.
            timeframe: The timeframe to check.
            signal: The signal to check ('LONG' or 'SHORT').
            performance_candidates: Dict containing 'long' and 'short' lists.

        Returns:
            1 if it's a match, -1 if it's a conflict, 0 otherwise.
        """
        try:
            norm_symbol = self.normalize_symbol(symbol)
            long_candidates = performance_candidates.get('long', [])
            short_candidates = performance_candidates.get('short', [])

            if signal == SIGNAL_LONG:
                if any(c['symbol'] == norm_symbol and timeframe in c.get('timeframe_scores', {}) for c in long_candidates):
                    return 1
                if any(c['symbol'] == norm_symbol and timeframe in c.get('timeframe_scores', {}) for c in short_candidates):
                    return -1
            elif signal == SIGNAL_SHORT:
                if any(c['symbol'] == norm_symbol and timeframe in c.get('timeframe_scores', {}) for c in short_candidates):
                    return 1
                if any(c['symbol'] == norm_symbol and timeframe in c.get('timeframe_scores', {}) for c in long_candidates):
                    return -1
        except (KeyError, TypeError) as e:
            logger.error(
                f"Error processing performance candidates for {symbol}/{timeframe}: {e}"
            )
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

            return None

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
            predicted_signal, _ = get_latest_random_forest_signal(market_data, model)
            return self._calculate_signal_match_score(predicted_signal, signal)
        except Exception as e:
            logger.error(f"Error getting Random Forest signal: {e}")
            return 0

    def get_hmm_signal_score(self, market_data: pd.DataFrame, signal: str) -> int:
        """Gets the signal score from the HMM model (strict and non-strict)."""
        try:
            hmm_results = hmm_signals(market_data)
            if not isinstance(hmm_results, (list, tuple)) or len(hmm_results) < 2:
                logger.warning("HMM signals returned an unexpected format or insufficient values.")
                return 0

            strict_signal, non_strict_signal = hmm_results[:2]
            signal_map = {1: SIGNAL_LONG, -1: SIGNAL_SHORT, 0: SIGNAL_NEUTRAL}
            strict_signal_str = signal_map.get(strict_signal, SIGNAL_NEUTRAL)
            non_strict_signal_str = signal_map.get(non_strict_signal, SIGNAL_NEUTRAL)

            strict_score = self._calculate_signal_match_score(signal, strict_signal_str)
            non_strict_score = self._calculate_signal_match_score(signal, non_strict_signal_str)

            return strict_score + non_strict_score
        except Exception as e:
            symbol_info = market_data.iloc[0]['symbol'] if not market_data.empty and 'symbol' in market_data.columns else 'unknown symbol'
            logger.error(f"Error calculating HMM signal for {symbol_info}: {e}")
            return 0

    def get_transformer_signal_score(self, market_data: pd.DataFrame, signal: str) -> int:
        """Gets the signal score from the global Transformer model."""
        try:
            model_path = MODELS_DIR / "transformer_model_global.pth"
            if not model_path.exists():
                return 0

            model_data = load_transformer_model(str(model_path))
            if not model_data:
                return 0

            model, scaler, feature_cols, target_idx = model_data
            if model is None or scaler is None or feature_cols is None or target_idx is None:
                logger.warning("Transformer model data is incomplete. Skipping prediction.")
                return 0

            predicted_signal = get_latest_transformer_signal(
                market_data, model, scaler, feature_cols, target_idx
            )
            return self._calculate_signal_match_score(predicted_signal, signal)
        except (IOError, ValueError, TypeError) as e:
            logger.error(f"Error getting Transformer signal: {e}")
            return 0

    def get_lstm_signal_score(self, market_data: pd.DataFrame, signal: str) -> int:
        """Gets the combined signal score from all 12 LSTM model variants."""
        total_score = 0
        model_files = list(MODELS_DIR.glob('*_model_global.pth'))
        lstm_model_files = [f for f in model_files if any(n in str(f) for n in ['lstm', 'cnn'])]

        if not lstm_model_files:
            return 0

        for model_path in lstm_model_files:
            try:
                model_data = load_cnn_lstm_attention_model(str(model_path))
                if not model_data:
                    logger.warning(f"Failed to load LSTM model from {model_path}")
                    continue

                model, config, data_info, opt_res = model_data
                if not model or not config:
                    logger.warning(f"Invalid model or config in {model_path}")
                    continue

                predicted_signal = get_latest_cnn_lstm_attention_signal(
                    market_data, model, config, data_info, opt_res
                )
                score = self._calculate_signal_match_score(signal, predicted_signal)
                total_score += score
                logger.signal(
                    f"LSTM model {model_path.name}: "
                    f"Predicted={predicted_signal}, Target={signal}, Score={score}"
                )
            except (IOError, ValueError, TypeError) as e:
                logger.error(f"Error getting LSTM signal from {model_path.name}: {e}")

        return total_score

    def _calculate_signal_match_score(self, main_signal: str, model_signal: str) -> int:
        """Calculates a score based on how well the model signal matches the main signal.

        Args:
            main_signal: The primary signal ('LONG', 'SHORT').
            model_signal: The signal from a model.

        Returns:
            1 for a match, -1 for a conflict, 0 for neutral/no signal.
        """
        main_signal = main_signal.upper()
        model_signal = str(model_signal).upper()

        if (not main_signal or main_signal == SIGNAL_NEUTRAL or
                not model_signal or model_signal == SIGNAL_NEUTRAL):
            return 0

        if main_signal == model_signal:
            return 1
        # This condition means main_signal is 'LONG' and model_signal is 'SHORT' or vice-versa
        if main_signal != model_signal:
            return -1
        return 0 # Should not be reached

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
        analysis_params: Dict[str, str],
        performance_candidates: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Union[str, int, float, Dict[str, int]]]:
        """Performs a comprehensive signal analysis for a single symbol.

        Args:
            analysis_params: Dict with 'symbol', 'timeframe', and 'signal'.
            performance_candidates: Dict containing 'long' and 'short' lists.

        Returns:
            A dictionary containing the detailed analysis results.
        """
        symbol = analysis_params['symbol']
        timeframe = analysis_params['timeframe']
        signal = analysis_params['signal']
        logger.analysis(f"Starting analysis: {symbol}/{timeframe}/{signal}...")

        scores: Dict[str, int] = {}
        performance_score = self.check_symbol_in_performance_list(
            symbol, timeframe, signal, performance_candidates
        )
        scores['performance'] = performance_score

        market_data = self._get_market_data_for_symbol(symbol, timeframe)
        if market_data is None or market_data.empty:
            logger.warning(
                f"No market data for {symbol}/{timeframe}, skipping model checks."
            )
            scores.update({'random_forest': 0, 'hmm': 0, 'transformer': 0, 'lstm': 0})
        else:
            scores['random_forest'] = self.get_random_forest_signal_score(
                market_data, signal
            )
            scores['hmm'] = self.get_hmm_signal_score(market_data, signal)
            scores['transformer'] = self.get_transformer_signal_score(
                market_data, signal
            )
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
        self, reload_model: bool, analysis_params: Dict[str, str]
    ) -> Dict[str, Any]:
        """Main entry point to run a full analysis for a given request.

        Args:
            reload_model: If True, retrains all models before analysis.
            analysis_params: Dict with 'symbol', 'timeframe', and 'signal'.

        Returns:
            A dictionary containing the analysis results or an error message.
        """
        try:
            symbol = analysis_params['symbol']
            timeframe = analysis_params['timeframe']
            signal = analysis_params['signal']

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
            performance_candidates = {
                'long': long_candidates, 'short': short_candidates
            }

            result = self.analyze_symbol_signal(
                analysis_params, performance_candidates
            )

            return result

        except (KeyError, TypeError) as e:
            logger.error(f"Invalid analysis parameters provided: {e}")
            return {'error': 'Invalid analysis parameters.'}
        except RuntimeError as e:
            logger.error(f"A critical error occurred during analysis: {e}")
            return {'error': str(e)}

def print_analysis_result(result: Dict[str, Any]) -> None:
    """Logs the analysis result in a formatted, user-friendly way.

    Args:
        result: The result dictionary from the `run_analysis` method.
    """
    if 'error' in result:
        logger.error(f"Analysis failed: {result['error']}")
        return

    symbol = result.get('symbol', 'N/A')
    timeframe = result.get('timeframe', 'N/A')
    signal = result.get('signal', 'N/A')
    scores = result.get('scores', {})
    total_score = result.get('total_score', 0)
    max_possible_score = result.get('max_possible_score', 1)
    threshold = result.get('threshold', 0.0)
    recommendation = result.get('recommendation', 'WAIT')

    logger.analysis("\n" + "="*60)
    logger.analysis("🔍 TRADING SIGNAL ANALYSIS RESULT")
    logger.analysis("="*60)

    logger.data(f"📊 Symbol:    {symbol}")
    logger.data(f"⏰ Timeframe: {timeframe}")
    logger.signal(f"📈 Signal:    {signal}")

    logger.info("\n📋 SCORE DETAILS:")
    if scores:
        logger.performance(f"  • Performance List:  {scores.get('performance', 0):+3d}")
        logger.model(f"  • Random Forest:     {scores.get('random_forest', 0):+3d}")
        logger.model(f"  • HMM (Combined):    {scores.get('hmm', 0):+3d}")
        logger.model(f"  • Transformer:       {scores.get('transformer', 0):+3d}")
        logger.model(f"  • LSTM (12 models):  {scores.get('lstm', 0):+3d}")

    logger.info("\n🎯 FINAL ASSESSMENT:")
    logger.analysis(
        f"  • Total Score:         {total_score:+3d} / {max_possible_score}"
    )
    logger.analysis(f"  • Confidence Score:    {threshold:.3f}")

    if recommendation == 'ENTER':
        logger.success(f"  • Recommendation:      ✅ {recommendation} (Confidence ≥ 0.7)")
    else:
        logger.warning(f"  • Recommendation:      ⏳ {recommendation} (Confidence < 0.7)")

    logger.analysis("="*60)

def main() -> None:
    """Main function to run the interactive command-line interface."""
    analyzer = TradingSignalAnalyzer()

    logger.info("\n" + "="*60)
    logger.info("Trading Signal Analyzer (Global Models)")
    logger.info("="*60)

    while True:
        try:
            reload_input = input("❓ Do you want to reload all models? (yes/no): ").lower()
            if reload_input in ['yes', 'y', 'no', 'n']:
                reload_model = reload_input in ['yes', 'y']
                break
            logger.warning("   ❌ Invalid input. Please enter 'yes' or 'no'.")
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting.")
            return

    while True:
        try:
            symbol_input = input("❓ Enter symbol to check (e.g., BTC-USDT): ").upper()
            if analyzer.validate_symbol(symbol_input):
                break
            logger.warning(f"   ❌ Invalid symbol '{symbol_input}'. Please try again.")
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting.")
            return

    while True:
        try:
            timeframe_input = input(f"❓ Enter timeframe ({', '.join(analyzer.valid_timeframes)}): ").lower()
            if timeframe_input in analyzer.valid_timeframes:
                break
            logger.warning("   ❌ Invalid timeframe. Please choose from the list.")
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting.")
            return

    while True:
        try:
            signal_input = input("❓ Enter signal to check (LONG/SHORT): ").upper()
            if signal_input in [SIGNAL_LONG, SIGNAL_SHORT]:
                break
            logger.warning("   ❌ Invalid input. Please enter 'LONG' or 'SHORT'.")
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting.")
            return

    logger.config("\n" + "="*60)
    logger.config("Starting analysis with parameters:")
    logger.config(f"  - Reload Models: {reload_model}")
    logger.config(f"  - Symbol:        {symbol_input}")
    logger.config(f"  - Timeframe:     {timeframe_input}")
    logger.config(f"  - Signal:        {signal_input}")
    logger.config("="*60)

    analysis_params = {
        'symbol': symbol_input,
        'timeframe': timeframe_input,
        'signal': signal_input
    }
    result = analyzer.run_analysis(
        reload_model=reload_model,
        analysis_params=analysis_params
    )

    print_analysis_result(result)

if __name__ == "__main__":
    main()
