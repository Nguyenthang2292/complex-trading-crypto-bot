#!/usr/bin/env python3
"""
Trading Signal Analyzer - All Symbols Scanner

Scans all available symbols to find LONG/SHORT signals using a chain-filtering
approach with multiple models. This script allows for market-wide analysis,
identifying candidates based on performance and then validating them with a
sequence of machine learning models.

Usage:
    python trading_signal_analyzer_all_symbols.py
"""

import logging
import os
import pandas as pd
import shutil
import sys
import textwrap
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from components._load_all_symbols_data import load_all_symbols_data
from components.tick_processor import TickProcessor
from components._combine_all_dataframes import combine_all_dataframes
from components.config import (
    DEFAULT_TIMEFRAMES, DEFAULT_CRYPTO_SYMBOLS, MODELS_DIR, 
    SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL
)
from signals.best_performance_pairs.signals_best_performance_symbols import signal_best_performance_symbols, get_short_signal_candidates
from signals.random_forest.signals_random_forest import get_latest_random_forest_signal, load_random_forest_model, train_and_save_global_rf_model
from signals.hmm.signals_hmm import hmm_signals
from signals.signals_transformer import get_latest_transformer_signal, load_transformer_model, train_and_save_transformer_model
from signals.signals_cnn_lstm_attention import (
    train_cnn_lstm_attention_model, 
    load_cnn_lstm_attention_model, 
    get_latest_cnn_lstm_attention_signal
)
from utilities._logger import setup_logging

logger = setup_logging(module_name="trading_signal_analyzer_all_symbols", log_level=logging.INFO)

class TradingSignalAnalyzer:
    """Scans and analyzes trading signals for all symbols using chain filtering."""
    
    def __init__(self) -> None:
        """Initializes the analyzer, fetching the list of valid symbols."""
        self.valid_timeframes: List[str] = DEFAULT_TIMEFRAMES
        self.processor = TickProcessor(trade_open_callback=None, trade_close_callback=None)
        
        try:
            self.valid_symbols: List[str] = self.processor.get_symbols_list_by_quote_usdt()
            logger.config(f"Initialized with {len(self.valid_symbols)} USDT pair symbols from the exchange.")
        except Exception as e:
            logger.warning(f"Could not fetch symbols from exchange: {e}. Using default list.")
            self.valid_symbols = DEFAULT_CRYPTO_SYMBOLS
        
        self.available_models: Dict[str, str] = {
            'random_forest': 'Random Forest',
            'hmm': 'HMM (Strict + Non-Strict)',
            'transformer': 'Transformer',
            'lstm': 'LSTM (12 Variants Combined)'
        }
        
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
                    # Ensure data is in the expected format Dict[str, DataFrame]
                    if isinstance(data, dict):
                        filtered_data[symbol] = {
                            tf: df for tf, df in data.items() if isinstance(df, pd.DataFrame)
                        }
                    elif isinstance(data, pd.DataFrame):
                        # If data is a DataFrame, wrap it in a dict with a default key
                        filtered_data[symbol] = {'default': data}
            
            combined_df = combine_all_dataframes(filtered_data)
            
            if combined_df.empty:
                raise RuntimeError("Combined dataframe is empty.")
            logger.success(f"Successfully combined dataframe with {len(combined_df)} rows.")
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
        try:
            logger.model("Training all models from the combined DataFrame...")
            if combined_df.empty:
                logger.error("Combined DataFrame is empty, cannot train models.")
                return
            
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
            
        except Exception as e:
            logger.error(f"Error training models from combined data: {e}")
            traceback.print_exc()
            raise
    
    def analyze_best_performance_signals(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Runs the best/worst performance analysis to find candidate symbols.

        Returns:
            A tuple containing lists of LONG and SHORT candidate dictionaries.
        """
        try:
            logger.analysis("Analyzing best/worst performing symbols...")
            
            symbol_data = load_all_symbols_data(
                processor=self.processor,
                symbols=self.valid_symbols,
                timeframes=DEFAULT_TIMEFRAMES
            )
            
            if not symbol_data:
                logger.error("No data available for performance analysis.")
                return [], []
            
            # Filter symbol_data to match expected type structure
            filtered_symbol_data: Dict[str, Dict[str, pd.DataFrame]] = {}
            for symbol, data in symbol_data.items():
                if data is not None and isinstance(data, dict):
                    
                    # Filter out None values within the inner dict
                    filtered_timeframes = {
                        tf: df for tf, df in data.items() if isinstance(df, pd.DataFrame)
                    }
                    if filtered_timeframes:
                        filtered_symbol_data[symbol] = filtered_timeframes
            
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
                preloaded_data=filtered_symbol_data
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
            
        except Exception as e:
            logger.error(f"Error during performance analysis: {e}")
            return [], []
    
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
                logger.warning(f"No data available for {symbol}")
                return None
            
            symbol_data_item = symbol_data.get(symbol)
            if symbol_data_item is None:
                logger.warning(f"Data is None for {symbol}")
                return None
                
            if isinstance(symbol_data_item, pd.DataFrame):
                return symbol_data_item
            
            if isinstance(symbol_data_item, dict):
                timeframe_data = symbol_data_item.get(timeframe)
                if isinstance(timeframe_data, pd.DataFrame):
                    return timeframe_data
            
            logger.warning(f"Data for {symbol} {timeframe} is not in a recognized format.")
            return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
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
            logger.debug(f"HMM analysis failed for a symbol: {e}")
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
                logger.warning("Transformer model components are None.")
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
                    total_score += self._calculate_signal_match_score(predicted, signal, variant_name.upper())
                    successful_predictions += 1
                except Exception as variant_error:
                    logger.debug(f"Error with LSTM variant {variant_name}: {variant_error}")
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
            logger.debug(f"{model_name}: No signal or neutral: 0")
            return 0
            
        if predicted_signal.upper() == target_signal.upper():
            logger.debug(f"{model_name}: Signal match ({predicted_signal}): +1")
            return 1
        
        logger.debug(f"{model_name}: Signal conflict ({predicted_signal} vs {target_signal}): -1")
        return -1
    
    def apply_chain_filtering(
        self,
        candidates: List[Dict],
        signal_type: str,
        model1: str,
        model2: str
    ) -> List[Dict]:
        """Applies a two-model chain filter to performance-based candidates.

        A candidate passes if both models give a positive score for the target
        signal on at least one of its analyzed timeframes.

        Args:
            candidates: A list of candidate symbols from performance analysis.
            signal_type: The target signal type ('LONG' or 'SHORT').
            model1: The key for the first model in the chain.
            model2: The key for the second model in the chain.

        Returns:
            A list of candidates that passed the chain filtering process.
        """
        filtered_candidates = []
        model1_name = self.available_models.get(model1, model1)
        model2_name = self.available_models.get(model2, model2)
        logger.analysis(
            f"Applying chain filter ({model1_name} → {model2_name}) for "
            f"{len(candidates)} {signal_type} candidates..."
        )
        
        for candidate in candidates:
            symbol = candidate.get('symbol')
            if not symbol:
                continue

            timeframe_scores = candidate.get('timeframe_scores', {})
            passed_timeframes = {}
            
            for timeframe in timeframe_scores.keys():
                if timeframe not in self.valid_timeframes:
                    continue
                    
                # Model 1: Must give a positive score to proceed
                score1 = self._get_model_score(symbol, timeframe, signal_type, model1)
                if score1 <= 0:
                    continue

                # Model 2: Must also give a positive score
                score2 = self._get_model_score(symbol, timeframe, signal_type, model2)
                if score2 > 0:
                    passed_timeframes[timeframe] = {
                        'original_score': timeframe_scores[timeframe],
                        'chain_score': score1 + score2,
                        f'{model1}_score': score1,
                        f'{model2}_score': score2
                    }
            
            # Keep candidate only if at least one timeframe passed the chain
            if passed_timeframes:
                filtered_candidate = candidate.copy()
                filtered_candidate['timeframe_scores'] = passed_timeframes
                filtered_candidate['chain_models'] = [model1, model2]
                filtered_candidates.append(filtered_candidate)
        
        logger.analysis(
            f"Chain filtering result: {len(filtered_candidates)}/{len(candidates)} "
            f"{signal_type} candidates passed."
        )
        return filtered_candidates
    
    def _get_model_score(self, symbol: str, timeframe: str, signal: str, model_key: str) -> int:
        """Gets the signal score for a specific model by its key.

        This method fetches the market data and dispatches the call to the
        appropriate model-scoring function.

        Args:
            symbol: The trading symbol.
            timeframe: The data timeframe.
            signal: The target signal ('LONG' or 'SHORT').
            model_key: The key of the model to use (e.g., 'random_forest').

        Returns:
            The integer score from the specified model.
        """
        try:
            market_data = self._get_market_data_for_symbol(symbol, timeframe)
            if market_data is None or market_data.empty:
                return 0

            model_functions = {
                'random_forest': self.get_random_forest_signal_score,
                'hmm': self.get_hmm_signal_score,
                'transformer': self.get_transformer_signal_score,
                'lstm': self.get_lstm_signal_score
            }
            
            if model_key in model_functions:
                return model_functions[model_key](market_data, signal)
            
            logger.warning(f"Unknown model key: {model_key}")
            return 0
        except Exception as e:
            logger.error(f"Error getting score from model {model_key}: {e}")
            return 0
    
    def run_full_market_scan(
        self,
        reload_models: bool,
        chain_model1: str,
        chain_model2: str
    ) -> Dict[str, Any]:
        """Runs a full market scan using a specified model chain.

        Args:
            reload_models: If True, retrains all models before analysis.
            chain_model1: The key for the first model in the chain.
            chain_model2: The key for the second model in the chain.

        Returns:
            A dictionary containing the results of the market scan.
        """
        try:
            logger.process("🔍 STARTING FULL MARKET SCAN")
            
            if reload_models:
                self.reload_all_models()
            
            # Step 1: Get candidates from performance analysis
            logger.analysis("📊 STEP 1: Analyzing symbol performance...")
            long_candidates, short_candidates = self.analyze_best_performance_signals()
            
            if not long_candidates and not short_candidates:
                return {'error': "No candidates found from performance analysis."}
            logger.analysis(
                f"✅ Found {len(long_candidates)} LONG and "
                f"{len(short_candidates)} SHORT candidates."
            )
            
            # Step 2: Apply the specified chain filter
            logger.analysis(f"🔗 STEP 2: Applying chain filter: {chain_model1} → {chain_model2}")
            filtered_long = self.apply_chain_filtering(
                long_candidates, SIGNAL_LONG, chain_model1, chain_model2
            )
            filtered_short = self.apply_chain_filtering(
                short_candidates, SIGNAL_SHORT, chain_model1, chain_model2
            )
            
            # Step 3: Prepare and return results
            result = {
                'success': True,
                'chain_models': [chain_model1, chain_model2],
                'original_candidates': {
                    'long_count': len(long_candidates),
                    'short_count': len(short_candidates)
                },
                'filtered_candidates': {
                    'long_candidates': filtered_long,
                    'short_candidates': filtered_short,
                    'long_count': len(filtered_long),
                    'short_count': len(filtered_short)
                },
                'filtering_efficiency': {
                    'long_pass_rate': len(filtered_long) / len(long_candidates) if long_candidates else 0,
                    'short_pass_rate': len(filtered_short) / len(short_candidates) if short_candidates else 0
                }
            }
            
            logger.success(
                f"🎯 COMPLETE: {len(filtered_long)} LONG and "
                f"{len(filtered_short)} SHORT symbols passed the chain filter."
            )
            return result
            
        except Exception as e:
            logger.error(f"Error during full market scan: {e}")
            traceback.print_exc()
            return {'error': str(e)}

    def run_comparison_analysis(
        self,
        reload_models: bool,
        main_chain_model1: str,
        main_chain_model2: str
    ) -> Dict[str, Any]:
        """Runs a comparison of all possible two-model chain combinations.

        Args:
            reload_models: If True, retrains all models before analysis.
            main_chain_model1: The key for the user's selected primary model.
            main_chain_model2: The key for the user's selected secondary model.

        Returns:
            A dictionary containing the comparative analysis results.
        """
        try:
            logger.process("🔍 STARTING MODEL CHAIN COMPARISON ANALYSIS")
            
            if reload_models:
                self.reload_all_models()
            
            # Step 1: Get base candidates once
            logger.analysis("📊 Analyzing symbol performance for all chains...")
            long_candidates, short_candidates = self.analyze_best_performance_signals()
            if not long_candidates and not short_candidates:
                return {'error': "No candidates found from performance analysis."}
            
            # Step 2: Generate all unique two-model combinations
            available_models = list(self.available_models.keys())
            all_combinations = [
                (m1, m2) for m1 in available_models for m2 in available_models if m1 != m2
            ]
            logger.analysis(f"🔗 Testing {len(all_combinations)} model chain combinations.")
            
            # Step 3: Run filtering for each combination
            comparison_results = {}
            main_chain_key = f"{main_chain_model1}_{main_chain_model2}"
            
            for model1, model2 in all_combinations:
                chain_key = f"{model1}_{model2}"
                model1_name = self.available_models.get(model1, model1)
                model2_name = self.available_models.get(model2, model2)
                logger.analysis(f"Testing chain: {model1_name} → {model2_name}")
                
                filtered_long = self.apply_chain_filtering(long_candidates, SIGNAL_LONG, model1, model2)
                filtered_short = self.apply_chain_filtering(short_candidates, SIGNAL_SHORT, model1, model2)
                
                comparison_results[chain_key] = {
                    'models': [model1, model2],
                    'model_names': [model1_name, model2_name],
                    'long_candidates': filtered_long,
                    'short_candidates': filtered_short,
                    'long_count': len(filtered_long),
                    'short_count': len(filtered_short),
                    'long_symbols': [c['symbol'] for c in filtered_long],
                    'short_symbols': [c['symbol'] for c in filtered_short],
                    'total_signals': len(filtered_long) + len(filtered_short),
                    'is_main_chain': chain_key == main_chain_key
                }
            
            # Step 4: Calculate rankings and prepare final result
            sorted_chains = sorted(
                comparison_results.items(),
                key=lambda item: item[1]['total_signals'],
                reverse=True
            )
            
            result = {
                'success': True,
                'main_chain': main_chain_key,
                'original_candidates': {
                    'long_count': len(long_candidates),
                    'short_count': len(short_candidates),
                },
                'comparison_results': comparison_results,
                'rankings': sorted_chains,
                'best_chain': sorted_chains[0][0] if sorted_chains else None,
                'main_chain_rank': next(
                    (i + 1 for i, (chain, _) in enumerate(sorted_chains) if chain == main_chain_key),
                    None
                )
            }
            
            logger.success(f"🎯 COMPARISON COMPLETE: {len(all_combinations)} chains tested.")
            return result
            
        except Exception as e:
            logger.error(f"Error during comparison analysis: {e}")
            traceback.print_exc()
            return {'error': str(e)}


def print_market_scan_results(result: Dict[str, Any]) -> None:
    """Prints the results of a full market scan in a formatted way.

    Args:
        result: The result dictionary from the `run_full_market_scan` method.
    """
    if 'error' in result:
        logger.error(f"❌ Scan failed: {result['error']}")
        return
    
    logger.success("\n" + "="*80)
    logger.success("MARKET SCAN CHAIN FILTERING RESULTS")
    logger.success("="*80)
    
    chain_models = result.get('chain_models', ['N/A', 'N/A'])
    model_names = result.get('model_names', chain_models)
    logger.info(f"🔗 Chain Used: {' → '.join(model_names)}")
    
    original = result.get('original_candidates', {})
    filtered = result.get('filtered_candidates', {})
    efficiency = result.get('filtering_efficiency', {})
    
    logger.info("\n📊 OVERALL STATISTICS:")
    logger.info(f"  • Original LONG candidates:  {original.get('long_count', 0)}")
    logger.info(f"  • Original SHORT candidates: {original.get('short_count', 0)}")
    logger.info(
        f"  • Filtered LONG candidates:  {filtered.get('long_count', 0)} "
        f"({efficiency.get('long_pass_rate', 0):.1%} pass rate)"
    )
    logger.info(
        f"  • Filtered SHORT candidates: {filtered.get('short_count', 0)} "
        f"({efficiency.get('short_pass_rate', 0):.1%} pass rate)"
    )
    
    # Print LONG Results
    long_candidates = filtered.get('long_candidates', [])
    if long_candidates:
        logger.info(f"\n📈 LONG SIGNALS ({len(long_candidates)} symbols):")
        logger.info("-" * 80)
        
        long_symbols = [c.get('symbol', 'N/A') for c in long_candidates]
        symbols_str = ', '.join(long_symbols)
        wrapped_symbols = textwrap.fill(symbols_str, width=75, initial_indent='  ', subsequent_indent='  ')
        logger.info(wrapped_symbols)
        
        logger.info("\nTop 5 detailed:")
        for i, candidate in enumerate(long_candidates[:5], 1):
            symbol = candidate.get('symbol', 'N/A')
            timeframes = list(candidate.get('timeframe_scores', {}).keys())
            chain_models_used = candidate.get('chain_models', [])
            
            logger.info(f"{i:2d}. {symbol:12s} | Timeframes: {', '.join(timeframes[:3])} | Chain: {' → '.join(chain_models_used)}")
            
            if timeframes:
                best_tf = timeframes[0]
                tf_data = candidate['timeframe_scores'][best_tf]
                logger.info(f"     └── {best_tf}: Chain Score: {tf_data.get('chain_score', 'N/A')}, Original: {tf_data.get('original_score', 0):.3f}")
    else:
        logger.info("\n📈 LONG SIGNALS: No symbols passed the filter.")
    
    # Print SHORT Results
    short_candidates = filtered.get('short_candidates', [])
    if short_candidates:
        logger.info(f"\n📉 SHORT SIGNALS ({len(short_candidates)} symbols):")
        logger.info("-" * 80)
        
        short_symbols = [c.get('symbol', 'N/A') for c in short_candidates]
        symbols_str = ', '.join(short_symbols)
        wrapped_symbols = textwrap.fill(symbols_str, width=75, initial_indent='  ', subsequent_indent='  ')
        logger.info(wrapped_symbols)
        
        logger.info("\nTop 5 detailed:")
        for i, candidate in enumerate(short_candidates[:5], 1):
            symbol = candidate.get('symbol', 'N/A')
            timeframes = list(candidate.get('timeframe_scores', {}).keys())
            chain_models_used = candidate.get('chain_models', [])
            
            logger.info(f"{i:2d}. {symbol:12s} | Timeframes: {', '.join(timeframes[:3])} | Chain: {' → '.join(chain_models_used)}")
            
            if timeframes:
                best_tf = timeframes[0]
                tf_data = candidate['timeframe_scores'][best_tf]
                logger.info(f"     └── {best_tf}: Chain Score: {tf_data.get('chain_score', 'N/A')}, Original: {tf_data.get('original_score', 0):.3f}")
    else:
        logger.info("\n📉 SHORT SIGNALS: No symbols passed the filter.")
    
    logger.info("="*80)


def print_comparison_results(result: Dict[str, Any]) -> None:
    """Prints the results of the chain comparison analysis.

    Args:
        result: The result dictionary from `run_comparison_analysis`.
    """
    if 'error' in result:
        logger.error(f"❌ Comparison failed: {result['error']}")
        return
    
    logger.success("\n" + "="*100)
    logger.success("MODEL CHAIN COMBINATION COMPARISON RESULTS")
    logger.success("="*100)
    
    main_chain_key = result.get('main_chain', '')
    main_chain_rank = result.get('main_chain_rank')
    
    logger.info(f"🎯 Main Chain: {main_chain_key.replace('_', ' → ')} (Rank: #{main_chain_rank})")
    
    original = result.get('original_candidates', {})
    logger.info("\n📊 ORIGINAL CANDIDATES:")
    logger.info(f"  • LONG candidates: {original.get('long_count', 0)} symbols")
    logger.info(f"  • SHORT candidates: {original.get('short_count', 0)} symbols")
    
    logger.info("\n🏅 RANKINGS BY TOTAL SIGNALS:")
    logger.info("-" * 100)
    logger.info(f"{'Rank':<6}{'Chain Models':<35}{'LONG':<8}{'SHORT':<8}{'Total':<8}{'Status':<15}")
    logger.info("-" * 100)
    
    rankings = result.get('rankings', [])
    for rank, (chain_name, data) in enumerate(rankings, 1):
        model_names = ' → '.join(data.get('model_names', []))
        status = "🎯 MAIN" if data.get('is_main_chain', False) else ""
        
        logger.info(
            f"{rank:<6}{model_names:<35}{data.get('long_count', 0):<8}"
            f"{data.get('short_count', 0):<8}{data.get('total_signals', 0):<8}{status:<15}"
        )
    
    comparison_results = result.get('comparison_results', {})
    main_chain_data = comparison_results.get(main_chain_key, {})
    
    if main_chain_data:
        logger.info("\n🎯 MAIN CHAIN DETAILED RESULTS:")
        logger.info(f"Chain: {' → '.join(main_chain_data.get('model_names', []))}")
        logger.info("-" * 100)
        
        long_symbols = main_chain_data.get('long_symbols', [])
        if long_symbols:
            logger.info(f"\n📈 LONG SIGNALS ({len(long_symbols)} symbols):")
            wrapped_symbols = textwrap.fill(', '.join(long_symbols), width=90, initial_indent='  ', subsequent_indent='  ')
            logger.info(wrapped_symbols)
        else:
            logger.info("\n📈 LONG SIGNALS: No symbols passed.")
        
        short_symbols = main_chain_data.get('short_symbols', [])
        if short_symbols:
            logger.info(f"\n📉 SHORT SIGNALS ({len(short_symbols)} symbols):")
            wrapped_symbols = textwrap.fill(', '.join(short_symbols), width=90, initial_indent='  ', subsequent_indent='  ')
            logger.info(wrapped_symbols)
        else:
            logger.info("\n📉 SHORT SIGNALS: No symbols passed.")
    
    logger.info("\n🏆 TOP 3 BEST PERFORMING CHAINS:")
    logger.info("-" * 100)
    for rank, (chain_name, data) in enumerate(rankings[:3], 1):
        model_names = ' → '.join(data.get('model_names', []))
        logger.info(f"\n{rank}. {model_names} (Total: {data.get('total_signals', 0)} signals)")
        
        long_symbols = data.get('long_symbols', [])
        if long_symbols:
            preview = ', '.join(long_symbols[:5])
            if len(long_symbols) > 5:
                preview += f" ... (+{len(long_symbols) - 5} more)"
            logger.info(f"   📈 LONG ({data.get('long_count', 0)}): {preview}")
        
        short_symbols = data.get('short_symbols', [])
        if short_symbols:
            preview = ', '.join(short_symbols[:5])
            if len(short_symbols) > 5:
                preview += f" ... (+{len(short_symbols) - 5} more)"
            logger.info(f"   📉 SHORT ({data.get('short_count', 0)}): {preview}")
    
    best_chain_key = result.get('best_chain')
    if best_chain_key and best_chain_key != main_chain_key:
        best_chain_data = comparison_results.get(best_chain_key, {})
        improvement = best_chain_data.get('total_signals', 0) - main_chain_data.get('total_signals', 0)
        
        if improvement > 0:
            logger.info("\n💡 PERFORMANCE ANALYSIS:")
            logger.info(f"  • Best chain: {' → '.join(best_chain_data.get('model_names', []))} ({best_chain_data.get('total_signals', 0)} signals)")
            logger.info(f"  • Your chain: {' → '.join(main_chain_data.get('model_names', []))} ({main_chain_data.get('total_signals', 0)} signals)")
            if main_chain_data.get('total_signals', 0) > 0:
                efficiency_gain = improvement / main_chain_data.get('total_signals', 1) * 100
                logger.info(f"  • Potential improvement: +{improvement} signals ({efficiency_gain:.1f}%)")
    
    logger.info("="*100)


def main() -> None:
    """Main function to run the interactive command-line interface."""
    analyzer = TradingSignalAnalyzer()

    print("="*80)
    print("MARKET-WIDE SCANNER - CHAIN FILTERING ANALYSIS")
    print("="*80)

    while True:
        reload_input = input("❓ Do you want to reload all models? (yes/no): ").lower()
        if reload_input in ['yes', 'y', 'no', 'n']:
            reload_models = reload_input in ['yes', 'y']
            break
        logger.error("   ❌ Invalid input. Please enter 'yes' or 'no'.")

    logger.info("\n📊 ANALYSIS MODE:")
    logger.info("  • 'simple': Test only your selected chain.")
    logger.info("  • 'compare': Compare your chain against all other combinations.")
    
    while True:
        mode_input = input("\n❓ Select mode (simple/compare): ").lower()
        if mode_input in ['simple', 'compare']:
            break
        logger.error("   ❌ Invalid input. Please select 'simple' or 'compare'.")

    logger.info("\n🔗 SELECT 2 MODELS FOR CHAIN FILTERING:")
    for key, name in analyzer.available_models.items():
        logger.info(f"  • {key}: {name}")
    
    model1_input = ""
    while model1_input not in analyzer.available_models:
        model1_input = input("\n❓ Select first model: ").lower()
        if model1_input not in analyzer.available_models:
            logger.error(f"   ❌ Invalid model. Please choose from: {', '.join(analyzer.available_models.keys())}")
    
    model2_input = ""
    while model2_input not in analyzer.available_models or model2_input == model1_input:
        model2_input = input("❓ Select second model: ").lower()
        if model2_input == model1_input:
            logger.error("   ❌ Please select a different model from the first one.")
        elif model2_input not in analyzer.available_models:
            logger.error(f"   ❌ Invalid model. Please choose from: {', '.join(analyzer.available_models.keys())}")
    
    print("="*80)
    logger.config("Starting market scan with parameters:")
    logger.config(f"  - Reload Models: {reload_models}")
    logger.config(f"  - Mode:          {mode_input}")
    logger.config(f"  - Chain Model 1: {analyzer.available_models[model1_input]}")
    logger.config(f"  - Chain Model 2: {analyzer.available_models[model2_input]}")
    logger.config(f"  - Total Symbols: {len(analyzer.valid_symbols)}")
    logger.config(f"  - Timeframes:    {', '.join(analyzer.valid_timeframes)}")
    print("="*80)

    if mode_input == 'compare':
        result = analyzer.run_comparison_analysis(
            reload_models=reload_models,
            main_chain_model1=model1_input,
            main_chain_model2=model2_input
        )
        print_comparison_results(result)
    else:
        result = analyzer.run_full_market_scan(
            reload_models=reload_models,
            chain_model1=model1_input,
            chain_model2=model2_input
        )
        print_market_scan_results(result)

if __name__ == "__main__":
    main()