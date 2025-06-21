#!/usr/bin/env python3
"""
Trading Signal Analyzer - All Symbols Scanner
Quét toàn bộ symbols để tìm tín hiệu LONG/SHORT sử dụng chain filtering models.

Usage:
    python trading_signal_analyzer_all_symbols.py
"""

import logging
import os
import pandas as pd
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from livetrade._components._load_all_symbols_data import load_all_symbols_data
from livetrade._components._tick_processor import tick_processor
from livetrade._components._combine_all_dataframes import combine_all_dataframes
from livetrade.config import (
    DEFAULT_TIMEFRAMES, DEFAULT_CRYPTO_SYMBOLS, MODELS_DIR, 
    SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL
)
from signals.signals_best_performance_symbols import signal_best_performance_symbols, get_short_signal_candidates
from signals.signals_random_forest import get_latest_random_forest_signal, load_random_forest_model, train_and_save_global_rf_model
from signals.signals_hmm import hmm_signals
from signals.signals_transformer import get_latest_transformer_signal, load_transformer_model, train_and_save_transformer_model
from signals.signals_cnn_lstm_attention import (
    train_and_save_global_cnn_lstm_attention_model, 
    load_cnn_lstm_attention_model, 
    get_latest_cnn_lstm_attention_signal
)
from utilities._logger import setup_logging

logger = setup_logging(module_name="trading_signal_analyzer_all_symbols", log_level=logging.INFO)

class TradingSignalAnalyzer:
    """Quét và phân tích tín hiệu giao dịch cho toàn bộ symbols sử dụng chain filtering models"""
    
    def __init__(self) -> None:
        """Khởi tạo analyzer với processor và danh sách symbols từ exchange"""
        self.valid_timeframes: List[str] = DEFAULT_TIMEFRAMES
        self.processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
        
        try:
            self.valid_symbols: List[str] = self.processor.get_symbols_list_by_quote_usdt()
            logger.config(f"Khởi tạo với {len(self.valid_symbols)} symbols USDT pairs từ exchange")
        except Exception as e:
            logger.warning(f"Không thể lấy symbols từ exchange: {e}. Sử dụng danh sách mặc định.")
            self.valid_symbols = DEFAULT_CRYPTO_SYMBOLS
        
        self.available_models: Dict[str, str] = {
            'random_forest': 'Random Forest',
            'hmm': 'HMM (Strict + Non-Strict Combined)',
            'transformer': 'Transformer',
            'lstm': 'LSTM (4 Variants Combined)'
        }
        
    def validate_symbol(self, symbol: str) -> bool:
        """Kiểm tra symbol có hợp lệ không"""
        if '-' in symbol:
            base, quote = symbol.split('-')
            symbol = f"{base}{quote}"
        return symbol.upper() in [s.upper() for s in self.valid_symbols]
    
    def normalize_symbol(self, symbol: str) -> str:
        """Chuẩn hóa tên symbol về format BTCUSDT"""
        if '-' in symbol:
            base, quote = symbol.split('-')
            return f"{base}{quote}".upper()
        return symbol.upper()
    
    def clear_models_directory(self) -> None:
        """Xóa và tạo lại thư mục models"""
        try:
            if MODELS_DIR.exists():
                logger.process(f"Đang xóa toàn bộ models trong {MODELS_DIR}")
                shutil.rmtree(MODELS_DIR)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            logger.success("Đã xóa và tạo lại thư mục models")      
        except Exception as e:
            logger.error(f"Lỗi khi xóa thư mục models: {e}")
            raise
    
    def reload_all_models(self) -> None:
        """Load toàn bộ dữ liệu và train lại tất cả models global"""
        try:
            logger.process("Bắt đầu reload toàn bộ models...")
            self.clear_models_directory()
            
            logger.data(f"Sử dụng {len(self.valid_symbols)} symbols USDT pairs đã load từ exchange")
            logger.data(f"Đang load dữ liệu cho {len(self.valid_symbols)} symbols và {len(DEFAULT_TIMEFRAMES)} timeframes...")
            
            all_symbols_data = load_all_symbols_data(
                processor=self.processor,
                symbols=self.valid_symbols,
                timeframes=DEFAULT_TIMEFRAMES
            )
            
            if not all_symbols_data:
                raise Exception("Load data thất bại")
            
            logger.data("Đang kết hợp tất cả dataframes...")
            combined_df = combine_all_dataframes(all_symbols_data)
            
            if combined_df.empty:
                raise Exception("Combine dataframes thất bại")
            
            logger.success(f"Đã kết hợp thành công DataFrame với {len(combined_df)} rows, {len(combined_df.columns)} columns")
            self._train_all_models_from_combined_data(combined_df)
            logger.success("Hoàn thành reload toàn bộ models")
            
        except Exception as e:
            logger.error(f"Lỗi khi reload models: {e}")
            raise
    
    def _train_all_models_from_combined_data(self, combined_df: pd.DataFrame) -> None:
        """Train tất cả các models từ DataFrame đã được kết hợp"""
        try:
            logger.model("Bắt đầu training tất cả models từ combined DataFrame...")
            logger.data(f"Combined DataFrame shape: {combined_df.shape}")
            
            if combined_df.empty:
                logger.error("Combined DataFrame is empty")
                return
            
            unique_symbols = combined_df['symbol'].nunique() if 'symbol' in combined_df.columns else 0
            unique_timeframes = combined_df['timeframe'].nunique() if 'timeframe' in combined_df.columns else 0
            logger.data(f"Training data contains {unique_symbols} unique symbols and {unique_timeframes} unique timeframes")
            
            model_configs = [
                ("Random Forest", lambda: train_and_save_global_rf_model(combined_df, model_filename="rf_model_global.joblib")),
                ("Transformer", lambda: train_and_save_transformer_model(combined_df, model_filename="transformer_model_global.pth")),
                ("LSTM", lambda: train_and_save_global_cnn_lstm_attention_model(combined_df, model_filename="lstm_model_global.pth", use_cnn=False, use_attention=False)),
                ("LSTM-Attention", lambda: train_and_save_global_cnn_lstm_attention_model(combined_df, model_filename="lstm_attention_model_global.pth", use_cnn=False, use_attention=True)),
                ("CNN-LSTM", lambda: train_and_save_global_cnn_lstm_attention_model(combined_df, model_filename="cnn_lstm_model_global.pth", use_cnn=True, use_attention=False)),
                ("CNN-LSTM-Attention", lambda: train_and_save_global_cnn_lstm_attention_model(combined_df, model_filename="cnn_lstm_attention_model_global.pth", use_cnn=True, use_attention=True))
            ]
            
            for model_name, train_func in model_configs:
                logger.model(f"Training {model_name} model...")
                try:
                    model, model_path = train_func()
                    if model:
                        logger.success(f"{model_name} model saved: {model_path}")
                    else:
                        logger.warning(f"{model_name} training failed")
                except Exception as e:
                    logger.error(f"Lỗi khi train {model_name} model: {e}")
            
            logger.success("Hoàn thành training tất cả models từ combined data")
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình train models từ combined data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def analyze_best_performance_signals(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Chạy phân tích best performance symbols và trả về LONG/SHORT candidates"""
        try:
            logger.analysis("Đang phân tích best performance symbols...")
            
            symbol_data = load_all_symbols_data(
                processor=self.processor,
                symbols=self.valid_symbols,
                timeframes=DEFAULT_TIMEFRAMES
            )
            
            if not symbol_data:
                logger.error("Không có dữ liệu để phân tích")
                return [], []
            
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
                preloaded_data=symbol_data
            )
            
            if not analysis_result:
                logger.error("Không có kết quả phân tích")
                return [], []
            
            long_candidates = analysis_result.get('best_performers', [])
            short_candidates = get_short_signal_candidates(analysis_result, min_short_score=0.6)
            
            logger.analysis(f"Tìm thấy {len(long_candidates)} LONG candidates, {len(short_candidates)} SHORT candidates")
            return long_candidates, short_candidates
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích best performance: {e}")
            return [], []
    
    def _get_market_data_for_symbol(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu thị trường cho symbol và timeframe cụ thể"""
        try:
            symbol_data = load_all_symbols_data(
                processor=self.processor,
                symbols=[symbol],
                timeframes=[timeframe]
            )
            
            if not symbol_data or symbol not in symbol_data:
                logger.warning(f"Không có dữ liệu cho {symbol}")
                return None
            
            if timeframe not in symbol_data[symbol]:
                logger.warning(f"Không có dữ liệu {timeframe} cho {symbol}")
                return None            
            return symbol_data[symbol][timeframe]
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu cho {symbol} {timeframe}: {e}")
            return None
    
    def get_random_forest_signal_score(self, symbol: str, timeframe: str, signal: str) -> int:
        """
        Lấy tín hiệu từ Random Forest model global
        
        Returns:
            +1: Signal match, -1: Signal conflict, 0: No signal hoặc lỗi
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            model_path = MODELS_DIR / "rf_model_global.joblib"
            
            if not model_path.exists():
                logger.warning("Không tìm thấy Random Forest model global")
                return 0
                
            model = load_random_forest_model(model_path)
            if not model:
                logger.warning("Không thể load Random Forest model global")
                return 0
            
            market_data = self._get_market_data_for_symbol(normalized_symbol, timeframe)
            if market_data is None or market_data.empty:
                logger.warning(f"Không có dữ liệu thị trường cho {symbol} {timeframe}")
                return 0
            
            predicted_signal = get_latest_random_forest_signal(market_data, model)
            score = self._calculate_signal_match_score(predicted_signal, signal, "Random Forest")
            
            logger.signal(f"Random Forest: {symbol} {timeframe} {signal} score: {score}")
            return score
                
        except Exception as e:
            logger.error(f"Lỗi khi lấy Random Forest signal: {e}")
            return 0
    
    def get_hmm_signal_score(self, symbol: str, timeframe: str, signal: str) -> int:
        """
        Lấy tín hiệu từ HMM model (cả strict và non-strict mode)
        
        Args:
            symbol: Symbol trading
            timeframe: Khung thời gian
            signal: Tín hiệu mong muốn (LONG/SHORT)
            
        Returns:
            Score từ -2 đến +2 (strict + non-strict)
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            market_data = self._get_market_data_for_symbol(normalized_symbol, timeframe)
            if market_data is None or market_data.empty:
                logger.warning(f"Không có dữ liệu thị trường cho {symbol} {timeframe}")
                return 0
            
            try:
                signals = hmm_signals(market_data)
                if len(signals) >= 2:
                    strict_signal, non_strict_signal = signals[:2]
                    
                    signal_map = {1: SIGNAL_LONG, -1: SIGNAL_SHORT, 0: SIGNAL_NEUTRAL}
                    
                    strict_signal_str = signal_map.get(strict_signal, SIGNAL_NEUTRAL)
                    non_strict_signal_str = signal_map.get(non_strict_signal, SIGNAL_NEUTRAL)
                    
                    score = (self._calculate_signal_match_score(strict_signal_str, signal, "HMM Strict") +
                            self._calculate_signal_match_score(non_strict_signal_str, signal, "HMM Non-strict"))
                    
                    logger.signal(f"HMM total score for {symbol} {timeframe} {signal}: {score}")
                    return score
                    
            except Exception as e:
                logger.debug(f"HMM analysis failed: {e}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy HMM signal: {e}")
            return 0
    
    def get_transformer_signal_score(self, symbol: str, timeframe: str, signal: str) -> int:
        """Lấy tín hiệu từ Transformer model global"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            model_path = MODELS_DIR / "transformer_model_global.pth"
            if not model_path.exists():
                logger.warning("Không tìm thấy Transformer model global")
                return 0
            
            model_data = load_transformer_model(model_path)
            if not model_data[0]:
                logger.warning("Không thể load Transformer model global")
                return 0
            
            model, scaler, feature_cols, input_dim = model_data
            
            market_data = self._get_market_data_for_symbol(normalized_symbol, timeframe)
            if market_data is None or market_data.empty:
                logger.warning(f"Không có dữ liệu thị trường cho {symbol} {timeframe}")
                return 0
            
            predicted_signal = get_latest_transformer_signal(market_data, model, scaler, feature_cols)
            score = self._calculate_signal_match_score(predicted_signal, signal, "Transformer")
            
            logger.signal(f"Transformer score for {symbol} {timeframe} {signal}: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy Transformer signal: {e}")
            return 0
    
    def get_lstm_signal_score(self, symbol: str, timeframe: str, signal: str) -> int:
        """
        Lấy tín hiệu tổng hợp từ 4 variants của LSTM models và combine thành 1 signal cuối

        Args:
            symbol: Symbol trading
            timeframe: Khung thời gian  
            signal: Tín hiệu mong muốn (LONG/SHORT)

        Returns:
            Score tổng hợp từ 4 LSTM variants: từ -4 đến +4
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            market_data = self._get_market_data_for_symbol(normalized_symbol, timeframe)
            if market_data is None or market_data.empty:
                logger.warning(f"Không có dữ liệu thị trường cho {symbol} {timeframe}")
                return 0            # Define 4 LSTM model variants với cấu hình tương ứng 
            lstm_variants = {
                'LSTM': MODELS_DIR / "lstm_model_global.pth",
                'LSTM-Attention': MODELS_DIR / "lstm_attention_model_global.pth", 
                'CNN-LSTM': MODELS_DIR / "cnn_lstm_model_global.pth",
                'CNN-LSTM-Attention': MODELS_DIR / "cnn_lstm_attention_model_global.pth"
            }
            
            total_score = 0
            successful_predictions = 0
            
            # Get signals from each LSTM variant
            for variant_name, model_path in lstm_variants.items():
                try:
                    if not model_path.exists():
                        logger.debug(f"Model {variant_name} not found at {model_path}")
                        continue

                    loaded_data = load_cnn_lstm_attention_model(model_path)
                    if not loaded_data:
                        logger.debug(f"Cannot load {variant_name} model from {model_path}")
                        continue
                    
                    model, model_config, data_info, optimization_results = loaded_data

                    predicted_signal = get_latest_cnn_lstm_attention_signal(
                        df_input=market_data,
                        model=model,
                        model_config=model_config,
                        data_info=data_info,
                        optimization_results=optimization_results
                    )

                    variant_score = self._calculate_signal_match_score(predicted_signal, signal, variant_name.upper())
                    total_score += variant_score
                    successful_predictions += 1
                    
                    logger.debug(f"{variant_name.upper()}: {predicted_signal} -> score: {variant_score}")
                    
                except Exception as variant_error:
                    logger.debug(f"Error with {variant_name} variant: {variant_error}")
                    continue

            # Log combined results
            if successful_predictions > 0:
                logger.signal(f"LSTM Combined ({successful_predictions}/4 variants): {symbol} {timeframe} {signal} total_score: {total_score}")
            else:
                logger.warning(f"No LSTM variants available for {symbol} {timeframe}")
                
            return total_score

        except Exception as e:
            logger.error(f"Lỗi khi lấy tín hiệu LSTM tổng hợp: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _calculate_signal_match_score(self, predicted_signal: str, target_signal: str, model_name: str) -> int:
        """Tính điểm dựa trên sự khớp của signal"""
        if not predicted_signal or predicted_signal == SIGNAL_NEUTRAL:
            logger.debug(f"{model_name}: No signal or neutral: 0")
            return 0
            
        if predicted_signal.upper() == target_signal.upper():
            logger.debug(f"{model_name}: Signal match ({predicted_signal}): +1")
            return 1
        else:
            logger.debug(f"{model_name}: Signal conflict ({predicted_signal} vs {target_signal}): -1")
            return -1
    
    def apply_chain_filtering(
        self,
        candidates: List[Dict],
        signal_type: str,
        model1: str,
        model2: str
    ) -> List[Dict]:
        """
        Áp dụng chain filtering với 2 models đã chọn
        
        Args:
            candidates: Danh sách candidates từ best performance analysis
            signal_type: 'LONG' hoặc 'SHORT'
            model1: Model đầu tiên trong chain
            model2: Model thứ hai trong chain
            
        Returns:
            Danh sách candidates đã được lọc qua chain
        """
        filtered_candidates = []
        
        logger.analysis(f"Áp dụng chain filtering {model1} → {model2} cho {len(candidates)} {signal_type} candidates")
        
        for candidate in candidates:
            symbol = candidate.get('symbol', '')
            
            # Test trên tất cả timeframes có trong candidate
            timeframe_scores = candidate.get('timeframe_scores', {})
            passed_timeframes = {}
            
            for timeframe in timeframe_scores.keys():
                if timeframe not in self.valid_timeframes:
                    continue
                    
                # Áp dụng model 1
                score1 = self._get_model_score(symbol, timeframe, signal_type, model1)
                
                # Chỉ tiếp tục nếu model 1 cho kết quả tích cực
                if score1 > 0:
                    # Áp dụng model 2
                    score2 = self._get_model_score(symbol, timeframe, signal_type, model2)
                    
                    # Cả 2 models đều phải cho kết quả tích cực
                    if score2 > 0:
                        passed_timeframes[timeframe] = {
                            'original_score': timeframe_scores[timeframe],
                            'chain_score': score1 + score2,
                            f'{model1}_score': score1,
                            f'{model2}_score': score2
                        }
            
            # Chỉ giữ candidates có ít nhất 1 timeframe pass chain
            if passed_timeframes:
                filtered_candidate = candidate.copy()
                filtered_candidate['timeframe_scores'] = passed_timeframes
                filtered_candidate['chain_models'] = [model1, model2]
                filtered_candidates.append(filtered_candidate)
        
        logger.analysis(f"Chain filtering kết quả: {len(filtered_candidates)}/{len(candidates)} {signal_type} candidates passed")
        return filtered_candidates
    
    def _get_model_score(self, symbol: str, timeframe: str, signal: str, model: str) -> int:
        """Lấy điểm số từ một model cụ thể"""
        try:
            model_functions = {
                'random_forest': self.get_random_forest_signal_score,
                'hmm': self.get_hmm_signal_score,
                'transformer': self.get_transformer_signal_score,
                'lstm': self.get_lstm_signal_score
            }
            
            if model in model_functions:
                return model_functions[model](symbol, timeframe, signal)
            else:
                logger.warning(f"Unknown model: {model}")
                return 0
        except Exception as e:
            logger.error(f"Lỗi khi lấy score từ model {model}: {e}")
            return 0
    
    def run_full_market_scan(
        self,
        reload_model: bool,
        chain_model1: str,
        chain_model2: str
    ) -> Dict:
        """
        Chạy quét toàn bộ thị trường với chain filtering
        
        Args:
            reload_model: Có reload models không
            chain_model1: Model đầu tiên trong chain
            chain_model2: Model thứ hai trong chain
            
        Returns:
            Dictionary chứa kết quả LONG và SHORT sau chain filtering
        """
        try:
            logger.process("🔍 BẮT ĐẦU QUÉT TOÀN BỘ THỊ TRƯỜNG")
            
            # 1. Reload models if requested
            if reload_model:
                logger.process("Reload models được yêu cầu...")
                self.reload_all_models()
            
            # 2. BƯỚC 1: Analyze best performance symbols
            logger.analysis("📊 BƯỚC 1: Phân tích best performance symbols...")
            long_candidates, short_candidates = self.analyze_best_performance_signals()
            
            if not long_candidates and not short_candidates:
                return {
                    'error': 'Không tìm thấy candidates nào từ best performance analysis'
                }
            
            logger.analysis(f"✅ Tìm thấy {len(long_candidates)} LONG candidates, {len(short_candidates)} SHORT candidates")
            
            # 3. BƯỚC 2: Apply chain filtering
            logger.analysis(f"🔗 BƯỚC 2: Áp dụng chain filtering {chain_model1} → {chain_model2}")
            
            filtered_long_candidates = self.apply_chain_filtering(
                long_candidates, SIGNAL_LONG, chain_model1, chain_model2
            )
            
            filtered_short_candidates = self.apply_chain_filtering(
                short_candidates, SIGNAL_SHORT, chain_model1, chain_model2
            )
              # 4. BƯỚC 3: Prepare results
            result = {
                'success': True,
                'chain_models': [chain_model1, chain_model2],
                'original_candidates': {
                    'long_count': len(long_candidates),
                    'short_count': len(short_candidates)
                },
                'filtered_candidates': {
                    'long_candidates': filtered_long_candidates,
                    'short_candidates': filtered_short_candidates,
                    'long_count': len(filtered_long_candidates),
                    'short_count': len(filtered_short_candidates)
                },
                'filtering_efficiency': {
                    'long_pass_rate': len(filtered_long_candidates) / len(long_candidates) if long_candidates else 0,
                    'short_pass_rate': len(filtered_short_candidates) / len(short_candidates) if short_candidates else 0
                }
            }
            
            logger.success(f"🎯 HOÀN THÀNH: {len(filtered_long_candidates)} LONG, {len(filtered_short_candidates)} SHORT symbols passed chain filtering")
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình quét thị trường: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def run_comparison_analysis(
        self,
        reload_model: bool,
        chain_model1: str,
        chain_model2: str
    ) -> Dict:
        """
        Chạy phân tích so sánh giữa chain đã chọn và các combinations khác
        
        Args:
            reload_model: Có reload models không
            chain_model1: Model đầu tiên trong chain chính
            chain_model2: Model thứ hai trong chain chính
            
        Returns:
            Dictionary chứa kết quả so sánh tất cả combinations
        """
        try:
            logger.process("🔍 BẮT ĐẦU PHÂN TÍCH SO SÁNH CHAIN MODELS")
            
            # 1. Reload models if requested
            if reload_model:
                logger.process("Reload models được yêu cầu...")
                self.reload_all_models()
            
            # 2. Analyze best performance symbols một lần cho tất cả
            logger.analysis("📊 Phân tích best performance symbols...")
            long_candidates, short_candidates = self.analyze_best_performance_signals()
            
            if not long_candidates and not short_candidates:
                return {
                    'error': 'Không tìm thấy candidates nào từ best performance analysis'
                }
              # 3. Tạo tất cả combinations có thể từ available models
            available_models = list(self.available_models.keys())
            all_combinations = [(model1, model2) for i, model1 in enumerate(available_models) 
                               for j, model2 in enumerate(available_models) if i != j]
            
            logger.analysis(f"🔗 Sẽ test {len(all_combinations)} chain combinations")
            
            # 4. Chạy chain filtering cho tất cả combinations
            comparison_results = {}
            main_chain = f"{chain_model1}_{chain_model2}"
            
            for model1, model2 in all_combinations:
                chain_name = f"{model1}_{model2}"
                
                logger.analysis(f"Testing chain: {self.available_models[model1]} → {self.available_models[model2]}")
                
                filtered_long = self.apply_chain_filtering(long_candidates, SIGNAL_LONG, model1, model2)
                filtered_short = self.apply_chain_filtering(short_candidates, SIGNAL_SHORT, model1, model2)
                
                comparison_results[chain_name] = {
                    'models': [model1, model2],
                    'model_names': [self.available_models[model1], self.available_models[model2]],
                    'long_candidates': filtered_long,
                    'short_candidates': filtered_short,
                    'long_count': len(filtered_long),
                    'short_count': len(filtered_short),
                    'long_symbols': [c['symbol'] for c in filtered_long],
                    'short_symbols': [c['symbol'] for c in filtered_short],
                    'total_signals': len(filtered_long) + len(filtered_short),
                    'is_main_chain': chain_name == main_chain
                }
              # 5. Tính toán efficiency và rankings
            sorted_chains = sorted(comparison_results.items(), key=lambda x: x[1]['total_signals'], reverse=True)
            
            result = {
                'success': True,
                'main_chain': main_chain,
                'original_candidates': {
                    'long_count': len(long_candidates),
                    'short_count': len(short_candidates),
                    'long_symbols': [c['symbol'] for c in long_candidates],
                    'short_symbols': [c['symbol'] for c in short_candidates]
                },
                'comparison_results': comparison_results,
                'rankings': sorted_chains,
                'best_chain': sorted_chains[0][0] if sorted_chains else None,
                'main_chain_rank': next((i+1 for i, (chain, _) in enumerate(sorted_chains) if chain == main_chain), None)
            }
            
            logger.success(f"🎯 HOÀN THÀNH SO SÁNH: {len(all_combinations)} chains tested")
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình so sánh: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


def print_market_scan_results(result: Dict[str, Any]) -> None:
    """In kết quả quét thị trường với chain filtering"""
    
    if 'error' in result:
        logger.error(f"❌ Lỗi: {result['error']}")
        return
    
    logger.success("\n" + "="*80)
    logger.success("🌍 KẾT QUẢ QUÉT TOÀN BỘ THỊ TRƯỜNG VỚI CHAIN FILTERING")
    logger.success("="*80)
    
    chain_models = result['chain_models']
    logger.info(f"🔗 Chain Models: {chain_models[0]} → {chain_models[1]}")
    
    original = result['original_candidates']
    filtered = result['filtered_candidates']
    efficiency = result['filtering_efficiency']
    
    logger.info(f"\n📊 THỐNG KÊ TỔNG QUAN:")
    logger.info(f"  • Original LONG candidates:  {original['long_count']}")
    logger.info(f"  • Original SHORT candidates: {original['short_count']}")
    logger.info(f"  • Filtered LONG candidates:  {filtered['long_count']} ({efficiency['long_pass_rate']:.1%} pass rate)")
    logger.info(f"  • Filtered SHORT candidates: {filtered['short_count']} ({efficiency['short_pass_rate']:.1%} pass rate)")
    
    # LONG Results
    if filtered['long_candidates']:
        logger.info(f"\n📈 LONG SIGNALS ({len(filtered['long_candidates'])} symbols):")
        logger.info("-" * 80)
        
        long_symbols = [c['symbol'] for c in filtered['long_candidates']]
        symbols_str = ', '.join(long_symbols)
        wrapped_symbols = textwrap.fill(symbols_str, width=75, initial_indent='  ', subsequent_indent='  ')
        logger.info(wrapped_symbols)
        
        logger.info("\nTop 5 detailed:")
        for i, candidate in enumerate(filtered['long_candidates'][:5], 1):
            symbol = candidate['symbol']
            timeframes = list(candidate['timeframe_scores'].keys())
            chain_models_used = candidate.get('chain_models', [])
            
            logger.info(f"{i:2d}. {symbol:12s} | Timeframes: {', '.join(timeframes[:3])} | Chain: {' → '.join(chain_models_used)}")
            
            if timeframes:
                best_tf = timeframes[0]
                tf_data = candidate['timeframe_scores'][best_tf]
                logger.info(f"     └── {best_tf}: Chain Score: {tf_data['chain_score']}, Original: {tf_data['original_score']:.3f}")
    else:
        logger.info(f"\n📈 LONG SIGNALS: Không có symbols nào")
    
    # SHORT Results
    if filtered['short_candidates']:
        logger.info(f"\n📉 SHORT SIGNALS ({len(filtered['short_candidates'])} symbols):")
        logger.info("-" * 80)
        
        short_symbols = [c['symbol'] for c in filtered['short_candidates']]
        symbols_str = ', '.join(short_symbols)
        wrapped_symbols = textwrap.fill(symbols_str, width=75, initial_indent='  ', subsequent_indent='  ')
        logger.info(wrapped_symbols)
        
        logger.info("\nTop 5 detailed:")
        for i, candidate in enumerate(filtered['short_candidates'][:5], 1):
            symbol = candidate['symbol']
            timeframes = list(candidate['timeframe_scores'].keys())
            chain_models_used = candidate.get('chain_models', [])
            
            logger.info(f"{i:2d}. {symbol:12s} | Timeframes: {', '.join(timeframes[:3])} | Chain: {' → '.join(chain_models_used)}")
            
            if timeframes:
                best_tf = timeframes[0]
                tf_data = candidate['timeframe_scores'][best_tf]
                logger.info(f"     └── {best_tf}: Chain Score: {tf_data['chain_score']}, Original: {tf_data['original_score']:.3f}")
    else:
        logger.info(f"\n📉 SHORT SIGNALS: Không có symbols nào")
    
    logger.info("="*80)


def print_comparison_results(result: Dict[str, Any]) -> None:
    """In kết quả so sánh tất cả chain combinations"""
    
    if 'error' in result:
        logger.error(f"❌ Lỗi: {result['error']}")
        return
    
    logger.success("\n" + "="*100)
    logger.success("🏆 KẾT QUẢ SO SÁNH TẤT CẢ CHAIN COMBINATIONS")
    logger.success("="*100)
    
    main_chain = result['main_chain']
    main_chain_rank = result['main_chain_rank']
    
    logger.info(f"🎯 Main Chain: {main_chain.replace('_', ' → ')} (Rank: #{main_chain_rank})")
    
    original = result['original_candidates']
    logger.info(f"\n📊 ORIGINAL CANDIDATES:")
    logger.info(f"  • LONG candidates: {original['long_count']} symbols")
    logger.info(f"  • SHORT candidates: {original['short_count']} symbols")
    
    logger.info(f"\n🏅 RANKINGS BY TOTAL SIGNALS:")
    logger.info("-" * 100)
    logger.info(f"{'Rank':<6}{'Chain Models':<35}{'LONG':<8}{'SHORT':<8}{'Total':<8}{'Status':<15}")
    logger.info("-" * 100)
    for rank, (chain_name, data) in enumerate(result['rankings'], 1):
        model_names = ' → '.join(data['model_names'])
        status = "🎯 MAIN" if data['is_main_chain'] else ""
        
        logger.info(f"{rank:<6}{model_names:<35}{data['long_count']:<8}{data['short_count']:<8}"
              f"{data['total_signals']:<8}{status:<15}")
    
    main_chain_data = result['comparison_results'][main_chain]
    
    logger.info(f"\n🎯 MAIN CHAIN DETAILED RESULTS:")
    logger.info(f"Chain: {' → '.join(main_chain_data['model_names'])}")
    logger.info("-" * 100)
    
    if main_chain_data['long_symbols']:
        logger.info(f"\n📈 LONG SIGNALS ({len(main_chain_data['long_symbols'])} symbols):")
        long_symbols_str = ', '.join(main_chain_data['long_symbols'])
        wrapped_symbols = textwrap.fill(long_symbols_str, width=90, initial_indent='  ', subsequent_indent='  ')
        logger.info(wrapped_symbols)
    else:
        logger.info(f"\n📈 LONG SIGNALS: Không có symbols nào")
    
    if main_chain_data['short_symbols']:
        logger.info(f"\n📉 SHORT SIGNALS ({len(main_chain_data['short_symbols'])} symbols):")
        short_symbols_str = ', '.join(main_chain_data['short_symbols'])
        wrapped_symbols = textwrap.fill(short_symbols_str, width=90, initial_indent='  ', subsequent_indent='  ')
        logger.info(wrapped_symbols)
    else:
        logger.info(f"\n📉 SHORT SIGNALS: Không có symbols nào")
    
    logger.info(f"\n🏆 TOP 3 BEST PERFORMING CHAINS:")
    logger.info("-" * 100)    
    for rank, (chain_name, data) in enumerate(result['rankings'][:3], 1):
        model_names = ' → '.join(data['model_names'])
        logger.info(f"\n{rank}. {model_names} (Total: {data['total_signals']} signals)")
        
        if data['long_symbols']:
            long_preview = ', '.join(data['long_symbols'][:5])
            if len(data['long_symbols']) > 5:
                long_preview += f" ... (+{len(data['long_symbols'])-5} more)"
            logger.info(f"   📈 LONG ({data['long_count']}): {long_preview}")
        
        if data['short_symbols']:
            short_preview = ', '.join(data['short_symbols'][:5])
            if len(data['short_symbols']) > 5:
                short_preview += f" ... (+{len(data['short_symbols'])-5} more)"
            logger.info(f"   📉 SHORT ({data['short_count']}): {short_preview}")
    
    best_chain_name = result['best_chain']
    if best_chain_name and best_chain_name != main_chain:
        best_chain_data = result['comparison_results'][best_chain_name]
        improvement = best_chain_data['total_signals'] - main_chain_data['total_signals']
        
        logger.info(f"\n💡 PERFORMANCE ANALYSIS:")
        logger.info(f"  • Best chain: {' → '.join(best_chain_data['model_names'])} ({best_chain_data['total_signals']} signals)")
        logger.info(f"  • Your chain: {' → '.join(main_chain_data['model_names'])} ({main_chain_data['total_signals']} signals)")
        if main_chain_data['total_signals'] > 0:
            logger.info(f"  • Potential improvement: +{improvement} signals ({improvement/main_chain_data['total_signals']*100:.1f}%)")
    
    logger.info("="*100)


def main() -> None:
    """Hàm main với giao diện tương tác cho chain filtering"""
    
    analyzer = TradingSignalAnalyzer()

    logger.info("="*80)
    logger.info("🌍 CHƯƠNG TRÌNH QUÉT TOÀN BỘ THỊ TRƯỜNG - CHAIN FILTERING")
    logger.info("Quét tất cả symbols với 2 models chain filtering")
    logger.info("="*80)

    # Nhập reload_model
    while True:
        reload_input = input("❓ Bạn có muốn reload lại toàn bộ models không? (yes/no): ").lower()
        if reload_input in ['yes', 'y', 'no', 'n']:
            reload_model = reload_input in ['yes', 'y']
            break
        else:
            logger.error("   ❌ Lỗi: Vui lòng nhập 'yes' hoặc 'no'.")

    # Chọn chế độ
    logger.info(f"\n📊 CHỌN CHỂ ĐỘ PHÂN TÍCH:")
    logger.info("  • simple: Chỉ test chain bạn chọn")
    logger.info("  • compare: So sánh với tất cả combinations khác")
    
    while True:
        mode_input = input("\n❓ Chọn chế độ (simple/compare): ").lower()
        if mode_input in ['simple', 'compare']:
            break
        else:
            logger.error("   ❌ Lỗi: Vui lòng chọn 'simple' hoặc 'compare'.")

    # Chọn chain models
    logger.info(f"\n🔗 CHỌN 2 MODELS CHO CHAIN FILTERING:")
    logger.info("Available models:")
    for key, name in analyzer.available_models.items():
        logger.info(f"  • {key}: {name}")
    
    # Model 1
    while True:
        model1_input = input("\n❓ Chọn model đầu tiên: ").lower()
        if model1_input in analyzer.available_models:
            break
        else:
            logger.error(f"   ❌ Lỗi: Vui lòng chọn từ: {', '.join(analyzer.available_models.keys())}")
    
    # Model 2
    while True:
        model2_input = input("❓ Chọn model thứ hai: ").lower()
        if model2_input in analyzer.available_models:
            if model2_input != model1_input:
                break
            else:
                logger.error("   ❌ Lỗi: Vui lòng chọn model khác với model đầu tiên.")
        else:
            logger.error(f"   ❌ Lỗi: Vui lòng chọn từ: {', '.join(analyzer.available_models.keys())}")
    
    logger.info("="*80)
    logger.config(f"Bắt đầu quét thị trường với tham số:")
    logger.config(f"  - Reload Model: {reload_model}")
    logger.config(f"  - Mode: {mode_input}")
    logger.config(f"  - Chain Model 1: {analyzer.available_models[model1_input]}")
    logger.config(f"  - Chain Model 2: {analyzer.available_models[model2_input]}")
    logger.config(f"  - Total Symbols: {len(analyzer.valid_symbols)}")
    logger.config(f"  - Timeframes: {', '.join(analyzer.valid_timeframes)}")

    # Chạy phân tích theo mode
    if mode_input == 'compare':
        result = analyzer.run_comparison_analysis(
            reload_model=reload_model,
            chain_model1=model1_input,
            chain_model2=model2_input
        )
        print_comparison_results(result)
    else:
        result = analyzer.run_full_market_scan(
            reload_model=reload_model,
            chain_model1=model1_input,
            chain_model2=model2_input
        )
        print_market_scan_results(result)

if __name__ == "__main__":
    main()