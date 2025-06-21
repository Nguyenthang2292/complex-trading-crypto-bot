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

from components._components._load_all_symbols_data import load_all_symbols_data
from components._components._tick_processor import tick_processor
from components._components._combine_all_dataframes import combine_all_dataframes
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
    """Phân tích tín hiệu giao dịch từ nhiều model ML sử dụng global models"""
    
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
        """Load lại toàn bộ dữ liệu và train models global"""
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
                ("LSTM", "lstm_model_global.pth", False, False),
                ("LSTM-Attention", "lstm_attention_model_global.pth", False, True),
                ("CNN-LSTM", "cnn_lstm_model_global.pth", True, False),
                ("CNN-LSTM-Attention", "cnn_lstm_attention_model_global.pth", True, True)
            ]
            
            for i, config in enumerate(model_configs):
                if i < 2:
                    model_name, train_func = config
                    logger.model(f"Training {model_name} model...")
                    try:
                        model, model_path = train_func()
                        if model:
                            logger.success(f"{model_name} model saved: {model_path}")
                        else:
                            logger.warning(f"{model_name} training failed")
                    except Exception as e:
                        logger.error(f"Lỗi khi train {model_name} model: {e}")
                else:
                    model_name, filename, use_cnn, use_attention = config
                    logger.model(f"Training {model_name} model...")
                    try:
                        lstm_model, lstm_path = train_cnn_lstm_attention_model(
                            combined_df,
                            model_filename=filename,
                            use_cnn=use_cnn,
                            use_attention=use_attention
                        )
                        if lstm_model:
                            logger.success(f"{model_name} model saved: {lstm_path}")
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
        """Chạy phân tích best performance symbols"""
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
    
    def check_symbol_in_performance_list(
        self, 
        symbol: str, 
        timeframe: str, 
        signal: str,
        long_candidates: List[Dict[str, Any]],
        short_candidates: List[Dict[str, Any]]
    ) -> int:
        """
        Kiểm tra symbol có trong danh sách top/worst performers không
        
        Returns:
            +1: Nằm trong top performers (LONG)
            -1: Nằm trong worst performers (SHORT)  
             0: Không nằm trong danh sách nào
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            for candidate in long_candidates:
                if (candidate.get('symbol', '').upper() == normalized_symbol and 
                    timeframe in candidate.get('timeframe_scores', {})):
                    if signal.upper() == SIGNAL_LONG:
                        logger.analysis(f"{symbol} {timeframe} {signal} nằm trong top performers: +1")
                        return 1
            
            for candidate in short_candidates:
                if (candidate.get('symbol', '').upper() == normalized_symbol and
                    timeframe in candidate.get('timeframe_scores', {})):
                    if signal.upper() == SIGNAL_SHORT:
                        logger.analysis(f"{symbol} {timeframe} {signal} nằm trong worst performers: +1") 
                        return 1
                    elif signal.upper() == SIGNAL_LONG:
                        logger.analysis(f"{symbol} {timeframe} {signal} nằm trong worst performers: -1")
                        return -1
            
            logger.analysis(f"{symbol} {timeframe} {signal} không nằm trong danh sách nào: 0")
            return 0
            
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra performance list: {e}")
            return 0
    
    def _get_market_data_for_symbol(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu thị trường cho symbol và timeframe"""
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
        """Lấy tín hiệu từ Random Forest model global"""
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
            return score
                
        except Exception as e:
            logger.error(f"Lỗi khi lấy Random Forest signal: {e}")
            return 0
    
    def get_hmm_signal_score(self, symbol: str, timeframe: str, signal: str) -> int:
        """Lấy tín hiệu từ HMM model (cả strict và non-strict mode)"""
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
                return 0

            lstm_variants = {
                'LSTM': MODELS_DIR / "lstm_model_global.pth",
                'LSTM-Attention': MODELS_DIR / "lstm_attention_model_global.pth", 
                'CNN-LSTM': MODELS_DIR / "cnn_lstm_model_global.pth",
                'CNN-LSTM-Attention': MODELS_DIR / "cnn_lstm_attention_model_global.pth"
            }
            
            total_score = 0
            successful_predictions = 0
            
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
    
    def calculate_final_threshold(self, total_score: int, max_possible_score: int) -> float:
        """Tính threshold cuối cùng"""
        if max_possible_score == 0:
            return 0.0
        
        normalized_score = (total_score + max_possible_score) / (2 * max_possible_score)
        return max(0.0, min(1.0, normalized_score))
    
    def _calculate_max_possible_score(self) -> int:
        """
        Tính toán max_possible_score dựa trên các models được sử dụng
        
        Returns:
            int: Tổng điểm tối đa có thể đạt được từ tất cả models
        """
        model_max_scores = {
            'performance': 1,      # Best performance analysis: +1 or -1
            'random_forest': 1,    # Random Forest: +1 or -1 
            'hmm': 2,             # HMM: +2 (strict +1, non-strict +1) or -2
            'transformer': 1,      # Transformer: +1 or -1
            'lstm': 4             # LSTM: +4 (4 variants × +1 each) or -4
        }
        
        # Tính tổng max score từ tất cả models được sử dụng
        total_max_score = sum(model_max_scores.values())
        
        logger.debug(f"Dynamic max possible score calculation:")
        for model, max_score in model_max_scores.items():
            logger.debug(f"  - {model}: {max_score}")
        logger.debug(f"  - Total: {total_max_score}")
        
        return total_max_score
    
    def analyze_symbol_signal(
        self, 
        symbol: str, 
        timeframe: str, 
        signal: str,
        long_candidates: List[Dict[str, Any]],
        short_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Union[str, int, float, Dict[str, int]]]:
        """Phân tích tổng hợp tín hiệu cho một symbol"""
        
        logger.analysis(f"Bắt đầu phân tích {symbol} {timeframe} {signal}")
        
        scores = {
            'performance': self.check_symbol_in_performance_list(symbol, timeframe, signal, long_candidates, short_candidates),
            'random_forest': self.get_random_forest_signal_score(symbol, timeframe, signal),
            'hmm': self.get_hmm_signal_score(symbol, timeframe, signal),
            'transformer': self.get_transformer_signal_score(symbol, timeframe, signal),
            'lstm': self.get_lstm_signal_score(symbol, timeframe, signal)
        }        
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
        """Chạy phân tích chính"""
        
        try:
            if not self.validate_symbol(symbol):
                return {
                    'error': f"Symbol '{symbol}' không hợp lệ. Danh sách hợp lệ: {self.valid_symbols}"
                }
            
            if timeframe not in self.valid_timeframes:
                return {
                    'error': f"Timeframe '{timeframe}' không hợp lệ. Danh sách hợp lệ: {self.valid_timeframes}"
                }
            
            if signal.upper() not in [SIGNAL_LONG, SIGNAL_SHORT]:
                return {
                    'error': f"Signal '{signal}' không hợp lệ. Chỉ chấp nhận: {SIGNAL_LONG}, {SIGNAL_SHORT}"
                }
            
            if reload_model:
                logger.process("Reload model được yêu cầu, bắt đầu reload...")
                self.reload_all_models()
            
            long_candidates, short_candidates = self.analyze_best_performance_signals()
            
            result = self.analyze_symbol_signal(
                symbol, timeframe, signal, long_candidates, short_candidates
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình phân tích: {e}")
            return {'error': str(e)}


def print_analysis_result(result: Dict[str, Any]) -> None:
    """In kết quả phân tích với màu sắc"""
    
    if 'error' in result:
        print(f"❌ Lỗi: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("🔍 KẾT QUẢ PHÂN TÍCH TÍN HIỆU GIAO DỊCH")
    print("="*60)
    
    print(f"📊 Symbol: {result['symbol']}")
    print(f"⏰ Timeframe: {result['timeframe']}")
    print(f"📈 Signal: {result['signal']}")
    
    print(f"\n📋 CHI TIẾT ĐIỂM SỐ:")
    scores = result['scores']
    print(f"  • Best Performance:  {scores['performance']:+2d}")
    print(f"  • Random Forest:     {scores['random_forest']:+2d}") 
    print(f"  • HMM (Strict+Non):  {scores['hmm']:+2d}")
    print(f"  • Transformer:       {scores['transformer']:+2d}")
    print(f"  • LSTM Combined:     {scores['lstm']:+2d} (4 variants)")
    
    print(f"\n🎯 TỔNG KẾT:")
    print(f"  • Tổng điểm:         {result['total_score']:+2d}/{result['max_possible_score']}")
    print(f"  • Threshold:         {result['threshold']:.3f}")
    
    recommendation = result['recommendation']
    if recommendation == 'ENTER':
        print(f"  • Khuyến nghị:       ✅ {recommendation} (Threshold ≥ 0.7)")
    else:
        print(f"  • Khuyến nghị:       ⏳ {recommendation} (Threshold < 0.7)")
    
    print("="*60)


def main() -> None:
    """Hàm main với input tương tác từ người dùng"""
    
    analyzer = TradingSignalAnalyzer()

    print("="*60)
    print("CHƯƠNG TRÌNH PHÂN TÍCH TÍN HIỆU GIAO DỊCH (REFACTORED)")
    print("Sử dụng Global Models được train từ tất cả symbols và timeframes")
    print("="*60)

    while True:
        reload_input = input("❓ Bạn có muốn reload lại toàn bộ models không? (yes/no): ").lower()
        if reload_input in ['yes', 'y', 'no', 'n']:
            reload_model = reload_input in ['yes', 'y']
            break
        else:
            print("   ❌ Lỗi: Vui lòng nhập 'yes' hoặc 'no'.")

    while True:
        symbol_input = input("❓ Nhập symbol cần kiểm tra (ví dụ: BTC-USDT): ").upper()
        if analyzer.validate_symbol(symbol_input):
            break
        else:
            print(f"   ❌ Lỗi: Symbol '{symbol_input}' không hợp lệ. Vui lòng thử lại.")

    while True:
        timeframe_input = input(f"❓ Nhập timeframe ({", ".join(analyzer.valid_timeframes)}): ").lower()
        if timeframe_input in analyzer.valid_timeframes:
            break
        else:
            print(f"   ❌ Lỗi: Timeframe '{timeframe_input}' không hợp lệ. Vui lòng chọn từ danh sách.")

    while True:
        signal_input = input("❓ Nhập signal cần kiểm tra (LONG/SHORT): ").upper()
        if signal_input in [SIGNAL_LONG, SIGNAL_SHORT]:
            break
        else:
            print("   ❌ Lỗi: Vui lòng nhập 'LONG' hoặc 'SHORT'.")

    print("="*60)
    logger.config(f"Bắt đầu phân tích với tham số:")
    logger.config(f"  - Reload Model: {reload_model}")
    logger.config(f"  - Symbol: {symbol_input}")
    logger.config(f"  - Timeframe: {timeframe_input}")
    logger.config(f"  - Signal: {signal_input}")

    result = analyzer.run_analysis(
        reload_model=reload_model,
        symbol=symbol_input,
        timeframe=timeframe_input,
        signal=signal_input
    )

    print_analysis_result(result)

if __name__ == "__main__":
    main()