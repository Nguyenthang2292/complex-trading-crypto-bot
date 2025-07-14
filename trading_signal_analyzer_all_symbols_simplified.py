import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from components.load_all_symbols_data import load_all_symbols_data
from components.tick_processor import TickProcessor
from components.combine_all_dataframes import combine_all_dataframes
from config.config import DEFAULT_TIMEFRAMES, SIGNAL_LONG, SIGNAL_SHORT
from signals.quant_models.random_forest.signals_random_forest import (
    get_latest_random_forest_signal, train_and_save_global_rf_model, load_random_forest_model
)
from signals.quant_models.best_performance_symbols.signals_best_performance_symbols import signal_best_performance_symbols
from signals.quant_models.hmm.signals_hmm import hmm_signals
from utilities._logger import setup_logging


logger = setup_logging(module_name="simplified_trading_signal_analyzer", log_level=20)

def normalize_columns(df):
    """Đảm bảo DataFrame có cả cột viết hoa và thường cho OHLCV."""
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns and col.upper() not in df.columns:
            df[col.upper()] = df[col]
        if col.upper() in df.columns and col not in df.columns:
            df[col] = df[col.upper()]
    return df

def clean_all_symbols_data(all_symbols_data, min_len=48):
    cleaned = {}
    for sym, tf_dict in all_symbols_data.items():
        if isinstance(tf_dict, dict):
            cleaned_tf = {tf: df for tf, df in tf_dict.items()
                          if isinstance(df, pd.DataFrame) and len(df) >= min_len}
            if cleaned_tf:
                cleaned[sym] = cleaned_tf
    return cleaned

def main():
    # Bước 1: Khởi tạo và nạp dữ liệu
    logger.process("Bước 1: Khởi tạo processor và nạp dữ liệu lịch sử")
    processor = TickProcessor(trade_open_callback=None, trade_close_callback=None)
    symbols = processor.get_symbols_list_by_quote_usdt()
    logger.info(f"Tổng số cặp USDT: {len(symbols)}")
    all_symbols_data = load_all_symbols_data(processor, symbols, timeframes=DEFAULT_TIMEFRAMES)
    if all_symbols_data is not None:
        for sym, tf_dict in all_symbols_data.items():
            if tf_dict is not None and isinstance(tf_dict, dict):
                for tf, df in tf_dict.items():
                    if isinstance(tf, str) and isinstance(df, pd.DataFrame):
                        tf_dict[tf] = normalize_columns(df)
        all_symbols_data = clean_all_symbols_data(all_symbols_data, min_len=48)
        logger.success("Nạp dữ liệu xong.")

    # Bước 2: Phân tích hiệu suất các cặp sử dụng signal_best_performance_symbols
    logger.analysis("Bước 2: Phân tích hiệu suất các cặp bằng signal_best_performance_symbols")
    perf_result = signal_best_performance_symbols(
        processor=processor,
        symbols=symbols,
        timeframes=DEFAULT_TIMEFRAMES,
        performance_period=48,
        top_percentage=0.2,
        include_short_signals=True,
        worst_percentage=0.2,
        min_volume_usdt=1000000,
        exclude_stable_coins=True,
        preloaded_data=all_symbols_data
    )
    top_long = perf_result.get('best_performers_long', [])
    top_short = perf_result.get('worst_performers_short', [])
    logger.info(f"Top LONG: {[item['symbol'] for item in top_long]}")
    logger.info(f"Top SHORT: {[item['symbol'] for item in top_short]}")

    # Lọc lại dữ liệu chỉ giữ các cặp tốt nhất/kém nhất
    long_symbols = set([item['symbol'] for item in top_long])
    short_symbols = set([item['symbol'] for item in top_short])
    filtered_data_long = {s: all_symbols_data[s] for s in long_symbols if s in all_symbols_data}
    filtered_data_short = {s: all_symbols_data[s] for s in short_symbols if s in all_symbols_data}

    # Bước 3: Kết hợp RF và HMM
    logger.process("Bước 3: Huấn luyện và sinh tín hiệu Random Forest")
    combined_df = combine_all_dataframes(all_symbols_data)
    rf_model, rf_model_path = train_and_save_global_rf_model(combined_df, model_filename="rf_model_global.joblib")
    rf_model = load_random_forest_model(Path(rf_model_path))
    rf_signals_long, rf_signals_short = [], []

    if rf_model is not None:
        for sym, tf_dict in filtered_data_long.items():
            for tf, df in tf_dict.items():
                signal = get_latest_random_forest_signal(df, rf_model)
                if signal == SIGNAL_LONG:
                    rf_signals_long.append({'symbol': sym, 'timeframe': tf, 'rf_signal': signal})
        for sym, tf_dict in filtered_data_short.items():
            for tf, df in tf_dict.items():
                signal = get_latest_random_forest_signal(df, rf_model)
                if signal == SIGNAL_SHORT:
                    rf_signals_short.append({'symbol': sym, 'timeframe': tf, 'rf_signal': signal})
    else:
        logger.error("Không thể load mô hình Random Forest. Bỏ qua bước sinh tín hiệu RF.")

    logger.info(f"Số cặp LONG qua RF: {len(rf_signals_long)}")
    logger.info(f"Số cặp SHORT qua RF: {len(rf_signals_short)}")

    # Chạy HMM trên các cặp đã lọc
    logger.process("Chạy HMM trên các cặp đã lọc")
    hmm_signals_long, hmm_signals_short = [], []
    for entry in rf_signals_long:
        df = all_symbols_data[entry['symbol']][entry['timeframe']]
        hmm_result = hmm_signals(df)
        if hmm_result[0] == 1 or hmm_result[1] == 1:  # strict hoặc non-strict LONG
            hmm_signals_long.append(entry)
    for entry in rf_signals_short:
        df = all_symbols_data[entry['symbol']][entry['timeframe']]
        hmm_result = hmm_signals(df)
        if hmm_result[0] == -1 or hmm_result[1] == -1:  # strict hoặc non-strict SHORT
            hmm_signals_short.append(entry)

    logger.info(f"Số cặp LONG đồng thuận RF+HMM: {len(hmm_signals_long)}")
    logger.info(f"Số cặp SHORT đồng thuận RF+HMM: {len(hmm_signals_short)}")

    # Bước 4: Lọc tín hiệu cuối cùng
    logger.process("Bước 4: Lọc tín hiệu cuối cùng và chuẩn hóa format")
    final_long = [
        {'pair': e['symbol'], 'direction': 'LONG', 'timeframe': e['timeframe']}
        for e in hmm_signals_long if e['symbol'] in long_symbols
    ]
    final_short = [
        {'pair': e['symbol'], 'direction': 'SHORT', 'timeframe': e['timeframe']}
        for e in hmm_signals_short if e['symbol'] in short_symbols
    ]
    logger.info(f"Tín hiệu LONG cuối cùng: {final_long}")
    logger.info(f"Tín hiệu SHORT cuối cùng: {final_short}")

    # Bước 5: Logging & Debug
    logger.process("Bước 5: Debug nâng cao")
    logger.info(f"Số lượng cặp trùng nhau giữa các tầng lọc LONG: {len(set([e['pair'] for e in final_long]))}")
    logger.info(f"Số lượng cặp trùng nhau giữa các tầng lọc SHORT: {len(set([e['pair'] for e in final_short]))}")

if __name__ == "__main__":
    main() 