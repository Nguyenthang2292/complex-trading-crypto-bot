import os
import sys
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm

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
from utilities.logger import setup_logging


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
        logger.info(f"Số lượng symbols sau clean_all_symbols_data: {len(all_symbols_data)}")
        for sym, tf_dict in all_symbols_data.items():
            logger.info(f"Symbol: {sym}, số timeframe: {len(tf_dict)}, timeframes: {list(tf_dict.keys())}")
    else:
        logger.error("Không load được all_symbols_data!")

    # Bước 2: Phân tích hiệu suất các cặp sử dụng signal_best_performance_symbols
    logger.analysis("Bước 2: Phân tích hiệu suất các cặp bằng signal_best_performance_symbols")
    perf_result = signal_best_performance_symbols(
        processor=processor,
        symbols=symbols,
        timeframes=DEFAULT_TIMEFRAMES,
        performance_period=48,  # Tăng từ 24 lên 48 để có nhiều dữ liệu hơn
        top_percentage=0.3,     # Giảm từ 0.4 xuống 0.3 để lọc chặt hơn
        include_short_signals=True,
        worst_percentage=0.3,   # Giảm từ 0.4 xuống 0.3 để lọc chặt hơn
        min_volume_usdt=500000, # Tăng từ 100k lên 500k để đảm bảo thanh khoản
        exclude_stable_coins=True,
        preloaded_data=all_symbols_data
    )
    top_long = perf_result.get('best_performers_long', [])
    top_short = perf_result.get('worst_performers_short', [])
    logger.info(f"Số lượng best_performers_long: {len(top_long)}")
    logger.info(f"Số lượng worst_performers_short: {len(top_short)}")
    logger.info(f"Top LONG: {[item['symbol'] for item in top_long]}")
    logger.info(f"Top SHORT: {[item['symbol'] for item in top_short]}")

    # Lọc lại dữ liệu chỉ giữ các cặp tốt nhất/kém nhất
    long_symbols = set([item['symbol'] for item in top_long])
    short_symbols = set([item['symbol'] for item in top_short])
    filtered_data_long = {s: all_symbols_data[s] for s in long_symbols if s in all_symbols_data}
    filtered_data_short = {s: all_symbols_data[s] for s in short_symbols if s in all_symbols_data}
    logger.info(f"Số lượng filtered_data_long: {len(filtered_data_long)}")
    logger.info(f"Số lượng filtered_data_short: {len(filtered_data_short)}")

    # Bước 3: Kết hợp RF và HMM
    logger.process("Bước 3: Huấn luyện và sinh tín hiệu Random Forest")
    # Xóa các file rf_model_global* trong thư mục models trước khi train lại
    models_dir = Path(__file__).parent / "models"
    rf_model_files = glob.glob(str(models_dir / "rf_model_global*"))
    if rf_model_files:
        for f in rf_model_files:
            try:
                os.remove(f)
                logger.info(f"Đã xóa file model cũ: {f}")
            except Exception as e:
                logger.error(f"Không thể xóa file model: {f}, lỗi: {e}")
    else:
        logger.info("Không có file rf_model_global cũ nào để xóa.")

    combined_df = combine_all_dataframes(all_symbols_data)
    logger.info(f"Số lượng dòng combined_df: {len(combined_df) if combined_df is not None else 0}")
    rf_model, rf_model_path = train_and_save_global_rf_model(combined_df, model_filename="rf_model_global.joblib")
    rf_model = load_random_forest_model(Path(rf_model_path))
    rf_signals_long, rf_signals_short = [], []

    if rf_model is not None:
        # Lọc LONG
        for item in top_long:
            sym = item['symbol']
            long_score = item.get('long_score', 0)
            if sym in filtered_data_long:
                for tf, df in filtered_data_long[sym].items():
                    rf_signal, rf_confidence = get_latest_random_forest_signal(df, rf_model)
                    if rf_signal == SIGNAL_LONG:
                        rf_signals_long.append({
                            'symbol': sym,
                            'timeframe': tf,
                            'rf_signal': rf_signal,
                            'confidence_rf': rf_confidence,
                            'long_score': long_score
                        })
        # Lọc SHORT
        for item in top_short:
            sym = item['symbol']
            short_score = item.get('short_score', 0)
            if sym in filtered_data_short:
                for tf, df in filtered_data_short[sym].items():
                    rf_signal, rf_confidence = get_latest_random_forest_signal(df, rf_model)
                    if rf_signal == SIGNAL_SHORT:
                        rf_signals_short.append({
                            'symbol': sym,
                            'timeframe': tf,
                            'rf_signal': rf_signal,
                            'confidence_rf': rf_confidence,
                            'short_score': short_score
                        })
    else:
        logger.error("Không thể load mô hình Random Forest. Bỏ qua bước sinh tín hiệu RF.")

    logger.info(f"rf_signals_long: {rf_signals_long}")
    logger.info(f"rf_signals_short: {rf_signals_short}")
    logger.info(f"Số cặp LONG qua RF: {len(rf_signals_long)}")
    logger.info(f"Số cặp SHORT qua RF: {len(rf_signals_short)}")

    # Chạy HMM trên các cặp đã lọc (bỏ qua confidence HMM)
    logger.process("Chạy HMM trên các cặp đã lọc")
    hmm_signals_long, hmm_signals_short = [], []
    for entry in tqdm(rf_signals_long, desc="HMM LONG", ncols=80):
        df = all_symbols_data[entry['symbol']][entry['timeframe']]
        logger.debug(f"Processing HMM LONG - {entry['symbol']} {entry['timeframe']}: shape={df.shape}, index_type={type(df.index)}")
        hmm_result = hmm_signals(df)
        logger.info(f"HMM signal LONG - Symbol: {entry['symbol']}, TF: {entry['timeframe']}, HMM result: {hmm_result}")
        if hmm_result[0] == 1 or hmm_result[1] == 1:  # strict hoặc non-strict LONG
            hmm_signals_long.append(entry)
    for entry in tqdm(rf_signals_short, desc="HMM SHORT", ncols=80):
        df = all_symbols_data[entry['symbol']][entry['timeframe']]
        logger.debug(f"Processing HMM SHORT - {entry['symbol']} {entry['timeframe']}: shape={df.shape}, index_type={type(df.index)}")
        hmm_result = hmm_signals(df)
        logger.info(f"HMM signal SHORT - Symbol: {entry['symbol']}, TF: {entry['timeframe']}, HMM result: {hmm_result}")
        if hmm_result[0] == -1 or hmm_result[1] == -1:  # strict hoặc non-strict SHORT
            hmm_signals_short.append(entry)

    logger.info(f"hmm_signals_long: {hmm_signals_long}")
    logger.info(f"hmm_signals_short: {hmm_signals_short}")
    logger.info(f"Số cặp LONG đồng thuận RF+HMM: {len(hmm_signals_long)}")
    logger.info(f"Số cặp SHORT đồng thuận RF+HMM: {len(hmm_signals_short)}")

    # Bước 4: Lọc tín hiệu cuối cùng
    logger.process("Bước 4: Lọc tín hiệu cuối cùng và chuẩn hóa format")
    final_long = [
        {
            'pair': e['symbol'],
            'direction': 'LONG',
            'timeframe': e['timeframe'],
            'confidence_rf': e['confidence_rf'],
            'long_score': e['long_score'],
            'confidence_total': (e['confidence_rf'] + e['long_score']) / 2
        }
        for e in hmm_signals_long if e['symbol'] in long_symbols
    ]
    final_short = [
        {
            'pair': e['symbol'],
            'direction': 'SHORT',
            'timeframe': e['timeframe'],
            'confidence_rf': e['confidence_rf'],
            'short_score': e['short_score'],
            'confidence_total': (e['confidence_rf'] + e['short_score']) / 2
        }
        for e in hmm_signals_short if e['symbol'] in short_symbols
    ]

    # Sắp xếp theo confidence tổng hợp giảm dần
    final_long_sorted = sorted(final_long, key=lambda x: x.get('confidence_total', 0.0), reverse=True)
    final_short_sorted = sorted(final_short, key=lambda x: x.get('confidence_total', 0.0), reverse=True)

    logger.info("Top 10 tín hiệu LONG cuối cùng:")
    if final_long_sorted:
        for i, entry in enumerate(final_long_sorted[:10], 1):
            logger.info(
                f"  {i:2d}. 🟢 {entry['pair']:12s} | Direction: {entry['direction']:<5s} | TF: {entry['timeframe']:<2s} | "
                f"RF: {entry['confidence_rf']:.2%} | Perf: {entry['long_score']:.2%} | Tổng hợp: {entry['confidence_total']:.2%}"
            )
    else:
        logger.info("  Không có tín hiệu LONG nào.")

    logger.info("Top 10 tín hiệu SHORT cuối cùng:")
    if final_short_sorted:
        for i, entry in enumerate(final_short_sorted[:10], 1):
            logger.info(
                f"  {i:2d}. 🔴 {entry['pair']:12s} | Direction: {entry['direction']:<5s} | TF: {entry['timeframe']:<2s} | "
                f"RF: {entry['confidence_rf']:.2%} | Perf: {entry['short_score']:.2%} | Tổng hợp: {entry['confidence_total']:.2%}"
            )
    else:
        logger.info("  Không có tín hiệu SHORT nào.")

    # Bước 5: Logging & Debug
    logger.process("Bước 5: Debug nâng cao")
    logger.info(f"Số lượng cặp trùng nhau giữa các tầng lọc LONG: {len(set([e['pair'] for e in final_long]))}")
    logger.info(f"Số lượng cặp trùng nhau giữa các tầng lọc SHORT: {len(set([e['pair'] for e in final_short]))}")

if __name__ == "__main__":
    main()