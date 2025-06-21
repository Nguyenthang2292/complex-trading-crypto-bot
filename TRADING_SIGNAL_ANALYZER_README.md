# Trading Signal Analyzer

Ứng dụng phân tích tín hiệu giao dịch crypto từ nhiều mô hình Machine Learning và đưa ra khuyến nghị.

## 🔧 Tính năng chính

### Đầu vào
- `--reload_model`: True/False - Reload toàn bộ models
- `--symbol`: Symbol cần kiểm tra (ví dụ: BTC-USDT, ETH-USDT)
- `--timeframe`: Timeframe (ví dụ: 1h, 4h, 1d)
- `--signal`: Signal cần kiểm tra (LONG/SHORT)

### Workflow
1. **Validate inputs**: Kiểm tra symbol, timeframe, signal hợp lệ
2. **Reload models** (nếu được yêu cầu): Xóa models cũ và train lại
3. **Best Performance Analysis**: Phân tích performance trên toàn bộ symbols
4. **Multi-model scoring**: Tính điểm từ 6 nguồn khác nhau
5. **Final recommendation**: Threshold ≥ 0.7 thì khuyến nghị vào lệnh

### Hệ thống tính điểm
- **Best Performance**: +1/-1/0 (top/worst performers)
- **Random Forest**: +1/-1/0 (signal match/conflict/neutral)
- **HMM**: +2/-2/0 (strict + non-strict modes)  
- **Transformer**: +1/-1/0 (signal match/conflict/neutral)
- **LSTM**: +4/-4/0 (4 scenarios: LSTM, LSTM Attention, CNN LSTM, CNN LSTM Attention)

**Tổng điểm tối đa**: 9 điểm

## 📖 Cách sử dụng

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy phân tích
```bash
# Reload models và phân tích BTC-USDT LONG 1h
python trading_signal_analyzer.py --reload_model True --symbol BTC-USDT --timeframe 1h --signal LONG

# Phân tích nhanh không reload models
python trading_signal_analyzer.py --reload_model False --symbol ETH-USDT --timeframe 4h --signal SHORT
```

### Test cơ bản
```bash
python test_analyzer.py
```

## 📊 Output mẫu

```
🔍 KẾT QUẢ PHÂN TÍCH TÍN HIỆU GIAO DỊCH
=============================================================
📊 Symbol: BTCUSDT
⏰ Timeframe: 1h
📈 Signal: LONG

📋 CHI TIẾT ĐIỂM SỐ:
  • Best Performance:  +1
  • Random Forest:     +1  
  • HMM (Strict+Non):  +2
  • Transformer:       +1
  • LSTM (4 models):   +3

🎯 TỔNG KẾT:
  • Tổng điểm:         +8/9
  • Threshold:         0.944
  • Khuyến nghị:       ✅ ENTER (Threshold ≥ 0.7)
```

## 🚨 Lưu ý quan trọng

1. **Dependencies**: App cần các file signals và models được cài đặt đúng
2. **Function signatures**: Có thể cần điều chỉnh import paths và function signatures
3. **Training time**: Reload models có thể mất thời gian lâu
4. **Data requirement**: Cần dữ liệu đầy đủ cho tất cả timeframes

## 🔧 Cấu trúc code

```
trading_signal_analyzer.py      # Main application
├── TradingSignalAnalyzer      # Main class
├── validate_symbol()          # Kiểm tra symbol hợp lệ
├── reload_all_models()        # Reload và train models
├── analyze_best_performance_signals()  # Phân tích performance
├── get_*_signal_score()       # Các hàm tính điểm từ models
├── analyze_symbol_signal()    # Phân tích tổng hợp
└── run_analysis()            # Workflow chính
```

## 🐛 Troubleshooting

**Lỗi import**: Kiểm tra các file signals có tồn tại và function signatures đúng

**Lỗi models**: Đảm bảo thư mục models có quyền write và đủ dung lượng

**Lỗi dữ liệu**: Kiểm tra kết nối internet và API Binance

**Lỗi memory**: Giảm số lượng symbols hoặc timeframes nếu thiếu RAM
