# Tài Liệu Hàm Crypto Trading Bot - tick_processor.py

## Tổng Quan

File `_tick_processor.py` là thành phần chính để xử lý giao dịch cryptocurrency, tương tác với Binance API và quản lý đơn hàng, dữ liệu tài khoản và thông tin thị trường.

## Danh Sách Các Hàm

### 1. `__init__(self, trade_open_callback, trade_close_callback)`
**Chức năng:** Khởi tạo class tick_processor với Binance client và các callback functions

**Đầu vào:**
- `trade_open_callback`: Function được gọi khi có đơn hàng mới mở
- `trade_close_callback`: Function được gọi khi có đơn hàng đóng

**Đầu ra:** Không có (constructor)

**Mô tả:** Thiết lập kết nối Binance API, logging, cache exchange info và các biến trạng thái của hệ thống.

---

### 2. `_validate_parameters(self, **kwargs)`
**Chức năng:** Validate các tham số quan trọng cho giao dịch

**Đầu vào:**
- `**kwargs`: Các cặp key-value cần validate (symbol, order_id, quantity, price, quote_quantity)

**Đầu ra:**
- `bool`: True nếu tất cả tham số hợp lệ, False nếu không

**Mô tả:** Kiểm tra tính hợp lệ của symbol, order_id, quantity, price và quote_quantity để đảm bảo an toàn giao dịch.

---

### 3. `_cancel_orders_batch(self, symbol, order_ids, max_retries=3)`
**Chức năng:** Hủy nhiều đơn hàng cùng lúc với cơ chế retry

**Đầu vào:**
- `symbol` (str): Symbol giao dịch
- `order_ids` (List[int]): Danh sách order ID cần hủy
- `max_retries` (int): Số lần retry tối đa (mặc định 3)

**Đầu ra:**
- `dict`: Kết quả hủy đơn với thông tin chi tiết (cancelled_orders, failed_orders, totals)

**Mô tả:** Hủy batch orders với error handling và retry logic, tránh rate limits.

---

### 4. `_klines_to_dataframe(self, klines_data)`
**Chức năng:** Chuyển đổi dữ liệu klines từ Binance thành DataFrame

**Đầu vào:**
- `klines_data` (list): Dữ liệu klines từ Binance API

**Đầu ra:**
- `pandas.DataFrame`: DataFrame đã format hoặc None nếu lỗi

**Mô tả:** Format dữ liệu OHLCV từ Binance thành pandas DataFrame với datetime index.

---

### 5. `get_account_info(self)`
**Chức năng:** Lấy thông tin tài khoản từ Binance

**Đầu vào:** Không có

**Đầu ra:**
- `dict`: Thông tin tài khoản hoặc None nếu lỗi

**Mô tả:** Truy xuất thông tin tài khoản bao gồm balances, fees, trading status.

---

### 6. `get_exchange_info_cached(self, force_refresh=False)`
**Chức năng:** Lấy thông tin exchange với caching 24 giờ

**Đầu vào:**
- `force_refresh` (bool): Buộc refresh cache (mặc định False)

**Đầu ra:**
- `Optional[Dict]`: Thông tin exchange hoặc None nếu lỗi

**Mô tả:** Cache exchange info trong 24 giờ để tối ưu performance, có thể force refresh khi cần.

---

### 7. `get_symbols_info(self)`
**Chức năng:** Lấy thông tin tất cả symbols từ exchange

**Đầu vào:** Không có

**Đầu ra:**
- `dict`: Dictionary chứa thông tin chi tiết các symbols

**Mô tả:** Sử dụng cached exchange info để lấy thông tin chi tiết về tất cả trading pairs đang active.

---

### 8. `get_base_asset_from_symbol(self, symbol)`
**Chức năng:** Trích xuất base asset từ trading symbol

**Đầu vào:**
- `symbol` (str): Trading symbol (ví dụ: BTCUSDT)

**Đầu ra:**
- `Optional[str]`: Base asset (ví dụ: BTC) hoặc None nếu không tìm thấy

**Mô tả:** Helper method sử dụng cached exchange info để extract base asset.

---

### 9. `get_asset_price_usdt(self, asset)`
**Chức năng:** Lấy giá USDT của một asset

**Đầu vào:**
- `asset` (str): Tên asset (ví dụ: BTC, ETH)

**Đầu ra:**
- `float`: Giá USDT hoặc None nếu lỗi

**Mô tả:** Tìm và lấy giá hiện tại của asset so với USDT.

---

### 10. `get_asset_balance(self, asset)`
**Chức năng:** Lấy số dư của một asset cụ thể

**Đầu vào:**
- `asset` (str): Tên asset

**Đầu ra:**
- `dict`: Thông tin balance (free, locked) hoặc None nếu lỗi

**Mô tả:** Truy xuất số dư available và locked của asset.

---

### 11. `get_all_assets_balances(self)`
**Chức năng:** Lấy tất cả số dư tài sản trong tài khoản

**Đầu vào:** Không có

**Đầu ra:**
- `dict`: Dictionary với asset làm key và balance info làm value

**Mô tả:** Lấy toàn bộ balances, chỉ trả về assets có số dư > 0.

---

### 12. `get_all_balances_usdt(self)`
**Chức năng:** Lấy tất cả balances được convert sang USDT

**Đầu vào:** Không có

**Đầu ra:**
- `dict`: Dictionary assets với giá trị USDT

**Mô tả:** Tính toán tổng giá trị portfolio bằng USDT cho tất cả assets.

---

### 13. `get_symbols_list_by_quote_usdt(self)`
**Chức năng:** Lấy tất cả symbols có quote asset là USDT

**Đầu vào:** Không có

**Đầu ra:**
- `list`: Danh sách symbols kết thúc bằng USDT

**Mô tả:** Filter các trading pairs có quote currency là USDT từ cached exchange info.

---

### 14. `get_symbols_list_by_quote_assets(self, quote_assets=['USDT', 'BTC', 'ETH', 'BNB'])`
**Chức năng:** Lấy symbols theo các quote assets chỉ định

**Đầu vào:**
- `quote_assets` (list): Danh sách quote assets

**Đầu ra:**
- `dict`: Dictionary với quote asset làm key, symbols làm values

**Mô tả:** Group symbols theo quote currency để phân loại trading pairs.

---

### 15. `get_historic_data_by_symbol(self, symbol, timeframe, num_candles=450)`
**Chức năng:** Lấy dữ liệu lịch sử OHLCV

**Đầu vào:**
- `symbol` (str): Trading symbol
- `timeframe` (str): Khung thời gian (1m, 5m, 1h, 1d...)
- `num_candles` (int): Số nến cần lấy (mặc định 450)

**Đầu ra:**
- `pandas.DataFrame`: DataFrame OHLCV hoặc None nếu lỗi

**Mô tả:** Download và format dữ liệu klines thành DataFrame cho phân tích kỹ thuật.

---

### 16. `get_historic_trades(self)`
**Chức năng:** Lấy tất cả lịch sử giao dịch đã hoàn thành

**Đầu vào:** Không có

**Đầu ra:**
- `list`: Danh sách tất cả trades đã executed

**Mô tả:** Duyệt qua tất cả symbols để collect toàn bộ lịch sử giao dịch.

---

### 17. `get_historic_trades_by_symbol(self, symbol, limit=100)`
**Chức năng:** Lấy lịch sử giao dịch theo symbol

**Đầu vào:**
- `symbol` (str): Trading symbol
- `limit` (int): Số lượng trades tối đa (mặc định 100)

**Đầu ra:**
- `list`: Danh sách trades của symbol

**Mô tả:** Lấy recent trades của một trading pair cụ thể.

---

### 18. `get_my_trades_by_symbol(self, symbol, limit=500, fromId=None, startTime=None, endTime=None)`
**Chức năng:** Lấy danh sách trades của tài khoản cho một symbol cụ thể

**Đầu vào:**
- `symbol` (str): Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
- `limit` (int): Số lượng trades tối đa (mặc định 500, tối đa 1000)
- `fromId` (str, optional): Trade ID để bắt đầu lấy từ đó
- `startTime` (str, optional): Timestamp bắt đầu (milliseconds)
- `endTime` (str, optional): Timestamp kết thúc (milliseconds)

**Đầu ra:**
- `list`: Danh sách trades của tài khoản hoặc None nếu lỗi

**Mô tả:** Truy xuất và phân tích lịch sử giao dịch của tài khoản cho một symbol cụ thể.

---

### 19. `get_my_trades_all_symbols(self, limit_per_symbol=100)`
**Chức năng:** Lấy trades của tài khoản cho tất cả symbols có balance hoặc có giao dịch gần đây

**Đầu vào:**
- `limit_per_symbol` (int): Số lượng trades tối đa cho mỗi symbol (mặc định 100)

**Đầu ra:**
- `dict`: Dictionary với symbol làm key và list trades làm value

**Mô tả:** Tổng hợp tất cả trades của tài khoản cho các symbols đang có balance hoặc giao dịch gần đây.

---

### 20. `get_my_trades_by_time_range(self, symbol, start_timestamp=None, end_timestamp=None, limit=1000)`
**Chức năng:** Lấy trades của tài khoản trong khoảng thời gian cụ thể

**Đầu vào:**
- `symbol` (str): Trading symbol
- `start_timestamp` (int, optional): Timestamp bắt đầu (milliseconds). Nếu None, lấy từ 24h trước
- `end_timestamp` (int, optional): Timestamp kết thúc (milliseconds). Nếu None, lấy đến hiện tại
- `limit` (int): Số lượng trades tối đa (mặc định 1000)

**Đầu ra:**
- `list`: Danh sách trades trong khoảng thời gian đã chỉ định

**Mô tả:** Truy xuất lịch sử giao dịch của tài khoản trong khoảng thời gian cụ thể, hỗ trợ phân tích hiệu suất theo thời gian.

---

### 21. `get_filled_orders_all_symbols(self)`
**Chức năng:** Lấy tất cả đơn hàng đã filled

**Đầu vào:** Không có

**Đầu ra:**
- `list`: Danh sách tất cả filled orders

**Mô tả:** Collect tất cả orders đã execute thành công trên mọi symbols.

---

### 22. `get_filled_orders_by_symbol(self, symbol, limit=50)`
**Chức năng:** Lấy filled orders theo symbol

**Đầu vào:**
- `symbol` (str): Trading symbol
- `limit` (int): Số lượng orders tối đa (mặc định 50)

**Đầu ra:**
- `list`: Danh sách filled orders của symbol

**Mô tả:** Lấy lịch sử orders đã filled của một trading pair.

---

### 23. `get_pending_orders_all_symbols(self)`
**Chức năng:** Lấy tất cả đơn hàng đang pending

**Đầu vào:** Không có

**Đầu ra:**
- `list`: Danh sách tất cả open orders

**Mô tả:** Lấy toàn bộ orders đang chờ execute trên tất cả symbols.

---

### 24. `order_send(self, symbol, order_type, test_mode=False, quantity=None, quote_quantity=None, price=None, time_in_force="GTC")`
**Chức năng:** Gửi đơn hàng với support test mode

**Đầu vào:**
- `symbol` (str): Trading symbol
- `order_type` (str): Loại order (BUY/SELL)
- `test_mode` (bool): Chế độ test (mặc định False)
- `quantity` (float): Số lượng base asset
- `quote_quantity` (float): Số lượng quote asset
- `price` (float): Giá cho limit order
- `time_in_force` (str): Thời gian hiệu lực (GTC, IOC, FOK)

**Đầu ra:**
- `dict`: Kết quả order hoặc None nếu lỗi

**Mô tả:** Core function để gửi orders, hỗ trợ cả test mode và live trading.

---

### 25. `order_send_market(self, symbol, side, test_mode=False, quote_quantity=None, quantity=None)`
**Chức năng:** Gửi market order

**Đầu vào:**
- `symbol` (str): Trading symbol
- `side` (str): BUY hoặc SELL
- `test_mode` (bool): Chế độ test
- `quote_quantity` (float): Số lượng quote asset
- `quantity` (float): Số lượng base asset

**Đầu ra:**
- `dict`: Kết quả market order

**Mô tả:** Convenience method cho market orders với execute ngay tại giá thị trường.

---

### 26. `order_send_limit(self, symbol, side, price, test_mode=False, quote_quantity=None, quantity=None, time_in_force="GTC")`
**Chức năng:** Gửi limit order

**Đầu vào:**
- `symbol` (str): Trading symbol
- `side` (str): BUY hoặc SELL
- `price` (float): Giá limit
- `test_mode` (bool): Chế độ test
- `quote_quantity` (float): Số lượng quote asset
- `quantity` (float): Số lượng base asset
- `time_in_force` (str): Thời gian hiệu lực

**Đầu ra:**
- `dict`: Kết quả limit order

**Mô tả:** Convenience method cho limit orders với giá xác định.

---

### 27. `cancel_order_limit(self, symbol, order_id)`
**Chức năng:** Hủy một limit order

**Đầu vào:**
- `symbol` (str): Trading symbol
- `order_id` (int): ID của order cần hủy

**Đầu ra:**
- `dict`: Kết quả hủy order hoặc None nếu lỗi

**Mô tả:** Hủy một open order cụ thể.

---

### 28. `get_order_status(self, symbol, order_id)`
**Chức năng:** Kiểm tra trạng thái order

**Đầu vào:**
- `symbol` (str): Trading symbol
- `order_id` (int): ID của order

**Đầu ra:**
- `dict`: Thông tin trạng thái order hoặc None nếu lỗi

**Mô tả:** Query status và thông tin chi tiết của một order.

---

### 29. `buy_market_usdt(self, symbol, usdt_amount, test=False)`
**Chức năng:** Mua market với số tiền USDT

**Đầu vào:**
- `symbol` (str): Trading symbol
- `usdt_amount` (float): Số USDT để mua
- `test` (bool): Chế độ test

**Đầu ra:**
- Kết quả từ `order_send_market`

**Mô tả:** Convenience method để mua asset bằng số tiền USDT cố định.

---

### 30. `sell_market_all_by_symbol(self, symbol, quantity, test=False)`
**Chức năng:** Bán market với số lượng cụ thể

**Đầu vào:**
- `symbol` (str): Trading symbol
- `quantity` (float): Số lượng cần bán
- `test` (bool): Chế độ test

**Đầu ra:**
- Kết quả từ `order_send_market`

**Mô tả:** Convenience method để bán một lượng asset nhất định.

---

### 31. `sell_market_all_by_symbol_and_cancel_orders(self, symbol)`
**Chức năng:** Bán toàn bộ asset và hủy tất cả orders

**Đầu vào:**
- `symbol` (str): Trading symbol

**Đầu ra:**
- `dict`: Kết quả tổng hợp (sell_result, cancel_result)

**Mô tả:** Emergency function để liquidate position và hủy pending orders.

---

### 32. `buy_limit_usdt(self, symbol, price, usdt_amount, test=False)`
**Chức năng:** Mua limit với số tiền USDT

**Đầu vào:**
- `symbol` (str): Trading symbol
- `price` (float): Giá limit
- `usdt_amount` (float): Số USDT
- `test` (bool): Chế độ test

**Đầu ra:**
- Kết quả từ `order_send_limit`

**Mô tả:** Convenience method để đặt buy limit order với giá trị USDT.

---

### 33. `sell_limit_quantity(self, symbol, price, quantity, test=False)`
**Chức năng:** Bán limit với số lượng cụ thể

**Đầu vào:**
- `symbol` (str): Trading symbol
- `price` (float): Giá limit
- `quantity` (float): Số lượng
- `test` (bool): Chế độ test

**Đầu ra:**
- Kết quả từ `order_send_limit`

**Mô tả:** Convenience method để đặt sell limit order.

---

### 34. `sell_market_all_by_order_id(self, order_id)`
**Chức năng:** Bán toàn bộ asset từ một order ID

**Đầu vào:**
- `order_id` (int): ID của order đã filled

**Đầu ra:**
- `dict`: Kết quả bán hoặc None nếu lỗi

**Mô tả:** Tìm filled order, extract base asset và bán toàn bộ số lượng.

---

### 35. `emergency_close_all(self)`
**Chức năng:** Đóng tất cả positions trong trường hợp khẩn cấp

**Đầu vào:** Không có

**Đầu ra:**
- `dict`: Báo cáo tổng hợp kết quả

**Mô tả:** Emergency function để liquidate toàn bộ portfolio và hủy mọi pending orders.

---

### 36. `stop(self)`
**Chức năng:** Dừng tick processor và cleanup resources

**Đầu vào:** Không có

**Đầu ra:**
- `bool`: True nếu thành công, False nếu có lỗi

**Mô tả:** Clean shutdown với memory cleanup và resource management.

---

### 37. `on_order_event(self)`
**Chức năng:** Xử lý events khi có thay đổi orders

**Đầu vào:** Không có

**Đầu ra:** Không có

**Mô tả:** Monitor order changes, detect new/closed orders và trigger callbacks.

---

### 38. `_process_new_crypto_orders(self, new_order_ids, current_orders_dict)`
**Chức năng:** Xử lý orders mới được phát hiện

**Đầu vào:**
- `new_order_ids` (set): Set các order ID mới
- `current_orders_dict` (dict): Dictionary thông tin orders hiện tại

**Đầu ra:** Không có

**Mô tả:** Helper method để process new orders và trigger callbacks.

---

### 39. `_process_closed_crypto_orders(self, closed_order_ids)`
**Chức năng:** Xử lý orders đã đóng

**Đầu vào:**
- `closed_order_ids` (set): Set các order ID đã đóng

**Đầu ra:** Không có

**Mô tả:** Helper method để process closed orders và trigger callbacks.

---

## Các Tính Năng Chính

### Exchange Info Cache
- Cache thông tin exchange trong 24 giờ
- Tự động refresh khi hết hạn
- Có thể force refresh khi cần

### Parameter Validation
- Validate symbol, order_id, quantity, price
- Đảm bảo type safety và data integrity
- Error logging chi tiết

### Batch Order Management
- Hủy nhiều orders cùng lúc
- Retry mechanism với exponential backoff
- Rate limiting protection

### Test Mode Support
- Tất cả trading functions hỗ trợ test mode
- Mock responses cho test orders
- Safe testing environment

### Comprehensive Logging
- Detailed logging cho tất cả operations
- Error tracking và debugging
- Performance monitoring

### Emergency Functions
- Close all positions trong trường hợp khẩn cấp
- Batch cancel tất cả orders
- Portfolio liquidation

## Cách Sử Dụng

```python
# Khởi tạo
processor = tick_processor(
    trade_open_callback=my_open_callback,
    trade_close_callback=my_close_callback
)

# Lấy thông tin tài khoản
account_info = processor.get_account_info()

# Lấy balances
balances = processor.get_all_assets_balances()
usdt_balances = processor.get_all_balances_usdt()

# Lấy lịch sử giao dịch
# Lấy giao dịch cho một symbol cụ thể
my_btc_trades = processor.get_my_trades_by_symbol("BTCUSDT", limit=100)

# Lấy giao dịch cho tất cả symbols
all_my_trades = processor.get_my_trades_all_symbols()

# Lấy giao dịch trong khoảng thời gian (48 giờ qua)
from datetime import datetime, timezone, timedelta
end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
start_time = int((datetime.now(timezone.utc) - timedelta(hours=48)).timestamp() * 1000)
recent_trades = processor.get_my_trades_by_time_range("ETHUSDT", start_time, end_time)

# Giao dịch test
result = processor.buy_market_usdt("BTCUSDT", 100, test=True)

# Giao dịch thực
result = processor.buy_market_usdt("BTCUSDT", 100, test=False)

# Bán hết một symbol và hủy tất cả orders
result = processor.sell_market_all_by_symbol_and_cancel_orders("BTCUSDT")

# Emergency close
result = processor.emergency_close_all()

# Cleanup
processor.stop()
```

## Lưu Ý An Toàn

1. **Test Mode**: Luôn test trước khi trade thực
2. **Parameter Validation**: Tất cả inputs đều được validate
3. **Error Handling**: Comprehensive error handling và logging
4. **Rate Limiting**: Built-in protection khỏi rate limits
5. **Emergency Functions**: Sẵn sàng cho các tình huống khẩn cấp

## Phụ Thuộc

- `binance.spot`: Binance Python connector
- `pandas`: Data manipulation
- `logging`: Logging framework
- `datetime`: Time management
- `typing`: Type hints

Tài liệu này cung cấp overview chi tiết về tất cả functions trong `tick_processor.py` với focus vào crypto trading và Binance integration.
