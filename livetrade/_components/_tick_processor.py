import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from time import sleep
from typing import Dict, Optional, List
import json  
from binance.spot import Spot as Client

from livetrade.config import API_KEY, API_SECRET

class tick_processor():
    def __init__(self, trade_open_callback, trade_close_callback,):
        # Initialize Binance client with API credentials
        self.spot_client = Client(
            api_key=API_KEY,
            api_secret=API_SECRET,
        )

        self.df_cache = {}
        self.last_open_time = datetime.now(timezone.utc)
        self.last_modification_time = datetime.now(timezone.utc)

        # Add logging
        self.logger = logging.getLogger(__name__)

        # Store callback functions
        self.trade_open_callback = trade_open_callback
        self.trade_close_callback = trade_close_callback

        # Log the status of the trade_open_callback
        if self.trade_open_callback:
            # Attempt to get the function name, default to 'anonymous function' if not available
            callback_name = getattr(self.trade_open_callback, '__name__', 'anonymous function')
            self.logger.info(f"trade_open_callback successfully set to: {callback_name}")
        else:
            self.logger.warning("trade_open_callback is None during initialization.")

        # Log the status of the trade_close_callback
        if self.trade_close_callback:
            # Attempt to get the function name, default to 'anonymous function' if not available
            callback_name = getattr(self.trade_close_callback, '__name__', 'anonymous function')
            self.logger.info(f"trade_close_callback successfully set to: {callback_name}")
        else:
            self.logger.warning("trade_close_callback is None during initialization.")

        print(f"System initialized at: {datetime.now(timezone.utc)}")

        # Enhanced Exchange Info Cache with expiry
        self.exchange_info_cache = None
        self.cache_expiry = None
        self.cache_duration_hours = 1  # Cache expires after 1 hour
    
    def _validate_parameters(self, **kwargs) -> bool:
        """
        Validate important parameters for trading operations
        
        Args:
            **kwargs: Key-value pairs to validate
            
        Returns:
            bool: True if all parameters are valid, False otherwise
        """
        try:
            for param_name, param_value in kwargs.items():
                if param_name == 'symbol':
                    if not param_value or not isinstance(param_value, str) or len(param_value) < 3:
                        self.logger.error(f"Invalid symbol: {param_value}")
                        return False
                
                elif param_name == 'order_id':
                    # Kiểm tra type trước
                    if not isinstance(param_value, (int, str)):
                        self.logger.error(f"Invalid order_id type: {param_value}, must be int or string")
                        return False
                    
                    # Nếu param_value là None hoặc empty string
                    if param_value is None or param_value == "" or param_value == 0:
                        self.logger.error(f"Invalid order_id: {param_value}, cannot be None, empty, or zero")
                        return False
                    
                    # Convert to int if string và validate
                    try:
                        order_id_int = int(param_value) if isinstance(param_value, str) else param_value
                        if order_id_int <= 0:  # Order ID phải là số dương
                            self.logger.error(f"Invalid order_id: {param_value}, must be positive")
                            return False
                    except (ValueError, TypeError):
                        self.logger.error(f"Invalid order_id format: {param_value}")
                        return False
        
                elif param_name == 'quantity':
                    if not isinstance(param_value, (int, float)) or param_value <= 0:
                        self.logger.error(f"Invalid quantity: {param_value}, must be positive number")
                        return False
            
                elif param_name == 'price':
                    if not isinstance(param_value, (int, float)) or param_value <= 0:
                        self.logger.error(f"Invalid price: {param_value}, must be positive number")
                        return False
            
                elif param_name == 'quote_quantity':
                    if not isinstance(param_value, (int, float)) or param_value <= 0:
                        self.logger.error(f"Invalid quote_quantity: {param_value}, must be positive number")
                        return False

            return True
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {str(e)}")
            return False
    
    def _cancel_orders_batch(self, symbol: str, order_ids: List[int], max_retries: int = 3) -> Dict:
        """
        Shared method to cancel multiple orders with error handling and retries
        
        Args:
            symbol (str): Trading symbol
            order_ids (List[int]): List of order IDs to cancel
            max_retries (int): Maximum number of retries per order
            
        Returns:
            dict: Summary of cancellation results
        """
        try:
            if not self._validate_parameters(symbol=symbol):
                return {'success': False, 'error': 'Invalid parameters'}
            
            if not order_ids:
                return {'success': True, 'cancelled_orders': [], 'failed_orders': []}
            
            cancelled_orders = []
            failed_orders = []
            
            self.logger.info(f"Cancelling {len(order_ids)} orders for {symbol}")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return {'success': False, 'error': 'Spot client not initialized'}
            
            for order_id in order_ids:
                if not self._validate_parameters(order_id=order_id):
                    failed_orders.append({'order_id': order_id, 'error': 'Invalid order_id'})
                    continue
                
                retry_count = 0
                cancelled = False
                
                while retry_count < max_retries and not cancelled:
                    try:
                        response = self.spot_client.cancel_order(symbol=symbol, orderId=int(order_id))
                        
                        if response and response.get('status') == 'CANCELED':
                            cancelled_orders.append(order_id)
                            cancelled = True
                            self.logger.info(f"Successfully cancelled order {order_id}")
                        else:
                            retry_count += 1
                            if retry_count < max_retries:
                                self.logger.warning(f"Retry {retry_count} for order {order_id}")
                                sleep(0.1)
                
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        
                        if retry_count >= max_retries:
                            failed_orders.append({'order_id': order_id, 'error': error_msg})
                            self.logger.error(f"Failed to cancel order {order_id} after {max_retries} retries: {error_msg}")
                        else:
                            self.logger.warning(f"Retry {retry_count} for order {order_id} due to error: {error_msg}")
                            sleep(0.1)
            
            result = {
                'success': True,
                'cancelled_orders': cancelled_orders,
                'failed_orders': failed_orders,
                'total_requested': len(order_ids),
                'total_cancelled': len(cancelled_orders),
                'total_failed': len(failed_orders)
            }
            
            self.logger.info(f"Batch cancellation complete: {len(cancelled_orders)}/{len(order_ids)} orders cancelled")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in batch cancel orders: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _klines_to_dataframe(self, klines_data):
        """
        Convert klines data from Binance to DataFrame format
    
        Args:
            klines_data (list): Raw klines data from Binance API
    
        Returns:
            pandas.DataFrame: Formatted DataFrame or None if error 
        """
        try:
            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            df = pd.DataFrame(klines_data, columns=columns)
            
            # Convert data types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convert price and volume columns to float
            price_volume_cols = ['open', 'high', 'low', 'close', 'volume', 
                                'quote_asset_volume', 'taker_buy_base_asset_volume', 
                                'taker_buy_quote_asset_volume']
            
            for col in price_volume_cols:
                df[col] = df[col].astype(float)
            
            df['number_of_trades'] = df['number_of_trades'].astype(int)
            
            # Drop unnecessary 'ignore' column
            df = df.drop('ignore', axis=1)
            
            # Set index name
            df.index.name = "time"
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting klines to dataframe: {str(e)}")
            return None
        
    def get_account_info(self):
        """
        Get account information from Binance
    
        Returns:
            dict: Dictionary containing account information or empty dict if error
        """
        try:
            self.logger.info("Fetching account information from Binance...")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Lấy thông tin tài khoản
            account_info = self.spot_client.account()
            
            if not account_info:
                self.logger.error("Failed to get account information from Binance")
                return {}
            
            self.logger.info("Account information retrieved successfully")
            return account_info
            
        except Exception as e:
            self.logger.error(f"Error getting account information: {str(e)}")
            return {}
        
    def get_exchange_info_cached(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get exchange info with caching mechanism
        
        Args:
            force_refresh (bool): Force refresh the cache
            
        Returns:
            dict: Exchange info or None if error
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Check if cache is valid
            if (not force_refresh and 
                self.exchange_info_cache is not None and 
                self.cache_expiry is not None and 
                current_time < self.cache_expiry):
                
                self.logger.debug("Using cached exchange info")
                return self.exchange_info_cache
            
            # Fetch fresh data
            self.logger.info("Fetching fresh exchange info from Binance API")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            exchange_info = self.spot_client.exchange_info()
            
            if not exchange_info or 'symbols' not in exchange_info:
                self.logger.error("Failed to get valid exchange info from Binance")
                return None
            
            # Update cache
            self.exchange_info_cache = exchange_info
            self.cache_expiry = current_time + timedelta(hours=self.cache_duration_hours)
            
            self.logger.info(f"Exchange info cached successfully. Cache expires at: {self.cache_expiry}")
            return exchange_info
            
        except Exception as e:
            self.logger.error(f"Error getting exchange info: {str(e)}")
            return None

    def get_symbols_info(self):
        """
        Get detailed information about all symbols from Binance using cached exchange info
    
        Returns:
            dict: Dictionary containing detailed symbol information or empty dict if error
        """
        try:
            print("Fetching detailed symbol information from Binance...")
            
            exchange_info = self.get_exchange_info_cached()
            
            if not exchange_info:
                self.logger.error("Failed to get exchange info from Binance")
                return {}
            
            # Create dictionary with symbol as key and detailed info as value
            symbols_info = {}
            
            for symbol_data in exchange_info['symbols']:
                if symbol_data['status'] == 'TRADING':  # Only get active symbols
                    symbol = symbol_data['symbol']
                    symbols_info[symbol] = {
                        'baseAsset': symbol_data['baseAsset'],
                        'quoteAsset': symbol_data['quoteAsset'],
                        'status': symbol_data['status'],
                        'orderTypes': symbol_data['orderTypes'],
                        'filters': symbol_data['filters']
                    }
            
            self.logger.info(f"Retrieved detailed info for {len(symbols_info)} active symbols")
            return symbols_info
            
        except Exception as e:
            self.logger.error(f"Error getting detailed symbols info: {str(e)}")
            return {}
    
    def get_base_asset_from_symbol(self, symbol: str) -> Optional[str]:
        """
        Helper method to extract base asset from trading symbol using cached exchange info
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            str: Base asset (e.g., 'BTC') or None if not found
        """
        try:
            if not self._validate_parameters(symbol=symbol):
                return None
            
            exchange_info = self.get_exchange_info_cached()
            if not exchange_info:
                return None
            
            for symbol_data in exchange_info['symbols']:
                if symbol_data['symbol'] == symbol:
                    base_asset = symbol_data['baseAsset']
                    self.logger.debug(f"Found base asset '{base_asset}' for symbol '{symbol}'")
                    return base_asset
            
            self.logger.warning(f"Base asset not found for symbol: {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting base asset for {symbol}: {str(e)}")
            return None
    
    def get_symbol_price(self, symbol):
        """
        Lấy giá hiện tại của một symbol
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Current price or None if error
        """
        try:
            if not self._validate_parameters(symbol=symbol):
                return None
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized")
                return None
            
            price_data = self.spot_client.ticker_price(symbol)
            
            if price_data and 'price' in price_data:
                return float(price_data['price'])
            
            self.logger.warning(f"No price data available for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
        
    def get_asset_price_usdt(self, asset):
            """
            Lấy giá của một asset so với USDT
        
            Args:
                asset (str): Tên asset (e.g., 'BTC', 'ETH')
        
            Returns:
                float: Giá của asset tính bằng USDT, hoặc None nếu lỗi
            """
            try:
                if asset == 'USDT':
                    return 1.0
                
                if self.spot_client is None:
                    self.logger.error("Spot client is not initialized, cannot cancel order")
                    return None
                
                # Thử {ASSET}USDT trước
                symbol_usdt = f"{asset}USDT"
                try:
                    price_data = self.spot_client.ticker_price(symbol_usdt)
                    if price_data and 'price' in price_data:
                        return float(price_data['price'])
                except:
                    pass
                
                # Thử USDT{ASSET} nếu không có
                symbol_reversed = f"USDT{asset}"
                try:
                    price_data = self.spot_client.ticker_price(symbol_reversed)
                    if price_data and 'price' in price_data:
                        return 1.0 / float(price_data['price'])
                except:
                    pass
                
                self.logger.warning(f"Cannot get USDT price for {asset}")
                return None
                
            except Exception as e:
                self.logger.error(f"Error getting price for {asset}: {str(e)}")
                return None
        
    def get_asset_balance(self, asset):
        """
        Lấy số dư của một tài sản cụ thể trong tài khoản Binance
    
        Args:
            asset (str): Mã tài sản (e.g., 'BTC', 'ETH', 'USDT')
    
        Returns:
            float: Số dư của tài sản, hoặc None nếu không tìm thấy hoặc xảy ra lỗi
        """
        try:
            self.logger.info(f"Fetching balance for asset: {asset}")
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized")
                return None
            
            # Sử dụng spot_client.account() để lấy thông tin tài khoản
            account_info = self.spot_client.account()
            
            if not account_info or "balances" not in account_info:
                self.logger.error("Failed to get account information from Binance")
                return None
            
            # Tìm tài sản trong danh sách balances
            for balance in account_info["balances"]:
                if balance["asset"] == asset:
                    free_balance = float(balance["free"])
                    locked_balance = float(balance["locked"])
                    self.logger.info(f"Balance for {asset}: Free={free_balance}, Locked={locked_balance}")
                    return free_balance
            
            self.logger.warning(f"Asset {asset} not found in account balances")
            return None

        except Exception as e:
            self.logger.error(f"Error getting balance for {asset}: {str(e)}")
            return None

    def get_all_assets_balances(self):
        """
        Lấy tất cả số dư tài sản trong tài khoản
    
        Returns:
        dict: Dictionary với asset làm key và balance info làm value, hoặc empty dict nếu lỗi
        """
        try:
            self.logger.info("Fetching all asset balances from Binance...")
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized")
                return {}
            
            # Sử dụng spot_client.account() để lấy thông tin tài khoản
            account_info = self.spot_client.account()
            
            if not account_info or "balances" not in account_info:
                self.logger.error("Failed to get account information from Binance")
                return {}
            
            # Chuyển đổi thành dictionary và chỉ lấy assets có balance > 0
            balances = {}
            for balance in account_info["balances"]:
                asset = balance["asset"]
                free_balance = float(balance["free"])
                locked_balance = float(balance["locked"])
                total_balance = free_balance + locked_balance
                
                # Chỉ thêm vào kết quả nếu có balance
                if total_balance > 0:
                    balances[asset] = {
                        'free': free_balance,
                        'locked': locked_balance,
                        'total': total_balance
                    }
        
            self.logger.info(f"Retrieved balances for {len(balances)} assets with non-zero balance")
            
            # Log một số ví dụ
            if balances:
                sample_assets = list(balances.keys())[:5]
                self.logger.info(f"Sample assets with balance: {sample_assets}")
        
            return balances
        
        except Exception as e:
            self.logger.error(f"Error getting all balances: {str(e)}")
            return {}
    
    def get_all_balances_usdt(self):
        """
        Lấy tất cả số dư tài sản và quy đổi về USDT
    
        Returns:
            dict: Dictionary chứa balance info đã quy đổi về USDT
        """
        try:
            self.logger.info("Fetching all balances and converting to USDT...")
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized")
                return {}
            
            # Get account information
            account_info = self.spot_client.account()
            
            if not account_info or "balances" not in account_info:
                self.logger.error("Failed to get account information from Binance")
                return {}
            
            balances_usdt = {}
            total_portfolio_usdt = 0.0
            
            # Process non-zero balances
            for balance in account_info["balances"]:
                asset = balance["asset"]
                free_balance = float(balance["free"])
                locked_balance = float(balance["locked"])
                total_balance = free_balance + locked_balance
                
                # Skip zero balances
                if total_balance <= 0:
                    continue
                
                # Handle USDT directly
                if asset == 'USDT':
                    usdt_value = total_balance
                    balances_usdt[asset] = {
                        'asset': asset,
                        'free': free_balance,
                        'locked': locked_balance,
                        'total': total_balance,
                        'usdt_value': usdt_value,
                        'percentage': 0.0  # Will calculate later
                    }
                    total_portfolio_usdt += usdt_value
                    continue
                # Get price for non-USDT assets using get_asset_price_usdt
                asset_price = self.get_asset_price_usdt(asset)
                if asset_price is None:
                    continue
                
                usdt_value = total_balance * asset_price
                
                balances_usdt[asset] = {
                    'asset': asset,
                    'free': free_balance,
                    'locked': locked_balance,
                    'total': total_balance,
                    'usdt_value': usdt_value,
                    'percentage': 0.0  # Will calculate later
                }
                
                total_portfolio_usdt += usdt_value
                
                # Small delay to avoid rate limit
                sleep(0.02)
            
            # Calculate percentages
            if total_portfolio_usdt > 0:
                for asset_data in balances_usdt.values():
                    asset_data['percentage'] = (asset_data['usdt_value'] / total_portfolio_usdt) * 100
            
            # Sort by USDT value descending
            sorted_balances = dict(sorted(
                balances_usdt.items(),
                key=lambda x: x[1]['usdt_value'],
                reverse=True
            ))
            
            result = {
                'total_portfolio_usdt': total_portfolio_usdt,
                'assets_count': len(sorted_balances),
                'balances': sorted_balances,
                'top_5_assets': list(sorted_balances.keys())[:5]
            }
            
            self.logger.info(f"Portfolio converted to USDT: {total_portfolio_usdt:.2f} USDT from {len(sorted_balances)} assets")
            
            # Log top asset
            if sorted_balances:
                top_asset = list(sorted_balances.keys())[0]
                top_asset_data = sorted_balances[top_asset]
                self.logger.info(f"Largest holding: {top_asset} = {top_asset_data['usdt_value']:.2f} USDT ({top_asset_data['percentage']:.1f}%)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error converting balances to USDT: {str(e)}")
            return {}

    def get_symbols_list_by_quote_usdt(self):
        """
        Lấy danh sách các trading pairs có sẵn trên Binance với quote asset là USDT
        
        Returns:
            list: Danh sách các symbol đang được giao dịch với quote asset đã chọn
        """
        
        quote_asset='USDT'
        
        try:
            self.logger.info(f"Fetching symbol list for {quote_asset} pairs from Binance...")
            
            # Sử dụng cached exchange_info
            exchange_info = self.get_exchange_info_cached()
            
            if not exchange_info or 'symbols' not in exchange_info:
                self.logger.error("Failed to get exchange info from cache")
                return []
            
            # Lọc symbols theo quote asset và trạng thái
            filtered_symbols = [
                s['symbol'] for s in exchange_info['symbols'] 
                if s['quoteAsset'] == quote_asset and s['status'] == 'TRADING'
            ]
            
            # Log kết quả
            symbol_count = len(filtered_symbols)
            self.logger.info(f"Found {symbol_count} active {quote_asset} trading pairs")
            
            # Log một số ví dụ symbols
            if filtered_symbols:
                sample_size = min(5, symbol_count)
                sample_symbols = filtered_symbols[:sample_size]
                self.logger.info(f"Sample symbols: {', '.join(sample_symbols)}")
            
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting symbols list: {str(e)}", exc_info=True)
            return []

    def get_symbols_list_by_quote_assets(self, quote_assets=['USDT', 'BTC', 'ETH', 'BNB']):
        """
        Lấy symbols được nhóm theo quote assets
    
        Args:
            quote_assets (list): Danh sách quote assets cần lọc
    
        Returns:
            dict: Dictionary với quote asset làm key và list symbols làm value
        """
        try:
            print(f"Fetching symbols grouped by quote assets: {quote_assets}")
            
            # Sử dụng cached exchange_info
            exchange_info = self.get_exchange_info_cached()
            
            if not exchange_info or 'symbols' not in exchange_info:
                self.logger.error("Failed to get exchange info from cache")
                return {}
            
            # Tạo dictionary nhóm symbols theo quote asset
            grouped_symbols = {asset: [] for asset in quote_assets}
            
            for symbol_data in exchange_info['symbols']:
                if (symbol_data['status'] == 'TRADING' and 
                    symbol_data['quoteAsset'] in quote_assets):
                    
                    quote_asset = symbol_data['quoteAsset']
                    grouped_symbols[quote_asset].append(symbol_data['symbol'])
            
            # Log thống kê
            for quote_asset, symbols in grouped_symbols.items():
                self.logger.info(f"{quote_asset} pairs: {len(symbols)}")
            
            return grouped_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting grouped symbols: {str(e)}")
            return {}
        
    def get_historic_data_by_symbol(self, symbol, timeframe, num_candles=450):
        """
        Lấy dữ liệu lịch sử từ Binance cho crypto trading
    
        Args:
            symbol (str): Symbol crypto (e.g., 'BTCUSDT', 'ETHUSDT')
            timeframe (str): Khung thời gian ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
            num_candles (int): Số lượng nến cần lấy (mặc định 450)
        
        Returns:
            pandas.DataFrame: DataFrame chứa dữ liệu OHLCV hoặc None nếu lỗi
        """
        try:
            
            # Ensure timeframe is not None
            if timeframe is None:
                self.logger.error(f"Invalid timeframe: {timeframe}, could not be mapped")
                return None
                
            print(f"Requesting {num_candles} candles of {symbol} {timeframe} data from Binance...")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Lấy dữ liệu klines từ Binance
            klines_data = self.spot_client.klines(symbol, timeframe, limit=num_candles)
            
            if not klines_data:
                print(f"Warning: No data received for {symbol} {timeframe}")
                return None
            
            # Chuyển đổi thành DataFrame
            df = self._klines_to_dataframe(klines_data)
            
            if df is not None and not df.empty:
                # Store in cache using symbol and timeframe as key
                cache_key = (symbol, timeframe)
                self.df_cache[cache_key] = df
                
                print(f"Cached {len(df)} candles for {symbol} {timeframe}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                
                return df
            else:
                print(f"Error: Failed to process data for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting historic data for {symbol} {timeframe}: {str(e)}")
            return None

    def get_historic_trades(self):
        """
        Lấy dữ liệu giao dịch lịch sử từ Binance và xử lý trades gần nhất
        """
        try:
            self.logger.info("Starting historic trades retrieval from Binance...")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Lấy danh sách symbols từ Binance
            symbols_list = self.get_symbols_list_by_quote_usdt()
            
            if not symbols_list:
                self.logger.warning("No symbols available for historic trades retrieval")
                return []
            
            self.logger.info(f"Processing historic trades for {len(symbols_list)} symbols")
            
            all_trades = []
            processed_symbols = 0
            
            # Lặp qua từng symbol để lấy historical trades
            for symbol in symbols_list:
                try:
                    # Lấy historical trades cho symbol này
                    trades_data = self.spot_client.historical_trades(symbol, limit=10)
                    
                    if trades_data:
                        # Thêm symbol vào mỗi trade record để dễ tracking
                        for trade in trades_data:
                            trade['symbol'] = symbol
                            all_trades.append(trade)
                        
                        self.logger.debug(f"Retrieved {len(trades_data)} trades for {symbol}")
                    
                    processed_symbols += 1
                    
                    # Log progress mỗi 50 symbols
                    if processed_symbols % 50 == 0:
                        self.logger.info(f"Processed {processed_symbols}/{len(symbols_list)} symbols")
                    
                    # Tạm dừng nhỏ để tránh rate limit
                    sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error getting historic trades for {symbol}: {str(e)}")
                    continue
            
            if not all_trades:
                self.logger.warning("No historic trades data retrieved from any symbol")
                return
            
            # Sắp xếp trades theo thời gian (gần nhất ở vị trí 0)
            sorted_trades = sorted(
                all_trades,
                key=lambda x: int(x['time']),  # time là timestamp trong milliseconds
                reverse=True  # Gần nhất trước
            )
            
            self.logger.info(f"Retrieved and sorted {len(sorted_trades)} total trades")
            
            # Xử lý trade gần nhất
            if sorted_trades and self.trade_close_callback:
                most_recent_trade = sorted_trades[0]
                
                # Chuyển đổi format để tương thích với callback
                trade_data = {
                    'symbol': most_recent_trade['symbol'],
                    'trade_id': most_recent_trade['id'],
                    'price': float(most_recent_trade['price']),
                    'qty': float(most_recent_trade['qty']),
                    'time': most_recent_trade['time'],
                    'isBuyerMaker': most_recent_trade['isBuyerMaker'],
                    'isBestMatch': most_recent_trade['isBestMatch']
                }
                
                self.logger.info(f"Processing most recent trade: {most_recent_trade['symbol']} at {most_recent_trade['time']}")
                
                # Gọi callback với trade gần nhất
                try:
                    self.trade_close_callback(most_recent_trade['id'], trade_data)
                    self.logger.info(f"Successfully sent trade {most_recent_trade['id']} to callback")
                except Exception as e:
                    self.logger.error(f"Error calling trade_close_callback: {str(e)}")
            
            elif not self.trade_close_callback:
                self.logger.warning("No trade_close_callback defined - cannot process trades")
                
            # Log thống kê
            if sorted_trades:
                recent_symbols = list(set([trade['symbol'] for trade in sorted_trades[:10]]))
                self.logger.info(f"Most active symbols in recent trades: {recent_symbols[:5]}")
                
        except Exception as e:
            self.logger.error(f"Error in on_historic_trades: {str(e)}", exc_info=True)

    def get_historic_trades_by_symbol(self, symbol, limit=100):
        """
        Lấy historic trades cho một symbol cụ thể cho TẤT CẢ CÁC GIAO DỊCH ĐANG XẢY RA TRÊN SÀN
    
        Args:
        symbol (str): Symbol cần lấy trades (e.g., 'BTCUSDT')
        limit (int): Số lượng trades tối đa (mặc định 100, max 1000)
    
        Returns:
        list: Danh sách trades đã được sắp xếp theo thời gian
        """
        try:
            self.logger.info(f"Getting historic trades for {symbol}")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Lấy historical trades
            trades_data = self.spot_client.historical_trades(symbol, limit=limit)
            
            if not trades_data:
                self.logger.warning(f"No historic trades data for {symbol}")
                return []
            
            # Sắp xếp theo thời gian (gần nhất trước)
            sorted_trades = sorted(
                trades_data,
                key=lambda x: int(x['time']),
                reverse=True
            )
            
            self.logger.info(f"Retrieved {len(sorted_trades)} trades for {symbol}")
            return sorted_trades
            
        except Exception as e:
            self.logger.error(f"Error getting historic trades for {symbol}: {str(e)}")
            return []
    
    def get_my_trades_by_symbol(self, symbol, limit=500, fromId=None, startTime=None, endTime=None):
        """
        Lấy danh sách trades của tài khoản cho một symbol cụ thể
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            limit (int): Số lượng trades tối đa (mặc định 500, tối đa 1000)
            fromId (str, optional): Trade ID để bắt đầu lấy từ đó
            startTime (str, optional): Timestamp bắt đầu (milliseconds)
            endTime (str, optional): Timestamp kết thúc (milliseconds)
        
        Returns:
            list: Danh sách trades của tài khoản hoặc None nếu lỗi
        """
        try:
            self.logger.info(f"Getting my trades for {symbol}")
            
            if not self._validate_parameters(symbol=symbol):
                return None
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized")
                return None
            
            # Prepare parameters
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            # Add optional parameters if provided
            if fromId is not None:
                params['fromId'] = str(fromId)
                
            if startTime is not None:
                params['startTime'] = str(startTime)
                
            if endTime is not None:
                params['endTime'] = str(endTime)
            
            self.logger.debug(f"My trades parameters: {params}")
            
            # Get my trades from Binance
            my_trades = self.spot_client.my_trades(**params)
            
            if not my_trades:
                self.logger.warning(f"No trades found for {symbol}")
                return []
            
            # Sort by time (newest first)
            sorted_trades = sorted(
                my_trades,
                key=lambda x: int(x['time']),
                reverse=True
            )
            
            self.logger.info(f"Retrieved {len(sorted_trades)} trades for {symbol}")
            
            # Log some statistics
            if sorted_trades:
                total_qty = sum(float(trade['qty']) for trade in sorted_trades)
                total_quote_qty = sum(float(trade['quoteQty']) for trade in sorted_trades)
                buy_trades = [trade for trade in sorted_trades if trade['isBuyer']]
                sell_trades = [trade for trade in sorted_trades if not trade['isBuyer']]
                
                self.logger.info(f"Trades summary for {symbol}:")
                self.logger.info(f"  Total quantity: {total_qty}")
                self.logger.info(f"  Total quote quantity: {total_quote_qty}")
                self.logger.info(f"  Buy trades: {len(buy_trades)}")
                self.logger.info(f"  Sell trades: {len(sell_trades)}")
                
                # Log most recent trade
                latest_trade = sorted_trades[0]
                self.logger.info(f"  Latest trade: {latest_trade['qty']} @ {latest_trade['price']} at {latest_trade['time']}")
            
            return sorted_trades
            
        except Exception as e:
            self.logger.error(f"Error getting my trades for {symbol}: {str(e)}")
            return None

    def get_my_trades_all_symbols(self, limit_per_symbol=100):
        """
        Lấy trades của tài khoản cho tất cả symbols có balance hoặc có giao dịch gần đây
        
        Args:
            limit_per_symbol (int): Số lượng trades tối đa cho mỗi symbol (mặc định 100)
        
        Returns:
            dict: Dictionary với symbol làm key và list trades làm value
        """
        try:
            self.logger.info("Getting my trades for all symbols...")
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized")
                return {}
            
            # Get symbols that have balance or recent activity
            all_balances = self.get_all_assets_balances()
            symbols_with_balance = []
            
            for asset in all_balances.keys():
                if asset != 'USDT':
                    symbol = f"{asset}USDT"
                    symbols_with_balance.append(symbol)
            
            # Also check recent filled orders to get symbols with recent activity
            try:
                recent_orders = self.get_filled_orders_all_symbols()
                symbols_with_recent_activity = []
                
                if recent_orders:
                    # Get unique symbols from recent orders (last 50)
                    symbols_with_recent_activity = list(set([
                        order['symbol'] for order in recent_orders[:50]
                    ]))
            except Exception as e:
                self.logger.warning(f"Failed to get recent orders: {str(e)}")
                symbols_with_recent_activity = []
            
            # Combine and deduplicate symbols
            all_symbols = list(set(symbols_with_balance + symbols_with_recent_activity))
            
            if not all_symbols:
                self.logger.warning("No symbols found with balance or recent activity")
                return {}
            
            self.logger.info(f"Checking my trades for {len(all_symbols)} symbols")
            
            all_my_trades = {}
            processed_count = 0
            
            for symbol in all_symbols:
                try:
                    trades = self.get_my_trades_by_symbol(symbol, limit=limit_per_symbol)
                    
                    if trades:
                        all_my_trades[symbol] = trades
                        self.logger.debug(f"Found {len(trades)} trades for {symbol}")
                    
                    processed_count += 1
                    
                    # Log progress every 10 symbols
                    if processed_count % 10 == 0:
                        self.logger.info(f"Processed {processed_count}/{len(all_symbols)} symbols")
                    
                    # Small delay to avoid rate limiting
                    sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error getting trades for {symbol}: {str(e)}")
                    continue
            
            # Log summary
            total_trades = sum(len(trades) for trades in all_my_trades.values())
            self.logger.info(f"Retrieved {total_trades} total trades across {len(all_my_trades)} symbols")
            
            # Log top symbols by trade count
            if all_my_trades:
                sorted_by_count = sorted(
                    all_my_trades.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )
                
                top_5 = sorted_by_count[:5]
                self.logger.info("Top symbols by trade count:")
                for symbol, trades in top_5:
                    self.logger.info(f"  {symbol}: {len(trades)} trades")
            
            return all_my_trades
            
        except Exception as e:
            self.logger.error(f"Error getting my trades for all symbols: {str(e)}")
            return {}

    def get_my_trades_by_time_range(self, symbol, start_timestamp=None, end_timestamp=None, limit=1000):
        """
        Lấy trades của tài khoản trong khoảng thời gian cụ thể
        
        Args:
            symbol (str): Trading symbol
            start_timestamp (int, optional): Timestamp bắt đầu (milliseconds). Nếu None, lấy từ 24h trước
            end_timestamp (int, optional): Timestamp kết thúc (milliseconds). Nếu None, lấy đến hiện tại
            limit (int): Số lượng trades tối đa (mặc định 1000)
        
        Returns:
            list: Danh sách trades trong khoảng thời gian đã chỉ định
        """
        try:
            self.logger.info(f"Getting my trades for {symbol} within time range")
            
            if not self._validate_parameters(symbol=symbol):
                return None
            
            # Validate limit
            if not isinstance(limit, int) or limit <= 0 or limit > 1000:
                self.logger.error(f"Invalid limit: {limit}. Must be between 1 and 1000")
                return None
            
            # Set default time range if not provided (last 24 hours)
            if start_timestamp is None:
                current_time = datetime.now(timezone.utc)
                start_time = current_time - timedelta(hours=24)
                start_timestamp = int(start_time.timestamp() * 1000)
            
            if end_timestamp is None:
                end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # Validate timestamps
            if start_timestamp >= end_timestamp:
                self.logger.error(f"Invalid time range: start_timestamp ({start_timestamp}) must be less than end_timestamp ({end_timestamp})")
                return None
            
            # Check if time range is too large (Binance has limits)
            time_diff_hours = (end_timestamp - start_timestamp) / (1000 * 60 * 60)
            if time_diff_hours > 24 * 7:  # More than 7 days
                self.logger.warning(f"Large time range detected: {time_diff_hours:.1f} hours. This might hit API limits.")
            
            self.logger.info(f"Time range: {start_timestamp} to {end_timestamp} ({time_diff_hours:.1f} hours)")
            
            trades = self.get_my_trades_by_symbol(
                symbol=symbol,
                limit=limit,
                startTime=str(start_timestamp),
                endTime=str(end_timestamp)
            )
            
            if trades:
                self.logger.info(f"Found {len(trades)} trades for {symbol} in specified time range")
                
                # Calculate some statistics for the time range
                if trades:
                    total_volume = sum(float(trade['qty']) for trade in trades)
                    total_value = sum(float(trade['quoteQty']) for trade in trades)
                    avg_price = total_value / total_volume if total_volume > 0 else 0
                    
                    buy_trades = [t for t in trades if t.get('isBuyer', False)]
                    sell_trades = [t for t in trades if not t.get('isBuyer', False)]
                    
                    self.logger.info(f"Time range statistics for {symbol}:")
                    self.logger.info(f"  Total volume: {total_volume}")
                    self.logger.info(f"  Total value: {total_value} USDT")
                    self.logger.info(f"  Average price: {avg_price}")
                    self.logger.info(f"  Buy trades: {len(buy_trades)}")
                    self.logger.info(f"  Sell trades: {len(sell_trades)}")
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error getting my trades by time range for {symbol}: {str(e)}")
            return None

    # Add new trading methods
    def get_filled_orders_all_symbols(self):
        """
        Lấy tất cả lệnh đã khớp hoàn toàn từ Binance cho tất cả symbols
    
        Returns:
            list: Danh sách orders đã được sắp xếp theo thời gian (gần nhất trước)
        """
        try:
            self.logger.info("Starting to retrieve all filled orders from Binance...")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Lấy danh sách tất cả symbols
            symbols_list = self.get_symbols_list_by_quote_usdt()
            
            if not symbols_list:
                self.logger.warning("No symbols available for orders retrieval")
                return []
            
            self.logger.info(f"Processing orders for {len(symbols_list)} symbols")
            
            all_filled_orders = []
            processed_symbols = 0
            
            # Lặp qua từng symbol để lấy orders
            for symbol in symbols_list:
                try:
                    # Lấy tất cả orders cho symbol này
                    orders = self.spot_client.get_orders(symbol, limit=100)  # Giới hạn 100 orders mỗi symbol
                    
                    if orders:
                        # Lọc ra những lệnh đã khớp hoàn toàn (tương tự test2.py)
                        filled_orders = [order for order in orders if order['status'] == 'FILLED']
                        
                        if filled_orders:
                            # Thêm symbol vào mỗi order record để dễ tracking
                            for order in filled_orders:
                                order['symbol'] = symbol
                                all_filled_orders.append(order)
                        
                            self.logger.debug(f"Retrieved {len(filled_orders)} filled orders for {symbol}")
                
                    processed_symbols += 1
                    
                    # Log progress mỗi 20 symbols để tránh spam log
                    if processed_symbols % 20 == 0:
                        self.logger.info(f"Processed {processed_symbols}/{len(symbols_list)} symbols")
                    
                    # Tạm dừng nhỏ để tránh rate limit
                    sleep(0.05)  # Tăng delay vì get_orders có rate limit cao hơn
                
                except Exception as e:
                    self.logger.error(f"Error getting orders for {symbol}: {str(e)}")
                    continue
        
            if not all_filled_orders:
                self.logger.warning("No filled orders found for any symbol")
                return []
            
            # Sắp xếp orders theo thời gian (gần nhất ở vị trí đầu tiên)
            sorted_orders = sorted(
                all_filled_orders,
                key=lambda x: int(x['time']),  # time là timestamp trong milliseconds
                reverse=True  # Gần nhất trước
            )
            
            self.logger.info(f"Retrieved and sorted {len(sorted_orders)} total filled orders")
            
            # Log thống kê
            if sorted_orders:
                recent_symbols = list(set([order['symbol'] for order in sorted_orders[:10]]))
                self.logger.info(f"Most recently active symbols: {recent_symbols[:5]}")
                
                # Log thông tin order gần nhất
                most_recent = sorted_orders[0]
                self.logger.info(f"Most recent order: {most_recent['symbol']} - ID: {most_recent['orderId']} - Price: {most_recent['price']}")
            
            return sorted_orders
            
        except Exception as e:
            self.logger.error(f"Error in orders_total: {str(e)}", exc_info=True)
            return []

    def get_filled_orders_by_symbol(self, symbol, limit=50):
        """
        Lấy lệnh đã khớp cho một symbol cụ thể
    
        Args:
        symbol (str): Symbol cần lấy orders (e.g., 'BTCUSDT')
        limit (int): Số lượng orders tối đa (mặc định 50)
    
        Returns:
        list: Danh sách filled orders đã được sắp xếp theo thời gian
        """
        try:
            self.logger.info(f"Getting filled orders for {symbol}")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Lấy tất cả orders
            all_orders = self.spot_client.get_orders(symbol, limit=limit)
            
            if not all_orders:
                self.logger.warning(f"No orders data for {symbol}")
                return []
            
            # Lọc ra lệnh đã khớp hoàn toàn
            filled_orders = [order for order in all_orders if order['status'] == 'FILLED']
            
            # Sắp xếp theo thời gian (gần nhất trước)
            sorted_orders = sorted(
                filled_orders,
                key=lambda x: int(x['time']),
                reverse=True
            )
            
            self.logger.info(f"Retrieved {len(sorted_orders)} filled orders for {symbol}")
            return sorted_orders
            
        except Exception as e:
            self.logger.error(f"Error getting filled orders for {symbol}: {str(e)}")
            return []

    def get_pending_orders_all_symbols(self):
        """
        Lấy tất cả lệnh đang mở (chưa khớp) từ tất cả symbols
    
        Returns:
        list: Danh sách open orders
        """
        try:
            self.logger.info("Getting all open orders from Binance...")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Lấy tất cả open orders (không cần specify symbol)
            open_orders = self.spot_client.get_open_orders()
            
            if not open_orders:
                self.logger.info("No open orders found")
                return []
            
            # Sắp xếp theo thời gian
            sorted_orders = sorted(
                open_orders,
                key=lambda x: int(x['time']),
                reverse=True
            )
            
            self.logger.info(f"Retrieved {len(sorted_orders)} open orders")
            
            # Log thống kê
            if sorted_orders:
                symbols_with_orders = list(set([order['symbol'] for order in sorted_orders]))
                self.logger.info(f"Symbols with open orders: {symbols_with_orders[:10]}")
            
            return sorted_orders
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {str(e)}")            
            return []    
        
    def order_send(self, 
                symbol, 
                order_type, 
                test_mode=False, 
                quantity=None, 
                quote_quantity=None, 
                price=None,  
                time_in_force="GTC"):
        """
        Gửi lệnh mới đến Binance Exchange
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            order_type (str): Order type ('buy_market', 'sell_market', 'buy_limit', 'sell_limit')
            quantity (float, optional): Số lượng base asset (e.g., BTC amount)
            quote_quantity (float, optional): Số lượng quote asset (e.g., USDT amount)
            price (float, optional): Giá lệnh (required cho limit orders)
            time_in_force (str): Time in force ('GTC', 'IOC', 'FOK')
            test_mode (bool): If True, use test order (new_order_testing), if False use real order (new_order)
            
        Returns:
            dict: Response từ Binance API hoặc None nếu lỗi
        """
        try:
            # Validate required parameters
            if not self._validate_parameters(symbol=symbol):
                return None
            
            # Validate order type
            valid_types = ['buy_market', 'sell_market', 'buy_limit', 'sell_limit']
            if order_type.lower() not in valid_types:
                self.logger.error(f"Invalid order type: {order_type}. Must be one of {valid_types}")
                return None
            
            # Determine side and type
            if order_type.lower().startswith('buy'):
                side = "BUY"
            else:
                side = "SELL"
            
            if order_type.lower().endswith('market'):
                binance_type = "MARKET"
            else:
                binance_type = "LIMIT"
            
            # Validate parameters
            if binance_type == "LIMIT":
                if not self._validate_parameters(price=price):
                    self.logger.error("Valid price is required for limit orders")
                    return None
        
            if quantity is None and quote_quantity is None:
                self.logger.error("Either quantity or quote_quantity must be specified")
                return None
            
            if quantity is not None and not self._validate_parameters(quantity=quantity):
                return None
                
            if quote_quantity is not None and not self._validate_parameters(quote_quantity=quote_quantity):
                return None
            
            # Check spot client
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return None
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': binance_type
            }
            
            # Add quantity parameters
            if quantity is not None:
                order_params['quantity'] = str(quantity)
            elif quote_quantity is not None:
                order_params['quoteOrderQty'] = str(quote_quantity)
        
            # Add price for limit orders
            if binance_type == "LIMIT":
                order_params['price'] = str(price)
                order_params['timeInForce'] = time_in_force
        
            self.logger.info(f"Sending {'TEST' if test_mode else 'LIVE'} order: {order_type} {symbol}")
            self.logger.debug(f"Order parameters: {order_params}")
        
            # Send order to Binance (test or live)
            if test_mode:
                response = self.spot_client.new_order_test(**order_params)
                self.logger.info("Order test completed successfully - no actual trade executed")
            
                # For test orders, create a mock response structure
                if response is not None:  
                    # Test API returns {} on success
                    test_response = {
                        'orderId': 'TEST_ORDER',
                        'status': 'TEST_SUCCESS',
                        'executedQty': '0',
                        'cummulativeQuoteQty': '0',
                        'symbol': symbol,
                        'side': order_params['side'],
                        'type': order_params['type'],
                        'test_mode': True
                    }
                    return test_response
                else:
                    self.logger.error("Test order failed")
                    return None
            else:
                response = self.spot_client.new_order(**order_params)
        
            if response and not test_mode:
                # Log successful live order
                order_id = response.get('orderId', 'Unknown')
                status = response.get('status', 'Unknown')
                executed_qty = response.get('executedQty', '0')
                cumulative_quote_qty = response.get('cummulativeQuoteQty', '0')
                
                self.logger.info(f"Order successful - ID: {order_id}, Status: {status}")
                self.logger.info(f"Executed: {executed_qty}, Quote: {cumulative_quote_qty}")
                
                # Log fills if available
                if 'fills' in response and response['fills']:
                    fills_count = len(response['fills'])
                    self.logger.info(f"Order filled in {fills_count} transaction(s)")
                    
                    for i, fill in enumerate(response['fills'][:3]):  # Log first 3 fills
                        fill_price = fill.get('price', 'N/A')
                        fill_qty = fill.get('qty', 'N/A')
                        self.logger.debug(f"Fill {i+1}: {fill_qty} @ {fill_price}")
            
                return response
            elif not test_mode:
                self.logger.error("No response received from Binance")
                return None
                
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error sending order: {error_msg}")
        
            # Parse Binance API errors more comprehensively
            if "code" in error_msg and "msg" in error_msg:
                try:
                    # Try to extract JSON error from string
                    import re
                    json_match = re.search(r'\{.*\}', error_msg)
                    if json_match:
                        error_data = json.loads(json_match.group())
                        error_code = error_data.get('code', 'Unknown')
                        error_message = error_data.get('msg', 'Unknown error')
                        self.logger.error(f"Binance API Error {error_code}: {error_message}")
                        
                        # Log specific error codes for debugging
                        if error_code == -1121:
                            self.logger.error("Invalid symbol - check if symbol exists and is trading")
                        elif error_code == -2010:
                            self.logger.error("Insufficient balance for this order")
                        elif error_code == -1013:
                            self.logger.error("Invalid quantity - check minimum/maximum order size")
                        elif error_code == -1111:
                            self.logger.error("Invalid precision - check decimal places for quantity/price")
                            
                except Exception as parse_error:
                    self.logger.error(f"Error parsing Binance error response: {parse_error}")
        
            return None

    def order_send_market(self, symbol, side, test_mode=False, quote_quantity=None, quantity=None):
        """
        Gửi lệnh market đơn giản
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            side (str): 'buy' hoặc 'sell'
            quote_quantity (float, optional): Số lượng USDT muốn mua/bán
            quantity (float, optional): Số lượng base asset muốn mua/bán
            test (bool): If True, use test mode
        
        Returns:
            dict: Response từ Binance API hoặc None nếu lỗi
        """
        order_type = f"{side.lower()}_market"
        return self.order_send(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            quote_quantity=quote_quantity,
            test_mode=test_mode
        )    
    
    def order_send_limit(self, symbol, side, price, test_mode=False, quote_quantity=None, quantity=None, time_in_force="GTC"):
        """
        Gửi lệnh limit đơn giản
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            side (str): 'buy' hoặc 'sell'
            price (float): Giá lệnh
            quote_quantity (float, optional): Số lượng USDT muốn mua/bán
            quantity (float, optional): Số lượng base asset muốn mua/bán
            time_in_force (str): Time in force ('GTC', 'IOC', 'FOK')
            test (bool): If True, use test mode
        
        Returns:
            dict: Response từ Binance API hoặc None nếu lỗi
        """
        order_type = f"{side.lower()}_limit"
        return self.order_send(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            quote_quantity=quote_quantity,
            price=price,
            time_in_force=time_in_force,
            test_mode=test_mode
        )

    def cancel_order_limit(self, symbol, order_id):
        """
        Hủy lệnh theo order ID
        
        Args:
            symbol (str): Trading symbol
            order_id (int): Order ID cần hủy
        
        Returns:
            dict: Response từ Binance API hoặc None nếu lỗi
        """
        try:
            self.logger.info(f"Cancelling order {order_id} for {symbol}")
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
                
            response = self.spot_client.cancel_order(symbol=symbol, orderId=order_id)
            
            if response:
                status = response.get('status', 'Unknown')
                self.logger.info(f"Order {order_id} cancelled successfully. Status: {status}")
                return response
            else:
                self.logger.error(f"Failed to cancel order {order_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return None

    def get_order_status(self, symbol, order_id):
        """
        Kiểm tra trạng thái lệnh
        
        Args:
            symbol (str): Trading symbol
            order_id (int): Order ID
        
        Returns:
            dict: Thông tin lệnh hoặc None nếu lỗi
        """
        try:
            self.logger.info(f"Checking status of order {order_id} for {symbol}")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            response = self.spot_client.get_order(symbol=symbol, orderId=order_id)
            
            if response:
                status = response.get('status', 'Unknown')
                executed_qty = response.get('executedQty', '0')
                self.logger.info(f"Order {order_id} status: {status}, Executed: {executed_qty}")
                return response
            else:
                self.logger.error(f"Failed to get order {order_id} status")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting order {order_id} status: {str(e)}")
            return None    # Convenience methods cho các use cases phổ biến
    
    def buy_market_usdt(self, symbol, usdt_amount, test=False):
        """Mua market với số USDT cụ thể"""
        return self.order_send_market(symbol, "buy", quote_quantity=usdt_amount, test_mode=test)

    def sell_market_all_by_symbol(self, symbol, quantity, test=False):
        """Bán market toàn bộ số lượng"""
        return self.order_send_market(symbol, "sell", quantity=quantity, test_mode=test)

    def sell_market_all_by_symbol_and_cancel_orders(self, symbol):
        """
        Hủy tất cả orders của một symbol và bán hết balance
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Kết quả thực hiện
        """
        try:
            self.logger.info(f"Cancelling all orders and selling all balance for {symbol}")

            if self.spot_client is None:
                self.logger.error("Spot client is not initialized, cannot cancel order")
                return None
            
            # Bước 1: Lấy tất cả open orders cho symbol này
            open_orders = self.spot_client.get_open_orders(symbol)
            
            # Bước 2: Hủy tất cả open orders bằng batch method
            if open_orders:
                self.logger.info(f"Found {len(open_orders)} open orders for {symbol}")
                order_ids = [order['orderId'] for order in open_orders]
                
                cancel_result = self._cancel_orders_batch(symbol, order_ids)
                cancelled_orders = cancel_result.get('cancelled_orders', [])
                
                if cancel_result.get('success'):
                    self.logger.info(f"Successfully cancelled {len(cancelled_orders)} orders")
                else:
                    self.logger.warning(f"Batch cancellation completed with some failures")
                
                # Đợi một chút để đảm bảo tất cả orders đã bị hủy
                sleep(1)
            else:
                cancelled_orders = []
            
            # Bước 3: Lấy base asset bằng helper method
            base_asset = self.get_base_asset_from_symbol(symbol)
            
            if not base_asset:
                self.logger.error(f"Cannot determine base asset for {symbol}")
                return None
            
            # Bước 4: Bán toàn bộ balance
            available_balance = self.get_asset_balance(base_asset)
            
            if not available_balance or available_balance <= 0:
                self.logger.warning(f"No available balance for {base_asset} to sell")
                return {
                    'symbol': symbol,
                    'base_asset': base_asset,
                    'cancelled_orders': cancelled_orders,
                    'sell_result': None,
                    'message': 'No balance to sell'
                }
            
            sell_result = self.sell_market_all_by_symbol(symbol, available_balance)
            
            return {
                'symbol': symbol,
                'base_asset': base_asset,
                'cancelled_orders': cancelled_orders,
                'sell_result': sell_result,
                'quantity_sold': sell_result.get('executedQty', '0') if sell_result else '0',
                'usdt_received': sell_result.get('cummulativeQuoteQty', '0') if sell_result else '0'
            }
        
        except Exception as e:
            self.logger.error(f"Error in sell_market_all_by_symbol_and_cancel_orders for {symbol}: {str(e)}")
            return None
        
    def buy_limit_usdt(self, symbol, price, usdt_amount, test=False):
        """Đặt lệnh mua limit với số USDT cụ thể"""
        return self.order_send_limit(symbol, "buy", price, quote_quantity=usdt_amount, test_mode=test)

    def sell_limit_quantity(self, symbol, price, quantity, test=False):
        """Đặt lệnh bán limit với số lượng cụ thể"""
        return self.order_send_limit(symbol, "sell", price, quantity=quantity, test_mode=test)
    
    def sell_market_all_by_order_id(self, order_id):
        """
        Bán market tất cả số lượng của một order cụ thể theo order ID
        
        Args:
            order_id (int): Order ID cần bán hết
            
        Returns:
            dict: Response từ Binance API hoặc None nếu lỗi
        """
        try:
            self.logger.info(f"Starting sell market all for order ID: {order_id}")
            
            if not self._validate_parameters(order_id=order_id):
                return None
            
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized")
                return None
            
            # Bước 1: Lấy thông tin order để biết symbol và quantity
            # Vì chúng ta chỉ có order_id, cần tìm order này trong tất cả open orders
            open_orders = self.get_pending_orders_all_symbols()

            if not open_orders:
                self.logger.warning("No open orders found, cannot proceed with sell")
                return None
            
            target_order = None
            for order in open_orders:
                if order.get('orderId') == order_id:
                    target_order = order
                    break
            
            if not target_order:
                # Nếu không tìm thấy trong open orders, có thể là filled order
                # Cần tìm trong filled orders để lấy symbol
                self.logger.warning(f"Order {order_id} not found in open orders, checking filled orders...")
                
                # Tìm trong tất cả symbols - này sẽ mất thời gian
                symbols_list = self.get_symbols_list_by_quote_usdt()  
                
                for symbol in symbols_list:
                    try:
                        # Kiểm tra order trong lịch sử
                        order_info = self.spot_client.get_order(symbol=symbol, orderId=order_id)
                        if order_info:
                            target_order = order_info
                            target_order['symbol'] = symbol
                            break
                    except:
                        continue
                
            if not target_order:
                self.logger.error(f"Order {order_id} not found in any symbol")
                return None
            
            symbol = target_order['symbol']
            self.logger.info(f"Found order {order_id} for symbol {symbol}")
            
            # Bước 2: Lấy base asset bằng helper method
            base_asset = self.get_base_asset_from_symbol(symbol)
            
            if not base_asset:
                self.logger.error(f"Cannot determine base asset for {symbol}")
                return None
            
            # Bước 3: Lấy balance hiện tại của base asset
            available_balance = self.get_asset_balance(base_asset)
            
            if not available_balance or available_balance <= 0:
                self.logger.warning(f"No available balance for {base_asset} to sell")
                return None
            
            # Bước 4: Hủy order cũ nếu nó vẫn đang open
            if target_order.get('status') in ['NEW', 'PARTIALLY_FILLED']:
                self.logger.info(f"Cancelling existing order {order_id} before selling...")
                cancel_result = self._cancel_orders_batch(symbol, [order_id])
                
                if cancel_result.get('success') and order_id in cancel_result.get('cancelled_orders', []):
                    self.logger.info(f"Successfully cancelled order {order_id}")
                    # Đợi một chút để đảm bảo order đã bị hủy
                    sleep(0.5)
                else:
                    self.logger.warning(f"Failed to cancel order {order_id}, proceeding with sell anyway...")
            
            # Bước 5: Thực hiện bán market toàn bộ balance
            self.logger.info(f"Selling {available_balance} {base_asset} at market price...")
            
            sell_response = self.sell_market_all_by_symbol(symbol, available_balance)
            
            if sell_response:
                sell_order_id = sell_response.get('orderId', 'Unknown')
                executed_qty = sell_response.get('executedQty', '0')
                cumulative_quote_qty = sell_response.get('cummulativeQuoteQty', '0')
                
                self.logger.info(f"Successfully sold {executed_qty} {base_asset} for {cumulative_quote_qty} USDT")
                self.logger.info(f"New sell order ID: {sell_order_id}")
                
                return {
                    'original_order_id': order_id,
                    'sell_order_id': sell_order_id,
                    'symbol': symbol,
                    'base_asset': base_asset,
                    'quantity_sold': executed_qty,
                    'usdt_received': cumulative_quote_qty,
                    'sell_response': sell_response
                }
            else:
                self.logger.error(f"Failed to execute market sell for {base_asset}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in sell_market_all_by_order_id for order {order_id}: {str(e)}")
            return None

    def emergency_close_all(self):
        """
        Emergency function: Đóng tất cả positions và hủy tất cả orders
        
        Returns:
            dict: Tóm tắt kết quả
        """
        try:
            self.logger.warning("EMERGENCY: Closing all positions and cancelling all orders!")
            
            # Bước 1: Lấy tất cả open orders
            all_open_orders = self.get_pending_orders_all_symbols()

            if not all_open_orders:
                self.logger.warning("No open orders found, nothing to cancel")
                all_open_orders = []
            
            # Bước 2: Hủy tất cả orders bằng batch method để hiệu quả hơn
            cancelled_count = 0
            symbols_processed = set()
            
            # Group orders by symbol for batch cancellation
            orders_by_symbol = {}
            for order in all_open_orders:
                try:
                    symbol = order['symbol']
                    order_id = order['orderId']
                    
                    if symbol not in orders_by_symbol:
                        orders_by_symbol[symbol] = []
                    orders_by_symbol[symbol].append(order_id)
                    
                except Exception as e:
                    self.logger.error(f"Error processing order for cancellation: {e}")
                    continue
            
            # Cancel orders by symbol using batch method
            for symbol, order_ids in orders_by_symbol.items():
                try:
                    cancel_result = self._cancel_orders_batch(symbol, order_ids)
                    if cancel_result.get('success'):
                        cancelled_count += len(cancel_result.get('cancelled_orders', []))
                    symbols_processed.add(symbol)
                    sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error cancelling orders for {symbol}: {e}")
                    continue
            
            # Bước 3: Bán tất cả balances
            all_balances = self.get_all_assets_balances()
            sold_assets = []

            if not all_balances:
                self.logger.warning("No balances found, nothing to sell")
                return {
                    'success': True,
                    'cancelled_orders_count': cancelled_count,
                    'symbols_processed': list(symbols_processed),
                    'sold_assets': sold_assets,
                    'total_assets_sold': 0,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            # Get available symbols for validation
            available_symbols = self.get_symbols_list_by_quote_usdt()
            
            for asset, balance_info in all_balances.items():
                if asset == 'USDT':  # Skip USDT
                    continue
                    
                if balance_info['free'] > 0:
                    try:
                        # Tìm symbol tương ứng
                        symbol = f"{asset}USDT"
                        
                        # Kiểm tra symbol có tồn tại không
                        if symbol in available_symbols:
                            sell_result = self.sell_market_all_by_symbol(symbol, balance_info['free'])
                            if sell_result:
                                sold_assets.append({
                                    'asset': asset,
                                    'symbol': symbol,
                                    'quantity': sell_result.get('executedQty', '0'),
                                    'usdt_received': sell_result.get('cummulativeQuoteQty', '0'),
                                    'order_id': sell_result.get('orderId', 'Unknown')
                                })
                                self.logger.info(f"Emergency sold {asset}: {sell_result.get('executedQty', '0')}")
                            else:
                                self.logger.error(f"Failed to sell {asset}")
                            
                            sleep(0.2)
                        else:
                            self.logger.warning(f"Symbol {symbol} not available for trading")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to sell {asset}: {str(e)}")
                        continue
            
            result = {
                'success': True,
                'cancelled_orders_count': cancelled_count,
                'symbols_processed': list(symbols_processed),
                'sold_assets': sold_assets,
                'total_assets_sold': len(sold_assets),
                'total_usdt_received': sum(float(asset.get('usdt_received', 0)) for asset in sold_assets),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.warning(f"Emergency close completed: {cancelled_count} orders cancelled, {len(sold_assets)} assets sold")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in emergency close: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def on_order_event(self):
        """
        Handles crypto order events, detecting new orders and processing closed orders.
        Monitors Binance spot orders for status changes.
        """
        try:
            self.logger.debug("Crypto order event triggered - checking for order updates")
            
            # Get all current open orders from Binance
            current_orders = self.get_pending_orders_all_symbols()
            if current_orders is None:
                self.logger.warning("Failed to retrieve current orders from Binance")
                return
            
            # Create set of current order IDs for comparison
            current_order_ids = set()
            current_orders_dict = {}
            
            for order in current_orders:
                order_id = order.get('orderId')
                if order_id:
                    current_order_ids.add(order_id)
                    current_orders_dict[order_id] = order
            
            open_orders_count = len(current_order_ids)
            self.logger.info(f'Crypto order event - Current open orders: {open_orders_count}')

            # Initialize previous order tracking if needed
            if not hasattr(self, 'previous_open_order_ids'):
                self.logger.info("Initializing previous_open_order_ids for crypto order tracking")
                self.previous_open_order_ids = set()

            # Detect new orders
            new_order_ids = current_order_ids - self.previous_open_order_ids
            closed_order_ids = self.previous_open_order_ids - current_order_ids
            
            # Process new orders
            if new_order_ids:
                self.logger.info(f"Detected new crypto order(s): {new_order_ids}")
                self._process_new_crypto_orders(new_order_ids, current_orders_dict)
            
            # Process closed orders
            if closed_order_ids:
                self.logger.info(f"Detected closed crypto order(s): {closed_order_ids}")
                self._process_closed_crypto_orders(closed_order_ids)

            # Update tracking set for next event
            self.previous_open_order_ids = current_order_ids

        except Exception as e:
            self.logger.error(f"Error in crypto order event handling: {str(e)}", exc_info=True)

    def _process_new_crypto_orders(self, new_order_ids: set, current_orders_dict: dict):
        """
        Process newly opened crypto orders
        
        Args:
            new_order_ids (set): Set of new order IDs
            current_orders_dict (dict): Dictionary of current orders by ID
        """
        try:
            if not self.trade_open_callback:
                self.logger.warning(f"trade_open_callback not configured, skipping callback for new orders: {new_order_ids}")
                return

            for order_id in new_order_ids:
                try:
                    order_details = current_orders_dict.get(order_id, {})
                    
                    if not order_details:
                        self.logger.warning(f"No order details found for order_id: {order_id}")
                        continue
                    
                    # Check for special order types (equivalent to MASTER_ORDER_ in forex)
                    client_order_id = order_details.get('clientOrderId', '')
                    symbol = order_details.get('symbol', '')
                    
                    # Validate essential order data
                    if not symbol:
                        self.logger.error(f"Order {order_id} missing symbol information")
                        continue
                    
                    # Create signal data for crypto orders
                    signal_data = {
                        'order_id': order_id,
                        'client_order_id': client_order_id,
                        'symbol': symbol,
                        'side': order_details.get('side', ''),
                        'type': order_details.get('type', ''),
                        'status': order_details.get('status', ''),
                        'quantity': order_details.get('origQty', '0'),
                        'price': order_details.get('price', '0'),
                        'time_in_force': order_details.get('timeInForce', ''),
                        'time_created': order_details.get('time', 0),
                        'executed_qty': order_details.get('executedQty', '0'),
                        'cumulative_quote_qty': order_details.get('cummulativeQuoteQty', '0')
                    }
                    
                   
                    # Check for special master orders (using clientOrderId pattern)
                    if 'MASTER_ORDER_' in client_order_id:
                        self.logger.info(f"Detected MASTER_ORDER_ in clientOrderId for order {order_id}: '{client_order_id}'")
                        try:
                            self.trade_open_callback(command="MASTER_ORDER_PASSED!!!", signal_data=signal_data)
                            self.logger.info(f"Successfully processed MASTER_ORDER_ callback for order {order_id}")
                        except Exception as e:
                            self.logger.error(f"Error calling trade_open_callback for MASTER_ORDER_: {e}", exc_info=True)
                        break  # Process only the first master order found
                    
                    else:
                        # Process regular crypto order
                        self.logger.info(f"Processing new crypto order {order_id} for {symbol}")
                        try:
                            self.trade_open_callback(command="UPDATE_ORDER_STATUS", signal_data=signal_data)
                            self.logger.debug(f"Successfully processed callback for order {order_id}")
                        except Exception as e:
                            self.logger.error(f"Error calling trade_open_callback for order {order_id}: {e}", exc_info=True)

                except Exception as e:
                    self.logger.error(f"Error processing new crypto order {order_id}: {e}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Error in _process_new_crypto_orders: {e}", exc_info=True)

    def _process_closed_crypto_orders(self, closed_order_ids: set):
        """
        Process closed crypto orders
        
        Args:
            closed_order_ids (set): Set of closed order IDs
        """
        try:
            for order_id in closed_order_ids:
                try:
                    self.logger.info(f"Processing closed crypto order: {order_id}")
                    
                    # Try to get order details from recent trades or order history
                    # Note: In crypto, we might need to check filled orders to get complete info
                    filled_orders = self.get_filled_orders_all_symbols()
                    
                    closed_order_info = None
                    if filled_orders:
                        for filled_order in filled_orders:
                            if filled_order.get('orderId') == order_id:
                                closed_order_info = filled_order
                                break
                    
                    # Call trade close callback if available
                    if self.trade_close_callback and closed_order_info:
                        try:
                            # Prepare signal data for closed order
                            close_signal_data = {
                                'order_id': order_id,
                                'symbol': closed_order_info.get('symbol', ''),
                                'side': closed_order_info.get('side', ''),
                                'executed_qty': closed_order_info.get('executedQty', '0'),
                                'cumulative_quote_qty': closed_order_info.get('cummulativeQuoteQty', '0'),
                                'status': closed_order_info.get('status', 'FILLED'),
                                'commission': closed_order_info.get('commission', '0'),
                                'commission_asset': closed_order_info.get('commissionAsset', ''),
                                'time_closed': closed_order_info.get('time', 0)
                            }
                            
                            self.trade_close_callback(command="ORDER_CLOSED", signal_data=close_signal_data)
                            
                        except Exception as e:
                            self.logger.error(f"Error calling trade_close_callback for order {order_id}: {e}", exc_info=True)
                    else:
                        self.logger.debug(f"No close callback or order info available for closed order {order_id}")

                except Exception as e:
                    self.logger.error(f"Error processing closed crypto order {order_id}: {e}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Error in _process_closed_crypto_orders: {e}", exc_info=True)
    
    def stop(self):
        """
        Comprehensive resource cleanup for crypto tick_processor
        """
        try:
            self.logger.info("Starting tick processor shutdown sequence...")
            
            # Clear data caches
            if hasattr(self, 'df_cache'):
                try:
                    self.df_cache.clear()
                    self.logger.info("Data cache cleared")
                except Exception as e:
                    self.logger.warning(f"Error clearing data cache: {e}")
            
            # Clear exchange info cache
            if hasattr(self, 'exchange_info_cache'):
                try:
                    self.exchange_info_cache = None
                    self.cache_expiry = None
                    self.logger.info("Exchange info cache cleared")
                except Exception as e:
                    self.logger.warning(f"Error clearing exchange info cache: {e}")
            
            # Clear callback references to prevent memory leaks
            try:
                self.trade_open_callback = None
                self.trade_close_callback = None
                self.logger.info("Callback references cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing callback references: {e}")
            
            # Clear client reference
            try:
                self.spot_client = None
                self.logger.info("Spot client reference cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing spot client reference: {e}")
            
            # Force garbage collection to clean up memory
            try:
                import gc
                gc.collect()
                self.logger.info("Garbage collection completed")
            except Exception as e:
                self.logger.warning(f"Error during garbage collection: {e}")
            
            self.logger.info("Crypto tick processor stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping crypto tick processor: {e}", exc_info=True)
            return False    # triggers when an order is added or removed in crypto trading
