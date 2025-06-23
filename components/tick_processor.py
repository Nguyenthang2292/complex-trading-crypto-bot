import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from time import sleep
from typing import Dict, Optional, List, Any, Callable
import json
from binance.spot import Spot as Client

from components.config_api import API_KEY, API_SECRET

class TickProcessor:
    """
    Handles interactions with the Binance API for trading operations.

    This class provides a comprehensive interface for managing trading accounts,
    fetching market data, executing orders, and processing real-time trade events.
    It includes functionalities for Spot trading.

    Attributes:
        spot_client: An instance of the Binance Spot client.
        df_cache: A cache to store historical data DataFrames.
        last_open_time: Timestamp of the last opened trade.
        last_modification_time: Timestamp of the last modification event.
        logger: A configured logger for the class.
        trade_open_callback: A callback function for trade opening events.
        trade_close_callback: A callback function for trade closing events.
        exchange_info_cache: A cache for exchange information.
        cache_expiry: The expiry timestamp for the exchange info cache.
        cache_duration_hours: The duration for which the cache is valid.
    """
    def __init__(
        self,
        trade_open_callback: Optional[Callable[..., Any]],
        trade_close_callback: Optional[Callable[..., Any]]
    ):
        """Initializes the TickProcessor.

        Args:
            trade_open_callback: Callback function for trade opening events.
            trade_close_callback: Callback function for trade closing events.
        """
        self.spot_client = Client(api_key=API_KEY, api_secret=API_SECRET)
        self.df_cache: Dict[tuple, pd.DataFrame] = {}
        self.last_open_time = datetime.now(timezone.utc)
        self.last_modification_time = datetime.now(timezone.utc)
        self.logger = logging.getLogger(__name__)

        self.trade_open_callback = trade_open_callback
        self.trade_close_callback = trade_close_callback

        if self.trade_open_callback:
            callback_name = getattr(self.trade_open_callback, '__name__', 'anonymous')
            self.logger.info(f"Trade open callback set to: {callback_name}")
        else:
            self.logger.warning("Trade open callback is not set.")

        if self.trade_close_callback:
            callback_name = getattr(self.trade_close_callback, '__name__', 'anonymous')
            self.logger.info(f"Trade close callback set to: {callback_name}")
        else:
            self.logger.warning("Trade close callback is not set.")

        print(f"System initialized at: {datetime.now(timezone.utc)}")

        self.exchange_info_cache: Optional[Dict[str, Any]] = None
        self.cache_expiry: Optional[datetime] = None
        self.cache_duration_hours = 1
        self.previous_open_order_ids: set = set()


    def _validate_parameters(self, **kwargs: Any) -> bool:
        """Validates parameters for trading operations.

        Args:
            **kwargs: Keyword arguments to validate. Supported keys are
                'symbol', 'order_id', 'quantity', 'price', 'quote_quantity'.

        Returns:
            True if all provided parameters are valid, False otherwise.
        """
        try:
            for param_name, param_value in kwargs.items():
                if param_name == 'symbol':
                    if not isinstance(param_value, str) or len(param_value) < 3:
                        self.logger.error(f"Invalid symbol: '{param_value}'")
                        return False
                
                elif param_name == 'order_id':
                    if not isinstance(param_value, (int, str)):
                        self.logger.error(f"Invalid order_id type: {type(param_value)}")
                        return False
                    if not param_value or param_value == 0:
                        self.logger.error("Invalid order_id: cannot be None, empty, or zero.")
                        return False
                    try:
                        if int(param_value) <= 0:
                            self.logger.error(f"Invalid order_id: must be positive.")
                            return False
                    except (ValueError, TypeError):
                        self.logger.error(f"Invalid order_id format: '{param_value}'")
                        return False
        
                elif param_name in ['quantity', 'price', 'quote_quantity']:
                    if not isinstance(param_value, (int, float)) or param_value <= 0:
                        self.logger.error(f"Invalid {param_name}: '{param_value}', must be a positive number.")
                        return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating parameters: {e}", exc_info=True)
            return False

    def _cancel_orders_batch(self, symbol: str, order_ids: List[int], max_retries: int = 3) -> Dict[str, Any]:
        """Cancels a batch of orders with error handling and retries.

        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT').
            order_ids: A list of order IDs to cancel.
            max_retries: The maximum number of retry attempts per order.

        Returns:
            A dictionary summarizing the cancellation results.
        """
        if not self._validate_parameters(symbol=symbol):
            return {'success': False, 'error': 'Invalid parameters'}
        
        if not order_ids:
            return {'success': True, 'cancelled_orders': [], 'failed_orders': []}
        
        cancelled_orders, failed_orders = [], []
        self.logger.info(f"Cancelling {len(order_ids)} orders for {symbol}")

        if self.spot_client is None:
            self.logger.error("Spot client not initialized, cannot cancel orders.")
            return {'success': False, 'error': 'Spot client not initialized'}
        
        for order_id in order_ids:
            if not self._validate_parameters(order_id=order_id):
                failed_orders.append({'order_id': order_id, 'error': 'Invalid order_id'})
                continue
            
            for attempt in range(max_retries):
                try:
                    response = self.spot_client.cancel_order(symbol=symbol, orderId=int(order_id))
                    if response and response.get('status') == 'CANCELED':
                        cancelled_orders.append(order_id)
                        self.logger.info(f"Successfully cancelled order {order_id}")
                        break
                    self.logger.warning(f"Attempt {attempt + 1} to cancel order {order_id} did not confirm cancellation.")
                    sleep(0.1 * (attempt + 1))
                except Exception as e:
                    error_msg = str(e)
                    if attempt >= max_retries - 1:
                        failed_orders.append({'order_id': order_id, 'error': error_msg})
                        self.logger.error(f"Failed to cancel order {order_id} after {max_retries} attempts: {error_msg}")
                    else:
                        self.logger.warning(f"Attempt {attempt + 1} to cancel order {order_id} failed: {error_msg}. Retrying...")
                        sleep(0.1 * (attempt + 1))
        
        result = {
            'success': not failed_orders,
            'cancelled_orders': cancelled_orders,
            'failed_orders': failed_orders,
        }
        
        self.logger.info(
            f"Batch cancellation complete: {len(cancelled_orders)}/{len(order_ids)} orders cancelled."
        )
        return result

    def _klines_to_dataframe(self, klines_data: List[List[Any]]) -> Optional[pd.DataFrame]:
        """Converts raw k-lines data from Binance to a pandas DataFrame.

        Args:
            klines_data: The raw k-lines data from the Binance API.

        Returns:
            A pandas DataFrame with structured k-line data, or None if an error occurs.
        """
        try:
            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            df = pd.DataFrame(klines_data, columns=columns)  # type: ignore
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_cols = [
                'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
            ]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            df['number_of_trades'] = df['number_of_trades'].astype(int)
            df = df.drop(columns=['ignore'])
            df.index.name = "time"
            
            return df
        except Exception as e:
            self.logger.error(f"Error converting klines to DataFrame: {e}", exc_info=True)
            return None
        
    def get_account_info(self) -> Dict[str, Any]:
        """Retrieves account information from Binance.

        Returns:
            A dictionary containing account information, or an empty dictionary on error.
        """
        try:
            self.logger.info("Fetching account information from Binance...")
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return {}
            
            account_info = self.spot_client.account()
            self.logger.info("Successfully fetched account information.")
            return account_info
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}", exc_info=True)
            return {}

    def get_exchange_info_cached(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieves exchange information, using a cache to avoid repeated API calls.

        Args:
            force_refresh: If True, forces a refresh of the cache from the API.

        Returns:
            A dictionary containing exchange information, or None if an error occurs.
        """
        current_time = datetime.now(timezone.utc)
        
        if (not force_refresh and self.exchange_info_cache and self.cache_expiry and
                current_time < self.cache_expiry):
            self.logger.debug("Using cached exchange info.")
            return self.exchange_info_cache
        
        try:
            self.logger.info("Fetching fresh exchange info from Binance API.")
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return None

            exchange_info = self.spot_client.exchange_info()
            if not exchange_info or 'symbols' not in exchange_info:
                self.logger.error("Failed to get valid exchange info from Binance.")
                return None
            
            self.exchange_info_cache = exchange_info
            self.cache_expiry = current_time + timedelta(hours=self.cache_duration_hours)
            self.logger.info(f"Exchange info cached. Expires at: {self.cache_expiry}")
            return self.exchange_info_cache
        except Exception as e:
            self.logger.error(f"Error getting exchange info: {e}", exc_info=True)
            return None

    def get_symbols_info(self) -> Dict[str, Any]:
        """Gets detailed information for all trading symbols.

        Returns:
            A dictionary where keys are symbol names and values are their details.
        """
        self.logger.info("Fetching detailed symbol information...")
        exchange_info = self.get_exchange_info_cached()
        if not exchange_info:
            return {}
        
        try:
            symbols_info = {
                s['symbol']: {
                    'baseAsset': s['baseAsset'],
                    'quoteAsset': s['quoteAsset'],
                    'status': s['status'],
                    'orderTypes': s['orderTypes'],
                    'filters': s['filters']
                }
                for s in exchange_info.get('symbols', []) if s.get('status') == 'TRADING'
            }
            self.logger.info(f"Retrieved info for {len(symbols_info)} active symbols.")
            return symbols_info
        except Exception as e:
            self.logger.error(f"Error processing symbols info: {e}", exc_info=True)
            return {}
    
    def get_base_asset_from_symbol(self, symbol: str) -> Optional[str]:
        """Extracts the base asset from a trading symbol.

        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT').

        Returns:
            The base asset (e.g., 'BTC'), or None if not found or on error.
        """
        if not self._validate_parameters(symbol=symbol):
            return None
            
        exchange_info = self.get_exchange_info_cached()
        if not exchange_info:
            return None
        
        try:
            for symbol_data in exchange_info.get('symbols', []):
                if symbol_data.get('symbol') == symbol:
                    base_asset = symbol_data.get('baseAsset')
                    self.logger.debug(f"Found base asset '{base_asset}' for symbol '{symbol}'")
                    return base_asset
            
            self.logger.warning(f"Base asset not found for symbol: {symbol}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting base asset for {symbol}: {e}", exc_info=True)
            return None
    
    def get_symbol_price(self, symbol: str) -> Optional[float]:
        """Gets the current price of a trading symbol.

        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT').

        Returns:
            The current price as a float, or None if an error occurs.
        """
        if not self._validate_parameters(symbol=symbol):
            return None
        
        try:
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return None
            
            price_data = self.spot_client.ticker_price(symbol)
            if price_data and 'price' in price_data:
                return float(price_data['price'])
            
            self.logger.warning(f"No price data available for {symbol}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}", exc_info=True)
            return None
        
    def get_asset_price_usdt(self, asset: str) -> Optional[float]:
        """Gets the price of an asset in USDT.

        Args:
            asset: The asset symbol (e.g., 'BTC', 'ETH').

        Returns:
            The price of the asset in USDT, or None if not found or on error.
        """
        if not isinstance(asset, str) or not asset:
            self.logger.error("Invalid asset provided.")
            return None

        if asset == 'USDT':
            return 1.0
        
        if self.spot_client is None:
            self.logger.error("Spot client is not initialized.")
            return None

        try:
            direct_symbol = f"{asset}USDT"
            price_data = self.spot_client.ticker_price(direct_symbol)
            if price_data and 'price' in price_data:
                return float(price_data['price'])
        except Exception:
            self.logger.debug(f"Direct symbol {direct_symbol} not found. Trying reverse.")

        try:
            reverse_symbol = f"USDT{asset}"
            price_data = self.spot_client.ticker_price(reverse_symbol)
            price = float(price_data.get('price'))
            if price > 0:
                return 1.0 / price
        except Exception:
            self.logger.warning(f"Could not determine USDT price for asset: {asset}")
        
        return None
        
    def get_asset_balance(self, asset: str) -> Optional[float]:
        """Retrieves the 'free' balance of a specific asset.

        Args:
            asset: The asset symbol (e.g., 'BTC', 'ETH', 'USDT').

        Returns:
            The free balance of the asset as a float, or 0.0 if not found.
        """
        if not isinstance(asset, str) or not asset:
            self.logger.error("Invalid asset provided.")
            return None

        try:
            self.logger.info(f"Fetching balance for asset: {asset}")
            account_info = self.get_account_info()
            
            if not account_info or "balances" not in account_info:
                self.logger.error("Failed to get account information from Binance.")
                return None
            
            for balance in account_info["balances"]:
                if balance.get("asset") == asset:
                    free_balance = float(balance.get("free", 0.0))
                    self.logger.info(f"Balance for {asset}: {free_balance}")
                    return free_balance
            
            self.logger.warning(f"Asset '{asset}' not found in account balances.")
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting balance for {asset}: {e}", exc_info=True)
            return None

    def get_all_assets_balances(self) -> Dict[str, Dict[str, float]]:
        """Retrieves all non-zero asset balances from the account.

        Returns:
            A dictionary of asset balances, or an empty dictionary on error.
        """
        self.logger.info("Fetching all asset balances...")
        account_info = self.get_account_info()
        if not account_info or "balances" not in account_info:
            return {}
        
        try:
            balances = {
                b['asset']: {
                    'free': float(b.get('free', 0.0)),
                    'locked': float(b.get('locked', 0.0)),
                    'total': float(b.get('free', 0.0)) + float(b.get('locked', 0.0))
                }
                for b in account_info["balances"]
                if float(b.get('free', 0.0)) + float(b.get('locked', 0.0)) > 0
            }
            self.logger.info(f"Retrieved {len(balances)} assets with non-zero balance.")
            return balances
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error processing asset balances: {e}", exc_info=True)
            return {}

    def get_all_balances_usdt(self) -> Dict[str, Any]:
        """Retrieves all asset balances and their approximate value in USDT.

        Returns:
            A dictionary containing portfolio summary in USDT.
        """
        self.logger.info("Fetching all balances and converting to USDT value...")
        all_balances = self.get_all_assets_balances()
        if not all_balances:
            return {}

        balances_usdt = {}
        total_portfolio_usdt = 0.0

        for asset, balance_info in all_balances.items():
            usdt_price = self.get_asset_price_usdt(asset)
            if usdt_price is None:
                self.logger.warning(f"Skipping {asset} as its USDT price could not be determined.")
                continue

            usdt_value = balance_info['total'] * usdt_price
            balances_usdt[asset] = {**balance_info, 'usdt_value': usdt_value}
            total_portfolio_usdt += usdt_value
            sleep(0.01)

        if total_portfolio_usdt > 0:
            for data in balances_usdt.values():
                data['percentage'] = (data['usdt_value'] / total_portfolio_usdt) * 100

        sorted_balances = dict(sorted(
            balances_usdt.items(), key=lambda item: item[1]['usdt_value'], reverse=True
        ))

        result = {
            'total_portfolio_usdt': total_portfolio_usdt,
            'assets_count': len(sorted_balances),
            'balances': sorted_balances,
        }
        self.logger.info(
            f"Portfolio value: {total_portfolio_usdt:.2f} USDT across {len(sorted_balances)} assets."
        )
        return result

    def get_symbols_list_by_quote_asset(self, quote_asset: str = 'USDT') -> List[str]:
        """Gets a list of all symbols for a given quote asset.

        Args:
            quote_asset: The quote asset to filter by (e.g., 'USDT', 'BTC').

        Returns:
            A list of trading symbols.
        """
        self.logger.info(f"Fetching symbol list for '{quote_asset}' pairs...")
        exchange_info = self.get_exchange_info_cached()
        if not exchange_info:
            return []

        try:
            symbols = [
                s['symbol'] for s in exchange_info.get('symbols', [])
                if s.get('quoteAsset') == quote_asset and s.get('status') == 'TRADING'
            ]
            self.logger.info(f"Found {len(symbols)} active '{quote_asset}' trading pairs.")
            return symbols
        except Exception as e:
            self.logger.error(f"Error getting symbols list: {e}", exc_info=True)
            return []

    def get_symbols_list_by_quote_assets(self, quote_assets: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Gets symbols grouped by specified quote assets.

        Args:
            quote_assets: A list of quote assets to filter by.

        Returns:
            A dictionary of symbols grouped by quote asset.
        """
        if quote_assets is None:
            quote_assets = ['USDT', 'BTC', 'ETH', 'BNB']
        
        self.logger.info(f"Fetching symbols grouped by quote assets: {quote_assets}")
        exchange_info = self.get_exchange_info_cached()
        if not exchange_info:
            return {}

        try:
            grouped_symbols = {asset: [] for asset in quote_assets}
            for symbol_data in exchange_info.get('symbols', []):
                if (symbol_data.get('status') == 'TRADING' and 
                        symbol_data.get('quoteAsset') in quote_assets):
                    quote = symbol_data['quoteAsset']
                    grouped_symbols[quote].append(symbol_data['symbol'])
            
            for asset, symbols in grouped_symbols.items():
                self.logger.info(f"Found {len(symbols)} '{asset}' pairs.")
            return grouped_symbols
        except Exception as e:
            self.logger.error(f"Error getting grouped symbols: {e}", exc_info=True)
            return {}
        
    def get_symbols_list_by_quote_usdt(self) -> List[str]:
        """Gets a list of all symbols for USDT pairs.

        Returns:
            A list of trading symbols.
        """
        return self.get_symbols_list_by_quote_asset('USDT')
        
    def get_historic_data_by_symbol(
        self, symbol: str, timeframe: str, num_candles: int = 450
    ) -> Optional[pd.DataFrame]:
        """Fetches historical k-line data for a symbol and caches it.

        Args:
            symbol: The trading symbol.
            timeframe: The timeframe for the k-lines.
            num_candles: The number of candles to retrieve.

        Returns:
            A DataFrame with historical data, or None on error.
        """
        if not self._validate_parameters(symbol=symbol) or not timeframe:
            self.logger.error(f"Invalid parameters for historic data: symbol='{symbol}', timeframe='{timeframe}'")
            return None

        cache_key = (symbol, timeframe)
        if cache_key in self.df_cache:
            self.logger.info(f"Returning cached data for {symbol} {timeframe}.")
            return self.df_cache[cache_key]

        try:
            self.logger.info(f"Requesting {num_candles} candles of {symbol} {timeframe} data...")
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return None
            
            klines = self.spot_client.klines(symbol, timeframe, limit=num_candles)
            if not klines:
                self.logger.warning(f"No k-line data received for {symbol} {timeframe}.")
                return None
            
            df = self._klines_to_dataframe(klines)
            if df is not None and not df.empty:
                self.df_cache[cache_key] = df
                self.logger.info(f"Cached {len(df)} candles for {symbol} {timeframe}.")
                return df
            
            self.logger.error(f"Failed to process k-line data for {symbol} {timeframe}.")
            return None
        except Exception as e:
            self.logger.error(f"Error getting historic data for {symbol} {timeframe}: {e}", exc_info=True)
            return None

    def get_historic_trades_by_symbol(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetches historical market trades for a specific symbol.

        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT').
            limit: The number of trades to retrieve (max 1000).

        Returns:
            A list of historical trade data, sorted by time descending.
        """
        self.logger.info(f"Getting historic trades for {symbol}")
        if not self._validate_parameters(symbol=symbol):
            return []

        try:
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return []
            
            trades = self.spot_client.historical_trades(symbol, limit=limit)
            if not trades:
                self.logger.warning(f"No historic trades data for {symbol}")
                return []
            
            sorted_trades = sorted(trades, key=lambda x: int(x['time']), reverse=True)
            self.logger.info(f"Retrieved {len(sorted_trades)} trades for {symbol}")
            return sorted_trades
        except Exception as e:
            self.logger.error(f"Error getting historic trades for {symbol}: {e}", exc_info=True)
            return []
    
    def get_my_trades_by_symbol(
        self,
        symbol: str,
        limit: int = 500,
        from_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches personal account trades for a specific symbol.

        Args:
            symbol: The trading symbol.
            limit: The number of trades to retrieve (max 1000).
            from_id: The trade ID to fetch from.
            start_time: The start timestamp in milliseconds.
            end_time: The end timestamp in milliseconds.

        Returns:
            A list of personal trade data, sorted by time descending.
        """
        self.logger.info(f"Getting my trades for {symbol}")
        if not self._validate_parameters(symbol=symbol):
            return []

        try:
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return []
                
            params = {'symbol': symbol, 'limit': limit}
            if from_id is not None:
                params['fromId'] = from_id
            if start_time is not None:
                params['startTime'] = start_time
            if end_time is not None:
                params['endTime'] = end_time

            my_trades = self.spot_client.my_trades(**params)
            if not my_trades:
                self.logger.info(f"No trades found for {symbol} with given parameters.")
                return []

            sorted_trades = sorted(my_trades, key=lambda x: int(x['time']), reverse=True)
            self.logger.info(f"Retrieved {len(sorted_trades)} of my trades for {symbol}")
            return sorted_trades
        except Exception as e:
            self.logger.error(f"Error getting my trades for {symbol}: {e}", exc_info=True)
            return []

    def get_my_trades_all_symbols(self, limit_per_symbol: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetches personal trades for all symbols with a non-zero balance.

        Args:
            limit_per_symbol: The number of trades to retrieve per symbol.

        Returns:
            A dictionary mapping symbols to a list of their trades.
        """
        self.logger.info("Getting my trades for all symbols with balance...")
        balances = self.get_all_assets_balances()
        symbols_to_check = {f"{asset}USDT" for asset in balances if asset != 'USDT'}
        
        all_my_trades = {}
        for symbol in symbols_to_check:
            try:
                trades = self.get_my_trades_by_symbol(symbol, limit=limit_per_symbol)
                if trades:
                    all_my_trades[symbol] = trades
                sleep(0.1)  # Avoid rate limiting
            except Exception as e:
                self.logger.error(f"Error fetching trades for {symbol}: {e}")
                continue
        
        self.logger.info(f"Retrieved trades for {len(all_my_trades)} symbols.")
        return all_my_trades

    def get_filled_orders_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetches filled orders for a specific symbol.

        Args:
            symbol: The trading symbol.
            limit: The number of orders to retrieve.

        Returns:
            A list of filled orders, sorted by time descending.
        """
        self.logger.info(f"Getting filled orders for {symbol}")
        if not self._validate_parameters(symbol=symbol):
            return []
            
        try:
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return []
                
            orders = self.spot_client.get_orders(symbol, limit=limit)
            if not orders:
                self.logger.info(f"No orders found for {symbol}.")
                return []

            filled_orders = [o for o in orders if o.get('status') == 'FILLED']
            sorted_orders = sorted(filled_orders, key=lambda x: int(x['time']), reverse=True)
            self.logger.info(f"Retrieved {len(sorted_orders)} filled orders for {symbol}")
            return sorted_orders
        except Exception as e:
            self.logger.error(f"Error getting filled orders for {symbol}: {e}", exc_info=True)
            return []

    def get_pending_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetches all open (pending) orders for a specific symbol.

        Args:
            symbol: The trading symbol.

        Returns:
            A list of open orders for the symbol.
        """
        if not self._validate_parameters(symbol=symbol):
            return []
        
        all_open_orders = self.get_pending_orders_all_symbols()
        return [order for order in all_open_orders if order.get('symbol') == symbol]

    def get_pending_orders_all_symbols(self) -> List[Dict[str, Any]]:
        """
        Fetches all open (pending) orders for the account.

        Returns:
            A list of open orders.
        """
        self.logger.info("Getting all open orders...")
        try:
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return []
                
            open_orders = self.spot_client.get_open_orders()
            if not open_orders:
                self.logger.info("No open orders found.")
                return []

            self.logger.info(f"Retrieved {len(open_orders)} open orders.")
            return open_orders
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}", exc_info=True)
            return []    
        
    def order_send(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        quote_quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        test_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Sends a new order to the exchange.

        Args:
            symbol: The trading symbol.
            side: 'BUY' or 'SELL'.
            order_type: 'MARKET' or 'LIMIT'.
            quantity: The amount of the base asset.
            quote_quantity: The amount of the quote asset (for market orders).
            price: The price for limit orders.
            time_in_force: The time in force for limit orders.
            test_mode: If True, send a test order.

        Returns:
            The API response from the exchange.
        """
        if not self._validate_parameters(symbol=symbol):
            return None
        
        order_params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper()
        }

        if quantity is not None:
            order_params['quantity'] = str(quantity)
        if quote_quantity is not None:
            order_params['quoteOrderQty'] = str(quote_quantity)
        if price is not None:
            order_params['price'] = str(price)
        if order_type.upper() == 'LIMIT':
            order_params['timeInForce'] = time_in_force
            
        self.logger.info(f"Sending {'TEST' if test_mode else 'LIVE'} order: {order_params}")

        try:
            if self.spot_client is None:
                self.logger.error("Spot client is not initialized.")
                return None

            if test_mode:
                response = self.spot_client.new_order_test(**order_params)
                self.logger.info("Test order successful.")
                return response if response else {'status': 'TEST_SUCCESS'}
            else:
                response = self.spot_client.new_order(**order_params)
                self.logger.info(f"Live order response: {response}")
                return response
        except Exception as e:
            self.logger.error(f"Error sending order: {e}", exc_info=True)
            return None

    def sell_market_all_by_symbol_and_cancel_orders(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Cancels all open orders for a symbol and sells the entire balance.

        Args:
            symbol: The trading symbol.

        Returns:
            A dictionary with the results of the cancellation and sale.
        """
        self.logger.info(f"Initiating cancel all and sell all for {symbol}...")
        
        try:
            open_orders = self.get_pending_orders_by_symbol(symbol)
            if open_orders:
                order_ids = [o['orderId'] for o in open_orders]
                cancel_result = self._cancel_orders_batch(symbol, order_ids)
                self.logger.info(f"Cancellation result for {symbol}: {cancel_result}")
                sleep(1)

            base_asset = self.get_base_asset_from_symbol(symbol)
            if not base_asset:
                self.logger.error(f"Could not determine base asset for {symbol}.")
                return None
                
            balance = self.get_asset_balance(base_asset)
            if balance is None or balance <= 0:
                self.logger.warning(f"No balance of {base_asset} to sell for {symbol}.")
                return {'message': 'No balance to sell.'}

            sell_result = self.order_send(symbol, 'SELL', 'MARKET', quantity=balance)
            return {'cancellation': cancel_result, 'sell_order': sell_result}
            
        except Exception as e:
            self.logger.error(f"Error in emergency close for {symbol}: {e}", exc_info=True)
            return None
        
    def emergency_close_all(self) -> Dict[str, Any]:
        """
        Cancels all open orders and sells all non-USDT assets.

        Returns:
            A summary of the actions taken.
        """
        self.logger.warning("EMERGENCY: Closing all positions and cancelling all orders!")
        
        all_open_orders = self.get_pending_orders_all_symbols()
        orders_by_symbol = {}
        for order in all_open_orders:
            symbol = order['symbol']
            if symbol not in orders_by_symbol:
                orders_by_symbol[symbol] = []
            orders_by_symbol[symbol].append(order['orderId'])
            
        cancellation_results = {}
        for symbol, order_ids in orders_by_symbol.items():
            cancellation_results[symbol] = self._cancel_orders_batch(symbol, order_ids)
            sleep(0.1)
        
        all_balances = self.get_all_assets_balances()
        sell_results = {}
        for asset, balance_info in all_balances.items():
            if asset == 'USDT' or balance_info.get('free', 0) <= 0:
                continue
            
            symbol = f"{asset}USDT"
            # Basic check if symbol might exist
            if self.get_symbol_price(symbol) is not None:
                 sell_results[asset] = self.order_send(symbol, 'SELL', 'MARKET', quantity=balance_info['free'])
                 sleep(0.2)
        
        return {
            'cancellations': cancellation_results,
            'sells': sell_results
        }
    
    def on_order_event(self):
        """
        Handles order update events by checking for new and closed orders.
        """
        self.logger.debug("Checking for order updates...")
        current_orders = self.get_pending_orders_all_symbols()
        if current_orders is None:
            return

        current_order_ids = {o['orderId'] for o in current_orders}
        new_order_ids = current_order_ids - self.previous_open_order_ids
        closed_order_ids = self.previous_open_order_ids - current_order_ids

        if new_order_ids:
            self.logger.info(f"Detected new order(s): {new_order_ids}")
            self._process_new_crypto_orders(new_order_ids, {o['orderId']: o for o in current_orders})

        if closed_order_ids:
            self.logger.info(f"Detected closed order(s): {closed_order_ids}")
            self._process_closed_crypto_orders(closed_order_ids)

        self.previous_open_order_ids = current_order_ids

    def _process_new_crypto_orders(self, new_order_ids: set, current_orders_dict: dict):
        """Processes new crypto orders and triggers the open callback."""
        if not self.trade_open_callback:
            return

        for order_id in new_order_ids:
            order_details = current_orders_dict.get(order_id)
            if order_details:
                self.trade_open_callback(command="UPDATE_ORDER_STATUS", signal_data=order_details)

    def _process_closed_crypto_orders(self, closed_order_ids: set):
        """Processes closed crypto orders and triggers the close callback."""
        if not self.trade_close_callback:
            return

        for order_id in closed_order_ids:
            # We need to fetch the details of the closed order
            # This is a simplification; a robust implementation would need to query order history
            self.trade_close_callback(command="ORDER_CLOSED", signal_data={'order_id': order_id})
    
    def stop(self):
        """
        Cleans up resources used by the TickProcessor.
        """
        self.logger.info("Stopping TickProcessor and cleaning up resources...")
        self.df_cache.clear()
        self.exchange_info_cache = None
        self.cache_expiry = None
        self.trade_open_callback = None
        self.trade_close_callback = None
        self.spot_client = None
        self.logger.info("TickProcessor stopped.")
