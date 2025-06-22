import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from binance.lib.utils import config_logging

from components.config_api import API_KEY, API_SECRET

class TickProcessorSocket:
    """
    Manages a WebSocket connection to Binance for real-time balance updates.

    This class establishes a connection to the Binance WebSocket API, listens for
    account-related events, processes balance updates, and provides the latest
    balance information through a callback.

    Attributes:
        logger: A configured logger for the class.
        balance_update_callback: A callback to invoke with updated balance data.
        ws_client: The Binance Spot WebSocket API client.
        current_balances: A dictionary holding the latest processed balance information.
        last_balance_update: Timestamp of the last successful balance update.
        is_connected: A boolean indicating the WebSocket connection status.
        stream_url: The URL for the WebSocket stream.
    """
    def __init__(
        self,
        balance_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stream_url: Optional[str] = None
    ):
        """Initializes the TickProcessorSocket.

        Args:
            balance_update_callback: A function to call with new balance data.
            stream_url: The WebSocket stream URL. Defaults to Binance mainnet.
        """
        self.logger = logging.getLogger(__name__)
        config_logging(logging, logging.INFO)
        
        self.balance_update_callback = balance_update_callback
        self.ws_client: Optional[SpotWebsocketAPIClient] = None
        self.current_balances: Dict[str, Any] = {}
        self.last_balance_update: Optional[datetime] = None
        self.is_connected = False
        self.stream_url = stream_url or "wss://ws-api.binance.com/ws-api/v3"
        
        self.price_cache: Dict[str, float] = {}
        self.price_cache_expiry: Dict[str, float] = {}
        
        self.logger.info("TickProcessorSocket initialized.")
        
    def _on_message(self, _: Any, message: str):
        """Handles incoming WebSocket messages."""
        try:
            data = json.loads(message)
            if 'result' in data and 'balances' in data['result']:
                self._process_balance_update(data['result'])
            elif 'balances' in data:
                self._process_balance_update(data)
            else:
                self.logger.debug(f"Received non-balance message: {data}")
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode WebSocket message: {message}")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
    
    def _on_open(self, _: Any):
        """Handles the WebSocket connection opening."""
        self.is_connected = True
        self.logger.info("WebSocket connection established.")
        
    def _on_close(self, _: Any):
        """Handles the WebSocket connection closing."""
        self.is_connected = False
        self.logger.warning("WebSocket connection closed.")
        
    def _on_error(self, _: Any, error: Exception):
        """Handles WebSocket errors."""
        self.logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        
    def _process_balance_update(self, balance_data: Dict[str, Any]):
        """Processes balance data from the WebSocket stream."""
        self.logger.info("Processing real-time balance update...")
        raw_balances = balance_data.get('balances', [])
        processed_balances = {}
        total_portfolio_usdt = 0.0
        
        for balance in raw_balances:
            asset = balance.get('asset')
            if not asset:
                continue
            
            free_balance = float(balance.get('free', 0.0))
            locked_balance = float(balance.get('locked', 0.0))
            total_balance = free_balance + locked_balance
            
            if total_balance <= 0:
                continue
            
            usdt_value = self._get_asset_usdt_value(asset, total_balance)
            if usdt_value > 0:
                processed_balances[asset] = {
                    'free': free_balance,
                    'locked': locked_balance,
                    'total': total_balance,
                    'usdt_value': usdt_value,
                }
                total_portfolio_usdt += usdt_value
        
        if total_portfolio_usdt > 0:
            for data in processed_balances.values():
                data['percentage'] = (data['usdt_value'] / total_portfolio_usdt) * 100
        
        sorted_balances = dict(sorted(
            processed_balances.items(), key=lambda item: item[1]['usdt_value'], reverse=True
        ))
        
        balance_result = {
            'total_portfolio_usdt': total_portfolio_usdt,
            'assets_count': len(sorted_balances),
            'balances': sorted_balances,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        self.current_balances = balance_result
        self.last_balance_update = datetime.now(timezone.utc)
        
        if self.balance_update_callback:
            try:
                self.balance_update_callback(balance_result)
            except Exception as e:
                self.logger.error(f"Error calling balance update callback: {e}", exc_info=True)
        
        self.logger.info(f"Balance update processed: {total_portfolio_usdt:.2f} USDT")

    def _get_asset_usdt_value(self, asset: str, quantity: float) -> float:
        """
        Gets the USDT value for an asset using a local price cache.

        Note: This method relies on an external process to populate the price
        cache via the `update_price_cache` method. It does not fetch prices
        itself.

        Args:
            asset: The asset symbol (e.g., 'BTC').
            quantity: The amount of the asset.

        Returns:
            The calculated value in USDT, or 0.0 if the price is not cached.
        """
        if asset == 'USDT':
            return quantity
        
        cache_key = f"{asset}USDT"
        current_time = time.time()
        
        if (cache_key in self.price_cache and
                self.price_cache_expiry.get(cache_key, 0) > current_time):
            return quantity * self.price_cache[cache_key]
        
        self.logger.debug(f"No valid cached price for {asset}, returning 0 USDT value.")
        return 0.0
            
    def update_price_cache(self, symbol: str, price: float, expiry_seconds: int = 60):
        """
        Updates the local price cache used for USDT conversion.

        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT').
            price: The current price of the symbol.
            expiry_seconds: The duration in seconds for which the cache is valid.
        """
        try:
            self.price_cache[symbol] = price
            self.price_cache_expiry[symbol] = time.time() + expiry_seconds
            self.logger.debug(f"Updated price cache: {symbol} = {price}")
        except Exception as e:
            self.logger.error(f"Error updating price cache: {e}", exc_info=True)
            
    def start_balance_stream(self) -> bool:
        """
        Starts the WebSocket connection and subscribes to balance updates.

        Returns:
            True if the stream was started successfully, False otherwise.
        """
        self.logger.info("Starting WebSocket balance stream...")
        if self.ws_client and self.is_connected:
            self.logger.warning("Stream is already active.")
            return True
            
        try:
            self.ws_client = SpotWebsocketAPIClient(
                stream_url=self.stream_url,
                api_key=API_KEY,
                api_secret=API_SECRET,
                on_message=self._on_message,
                on_close=self._on_close,
                on_open=self._on_open,
                on_error=self._on_error
            )
            self.ws_client.account()
            self.logger.info("WebSocket balance stream started.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket stream: {e}", exc_info=True)
            return False

    def stop_balance_stream(self):
        """Stops the WebSocket connection."""
        self.logger.info("Stopping WebSocket balance stream...")
        if self.ws_client:
            self.ws_client.stop()
            self.ws_client = None
        self.is_connected = False
        self.logger.info("WebSocket balance stream stopped.")
        
    def get_current_balances(self) -> Dict[str, Any]:
        """
        Returns the most recent balance information.

        Returns:
            A dictionary containing the latest balance data.
        """
        return self.current_balances
        
    def is_stream_active(self) -> bool:
        """
        Checks if the WebSocket stream is currently connected.

        Returns:
            True if the stream is active, False otherwise.
        """
        return self.is_connected

    def get_stream_status(self) -> Dict[str, Any]:
        """
        Gets the current status of the WebSocket connection.

        Returns:
            A dictionary with status details.
        """
        return {
            "is_connected": self.is_connected,
            "stream_url": self.stream_url,
            "last_balance_update": self.last_balance_update.isoformat() if self.last_balance_update else None
        }
        
    def request_balance_update(self):
        """Sends a request to the WebSocket for a fresh balance update."""
        if self.ws_client and self.is_connected:
            self.logger.info("Requesting a manual balance update.")
            self.ws_client.account()
        else:
            self.logger.warning("Cannot request update, stream is not active.")

    def cleanup(self):
        """Stops the stream and cleans up resources."""
        self.stop_balance_stream()
        self.price_cache.clear()
        self.price_cache_expiry.clear()
        self.current_balances.clear()
        self.logger.info("TickProcessorSocket cleaned up.")

