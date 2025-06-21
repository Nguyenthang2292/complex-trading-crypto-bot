import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Callable
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from binance.lib.utils import config_logging

from components.config_api import API_KEY, API_SECRET

class tick_processor_socket():
    def __init__(self, balance_update_callback: Optional[Callable] = None, stream_url: Optional[str] = None):
        """
        Initialize WebSocket-based tick processor for real-time balance updates
        
        Args:
            balance_update_callback: Callback function to handle balance updates
            stream_url: WebSocket stream URL (default: live-net mainnet)
        """
        self.logger = logging.getLogger(__name__)
        config_logging(logging, logging.INFO)
        
        # Store callback function
        self.balance_update_callback = balance_update_callback
        
        # WebSocket client
        self.ws_client = None
        
        # Data storage
        self.current_balances = {}
        self.last_balance_update = None
        self.is_connected = False
        
        # Stream URL - default to live-net (mainnet)
        self.stream_url = stream_url or "wss://ws-api.binance.com/ws-api/v3"
        
        # Price cache for USDT conversion
        self.price_cache = {}
        self.price_cache_expiry = {}
        
        self.logger.info("tick_processor_socket initialized")
        
    def _on_message(self, _, message):
        """
        Handle incoming WebSocket messages
        
        Args:
            _: WebSocket connection (unused)
            message: JSON message from Binance WebSocket
        """
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
                
            # Check if this is an account balance update
            if 'result' in data and 'balances' in data['result']:
                self._process_balance_update(data['result'])
            elif 'balances' in data:
                self._process_balance_update(data)
            else:
                self.logger.debug(f"Received non-balance message: {data}")
                
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _on_open(self, _):
        """Handle WebSocket connection open"""
        self.is_connected = True
        self.logger.info("WebSocket connection established")
        
    def _on_close(self, _):
        """Handle WebSocket connection close"""
        self.is_connected = False
        self.logger.warning("WebSocket connection closed")
        
        # Call custom close handler if needed
        if hasattr(self, 'custom_close_handler'):
            try:
                self.custom_close_handler()
            except Exception as e:
                self.logger.error(f"Error in custom close handler: {str(e)}")
                
    def _on_error(self, _, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {str(error)}")
        self.is_connected = False
        
    def _process_balance_update(self, balance_data):
        """
        Process balance update from WebSocket and convert to USDT values
        
        Args:
            balance_data: Balance data from Binance WebSocket
        """
        try:
            self.logger.info("Processing real-time balance update...")
            
            # Parse balance data
            if 'balances' in balance_data:
                raw_balances = balance_data['balances']
            else:
                raw_balances = balance_data
                
            # Process and convert balances
            processed_balances = {}
            total_portfolio_usdt = 0.0
            
            for balance in raw_balances:
                asset = balance.get('asset', '')
                free_balance = float(balance.get('free', 0))
                locked_balance = float(balance.get('locked', 0))
                total_balance = free_balance + locked_balance
                
                # Skip zero balances
                if total_balance <= 0:
                    continue
                    
                # Handle USDT directly
                if asset == 'USDT':
                    usdt_value = total_balance
                else:
                    # Get USDT price for other assets
                    usdt_value = self._get_asset_usdt_value(asset, total_balance)
                    
                if usdt_value > 0:
                    processed_balances[asset] = {
                        'asset': asset,
                        'free': free_balance,
                        'locked': locked_balance,
                        'total': total_balance,
                        'usdt_value': usdt_value,
                        'percentage': 0.0  # Will calculate later
                    }
                    total_portfolio_usdt += usdt_value
            
            # Calculate percentages
            if total_portfolio_usdt > 0:
                for asset_data in processed_balances.values():
                    asset_data['percentage'] = (asset_data['usdt_value'] / total_portfolio_usdt) * 100
            
            # Sort by USDT value
            sorted_balances = dict(sorted(
                processed_balances.items(),
                key=lambda x: x[1]['usdt_value'],
                reverse=True
            ))
            
            # Create result structure similar to get_all_balances_usdt
            balance_result = {
                'total_portfolio_usdt': total_portfolio_usdt,
                'assets_count': len(sorted_balances),
                'balances': sorted_balances,
                'top_5_assets': list(sorted_balances.keys())[:5],
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'websocket_realtime'
            }
            
            # Update internal storage
            self.current_balances = balance_result
            self.last_balance_update = datetime.now(timezone.utc)
            
            # Call callback function if provided
            if self.balance_update_callback:
                try:
                    self.balance_update_callback(balance_result)
                except Exception as e:
                    self.logger.error(f"Error calling balance update callback: {str(e)}")
            
            self.logger.info(f"Balance update processed: {total_portfolio_usdt:.2f} USDT across {len(sorted_balances)} assets")
            
        except Exception as e:
            self.logger.error(f"Error processing balance update: {str(e)}")
            
    def _get_asset_usdt_value(self, asset: str, quantity: float) -> float:
        """
        Get USDT value for an asset quantity using cached prices
        
        Args:
            asset: Asset symbol
            quantity: Asset quantity
            
        Returns:
            USDT value
        """
        try:
            if asset == 'USDT':
                return quantity
                
            # Check cache first
            cache_key = f"{asset}USDT"
            current_time = time.time()
            
            if (cache_key in self.price_cache and 
                cache_key in self.price_cache_expiry and
                current_time < self.price_cache_expiry[cache_key]):
                
                price = self.price_cache[cache_key]
                return quantity * price
            
            # If not in cache or expired, we need to get price
            # For real-time WebSocket, we should ideally subscribe to price streams
            # For now, return 0 and let the main processor handle price updates
            self.logger.warning(f"No cached price for {asset}, returning 0 USDT value")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting USDT value for {asset}: {str(e)}")
            return 0.0
            
    def update_price_cache(self, symbol: str, price: float, expiry_seconds: int = 60):
        """
        Update price cache for USDT conversion
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            price: Current price
            expiry_seconds: Cache expiry time in seconds
        """
        try:
            self.price_cache[symbol] = price
            self.price_cache_expiry[symbol] = time.time() + expiry_seconds
            self.logger.debug(f"Updated price cache: {symbol} = {price}")
        except Exception as e:
            self.logger.error(f"Error updating price cache: {str(e)}")
            
    def start_balance_stream(self):
        """
        Start WebSocket connection and begin streaming balance updates
        """
        try:
            self.logger.info("Starting WebSocket balance stream...")
            
            # Create WebSocket client
            self.ws_client = SpotWebsocketAPIClient(
                stream_url=self.stream_url,
                api_key=API_KEY,
                api_secret=API_SECRET,
                on_message=self._on_message,
                on_close=self._on_close,
                on_open=self._on_open,
                on_error=self._on_error
            )
            
            # Request account info to get initial balance
            self.ws_client.account()
            
            self.logger.info("WebSocket balance stream started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting WebSocket balance stream: {str(e)}")
            return False
            
    def stop_balance_stream(self):
        """
        Stop WebSocket connection
        """
        try:
            self.logger.info("Stopping WebSocket balance stream...")
            
            if self.ws_client:
                self.ws_client.stop()
                self.ws_client = None
                
            self.is_connected = False
            self.logger.info("WebSocket balance stream stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket balance stream: {str(e)}")
            return False
            
    def get_current_balances(self) -> Optional[Dict]:
        """
        Get current cached balances (similar to get_all_balances_usdt)
        
        Returns:
            Current balance data or None if not available
        """
        return self.current_balances.copy() if self.current_balances else None
        
    def is_stream_active(self) -> bool:
        """
        Check if WebSocket stream is active
        
        Returns:
            True if connected and receiving data
        """
        return self.is_connected and self.ws_client is not None
        
    def get_stream_status(self) -> Dict:
        """
        Get detailed stream status information
        
        Returns:
            Status information dictionary
        """
        return {
            'is_connected': self.is_connected,
            'has_client': self.ws_client is not None,
            'last_update': self.last_balance_update.isoformat() if self.last_balance_update else None,
            'assets_count': len(self.current_balances.get('balances', {})),
            'total_portfolio_usdt': self.current_balances.get('total_portfolio_usdt', 0.0),
            'stream_url': self.stream_url
        }
        
    def request_balance_update(self):
        """
        Manually request a balance update via WebSocket
        """
        try:
            if self.ws_client and self.is_connected:
                self.ws_client.account()
                self.logger.info("Manual balance update requested")
                return True
            else:
                self.logger.warning("WebSocket not connected, cannot request balance update")
                return False
                
        except Exception as e:
            self.logger.error(f"Error requesting balance update: {str(e)}")
            return False
            
    def set_custom_close_handler(self, handler: Callable):
        """
        Set custom handler for WebSocket close events
        
        Args:
            handler: Function to call when WebSocket closes
        """
        self.custom_close_handler = handler
        
    def cleanup(self):
        """
        Clean up resources
        """
        try:
            self.stop_balance_stream()
            self.current_balances.clear()
            self.price_cache.clear()
            self.price_cache_expiry.clear()
            self.balance_update_callback = None
            self.logger.info("tick_processor_socket cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

