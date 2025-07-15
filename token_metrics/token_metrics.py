"""
Token Metrics API Client
A comprehensive Python client for interacting with Token Metrics API using the official SDK
"""

import sys
import time
from typing import Dict, List, Optional
from pathlib import Path
import logging
from utilities.logger import setup_logging

logger = setup_logging(module_name="token_metrics_api", log_level=logging.INFO)

# Import the official Token Metrics SDK
try:
    from tmai_api import TokenMetricsClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("Token Metrics SDK not found. Will use fallback REST API implementation.")

class TokenMetricsAPIError(Exception):
    """Custom exception for Token Metrics API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class TokenMetricsAPI:
    """
    Token Metrics API Client using REST API (SDK bypassed due to parameter issues)
    
    Supports various endpoints for crypto data analysis:
    - Token prices and metrics
    - Trader grades and signals
    - AI-powered analytics
    - Market sentiment
    - Performance indices
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Token Metrics API client using REST API
        
        Args:
            api_key: Token Metrics API key
        """
        self.api_key = api_key
        
        # Force use of REST API due to SDK parameter issues
        logger.info("Using REST API implementation (SDK bypassed due to parameter issues)")
        self.use_sdk = False
        self._init_rest_client()
    
    def _init_rest_client(self):
        """Initialize REST API client"""
        import requests
        self.base_url = "https://api.tokenmetrics.com/v2"
        self.headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'TokenMetrics-Python-Client/1.0'
        }
        self.last_request_time = 0
        self.min_request_interval = 1.0
        logger.info("Token Metrics client initialized with REST API")
    
    def _make_rest_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make REST API request"""
        import requests
        import time
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        try:
            logger.debug(f"Making REST GET request to {url}")
            
            response = requests.get(
                url=url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            self.last_request_time = time.time()
            
            # Handle different status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                raise TokenMetricsAPIError(
                    f"Bad Request: Invalid parameters or endpoint - {response.text}",
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )
            elif response.status_code == 401:
                raise TokenMetricsAPIError(
                    "Unauthorized: Invalid or missing API key",
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )
            elif response.status_code == 403:
                raise TokenMetricsAPIError(
                    "Forbidden: API key inactive or insufficient permissions",
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )
            elif response.status_code == 429:
                raise TokenMetricsAPIError(
                    "Rate limit exceeded. Please wait before making more requests.",
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )
            else:
                raise TokenMetricsAPIError(
                    f"API request failed with status {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )
                    
        except requests.exceptions.RequestException as e:
            raise TokenMetricsAPIError(f"Request failed: {str(e)}")
    
    def get_token_price(self, symbol: str) -> Dict:
        """
        Get current price and basic metrics for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Token price data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get token price for {symbol}: {str(e)}")
    
    def get_token_metrics(self, symbol: str) -> Dict:
        """
        Get comprehensive metrics for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Token metrics data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get token metrics for {symbol}: {str(e)}")
    
    def get_trader_grade(self, symbol: str) -> Dict:
        """
        Get trader grade for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Trader grade data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get trader grade for {symbol}: {str(e)}")
    
    def get_trading_signals(self, symbol: str) -> Dict:
        """
        Get trading signals for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Trading signals data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get trading signals for {symbol}: {str(e)}")
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get market sentiment for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Market sentiment data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get market sentiment for {symbol}: {str(e)}")
    
    def get_performance_indices(self) -> Dict:
        """
        Get performance indices data
        
        Returns:
            Performance indices data
        """
        try:
            response = self._make_rest_request("tokens")
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get performance indices: {str(e)}")
    
    def get_top_tokens(self, limit: int = 100) -> Dict:
        """
        Get top performing tokens
        
        Args:
            limit: Number of tokens to return (max 100)
            
        Returns:
            Top tokens data
        """
        try:
            response = self._make_rest_request("tokens")
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get top tokens: {str(e)}")
    
    def search_tokens(self, query: str) -> Dict:
        """
        Search for tokens by name or symbol
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": query})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to search tokens for '{query}': {str(e)}")
    
    def get_ai_analysis(self, symbol: str) -> Dict:
        """
        Get AI-powered analysis for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            AI analysis data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get AI analysis for {symbol}: {str(e)}")
    
    def get_quantitative_metrics(self, symbol: str) -> Dict:
        """
        Get quantitative metrics for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Quantitative metrics data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get quantitative metrics for {symbol}: {str(e)}")
    
    def get_resistance_support(self, symbol: str) -> Dict:
        """
        Get resistance and support levels for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Resistance and support data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get resistance/support for {symbol}: {str(e)}")
    
    def get_scenario_analysis(self, symbol: str) -> Dict:
        """
        Get scenario analysis for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Scenario analysis data
        """
        try:
            response = self._make_rest_request("tokens", params={"symbol": symbol})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get scenario analysis for {symbol}: {str(e)}")
    
    def get_multiple_tokens(self, symbols: List[str]) -> Dict:
        """
        Get data for multiple tokens at once
        
        Args:
            symbols: List of token symbols (e.g., ['BTC', 'ETH', 'ADA'])
            
        Returns:
            Data for multiple tokens
        """
        try:
            symbols_str = ','.join(symbols)
            response = self._make_rest_request("tokens", params={"symbol": symbols_str})
            return response
        except Exception as e:
            raise TokenMetricsAPIError(f"Failed to get data for multiple tokens: {str(e)}")

def load_api_key_from_config() -> str:
    """
    Load API key from config_api.py file
    
    Returns:
        API key string
        
    Raises:
        ImportError: If config_api module cannot be imported
        AttributeError: If TOKEN_METRICS_API_KEY not found in config
    """
    try:
        # Add config directory to path
        config_dir = Path(__file__).parent.parent / "config"
        sys.path.insert(0, str(config_dir))
        
        # Import config_api module using importlib
        import importlib.util
        config_api_path = config_dir / "config_api.py"
        spec = importlib.util.spec_from_file_location("config_api", config_api_path)
        config_api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_api)
        
        # Get API key
        api_key = getattr(config_api, 'TOKEN_METRICS_API_KEY', None)
        
        if not api_key or api_key == "YOUR_API_KEY":
            raise AttributeError("TOKEN_METRICS_API_KEY not found in config_api.py or is set to placeholder value")
        
        return api_key
        
    except ImportError as e:
        raise ImportError(f"Failed to import config_api module: {str(e)}")
    except AttributeError as e:
        raise AttributeError(f"API key error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error loading API key: {str(e)}")

def create_token_metrics_client() -> TokenMetricsAPI:
    """
    Create Token Metrics API client with configuration from file
    
    Returns:
        TokenMetricsAPI client instance
    """
    try:
        api_key = load_api_key_from_config()
        return TokenMetricsAPI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Token Metrics client: {str(e)}")
        raise

# Example usage and testing functions
def test_api_connection():
    """Test API connection and basic functionality"""
    try:
        client = create_token_metrics_client()
        logger.info("Testing Token Metrics API connection...")
        
        # First, test if we can get any tokens without specifying a symbol
        logger.info("Testing basic API connectivity...")
        try:
            all_tokens = client.get_performance_indices()
            logger.success(f"Successfully connected to Token Metrics API")
            logger.info(f"API response type: {type(all_tokens)}")
            if isinstance(all_tokens, dict):
                logger.info(f"Response keys: {list(all_tokens.keys())}")
            return True
        except Exception as e:
            logger.error(f"Basic API connectivity test failed: {str(e)}")
            return False
        
    except TokenMetricsAPIError as e:
        logger.error(f"API Error: {e.message}")
        if e.status_code:
            logger.error(f"Status Code: {e.status_code}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

def test_specific_token():
    """Test getting data for a specific token"""
    try:
        client = create_token_metrics_client()
        logger.info("Testing specific token retrieval...")
        
        # Test basic token price endpoint
        btc_data = client.get_token_price('BTC')
        logger.success(f"Successfully retrieved BTC data")
        
        # Test multiple tokens endpoint
        tokens_data = client.get_multiple_tokens(['BTC', 'ETH'])
        logger.success(f"Successfully retrieved data for multiple tokens")
        
        return True
        
    except TokenMetricsAPIError as e:
        logger.error(f"API Error: {e.message}")
        if e.status_code:
            logger.error(f"Status Code: {e.status_code}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

def get_comprehensive_token_analysis(symbol: str) -> Dict:
    """
    Get comprehensive analysis for a token including all available metrics
    
    Args:
        symbol: Token symbol (e.g., 'BTC', 'ETH')
        
    Returns:
        Comprehensive analysis data
    """
    client = create_token_metrics_client()
    
    analysis = {
        'symbol': symbol,
        'timestamp': time.time(),
        'data': {}
    }
    
    try:
        # Get comprehensive token data (includes all metrics)
        analysis['data']['comprehensive'] = client.get_token_price(symbol)
        logger.info(f"Retrieved comprehensive data for {symbol}")
    except Exception as e:
        logger.warning(f"Failed to get comprehensive data for {symbol}: {str(e)}")
    
    try:
        # Get trader grade
        analysis['data']['trader_grade'] = client.get_trader_grade(symbol)
        logger.info(f"Retrieved trader grade for {symbol}")
    except Exception as e:
        logger.warning(f"Failed to get trader grade for {symbol}: {str(e)}")
    
    try:
        # Get trading signals
        analysis['data']['signals'] = client.get_trading_signals(symbol)
        logger.info(f"Retrieved trading signals for {symbol}")
    except Exception as e:
        logger.warning(f"Failed to get trading signals for {symbol}: {str(e)}")
    
    try:
        # Get market sentiment
        analysis['data']['sentiment'] = client.get_market_sentiment(symbol)
        logger.info(f"Retrieved market sentiment for {symbol}")
    except Exception as e:
        logger.warning(f"Failed to get market sentiment for {symbol}: {str(e)}")
    
    try:
        # Get AI analysis
        analysis['data']['ai_analysis'] = client.get_ai_analysis(symbol)
        logger.info(f"Retrieved AI analysis for {symbol}")
    except Exception as e:
        logger.warning(f"Failed to get AI analysis for {symbol}: {str(e)}")
    
    return analysis

def get_multiple_tokens_analysis(symbols: List[str]) -> Dict:
    """
    Get comprehensive analysis for multiple tokens
    
    Args:
        symbols: List of token symbols (e.g., ['BTC', 'ETH', 'ADA'])
        
    Returns:
        Comprehensive analysis data for multiple tokens
    """
    client = create_token_metrics_client()
    
    analysis = {
        'symbols': symbols,
        'timestamp': time.time(),
        'data': {}
    }
    
    try:
        # Get data for all tokens at once
        analysis['data']['tokens'] = client.get_multiple_tokens(symbols)
        logger.info(f"Retrieved data for {len(symbols)} tokens")
    except Exception as e:
        logger.warning(f"Failed to get data for multiple tokens: {str(e)}")
    
    return analysis

if __name__ == "__main__":
    # Test the API connection
    if test_api_connection():
        logger.success("Token Metrics API connection test passed!")
        
        # Example: Get comprehensive analysis for BTC
        btc_analysis = get_comprehensive_token_analysis('BTC')
        logger.info(f"BTC Analysis completed. Data keys: {list(btc_analysis['data'].keys())}")
        
        # Example: Get analysis for multiple tokens
        multi_analysis = get_multiple_tokens_analysis(['BTC', 'ETH', 'ADA'])
        logger.info(f"Multiple tokens analysis completed for {len(multi_analysis['symbols'])} tokens")
    else:
        logger.error("Token Metrics API connection test failed!")