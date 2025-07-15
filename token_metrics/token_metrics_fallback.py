"""
Token Metrics API Fallback Implementation
Direct REST API calls as an alternative to the SDK
"""

import requests
import time
import sys
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from utilities.logger import setup_logging

logger = setup_logging(module_name="token_metrics_fallback", log_level=logging.INFO)

class TokenMetricsAPIError(Exception):
    """Custom exception for Token Metrics API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class TokenMetricsAPIFallback:
    """
    Token Metrics API Client using direct REST calls
    
    This is a fallback implementation when the SDK doesn't work properly
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Token Metrics API client using direct REST calls
        
        Args:
            api_key: Token Metrics API key
        """
        self.api_key = api_key
        self.base_url = "https://api.tokenmetrics.com/v2"
        self.headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'TokenMetrics-Python-Client/1.0'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        logger.info("Token Metrics fallback client initialized successfully")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make authenticated request to Token Metrics API
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            API response as dictionary
            
        Raises:
            TokenMetricsAPIError: If request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        try:
            logger.debug(f"Making GET request to {url}")
            
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
        return self._make_request(f'tokens/{symbol}')
    
    def get_token_metrics(self, symbol: str) -> Dict:
        """
        Get comprehensive metrics for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Token metrics data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_trader_grade(self, symbol: str) -> Dict:
        """
        Get trader grade for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Trader grade data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_trading_signals(self, symbol: str) -> Dict:
        """
        Get trading signals for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Trading signals data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get market sentiment for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Market sentiment data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_performance_indices(self) -> Dict:
        """
        Get performance indices data
        
        Returns:
            Performance indices data
        """
        return self._make_request('tokens')
    
    def get_top_tokens(self, limit: int = 100) -> Dict:
        """
        Get top performing tokens
        
        Args:
            limit: Number of tokens to return (max 100)
            
        Returns:
            Top tokens data
        """
        return self._make_request('tokens')
    
    def search_tokens(self, query: str) -> Dict:
        """
        Search for tokens by name or symbol
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        return self._make_request(f'tokens/{query}')
    
    def get_ai_analysis(self, symbol: str) -> Dict:
        """
        Get AI-powered analysis for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            AI analysis data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_quantitative_metrics(self, symbol: str) -> Dict:
        """
        Get quantitative metrics for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Quantitative metrics data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_resistance_support(self, symbol: str) -> Dict:
        """
        Get resistance and support levels for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Resistance and support data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_scenario_analysis(self, symbol: str) -> Dict:
        """
        Get scenario analysis for a token
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Scenario analysis data
        """
        return self._make_request(f'tokens/{symbol}')
    
    def get_multiple_tokens(self, symbols: List[str]) -> Dict:
        """
        Get data for multiple tokens at once
        
        Args:
            symbols: List of token symbols (e.g., ['BTC', 'ETH', 'ADA'])
            
        Returns:
            Data for multiple tokens
        """
        symbols_str = ','.join(symbols)
        return self._make_request('tokens', params={'symbol': symbols_str})

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
        if spec is None:
            raise ImportError("Could not create spec for config_api module")
        config_api = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Could not load config_api module")
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

def create_token_metrics_fallback_client() -> TokenMetricsAPIFallback:
    """
    Create Token Metrics API fallback client with configuration from file
    
    Returns:
        TokenMetricsAPIFallback client instance
    """
    try:
        api_key = load_api_key_from_config()
        return TokenMetricsAPIFallback(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Token Metrics fallback client: {str(e)}")
        raise

def test_fallback_api_connection():
    """Test fallback API connection and basic functionality"""
    try:
        client = create_token_metrics_fallback_client()
        logger.info("Testing Token Metrics fallback API connection...")
        
        # Test basic connectivity without parameters
        logger.info("Testing basic API connectivity...")
        try:
            all_tokens = client.get_performance_indices()
            logger.success(f"Successfully connected to Token Metrics API using fallback method")
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

if __name__ == "__main__":
    # Test the fallback API connection
    if test_fallback_api_connection():
        logger.success("Token Metrics fallback API connection test passed!")
    else:
        logger.error("Token Metrics fallback API connection test failed!") 