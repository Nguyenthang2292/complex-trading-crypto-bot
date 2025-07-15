# Token Metrics API Integration

This project now uses the official Token Metrics Python SDK (`tmai-api`) for interacting with the Token Metrics API.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your API key in `config/config_api.py`:
```python
TOKEN_METRICS_API_KEY = "your-actual-api-key-here"
```

## Usage

### Basic Usage

```python
from token_metrics import create_token_metrics_client

# Create client
client = create_token_metrics_client()

# Get single token data
btc_data = client.get_token_price('BTC')

# Get multiple tokens data
tokens_data = client.get_multiple_tokens(['BTC', 'ETH', 'ADA'])
```

### Available Methods

The `TokenMetricsAPI` class provides the following methods:

- `get_token_price(symbol)` - Get current price and basic metrics
- `get_token_metrics(symbol)` - Get comprehensive metrics
- `get_trader_grade(symbol)` - Get trader grade information
- `get_trading_signals(symbol)` - Get trading signals
- `get_market_sentiment(symbol)` - Get market sentiment
- `get_ai_analysis(symbol)` - Get AI-powered analysis
- `get_quantitative_metrics(symbol)` - Get quantitative metrics
- `get_resistance_support(symbol)` - Get resistance and support levels
- `get_scenario_analysis(symbol)` - Get scenario analysis
- `get_multiple_tokens(symbols)` - Get data for multiple tokens at once
- `get_top_tokens(limit)` - Get top performing tokens
- `search_tokens(query)` - Search for tokens
- `get_performance_indices()` - Get performance indices

### Helper Functions

- `get_comprehensive_token_analysis(symbol)` - Get all available data for a token
- `get_multiple_tokens_analysis(symbols)` - Get comprehensive analysis for multiple tokens

### Example Usage

```python
from token_metrics import (
    create_token_metrics_client,
    get_comprehensive_token_analysis,
    get_multiple_tokens_analysis
)

# Create client
client = create_token_metrics_client()

# Get comprehensive analysis for BTC
btc_analysis = get_comprehensive_token_analysis('BTC')
print(f"BTC Analysis: {list(btc_analysis['data'].keys())}")

# Get analysis for multiple tokens
multi_analysis = get_multiple_tokens_analysis(['BTC', 'ETH', 'ADA'])
print(f"Multi-token Analysis: {len(multi_analysis['symbols'])} tokens")

# Get specific data
btc_price = client.get_token_price('BTC')
eth_signals = client.get_trading_signals('ETH')
ada_sentiment = client.get_market_sentiment('ADA')
```

## Testing

Run the test script to verify your API connection:

```bash
python token_metrics/token_metrics.py
```

Or run the example script:

```bash
python token_metrics_example.py
```

## Error Handling

The API client includes comprehensive error handling:

- `TokenMetricsAPIError` - Custom exception for API-related errors
- Automatic retry logic with exponential backoff
- Rate limiting support
- Detailed error messages with status codes

## Migration from Custom Implementation

The main changes from the previous custom implementation:

1. **Uses Official SDK**: Now uses `tmai_api.TokenMetricsClient` instead of custom HTTP requests
2. **Simplified Configuration**: No need for custom config classes, just pass the API key
3. **Better Error Handling**: Leverages the SDK's built-in error handling
4. **Multiple Tokens Support**: Added `get_multiple_tokens()` method for batch requests
5. **Maintained Interface**: All existing method signatures remain the same for backward compatibility

## Dependencies

- `tmai-api>=1.0.0` - Official Token Metrics Python SDK
- `requests>=2.28.0` - HTTP library (used by the SDK)
- Other dependencies listed in `requirements.txt`

## Configuration

Make sure your `config/config_api.py` file contains:

```python
TOKEN_METRICS_API_KEY = "your-actual-api-key-here"
```

Replace `"your-actual-api-key-here"` with your real Token Metrics API key. 