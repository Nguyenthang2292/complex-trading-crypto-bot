# Token Metrics API Implementation Guide

## 🔍 **Problem Analysis**

Based on the error logs, the issue is that the `tmai_api` SDK is automatically adding unsupported parameters (`limit=1000&page=0`) to API requests, causing 400 Bad Request errors.

**Error Pattern:**
```
400 Client Error: Bad Request for url: https://api.tokenmetrics.com/v2/tokens?symbol=BTC&limit=1000&page=0
```

## 🛠️ **Solution Implemented**

### **1. REST API Implementation (Primary Solution)**

I've updated the implementation to **bypass the SDK entirely** and use direct REST API calls:

```python
class TokenMetricsAPI:
    def __init__(self, api_key: str):
        # Force use of REST API due to SDK parameter issues
        self.use_sdk = False
        self._init_rest_client()
    
    def _make_rest_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        # Direct HTTP requests without problematic parameters
        response = requests.get(url, headers=self.headers, params=params)
        # Proper error handling for different status codes
```

### **2. Correct API Endpoints**

Based on the [Token Metrics API documentation](https://developers.tokenmetrics.com/reference/tokens), the correct endpoints are:

- **Get All Tokens**: `GET /v2/tokens`
- **Get Specific Token**: `GET /v2/tokens/{symbol}`
- **Get Multiple Tokens**: `GET /v2/tokens?symbol=BTC,ETH,ADA`

### **3. Proper Error Handling**

The implementation now handles all HTTP status codes correctly:

- **400 Bad Request**: Invalid parameters or endpoint
- **401 Unauthorized**: Invalid API key
- **403 Forbidden**: Insufficient permissions
- **429 Rate Limited**: Too many requests

## 🧪 **Testing Strategy**

### **Test 1: Main Implementation**
```bash
python token_metrics/token_metrics.py
```

### **Test 2: Fallback Implementation**
```bash
python token_metrics/token_metrics_fallback.py
```

### **Test 3: Comprehensive Test Suite**
```bash
python test_token_metrics_api.py
```

### **Test 4: Direct REST API Call**
```python
import requests

headers = {
    'x-api-key': 'your-api-key',
    'Content-Type': 'application/json'
}

response = requests.get(
    "https://api.tokenmetrics.com/v2/tokens",
    headers=headers
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

## 📋 **API Key Verification**

### **1. Check API Key Format**
Your API key should be in `config/config_api.py`:
```python
TOKEN_METRICS_API_KEY = "your-actual-api-key-here"
```

### **2. Verify API Key Status**
- Log into your Token Metrics account
- Check API key status in the dashboard
- Verify subscription plan and usage limits

### **3. Test API Key Manually**
```bash
curl -H "x-api-key: your-api-key" https://api.tokenmetrics.com/v2/tokens
```

## 🔧 **Troubleshooting Steps**

### **Step 1: Verify API Key**
1. Check `config/config_api.py` file
2. Ensure API key is valid and active
3. Verify subscription plan limits

### **Step 2: Test Basic Connectivity**
```bash
python test_token_metrics_api.py
```

### **Step 3: Check Network Connectivity**
```bash
curl -I https://api.tokenmetrics.com/v2/tokens
```

### **Step 4: Verify API Plan**
According to the documentation, check your plan:
- **Basic**: 5,000 calls/month, 20 req/min
- **Advanced**: 20,000 calls/month, 60 req/min
- **Premium**: 100,000 calls/month, 180 req/min
- **VIP**: 500,000 calls/month, 600 req/min

## 🚀 **Usage Examples**

### **Command Line Interface**
```bash
python token_metrics/token_metrics_main.py
```

### **Programmatic Usage**
```python
from token_metrics import create_token_metrics_client

# Create client (uses REST API automatically)
client = create_token_metrics_client()

# Get token data
btc_data = client.get_token_price('BTC')
eth_data = client.get_token_metrics('ETH')

# Get multiple tokens
tokens_data = client.get_multiple_tokens(['BTC', 'ETH', 'ADA'])
```

## 📊 **Expected Response Format**

Based on the API documentation, responses should include:

```json
{
  "data": [
    {
      "symbol": "BTC",
      "name": "Bitcoin",
      "price": 50000.0,
      "market_cap": 1000000000,
      "volume_24h": 50000000,
      "change_24h": 2.5
    }
  ],
  "meta": {
    "total": 100,
    "page": 1,
    "limit": 10
  }
}
```

## 🔄 **Fallback Options**

If the Token Metrics API continues to have issues:

### **1. Alternative APIs**
- **CoinGecko API**: Free, no API key required
- **CoinMarketCap API**: Comprehensive data
- **Binance API**: Real-time trading data

### **2. Mock Data Implementation**
```python
def get_mock_token_data(symbol: str):
    return {
        "symbol": symbol,
        "price": 50000.0,
        "market_cap": 1000000000,
        "volume_24h": 50000000,
        "change_24h": 2.5
    }
```

## 📞 **Getting Help**

1. **Check API Documentation**: [Token Metrics API Reference](https://developers.tokenmetrics.com/reference/tokens)
2. **Contact Support**: Token Metrics support team
3. **Check Status Page**: API service status
4. **Community Forums**: Developer community discussions

## ✅ **Success Criteria**

The implementation is working correctly when:
- ✅ API calls return 200 status codes
- ✅ Response data is properly formatted
- ✅ No 400 Bad Request errors
- ✅ Rate limiting is respected
- ✅ All endpoints are accessible

## 🎯 **Next Steps**

1. **Run the test suite**: `python test_token_metrics_api.py`
2. **Verify API key**: Check your Token Metrics dashboard
3. **Test CLI interface**: `python token_metrics/token_metrics_main.py`
4. **Monitor usage**: Check API call limits and usage

The updated implementation should now work correctly with the Token Metrics API without the 400 Bad Request errors! 