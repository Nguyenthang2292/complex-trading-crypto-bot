
## Complex Trading Crypto Bot for Binance

*Currently under development and updates*

### Setup Instructions

1. **Environment Setup**
    - Create a virtual environment (`.venv`)
    - Install all required libraries

2. **API Configuration**
    - Obtain your Binance API credentials
    - Configure the API keys in `components/_components/_tick_processor.py`:
      ```python
      API_KEY = 'xxxxxxxxxxxxxxxxxxxx_1'
      API_SECRET = 'xxxxxxxxxxxxxxxxxxxx_2'
      ```

3. **Run the Bot**
    - Run `trading_signal_analyzer.py` if you want to check signals for a specific symbol (LONG or SHORT)
    - Run `trading_signal_analyzer_all_symbols.py` if you want to analyze the entire market to see how many LONG and SHORT signals are available
