
## Complex Trading Crypto Bot for Binance

*Currently under development and updates*

### Overview

This advanced crypto trading bot combines multiple AI/ML models to generate trading signals for Binance USDT pairs. The system uses a sophisticated multi-layered approach combining performance analysis, Random Forest classification, and Hidden Markov Models (HMM) for signal generation.

### Key Features

- **Multi-Model Signal Generation**: Combines Random Forest and HMM models for robust signal prediction
- **Performance-Based Filtering**: Uses historical performance analysis to identify top-performing and worst-performing symbols
- **Multi-Timeframe Analysis**: Supports multiple timeframes for comprehensive market analysis
- **Volume and Liquidity Filtering**: Ensures only liquid pairs with sufficient trading volume
- **Stable Coin Exclusion**: Automatically filters out stable coins to focus on volatile assets
- **Confidence Scoring**: Provides confidence levels for each trading signal
- **Real-time Market Analysis**: Analyzes entire Binance USDT market for opportunities

### Signal Generation Process

1. **Data Loading**: Loads historical data for all USDT pairs across multiple timeframes
2. **Performance Analysis**: Identifies best-performing (LONG candidates) and worst-performing (SHORT candidates) symbols
3. **Random Forest Training**: Trains a global Random Forest model on combined market data
4. **Signal Filtering**: Applies RF model to filtered symbols to generate initial signals
5. **HMM Validation**: Uses Hidden Markov Models to validate and refine signals
6. **Final Ranking**: Ranks signals by combined confidence score

### Setup Instructions

1. **Environment Setup**
    - Create a virtual environment (`.venv`)
    - Install all required libraries:

      ```bash
      pip install pandas numpy scikit-learn hmmlearn tqdm
      ```

2. **API Configuration**
    - Obtain your Binance API credentials
    - Configure the API keys in `components/tick_processor.py`:

      ```python
      API_KEY = 'xxxxxxxxxxxxxxxxxxxx_1'
      API_SECRET = 'xxxxxxxxxxxxxxxxxxxx_2'
      ```

3. **Configuration**
    - Adjust parameters in `config/config.py`:
      - `DEFAULT_TIMEFRAMES`: Timeframes to analyze
      - `SIGNAL_LONG`, `SIGNAL_SHORT`: Signal constants
      - Performance thresholds and filtering criteria

### Usage

#### Basic Signal Analysis

```bash
# Analyze specific symbol signals
python trading_signal_analyzer.py
```

#### Full Market Analysis

```bash
# Analyze entire market for opportunities
python trading_signal_analyzer_all_symbols.py
```

#### Simplified Advanced Analysis

```bash
# Run the simplified version with enhanced filtering
python trading_signal_analyzer_all_symbols_simplified.py
```

### Output Format

The bot provides detailed signal analysis including:

- **Symbol**: Trading pair (e.g., BTCUSDT)
- **Direction**: LONG or SHORT signal
- **Timeframe**: Analysis timeframe (1m, 5m, 15m, etc.)
- **RF Confidence**: Random Forest model confidence
- **Performance Score**: Historical performance score
- **Total Confidence**: Combined confidence score

### Example Output
```
Top 10 tín hiệu LONG cuối cùng:
   1. 🟢 BTCUSDT      | Direction: LONG  | TF: 1h | RF: 85.2% | Perf: 92.1% | Tổng hợp: 88.7%
   2. 🟢 ETHUSDT      | Direction: LONG  | TF: 4h | RF: 78.9% | Perf: 89.3% | Tổng hợp: 84.1%
   
Top 10 tín hiệu SHORT cuối cùng:
   1. 🔴 DOGEUSDT     | Direction: SHORT | TF: 1h | RF: 82.4% | Perf: 91.7% | Tổng hợp: 87.1%
   2. 🔴 ADAUSDT      | Direction: SHORT | TF: 4h | RF: 76.2% | Perf: 88.9% | Tổng hợp: 82.6%
```

### Advanced Features

- **Automatic Model Management**: Automatically removes old models before retraining
- **Data Validation**: Ensures data quality and minimum length requirements
- **Error Handling**: Comprehensive error handling and logging
- **Progress Tracking**: Real-time progress bars for long-running operations
- **Memory Optimization**: Efficient data handling for large datasets

### Model Components

1. **Performance Analyzer**: Identifies top/bottom performing symbols based on historical data
2. **Random Forest Classifier**: Machine learning model for signal classification
3. **Hidden Markov Model**: Probabilistic model for state prediction and signal validation
4. **Data Processor**: Handles data loading, cleaning, and preprocessing
5. **Signal Combiner**: Merges signals from multiple models with confidence scoring
