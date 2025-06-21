from pathlib import Path

# =============================================================================
# SYSTEM & ENVIRONMENT CONFIGURATION
# =============================================================================

# File and Directory Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_BACKUP_DIR = MODELS_DIR / "backup"
MODELS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Model Filenames
RANDOM_FOREST_MODEL_FILENAME = "random_forest_model.joblib"
TRANSFORMER_MODEL_FILENAME = "transformer_model.pth"
LSTM_MODEL_FILENAME = "LSTM_model.pth"

# =============================================================================
# RESOURCE & GPU CONFIGURATION
# =============================================================================
MAX_TRAINING_ROWS = 1000000
LARGE_DATASET_THRESHOLD_FOR_SMOTE = 50000
MIN_MEMORY_GB = 1.0
MAX_CPU_USAGE_FRACTION = 0.5                    # Use 50% of available CPUs
DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 2

GPU_MEMORY_FRACTION = 0.8                       # Use 80% of GPU memory
CLEAR_GPU_CACHE_EVERY_N_EPOCHS = 5
GPU_BATCH_SIZE = 64
CPU_BATCH_SIZE = 32

# =============================================================================
# DATA STRUCTURE & MARKET CONFIGURATION
# =============================================================================
# Column Names
COL_OPEN = 'open'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_CLOSE = 'close'
COL_VOLUME = 'volume'
COL_BB_UPPER = 'bb_upper'
COL_BB_LOWER = 'bb_lower'

# Technical Indicators and Target Parameters
FUTURE_RETURN_SHIFT = -5
RSI_PERIOD = 14
SMA_PERIOD = 20
BB_WINDOW = 20
BB_STD_MULTIPLIER = 2
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
DEFAULT_WINDOW_SIZE = 20

# Market Data & Testing
DEFAULT_TIMEFRAMES = ['1h', '4h', '1d']
DEFAULT_CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT"]
DEFAULT_TEST_SYMBOL = "BTCUSDT"
DEFAULT_TEST_TIMEFRAME = '1h'
DEFAULT_TOP_SYMBOLS = 10

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# General Settings
MODEL_RANDOM_STATE = 42
MODEL_TEST_SIZE = 0.2
MIN_DATA_POINTS = 100
MIN_TRAINING_SAMPLES = 50
MAX_NAN_PERCENTAGE = 0.1
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2

# Model Features
MODEL_FEATURES = [
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME,
    'rsi', 'macd', 'macd_signal',
    COL_BB_LOWER, COL_BB_UPPER,
    'ma_20'
]

# Model Architectures
GPU_MODEL_CONFIG = {
    'feature_size': 11,  # Can be set dynamically using len(MODEL_FEATURES) if desired
    'num_layers': 4,
    'd_model': 128,
    'nhead': 8,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'seq_length': 30,
    'prediction_length': 1
}

CPU_MODEL_CONFIG = {
    'feature_size': 11,
    'num_layers': 2,
    'd_model': 64,
    'nhead': 8,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'seq_length': 30,
    'prediction_length': 1
}

# Training Parameters (for Transformer / LSTM and similar models)
DEFAULT_EPOCHS = 20
FAST_EPOCHS = 10
FULL_EPOCHS = 50

# =============================================================================
# TRADING STRATEGY CONFIGURATION
# =============================================================================
SIGNAL_LONG = 'LONG'
SIGNAL_SHORT = 'SHORT'
SIGNAL_NEUTRAL = 'NEUTRAL'

BUY_THRESHOLD = 0.01
SELL_THRESHOLD = -0.01
CONFIDENCE_THRESHOLD = 0.65
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]

# =============================================================================
# SPECIALIZED MODEL CONFIGURATION
# =============================================================================
# HMM Configuration
SIGNAL_LONG_HMM = 1
SIGNAL_HOLD_HMM = 0
SIGNAL_SHORT_HMM = -1
HMM_PROBABILITY_THRESHOLD = 0.35

# LSTM Specific Configuration
WINDOW_SIZE_LSTM = 60
TARGET_THRESHOLD_LSTM = 0.005  
NEUTRAL_ZONE_LSTM = 0.001
