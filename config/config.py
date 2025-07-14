"""
Configuration module for the crypto trading bot.

This module contains all configuration constants and settings for the trading system,
including system resources, model parameters, trading strategies, and market data
configuration. All constants are organized by functional areas for better maintainability.

Key sections:
- System & Environment: File paths and directory management
- Resource & GPU: Memory and processing limits
- Data Structure & Market: Column names and market parameters  
- Model Configuration: ML model settings and architectures
- Trading Strategy: Signal thresholds and trading parameters
- Specialized Models: HMM and LSTM specific configurations
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass

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
# Memory and processing limits
MAX_TRAINING_ROWS = 1000000  # Maximum rows for training data
LARGE_DATASET_THRESHOLD_FOR_SMOTE = 50000  # Threshold for SMOTE sampling
MIN_MEMORY_GB = 1.0  # Minimum required memory in GB
MAX_CPU_MEMORY_FRACTION = 0.5  # Use 50% of available CPU memory
DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 2  # Wait time between data processing

# GPU configuration
MAX_GPU_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory
CLEAR_GPU_CACHE_EVERY_N_EPOCHS = 5  # Clear GPU cache every N epochs
GPU_BATCH_SIZE = 64  # Batch size for GPU training
CPU_BATCH_SIZE = 32  # Batch size for CPU training

# =============================================================================
# DATA STRUCTURE & MARKET CONFIGURATION
# =============================================================================
# Column Names for OHLCV data
COL_OPEN = 'open'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_CLOSE = 'close'
COL_VOLUME = 'volume'
COL_BB_UPPER = 'bb_upper'
COL_BB_LOWER = 'bb_lower'

# Technical Indicators Parameters
FUTURE_RETURN_SHIFT = -5  # Shift for future return calculation
RSI_PERIOD = 14  # RSI calculation period
SMA_PERIOD = 20  # Simple Moving Average period
BB_WINDOW = 20  # Bollinger Bands window
BB_STD_MULTIPLIER = 2  # Bollinger Bands standard deviation multiplier
MACD_FAST_PERIOD = 12  # MACD fast EMA period
MACD_SLOW_PERIOD = 26  # MACD slow EMA period
MACD_SIGNAL_PERIOD = 9  # MACD signal line period
DEFAULT_WINDOW_SIZE = 20  # Default window size for calculations

# Market Data & Testing Configuration
DEFAULT_TIMEFRAMES: List[str] = ['15m','30m', '1h', '4h', '1d']
DEFAULT_CRYPTO_SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", 
    "XRPUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT"
]
DEFAULT_TEST_SYMBOL = "BTCUSDT"
DEFAULT_TEST_TIMEFRAME = '1h'
DEFAULT_TOP_SYMBOLS = 10

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# General Model Settings
MODEL_RANDOM_STATE = 42  # Random seed for reproducibility
MODEL_TEST_SIZE = 0.2  # 20% of data for testing
MIN_DATA_POINTS = 100  # Minimum data points required
MIN_TRAINING_SAMPLES = 50  # Minimum training samples
MAX_NAN_PERCENTAGE = 0.1  # Maximum allowed NaN percentage (10%)
TRAIN_TEST_SPLIT = 0.8  # 80% for training
VALIDATION_SPLIT = 0.2  # 20% for validation

# Model Features
MODEL_FEATURES: List[str] = [
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME,
    'rsi', 'macd', 'macd_signal',
    COL_BB_LOWER, COL_BB_UPPER,
    'ma_20'
]

# Model Architecture Configuration
@dataclass
class ModelArchitecture:
    """Configuration for model architecture parameters."""
    feature_size: int
    num_layers: int
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    seq_length: int
    prediction_length: int

# GPU-optimized model configuration
GPU_MODEL_CONFIG = ModelArchitecture(
    feature_size=11,
    num_layers=4,
    d_model=128,
    nhead=8,
    dim_feedforward=512,
    dropout=0.1,
    seq_length=30,
    prediction_length=1
)

# CPU-optimized model configuration
CPU_MODEL_CONFIG = ModelArchitecture(
    feature_size=11,
    num_layers=2,
    d_model=64,
    nhead=8,
    dim_feedforward=256,
    dropout=0.1,
    seq_length=30,
    prediction_length=1
)

# Training Parameters
DEFAULT_EPOCHS = 20  # Default training epochs
FAST_EPOCHS = 10  # Fast training epochs for quick testing
FULL_EPOCHS = 50  # Full training epochs for production

# =============================================================================
# TRADING STRATEGY CONFIGURATION
# =============================================================================
# Trading Signals
SIGNAL_LONG = 'LONG'
SIGNAL_SHORT = 'SHORT'
SIGNAL_NEUTRAL = 'NEUTRAL'

# Trading Thresholds
BUY_THRESHOLD = 0.01  # 1% price increase threshold for buy signals
SELL_THRESHOLD = -0.01  # 1% price decrease threshold for sell signals
CONFIDENCE_THRESHOLD = 0.65  # 65% confidence threshold for trading decisions
CONFIDENCE_THRESHOLDS: List[float] = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]

# =============================================================================
# SPECIALIZED MODEL CONFIGURATION
# =============================================================================
# HMM (Hidden Markov Model) Configuration
SIGNAL_LONG_HMM = 1
SIGNAL_HOLD_HMM = 0
SIGNAL_SHORT_HMM = -1
HMM_PROBABILITY_THRESHOLD = 0.35  # 35% probability threshold for HMM signals

# LSTM Specific Configuration
WINDOW_SIZE_LSTM = 60  # LSTM input window size
TARGET_THRESHOLD_LSTM = 0.005  # 0.5% threshold for LSTM targets
NEUTRAL_ZONE_LSTM = 0.001  # 0.1% neutral zone for LSTM predictions

# ARIMAX-GARCH Model Configuration
# ARIMAX Model Parameters
ARIMAX_MAX_P = 5  # Maximum AR order
ARIMAX_MAX_D = 2  # Maximum differencing order  
ARIMAX_MAX_Q = 5  # Maximum MA order
ARIMAX_DEFAULT_ORDER = (1, 1, 1)  # Default (p, d, q) order

# GARCH Model Parameters
GARCH_P = 1  # GARCH lag order
GARCH_Q = 1  # ARCH lag order
GARCH_MEAN_MODEL = 'Zero'  # Mean model specification

# Signal Generation Parameters
SIGNAL_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for signal generation
STRONG_SIGNAL_THRESHOLD = 0.02  # 2% threshold for strong signals
WEAK_SIGNAL_THRESHOLD = 0.005  # 0.5% threshold for weak signals
RISK_FREE_RATE = 0.02 / 252  # Daily risk-free rate (2% annual)

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config() -> bool:
    """
    Validate critical configuration values.
    
    Returns:
        bool: True if all validations pass
        
    Raises:
        ValueError: If any configuration value is invalid
    """
    # Validate memory fractions
    if not (0 < MAX_CPU_MEMORY_FRACTION <= 1):
        raise ValueError("MAX_CPU_MEMORY_FRACTION must be between 0 and 1")
    if not (0 < MAX_GPU_MEMORY_FRACTION <= 1):
        raise ValueError("MAX_GPU_MEMORY_FRACTION must be between 0 and 1")
    
    # Validate thresholds
    if BUY_THRESHOLD <= 0:
        raise ValueError("BUY_THRESHOLD must be positive")
    if SELL_THRESHOLD >= 0:
        raise ValueError("SELL_THRESHOLD must be negative")
    if not (0 < CONFIDENCE_THRESHOLD < 1):
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    # Validate model parameters
    if MIN_DATA_POINTS <= 0:
        raise ValueError("MIN_DATA_POINTS must be positive")
    if not (0 < MODEL_TEST_SIZE < 1):
        raise ValueError("MODEL_TEST_SIZE must be between 0 and 1")
    
    return True

# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    raise RuntimeError(f"Configuration validation failed: {e}")
