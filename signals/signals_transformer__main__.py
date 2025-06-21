import logging
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
from torch.utils.data import DataLoader

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.tick_processor import tick_processor
from utilities._gpu_resource_manager import get_gpu_resource_manager, check_gpu_availability
from utilities._logger import setup_logging

# Setup logging
logger = setup_logging(module_name="signals_transformer__main__", log_level=logging.DEBUG)

from components.config import (
    COL_CLOSE,
    CPU_BATCH_SIZE,
    CPU_MODEL_CONFIG,
    DATA_PROCESSING_WAIT_TIME_IN_SECONDS,
    DEFAULT_TEST_SYMBOL,
    DEFAULT_TEST_TIMEFRAME,
    GPU_BATCH_SIZE,
    GPU_MODEL_CONFIG,
)

from signals.signals_transformer import (
    CryptoDataset,
    TimeSeriesTransformer,
    evaluate_model,
    get_latest_transformer_signal,
    select_and_scale_features,
    train_transformer_model,
)

# Import the calculate_features function
from components._generate_indicator_features import generate_indicator_features

# Check GPU availability at startup
gpu_available = check_gpu_availability()

def main():
    """
    Executes the transformer model training pipeline from data loading through signal generation.
    
    This function:
    1. Loads historical market data for the default test pair
    2. Preprocesses data with technical indicators 
    3. Trains a transformer model on the processed data
    4. Evaluates model performance
    5. Generates a trading signal based on the latest data
    """
    logger.model("Starting transformer model training pipeline")
    
    # Initialize resources and load data
    gpu_manager = get_gpu_resource_manager()
    
    with gpu_manager.gpu_scope():
        # Log GPU memory info
        gpu_info = gpu_manager.get_memory_info()
        logger.memory(f"GPU Memory Info - Total: {gpu_info['total'] / (1024**3):.2f}GB, "
                     f"Allocated: {gpu_info['allocated'] / (1024**3):.2f}GB, "
                     f"Cached: {gpu_info['cached'] / (1024**3):.2f}GB")
        
        # Initialize tick processor and fetch historical data
        processor = tick_processor(trade_open_callback=None, trade_close_callback=None)
        symbol = DEFAULT_TEST_SYMBOL           
        timeframe = DEFAULT_TEST_TIMEFRAME  
        logger.data(f"Requesting historic data for {symbol} / {timeframe}")
        processor.get_historic_data_by_symbol(symbol, timeframe)
    
    # Wait for data processing to complete
    time.sleep(DATA_PROCESSING_WAIT_TIME_IN_SECONDS)  
    
    # Retrieve and validate data
    df = processor.df_cache.get((symbol, timeframe), pd.DataFrame())
    if df is None or df.empty:
        logger.error(f"No data loaded for {symbol} {timeframe}")
        raise ValueError(f"No data loaded for {symbol} {timeframe}")
    
    logger.data(f"Loaded {len(df)} rows of data for {symbol} {timeframe}")
    
    # Data preprocessing
    df_ind = generate_indicator_features(df)
    data_scaled, scaler, feature_cols = select_and_scale_features(df_ind)
    target_idx = feature_cols.index(COL_CLOSE)
    logger.config(f"Target index (close price) at position {target_idx} in features")
    
    # Create dataset with appropriate sequence length
    seq_len, pred_len = GPU_MODEL_CONFIG['seq_length'], GPU_MODEL_CONFIG['prediction_length']
    dataset = CryptoDataset(data_scaled, seq_len, pred_len, len(feature_cols), target_idx)
    
    # Split dataset into training, validation, and test sets
    n_samples = len(dataset)
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)
    n_test = n_samples - n_train - n_val
    logger.data(f"Dataset split: Train={n_train}, Validation={n_val}, Test={n_test}")
    
    # Create data subsets
    train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_ds = torch.utils.data.Subset(dataset, range(n_train + n_val, n_samples))
    
    # Create data loaders with appropriate batch size
    batch_size = GPU_BATCH_SIZE if gpu_available else CPU_BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize model with appropriate configuration - Fix feature size mismatch
    model_config = GPU_MODEL_CONFIG.copy() if gpu_available else CPU_MODEL_CONFIG.copy()
    model_config['feature_size'] = len(feature_cols)
    model = TimeSeriesTransformer(**model_config)

    # Configure device for training
    device = 'cuda' if gpu_available and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        logger.success("Using GPU acceleration for training")
    else:
        logger.info("Using CPU for training")
    
    logger.config(f"Using device: {device}")
    logger.config(f"Model feature size: {len(feature_cols)}")
    
    # Train and evaluate model
    trained_model = train_transformer_model(model, train_loader, val_loader, device=device)
    mse, mae = evaluate_model(trained_model, test_loader, scaler, feature_cols, target_idx, device=device)
    
    # Log performance metrics
    rmse = np.sqrt(mse)
    logger.performance("="*80)
    logger.performance("MODEL METRICS:")
    logger.performance(f"  • MSE (Mean Squared Error): {mse:.6f}")
    logger.performance(f"  • MAE (Mean Absolute Error): {mae:.6f}")
    logger.performance(f"  • RMSE (Root Mean Squared Error): {rmse:.6f}")
    logger.performance("="*80)
    
    # Generate trading signal
    signal = get_latest_transformer_signal(df, trained_model, scaler, feature_cols, target_idx, device)
    logger.signal(f"Generated signal: {signal}")
    
    # Clean up resources
    processor.stop()
    logger.success("Pipeline completed successfully")

if __name__ == "__main__":
    main()