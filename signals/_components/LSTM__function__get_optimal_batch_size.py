import logging
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from utilities._logger import setup_logging
# Initialize logger for LSTM Attention module
logger = setup_logging(module_name="LSTM_get_optimal_batch_size", log_level=logging.DEBUG)

# Add error handling for missing config imports
try:
    from livetrade.config import (CPU_BATCH_SIZE, GPU_BATCH_SIZE)
except ImportError as e:
    logger.warning("Could not import batch size configs: {0}. Using defaults.".format(e))
    CPU_BATCH_SIZE = 32
    GPU_BATCH_SIZE = 64

def get_optimal_batch_size(device, input_size, sequence_length, model_type='lstm'):
    """
    Dynamically determine optimal batch size based on GPU memory and model complexity.
    
    Args:
        device: torch device
        input_size: Number of input features
        sequence_length: LSTM sequence length
        model_type: Type of model ('lstm', 'lstm_attention', 'cnn_lstm')
        
    Returns:
        int: Optimal batch size
    """
    if device.type == 'cpu':
        # CPU optimization based on available RAM and cores
        if model_type == 'cnn_lstm':
            return max(8, CPU_BATCH_SIZE // 4)  # CNN requires more memory
        elif model_type == 'lstm_attention':
            return max(16, CPU_BATCH_SIZE // 2)  # Attention models need more memory
        else:
            return CPU_BATCH_SIZE
    
    try:
        # Get GPU memory info
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Estimate memory per sample (more accurate calculation)
        base_memory_mb = (input_size * sequence_length * 4) / 1024**2  # 4 bytes per float32
        
        # Model complexity multipliers
        complexity_multiplier = {
            'lstm': 1.0,
            'lstm_attention': 2.5,  # Attention mechanism requires more memory
            'cnn_lstm': 3.0,        # CNN + LSTM requires most memory
        }
        
        memory_per_sample_mb = base_memory_mb * complexity_multiplier.get(model_type, 1.0)
        
        # Calculate optimal batch size based on GPU memory
        if gpu_memory_gb >= 12:  # High-end GPU (RTX 4080/4090, etc.)
            if model_type == 'cnn_lstm':
                optimal_batch = min(256, max(32, int(gpu_memory_gb * 8)))
            elif model_type == 'lstm_attention':
                optimal_batch = min(512, max(64, int(gpu_memory_gb * 16)))
            else:
                optimal_batch = min(1024, max(128, int(gpu_memory_gb * 32)))
        elif gpu_memory_gb >= 8:  # Mid-high range GPU (RTX 4070, RTX 3080, etc.)
            if model_type == 'cnn_lstm':
                optimal_batch = min(128, max(16, int(gpu_memory_gb * 6)))
            elif model_type == 'lstm_attention':
                optimal_batch = min(256, max(32, int(gpu_memory_gb * 12)))
            else:
                optimal_batch = min(512, max(64, int(gpu_memory_gb * 24)))
        elif gpu_memory_gb >= 6:  # Mid-range GPU (RTX 4060, RTX 3070, etc.)
            if model_type == 'cnn_lstm':
                optimal_batch = min(64, max(8, int(gpu_memory_gb * 4)))
            elif model_type == 'lstm_attention':
                optimal_batch = min(128, max(16, int(gpu_memory_gb * 8)))
            else:
                optimal_batch = min(256, max(32, int(gpu_memory_gb * 16)))
        else:  # Lower-end GPU (RTX 3060, GTX series, etc.)
            if model_type == 'cnn_lstm':
                optimal_batch = min(32, max(4, GPU_BATCH_SIZE // 4))
            elif model_type == 'lstm_attention':
                optimal_batch = min(64, max(8, GPU_BATCH_SIZE // 2))
            else:
                optimal_batch = GPU_BATCH_SIZE
        
        # Safety check: ensure batch size doesn't exceed memory limits
        estimated_memory_gb = (optimal_batch * memory_per_sample_mb) / 1024
        if estimated_memory_gb > gpu_memory_gb * 0.8:  # Use max 80% of GPU memory
            optimal_batch = max(4, int((gpu_memory_gb * 0.8 * 1024) / memory_per_sample_mb))
            
        logger.gpu("GPU Memory: {0:.1f}GB, Model: {1}, Optimal batch size: {2}".format(
            gpu_memory_gb, model_type, optimal_batch))
        logger.debug("Estimated memory usage: {0:.2f}GB ({1:.1f}%)".format(
            estimated_memory_gb, (estimated_memory_gb / gpu_memory_gb) * 100))
        
        return optimal_batch
        
    except Exception as e:
        logger.warning("Could not determine optimal batch size: {0}".format(e))
        # Fallback batch sizes based on model type
        fallback_sizes = {
            'cnn_lstm': max(8, GPU_BATCH_SIZE // 4),
            'lstm_attention': max(16, GPU_BATCH_SIZE // 2),
            'lstm': GPU_BATCH_SIZE
        }
        return fallback_sizes.get(model_type, GPU_BATCH_SIZE)