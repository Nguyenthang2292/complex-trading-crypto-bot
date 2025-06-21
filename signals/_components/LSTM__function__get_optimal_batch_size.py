import logging
import sys
from typing import Literal
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="LSTM__function__get_optimal_batch_size", log_level=logging.DEBUG)

try:
    from components.config import (CPU_BATCH_SIZE, GPU_BATCH_SIZE)
except ImportError as e:
    logger.warning("Could not import batch size configs: {0}. Using defaults.".format(e))
    CPU_BATCH_SIZE = 32
    GPU_BATCH_SIZE = 64

def get_optimal_batch_size(
    device: torch.device, 
    input_size: int, 
    sequence_length: int, 
    model_type: Literal['lstm', 'lstm_attention', 'cnn_lstm'] = 'lstm'
) -> int:
    """
    Dynamically determine optimal batch size based on device capabilities and model complexity.
    
    Args:
        device: PyTorch device (CPU or CUDA)
        input_size: Number of input features per timestep
        sequence_length: Length of input sequences for LSTM
        model_type: Architecture type affecting memory requirements
        
    Returns:
        Optimal batch size as integer value
    """
    if device.type == 'cpu':
        complexity_divisors = {'cnn_lstm': 4, 'lstm_attention': 2, 'lstm': 1}
        return max(8 if model_type == 'cnn_lstm' else 16 if model_type == 'lstm_attention' else CPU_BATCH_SIZE, 
                  CPU_BATCH_SIZE // complexity_divisors[model_type])
    
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        base_memory_mb = (input_size * sequence_length * 4) / (1024**2)
        
        complexity_multipliers = {'lstm': 1.0, 'lstm_attention': 2.5, 'cnn_lstm': 3.0}
        memory_per_sample_mb = base_memory_mb * complexity_multipliers[model_type]
        
        if gpu_memory_gb >= 12:
            memory_factors = {'cnn_lstm': 8, 'lstm_attention': 16, 'lstm': 32}
            max_batches = {'cnn_lstm': 256, 'lstm_attention': 512, 'lstm': 1024}
            min_batches = {'cnn_lstm': 32, 'lstm_attention': 64, 'lstm': 128}
        elif gpu_memory_gb >= 8:
            memory_factors = {'cnn_lstm': 6, 'lstm_attention': 12, 'lstm': 24}
            max_batches = {'cnn_lstm': 128, 'lstm_attention': 256, 'lstm': 512}
            min_batches = {'cnn_lstm': 16, 'lstm_attention': 32, 'lstm': 64}
        elif gpu_memory_gb >= 6:
            memory_factors = {'cnn_lstm': 4, 'lstm_attention': 8, 'lstm': 16}
            max_batches = {'cnn_lstm': 64, 'lstm_attention': 128, 'lstm': 256}
            min_batches = {'cnn_lstm': 8, 'lstm_attention': 16, 'lstm': 32}
        else:
            fallback_divisors = {'cnn_lstm': 4, 'lstm_attention': 2, 'lstm': 1}
            fallback_mins = {'cnn_lstm': 4, 'lstm_attention': 8, 'lstm': GPU_BATCH_SIZE}
            return max(fallback_mins[model_type], GPU_BATCH_SIZE // fallback_divisors[model_type])
        
        optimal_batch = min(max_batches[model_type], 
                           max(min_batches[model_type], int(gpu_memory_gb * memory_factors[model_type])))
        
        estimated_memory_gb = (optimal_batch * memory_per_sample_mb) / 1024
        if estimated_memory_gb > gpu_memory_gb * 0.8:
            optimal_batch = max(4, int((gpu_memory_gb * 0.8 * 1024) / memory_per_sample_mb))
            
        logger.gpu("GPU Memory: {0:.1f}GB, Model: {1}, Optimal batch size: {2}".format(
            gpu_memory_gb, model_type, optimal_batch))
        logger.debug("Estimated memory usage: {0:.2f}GB ({1:.1f}%)".format(
            estimated_memory_gb, (estimated_memory_gb / gpu_memory_gb) * 100))
        
        return optimal_batch
        
    except Exception as e:
        logger.warning("Could not determine optimal batch size: {0}".format(e))
        fallback_sizes = {'cnn_lstm': max(8, GPU_BATCH_SIZE // 4),
                         'lstm_attention': max(16, GPU_BATCH_SIZE // 2),
                         'lstm': GPU_BATCH_SIZE}
        return fallback_sizes[model_type]