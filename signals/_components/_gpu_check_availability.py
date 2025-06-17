import logging
import os
import sys
from typing import Union
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.dirname(current_dir)
signals_dir = os.path.dirname(components_dir)
if signals_dir not in sys.path:
    sys.path.insert(0, signals_dir)

from utilities._logger import setup_logging
# Initialize logger for LSTM Attention module
logger = setup_logging(module_name="_check_gpu_availability", log_level=logging.DEBUG)

def check_gpu_availability() -> bool:
    """
    Check if GPU is available for PyTorch and provide detailed diagnostic information.
    
    Returns:
        bool: True if GPU is available and functioning, False otherwise
    """
    logger.info("="*60)
    logger.gpu("GPU AVAILABILITY CHECK (PyTorch)")
    logger.info("="*60)
    
    gpu_available = torch.cuda.is_available()
    logger.gpu("PyTorch CUDA available: {0}".format(gpu_available))
    
    if gpu_available:
        try:
            gpu_count = torch.cuda.device_count()
            logger.gpu("Number of GPUs: {0}".format(gpu_count))
            
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.gpu("GPU {0}: {1} ({2:.1f} GB)".format(i, gpu_name, gpu_memory))
                except Exception as gpu_info_error:
                    logger.warning("Could not get info for GPU {0}: {1}".format(i, gpu_info_error))
                
            try:
                device = torch.device('cuda:0')
                test_tensor = torch.randn(10, 10, device=device)
                result = torch.sum(test_tensor)
                logger.success("GPU test successful - tensor operations working")
                return True
            except Exception as gpu_test_error:
                error_msg = str(gpu_test_error).lower()
                if "cudnn" in error_msg or "cudnn_graph64_9.dll" in error_msg:
                    logger.error("cuDNN library error detected: {0}".format(gpu_test_error))
                    logger.warning("cuDNN not properly installed or incompatible version")
                    logger.config("Solutions:")
                    logger.config("1. Install cuDNN: conda install cudnn -c conda-forge")
                    logger.config("2. Reinstall PyTorch with CUDA: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
                    logger.config("3. Use CPU-only version: conda install pytorch torchvision torchaudio cpuonly -c pytorch")
                elif "invalid handle" in error_msg:
                    logger.error("CUDA driver/runtime mismatch: {0}".format(gpu_test_error))
                    logger.warning("CUDA driver and runtime versions may be incompatible")
                    logger.config("Check CUDA installation: nvidia-smi")
                else:
                    logger.error("GPU tensor test failed: {0}".format(gpu_test_error))
                
                logger.warning("Falling back to CPU mode due to GPU issues")
                return False
                
        except Exception as e:
            logger.error("GPU initialization failed: {0}".format(e))
            logger.warning("This might be due to missing or incompatible CUDA/cuDNN libraries")
            logger.config("Consider installing CPU-only PyTorch: conda install pytorch torchvision torchaudio cpuonly -c pytorch")
            return False
    else:
        logger.warning("No GPU detected - will use CPU")
        
    logger.info("="*60)
    return gpu_available

def configure_gpu_memory() -> bool:
    """
    Configure PyTorch GPU memory settings to avoid memory allocation issues.
    
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.75)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
            
            logger.memory("GPU memory configuration optimized for performance")
            return True
    except Exception as e:
        error_msg = str(e).lower()
        if "cudnn" in error_msg:
            logger.warning("cuDNN configuration failed: {0}".format(e))
            logger.config("cuDNN may not be properly installed - continuing with basic GPU settings")
        else:
            logger.warning("Could not configure GPU memory: {0}".format(e))
            logger.config("Continuing with default GPU settings")
    return False

