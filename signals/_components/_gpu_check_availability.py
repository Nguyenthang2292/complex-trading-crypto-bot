import logging
import os
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="_gpu_check_availability", log_level=logging.DEBUG)

def check_gpu_availability() -> bool:
    """
    Check GPU availability and perform comprehensive diagnostic testing.
    
    Verifies CUDA availability, enumerates GPU devices, tests tensor operations,
    and provides specific error diagnostics for common CUDA/cuDNN issues.
    
    Returns:
        True if GPU is available and functioning correctly, False if CPU fallback required
    """
    logger.info("="*60)
    logger.gpu("GPU AVAILABILITY CHECK (PyTorch)")
    logger.info("="*60)
    
    gpu_available: bool = torch.cuda.is_available()
    logger.gpu("PyTorch CUDA available: {0}".format(gpu_available))
    
    if not gpu_available:
        logger.warning("No GPU detected - will use CPU")
        logger.info("="*60)
        return False
    
    try:
        gpu_count: int = torch.cuda.device_count()
        logger.gpu("Number of GPUs: {0}".format(gpu_count))
        
        for i in range(gpu_count):
            try:
                gpu_name: str = torch.cuda.get_device_name(i)
                gpu_memory: float = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.gpu("GPU {0}: {1} ({2:.1f} GB)".format(i, gpu_name, gpu_memory))
            except Exception as gpu_info_error:
                logger.warning("Could not get info for GPU {0}: {1}".format(i, gpu_info_error))
            
        try:
            device: torch.device = torch.device('cuda:0')
            test_tensor: torch.Tensor = torch.randn(10, 10, device=device)
            result: torch.Tensor = torch.sum(test_tensor)
            logger.success("GPU test successful - tensor operations working")
            logger.info("="*60)
            return True
        except Exception as gpu_test_error:
            error_msg: str = str(gpu_test_error).lower()
            
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
            logger.info("="*60)
            return False
            
    except Exception as e:
        logger.error("GPU initialization failed: {0}".format(e))
        logger.warning("This might be due to missing or incompatible CUDA/cuDNN libraries")
        logger.config("Consider installing CPU-only PyTorch: conda install pytorch torchvision torchaudio cpuonly -c pytorch")
        logger.info("="*60)
        return False

def configure_gpu_memory() -> bool:
    """
    Configure PyTorch GPU memory settings for optimal performance.
    
    Sets memory allocation limits, enables cuDNN optimizations, and configures
    PyTorch memory management to prevent allocation issues during training.
    
    Returns:
        True if GPU memory configuration successful, False if configuration failed
    """
    if not torch.cuda.is_available():
        return False
        
    try:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.75)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        
        logger.memory("GPU memory configuration optimized for performance")
        return True
    except Exception as e:
        error_msg: str = str(e).lower()
        if "cudnn" in error_msg:
            logger.warning("cuDNN configuration failed: {0}".format(e))
            logger.config("cuDNN may not be properly installed - continuing with basic GPU settings")
        else:
            logger.warning("Could not configure GPU memory: {0}".format(e))
            logger.config("Continuing with default GPU settings")
        return False

