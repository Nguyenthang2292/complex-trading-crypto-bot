import os
import sys
import logging
from typing import List, bool

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities._logger import setup_logging
logger = setup_logging(module_name="_gpu_setup_cuda_environment", log_level=logging.DEBUG)

def setup_cuda_environment() -> None:
    """
    Setup CUDA environment variables for PyTorch compatibility.
    
    Configures PATH environment variable with CUDA/cuDNN binary directories
    and sets required environment variables for GPU acceleration.
    """
    cuda_paths: List[str] = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.8",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\lib\x64",
    ]
    
    current_path: str = os.environ.get('PATH', '')
    paths_added: int = 0
    
    for cuda_path in cuda_paths:
        if os.path.isdir(cuda_path) and cuda_path not in current_path:
            os.environ['PATH'] = cuda_path + os.pathsep + current_path
            logger.info("Added to PATH: {0}".format(cuda_path))
            paths_added += 1
    
    cudnn_path: str = r"C:\Program Files\NVIDIA\CUDNN\v9.10"
    cuda_toolkit_path: str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    
    if os.path.exists(cudnn_path):
        os.environ['CUDNN_PATH'] = cudnn_path
    if os.path.exists(cuda_toolkit_path):
        os.environ['CUDA_PATH'] = cuda_toolkit_path
    
    logger.info("CUDA environment setup complete - {0} paths added".format(paths_added))


def load_cuda_dlls() -> bool:
    """
    Load CUDA and cuDNN DLLs for GPU acceleration support.
    
    Searches known installation directories for essential cuDNN libraries
    and adds them to the DLL search path for PyTorch compatibility.
    
    Returns:
        True if any cuDNN DLLs were successfully loaded, False otherwise
    """
    potential_dirs: List[str] = [
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.8", 
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v9.0\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v8.9\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",  
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin",
        r"C:\tools\cuda\bin",
        r"C:\cuda\bin",
        r"C:\Users\Admin\Desktop\NGUYEN QUANG THANG\Quantitative-trading-project\[2025] ComplexTradingBotCryptoRefactor\.conda\Lib\site-packages\torch\lib",
    ]

    essential_dlls: List[str] = [
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_graph64_9.dll",
    ]

    dll_loaded: bool = False
    successful_dirs: List[str] = []
    
    for directory in potential_dirs:
        if not os.path.isdir(directory):
            continue
            
        found_dlls: List[str] = [
            dll for dll in essential_dlls 
            if os.path.isfile(os.path.join(directory, dll))
        ]
        
        if found_dlls:
            try:
                os.add_dll_directory(directory)
                successful_dirs.append(directory)
                logger.info("Successfully added DLL directory: {0}".format(directory))
                logger.debug("  Found DLLs: {0}".format(found_dlls))
                dll_loaded = True
            except Exception as e:
                logger.error("Failed to add dll directory {0}: {1}".format(directory, e))
    
    if dll_loaded:
        logger.info("Successfully loaded cuDNN from {0} directories".format(len(successful_dirs)))
        
        missing_dlls: List[str] = [
            dll for dll in essential_dlls
            if not any(os.path.isfile(os.path.join(d, dll)) for d in successful_dirs)
        ]
        
        if missing_dlls:
            logger.warning("Missing critical DLLs: {0}".format(missing_dlls))
            logger.warning("This may cause GPU acceleration issues")
    else:
        logger.warning("No cuDNN DLLs found in any known path. GPU acceleration may be unavailable.")
    
    return dll_loaded


def check_cuda_compatibility() -> bool:
    """
    Check CUDA and cuDNN compatibility with current PyTorch installation.
    
    Verifies PyTorch can detect CUDA devices and reports system configuration
    including CUDA version, cuDNN version, and available GPU devices.
    
    Returns:
        True if CUDA is available and compatible, False otherwise
    """
    try:
        import torch
        logger.info("PyTorch version: {0}".format(torch.__version__))
        cuda_available: bool = torch.cuda.is_available()
        logger.info("CUDA available: {0}".format(cuda_available))
        
        if cuda_available:
            logger.info("CUDA version: {0}".format(torch.version.cuda))  # type: ignore
            logger.info("cuDNN version: {0}".format(torch.backends.cudnn.version()))
            device_count: int = torch.cuda.device_count()
            logger.info("GPU device count: {0}".format(device_count))
            
            for i in range(device_count):
                device_name: str = torch.cuda.get_device_name(i)
                logger.info("GPU {0}: {1}".format(i, device_name))
                
        return cuda_available
    except Exception as e:
        logger.error("Error checking CUDA compatibility: {0}".format(e))
        return False