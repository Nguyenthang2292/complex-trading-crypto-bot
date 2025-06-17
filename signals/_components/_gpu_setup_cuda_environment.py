import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.dirname(current_dir)
signals_dir = os.path.dirname(components_dir)
if signals_dir not in sys.path:
    sys.path.insert(0, signals_dir)

from utilities._logger import setup_logging
# Initialize logger for CUDA setup
logger = setup_logging(module_name="_setup_cuda_environment", log_level=logging.DEBUG)

# Enhanced CUDA/cuDNN DLL loading with comprehensive DLL support
def setup_cuda_environment():
    """Setup CUDA environment variables for better compatibility"""
    # CUDA and cuDNN paths for version 12.8
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.8",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9",
        r"C:\Program Files\NVIDIA\CUDNN\v9.10\lib\x64",
    ]
    
    # Add to PATH if not already present
    current_path = os.environ.get('PATH', '')
    paths_added = 0
    for cuda_path in cuda_paths:
        if os.path.isdir(cuda_path) and cuda_path not in current_path:
            os.environ['PATH'] = cuda_path + os.pathsep + current_path
            logger.info("Added to PATH: {0}".format(cuda_path))  
            paths_added += 1
    
    # Set cuDNN environment variables
    if os.path.exists(r"C:\Program Files\NVIDIA\CUDNN\v9.10"):
        os.environ['CUDNN_PATH'] = r"C:\Program Files\NVIDIA\CUDNN\v9.10"
    if os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"):
        os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    
    logger.info("CUDA environment setup complete - {0} paths added".format(paths_added))  

def load_cuda_dlls():
    """Enhanced CUDA/cuDNN DLL loading with comprehensive DLL support"""
    potential_dirs = [
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

    # List of essential cuDNN DLLs for version 9.x
    essential_dlls = [
        "cudnn64_9.dll",           # Core cuDNN library
        "cudnn_ops64_9.dll",       # Operations library
        "cudnn_cnn64_9.dll",       # CNN operations
        "cudnn_adv64_9.dll",       # Advanced operations
        "cudnn_graph64_9.dll",     # Graph operations (the missing one)
    ]

    dll_loaded = False
    successful_dirs = []
    
    for d in potential_dirs:
        if os.path.isdir(d):
            # Check how many essential DLLs exist in this directory
            found_dlls = []
            for dll in essential_dlls:
                dll_path = os.path.join(d, dll)
                if os.path.isfile(dll_path):
                    found_dlls.append(dll)
            
            if found_dlls:  # If any DLL is found, try to add directory
                try:
                    os.add_dll_directory(d)
                    successful_dirs.append(d)
                    logger.info("Successfully added DLL directory: {0}".format(d))  
                    logger.debug("  Found DLLs: {0}".format(found_dlls))  
                    dll_loaded = True
                except Exception as e:
                    logger.error("Failed to add dll directory {0}: {1}".format(d, e))  
    
    if dll_loaded:
        logger.info("Successfully loaded cuDNN from {0} directories".format(len(successful_dirs)))  
        
        # Check for missing critical DLLs
        missing_dlls = []
        for dll in essential_dlls:
            dll_found = False
            for d in successful_dirs:
                if os.path.isfile(os.path.join(d, dll)):
                    dll_found = True
                    break
            if not dll_found:
                missing_dlls.append(dll)
        
        if missing_dlls:
            logger.warning("Missing critical DLLs: {0}".format(missing_dlls))  
            logger.warning("This may cause GPU acceleration issues")
    else:
        logger.warning("No cuDNN DLLs found in any known path. GPU acceleration may be unavailable.")
    
    return dll_loaded

def check_cuda_compatibility():
    """Check CUDA and cuDNN compatibility"""
    try:
        import torch
        logger.info("PyTorch version: {0}".format(torch.__version__))  
        logger.info("CUDA available: {0}".format(torch.cuda.is_available()))  
        if torch.cuda.is_available():
            logger.info("CUDA version: {0}".format(torch.version.cuda))  # type: ignore 
            logger.info("cuDNN version: {0}".format(torch.backends.cudnn.version()))  
            logger.info("GPU device count: {0}".format(torch.cuda.device_count()))  
            for i in range(torch.cuda.device_count()):
                logger.info("GPU {0}: {1}".format(i, torch.cuda.get_device_name(i)))  
        return torch.cuda.is_available()
    except Exception as e:
        logger.error("Error checking CUDA compatibility: {0}".format(e))  
        return False