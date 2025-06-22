import torch
import logging
import os
from typing import Optional, Dict, Union, Generator, List, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class GPUResourceManager:
    """
    Unified GPU resource management for PyTorch operations.
    
    Provides centralized GPU device management, memory cleanup, resource monitoring,
    CUDA environment setup, and compatibility checking with automatic fallback to CPU.
    """
    
    def __init__(self) -> None:
        self._current_device: Optional[torch.device] = None
        self._is_cuda_available: bool = False
        self._initialization_attempted: bool = False
        self._cuda_dlls_loaded: bool = False
        
    def setup_environment(self) -> None:
        """Setup complete CUDA environment including DLLs and environment variables."""
        if not self._initialization_attempted:
            self._setup_cuda_environment()
            self._cuda_dlls_loaded = self._load_cuda_dlls()
            self._is_cuda_available = self._check_gpu_availability()
            self._initialization_attempted = True
            
            if self._is_cuda_available:
                self._configure_gpu_memory()
                
    def _setup_cuda_environment(self) -> None:
        """Setup CUDA environment variables for PyTorch compatibility."""
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
                logger.debug(f"Added to PATH: {cuda_path}")
                paths_added += 1
        
        # Set CUDA environment variables
        cudnn_path: str = r"C:\Program Files\NVIDIA\CUDNN\v9.10"
        cuda_toolkit_path: str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
        
        if os.path.exists(cudnn_path):
            os.environ['CUDNN_PATH'] = cudnn_path
        if os.path.exists(cuda_toolkit_path):
            os.environ['CUDA_PATH'] = cuda_toolkit_path
        
        logger.debug(f"CUDA environment setup complete - {paths_added} paths added")

    def _load_cuda_dlls(self) -> bool:
        """Load CUDA and cuDNN DLLs for GPU acceleration support."""
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
                    logger.debug(f"Added DLL directory: {directory}")
                    dll_loaded = True
                except Exception as e:
                    logger.debug(f"Failed to add dll directory {directory}: {e}")
        
        if dll_loaded:
            logger.debug(f"Successfully loaded cuDNN from {len(successful_dirs)} directories")
        else:
            logger.debug("No cuDNN DLLs found in any known path")
        
        return dll_loaded

    def _check_gpu_availability(self) -> bool:
        """Check GPU availability and perform comprehensive diagnostic testing."""
        logger.info("="*60)
        logger.info("GPU AVAILABILITY CHECK (PyTorch)")
        logger.info("="*60)
        
        gpu_available: bool = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {gpu_available}")
        
        if not gpu_available:
            logger.warning("No GPU detected - will use CPU")
            logger.info("="*60)
            return False
        
        try:
            gpu_count: int = torch.cuda.device_count()
            logger.info(f"Number of GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                try:
                    gpu_name: str = torch.cuda.get_device_name(i)
                    gpu_memory: float = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                except Exception as gpu_info_error:
                    logger.warning(f"Could not get info for GPU {i}: {gpu_info_error}")
                
            # Test GPU functionality
            try:
                device: torch.device = torch.device('cuda:0')
                test_tensor: torch.Tensor = torch.randn(10, 10, device=device)
                result: torch.Tensor = torch.sum(test_tensor)
                logger.info("GPU test successful - tensor operations working")
                logger.info("="*60)
                return True
            except Exception as gpu_test_error:
                self._handle_gpu_error(gpu_test_error)
                logger.info("="*60)
                return False
                
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            logger.warning("Consider installing CPU-only PyTorch if GPU issues persist")
            logger.info("="*60)
            return False

    def _handle_gpu_error(self, error: Exception) -> None:
        """Handle specific GPU errors with targeted solutions."""
        error_msg: str = str(error).lower()
        
        if "cudnn" in error_msg or "cudnn_graph64_9.dll" in error_msg:
            logger.error(f"cuDNN library error detected: {error}")
            logger.warning("cuDNN not properly installed or incompatible version")
            logger.info("Solutions:")
            logger.info("1. Install cuDNN: conda install cudnn -c conda-forge")
            logger.info("2. Reinstall PyTorch with CUDA: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
            logger.info("3. Use CPU-only version: conda install pytorch torchvision torchaudio cpuonly -c pytorch")
        elif "invalid handle" in error_msg:
            logger.error(f"CUDA driver/runtime mismatch: {error}")
            logger.warning("CUDA driver and runtime versions may be incompatible")
            logger.info("Check CUDA installation: nvidia-smi")
        else:
            logger.error(f"GPU tensor test failed: {error}")
        
        logger.warning("Falling back to CPU mode due to GPU issues")

    def _configure_gpu_memory(self) -> bool:
        """Configure PyTorch GPU memory settings for optimal performance."""
        if not self._is_cuda_available:
            return False
            
        try:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.75)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
            
            logger.info("GPU memory configuration optimized for performance")
            return True
        except Exception as e:
            error_msg: str = str(e).lower()
            if "cudnn" in error_msg:
                logger.warning(f"cuDNN configuration failed: {e}")
                logger.info("cuDNN may not be properly installed - continuing with basic GPU settings")
            else:
                logger.warning(f"Could not configure GPU memory: {e}")
                logger.info("Continuing with default GPU settings")
            return False
        
    def initialize(self, device_id: int = 0) -> bool:
        """
        Initialize GPU resource management with specified device.
        
        Args:
            device_id: CUDA device ID to use for operations
            
        Returns:
            True if GPU initialization successful, False if falling back to CPU
        """
        # Ensure environment is set up first
        if not self._initialization_attempted:
            self.setup_environment()
            
        if not self._is_cuda_available:
            logger.debug("CUDA is not available. Using CPU.")
            return False
            
        try:
            self._current_device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(self._current_device)
            logger.debug(f"GPU resource manager initialized on device: {self._current_device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GPU resource manager: {e}")
            return False
            
    def cleanup(self) -> None:
        """Clean up GPU memory cache and synchronize operations."""
        if not self._is_cuda_available or not self._current_device:
            return
            
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")
            
    @contextmanager
    def gpu_scope(self, device_id: int = 0) -> Generator[Optional[torch.device], None, None]:
        """
        Context manager for automatic GPU resource management.
        
        Initializes GPU device on entry and cleans up resources on exit.
        Yields None if GPU initialization fails.
        
        Args:
            device_id: CUDA device ID to use within context
            
        Yields:
            torch.device object if successful, None if CPU fallback
        """
        try:
            device_to_yield: Optional[torch.device] = self._current_device if self.initialize(device_id) else None
            yield device_to_yield
        finally:
            self.cleanup()
            
    def get_memory_info(self) -> Dict[str, Union[int, str]]:
        """
        Get current GPU memory usage statistics.
        
        Returns:
            Dictionary containing memory information:
            - total: Total device memory in bytes
            - allocated: Currently allocated memory in bytes  
            - cached: Reserved memory in bytes
            - device: Device identifier string
        """
        if not self._is_cuda_available or not self._current_device:
            return {
                'total': 0,
                'allocated': 0,
                'cached': 0,
                'device': 'cpu'
            }
            
        return {
            'total': torch.cuda.get_device_properties(self._current_device).total_memory,
            'allocated': torch.cuda.memory_allocated(self._current_device),
            'cached': torch.cuda.memory_reserved(self._current_device),
            'device': str(self._current_device)
        }

    def check_cuda_compatibility(self) -> bool:
        """
        Check CUDA and cuDNN compatibility with current PyTorch installation.
        
        Returns:
            True if CUDA is available and compatible, False otherwise
        """
        try:
            logger.info(f"PyTorch version: {torch.__version__}")
            cuda_available: bool = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            
            if cuda_available:
                if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):  # type: ignore
                    logger.info(f"CUDA version: {torch.version.cuda}")  # type: ignore
                if torch.backends.cudnn.is_available():
                    logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
                    
                device_count: int = torch.cuda.device_count()
                logger.info(f"GPU device count: {device_count}")
                
                for i in range(device_count):
                    device_name: str = torch.cuda.get_device_name(i)
                    logger.info(f"GPU {i}: {device_name}")
                    
            return cuda_available
        except Exception as e:
            logger.error(f"Error checking CUDA compatibility: {e}")
            return False

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available after environment setup."""
        if not self._initialization_attempted:
            self.setup_environment()
        return self._is_cuda_available

    @property
    def current_device(self) -> Optional[torch.device]:
        """Get current GPU device."""
        return self._current_device

    def get_tensor_core_info(self) -> Dict[str, Any]:
        """Get comprehensive Tensor Core information for optimization.
        
        Returns:
            Dictionary containing Tensor Core capabilities and optimization info
        """
        info = {
            'has_tensor_cores': False,
            'compute_capability': None,
            'optimal_batch_multiple': 1,
            'optimal_dim_multiple': 1,
            'gpu_name': 'CPU',
            'generation': 'N/A'
        }
        
        if not self._is_cuda_available:
            return info
        
        try:
            compute_capability = torch.cuda.get_device_capability(0)
            major, minor = compute_capability
            info['compute_capability'] = f"{major}.{minor}"
            info['gpu_name'] = torch.cuda.get_device_name(0)
            
            # Tensor Core support matrix
            if major >= 9:  # Hopper H100+
                info['has_tensor_cores'] = True
                info['optimal_batch_multiple'] = 8
                info['optimal_dim_multiple'] = 8
                info['generation'] = 'Hopper'
            elif major == 8:  # Ampere/Ada Lovelace
                info['has_tensor_cores'] = True
                info['optimal_batch_multiple'] = 8
                info['optimal_dim_multiple'] = 8
                if minor >= 9:  # Ada Lovelace RTX 40xx
                    info['generation'] = 'Ada Lovelace'
                else:  # Ampere RTX 30xx, A100
                    info['generation'] = 'Ampere'
            elif major == 7:  # Volta/Turing
                info['has_tensor_cores'] = True
                info['optimal_batch_multiple'] = 8
                info['optimal_dim_multiple'] = 8
                if minor >= 5:  # Turing RTX 20xx
                    info['generation'] = 'Turing'
                else:  # Volta V100
                    info['generation'] = 'Volta'
            
            logger.info(f"GPU: {info['gpu_name']} ({info['generation']}) - "
                        f"Tensor Cores: {'Yes' if info['has_tensor_cores'] else 'No'}")
                      
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")
        
        return info

# Legacy wrapper function for backward compatibility
def get_tensor_core_info() -> Dict[str, Any]:
    """Get comprehensive Tensor Core information for optimization.
    
    Legacy wrapper function that uses the singleton GPU resource manager.
    
    Returns:
        Dictionary containing Tensor Core capabilities and optimization info
    """
    return gpu_resource_manager.get_tensor_core_info()
  
# Singleton instance
gpu_resource_manager: GPUResourceManager = GPUResourceManager()

def get_gpu_resource_manager() -> GPUResourceManager:
    """Get the singleton GPU resource manager instance."""
    return gpu_resource_manager

# Legacy wrapper functions for backward compatibility
def check_gpu_availability() -> bool:
    """Legacy wrapper for GPU availability check."""
    return gpu_resource_manager.is_cuda_available

def configure_gpu_memory() -> bool:
    """Legacy wrapper for GPU memory configuration."""
    gpu_resource_manager.setup_environment()
    return gpu_resource_manager._configure_gpu_memory()

def setup_cuda_environment() -> None:
    """Legacy wrapper for CUDA environment setup."""
    gpu_resource_manager._setup_cuda_environment()

def load_cuda_dlls() -> bool:
    """Legacy wrapper for CUDA DLL loading."""
    return gpu_resource_manager._load_cuda_dlls()

def check_cuda_compatibility() -> bool:
    """Legacy wrapper for CUDA compatibility check."""
    return gpu_resource_manager.check_cuda_compatibility()
