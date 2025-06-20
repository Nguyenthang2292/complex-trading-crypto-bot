import torch
import logging
from typing import Optional, Dict, Union, Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class GPUResourceManager:
    """
    GPU resource management for PyTorch operations.
    
    Provides centralized GPU device management, memory cleanup, and resource monitoring
    for machine learning workflows with automatic fallback to CPU when CUDA unavailable.
    """
    
    def __init__(self) -> None:
        self._current_device: Optional[torch.device] = None
        self._is_cuda_available: bool = torch.cuda.is_available()
        
    def initialize(self, device_id: int = 0) -> bool:
        """
        Initialize GPU resource management with specified device.
        
        Args:
            device_id: CUDA device ID to use for operations
            
        Returns:
            True if GPU initialization successful, False if falling back to CPU
        """
        if not self._is_cuda_available:
            logger.warning("CUDA is not available. Falling back to CPU.")
            return False
            
        try:
            self._current_device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(self._current_device)
            logger.info(f"GPU resource manager initialized on device: {self._current_device}")
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
            logger.info("GPU resources cleaned up successfully")
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

gpu_resource_manager: GPUResourceManager = GPUResourceManager()

def get_gpu_resource_manager() -> GPUResourceManager:
    """Get the singleton GPU resource manager instance."""
    return gpu_resource_manager
