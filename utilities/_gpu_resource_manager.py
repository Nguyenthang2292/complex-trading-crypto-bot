import torch
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class GPUResourceManager:
    def __init__(self):
        self._current_device: Optional[torch.device] = None
        self._is_cuda_available = torch.cuda.is_available()
        
    def initialize(self, device_id: int = 0) -> bool:
        """Initialize GPU resource management
        
        Args:
            device_id: CUDA device ID to use
            
        Returns:
            bool: True if initialization successful, False otherwise
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
            
    def cleanup(self):
        """Clean up GPU resources"""
        if not self._is_cuda_available:
            return
            
        try:
            if self._current_device:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")
            
    @contextmanager
    def gpu_scope(self, device_id: int = 0):
        """Context manager for GPU resource management"""
        try:
            if not self.initialize(device_id):
                yield None
                return
                
            yield self._current_device
            
        finally:
            self.cleanup()
            
    def get_memory_info(self) -> dict:
        """Get current GPU memory information"""
        if not self._is_cuda_available:
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

# Singleton instance
gpu_resource_manager = GPUResourceManager()

def get_gpu_resource_manager():
    return gpu_resource_manager
