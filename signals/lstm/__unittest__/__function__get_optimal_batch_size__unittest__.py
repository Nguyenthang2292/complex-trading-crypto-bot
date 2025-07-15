import unittest
from unittest.mock import patch, MagicMock
import torch
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from signals._components.LSTM__function__get_optimal_batch_size import get_optimal_batch_size


class TestGetOptimalBatchSize(unittest.TestCase):
    """Test cases for LSTM batch size optimization function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.sequence_length = 50
        self.cpu_device = torch.device('cpu')
        self.cuda_device = torch.device('cuda')
    
    def test_cpu_device_lstm(self):
        """Test batch size calculation for CPU with LSTM model."""
        with patch('signals._components.LSTM__function__get_optimal_batch_size.CPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cpu_device, self.input_size, self.sequence_length, 'lstm'
            )
            self.assertEqual(result, 64)
    
    def test_cpu_device_lstm_attention(self):
        """Test batch size calculation for CPU with LSTM attention model."""
        with patch('signals._components.LSTM__function__get_optimal_batch_size.CPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cpu_device, self.input_size, self.sequence_length, 'lstm_attention'
            )
            self.assertEqual(result, 32)
    
    def test_cpu_device_cnn_lstm(self):
        """Test batch size calculation for CPU with CNN-LSTM model."""
        with patch('signals._components.LSTM__function__get_optimal_batch_size.CPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cpu_device, self.input_size, self.sequence_length, 'cnn_lstm'
            )
            self.assertEqual(result, 16)
    
    @patch('torch.cuda.get_device_properties')
    def test_gpu_high_memory_lstm(self, mock_get_props):
        """Test batch size for high-end GPU with LSTM."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_get_props.return_value = mock_props
        
        result = get_optimal_batch_size(
            self.cuda_device, self.input_size, self.sequence_length, 'lstm'
        )
        self.assertGreaterEqual(result, 128)
        self.assertLessEqual(result, 1024)
    
    @patch('torch.cuda.get_device_properties')
    def test_gpu_high_memory_cnn_lstm(self, mock_get_props):
        """Test batch size for high-end GPU with CNN-LSTM."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_get_props.return_value = mock_props
        
        result = get_optimal_batch_size(
            self.cuda_device, self.input_size, self.sequence_length, 'cnn_lstm'
        )
        self.assertGreaterEqual(result, 32)
        self.assertLessEqual(result, 256)
    
    @patch('torch.cuda.get_device_properties')
    def test_gpu_mid_memory_lstm_attention(self, mock_get_props):
        """Test batch size for mid-range GPU with LSTM attention."""
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_get_props.return_value = mock_props
        
        result = get_optimal_batch_size(
            self.cuda_device, self.input_size, self.sequence_length, 'lstm_attention'
        )
        self.assertGreaterEqual(result, 32)
        self.assertLessEqual(result, 256)
    
    @patch('torch.cuda.get_device_properties')
    def test_gpu_low_memory_fallback(self, mock_get_props):
        """Test fallback behavior for low-memory GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 4 * 1024**3  # 4GB
        mock_get_props.return_value = mock_props
        
        with patch('signals._components.LSTM__function__get_optimal_batch_size.GPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cuda_device, self.input_size, self.sequence_length, 'cnn_lstm'
            )
            self.assertEqual(result, 16)
    
    @patch('torch.cuda.get_device_properties')
    def test_memory_safety_check(self, mock_get_props):
        """Test memory safety check prevents excessive memory usage."""
        mock_props = MagicMock()
        mock_props.total_memory = 6 * 1024**3  # 6GB
        mock_get_props.return_value = mock_props
        
        # Use very large input to trigger safety check
        large_input_size = 1000
        large_sequence_length = 1000
        
        result = get_optimal_batch_size(
            self.cuda_device, large_input_size, large_sequence_length, 'cnn_lstm'
        )
        self.assertGreaterEqual(result, 4)
    
    @patch('torch.cuda.get_device_properties')
    def test_exception_handling(self, mock_get_props):
        """Test exception handling when GPU properties cannot be retrieved."""
        mock_get_props.side_effect = RuntimeError("CUDA not available")
        
        with patch('signals._components.LSTM__function__get_optimal_batch_size.GPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cuda_device, self.input_size, self.sequence_length, 'lstm'
            )
            self.assertEqual(result, 64)
    
    @patch('torch.cuda.get_device_properties')
    def test_exception_handling_cnn_lstm(self, mock_get_props):
        """Test exception handling for CNN-LSTM model type."""
        mock_get_props.side_effect = RuntimeError("CUDA not available")
        
        with patch('signals._components.LSTM__function__get_optimal_batch_size.GPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cuda_device, self.input_size, self.sequence_length, 'cnn_lstm'
            )
            self.assertEqual(result, 16)
    
    def test_default_model_type(self):
        """Test default model type parameter."""
        with patch('signals._components.LSTM__function__get_optimal_batch_size.CPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cpu_device, self.input_size, self.sequence_length
            )
            self.assertEqual(result, 64)
    
    @patch('torch.cuda.get_device_properties')
    def test_memory_calculation_accuracy(self, mock_get_props):
        """Test memory calculation accuracy for different model types."""
        mock_props = MagicMock()
        mock_props.total_memory = 12 * 1024**3  # 12GB
        mock_get_props.return_value = mock_props
        
        lstm_batch = get_optimal_batch_size(
            self.cuda_device, self.input_size, self.sequence_length, 'lstm'
        )
        attention_batch = get_optimal_batch_size(
            self.cuda_device, self.input_size, self.sequence_length, 'lstm_attention'
        )
        cnn_batch = get_optimal_batch_size(
            self.cuda_device, self.input_size, self.sequence_length, 'cnn_lstm'
        )
        
        # CNN-LSTM should have smallest batch due to highest memory requirements
        # LSTM should have largest batch due to lowest memory requirements
        self.assertGreater(lstm_batch, attention_batch)
        self.assertGreater(attention_batch, cnn_batch)
    
    @patch('torch.cuda.get_device_properties')
    def test_edge_case_zero_input_size(self, mock_get_props):
        """Test edge case with zero input size."""
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_get_props.return_value = mock_props
        
        result = get_optimal_batch_size(
            self.cuda_device, 0, self.sequence_length, 'lstm'
        )
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
    
    @patch('torch.cuda.get_device_properties')
    def test_edge_case_zero_sequence_length(self, mock_get_props):
        """Test edge case with zero sequence length."""
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_get_props.return_value = mock_props
        
        result = get_optimal_batch_size(
            self.cuda_device, self.input_size, 0, 'lstm'
        )
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
    
    def test_type_hints_compliance(self):
        """Test that function returns correct type."""
        with patch('signals._components.LSTM__function__get_optimal_batch_size.CPU_BATCH_SIZE', 64):
            result = get_optimal_batch_size(
                self.cpu_device, self.input_size, self.sequence_length, 'lstm'
            )
            self.assertIsInstance(result, int)


if __name__ == '__main__':
    unittest.main()