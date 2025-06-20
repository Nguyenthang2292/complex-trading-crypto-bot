# filepath: c:\Users\Admin\Desktop\complex-trading-crypto-bot\tests\test_LSTM_evaluate_models_clean.py
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from signals._components.LSTM__function__evaluate_models import (
    apply_confidence_threshold,
    evaluate_model_in_batches,
    evaluate_model_with_confidence
)


class TestApplyConfidenceThreshold(unittest.TestCase):
    """Test cases for apply_confidence_threshold function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample probability data
        self.y_proba = np.array([
            [0.1, 0.2, 0.7],  # High confidence BUY (class 1)
            [0.8, 0.1, 0.1],  # High confidence SELL (class -1)
            [0.3, 0.4, 0.3],  # Low confidence - should be NEUTRAL
            [0.2, 0.9, 0.1],  # High confidence NEUTRAL (class 0)
            [0.4, 0.3, 0.3],  # Low confidence - should be NEUTRAL
        ])
    
    def test_high_confidence_predictions(self):
        """Test predictions with high confidence threshold."""
        threshold = 0.7
        result = apply_confidence_threshold(self.y_proba, threshold)
        expected = np.array([1, -1, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_low_confidence_predictions(self):
        """Test predictions with low confidence threshold."""
        threshold = 0.3
        result = apply_confidence_threshold(self.y_proba, threshold)
        expected = np.array([1, -1, 0, 0, -1])
        np.testing.assert_array_equal(result, expected)
    
    def test_medium_confidence_predictions(self):
        """Test predictions with medium confidence threshold."""
        threshold = 0.5
        result = apply_confidence_threshold(self.y_proba, threshold)
        expected = np.array([1, -1, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_class_mapping(self):
        """Test correct mapping from probabilities to class labels."""
        y_proba = np.array([
            [0.9, 0.05, 0.05],  # SELL (-1)
            [0.1, 0.8, 0.1],    # NEUTRAL (0)
            [0.1, 0.1, 0.8],    # BUY (1)
        ])
        threshold = 0.5
        result = apply_confidence_threshold(y_proba, threshold)
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_edge_case_equal_threshold(self):
        """Test behavior when confidence equals threshold."""
        y_proba = np.array([[0.5, 0.3, 0.2]])
        threshold = 0.5
        result = apply_confidence_threshold(y_proba, threshold)
        expected = np.array([-1])
        np.testing.assert_array_equal(result, expected)
    
    def test_empty_input(self):
        """Test with empty input array."""
        y_proba = np.array([]).reshape(0, 3)
        threshold = 0.5
        result = apply_confidence_threshold(y_proba, threshold)
        self.assertEqual(len(result), 0)
    
    def test_single_sample(self):
        """Test with single sample."""
        y_proba = np.array([[0.2, 0.1, 0.7]])
        threshold = 0.6
        result = apply_confidence_threshold(y_proba, threshold)
        expected = np.array([1])
        np.testing.assert_array_equal(result, expected)


class TestEvaluateModelInBatches(unittest.TestCase):
    """Test cases for evaluate_model_in_batches function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.cuda_device = torch.device('cuda')
        
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.eval = MagicMock()
        
        # Create test data
        self.X_test = torch.randn(100, 10, 5)
        
        # Mock model output
        self.mock_output = torch.randn(32, 3)
        self.mock_model.return_value = self.mock_output
    
    def test_cpu_evaluation(self):
        """Test model evaluation on CPU."""
        with patch('torch.no_grad'):
            result = evaluate_model_in_batches(
                self.mock_model, self.X_test, self.device, batch_size=32
            )
            
            self.mock_model.eval.assert_called_once()
            self.assertIsInstance(result, np.ndarray)
    
    @patch('torch.cuda.empty_cache')
    @patch.object(torch.Tensor, 'to', lambda self, device: self)
    def test_gpu_evaluation_with_cache_clearing(self, mock_empty_cache):
        """Test GPU evaluation with cache clearing."""
        with patch('torch.no_grad'):
            batch_size = 10
            result = evaluate_model_in_batches(
                self.mock_model, self.X_test, self.cuda_device, batch_size=batch_size
            )
            
            self.assertTrue(mock_empty_cache.called)    
    
    @patch('signals._components.LSTM__function__evaluate_models.logger')
    def test_oom_error_handling(self, mock_logger):
        """Test out of memory error handling."""
        # Create a model that raises OOM on first call
        # Then returns valid tensor on subsequent calls
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        
        # Create test data and mock tensors
        test_data = self.X_test[:32]
        mock_output = torch.randn(16, 3)
        
        # Configure the mock to raise OOM on first call and return tensor on second call
        first_call = True
        def side_effect(*args, **kwargs):
            nonlocal first_call
            if first_call:
                first_call = False
                raise RuntimeError("CUDA out of memory")
            return mock_output
        
        mock_model.side_effect = side_effect
        
        with patch('torch.no_grad'):
            with patch.object(torch.Tensor, 'to', return_value=test_data):
                result = evaluate_model_in_batches(
                    mock_model, test_data, self.cuda_device, batch_size=32
                )
                
                mock_logger.warning.assert_called()
                self.assertIsInstance(result, np.ndarray)
    
    def test_non_oom_error_propagation(self):
        """Test that non-OOM errors are properly propagated."""
        other_error = RuntimeError("Different error")
        self.mock_model.side_effect = other_error
        
        with patch('torch.no_grad'):
            with self.assertRaises(RuntimeError):
                evaluate_model_in_batches(
                    self.mock_model, self.X_test, self.device, batch_size=32
                )
    
    @patch('signals._components.LSTM__function__evaluate_models.logger')
    @patch.object(torch.Tensor, 'to', lambda self, device: self)
    def test_batch_size_reduction_limit(self, mock_logger):
        """Test that batch size reduction has a limit."""
        oom_error = RuntimeError("CUDA out of memory")
        self.mock_model.side_effect = oom_error
        
        with patch('torch.no_grad'):
            with self.assertRaises(RuntimeError):
                evaluate_model_in_batches(
                    self.mock_model, self.X_test[:1], self.cuda_device, batch_size=1
                )
            
            mock_logger.error.assert_called()


class TestEvaluateModelWithConfidence(unittest.TestCase):
    """Test cases for evaluate_model_with_confidence function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.mock_model = MagicMock()
        self.X_test = torch.randn(50, 10, 5)
        self.y_test = np.array([-1, 0, 1] * 16 + [-1, 0])
        
        # Mock prediction probabilities
        self.mock_y_pred_prob = np.random.rand(50, 3)
    
    @patch('signals._components.LSTM__function__evaluate_models.evaluate_model_in_batches')
    @patch('signals._components.LSTM__function__evaluate_models.CONFIDENCE_THRESHOLDS', [0.5, 0.7])
    @patch('signals._components.LSTM__function__evaluate_models.CONFIDENCE_THRESHOLD', 0.6)
    @patch('signals._components.LSTM__function__evaluate_models.logger')
    def test_successful_evaluation(self, mock_logger, mock_eval_batches):
        """Test successful model evaluation with confidence thresholds."""
        mock_eval_batches.return_value = self.mock_y_pred_prob
        
        evaluate_model_with_confidence(
            self.mock_model, self.X_test, self.y_test, self.device
        )
        
        # Should log analysis results
        self.assertTrue(mock_logger.analysis.called)
        # Should evaluate for each threshold
        self.assertGreaterEqual(mock_logger.analysis.call_count, 4)
    
    @patch('signals._components.LSTM__function__evaluate_models.evaluate_model_in_batches')
    @patch('signals._components.LSTM__function__evaluate_models.logger')
    def test_evaluation_failure_handling(self, mock_logger, mock_eval_batches):
        """Test handling of evaluation failures."""
        mock_eval_batches.side_effect = RuntimeError("Evaluation failed")
        
        result = evaluate_model_with_confidence(
            self.mock_model, self.X_test, self.y_test, self.device
        )
        
        # Should return None and log error
        self.assertIsNone(result)
        mock_logger.error.assert_called()
    
    @patch('signals._components.LSTM__function__evaluate_models.evaluate_model_in_batches')
    @patch('signals._components.LSTM__function__evaluate_models.logger')
    def test_gpu_batch_size_selection(self, mock_logger, mock_eval_batches):
        """Test GPU uses smaller batch size for evaluation."""
        cuda_device = torch.device('cuda')
        mock_eval_batches.return_value = self.mock_y_pred_prob
        
        evaluate_model_with_confidence(
            self.mock_model, self.X_test, self.y_test, cuda_device
        )
        
        # Should call with batch size 16 for GPU
        mock_eval_batches.assert_called_with(
            self.mock_model, self.X_test, cuda_device, 16
        )
    
    @patch('signals._components.LSTM__function__evaluate_models.evaluate_model_in_batches')
    @patch('signals._components.LSTM__function__evaluate_models.logger')
    def test_cpu_batch_size_selection(self, mock_logger, mock_eval_batches):
        """Test CPU uses larger batch size for evaluation."""
        mock_eval_batches.return_value = self.mock_y_pred_prob
        
        evaluate_model_with_confidence(
            self.mock_model, self.X_test, self.y_test, self.device
        )
        
        # Should call with batch size 32 for CPU
        mock_eval_batches.assert_called_with(
            self.mock_model, self.X_test, self.device, 32
        )


if __name__ == '__main__':
    unittest.main()