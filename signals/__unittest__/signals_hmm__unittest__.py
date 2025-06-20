import unittest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from signals.signals_hmm import hmm_signals, initialize_ray, Signal
from signals._components.HMM__class__OptimizingParameters import OptimizingParameters
from signals.signals_hmm import Signal

class TestSignalsHMM(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        self.optimizing_params = OptimizingParameters()
    
    @patch('signals.signals_hmm.ray')
    def test_initialize_ray_not_initialized(self, mock_ray):
        """Test Ray initialization when not already initialized"""
        mock_ray.is_initialized.return_value = False
        
        initialize_ray()
        
        mock_ray.is_initialized.assert_called_once()
        mock_ray.init.assert_called_once()
    
    @patch('signals.signals_hmm.ray')
    def test_initialize_ray_already_initialized(self, mock_ray):
        """Test Ray initialization when already initialized"""
        mock_ray.is_initialized.return_value = True
        
        initialize_ray()
        
        mock_ray.is_initialized.assert_called_once()
        mock_ray.init.assert_not_called()
    
    @patch('signals.signals_hmm.hmm_high_order')
    @patch('signals.signals_hmm.hmm_kama')
    def test_hmm_signals_long_signals(self, mock_hmm_kama, mock_hmm_high_order):
        """Test hmm_signals returning LONG signals"""
        # Mock high-order HMM result
        mock_high_order_result = Mock()
        mock_high_order_result.next_state_with_high_order_hmm = 1
        mock_high_order_result.next_state_probability = 0.85
        mock_hmm_high_order.return_value = mock_high_order_result
        
        # Mock HMM-KAMA result for LONG signal
        mock_kama_result = Mock()
        mock_kama_result.next_state_with_hmm_kama = 1
        mock_kama_result.current_state_of_state_using_std = 1
        mock_kama_result.current_state_of_state_using_hmm = 1
        mock_kama_result.current_state_of_state_using_kmeans = 1
        mock_kama_result.state_high_probabilities_using_arm_apriori = 1
        mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 3
        mock_hmm_kama.return_value = mock_kama_result
        
        result = hmm_signals(self.sample_df, self.optimizing_params)
        
        self.assertEqual(result, (1, 1))  # Both signals should be LONG
    
    @patch('signals.signals_hmm.hmm_high_order')
    @patch('signals.signals_hmm.hmm_kama')
    def test_hmm_signals_short_signals(self, mock_hmm_kama, mock_hmm_high_order):
        """Test hmm_signals returning SHORT signals"""
        # Mock high-order HMM result
        mock_high_order_result = Mock()
        mock_high_order_result.next_state_with_high_order_hmm = -1
        mock_high_order_result.next_state_probability = 0.85
        mock_hmm_high_order.return_value = mock_high_order_result
        
        # Mock HMM-KAMA result for SHORT signal
        mock_kama_result = Mock()
        mock_kama_result.next_state_with_hmm_kama = 0
        mock_kama_result.current_state_of_state_using_std = 1
        mock_kama_result.current_state_of_state_using_hmm = 1
        mock_kama_result.current_state_of_state_using_kmeans = 1
        mock_kama_result.state_high_probabilities_using_arm_apriori = 0
        mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 2
        mock_hmm_kama.return_value = mock_kama_result
        
        result = hmm_signals(self.sample_df, self.optimizing_params)
        
        self.assertEqual(result, (-1, -1))  # Both signals should be SHORT
    
    @patch('signals.signals_hmm.hmm_high_order')
    @patch('signals.signals_hmm.hmm_kama')
    def test_hmm_signals_hold_low_probability(self, mock_hmm_kama, mock_hmm_high_order):
        """Test hmm_signals returning HOLD due to low probability"""
        # Mock high-order HMM result with low probability
        mock_high_order_result = Mock()
        mock_high_order_result.next_state_with_high_order_hmm = 0  # Changed to 0 for HOLD
        mock_high_order_result.next_state_probability = 0.5  # Below threshold
        mock_hmm_high_order.return_value = mock_high_order_result
        
        # Mock HMM-KAMA result with low scores
        mock_kama_result = Mock()
        mock_kama_result.next_state_with_hmm_kama = 5  # Neutral state
        mock_kama_result.current_state_of_state_using_std = 0
        mock_kama_result.current_state_of_state_using_hmm = 0
        mock_kama_result.current_state_of_state_using_kmeans = 0
        mock_kama_result.state_high_probabilities_using_arm_apriori = 5
        mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 5
        mock_hmm_kama.return_value = mock_kama_result
        
        result = hmm_signals(self.sample_df, self.optimizing_params)
        
        self.assertEqual(result, (0, 0))  # Both signals should be HOLD
    
    @patch('signals.signals_hmm.hmm_high_order')
    @patch('signals.signals_hmm.hmm_kama')
    def test_hmm_signals_exception_handling(self, mock_hmm_kama, mock_hmm_high_order):
        """Test hmm_signals exception handling"""
        mock_hmm_kama.side_effect = Exception("Model error")
        
        with patch('signals.signals_hmm.logger') as mock_logger:
            result = hmm_signals(self.sample_df, self.optimizing_params)
            
            self.assertEqual(result, (0, 0))  # Should return HOLD, HOLD
            mock_logger.error.assert_called_once()
    
    @patch('signals.signals_hmm.hmm_high_order')
    @patch('signals.signals_hmm.hmm_kama')
    def test_hmm_signals_kama_scoring_system(self, mock_hmm_kama, mock_hmm_high_order):
        """Test HMM-KAMA scoring system with edge cases"""
        # Mock high-order HMM result (HOLD)
        mock_high_order_result = Mock()
        mock_high_order_result.next_state_with_high_order_hmm = 0
        mock_high_order_result.next_state_probability = 0.5
        mock_hmm_high_order.return_value = mock_high_order_result
        
        # Mock HMM-KAMA result with exact threshold scores
        mock_kama_result = Mock()
        mock_kama_result.next_state_with_hmm_kama = 1  # +2 to long
        mock_kama_result.current_state_of_state_using_std = 1  # +1 to both
        mock_kama_result.current_state_of_state_using_hmm = 0  # +0
        mock_kama_result.current_state_of_state_using_kmeans = 0  # +0
        mock_kama_result.state_high_probabilities_using_arm_apriori = 5  # +0
        mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 5  # +0
        mock_hmm_kama.return_value = mock_kama_result
        
        result = hmm_signals(self.sample_df, self.optimizing_params)
        
        # Long score: 2 + 1 = 3, Short score: 1, should be LONG
        self.assertEqual(result, (0, 1))
    
    @patch('signals.signals_hmm.hmm_high_order')
    @patch('signals.signals_hmm.hmm_kama')
    def test_hmm_signals_logging(self, mock_hmm_kama, mock_hmm_high_order):
        """Test logging behavior for active signals"""
        # Mock results for active signals
        mock_high_order_result = Mock()
        mock_high_order_result.next_state_with_high_order_hmm = 1
        mock_high_order_result.next_state_probability = 0.85
        mock_hmm_high_order.return_value = mock_high_order_result
        
        mock_kama_result = Mock()
        mock_kama_result.next_state_with_hmm_kama = 1
        mock_kama_result.current_state_of_state_using_std = 1
        mock_kama_result.current_state_of_state_using_hmm = 1
        mock_kama_result.current_state_of_state_using_kmeans = 1
        mock_kama_result.state_high_probabilities_using_arm_apriori = 1
        mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 1
        mock_hmm_kama.return_value = mock_kama_result
        
        with patch('signals.signals_hmm.logger') as mock_logger:
            hmm_signals(self.sample_df, self.optimizing_params)
            mock_logger.info.assert_called_once()
    
    def test_signal_type_annotation(self):
        """Test that Signal type is properly defined"""
        # This test ensures the type annotation is correct
        self.assertEqual(Signal.__args__, (-1, 0, 1))
    
    @patch('signals.signals_hmm.hmm_high_order')
    @patch('signals.signals_hmm.hmm_kama')
    def test_hmm_signals_return_type(self, mock_hmm_kama, mock_hmm_high_order):
        """Test that hmm_signals returns correct types"""
        mock_high_order_result = Mock()
        mock_high_order_result.next_state_with_high_order_hmm = 0
        mock_high_order_result.next_state_probability = 0.5
        mock_hmm_high_order.return_value = mock_high_order_result
        
        mock_kama_result = Mock()
        mock_kama_result.next_state_with_hmm_kama = 5
        mock_kama_result.current_state_of_state_using_std = 0
        mock_kama_result.current_state_of_state_using_hmm = 0
        mock_kama_result.current_state_of_state_using_kmeans = 0
        mock_kama_result.state_high_probabilities_using_arm_apriori = 5
        mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 5
        mock_hmm_kama.return_value = mock_kama_result
        
        result = hmm_signals(self.sample_df, self.optimizing_params)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIn(result[0], [-1, 0, 1])
        self.assertIn(result[1], [-1, 0, 1])


if __name__ == '__main__':
    unittest.main()