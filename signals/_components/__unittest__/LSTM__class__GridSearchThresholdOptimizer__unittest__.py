import unittest
import numpy as np
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent.parent)) if str(current_dir.parent.parent.parent) not in sys.path else None

from signals._components.LSTM__class__GridSearchThresholdOptimizer import GridSearchThresholdOptimizer

class TestGridSearchThresholdOptimizer(unittest.TestCase):
    """Test cases for GridSearchThresholdOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)  # For reproducible tests
        self.n_samples = 100
        
        # Sample data for regression testing
        self.predictions = np.random.randn(self.n_samples) * 0.02
        self.returns = np.random.randn(self.n_samples) * 0.015
        self.prices = np.cumprod(1 + self.returns) * 100  # Starting price of 100
        
        # Sample data for classification testing
        self.n_classes = 3
        self.probabilities = np.random.rand(self.n_samples, self.n_classes)
        # Normalize to sum to 1 (softmax-like)
        self.probabilities = self.probabilities / np.sum(self.probabilities, axis=1, keepdims=True)
        
        # Default threshold range
        self.default_threshold_range = np.arange(0.01, 0.15, 0.01)
        
    def test_initialization_default_params(self):
        """Test GridSearchThresholdOptimizer initialization with default parameters."""
        optimizer = GridSearchThresholdOptimizer()
        
        np.testing.assert_array_equal(optimizer.threshold_range, self.default_threshold_range)
        self.assertIsNone(optimizer.best_threshold)
        self.assertEqual(optimizer.best_sharpe, -np.inf)
        
    def test_initialization_custom_params(self):
        """Test GridSearchThresholdOptimizer initialization with custom parameters."""
        custom_range = np.arange(0.005, 0.1, 0.005)
        optimizer = GridSearchThresholdOptimizer(threshold_range=custom_range)
        
        np.testing.assert_array_equal(optimizer.threshold_range, custom_range)
        self.assertIsNone(optimizer.best_threshold)
        self.assertEqual(optimizer.best_sharpe, -np.inf)
        
    def test_optimize_regression_threshold_basic(self):
        """Test basic regression threshold optimization."""
        optimizer = GridSearchThresholdOptimizer()
        
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            self.predictions, self.returns, self.prices
        )
        
        # Check return types
        self.assertIsInstance(best_sharpe, float)
        if best_threshold is not None:
            self.assertIsInstance(best_threshold, float)
            self.assertIn(best_threshold, optimizer.threshold_range)
        
        # Check that optimizer state is updated
        self.assertEqual(optimizer.best_threshold, best_threshold)
        self.assertEqual(optimizer.best_sharpe, best_sharpe)
        
    def test_optimize_regression_threshold_with_trend(self):
        """Test regression optimization with trending data."""
        # Create trending predictions and returns
        trend_predictions = np.linspace(-0.05, 0.05, self.n_samples)
        trend_returns = trend_predictions * 0.8 + np.random.randn(self.n_samples) * 0.01
        trend_prices = np.cumprod(1 + trend_returns) * 100
        
        optimizer = GridSearchThresholdOptimizer()
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            trend_predictions, trend_returns, trend_prices
        )
        
        self.assertIsInstance(best_sharpe, float)
        if best_threshold is not None:
            self.assertGreater(best_threshold, 0)
            
    def test_optimize_classification_threshold_basic(self):
        """Test basic classification threshold optimization."""
        optimizer = GridSearchThresholdOptimizer()
        
        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(
            self.probabilities, self.returns
        )
        
        # Check return types
        self.assertIsInstance(best_sharpe, float)
        if best_confidence is not None:
            self.assertIsInstance(best_confidence, float)
            self.assertGreaterEqual(best_confidence, 0.5)
            self.assertLessEqual(best_confidence, 0.95)
            
    def test_optimize_classification_threshold_with_confident_predictions(self):
        """Test classification optimization with confident predictions."""
        # Create more confident probabilities
        confident_probs = np.zeros((self.n_samples, self.n_classes))
        # Make random class predictions with high confidence
        for i in range(self.n_samples):
            class_idx = np.random.randint(0, self.n_classes)
            confident_probs[i, class_idx] = 0.9
            remaining_prob = 0.1
            for j in range(self.n_classes):
                if j != class_idx:
                    confident_probs[i, j] = remaining_prob / (self.n_classes - 1)
        
        optimizer = GridSearchThresholdOptimizer()
        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(
            confident_probs, self.returns
        )
        
        self.assertIsInstance(best_sharpe, float)
        if best_confidence is not None:
            self.assertGreaterEqual(best_confidence, 0.5)
            
    def test_optimize_regression_with_zero_predictions(self):
        """Test regression optimization with all zero predictions."""
        zero_predictions = np.zeros(self.n_samples)
        
        optimizer = GridSearchThresholdOptimizer()
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            zero_predictions, self.returns, self.prices
        )
        
        # With zero predictions, no signals should be generated
        self.assertIsNone(best_threshold)
        self.assertEqual(best_sharpe, -np.inf)
        
    def test_optimize_regression_with_constant_returns(self):
        """Test regression optimization with constant returns."""
        constant_returns = np.full(self.n_samples, 0.01)
        constant_prices = np.cumprod(1 + constant_returns) * 100
        
        optimizer = GridSearchThresholdOptimizer()
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            self.predictions, constant_returns, constant_prices
        )
        
        # Should handle constant returns gracefully
        self.assertIsInstance(best_sharpe, float)
        
    def test_optimize_classification_with_uniform_probabilities(self):
        """Test classification optimization with uniform probabilities."""
        uniform_probs = np.full((self.n_samples, self.n_classes), 1.0 / self.n_classes)
        
        optimizer = GridSearchThresholdOptimizer()
        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(
            uniform_probs, self.returns
        )
        
        # With uniform probabilities, no confident predictions should be made
        self.assertIsNone(best_confidence)
        self.assertEqual(best_sharpe, -np.inf)
        
    def test_empty_data_regression(self):
        """Test regression optimization with empty data."""
        empty_predictions = np.array([])
        empty_returns = np.array([])
        empty_prices = np.array([])
        
        optimizer = GridSearchThresholdOptimizer()
        
        with self.assertRaises(ValueError):
            optimizer.optimize_regression_threshold(empty_predictions, empty_returns, empty_prices)
            
    def test_empty_data_classification(self):
        """Test classification optimization with empty data."""
        empty_probs = np.array([]).reshape(0, self.n_classes)
        empty_returns = np.array([])
        
        optimizer = GridSearchThresholdOptimizer()
        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(
            empty_probs, empty_returns
        )
        
        self.assertIsNone(best_confidence)
        self.assertEqual(best_sharpe, -np.inf)
        
    def test_single_sample_regression(self):
        """Test regression optimization with single sample."""
        single_prediction = np.array([0.05])
        single_return = np.array([0.02])
        single_price = np.array([100.0])
        
        optimizer = GridSearchThresholdOptimizer()
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            single_prediction, single_return, single_price
        )
        
        # Single sample should not produce meaningful results
        self.assertIsNone(best_threshold)
        self.assertEqual(best_sharpe, -np.inf)
        
    def test_mismatched_data_lengths_regression(self):
        """Test regression optimization with mismatched data lengths."""
        short_predictions = self.predictions[:50]
        
        optimizer = GridSearchThresholdOptimizer()
        
        with self.assertRaises(ValueError):
            optimizer.optimize_regression_threshold(short_predictions, self.returns, self.prices)
            
    def test_mismatched_data_lengths_classification(self):
        """Test classification optimization with mismatched data lengths."""
        short_probs = self.probabilities[:50]
        
        optimizer = GridSearchThresholdOptimizer()
        
        with self.assertRaises(ValueError):
            optimizer.optimize_classification_threshold(short_probs, self.returns)
            
    def test_custom_threshold_range(self):
        """Test optimization with custom threshold range."""
        custom_range = np.array([0.02, 0.05, 0.08])
        optimizer = GridSearchThresholdOptimizer(threshold_range=custom_range)
        
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            self.predictions, self.returns, self.prices
        )
        
        if best_threshold is not None:
            self.assertIn(best_threshold, custom_range)
            
    def test_negative_returns_regression(self):
        """Test regression optimization with mostly negative returns."""
        negative_returns = -np.abs(np.random.randn(self.n_samples) * 0.015)
        negative_prices = np.cumprod(1 + negative_returns) * 100
        
        optimizer = GridSearchThresholdOptimizer()
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            self.predictions, negative_returns, negative_prices
        )
        
        # Should handle negative returns
        self.assertIsInstance(best_sharpe, float)
        
    def test_high_volatility_data(self):
        """Test optimization with high volatility data."""
        high_vol_predictions = np.random.randn(self.n_samples) * 0.1
        high_vol_returns = np.random.randn(self.n_samples) * 0.05
        high_vol_prices = np.cumprod(1 + high_vol_returns) * 100
        
        optimizer = GridSearchThresholdOptimizer()
        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(
            high_vol_predictions, high_vol_returns, high_vol_prices
        )
        
        # Should handle high volatility data
        self.assertIsInstance(best_sharpe, float)
        
    def test_optimizer_state_persistence(self):
        """Test that optimizer state persists across calls."""
        optimizer = GridSearchThresholdOptimizer()
        
        # First optimization
        best_threshold_1, best_sharpe_1 = optimizer.optimize_regression_threshold(
            self.predictions, self.returns, self.prices
        )
        
        # Check state
        self.assertEqual(optimizer.best_threshold, best_threshold_1)
        self.assertEqual(optimizer.best_sharpe, best_sharpe_1)
        
        # Second optimization with different data
        new_predictions = np.random.randn(self.n_samples) * 0.03
        best_threshold_2, best_sharpe_2 = optimizer.optimize_regression_threshold(
            new_predictions, self.returns, self.prices
        )
        
        # State should be updated
        self.assertEqual(optimizer.best_threshold, best_threshold_2)
        self.assertEqual(optimizer.best_sharpe, best_sharpe_2)


if __name__ == '__main__':
    unittest.main()
