import unittest
import pandas as pd
import numpy as np
import sys
from unittest.mock import patch, ANY

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from signals._components.LSTM__function__create_classification_targets import create_classification_targets

class TestCreateClassificationTargets(unittest.TestCase):
    """Test cases for create_classification_targets function."""

    def setUp(self):
        """Set up test data."""
        # Create sample price data
        self.dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        self.prices = np.array([
            100, 102, 105, 103, 101, 102, 104, 106, 105, 107,
            110, 112, 115, 113, 111, 112, 114, 116, 115, 117
        ])
        
        # Create DataFrame with OHLCV data
        self.df = pd.DataFrame({
            'close': self.prices,
            'open': self.prices * 0.99,
            'high': self.prices * 1.01,
            'low': self.prices * 0.98,
            'volume': np.random.randint(1000, 10000, size=len(self.prices))
        }, index=self.dates)
        
        # Test parameters
        self.threshold = 0.05
        self.neutral_zone = 0.01

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_successful_classification(self, mock_regression_targets):
        """Test successful creation of classification targets."""
        # Mock regression targets function to return known values
        test_df = self.df.copy()
        test_df['return_target'] = [0.06, -0.07, 0.008, -0.003, 0.04, -0.02, 0.001, 0.08, -0.09, 0.002,
                                   0.03, -0.04, 0.007, -0.001, 0.05, -0.06, 0.009, 0.07, -0.08, 0.0]
        mock_regression_targets.return_value = test_df
        
        result = create_classification_targets(self.df, threshold=self.threshold, neutral_zone=self.neutral_zone)
        
        # Verify that regression targets function was called with correct parameters
        mock_regression_targets.assert_called_once_with(ANY, 'close', -1)
        
        # Check that class_target column was created
        self.assertIn('class_target', result.columns)
        
        # Check specific classifications
        self.assertEqual(result['class_target'].iloc[0], 1)   # 0.06 > 0.05 -> Strong UP
        self.assertEqual(result['class_target'].iloc[1], -1)  # -0.07 < -0.05 -> Strong DOWN
        self.assertEqual(result['class_target'].iloc[2], 0)   # 0.008 <= 0.01 -> Neutral
        self.assertEqual(result['class_target'].iloc[3], 0)   # -0.003 <= 0.01 -> Neutral

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_regression_targets_failure(self, mock_regression_targets):
        """Test handling when regression targets creation fails."""
        # Mock regression targets to return DataFrame without return_target column
        mock_regression_targets.return_value = self.df.copy()
        
        result = create_classification_targets(self.df, threshold=self.threshold, neutral_zone=self.neutral_zone)
        
        # Should return original DataFrame without class_target column
        self.assertNotIn('class_target', result.columns)
        self.assertEqual(len(result), len(self.df))

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_edge_case_thresholds(self, mock_regression_targets):
        """Test classification with edge case return values."""
        test_df = self.df.copy()
        # Create returns that will actually trigger the classification conditions
        # Need values that are clearly above/below thresholds, not equal to them
        edge_values = [
            self.threshold + 0.001,      # Slightly above positive threshold -> Strong UP
            -self.threshold - 0.001,     # Slightly below negative threshold -> Strong DOWN  
            self.neutral_zone - 0.001,   # Within neutral zone -> Neutral
            -self.neutral_zone + 0.001   # Within neutral zone -> Neutral
        ]
        # Repeat pattern and trim to exact DataFrame length
        test_df['return_target'] = (edge_values * ((len(self.df) // len(edge_values)) + 1))[:len(self.df)]
        mock_regression_targets.return_value = test_df
        
        result = create_classification_targets(self.df, threshold=self.threshold, neutral_zone=self.neutral_zone)
        
        # Check boundary classifications with values that actually satisfy the conditions
        self.assertEqual(result['class_target'].iloc[0], 1)   # threshold + 0.001 > threshold -> Strong UP  
        self.assertEqual(result['class_target'].iloc[1], -1)  # -threshold - 0.001 < -threshold -> Strong DOWN
        self.assertEqual(result['class_target'].iloc[2], 0)   # neutral_zone - 0.001 <= neutral_zone -> Neutral
        self.assertEqual(result['class_target'].iloc[3], 0)   # -neutral_zone + 0.001 <= neutral_zone -> Neutral

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_intermediate_values(self, mock_regression_targets):
        """Test classification of intermediate values (between neutral_zone and threshold)."""
        test_df = self.df.copy()
        # Create returns between neutral_zone and threshold
        intermediate_positive = (self.threshold + self.neutral_zone) / 2
        intermediate_negative = -(self.threshold + self.neutral_zone) / 2
        
        # Create array with proper length matching the DataFrame
        return_values = [
            intermediate_positive,   # Between neutral_zone and threshold
            intermediate_negative,   # Between -neutral_zone and -threshold
            0.0                     # Exactly zero
        ]
        # Repeat pattern to match DataFrame length
        test_df['return_target'] = (return_values * (len(self.df) // len(return_values) + 1))[:len(self.df)]
        mock_regression_targets.return_value = test_df
        
        result = create_classification_targets(self.df, threshold=self.threshold, neutral_zone=self.neutral_zone)
        
        # Intermediate values should be classified as neutral (default=0)
        self.assertEqual(result['class_target'].iloc[0], 0)
        self.assertEqual(result['class_target'].iloc[1], 0)
        self.assertEqual(result['class_target'].iloc[2], 0)  # Zero should be neutral

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_custom_parameters(self, mock_regression_targets):
        """Test function with custom parameters."""
        test_df = self.df.copy()
        # Fix array length by trimming to exact DataFrame length
        test_values = [0.03, -0.04, 0.001] * ((len(self.df) // 3) + 1)
        test_df['return_target'] = test_values[:len(self.df)]
        mock_regression_targets.return_value = test_df
        
        # Custom parameters
        custom_threshold = 0.025
        custom_neutral_zone = 0.005
        custom_shift = -2
        custom_target_col = 'high'
        
        result = create_classification_targets(
            self.df, 
            target_col=custom_target_col,
            future_shift=custom_shift,
            threshold=custom_threshold, 
            neutral_zone=custom_neutral_zone
        )
        
        # Verify custom parameters were passed to regression targets
        mock_regression_targets.assert_called_once_with(ANY, custom_target_col, custom_shift)
        
        # Check classifications with custom thresholds
        self.assertEqual(result['class_target'].iloc[0], 1)   # 0.03 > 0.025 -> Strong UP
        self.assertEqual(result['class_target'].iloc[1], -1)  # -0.04 < -0.025 -> Strong DOWN
        self.assertEqual(result['class_target'].iloc[2], 0)   # 0.001 <= 0.005 -> Neutral

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_empty_dataframe(self, mock_regression_targets):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        mock_regression_targets.return_value = empty_df
        
        result = create_classification_targets(empty_df)
        
        self.assertTrue(result.empty)

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_class_distribution_logging(self, mock_regression_targets):
        """Test that class distribution is logged correctly."""
        test_df = self.df.copy()
        # Fix array length by trimming to exact DataFrame length
        test_values = [0.06, -0.07, 0.008] * ((len(self.df) // 3) + 1) 
        test_df['return_target'] = test_values[:len(self.df)]
        mock_regression_targets.return_value = test_df
        
        with patch('signals._components.LSTM__function__create_classification_targets.logger') as mock_logger:
            result = create_classification_targets(self.df, threshold=self.threshold, neutral_zone=self.neutral_zone)
            
            # Verify logging was called
            mock_logger.debug.assert_called_once()
            
            # Check that the logged message contains distribution info
            log_call_args = mock_logger.debug.call_args[0][0]
            self.assertIn("Classification targets distribution:", log_call_args)

    @patch('signals._components.LSTM__function__create_classification_targets.create_regression_targets')
    def test_data_integrity(self, mock_regression_targets):
        """Test that original data is preserved and only classification target is added."""
        test_df = self.df.copy()
        test_df['return_target'] = [0.06] * len(self.df)
        mock_regression_targets.return_value = test_df
        
        result = create_classification_targets(self.df)
        
        # Check that all original columns are preserved (comparing column names only)
        for col in self.df.columns:
            self.assertIn(col, result.columns)
        
        # Check that new columns were added
        self.assertIn('return_target', result.columns)
        self.assertIn('class_target', result.columns)


if __name__ == '__main__':
    unittest.main()
