from pathlib import Path
import unittest
import pandas as pd
import numpy as np
import sys
from unittest.mock import patch

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent.parent)) if str(current_dir.parent.parent.parent) not in sys.path else None

from signals._components.LSTM__function__create_regression_targets import create_regression_targets

class TestCreateRegressionTargets(unittest.TestCase):
    """Test cases for create_regression_targets function."""

    def setUp(self):
        """Set up test data."""
        self.dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        self.prices = np.array([100, 102, 105, 103, 101, 102, 104, 106, 105, 107])
        
        self.df = pd.DataFrame({
            'close': self.prices,
            'open': self.prices * 0.99,
            'high': self.prices * 1.01,
            'low': self.prices * 0.98,
            'volume': np.random.randint(1000, 10000, size=len(self.prices))
        }, index=self.dates)

    def test_basic_functionality(self):
        """Test basic regression target creation with default parameters."""
        result = create_regression_targets(self.df)
        
        # Check that return_target column was created
        self.assertIn('return_target', result.columns)
        
        # Check that original columns are preserved
        for col in self.df.columns:
            self.assertIn(col, result.columns)
        
        # Check that DataFrame length is preserved
        self.assertEqual(len(result), len(self.df))

    def test_custom_target_column(self):
        """Test with custom target column."""
        result = create_regression_targets(self.df, target_col='high')
        
        self.assertIn('return_target', result.columns)
        # Verify calculation is based on 'high' column
        expected_returns = (self.df['high'].shift(-1) - self.df['high']) / self.df['high']
        expected_returns = np.clip(expected_returns, -0.1, 0.1)
        
        # Compare non-NaN values
        mask = ~result['return_target'].isna()
        pd.testing.assert_series_equal(
            result['return_target'][mask], 
            expected_returns[mask], 
            check_names=False
        )

    def test_custom_future_shift(self):
        """Test with custom future shift parameter."""
        shift = -2
        result = create_regression_targets(self.df, future_shift=shift)
        
        self.assertIn('return_target', result.columns)
        
        # Verify calculation with custom shift
        expected_returns = (self.df['close'].shift(shift) - self.df['close']) / self.df['close']
        expected_returns = np.clip(expected_returns, -0.1, 0.1)
        
        mask = ~result['return_target'].isna()
        pd.testing.assert_series_equal(
            result['return_target'][mask], 
            expected_returns[mask], 
            check_names=False
        )

    def test_positive_future_shift(self):
        """Test with positive future shift (looking backward)."""
        shift = 1
        result = create_regression_targets(self.df, future_shift=shift)
        
        self.assertIn('return_target', result.columns)
        
        # Verify calculation with positive shift
        expected_returns = (self.df['close'].shift(shift) - self.df['close']) / self.df['close']
        expected_returns = np.clip(expected_returns, -0.1, 0.1)
        
        mask = ~result['return_target'].isna()
        pd.testing.assert_series_equal(
            result['return_target'][mask], 
            expected_returns[mask], 
            check_names=False
        )

    def test_missing_target_column(self):
        """Test with missing target column."""
        result = create_regression_targets(self.df, target_col='nonexistent')
        
        # Should return original DataFrame without return_target column
        self.assertNotIn('return_target', result.columns)
        self.assertEqual(len(result), len(self.df))
        
        # Original columns should be preserved
        for col in self.df.columns:
            self.assertIn(col, result.columns)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = create_regression_targets(empty_df)
        
        self.assertTrue(result.empty)

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        single_row_df = self.df.iloc[[0]]
        result = create_regression_targets(single_row_df)
        
        self.assertIn('return_target', result.columns)
        self.assertEqual(len(result), 1)
        # With single row and default shift=-1, return should be NaN
        self.assertTrue(pd.isna(result['return_target'].iloc[0]))

    def test_outlier_clipping(self):
        """Test that extreme returns are clipped to [-10%, +10%] range."""
        # Create DataFrame with extreme price movements
        extreme_prices = [100, 150, 50, 200, 25]  # Extreme movements > 10%
        extreme_df = pd.DataFrame({
            'close': extreme_prices,
            'open': [p * 0.99 for p in extreme_prices],
            'high': [p * 1.01 for p in extreme_prices],
            'low': [p * 0.98 for p in extreme_prices],
            'volume': [1000] * len(extreme_prices)
        })
        
        result = create_regression_targets(extreme_df)
        
        # All returns should be clipped to [-0.1, 0.1] range
        valid_returns = result['return_target'].dropna()
        self.assertTrue(all(valid_returns >= -0.1))
        self.assertTrue(all(valid_returns <= 0.1))

    def test_nan_handling(self):
        """Test handling of NaN values in price data."""
        df_with_nan = self.df.copy()
        df_with_nan.loc[df_with_nan.index[2], 'close'] = np.nan
        
        result = create_regression_targets(df_with_nan)
        
        self.assertIn('return_target', result.columns)
        # Should handle NaN gracefully
        self.assertEqual(len(result), len(df_with_nan))

    @patch('signals._components.LSTM__function__create_regression_targets.logger')
    def test_logging(self, mock_logger):
        """Test that statistics are logged correctly."""
        result = create_regression_targets(self.df)
        
        # Verify logging was called
        mock_logger.debug.assert_called_once()
        
        # Check that logged message contains statistics
        log_call_args = mock_logger.debug.call_args[0][0]
        self.assertIn("Created regression targets", log_call_args)
        self.assertIn("mean return:", log_call_args)
        self.assertIn("std:", log_call_args)

    def test_calculation_accuracy(self):
        """Test accuracy of return calculations."""
        # Use simple prices for easy verification
        simple_prices = [100, 110, 90, 105, 95]
        simple_df = pd.DataFrame({
            'close': simple_prices,
            'open': [p * 0.99 for p in simple_prices],
            'high': [p * 1.01 for p in simple_prices],
            'low': [p * 0.98 for p in simple_prices],
            'volume': [1000] * len(simple_prices)
        })
        
        result = create_regression_targets(simple_df, future_shift=-1)
        
        # Manual calculation for verification
        # Return[0] = (110 - 100) / 100 = 0.1
        # Return[1] = (90 - 110) / 110 = -0.181818... -> clipped to -0.1
        # Return[2] = (105 - 90) / 90 = 0.166666... -> clipped to 0.1
        # Return[3] = (95 - 105) / 105 = -0.095238...
        # Return[4] = NaN (no future price)
        
        expected_returns = [0.1, -0.1, 0.1, -0.095238, np.nan]
        
        for i, expected in enumerate(expected_returns[:-1]):  # Exclude NaN case
            self.assertAlmostEqual(result['return_target'].iloc[i], expected, places=5)
        
        # Last value should be NaN
        self.assertTrue(pd.isna(result['return_target'].iloc[-1]))

    def test_data_integrity(self):
        """Test that original data is not modified."""
        original_df = self.df.copy()
        result = create_regression_targets(self.df)
        
        # Original DataFrame should be unchanged
        pd.testing.assert_frame_equal(self.df, original_df)
        
        # Result should have additional column
        self.assertEqual(len(result.columns), len(original_df.columns) + 1)


if __name__ == '__main__':
    unittest.main()
