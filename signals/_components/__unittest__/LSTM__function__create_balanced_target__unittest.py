import unittest
import pandas as pd
import numpy as np
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the function to test
from signals._components.LSTM__function__create_balanced_target import create_balanced_target
from livetrade.config import FUTURE_RETURN_SHIFT

class TestCreateBalancedTarget(unittest.TestCase):
    """Test cases for create_balanced_target function."""

    def setUp(self):
        """Set up test data."""
        # Create sample price data
        self.dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        self.prices = np.array([
            100, 102, 105, 103, 101, 102, 104, 106, 105, 107,
            110, 112, 115, 113, 111, 112, 114, 116, 115, 117,
            120, 118, 115, 113, 111, 110, 108, 106, 105, 103
        ])
        
        # Create DataFrame with OHLCV data
        self.df = pd.DataFrame({
            'close': self.prices,
            'open': self.prices * 0.99,  # Just dummy values
            'high': self.prices * 1.01,  # Just dummy values
            'low': self.prices * 0.98,   # Just dummy values
            'volume': np.random.randint(1000, 10000, size=len(self.prices))
        }, index=self.dates)
        
        # Parameters for testing
        self.threshold = 0.05
        self.neutral_zone = 0.01

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = create_balanced_target(empty_df, self.threshold, self.neutral_zone)
        # Should return the empty dataframe without error
        self.assertTrue(result.empty)

    def test_insufficient_data(self):
        """Test with insufficient data points."""
        small_df = self.df.iloc[:abs(FUTURE_RETURN_SHIFT)]  # Not enough data
        result = create_balanced_target(small_df, self.threshold, self.neutral_zone)
        # Should return the original dataframe without adding Target column
        self.assertEqual(len(result), len(small_df))
        self.assertNotIn('Target', result.columns)

    def test_strong_movement_classification(self):
        """Test classification of strong price movements."""
        # Create a test dataframe with more data points to accommodate FUTURE_RETURN_SHIFT
        shift = abs(FUTURE_RETURN_SHIFT)
        # Need at least shift+4 rows for our test cases plus handling shift
        total_rows = shift + 4
        
        # Initialize with base values
        df = pd.DataFrame({
            'close': [100.0] * total_rows,
            'open': [99.0] * total_rows,
            'high': [101.0] * total_rows,
            'low': [98.0] * total_rows,
            'volume': [1000] * total_rows
        })
        
        # Explicitly set the base prices that we'll use to calculate returns
        base_prices = df['close'].copy()
        
        # Create synthetic future prices to generate specific return scenarios
        if FUTURE_RETURN_SHIFT < 0:  # Historical data affects future (typical case)
            # For the first 4 data points (which we'll test), set specific future returns
            future_prices = base_prices.copy()
            
            # First point: Strong upward movement
            future_prices.iloc[0 - FUTURE_RETURN_SHIFT] = base_prices.iloc[0] * (1 + self.threshold + 0.01)
            
            # Second point: Strong downward movement
            future_prices.iloc[1 - FUTURE_RETURN_SHIFT] = base_prices.iloc[1] * (1 - self.threshold - 0.01)
            
            # Third point: Neutral (small upward)
            future_prices.iloc[2 - FUTURE_RETURN_SHIFT] = base_prices.iloc[2] * (1 + self.neutral_zone / 2)
            
            # Fourth point: Neutral (small downward)
            future_prices.iloc[3 - FUTURE_RETURN_SHIFT] = base_prices.iloc[3] * (1 - self.neutral_zone / 2)
            
            # Replace close prices with our engineered scenario
            df['close'] = future_prices
        else:  # Future data affects historical (unusual but handle it)
            # Similar logic but with indices reversed
            for i in range(4):
                if i == 0:
                    # Strong upward movement
                    df.loc[i, 'close'] = base_prices.iloc[i + FUTURE_RETURN_SHIFT] / (1 + self.threshold + 0.01)
                elif i == 1:
                    # Strong downward movement
                    df.loc[i, 'close'] = base_prices.iloc[i + FUTURE_RETURN_SHIFT] / (1 - self.threshold - 0.01)
                elif i == 2:
                    # Neutral (small upward)
                    df.loc[i, 'close'] = base_prices.iloc[i + FUTURE_RETURN_SHIFT] / (1 + self.neutral_zone / 2)
                elif i == 3:
                    # Neutral (small downward)
                    df.loc[i, 'close'] = base_prices.iloc[i + FUTURE_RETURN_SHIFT] / (1 - self.neutral_zone / 2)
        
        # Run the function with fixed random seed
        np.random.seed(42)
        result = create_balanced_target(df, self.threshold, self.neutral_zone)
        
        # Verify results exist
        self.assertIn('Target', result.columns)
        
        # We need at least 4 rows in the result to run our tests
        self.assertGreaterEqual(len(result), 4)
        
        # Check classifications
        self.assertEqual(result['Target'].iloc[0], 1.0)  # Strong up -> Buy
        self.assertEqual(result['Target'].iloc[1], -1.0)  # Strong down -> Sell
        self.assertEqual(result['Target'].iloc[2], 0.0)  # Small up -> Neutral
        self.assertEqual(result['Target'].iloc[3], 0.0)  # Small down -> Neutral

    def test_intermediate_classification(self):
        """Test probabilistic classification of intermediate movements."""
        # Similar fix is needed here
        shift = abs(FUTURE_RETURN_SHIFT)
        total_rows = shift + 3  # Need at least this many rows
        
        # Initialize with base values
        df = pd.DataFrame({
            'close': [100.0] * total_rows,
            'open': [99.0] * total_rows,
            'high': [101.0] * total_rows,
            'low': [98.0] * total_rows,
            'volume': [1000] * total_rows
        })
        
        # Base prices to calculate returns from
        base_prices = df['close'].copy()
        
        # Create intermediate returns (between neutral_zone and threshold)
        positive_intermediate_return = (self.threshold + self.neutral_zone) / 2  # Between neutral and threshold (positive)
        negative_intermediate_return = -(self.threshold + self.neutral_zone) / 2  # Between neutral and threshold (negative)
        
        if FUTURE_RETURN_SHIFT < 0:  # Historical data affects future
            future_prices = base_prices.copy()
            
            # Set specific future returns for test cases
            future_prices.iloc[0 - FUTURE_RETURN_SHIFT] = base_prices.iloc[0] * (1 + positive_intermediate_return)
            future_prices.iloc[1 - FUTURE_RETURN_SHIFT] = base_prices.iloc[1] * (1 + negative_intermediate_return)
            
            # Replace close prices with our engineered scenario
            df['close'] = future_prices
        else:  # Future data affects historical
            df.loc[0, 'close'] = base_prices.iloc[0 + FUTURE_RETURN_SHIFT] / (1 + positive_intermediate_return)
            df.loc[1, 'close'] = base_prices.iloc[1 + FUTURE_RETURN_SHIFT] / (1 + negative_intermediate_return)
        
        # Run with fixed random seed
        np.random.seed(42)
        result = create_balanced_target(df, self.threshold, self.neutral_zone)
        
        # Verify results exist
        self.assertIn('Target', result.columns)
        self.assertGreaterEqual(len(result), 2)
        
        # Check that intermediate cases are assigned (could be 1/0 or -1/0)
        self.assertIn(result['Target'].iloc[0], [0.0, 1.0])
        self.assertIn(result['Target'].iloc[1], [-1.0, 0.0])

    def test_nan_handling(self):
        """Test that NaN targets are removed."""
        result = create_balanced_target(self.df, self.threshold, self.neutral_zone)
        
        # There should be abs(FUTURE_RETURN_SHIFT) fewer rows due to NaN targets at the end
        expected_length = len(self.df) - abs(FUTURE_RETURN_SHIFT)
        self.assertEqual(len(result), expected_length)
        
        # All target values should be valid (-1, 0, or 1)
        self.assertTrue(all(result['Target'].isin([-1.0, 0.0, 1.0])))


if __name__ == '__main__':
    unittest.main()
