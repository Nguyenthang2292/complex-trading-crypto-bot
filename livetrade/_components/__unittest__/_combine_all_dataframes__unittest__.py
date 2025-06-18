import logging
import numpy as np
import pandas as pd
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

current_dir = Path(__file__).parent
main_dir = current_dir.parent.parent
if str(main_dir) not in sys.path:
    sys.path.insert(0, str(main_dir))
# Import function to test
from livetrade._components._combine_all_dataframes import combine_all_dataframes, logger

# Disable logger during tests
logger.setLevel(logging.CRITICAL)

class TestCombineAllDataframes(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample date range
        start_date = datetime.now() - timedelta(days=10)
        self.dates = [start_date + timedelta(hours=i) for i in range(24)]
        
        # Create sample dataframes with all required columns
        self.valid_df1 = pd.DataFrame({
            'open': [100.0 + i for i in range(24)],
            'high': [105.0 + i for i in range(24)],
            'low': [95.0 + i for i in range(24)],
            'close': [101.0 + i for i in range(24)],
            'volume': [1000.0 + i*10 for i in range(24)]
        }, index=self.dates)
        
        self.valid_df2 = pd.DataFrame({
            'open': [200.0 + i for i in range(24)],
            'high': [205.0 + i for i in range(24)],
            'low': [195.0 + i for i in range(24)],
            'close': [201.0 + i for i in range(24)],
            'volume': [2000.0 + i*10 for i in range(24)]
        }, index=self.dates)
        
        # Create dataframe with missing columns
        self.missing_cols_df = pd.DataFrame({
            'open': [300.0 + i for i in range(24)],
            'close': [301.0 + i for i in range(24)],
        }, index=self.dates)
        
        # Create dataframe with NaN values
        self.nan_df = pd.DataFrame({
            'open': [400.0 + i if i % 2 == 0 else np.nan for i in range(24)],
            'high': [405.0 + i if i % 3 == 0 else np.nan for i in range(24)],
            'low': [395.0 + i if i % 4 == 0 else np.nan for i in range(24)],
            'close': [401.0 + i if i % 5 == 0 else np.nan for i in range(24)],
            'volume': [4000.0 + i*10 for i in range(24)]
        }, index=self.dates)
        
        # Create dataframe with string values that need conversion
        self.string_df = pd.DataFrame({
            'open': [str(500.0 + i) for i in range(24)],
            'high': [str(505.0 + i) for i in range(24)],
            'low': [str(495.0 + i) for i in range(24)],
            'close': [str(501.0 + i) for i in range(24)],
            'volume': [str(5000.0 + i*10) for i in range(24)]
        }, index=self.dates)
        
    def test_valid_input(self):
        """Test with valid input data"""
        # Create test input with multiple symbols and timeframes
        input_data = {
            'BTC': {
                '1h': self.valid_df1,
                '4h': self.valid_df2,
            },
            'ETH': {
                '1h': self.valid_df2,
                '4h': self.valid_df1,
            }
        }
        
        # Call the function
        result = combine_all_dataframes(input_data)
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24*4)  # 24 rows * 4 dataframes
        self.assertTrue('symbol' in result.columns)
        self.assertTrue('timeframe' in result.columns)
        self.assertEqual(result['symbol'].nunique(), 2)  # BTC, ETH
        self.assertEqual(result['timeframe'].nunique(), 2)  # 1h, 4h
        
        # Check if all required columns exist
        for col in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']:
            self.assertTrue(col in result.columns)
    
    def test_empty_input(self):
        """Test with empty input"""
        # Empty dictionary
        result = combine_all_dataframes({})
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        
        # None values in nested dictionaries
        input_data = {'BTC': {'1h': None}}  # type: ignore
        result = combine_all_dataframes(input_data)  # type: ignore
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        
        # Empty dataframes
        input_data = {'BTC': {'1h': pd.DataFrame()}}
        result = combine_all_dataframes(input_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    def test_invalid_input_type(self):
        """Test with invalid input type"""
        with self.assertRaises(TypeError):
            combine_all_dataframes("not a dictionary")  # type: ignore
        
        with self.assertRaises(TypeError):
            combine_all_dataframes(None)  # type: ignore
    
    def test_missing_columns(self):
        """Test with dataframes missing required columns"""
        input_data = {
            'BTC': {
                '1h': self.valid_df1,
                '4h': self.missing_cols_df,
            }
        }
        
        result = combine_all_dataframes(input_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24)  # Only the valid dataframe should be included
        self.assertEqual(result['symbol'].nunique(), 1)
        self.assertEqual(result['timeframe'].nunique(), 1)
    
    def test_type_conversion(self):
        """Test with dataframes containing string values that need conversion"""
        input_data = {
            'BTC': {
                '1h': self.string_df,
            }
        }
        
        result = combine_all_dataframes(input_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24)
        
        # Verify numeric conversion
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[col]))
    
    def test_nan_handling(self):
        """Test with dataframes containing NaN values"""
        input_data = {
            'BTC': {
                '1h': self.nan_df,
            }
        }
        
        result = combine_all_dataframes(input_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24)
        
        # Verify NaN values are preserved
        self.assertTrue(result['open'].isna().any())
        self.assertTrue(result['high'].isna().any())
        self.assertTrue(result['low'].isna().any())
        self.assertTrue(result['close'].isna().any())
    
    def test_mixed_input(self):
        """Test with mixed valid and invalid inputs"""
        input_data = {
            'BTC': {
                '1h': self.valid_df1,
                '4h': None,  # type: ignore
                '1d': "not a dataframe"  # type: ignore
            },
            'ETH': {},
            'XRP': {
                '1h': self.nan_df,
                '4h': self.string_df
            },
            'LTC': "not a dictionary"  # type: ignore
        }
        
        result = combine_all_dataframes(input_data)  # type: ignore
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['symbol'].nunique(), 2)  # BTC and XRP
        # Changed: count unique (symbol, timeframe) pairs, which should be 3
        self.assertEqual(result[['symbol','timeframe']].drop_duplicates().shape[0], 3)
        
        # Count expected dataframes (valid_df1, nan_df, string_df)
        expected_count = 24 * 3
        self.assertEqual(len(result), expected_count)

if __name__ == '__main__':
    unittest.main()
