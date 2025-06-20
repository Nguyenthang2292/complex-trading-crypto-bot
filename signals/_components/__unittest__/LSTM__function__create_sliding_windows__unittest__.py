import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from signals._components.LSTM__function__create_sliding_windows import create_sliding_windows


class TestCreateSlidingWindows(unittest.TestCase):
    """Test cases for create_sliding_windows function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample time series data
        np.random.seed(42)
        self.n_rows = 100
        self.feature_data = {
            'feature1': np.random.randn(self.n_rows),
            'feature2': np.random.randn(self.n_rows),
            'feature3': np.random.randn(self.n_rows),
            'target': np.random.choice([-1, 0, 1], self.n_rows)
        }
        self.df = pd.DataFrame(self.feature_data)
        
        # Small dataset for edge cases
        self.small_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, -1, 1, 0]
        })
    
    def test_basic_functionality(self):
        """Test basic sliding window creation."""
        look_back = 10
        X, y, feature_cols = create_sliding_windows(
            self.df, look_back=look_back, target_col='target'
        )
        
        expected_samples = len(self.df) - look_back
        expected_features = len(self.df.columns) - 1  # Exclude target
        
        self.assertEqual(X.shape, (expected_samples, look_back, expected_features))
        self.assertEqual(y.shape, (expected_samples,))
        self.assertEqual(len(feature_cols), expected_features)
        self.assertNotIn('target', feature_cols)
    
    def test_with_specified_feature_columns(self):
        """Test sliding windows with specific feature columns."""
        look_back = 5
        specified_features = ['feature1', 'feature3']
        
        X, y, feature_cols = create_sliding_windows(
            self.df, 
            look_back=look_back, 
            target_col='target',
            feature_cols=specified_features
        )
        
        expected_samples = len(self.df) - look_back
        
        self.assertEqual(X.shape, (expected_samples, look_back, 2))
        self.assertEqual(y.shape, (expected_samples,))
        self.assertEqual(feature_cols, specified_features)
    
    def test_window_values_correctness(self):
        """Test that window values are correctly extracted."""
        look_back = 3
        X, y, feature_cols = create_sliding_windows(
            self.small_df, look_back=look_back, target_col='target'
        )
        
        # Check first window
        expected_first_window = self.small_df[['feature1', 'feature2']].iloc[0:3].values
        np.testing.assert_array_equal(X[0], expected_first_window)
        self.assertEqual(y[0], self.small_df['target'].iloc[3])
        
        # Check second window
        expected_second_window = self.small_df[['feature1', 'feature2']].iloc[1:4].values
        np.testing.assert_array_equal(X[1], expected_second_window)
        self.assertEqual(y[1], self.small_df['target'].iloc[4])
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with patch('signals._components.LSTM__function__create_sliding_windows.logger') as mock_logger:
            X, y, feature_cols = create_sliding_windows(empty_df)
            
            mock_logger.warning.assert_called_with("Input data is empty")
            self.assertEqual(len(X), 0)
            self.assertEqual(len(y), 0)
            self.assertEqual(feature_cols, [])
    
    def test_missing_target_column(self):
        """Test handling when target column is missing."""
        df_no_target = self.df.drop('target', axis=1)
        
        with patch('signals._components.LSTM__function__create_sliding_windows.logger') as mock_logger:
            X, y, feature_cols = create_sliding_windows(df_no_target, target_col='target')
            
            mock_logger.error.assert_called_with("Target column 'target' not found in data")
            self.assertEqual(len(X), 0)
            self.assertEqual(len(y), 0)
            self.assertEqual(feature_cols, [])
    
    def test_insufficient_data(self):
        """Test handling when data has insufficient rows."""
        look_back = 10
        insufficient_df = self.small_df  # Only 5 rows
        
        with patch('signals._components.LSTM__function__create_sliding_windows.logger') as mock_logger:
            X, y, feature_cols = create_sliding_windows(
                insufficient_df, look_back=look_back, target_col='target'
            )
            
            mock_logger.warning.assert_called_with(
                "Insufficient data: need at least 11 rows, got 5"
            )
            self.assertEqual(len(X), 0)
            self.assertEqual(len(y), 0)
            self.assertEqual(feature_cols, [])
    
    def test_minimum_data_requirement(self):
        """Test with exactly minimum required data."""
        look_back = 4
        min_df = self.small_df  # 5 rows, need look_back + 1 = 5
        
        X, y, feature_cols = create_sliding_windows(
            min_df, look_back=look_back, target_col='target'
        )
        
        # Should create exactly 1 sample
        self.assertEqual(X.shape, (1, look_back, 2))
        self.assertEqual(y.shape, (1,))
        self.assertEqual(len(feature_cols), 2)
    
    def test_custom_target_column_name(self):
        """Test with custom target column name."""
        df_custom = self.df.rename(columns={'target': 'custom_target'})
        look_back = 5
        
        X, y, feature_cols = create_sliding_windows(
            df_custom, look_back=look_back, target_col='custom_target'
        )
        
        expected_samples = len(df_custom) - look_back
        self.assertEqual(X.shape, (expected_samples, look_back, 3))
        self.assertEqual(y.shape, (expected_samples,))
        self.assertNotIn('custom_target', feature_cols)
    
    def test_single_feature_column(self):
        """Test with DataFrame having single feature column."""
        single_feature_df = pd.DataFrame({
            'single_feature': np.random.randn(20),
            'target': np.random.choice([-1, 0, 1], 20)
        })
        look_back = 5
        
        X, y, feature_cols = create_sliding_windows(
            single_feature_df, look_back=look_back, target_col='target'
        )
        
        expected_samples = len(single_feature_df) - look_back
        self.assertEqual(X.shape, (expected_samples, look_back, 1))
        self.assertEqual(y.shape, (expected_samples,))
        self.assertEqual(feature_cols, ['single_feature'])
    
    def test_different_look_back_values(self):
        """Test with different look_back values."""
        test_cases = [1, 5, 10, 30]
        
        for look_back in test_cases:
            with self.subTest(look_back=look_back):
                X, y, feature_cols = create_sliding_windows(
                    self.df, look_back=look_back, target_col='target'
                )
                
                expected_samples = len(self.df) - look_back
                expected_features = len(self.df.columns) - 1
                
                self.assertEqual(X.shape, (expected_samples, look_back, expected_features))
                self.assertEqual(y.shape, (expected_samples,))
    
    def test_data_types_preserved(self):
        """Test that data types are preserved correctly."""
        X, y, feature_cols = create_sliding_windows(
            self.df, look_back=5, target_col='target'
        )
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(feature_cols, list)
        self.assertTrue(np.issubdtype(X.dtype, np.floating))
        self.assertTrue(np.issubdtype(y.dtype, np.number))
    
    def test_feature_column_order(self):
        """Test that feature column order is maintained."""
        specified_features = ['feature3', 'feature1', 'feature2']
        
        X, y, feature_cols = create_sliding_windows(
            self.df,
            look_back=5,
            target_col='target',
            feature_cols=specified_features
        )
        
        self.assertEqual(feature_cols, specified_features)
        
        # Verify data matches the specified order
        expected_first_window = self.df[specified_features].iloc[0:5].values
        np.testing.assert_array_equal(X[0], expected_first_window)
    
    @patch('signals._components.LSTM__function__create_sliding_windows.logger')
    def test_logging_behavior(self, mock_logger):
        """Test that logging is called appropriately."""
        look_back = 10
        X, y, feature_cols = create_sliding_windows(
            self.df, look_back=look_back, target_col='target'
        )
        
        expected_samples = len(self.df) - look_back
        mock_logger.model.assert_called_with(
            "Created {0} sliding windows with shape X: {1}, y: {2}".format(
                expected_samples, X.shape, y.shape
            )
        )
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create larger dataset
        large_n = 10000
        large_df = pd.DataFrame({
            'feature1': np.random.randn(large_n),
            'feature2': np.random.randn(large_n),
            'feature3': np.random.randn(large_n),
            'target': np.random.choice([-1, 0, 1], large_n)
        })
        
        look_back = 60
        X, y, feature_cols = create_sliding_windows(
            large_df, look_back=look_back, target_col='target'
        )
        
        expected_samples = large_n - look_back
        self.assertEqual(X.shape, (expected_samples, look_back, 3))
        self.assertEqual(y.shape, (expected_samples,))
    
    def test_return_type_consistency(self):
        """Test that return types are consistent across different inputs."""
        test_configs = [
            {'look_back': 5, 'feature_cols': None},
            {'look_back': 10, 'feature_cols': ['feature1']},
            {'look_back': 3, 'feature_cols': ['feature1', 'feature2']}
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                result = create_sliding_windows(
                    self.df, target_col='target', **config
                )
                
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 3)
                X, y, feature_cols = result
                self.assertIsInstance(X, np.ndarray)
                self.assertIsInstance(y, np.ndarray)
                self.assertIsInstance(feature_cols, list)


if __name__ == '__main__':
    unittest.main()