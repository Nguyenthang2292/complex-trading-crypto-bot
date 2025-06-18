import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from signals.signals_random_forest import (
    _calculate_features,
    train_random_forest_model,
    get_latest_random_forest_signal,
    train_and_save_global_rf_model,
)

from livetrade.config import (
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE,
    COL_BB_UPPER, COL_BB_LOWER,
    SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL,
    MAX_TRAINING_ROWS)

class TestSignalRandomForest(unittest.TestCase):
    """Test cases for signals_random_forest module"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample OHLC data
        np.random.seed(42)
        self.sample_size = 100
        
        # Generate realistic OHLC data
        close_prices = 100 + np.cumsum(np.random.randn(self.sample_size) * 0.5)
        high_prices = close_prices + np.random.uniform(0, 2, self.sample_size)
        low_prices = close_prices - np.random.uniform(0, 2, self.sample_size)
        open_prices = close_prices + np.random.uniform(-1, 1, self.sample_size)
        
        self.sample_df = pd.DataFrame({
            COL_OPEN: open_prices,
            COL_HIGH: high_prices,
            COL_LOW: low_prices,
            COL_CLOSE: close_prices
        })
        
        # Create a larger dataset for training
        self.training_size = 200
        close_prices_large = 100 + np.cumsum(np.random.randn(self.training_size) * 0.5)
        high_prices_large = close_prices_large + np.random.uniform(0, 2, self.training_size)
        low_prices_large = close_prices_large - np.random.uniform(0, 2, self.training_size)
        open_prices_large = close_prices_large + np.random.uniform(-1, 1, self.training_size)
        
        self.training_df = pd.DataFrame({
            COL_OPEN: open_prices_large,
            COL_HIGH: high_prices_large,
            COL_LOW: low_prices_large,
            COL_CLOSE: close_prices_large
        })
        
        # Create empty DataFrame for edge cases
        self.empty_df = pd.DataFrame()
        
        # Create DataFrame with missing columns
        self.incomplete_df = pd.DataFrame({
            COL_OPEN: [100, 101, 102],
            COL_HIGH: [105, 106, 107]
            # Missing LOW and CLOSE columns
        })

        # Create very large dataset for testing memory constraints
        self.large_dataset_size = MAX_TRAINING_ROWS + 100
        large_close = 100 + np.cumsum(np.random.randn(self.large_dataset_size) * 0.5)
        self.large_df = pd.DataFrame({
            COL_OPEN: large_close + np.random.uniform(-1, 1, self.large_dataset_size),
            COL_HIGH: large_close + np.random.uniform(0, 2, self.large_dataset_size),
            COL_LOW: large_close - np.random.uniform(0, 2, self.large_dataset_size),
            COL_CLOSE: large_close
        })

    def test_calculate_features_success(self):
        """Test successful feature calculation"""
        result = _calculate_features(self.sample_df)
        
        # Check that result is not empty
        self.assertFalse(result.empty)
        
        # Check that all expected features are present
        expected_features = ['rsi', 'macd', 'macd_signal', COL_BB_UPPER, COL_BB_LOWER, 'ma_20']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check that OHLC columns are preserved
        for col in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]:
            self.assertIn(col, result.columns)
        
        # Check that RSI values are in valid range (0-100)
        rsi_values = result['rsi'].dropna()
        if not rsi_values.empty:
            self.assertTrue(all(0 <= val <= 100 for val in rsi_values))

    def test_calculate_features_empty_input(self):
        """Test feature calculation with empty DataFrame"""
        result = _calculate_features(self.empty_df)
        self.assertTrue(result.empty)

    def test_calculate_features_missing_columns(self):
        """Test feature calculation with missing required columns"""
        result = _calculate_features(self.incomplete_df)
        self.assertTrue(result.empty)

    def test_calculate_features_single_row(self):
        """Test feature calculation with single row of data"""
        single_row_df = self.sample_df.head(1)
        result = _calculate_features(single_row_df)
        
        # Should handle single row gracefully
        self.assertIsInstance(result, pd.DataFrame)

    def test_calculate_features_bollinger_bands_calculation(self):
        """Test specific Bollinger Bands calculation logic"""
        result = _calculate_features(self.sample_df)
        
        if not result.empty:
            # Check that Bollinger Bands are calculated correctly
            self.assertIn(COL_BB_UPPER, result.columns)
            self.assertIn(COL_BB_LOWER, result.columns)
            
            # Upper band should be greater than lower band
            valid_rows = result.dropna(subset=[COL_BB_UPPER, COL_BB_LOWER])
            if len(valid_rows) > 0:
                self.assertTrue(all(valid_rows[COL_BB_UPPER] >= valid_rows[COL_BB_LOWER]))

    def test_calculate_features_macd_edge_case(self):
        """Test MACD calculation with insufficient data"""
        small_df = self.sample_df.head(5)  # Very small dataset
        result = _calculate_features(small_df)
        
        # Should handle small datasets without crashing
        self.assertIsInstance(result, pd.DataFrame)

    @patch('signals.signals_random_forest.ta.rsi')
    def test_calculate_features_rsi_error(self, mock_rsi):
        """Test RSI calculation error handling"""
        mock_rsi.side_effect = Exception("RSI calculation error")
        
        result = _calculate_features(self.sample_df)
        
        # Should handle RSI calculation errors gracefully
        self.assertIsInstance(result, pd.DataFrame)

    @patch('signals.signals_random_forest.ta.macd')
    def test_calculate_features_macd_error(self, mock_macd):
        """Test MACD calculation error handling"""
        mock_macd.side_effect = Exception("MACD calculation error")
        
        result = _calculate_features(self.sample_df)
        
        # Should handle MACD calculation errors gracefully
        self.assertIsInstance(result, pd.DataFrame)

    @patch('signals.signals_random_forest.ta.sma')
    def test_calculate_features_sma_error(self, mock_sma):
        """Test SMA calculation error handling"""
        mock_sma.side_effect = Exception("SMA calculation error")
        
        result = _calculate_features(self.sample_df)
        
        # Should handle SMA calculation errors gracefully
        self.assertIsInstance(result, pd.DataFrame)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    def test_train_random_forest_model_insufficient_memory(self, mock_memory):
        """Test model training with insufficient memory"""
        # Mock insufficient memory
        mock_memory.return_value.available = 500 * 1024**2  # 500MB
        
        result = train_random_forest_model(self.training_df)
        self.assertIsNone(result)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_train_random_forest_model_success(self, mock_memory):
        """Test successful model training"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        with patch('signals.signals_random_forest.joblib.dump'):
            result = train_random_forest_model(self.training_df, save_model=False)
            
            # Should return a trained model
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, 'predict'))
            self.assertTrue(hasattr(result, 'predict_proba'))

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_train_random_forest_model_large_dataset_sampling(self, mock_memory):
        """Test model training with large dataset that needs sampling"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        with patch('signals.signals_random_forest.joblib.dump'):
            result = train_random_forest_model(self.large_df, save_model=False)
            
            # Should handle large datasets by sampling
            self.assertIsNotNone(result)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_train_random_forest_model_column_mapping(self, mock_memory):
        """Test model training with lowercase column names"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        # Create DataFrame with lowercase column names
        df_lowercase = self.training_df.copy()
        df_lowercase.columns = ['open', 'high', 'low', 'close']
        
        with patch('signals.signals_random_forest.joblib.dump'):
            result = train_random_forest_model(df_lowercase, save_model=False)
            
            # Should handle column name mapping
            self.assertIsNotNone(result)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    @patch('signals.signals_random_forest.SMOTE')
    def test_train_random_forest_model_smote_error(self, mock_smote_class, mock_memory):
        """Test model training when SMOTE fails"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        # Mock SMOTE to raise an exception
        mock_smote = MagicMock()
        mock_smote.fit_resample.side_effect = Exception("SMOTE failed")
        mock_smote_class.return_value = mock_smote
        
        with patch('signals.signals_random_forest.joblib.dump'):
            result = train_random_forest_model(self.training_df, save_model=False)
            
            # Should handle SMOTE failure and continue with original data
            self.assertIsNotNone(result)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    @patch('signals.signals_random_forest.compute_class_weight')
    def test_train_random_forest_model_class_weight_error(self, mock_weight, mock_memory):
        """Test model training when class weight computation fails"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        mock_weight.side_effect = Exception("Class weight computation failed")
        
        with patch('signals.signals_random_forest.joblib.dump'):
            result = train_random_forest_model(self.training_df, save_model=False)
            
            # Should handle class weight error and continue
            self.assertIsNotNone(result)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_train_random_forest_model_save_error(self, mock_memory):
        """Test model training when saving fails"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        with patch('signals.signals_random_forest.joblib.dump') as mock_dump:
            mock_dump.side_effect = Exception("Save failed")
            
            result = train_random_forest_model(self.training_df, save_model=True)
            
            # Should return model even if saving fails
            self.assertIsNotNone(result)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_success(self, mock_memory):
        """Test successful signal generation"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        # Create and train a model
        with patch('signals.signals_random_forest.joblib.dump'):
            model = train_random_forest_model(self.training_df, save_model=False)
        
        if model is not None:
            signal = get_latest_random_forest_signal(self.sample_df, model)
            self.assertIn(signal, [SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL])

    def test_get_latest_random_forest_signal_empty_input(self):
        """Test signal generation with empty DataFrame"""
        mock_model = MagicMock()
        signal = get_latest_random_forest_signal(self.empty_df, mock_model)
        self.assertEqual(signal, SIGNAL_NEUTRAL)

    def test_get_latest_random_forest_signal_missing_columns(self):
        """Test signal generation with missing OHLC columns"""
        mock_model = MagicMock()
        signal = get_latest_random_forest_signal(self.incomplete_df, mock_model)
        self.assertEqual(signal, SIGNAL_NEUTRAL)

    def test_get_latest_random_forest_signal_features_empty(self):
        """Test signal generation when feature calculation returns empty DataFrame"""
        mock_model = MagicMock()
        
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_calc.return_value = pd.DataFrame()  # Empty DataFrame
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_NEUTRAL)

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_features_with_nan(self):
        """Test signal generation when features contain NaN values"""
        mock_model = MagicMock()
        
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_features_df = self.sample_df.copy()
            # Add features but with NaN values
            for feature in ['rsi', 'macd', 'macd_signal', COL_BB_UPPER, COL_BB_LOWER, 'ma_20']:
                mock_features_df[feature] = np.nan
            mock_calc.return_value = mock_features_df
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_NEUTRAL)

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_low_confidence(self):
        """Test signal generation with low confidence prediction"""
        # Create a mock model that returns low confidence
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.3, 0.3]])  # Low confidence
        mock_model.classes_ = np.array([-1, 0, 1])
        
        # Mock the feature calculation to return valid data
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_features_df = self.sample_df.copy()
            # Add required features
            for feature in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20']:
                mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_calc.return_value = mock_features_df
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_NEUTRAL)

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_high_confidence_long(self):
        """Test signal generation with high confidence LONG prediction"""
        # Create a mock model that returns high confidence for LONG
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # High confidence for class 1
        mock_model.classes_ = np.array([-1, 0, 1])
        
        # Mock the feature calculation
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_features_df = self.sample_df.copy()
            for feature in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20']:
                mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_calc.return_value = mock_features_df
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_LONG)

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_high_confidence_short(self):
        """Test signal generation with high confidence SHORT prediction"""
        # Create a mock model that returns high confidence for SHORT
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])  # High confidence for class -1
        mock_model.classes_ = np.array([-1, 0, 1])
        
        # Mock the feature calculation
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_features_df = self.sample_df.copy()
            for feature in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20']:
                mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_calc.return_value = mock_features_df
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_SHORT)

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_high_confidence_neutral(self):
        """Test signal generation with high confidence NEUTRAL prediction"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])  # High confidence for class 0
        mock_model.classes_ = np.array([-1, 0, 1])
        
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_features_df = self.sample_df.copy()
            for feature in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20']:
                mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_calc.return_value = mock_features_df
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_NEUTRAL)

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_unknown_class(self):
        """Test signal generation with unknown predicted class"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # High confidence
        mock_model.classes_ = np.array([2, 3, 4])  # Unknown classes
        
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_features_df = self.sample_df.copy()
            for feature in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20']:
                mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_calc.return_value = mock_features_df
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_NEUTRAL)

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_get_latest_random_forest_signal_prediction_error(self):
        """Test signal generation when prediction raises an exception"""
        # Create a mock model that raises an exception
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = Exception("Prediction error")
        
        # Mock the feature calculation
        with patch('signals.signals_random_forest._calculate_features') as mock_calc:
            mock_features_df = self.sample_df.copy()
            for feature in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20']:
                mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_calc.return_value = mock_features_df
            
            signal = get_latest_random_forest_signal(self.sample_df, mock_model)
            self.assertEqual(signal, SIGNAL_NEUTRAL)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.joblib.dump')
    def test_train_and_save_global_rf_model_success(self, mock_dump, mock_memory):
        """Test successful global model training and saving"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        model, model_path = train_and_save_global_rf_model(self.training_df)
        
        if model is not None:
            self.assertIsNotNone(model)
            self.assertIsInstance(model_path, str)
            self.assertTrue(mock_dump.called)

    def test_train_and_save_global_rf_model_empty_input(self):
        """Test global model training with empty DataFrame"""
        model, model_path = train_and_save_global_rf_model(self.empty_df)
        self.assertIsNone(model)
        self.assertEqual(model_path, "")

    @patch('signals.signals_random_forest.MODEL_FEATURES', ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20'])
    def test_train_and_save_global_rf_model_custom_filename(self):
        """Test global model training with custom filename"""
        custom_filename = "custom_rf_model.joblib"
        
        with patch('signals.signals_random_forest.psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 4 * 1024**3  # 4GB
            with patch('signals.signals_random_forest.joblib.dump') as mock_dump:
                model, model_path = train_and_save_global_rf_model(self.training_df, custom_filename)
                
                if model is not None:
                    self.assertIn(custom_filename, model_path)

    @patch('signals.signals_random_forest.psutil.virtual_memory')
    @patch('signals.signals_random_forest.joblib.dump')
    def test_train_and_save_global_rf_model_save_error(self, mock_dump, mock_memory):
        """Test global model training when saving fails"""
        # Mock sufficient memory
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        mock_dump.side_effect = Exception("Save failed")
        
        model, model_path = train_and_save_global_rf_model(self.training_df)
        self.assertIsNone(model)
        self.assertEqual(model_path, "")

    def test_calculate_features_with_nan_values(self):
        """Test feature calculation with NaN values in input"""
        df_with_nan = self.sample_df.copy()
        df_with_nan.loc[10:15, COL_CLOSE] = np.nan
        
        result = _calculate_features(df_with_nan)
        
        # Should handle NaN values and still return a DataFrame
        # The exact behavior depends on how pandas_ta handles NaN values
        self.assertIsInstance(result, pd.DataFrame)

    def test_calculate_features_with_constant_prices(self):
        """Test feature calculation with constant prices"""
        constant_df = pd.DataFrame({
            COL_OPEN: [100] * 50,
            COL_HIGH: [100] * 50,
            COL_LOW: [100] * 50,
            COL_CLOSE: [100] * 50
        })
        
        result = _calculate_features(constant_df)
        
        # Should handle constant prices without crashing
        self.assertIsInstance(result, pd.DataFrame)

    def tearDown(self):
        """Clean up after each test"""
        # Clean up any temporary files or resources if needed
        pass

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
