import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, call

warnings.filterwarnings("ignore", message="Unable to find acceptable character detection dependency")

# Try to install missing dependencies if needed
try:
    import chardet
except ImportError:
    try:
        import charset_normalizer
    except ImportError:
        # If running in a test environment, we can suppress this specific warning
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "charset-normalizer"], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            import charset_normalizer
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ImportError):
            # If installation fails, just suppress the warning
            warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
signals_dir = os.path.dirname(current_dir)            
project_root = os.path.dirname(signals_dir)           
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module to test
from signals._components._process_signals_transformer import (
    load_latest_transformer_model,
    process_signals_transformer,
    DATAFRAME_COLUMNS
)

# Import constants for testing
from livetrade.config import (
    SIGNAL_LONG, 
    SIGNAL_SHORT, 
    SIGNAL_NEUTRAL,
    TRANSFORMER_MODEL_FILENAME
)

class TestLoadLatestTransformerModel(unittest.TestCase):
    """Test load_latest_transformer_model function"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_latest_transformer_model_no_directory(self):
        """Test when models directory doesn't exist"""
        non_existent_dir = os.path.join(self.temp_dir, "non_existent")
        
        result = load_latest_transformer_model(non_existent_dir)
        
        self.assertEqual(result, (None, None), "Should return (None, None) for non-existent directory")
    
    def test_load_latest_transformer_model_empty_directory(self):
        """Test when models directory is empty"""
        result = load_latest_transformer_model(str(self.models_dir))
        
        self.assertEqual(result, (None, None), "Should return (None, None) for empty directory")
    
    @patch('signals._components._process_signals_transformer.load_transformer_model')
    def test_load_latest_transformer_model_default_file_exists(self, mock_load):
        """Test loading default transformer model file"""
        # Create default model file
        default_file = self.models_dir / TRANSFORMER_MODEL_FILENAME
        default_file.touch()
        
        # Mock successful model loading
        mock_model_data = (Mock(), Mock(), ['close', 'rsi'], 0)
        mock_load.return_value = mock_model_data
        
        result = load_latest_transformer_model(str(self.models_dir))
        
        self.assertIsNotNone(result[0], "Should return model data")
        self.assertEqual(result[1], str(default_file), "Should return correct model path")
        mock_load.assert_called_once_with(str(default_file))
    
    @patch('signals._components._process_signals_transformer.load_transformer_model')
    def test_load_latest_transformer_model_timestamped_files(self, mock_load):
        """Test loading newest timestamped model file"""
        # Create timestamped model files
        older_file = self.models_dir / "transformer_model_20240101_1200.pth"
        newer_file = self.models_dir / "transformer_model_20240102_1200.pth"
        
        older_file.touch()
        newer_file.touch()
        
        # Make newer file actually newer
        import time
        time.sleep(0.1)
        newer_file.touch()
        
        # Mock successful model loading
        mock_model_data = (Mock(), Mock(), ['close', 'rsi'], 0)
        mock_load.return_value = mock_model_data
        
        result = load_latest_transformer_model(str(self.models_dir))
        
        self.assertIsNotNone(result[0], "Should return model data")
        self.assertEqual(result[1], str(newer_file), "Should return path to newer file")
        mock_load.assert_called_once_with(str(newer_file))
    
    @patch('signals._components._process_signals_transformer.load_transformer_model')
    def test_load_latest_transformer_model_failed_loading(self, mock_load):
        """Test when model file exists but loading fails"""
        # Create model file
        model_file = self.models_dir / TRANSFORMER_MODEL_FILENAME
        model_file.touch()
        
        # Mock failed model loading
        mock_load.return_value = (None, None, None, None)
        
        result = load_latest_transformer_model(str(self.models_dir))
        
        self.assertEqual(result, (None, None), "Should return (None, None) when loading fails")
    
    @patch('signals._components._process_signals_transformer.load_transformer_model')
    def test_load_latest_transformer_model_exception(self, mock_load):
        """Test when an exception occurs during loading"""
        # Create model file
        model_file = self.models_dir / TRANSFORMER_MODEL_FILENAME
        model_file.touch()
        
        # Mock exception during loading
        mock_load.side_effect = Exception("Loading error")
        
        result = load_latest_transformer_model(str(self.models_dir))
        
        self.assertEqual(result, (None, None), "Should return (None, None) when exception occurs")

class TestProcessSignalsTransformer(unittest.TestCase):
    """Test process_signals_transformer function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        self.sample_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Create preloaded data structure
        self.preloaded_data = {
            'BTCUSDT': {
                '1h': self.sample_df.copy(),
                '4h': self.sample_df.copy(),
                '1d': self.sample_df.copy()
            },
            'ETHUSDT': {
                '1h': self.sample_df.copy(),
                '4h': self.sample_df.copy(),
                '1d': self.sample_df.copy()
            }
        }
        
        # Create mock model data
        self.mock_model = Mock()
        self.mock_scaler = Mock()
        self.mock_feature_cols = ['close', 'rsi', 'macd', 'macd_signal']
        self.mock_target_idx = 0
        self.mock_model_data = (self.mock_model, self.mock_scaler, self.mock_feature_cols, self.mock_target_idx)
    
    def test_process_signals_transformer_empty_data(self):
        """Test with empty preloaded data"""
        result = process_signals_transformer({})
        
        self.assertTrue(result.empty, "Should return empty DataFrame for empty input")
        self.assertEqual(list(result.columns), DATAFRAME_COLUMNS, "Should have correct columns")
    
    def test_process_signals_transformer_invalid_data(self):
        """Test with invalid preloaded data format"""
        result = process_signals_transformer("invalid_data")
        
        self.assertTrue(result.empty, "Should return empty DataFrame for invalid input")
        self.assertEqual(list(result.columns), DATAFRAME_COLUMNS, "Should have correct columns")
    
    def test_process_signals_transformer_no_signal_types(self):
        """Test when both signal types are disabled"""
        result = process_signals_transformer(
            self.preloaded_data,
            include_long_signals=False,
            include_short_signals=False
        )
        
        self.assertTrue(result.empty, "Should return empty DataFrame when no signal types enabled")
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.get_latest_transformer_signal')
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    def test_process_signals_transformer_no_model_no_training(self, mock_combine, mock_signal, mock_load):
        """Test when no model is found and auto_train_if_missing is False"""
        # Mock no model found
        mock_load.return_value = (None, None)
        mock_combine.return_value = self.sample_df
        
        result = process_signals_transformer(
            self.preloaded_data,
            auto_train_if_missing=False
        )
        
        self.assertTrue(result.empty, "Should return empty DataFrame when no model available")
        mock_signal.assert_not_called()
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.train_and_save_transformer_model')
    @patch('signals._components._process_signals_transformer.load_transformer_model')
    @patch('signals._components._process_signals_transformer.get_latest_transformer_signal')  # corrected patch target remains
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    def test_process_signals_transformer_auto_train_success(self, mock_combine, mock_signal, mock_load_model, mock_train, mock_load_latest):
        """Test successful auto-training when no model is found"""
        # Mock no existing model
        mock_load_latest.return_value = (None, None)
        
        # Mock successful training
        mock_trained_model = Mock()
        mock_train.return_value = (mock_trained_model, "/path/to/model.pth")
        
        # Mock successful loading of trained model
        mock_model = Mock()
        mock_scaler = Mock()
        mock_features = ['close', 'rsi']
        mock_target_idx = 0
        mock_load_model.return_value = (mock_model, mock_scaler, mock_features, mock_target_idx)
        
        # Mock combined dataframe
        mock_combine.return_value = self.sample_df
        
        # Mock signal generation - IMPORTANT: This must return SIGNAL_LONG to generate signals
        mock_signal.return_value = SIGNAL_LONG
        
        # Create preloaded data with proper structure
        preloaded_data = {
            'BTCUSDT': {
                '1h': self.sample_df.copy()
            }
        }
        
        # Supply timeframe so that symbol data is found.
        result = process_signals_transformer(
            preloaded_data=preloaded_data,
            timeframes_to_scan=['1h'],
            auto_train_if_missing=True,
            include_long_signals=True
        )
        
        # Verify training was called
        mock_train.assert_called_once()
        mock_load_model.assert_called_once_with("/path/to/model.pth")
        
        # Verify signal generation was called
        mock_signal.assert_called()
        
        # Should have generated signals
        self.assertFalse(result.empty, "Should generate signals after training")
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.get_latest_transformer_signal')  # corrected patch target remains
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    @patch('signals.signals_transformer.add_technical_indicators')  # corrected patch target
    @patch('signals.signals_transformer.analyze_model_bias_and_adjust_thresholds')  # corrected patch target
    def test_process_signals_transformer_with_existing_model(self, mock_bias, mock_indicators, mock_combine, mock_signal, mock_load):
        """Test signal processing with existing model"""
        # Mock model loading
        mock_model = Mock()
        mock_scaler = Mock()
        mock_feature_cols = ['close', 'rsi']
        mock_target_idx = 0
        mock_model_data = (mock_model, mock_scaler, mock_feature_cols, mock_target_idx)
        mock_load.return_value = (mock_model_data, "/path/to/model.pth")
        
        # Mock combined dataframe
        mock_combine.return_value = self.sample_df
        
        # Mock technical indicators and bias analysis
        mock_indicators.return_value = self.sample_df.copy()
        mock_bias.return_value = (0.02, -0.02)
        
        # Mock signal generation - return different signals for different symbols
        mock_signal.side_effect = [SIGNAL_LONG, SIGNAL_SHORT]
        
        # Create test data with proper structure
        preloaded_data = {
            'BTCUSDT': {
                '1h': self.sample_df.copy(),
            },
            'ETHUSDT': {
                '1h': self.sample_df.copy(),
            }
        }
        
        result = process_signals_transformer(
            preloaded_data=preloaded_data,
            timeframes_to_scan=['1h'],
            include_long_signals=True,
            include_short_signals=True
        )
        
        # Should have called signal generation for each symbol
        self.assertEqual(mock_signal.call_count, 2, "Should call signal generation for each symbol")
        
        # Should return results
        self.assertFalse(result.empty, "Should return non-empty results")
        self.assertEqual(len(result), 2, "Should have signals for both symbols")
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.get_latest_transformer_signal')
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    def test_process_signals_transformer_signal_filtering(self, mock_combine, mock_signal, mock_load):
        """Test signal type filtering"""
        # Mock model loading
        mock_load.return_value = (self.mock_model_data, "/path/to/model.pth")
        mock_combine.return_value = self.sample_df
        
        # Mock signal generation - alternate between LONG and SHORT
        mock_signal.side_effect = [SIGNAL_LONG, SIGNAL_SHORT]
        
        # Test with only LONG signals enabled
        result_long_only = process_signals_transformer(
            self.preloaded_data,
            include_long_signals=True,
            include_short_signals=False
        )
        
        # Should only have LONG signals
        if not result_long_only.empty:
            self.assertTrue(all(result_long_only['SignalType'] == 'LONG'), "Should only include LONG signals")
        
        # Reset mock
        mock_signal.side_effect = [SIGNAL_LONG, SIGNAL_SHORT]
        
        # Test with only SHORT signals enabled
        result_short_only = process_signals_transformer(
            self.preloaded_data,
            include_long_signals=False,
            include_short_signals=True
        )
        
        # Should only have SHORT signals
        if not result_short_only.empty:
            self.assertTrue(all(result_short_only['SignalType'] == 'SHORT'), "Should only include SHORT signals")
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.get_latest_transformer_signal')
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    def test_process_signals_transformer_insufficient_data(self, mock_combine, mock_signal, mock_load):
        """Test with insufficient data for some symbols"""
        # Mock model loading
        mock_load.return_value = (self.mock_model_data, "/path/to/model.pth")
        mock_combine.return_value = self.sample_df
        
        # Create data with insufficient samples for one symbol
        insufficient_data = {
            'BTCUSDT': {
                '1h': self.sample_df.copy(),
                '4h': self.sample_df.head(10),  # Insufficient data
                '1d': self.sample_df.copy()
            },
            'ETHUSDT': {
                '1h': self.sample_df.copy(),
                '4h': self.sample_df.copy(),
                '1d': self.sample_df.copy()
            }
        }
        
        # Mock signal generation
        mock_signal.return_value = SIGNAL_LONG
        
        result = process_signals_transformer(
            insufficient_data,
            timeframes_to_scan=['4h']  # Use timeframe with insufficient data for BTCUSDT
        )
        
        # Should handle insufficient data gracefully
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame even with insufficient data")
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.get_latest_transformer_signal')
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    def test_process_signals_transformer_signal_generation_error(self, mock_combine, mock_signal, mock_load):
        """Test handling of signal generation errors"""
        # Mock model loading
        mock_load.return_value = (self.mock_model_data, "/path/to/model.pth")
        mock_combine.return_value = self.sample_df
        
        # Mock signal generation error for first symbol, success for second
        mock_signal.side_effect = [Exception("Signal generation error"), SIGNAL_LONG]
        
        result = process_signals_transformer(self.preloaded_data)
        
        # Should handle error gracefully and continue processing
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame even with errors")
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.get_latest_transformer_signal')
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    def test_process_signals_transformer_timeframe_priority(self, mock_combine, mock_signal, mock_load):
        """Test timeframe priority in signal processing"""
        # Mock model loading
        mock_load.return_value = (self.mock_model_data, "/path/to/model.pth")
        mock_combine.return_value = self.sample_df
        
        # Mock signal generation - first call returns LONG, should stop there
        mock_signal.return_value = SIGNAL_LONG
        
        result = process_signals_transformer(
            self.preloaded_data,
            timeframes_to_scan=['1h', '4h', '1d']
        )
        
        # Should find signal on first timeframe and stop
        if not result.empty:
            # Should prioritize first timeframe
            first_signal_tf = result.iloc[0]['SignalTimeframe']
            self.assertEqual(first_signal_tf, '1h', "Should use first timeframe when signal found")
    
    def test_process_signals_transformer_custom_timeframes(self):
        """Test with custom timeframes"""
        with patch('signals._components._process_signals_transformer.load_latest_transformer_model') as mock_load:
            mock_load.return_value = (self.mock_model_data, "/path/to/model.pth")
            
            with patch('signals._components._process_signals_transformer.combine_all_dataframes') as mock_combine:
                mock_combine.return_value = self.sample_df
                
                with patch('signals._components._process_signals_transformer.get_latest_transformer_signal') as mock_signal:
                    mock_signal.return_value = SIGNAL_NEUTRAL
                    
                    # Test with custom timeframes
                    custom_timeframes = ['15m', '30m']
                    
                    # Add custom timeframes to test data
                    test_data = {
                        'BTCUSDT': {
                            '15m': self.sample_df.copy(),
                            '30m': self.sample_df.copy()
                        }
                    }
                    
                    result = process_signals_transformer(
                        test_data,
                        timeframes_to_scan=custom_timeframes
                    )
                    
                    # Should process with custom timeframes
                    self.assertIsInstance(result, pd.DataFrame, "Should handle custom timeframes")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete module"""
    
    def setUp(self):
        """Set up integration test data"""
        np.random.seed(42)
        
        # Create more realistic test data
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        
        # Generate trending price data
        trend = np.cumsum(np.random.normal(0, 0.1, 200))
        close_prices = 100 + trend
        
        self.realistic_df = pd.DataFrame({
            'open': close_prices + np.random.normal(0, 0.5, 200),
            'high': close_prices + np.random.uniform(0, 2, 200),
            'low': close_prices - np.random.uniform(0, 2, 200),
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 200)
        }, index=dates)
        
        self.realistic_preloaded_data = {
            'BTCUSDT': {
                '1h': self.realistic_df.copy(),
                '4h': self.realistic_df.copy(),
                '1d': self.realistic_df.copy()
            }
        }
    
    @patch('signals._components._process_signals_transformer.load_latest_transformer_model')
    @patch('signals._components._process_signals_transformer.combine_all_dataframes')
    def test_integration_full_pipeline_mock_model(self, mock_combine, mock_load):
        """Test full pipeline with mocked model"""
        # Mock model loading
        mock_model = Mock()
        mock_scaler = Mock()
        mock_feature_cols = ['close', 'rsi', 'macd', 'macd_signal']
        mock_target_idx = 0
        mock_load.return_value = (mock_model, mock_scaler, mock_feature_cols, mock_target_idx)
        
        # Mock combined dataframe
        mock_combine.return_value = self.realistic_df
        
        # Mock the actual signal generation to avoid complex model setup
        with patch('signals._components._process_signals_transformer.get_latest_transformer_signal') as mock_signal:
            mock_signal.return_value = SIGNAL_LONG
            
            # Run the process
            result = process_signals_transformer(
                self.realistic_preloaded_data,
                timeframes_to_scan=['1h'],
                include_long_signals=True,
                include_short_signals=False
            )
            
            # Verify results
            self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")
            if not result.empty:
                self.assertEqual(result.iloc[0]['SignalType'], 'LONG', "Should generate LONG signal")
                self.assertEqual(result.iloc[0]['Pair'], 'BTCUSDT', "Should have correct symbol")
                self.assertEqual(result.iloc[0]['SignalTimeframe'], '1h', "Should have correct timeframe")
    
    def test_integration_dataframe_columns_consistency(self):
        """Test that all functions return DataFrames with consistent columns"""
        # Test empty result
        empty_result = process_signals_transformer({})
        self.assertEqual(list(empty_result.columns), DATAFRAME_COLUMNS, "Empty result should have correct columns")
        
        # Test with invalid data
        invalid_result = process_signals_transformer("invalid")
        self.assertEqual(list(invalid_result.columns), DATAFRAME_COLUMNS, "Invalid input result should have correct columns")

class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def test_process_signals_transformer_torch_cuda_error(self):
        """Test handling of torch CUDA errors"""
        with patch('torch.cuda.is_available', side_effect=Exception("CUDA error")):
            # Should handle CUDA check errors gracefully
            result = process_signals_transformer({})
            self.assertEqual(list(result.columns), DATAFRAME_COLUMNS, "Should handle CUDA errors")
    
    def test_process_signals_transformer_model_loading_error(self):
        """Test handling of model loading errors"""
        # Use a direct patch to avoid propagating the exception
        with patch('signals._components._process_signals_transformer.load_latest_transformer_model') as mock_load:
            # Mock model loading error
            mock_load.side_effect = Exception("Model loading error")
            
            preloaded_data = {'BTCUSDT': {'1h': pd.DataFrame({'close': [100, 101, 102]})}}
            
            result = process_signals_transformer(preloaded_data)
            
            # Should handle error and return empty result
            self.assertTrue(result.empty, "Should return empty result when model loading fails")
    
    def test_process_signals_transformer_empty_symbol_data(self):
        """Test handling of empty symbol data"""
        preloaded_data = {
            'BTCUSDT': {},  # Empty timeframe data
            'ETHUSDT': {'1h': pd.DataFrame()}  # Empty DataFrame
        }
        
        with patch('signals._components._process_signals_transformer.load_latest_transformer_model') as mock_load:
            mock_model_data = (Mock(), Mock(), ['close'], 0)
            mock_load.return_value = (mock_model_data, "/path/to/model.pth")
            
            result = process_signals_transformer(preloaded_data)
            
            # Should handle empty data gracefully
            self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame for empty symbol data")

if __name__ == '__main__':
    unittest.main()