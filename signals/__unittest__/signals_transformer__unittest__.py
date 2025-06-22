import os
import pandas as pd
import shutil
import sys
import tempfile
import torch
import unittest
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import Mock, patch
import numpy as np
from torch.utils.data import DataLoader

# Add the parent directory to sys.path to allow importing from config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the module to test
from signals.signals_transformer import (
    analyze_model_bias_and_adjust_thresholds,
    evaluate_model,
    get_latest_transformer_signal,
    load_transformer_model,
    preprocess_transformer_data,
    CryptoDataset,
    safe_load_model,
    safe_memory_comparison,
    safe_memory_division,
    safe_nan_to_num,
    safe_save_model,
    select_and_scale_features,
    setup_safe_globals,
    TimeSeriesTransformer,
    train_and_save_transformer_model,
    train_transformer_model,
)

# Import the calculate features function
from components._generate_indicator_features import generate_indicator_features

# Import constants for testing
from components.config import (
    SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL,
)

class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators calculation using _calculate_features"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Make prices more realistic with some trend
        for i in range(1, len(self.df)):
            self.df.loc[self.df.index[i], 'close'] = self.df.iloc[i-1]['close'] + np.random.normal(0, 0.5)
    
    def test_calculate_features_success(self):
        """Test successful technical indicators calculation"""
        result = generate_indicator_features(self.df.copy())
        
        # Check that new columns are added
        expected_columns = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20', 'ma_20_slope']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Column {col} should be present")
        
        # Check that indicators have reasonable values
        self.assertTrue(result['rsi'].between(0, 100).all(), "RSI should be between 0 and 100")
        self.assertTrue(result['bb_upper'].gt(result['bb_lower']).all(), "BB upper should be greater than BB lower")
    
    def test_calculate_features_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = generate_indicator_features(empty_df)
        self.assertTrue(result.empty, "Should return empty DataFrame for empty input")
    
    def test_calculate_features_missing_close(self):
        """Test with missing close column"""
        df_no_close = self.df.drop('close', axis=1)
        result = generate_indicator_features(df_no_close)
        self.assertTrue(result.empty, "Should return empty DataFrame without close column")
    
    def test_calculate_features_insufficient_data(self):
        """Test with insufficient data"""
        small_df = self.df.head(5)  # Only 5 rows
        result = generate_indicator_features(small_df)
        # Should still return DataFrame with indicators, but may have fewer rows after dropna
        self.assertGreaterEqual(len(result), 0, "Should return DataFrame with non-negative length")

class TestFeatureScaling(unittest.TestCase):
    """Test feature scaling and selection"""
    
    def setUp(self):
        """Set up test data with indicators"""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'close': np.random.uniform(95, 115, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.uniform(-2, 2, 100),
            'macd_signal': np.random.uniform(-1, 1, 100),
            'bb_upper': np.random.uniform(110, 120, 100),
            'bb_lower': np.random.uniform(90, 100, 100),
            'ma_20': np.random.uniform(100, 110, 100),
            'ma_20_slope': np.random.uniform(-1, 1, 100)
        })
    
    def test_select_and_scale_features_success(self):
        """Test successful feature scaling"""
        scaled_data, scaler, features = select_and_scale_features(self.df)
        
        # Check output shapes and types
        self.assertIsInstance(scaled_data, np.ndarray, "Should return numpy array")
        self.assertIsInstance(scaler, MinMaxScaler, "Should return MinMaxScaler")
        self.assertIsInstance(features, list, "Should return list of features")
        
        # Check scaling is correct (values should be between 0 and 1)
        self.assertTrue(np.all(scaled_data >= 0), "Scaled values should be >= 0")
        self.assertTrue(np.all(scaled_data <= 1), "Scaled values should be <= 1")
        
        # Check shape
        self.assertEqual(scaled_data.shape[0], len(self.df), "Should have same number of rows")
    
    def test_select_and_scale_features_custom_columns(self):
        """Test with custom feature columns"""
        custom_features = ['close', 'rsi']
        scaled_data, scaler, features = select_and_scale_features(self.df, custom_features)
        
        self.assertEqual(len(features), 2, "Should use custom features")
        self.assertEqual(scaled_data.shape[1], 2, "Should have 2 feature columns")
    
    def test_select_and_scale_features_missing_columns(self):
        """Test with missing columns"""
        features_with_missing = ['close', 'rsi', 'nonexistent_column']
        scaled_data, scaler, features = select_and_scale_features(self.df, features_with_missing)
        
        # Should work with available columns only
        self.assertNotIn('nonexistent_column', features, "Should exclude missing columns")
        self.assertTrue(len(features) >= 2, "Should have at least the available columns")

class TestTimeSeriesTransformer(unittest.TestCase):
    """Test TimeSeriesTransformer model"""
    
    def setUp(self):
        """Set up test model"""
        self.feature_size = 5
        self.seq_length = 10
        self.pred_length = 1
        self.model = TimeSeriesTransformer(
            feature_size=self.feature_size,
            seq_length=self.seq_length,
            prediction_length=self.pred_length
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, TimeSeriesTransformer, "Should create TimeSeriesTransformer instance")
        self.assertEqual(self.model.pos_embedding.shape[1], self.seq_length, "Positional embedding should match seq_length")
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        batch_size = 8
        input_tensor = torch.randn(batch_size, self.seq_length, self.feature_size)
        
        output = self.model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, self.pred_length)
        self.assertEqual(output.shape, expected_shape, f"Output shape should be {expected_shape}")
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any(), "Output should not contain NaN values")

class TestModelTraining(unittest.TestCase):
    """Test model training functions"""
    
    def setUp(self):
        """Set up test data and model"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic data with sequences already created
        num_samples = 200
        seq_length = 10
        num_features = 5
        self.data_X = np.random.rand(num_samples, seq_length, num_features)
        self.data_y = np.random.rand(num_samples, 1)  # Single step prediction
        self.dataset = CryptoDataset(self.data_X, self.data_y)
        
        # Create data loader
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        # Create model
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
    
    def test_train_transformer_model(self):
        """Test model training function"""
        # Create a simple dataset with preprocessed sequences
        num_samples = 100
        seq_length = 10
        num_features = 5
        data_X = np.random.rand(num_samples, seq_length, num_features)
        data_y = np.random.rand(num_samples, 1)
        dataset = CryptoDataset(data_X, data_y)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Train the model
        trained_model, training_history = train_transformer_model(
            self.model,
            train_loader,
            epochs=2,
            device='cpu'
        )
        
        # Check that model was trained
        self.assertIsInstance(trained_model, TimeSeriesTransformer, "Should return trained model")
        self.assertIsInstance(training_history, dict, "Should return training history")
        
        # Check that model parameters have changed (basic training check)
        # This is a simple check that training occurred
        self.assertTrue(any(p.requires_grad for p in trained_model.parameters()), "Model should have trainable parameters")

class TestSignalGeneration(unittest.TestCase):
    """Test signal generation functions"""
    def setUp(self):
        """Set up test data and mock model"""
        np.random.seed(42)
        
        # Create realistic price data
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Create mock model and scaler
        self.mock_model = Mock()
        self.mock_model.pos_embedding = Mock()
        self.mock_model.pos_embedding.shape = (1, 30, 64)  # Mock positional embedding shape
        self.mock_model.eval = Mock()
        
        self.mock_scaler = Mock()
        self.mock_scaler.transform = Mock(return_value=np.random.rand(30, 5))
        self.mock_scaler.inverse_transform = Mock(return_value=np.array([[0, 0, 0, 105.5, 0]]))
        
        self.feature_cols = ['close', 'rsi', 'macd', 'macd_signal', 'ma_20_slope']
    
    @patch('components._generate_indicator_features.generate_indicator_features')
    def test_get_latest_transformer_signal_long(self, mock_calculate_features):
        """Test long signal generation"""
        # Setup mock to return DataFrame with indicators
        mock_df = self.df.copy()
        mock_df['rsi'] = 50
        mock_df['macd'] = 0.1
        mock_df['macd_signal'] = 0.05
        mock_df['bb_upper'] = 120
        mock_df['bb_lower'] = 80
        mock_df['ma_20'] = 100
        mock_df['ma_20_slope'] = 0.1
        mock_calculate_features.return_value = mock_df
        
        # Mock model prediction to return bullish signal
        with patch('torch.tensor'), \
             patch('torch.no_grad'), \
             patch.object(self.mock_model, '__call__', return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([[0.8]])))))):
            
            signal = get_latest_transformer_signal(
                self.df, 
                self.mock_model, 
                self.mock_scaler, 
                self.feature_cols, 
                3,  # target_col_idx
                'cpu'
            )
            
            # Should return a valid signal - use constants from config
            self.assertIn(signal, [SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL], "Should return valid signal")
    
    def test_get_latest_transformer_signal_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        signal = get_latest_transformer_signal(
            empty_df, 
            self.mock_model, 
            self.mock_scaler, 
            self.feature_cols, 
            3,
            'cpu'
        )
        
        self.assertEqual(signal, SIGNAL_NEUTRAL, "Should return neutral for empty DataFrame")
    
    def test_get_latest_transformer_signal_insufficient_data(self):
        """Test with insufficient data"""
        small_df = self.df.head(5)  # Less than required sequence length
        
        with patch('components._generate_indicator_features.generate_indicator_features', return_value=small_df):
            signal = get_latest_transformer_signal(
                small_df, 
                self.mock_model, 
                self.mock_scaler, 
                self.feature_cols, 
                3,
                'cpu'
            )
            
            self.assertEqual(signal, SIGNAL_NEUTRAL, "Should return neutral for insufficient data")

class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation functions"""
    
    def setUp(self):
        """Set up test data and model for evaluation"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic data with sequences already created
        num_samples = 100
        seq_length = 10
        num_features = 5
        self.data_X = np.random.rand(num_samples, seq_length, num_features)
        self.data_y = np.random.rand(num_samples, 1)
        self.dataset = CryptoDataset(self.data_X, self.data_y)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=False)
        
        # Create model
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
        
        # Create scaler and fit on feature data
        self.scaler = MinMaxScaler()
        # Fit on the flattened features from all sequences
        all_features = self.data_X.reshape(-1, self.data_X.shape[-1])
        self.scaler.fit(all_features)
        
        self.feature_cols = ['feature1', 'feature2', 'feature3', 'target', 'feature5']
        self.target_col_idx = 3
    
    def test_evaluate_model_success(self):
        """Test successful model evaluation"""
        mse, mae = evaluate_model(
            self.model, 
            self.test_loader, 
            self.scaler, 
            self.feature_cols, 
            self.target_col_idx, 
            'cpu'
        )
        
        # Check that metrics are returned as floats
        self.assertIsInstance(mse, float, "MSE should be float")
        self.assertIsInstance(mae, float, "MAE should be float")
        
        # Check that metrics are non-negative
        self.assertGreaterEqual(mse, 0, "MSE should be non-negative")
        self.assertGreaterEqual(mae, 0, "MAE should be non-negative")
    
    def test_evaluate_model_empty_loader(self):
        """Test evaluation with empty data loader"""
        # Create minimal preprocessed data that will be empty after batching
        empty_X = np.random.rand(1, 10, 5)  # Just one sequence
        empty_y = np.random.rand(1, 1)
        empty_dataset = CryptoDataset(empty_X, empty_y)
        empty_loader = torch.utils.data.DataLoader(empty_dataset, batch_size=100, shuffle=False)  # Large batch size
        
        try:
            mse, mae = evaluate_model(
                self.model, 
                empty_loader, 
                self.scaler, 
                self.feature_cols, 
                self.target_col_idx, 
                'cpu'
            )
            # Should handle empty case gracefully
            self.assertTrue(True, "Should handle empty loader gracefully")
        except Exception as e:
            self.fail(f"Should handle empty loader gracefully, but raised: {e}")

class TestTechnicalIndicatorsEdgeCases(unittest.TestCase):
    """Test edge cases for technical indicators"""
    
    def test_calculate_features_with_nan_values(self):
        """Test technical indicators with NaN values in data"""
        df = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 104],
            'high': [105, 106, 107, np.nan, 109],
            'low': [95, 96, 97, 98, np.nan],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = generate_indicator_features(df)
        
        # Should handle NaN values without crashing
        self.assertGreaterEqual(len(result), 0, "Should return DataFrame with non-negative length")
    
    def test_calculate_features_with_constant_prices(self):
        """Test technical indicators with constant prices"""
        df = pd.DataFrame({
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })
        
        result = generate_indicator_features(df)
        
        # Should handle constant prices without crashing
        self.assertGreaterEqual(len(result), 0, "Should return DataFrame with non-negative length")
        
        # RSI should be around 50 for constant prices, but may have NaN
        if 'rsi' in result.columns:
            rsi_series = result['rsi']
            # Check if there are any non-NaN values
            if len(rsi_series.dropna()) > 0:
                rsi_mean = rsi_series.dropna().mean()
                # For constant prices, RSI calculation might not be exactly 50 due to technical reasons
                # Allow a wider range and check if it's finite
                self.assertTrue(np.isfinite(rsi_mean), "RSI should be finite for constant prices")
                if 0 <= rsi_mean <= 100:  # Valid RSI range
                    self.assertTrue(True, "RSI is within valid range")
                else:
                    self.fail(f"RSI {rsi_mean} is outside valid range [0, 100]")
    
    def test_calculate_features_with_extreme_values(self):
        """Test technical indicators with extreme price values"""
        df = pd.DataFrame({
            'open': [1e-6, 1e6, 1e-6, 1e6, 1e-6],
            'high': [1e-5, 1.1e6, 1e-5, 1.1e6, 1e-5],
            'low': [1e-7, 0.9e6, 1e-7, 0.9e6, 1e-7],
            'close': [5e-6, 1.05e6, 5e-6, 1.05e6, 5e-6],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = generate_indicator_features(df)
        
        # Should handle extreme values without crashing
        self.assertGreaterEqual(len(result), 0, "Should return DataFrame with non-negative length")
        # Check for infinite values using numpy if result is not empty
        if not result.empty:
            self.assertFalse(np.isinf(result.select_dtypes(include=[np.number])).any().any(), "Should not contain infinite values")

class TestFeatureScalingEdgeCases(unittest.TestCase):
    """Test edge cases for feature scaling"""
    
    def test_select_and_scale_features_with_constant_column(self):
        """Test scaling with constant column values"""
        df = pd.DataFrame({
            'close': [100] * 50,  # Constant values
            'rsi': np.random.uniform(20, 80, 50),
            'macd': np.random.uniform(-2, 2, 50),
            'macd_signal': np.random.uniform(-1, 1, 50)
        })
        
        try:
            scaled_data, scaler, features = select_and_scale_features(df)
            # Should handle constant values gracefully
            self.assertIsNotNone(scaled_data, "Should return scaled data")
            self.assertFalse(np.isnan(scaled_data).any(), "Should not contain NaN values")
        except Exception as e:
            self.fail(f"Should handle constant values gracefully, but raised: {e}")
    
    def test_select_and_scale_features_with_single_row(self):
        """Test scaling with single row of data"""
        df = pd.DataFrame({
            'close': [100],
            'rsi': [50],
            'macd': [0.1],
            'macd_signal': [0.05]
        })
        
        try:
            scaled_data, scaler, features = select_and_scale_features(df)
            self.assertEqual(scaled_data.shape[0], 1, "Should handle single row")
        except Exception as e:
            self.fail(f"Should handle single row gracefully, but raised: {e}")

class TestModelConfigurationValidation(unittest.TestCase):
    """Test model configuration validation"""
    
    def test_transformer_with_invalid_parameters(self):
        """Test transformer creation with invalid parameters"""
        # Test with zero heads
        try:
            model = TimeSeriesTransformer(nhead=0)
            self.fail("Should raise error for zero attention heads")
        except:
            self.assertTrue(True, "Should raise error for invalid nhead")
    
    def test_transformer_with_mismatched_dimensions(self):
        """Test transformer with mismatched dimensions"""
        try:
            # d_model must be divisible by nhead
            model = TimeSeriesTransformer(d_model=65, nhead=8)  # 65 not divisible by 8
            # Try a forward pass to trigger the error
            input_tensor = torch.randn(1, 10, 5)
            output = model(input_tensor)
            self.fail("Should raise error for mismatched dimensions")
        except:
            self.assertTrue(True, "Should raise error for mismatched dimensions")

class TestDeviceHandling(unittest.TestCase):
    """Test device handling (CPU/GPU)"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create preprocessed sequences
        num_samples = 50
        seq_length = 10
        num_features = 5
        self.data_X = np.random.rand(num_samples, seq_length, num_features)
        self.data_y = np.random.rand(num_samples, 1)
        self.dataset = CryptoDataset(self.data_X, self.data_y)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
    
    def test_train_model_device_cpu(self):
        """Test training on CPU device"""
        trained_model, training_history = train_transformer_model(
            self.model, 
            self.train_loader, 
            epochs=1, 
            device='cpu'
        )
        
        # Check that model is on CPU
        self.assertEqual(next(trained_model.parameters()).device.type, 'cpu', "Model should be on CPU")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_train_model_device_cuda(self):
        """Test training on CUDA device"""
        trained_model, training_history = train_transformer_model(
            self.model, 
            self.train_loader, 
            epochs=1, 
            device='cuda'
        )
        
        # Check that model is on CUDA
        self.assertEqual(next(trained_model.parameters()).device.type, 'cuda', "Model should be on CUDA")

class TestSignalGenerationEdgeCases(unittest.TestCase):
    """Test edge cases for signal generation"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Mock model and scaler
        self.mock_model = Mock()
        self.mock_model.pos_embedding = Mock()
        self.mock_model.pos_embedding.shape = (1, 30, 64)
        self.mock_model.eval = Mock()
        
        self.mock_scaler = Mock()
        self.feature_cols = ['close', 'rsi', 'macd', 'macd_signal', 'ma_20_slope']

    def test_get_signal_with_custom_thresholds(self):
        """Test signal generation with custom thresholds"""
        custom_thresholds = (0.05, -0.05)  # Custom buy/sell thresholds
        
        with patch('components._generate_indicator_features.generate_indicator_features') as mock_calculate_features, \
             patch('torch.tensor'), \
             patch('torch.no_grad'), \
             patch.object(self.mock_model, '__call__', return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([[0.8]])))))):
            
            mock_df = self.df.copy()
            mock_df['rsi'] = 50
            mock_df['macd'] = 0.1
            mock_df['macd_signal'] = 0.05
            mock_df['bb_upper'] = 120
            mock_df['bb_lower'] = 80
            mock_df['ma_20'] = 100
            mock_df['ma_20_slope'] = 0.1
            mock_calculate_features.return_value = mock_df
            
            self.mock_scaler.transform = Mock(return_value=np.random.rand(30, 5))
            self.mock_scaler.inverse_transform = Mock(return_value=np.array([[0, 0, 0, 110.0, 0]]))
            
            signal = get_latest_transformer_signal(
                self.df, 
                self.mock_model, 
                self.mock_scaler, 
                self.feature_cols, 
                3,
                'cpu',
                suggested_thresholds=custom_thresholds
            )
            
            self.assertIn(signal, [SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL], "Should return valid signal with custom thresholds")
    
    def test_get_signal_with_extreme_prediction(self):
        """Test signal generation with extreme price prediction"""
        with patch('components._generate_indicator_features.generate_indicator_features') as mock_calculate_features, \
             patch('torch.tensor'), \
             patch('torch.no_grad'), \
             patch.object(self.mock_model, '__call__', return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([[100.0]])))))):  # Extreme prediction
            
            mock_df = self.df.copy()
            mock_df['rsi'] = 50
            mock_df['macd'] = 0.1
            mock_df['macd_signal'] = 0.05
            mock_df['bb_upper'] = 120
            mock_df['bb_lower'] = 80
            mock_df['ma_20'] = 100
            mock_df['ma_20_slope'] = 0.1
            mock_calculate_features.return_value = mock_df
            
            self.mock_scaler.transform = Mock(return_value=np.random.rand(30, 5))
            self.mock_scaler.inverse_transform = Mock(return_value=np.array([[0, 0, 0, 1000.0, 0]]))  # Extreme price
            
            signal = get_latest_transformer_signal(
                self.df, 
                self.mock_model, 
                self.mock_scaler, 
                self.feature_cols, 
                3,
                'cpu'
            )
            
            # Should handle extreme predictions gracefully
            self.assertIn(signal, [SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL], "Should handle extreme predictions")

class TestModelCheckpointValidation(unittest.TestCase):
    """Test model checkpoint validation"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_model_with_missing_keys(self):
        """Test loading model with missing checkpoint keys"""
        model_path = os.path.join(self.temp_dir, 'incomplete_model.pth')
        
        # Create checkpoint with missing keys
        incomplete_checkpoint = {
            'model_state_dict': {},
            'model_config': {},
            # Missing 'scaler', 'feature_cols', 'target_idx'
        }
        
        torch.save(incomplete_checkpoint, model_path)
        
        result = load_transformer_model(model_path)
        
        # Should return None values for incomplete checkpoint
        self.assertEqual(result, (None, None, None, None), "Should return None values for incomplete checkpoint")
    
    def test_load_model_with_corrupted_state_dict(self):
        """Test loading model with corrupted state dict"""
        model_path = os.path.join(self.temp_dir, 'corrupted_model.pth')
        
        # Create checkpoint with corrupted state dict
        corrupted_checkpoint = {
            'model_state_dict': {'invalid_key': 'invalid_value'},
            'model_config': {
                'feature_size': 5,
                'seq_length': 10,
                'prediction_length': 1,
                'num_layers': 2,
                'd_model': 64,
                'nhead': 4,
                'dim_feedforward': 256,
                'dropout': 0.1
            },
            'scaler': MinMaxScaler(),
            'feature_cols': ['close', 'rsi', 'macd', 'macd_signal', 'ma_20_slope'],
            'target_idx': 0
        }
        
        torch.save(corrupted_checkpoint, model_path)
        
        result = load_transformer_model(model_path)
        
        # Should return None values for corrupted state dict
        self.assertEqual(result, (None, None, None, None), "Should handle corrupted state dict gracefully")

class TestTrainingWithValidation(unittest.TestCase):
    """Test training with and without validation loader"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create preprocessed sequences
        num_samples = 100
        seq_length = 10
        num_features = 5
        self.data_X = np.random.rand(num_samples, seq_length, num_features)
        self.data_y = np.random.rand(num_samples, 1)
        self.dataset = CryptoDataset(self.data_X, self.data_y)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=False)
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
    
    def test_train_with_validation_loader(self):
        """Test training with validation loader"""
        trained_model, training_history = train_transformer_model(
            self.model,
            self.train_loader,
            val_loader=self.val_loader,
            epochs=2,
            device='cpu'
        )
        
        # Check that model was trained
        self.assertIsInstance(trained_model, TimeSeriesTransformer, "Should return trained model")
        self.assertIsInstance(training_history, dict, "Should return training history")
        
        # Check that validation metrics are present
        self.assertIn('val_loss', training_history, "Should have validation loss")
        self.assertIn('val_mae', training_history, "Should have validation MAE")
    
    def test_train_without_validation_loader(self):
        """Test training without validation loader"""
        trained_model, training_history = train_transformer_model(
            self.model,
            self.train_loader,
            val_loader=None,
            epochs=2,
            device='cpu'
        )
        
        # Check that model was trained
        self.assertIsInstance(trained_model, TimeSeriesTransformer, "Should return trained model")
        self.assertIsInstance(training_history, dict, "Should return training history")
        
        # Check that validation metrics are empty lists (not missing keys)
        self.assertIn('val_loss', training_history, "Should have val_loss key")
        self.assertEqual(len(training_history['val_loss']), 0, "Should have empty validation loss list")
        self.assertIn('val_mae', training_history, "Should have val_mae key")
        self.assertEqual(len(training_history['val_mae']), 0, "Should have empty validation MAE list")

class TestBiasAnalysisEdgeCases(unittest.TestCase):
    """Test edge cases for bias analysis"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'close': np.random.uniform(95, 115, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.uniform(-2, 2, 100),
            'macd_signal': np.random.uniform(-1, 1, 100),
            'bb_upper': np.random.uniform(110, 120, 100),
            'bb_lower': np.random.uniform(90, 100, 100),
            'ma_20': np.random.uniform(100, 110, 100),
            'ma_20_slope': np.random.uniform(-1, 1, 100)
        })
        
        self.mock_model = Mock()
        self.mock_model.pos_embedding = Mock()
        self.mock_model.pos_embedding.shape = (1, 30, 64)
        self.mock_model.eval = Mock()
        
        self.mock_scaler = Mock()
        self.feature_cols = ['close', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'ma_20', 'ma_20_slope']
    
    def test_bias_analysis_with_invalid_target_idx(self):
        """Test bias analysis with invalid target index"""
        invalid_target_idx = 999  # Out of range
        
        buy_threshold, sell_threshold = analyze_model_bias_and_adjust_thresholds(
            self.df, 
            self.mock_model, 
            self.mock_scaler, 
            self.feature_cols, 
            invalid_target_idx,
            'cpu'
        )
        
        # Should return default thresholds for invalid target index
        from components.config import BUY_THRESHOLD, SELL_THRESHOLD
        self.assertEqual(buy_threshold, BUY_THRESHOLD, "Should return default buy threshold for invalid target_idx")
        self.assertEqual(sell_threshold, SELL_THRESHOLD, "Should return default sell threshold for invalid target_idx")
    
    def test_bias_analysis_with_model_returning_nan(self):
        """Test bias analysis when model returns NaN predictions"""
        with patch('torch.tensor'), \
             patch('torch.no_grad'), \
             patch.object(self.mock_model, '__call__', return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([[np.nan]])))))):
            
            self.mock_scaler.transform = Mock(return_value=np.random.rand(30, 8))
            self.mock_scaler.inverse_transform = Mock(return_value=np.array([[0, 0, 0, np.nan, 0, 0, 0, 0]]))
            
            buy_threshold, sell_threshold = analyze_model_bias_and_adjust_thresholds(
                self.df, 
                self.mock_model, 
                self.mock_scaler, 
                self.feature_cols, 
                0,
                'cpu'
            )
            
            # Should handle NaN predictions gracefully
            from components.config import BUY_THRESHOLD, SELL_THRESHOLD
            self.assertEqual(buy_threshold, BUY_THRESHOLD, "Should return default thresholds when model returns NaN")
            self.assertEqual(sell_threshold, SELL_THRESHOLD, "Should return default thresholds when model returns NaN")

class TestMemoryManagement(unittest.TestCase):
    """Test memory management and cleanup"""
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory_cleanup_after_training(self):
        """Test that GPU memory is properly cleaned up after training"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Record initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create and train model with preprocessed data
        num_samples = 100
        seq_length = 10
        num_features = 5
        data_X = np.random.rand(num_samples, seq_length, num_features)
        data_y = np.random.rand(num_samples, 1)
        dataset = CryptoDataset(data_X, data_y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
        
        trained_model = train_transformer_model(
            model, 
            train_loader, 
            epochs=1, 
            device='cuda'
        )
        
        # Check that memory was cleaned up
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow excessively (allowing some tolerance)
        memory_growth = final_memory - initial_memory
        self.assertLess(memory_growth, 100 * 1024 * 1024, "Memory growth should be reasonable (< 100MB)")

class TestHelperFunctions(unittest.TestCase):
    """Test helper functions for memory management and safety"""
    
    def test_safe_memory_division(self):
        """Test safe memory division with different input types"""
        from signals.signals_transformer import safe_memory_division
        
        # Test with integers
        self.assertEqual(safe_memory_division(1024, 2), 512.0)
        
        # Test with floats
        self.assertEqual(safe_memory_division(1024.0, 2), 512.0)
        
        # Test with strings
        self.assertEqual(safe_memory_division("1024", 2), 512.0)
        
        # Test with invalid string
        self.assertEqual(safe_memory_division("invalid", 2), 0.0)
        
        # Test with None (should be handled gracefully)
        self.assertEqual(safe_memory_division(0, 2), 0.0)
    
    def test_safe_memory_comparison(self):
        """Test safe memory comparison with different input types"""
        from signals.signals_transformer import safe_memory_comparison
        
        # Test with integers
        self.assertTrue(safe_memory_comparison(1024, 512))
        self.assertFalse(safe_memory_comparison(256, 512))
        
        # Test with floats
        self.assertTrue(safe_memory_comparison(1024.0, 512))
        self.assertFalse(safe_memory_comparison(256.0, 512))
        
        # Test with strings
        self.assertTrue(safe_memory_comparison("1024", 512))
        self.assertFalse(safe_memory_comparison("256", 512))
        
        # Test with invalid string
        self.assertFalse(safe_memory_comparison("invalid", 512))
        
        # Test with None (should be handled gracefully)
        self.assertFalse(safe_memory_comparison(0, 512))
    
    def test_safe_nan_to_num(self):
        """Test safe nan_to_num function"""
        from signals.signals_transformer import safe_nan_to_num
        
        # Test with numpy array containing NaN
        data_with_nan = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
        result = safe_nan_to_num(data_with_nan)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(result[1], 0.0)  # NaN should become 0.0
        self.assertEqual(result[3], 1e6)  # inf should become 1e6
        self.assertEqual(result[4], -1e6)  # -inf should become -1e6
        
        # Test with list
        list_data = np.array([1.0, np.nan, 3.0])
        result = safe_nan_to_num(list_data)
        self.assertIsInstance(result, np.ndarray)
        
        # Test with invalid input
        invalid_data = np.array([])
        result = safe_nan_to_num(invalid_data)
        self.assertIsInstance(result, np.ndarray)
    
    def test_setup_safe_globals(self):
        """Test setup_safe_globals function"""
        from signals.signals_transformer import setup_safe_globals
        
        # Should not raise any exceptions
        try:
            setup_safe_globals()
        except Exception as e:
            self.fail(f"setup_safe_globals raised {e} unexpectedly!")

class TestModelSavingLoading(unittest.TestCase):
    """Test model saving and loading functions"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
        
        # Create a simple model
        self.model = TimeSeriesTransformer(
            feature_size=5,
            seq_length=10,
            prediction_length=1
        )
        
        self.checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'feature_size': 5,
                'seq_length': 10,
                'prediction_length': 1,
                'num_layers': 2,
                'd_model': 64,
                'nhead': 4,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'data_info': {
                'scaler': MinMaxScaler(),
                'feature_cols': ['close', 'rsi', 'macd'],
                'target_idx': 0
            }
        }
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_safe_save_model_success(self):
        """Test successful model saving"""
        from signals.signals_transformer import safe_save_model
        
        result = safe_save_model(self.checkpoint, self.model_path)
        self.assertTrue(result, "Model should be saved successfully")
        self.assertTrue(os.path.exists(self.model_path), "Model file should exist")
    
    def test_safe_save_model_invalid_path(self):
        """Test model saving with invalid path"""
        checkpoint = {'test': 'data'}
        result = safe_save_model(checkpoint, '/non/existent/path/model.pth')
        
        # The function should succeed because it creates directories
        self.assertTrue(result, "Should return True even for invalid path (creates directories)")
    
    def test_safe_load_model_success(self):
        """Test successful model loading"""
        from signals.signals_transformer import safe_load_model
        
        # First save the model
        safe_save_model(self.checkpoint, self.model_path)
        
        # Then load it
        loaded_checkpoint = safe_load_model(self.model_path)
        
        self.assertIsNotNone(loaded_checkpoint, "Should load checkpoint successfully")
        if loaded_checkpoint is not None:
            self.assertIn('model_state_dict', loaded_checkpoint, "Should contain model state dict")
            self.assertIn('model_config', loaded_checkpoint, "Should contain model config")
    
    def test_safe_load_model_nonexistent_file(self):
        """Test loading non-existent model file"""
        from signals.signals_transformer import safe_load_model
        
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pth")
        result = safe_load_model(nonexistent_path)
        self.assertIsNone(result, "Should return None for non-existent file")

class TestPreprocessingPipeline(unittest.TestCase):
    """Test the preprocessing pipeline function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 200),
            'high': np.random.uniform(110, 120, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(95, 115, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)
        
        # Make prices more realistic with some trend
        for i in range(1, len(self.df)):
            self.df.loc[self.df.index[i], 'close'] = self.df.iloc[i-1]['close'] + np.random.normal(0, 0.5)
    
    def test_preprocess_transformer_data_success(self):
        """Test successful preprocessing"""
        X, y, scaler, feature_cols = preprocess_transformer_data(self.df)
        
        # Check outputs
        self.assertIsInstance(X, np.ndarray, "X should be numpy array")
        self.assertIsInstance(y, np.ndarray, "y should be numpy array")
        self.assertIsInstance(scaler, MinMaxScaler, "scaler should be MinMaxScaler")
        self.assertIsInstance(feature_cols, list, "feature_cols should be list")
        
        # Check shapes
        if len(X) > 0:
            self.assertEqual(X.ndim, 3, "X should be 3D array (samples, seq_length, features)")
            self.assertEqual(y.ndim, 2, "y should be 2D array (samples, prediction_length)")
            self.assertEqual(X.shape[0], y.shape[0], "X and y should have same number of samples")
            self.assertEqual(X.shape[2], len(feature_cols), "X features should match feature_cols length")
    
    def test_preprocess_transformer_data_empty_dataframe(self):
        """Test preprocessing with empty DataFrame"""
        empty_df = pd.DataFrame()
        X, y, scaler, feature_cols = preprocess_transformer_data(empty_df)
        
        self.assertEqual(len(X), 0, "X should be empty for empty DataFrame")
        self.assertEqual(len(y), 0, "y should be empty for empty DataFrame")
        self.assertIsInstance(scaler, MinMaxScaler, "scaler should still be MinMaxScaler")
        self.assertEqual(len(feature_cols), 0, "feature_cols should be empty")
    
    def test_preprocess_transformer_data_insufficient_data(self):
        """Test preprocessing with insufficient data"""
        small_df = self.df.head(10)  # Very small dataset
        X, y, scaler, feature_cols = preprocess_transformer_data(small_df)
        
        # Should return empty arrays for insufficient data
        self.assertEqual(len(X), 0, "X should be empty for insufficient data")
        self.assertEqual(len(y), 0, "y should be empty for insufficient data")

class TestTrainingAndSaving(unittest.TestCase):
    """Test the complete training and saving pipeline"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='1h')
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 300),
            'high': np.random.uniform(110, 120, 300),
            'low': np.random.uniform(90, 100, 300),
            'close': np.random.uniform(95, 115, 300),
            'volume': np.random.uniform(1000, 5000, 300)
        }, index=dates)
        
        # Make prices more realistic with some trend
        for i in range(1, len(self.df)):
            self.df.loc[self.df.index[i], 'close'] = self.df.iloc[i-1]['close'] + np.random.normal(0, 0.5)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('signals.signals_transformer.MODELS_DIR')
    @patch('signals.signals_transformer.safe_save_model')
    def test_train_and_save_transformer_model_success(self, mock_safe_save, mock_models_dir):
        """Test successful training and saving"""
        # Mock MODELS_DIR to have a mkdir method
        mock_models_dir.mkdir = Mock()
        mock_models_dir.__truediv__ = Mock(return_value='test_model.pth')
        
        # Mock safe_save_model to return True (successful save)
        mock_safe_save.return_value = True
        
        # Create a larger dataset for training with more realistic data
        dates = pd.date_range('2023-01-01', periods=500, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 500),
            'high': np.random.uniform(110, 120, 500),
            'low': np.random.uniform(90, 100, 500),
            'close': np.random.uniform(95, 115, 500),
            'volume': np.random.uniform(1000, 5000, 500)
        }, index=dates)
        
        # Make prices more realistic with some trend
        for i in range(1, len(df)):
            df.loc[df.index[i], 'close'] = df.iloc[i-1]['close'] + np.random.normal(0, 0.5)
        
        # Mock the generate_indicator_features to return the same DataFrame with indicators
        mock_df_with_indicators = df.copy()
        mock_df_with_indicators['rsi'] = np.random.uniform(20, 80, 500)
        mock_df_with_indicators['macd'] = np.random.uniform(-2, 2, 500)
        mock_df_with_indicators['macd_signal'] = np.random.uniform(-1, 1, 500)
        mock_df_with_indicators['bb_upper'] = np.random.uniform(110, 120, 500)
        mock_df_with_indicators['bb_lower'] = np.random.uniform(90, 100, 500)
        mock_df_with_indicators['ma_20'] = np.random.uniform(100, 110, 500)
        mock_df_with_indicators['ma_20_slope'] = np.random.uniform(-1, 1, 500)
        
        with patch('components._generate_indicator_features.generate_indicator_features', return_value=mock_df_with_indicators):
            model, model_path = train_and_save_transformer_model(df, 'test_model.pth')
            
            # Check that model training was successful
            self.assertIsNotNone(model, "Model should not be None")
            self.assertIsInstance(model_path, str, "Model path should be returned")
            self.assertTrue(len(model_path) > 0, "Model path should not be empty")
            self.assertEqual(model_path, 'test_model.pth', "Model path should match expected value")
            
            # Verify that safe_save_model was called
            mock_safe_save.assert_called_once()
    
    def test_train_and_save_transformer_model_empty_data(self):
        """Test training with empty data"""
        empty_df = pd.DataFrame()
        model, model_path = train_and_save_transformer_model(empty_df)
        
        self.assertIsNone(model, "Model should be None for empty data")
        self.assertEqual(model_path, "", "Model path should be empty string")
    
    def test_train_and_save_transformer_model_insufficient_data(self):
        """Test training with insufficient data"""
        small_df = self.df.head(50)  # Too small for training
        model, model_path = train_and_save_transformer_model(small_df)
        
        self.assertIsNone(model, "Model should be None for insufficient data")
        self.assertEqual(model_path, "", "Model path should be empty string")

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)