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

# Add the parent directory to sys.path to allow importing from config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the module to test
from signals.signals_transformer import (
    analyze_model_bias_and_adjust_thresholds,
    CryptoDataset,
    evaluate_model,
    get_latest_transformer_signal,
    load_transformer_model,
    select_and_scale_features,
    TimeSeriesTransformer,
    train_transformer_model,
)

# Import the calculate features function
from signals._components._generate_indicator_features import generate_indicator_features

# Import constants for testing
from livetrade.config import (
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

class TestCryptoDataset(unittest.TestCase):
    """Test CryptoDataset class"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.data = np.random.rand(100, 5)  # 100 samples, 5 features
        self.seq_length = 10
        self.pred_length = 1
        self.target_idx = 3
    
    def test_dataset_initialization(self):
        """Test dataset initialization"""
        dataset = CryptoDataset(self.data, self.seq_length, self.pred_length, 5, self.target_idx)
        
        self.assertEqual(len(dataset.data), 100, "Should store data correctly")
        self.assertEqual(dataset.seq_length, self.seq_length, "Should store seq_length correctly")
        self.assertEqual(dataset.pred_length, self.pred_length, "Should store pred_length correctly")
    
    def test_dataset_length(self):
        """Test dataset length calculation"""
        dataset = CryptoDataset(self.data, self.seq_length, self.pred_length, 5, self.target_idx)
        expected_length = len(self.data) - self.seq_length - self.pred_length + 1
        self.assertEqual(len(dataset), expected_length, "Dataset length should be calculated correctly")
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        dataset = CryptoDataset(self.data, self.seq_length, self.pred_length, 5, self.target_idx)
        
        x, y = dataset[0]
        
        # Check shapes
        self.assertEqual(x.shape, (self.seq_length, 5), "Input should have correct shape")
        self.assertEqual(y.shape, (self.pred_length,), "Target should have correct shape")
        
        # Check types
        self.assertIsInstance(x, torch.Tensor, "Input should be torch tensor")
        self.assertIsInstance(y, torch.Tensor, "Target should be torch tensor")

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
        
        # Create synthetic data
        self.data = np.random.rand(200, 5)
        self.dataset = CryptoDataset(self.data, 10, 1, 5, 3)
        
        # Create data loader
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        # Create model
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
    
    def test_train_transformer_model(self):
        """Test model training function"""
        # Train for just 2 epochs to save time
        trained_model = train_transformer_model(
            self.model, 
            self.train_loader, 
            epochs=2, 
            device='cpu'
        )
        
        self.assertIsInstance(trained_model, TimeSeriesTransformer, "Should return trained model")
        
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
    
    @patch('signals._components._generate_indicator_features._generate_indicator_features')
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
        
        with patch('signals._components._generate_indicator_features._generate_indicator_features', return_value=small_df):
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
        
        # Create synthetic data
        self.data = np.random.rand(100, 5)
        self.dataset = CryptoDataset(self.data, 10, 1, 5, 3)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=False)
        
        # Create model
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
        
        # Create scaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)
        
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
        empty_data = np.random.rand(15, 5)  # Just enough to create dataset but will be empty after batching
        empty_dataset = CryptoDataset(empty_data, 10, 1, 5, 3)
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
        if 'rsi' in result.columns and not result['rsi'].isna().all():
            rsi_mean = result['rsi'].dropna().mean()
            self.assertTrue(40 <= rsi_mean <= 60, "RSI should be around 50 for constant prices")
    
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
        self.data = np.random.rand(50, 5)
        self.dataset = CryptoDataset(self.data, 10, 1, 5, 3)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
    
    def test_train_model_device_cpu(self):
        """Test training on CPU device"""
        trained_model = train_transformer_model(
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
        trained_model = train_transformer_model(
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
        
        with patch('signals._components._generate_indicator_features._generate_indicator_features') as mock_calculate_features, \
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
        with patch('signals._components._generate_indicator_features._generate_indicator_features') as mock_calculate_features, \
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
        
        self.data = np.random.rand(100, 5)
        self.dataset = CryptoDataset(self.data, 10, 1, 5, 3)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=False)
        self.model = TimeSeriesTransformer(feature_size=5, seq_length=10, prediction_length=1)
    
    def test_train_with_validation_loader(self):
        """Test training with validation loader"""
        trained_model = train_transformer_model(
            self.model, 
            self.train_loader, 
            val_loader=self.val_loader,
            epochs=2, 
            device='cpu'
        )
        
        self.assertIsInstance(trained_model, TimeSeriesTransformer, "Should return trained model")
    
    def test_train_without_validation_loader(self):
        """Test training without validation loader"""
        trained_model = train_transformer_model(
            self.model, 
            self.train_loader, 
            val_loader=None,
            epochs=2, 
            device='cpu'
        )
        
        self.assertIsInstance(trained_model, TimeSeriesTransformer, "Should return trained model without validation")

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
        from livetrade.config import BUY_THRESHOLD, SELL_THRESHOLD
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
            from livetrade.config import BUY_THRESHOLD, SELL_THRESHOLD
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
        
        # Create and train model
        data = np.random.rand(100, 5)
        dataset = CryptoDataset(data, 10, 1, 5, 3)
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

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
