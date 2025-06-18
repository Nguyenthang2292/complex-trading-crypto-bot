import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent.parent)) if str(current_dir.parent.parent.parent) not in sys.path else None

from signals._components.LSTM__class__Models import (
    LSTMModel, 
    LSTMAttentionModel, 
    CNN1DExtractor, 
    CNNLSTMAttentionModel
)

class TestLSTMModel(unittest.TestCase):
    """Test cases for LSTMModel class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 4
        self.seq_len = 10
        self.input_size = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sample input data
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
    def test_initialization_default_params(self):
        """Test LSTMModel initialization with default parameters."""
        model = LSTMModel(input_size=self.input_size)
        
        self.assertEqual(model.hidden_size, 64)
        self.assertEqual(model.num_layers, 3)
        self.assertIsInstance(model.lstm1, nn.LSTM)
        self.assertIsInstance(model.lstm2, nn.LSTM)
        self.assertIsInstance(model.lstm3, nn.LSTM)
        
    def test_initialization_custom_params(self):
        """Test LSTMModel initialization with custom parameters."""
        hidden_size = 128
        num_layers = 2
        num_classes = 5
        dropout = 0.5
        
        model = LSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.assertEqual(model.hidden_size, hidden_size)
        self.assertEqual(model.num_layers, num_layers)
        
    def test_forward_pass(self):
        """Test forward pass through LSTMModel."""
        model = LSTMModel(input_size=self.input_size)
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        self.assertTrue(torch.allclose(torch.sum(output, dim=1), torch.ones(self.batch_size), atol=1e-6))
        
    def test_invalid_input_size(self):
        """Test LSTMModel with invalid input size."""
        with self.assertRaises(ValueError):
            LSTMModel(input_size=0)
            
        with self.assertRaises(ValueError):
            LSTMModel(input_size=-1)
            
    def test_invalid_hidden_size(self):
        """Test LSTMModel with invalid hidden size."""
        with self.assertRaises(ValueError):
            LSTMModel(input_size=self.input_size, hidden_size=0)
            
    def test_invalid_dropout(self):
        """Test LSTMModel with invalid dropout."""
        with self.assertRaises(ValueError):
            LSTMModel(input_size=self.input_size, dropout=1.5)
            
        with self.assertRaises(ValueError):
            LSTMModel(input_size=self.input_size, dropout=-0.1)
            
    def test_different_batch_sizes(self):
        """Test LSTMModel with different batch sizes."""
        model = LSTMModel(input_size=self.input_size)
        model.eval()
        
        for batch_size in [1, 2, 8, 16]:
            input_tensor = torch.randn(batch_size, self.seq_len, self.input_size)
            with torch.no_grad():
                output = model(input_tensor)
            self.assertEqual(output.shape[0], batch_size)
            
    def test_gradient_computation(self):
        """Test gradient computation for LSTMModel."""
        model = LSTMModel(input_size=self.input_size)
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_size, requires_grad=True)
        
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(input_tensor.grad)


class TestLSTMAttentionModel(unittest.TestCase):
    """Test cases for LSTMAttentionModel class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 4
        self.seq_len = 10
        self.input_size = 5
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
    def test_initialization_default_params(self):
        """Test LSTMAttentionModel initialization with default parameters."""
        model = LSTMAttentionModel(input_size=self.input_size)
        
        self.assertEqual(model.hidden_size, 64)
        self.assertEqual(model.num_layers, 3)
        self.assertTrue(model.use_positional_encoding)
        self.assertEqual(model.attention_dim, 16)
        
    def test_initialization_custom_params(self):
        """Test LSTMAttentionModel initialization with custom parameters."""
        model = LSTMAttentionModel(
            input_size=self.input_size,
            hidden_size=128,
            num_heads=4,
            use_positional_encoding=False
        )
        
        self.assertEqual(model.hidden_size, 128)
        self.assertFalse(model.use_positional_encoding)
        
    def test_forward_pass(self):
        """Test forward pass through LSTMAttentionModel."""
        model = LSTMAttentionModel(input_size=self.input_size)
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        self.assertTrue(torch.allclose(torch.sum(output, dim=1), torch.ones(self.batch_size), atol=1e-6))
        
    def test_attention_head_adjustment(self):
        """Test automatic adjustment of attention heads."""
        # This should automatically adjust num_heads to be compatible
        model = LSTMAttentionModel(input_size=self.input_size, num_heads=32)
        
        # Should not raise an error and should work
        model.eval()
        with torch.no_grad():
            output = model(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    def test_without_positional_encoding(self):
        """Test LSTMAttentionModel without positional encoding."""
        model = LSTMAttentionModel(
            input_size=self.input_size,
            use_positional_encoding=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    def test_invalid_parameters(self):
        """Test LSTMAttentionModel with invalid parameters."""
        with self.assertRaises(ValueError):
            LSTMAttentionModel(input_size=0)
            
        with self.assertRaises(ValueError):
            LSTMAttentionModel(input_size=self.input_size, num_heads=0)


class TestCNN1DExtractor(unittest.TestCase):
    """Test cases for CNN1DExtractor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 4
        self.seq_len = 20
        self.input_channels = 5
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.input_channels)
        
    def test_initialization_default_params(self):
        """Test CNN1DExtractor initialization with default parameters."""
        extractor = CNN1DExtractor(input_channels=self.input_channels)
        
        self.assertEqual(extractor.input_channels, self.input_channels)
        self.assertEqual(extractor.cnn_features, 64)
        self.assertEqual(len(extractor.conv_layers), 3)  # Default kernel_sizes=[3,5,7]
        
    def test_initialization_custom_params(self):
        """Test CNN1DExtractor initialization with custom parameters."""
        cnn_features = 128
        kernel_sizes = [3, 7, 11]
        
        extractor = CNN1DExtractor(
            input_channels=self.input_channels,
            cnn_features=cnn_features,
            kernel_sizes=kernel_sizes
        )
        
        self.assertEqual(extractor.cnn_features, cnn_features)
        self.assertEqual(len(extractor.conv_layers), len(kernel_sizes))
        
    def test_forward_pass(self):
        """Test forward pass through CNN1DExtractor."""
        extractor = CNN1DExtractor(input_channels=self.input_channels)
        extractor.eval()
        
        with torch.no_grad():
            output = extractor(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))
        
    def test_different_kernel_sizes(self):
        """Test CNN1DExtractor with different kernel sizes."""
        kernel_sizes = [2, 4, 6, 8]
        extractor = CNN1DExtractor(
            input_channels=self.input_channels,
            kernel_sizes=kernel_sizes
        )
        extractor.eval()
        
        with torch.no_grad():
            output = extractor(self.sample_input)
        
        # The output sequence length may vary slightly due to padding calculations
        self.assertEqual(output.shape[0], self.batch_size)  # Batch size should match
        self.assertEqual(output.shape[2], 64)  # Features should match
        # Allow some flexibility in sequence length due to padding
        self.assertGreaterEqual(output.shape[1], self.seq_len - 2)
        self.assertLessEqual(output.shape[1], self.seq_len + 2)
        
    def test_invalid_input_channels(self):
        """Test CNN1DExtractor with invalid input channels."""
        with self.assertRaises(ValueError):
            CNN1DExtractor(input_channels=0)
            
        with self.assertRaises(ValueError):
            CNN1DExtractor(input_channels=-1)
            
    def test_single_kernel_size(self):
        """Test CNN1DExtractor with single kernel size."""
        extractor = CNN1DExtractor(
            input_channels=self.input_channels,
            kernel_sizes=[5]
        )
        extractor.eval()
        
        with torch.no_grad():
            output = extractor(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))


class TestCNNLSTMAttentionModel(unittest.TestCase):
    """Test cases for CNNLSTMAttentionModel class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 4
        self.seq_len = 20
        self.input_size = 5
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
    def test_initialization_classification_mode(self):
        """Test CNNLSTMAttentionModel initialization in classification mode."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            output_mode='classification'
        )
        
        self.assertEqual(model.input_size, self.input_size)
        self.assertEqual(model.look_back, self.seq_len)
        self.assertEqual(model.output_mode, 'classification')
        self.assertTrue(model.use_attention)
        
    def test_initialization_regression_mode(self):
        """Test CNNLSTMAttentionModel initialization in regression mode."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            output_mode='regression'
        )
        
        self.assertEqual(model.output_mode, 'regression')
        self.assertTrue(hasattr(model, 'regressor'))
        
    def test_forward_pass_classification(self):
        """Test forward pass in classification mode."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            output_mode='classification'
        )
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        self.assertTrue(torch.allclose(torch.sum(output, dim=1), torch.ones(self.batch_size), atol=1e-6))
        
    def test_forward_pass_regression(self):
        """Test forward pass in regression mode."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            output_mode='regression'
        )
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        # Tanh output should be between -1 and 1
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1))
        
    def test_without_attention(self):
        """Test CNNLSTMAttentionModel without attention mechanism."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            use_attention=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    def test_without_positional_encoding(self):
        """Test CNNLSTMAttentionModel without positional encoding."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            use_positional_encoding=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    def test_custom_parameters(self):
        """Test CNNLSTMAttentionModel with custom parameters."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            cnn_features=32,
            lstm_hidden=16,
            num_layers=1,
            num_classes=5,
            num_heads=2,
            dropout=0.1
        )
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 5))
        
    def test_attention_head_adjustment(self):
        """Test automatic adjustment of attention heads in CNNLSTMAttentionModel."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            lstm_hidden=8,  # This will make attention_dim = 4
            num_heads=8     # This should be adjusted down
        )
        model.eval()
        
        with torch.no_grad():
            output = model(self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    def test_invalid_parameters(self):
        """Test CNNLSTMAttentionModel with invalid parameters."""
        with self.assertRaises(ValueError):
            CNNLSTMAttentionModel(input_size=0, look_back=self.seq_len)
            
        with self.assertRaises(ValueError):
            CNNLSTMAttentionModel(input_size=self.input_size, look_back=0)
            
        with self.assertRaises(ValueError):
            CNNLSTMAttentionModel(
                input_size=self.input_size,
                look_back=self.seq_len,
                output_mode='invalid'
            )
            
    def test_different_sequence_lengths(self):
        """Test CNNLSTMAttentionModel with different sequence lengths."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=30
        )
        model.eval()
        
        # Test with different sequence lengths
        for seq_len in [10, 20, 30, 50]:
            input_tensor = torch.randn(self.batch_size, seq_len, self.input_size)
            with torch.no_grad():
                output = model(input_tensor)
            self.assertEqual(output.shape, (self.batch_size, 3))
            
    def test_gradient_computation(self):
        """Test gradient computation for CNNLSTMAttentionModel."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len
        )
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_size, requires_grad=True)
        
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(input_tensor.grad)
        
    def test_model_parameters_count(self):
        """Test that models have reasonable parameter counts."""
        models = [
            LSTMModel(input_size=self.input_size),
            LSTMAttentionModel(input_size=self.input_size),
            CNNLSTMAttentionModel(input_size=self.input_size, look_back=self.seq_len)
        ]
        
        for model in models:
            param_count = sum(p.numel() for p in model.parameters())
            self.assertGreater(param_count, 0)
            self.assertLess(param_count, 10_000_000)  # Reasonable upper bound
            
    def test_model_training_eval_modes(self):
        """Test that models behave correctly in train/eval modes."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len
        )
        
        # Test training mode
        model.train()
        output_train1 = model(self.sample_input)
        output_train2 = model(self.sample_input)
        
        # Test eval mode
        model.eval()
        with torch.no_grad():
            output_eval1 = model(self.sample_input)
            output_eval2 = model(self.sample_input)
        
        # In eval mode, outputs should be identical
        self.assertTrue(torch.allclose(output_eval1, output_eval2, atol=1e-6))
        
        # Shapes should be consistent
        self.assertEqual(output_train1.shape, output_eval1.shape)


class TestModelPersistence(unittest.TestCase):
    """Test cases for model saving/loading and serialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.input_size = 5
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
    def test_lstm_model_state_dict(self):
        """Test LSTMModel state dict save/load."""
        model1 = LSTMModel(input_size=self.input_size)
        model1.eval()
        
        # Get original output
        with torch.no_grad():
            output1 = model1(self.sample_input)
        
        # Save and load state dict
        state_dict = model1.state_dict()
        model2 = LSTMModel(input_size=self.input_size)
        model2.load_state_dict(state_dict)
        model2.eval()
        
        # Compare outputs
        with torch.no_grad():
            output2 = model2(self.sample_input)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
        
    def test_cnn_lstm_attention_model_state_dict(self):
        """Test CNNLSTMAttentionModel state dict save/load."""
        model1 = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            use_attention=True
        )
        model1.eval()
        
        with torch.no_grad():
            output1 = model1(self.sample_input)
        
        state_dict = model1.state_dict()
        model2 = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            use_attention=True
        )
        model2.load_state_dict(state_dict)
        model2.eval()
        
        with torch.no_grad():
            output2 = model2(self.sample_input)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestDeviceCompatibility(unittest.TestCase):
    """Test cases for CPU/GPU device compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.input_size = 5
        self.cpu_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
    def test_lstm_model_cpu(self):
        """Test LSTMModel on CPU."""
        model = LSTMModel(input_size=self.input_size)
        model = model.to('cpu')
        input_tensor = self.cpu_input.to('cpu')
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.device.type, 'cpu')
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_lstm_model_gpu(self):
        """Test LSTMModel on GPU."""
        model = LSTMModel(input_size=self.input_size)
        model = model.to('cuda')
        input_tensor = self.cpu_input.to('cuda')
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.device.type, 'cuda')
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cnn_lstm_attention_model_gpu(self):
        """Test CNNLSTMAttentionModel on GPU."""
        model = CNNLSTMAttentionModel(
            input_size=self.input_size,
            look_back=self.seq_len,
            use_attention=True
        )
        model = model.to('cuda')
        input_tensor = self.cpu_input.to('cuda')
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.device.type, 'cuda')
        self.assertEqual(output.shape, (self.batch_size, 3))


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and boundary conditions."""
    
    def test_lstm_model_single_feature(self):
        """Test LSTMModel with single input feature."""
        model = LSTMModel(input_size=1)
        input_tensor = torch.randn(2, 10, 1)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (2, 3))
        
    def test_lstm_model_single_timestep(self):
        """Test LSTMModel with single timestep."""
        model = LSTMModel(input_size=5)
        input_tensor = torch.randn(2, 1, 5)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (2, 3))
        
    def test_cnn_extractor_large_kernel_sizes(self):
        """Test CNN1DExtractor with large kernel sizes."""
        seq_len = 50
        extractor = CNN1DExtractor(
            input_channels=5,
            kernel_sizes=[15, 25, 35]
        )
        input_tensor = torch.randn(2, seq_len, 5)
        
        extractor.eval()
        with torch.no_grad():
            output = extractor(input_tensor)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[2], 64)
        
    def test_attention_model_minimal_dimensions(self):
        """Test LSTMAttentionModel with minimal dimensions."""
        model = LSTMAttentionModel(
            input_size=2,
            hidden_size=4,
            num_heads=1
        )
        input_tensor = torch.randn(1, 5, 2)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (1, 3))
        
    def test_cnn_lstm_attention_extreme_small_params(self):
        """Test CNNLSTMAttentionModel with extremely small parameters."""
        model = CNNLSTMAttentionModel(
            input_size=1,
            look_back=5,
            cnn_features=4,
            lstm_hidden=4,
            num_heads=1,
            use_attention=True
        )
        input_tensor = torch.randn(1, 5, 1)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (1, 3))


class TestModelRobustness(unittest.TestCase):
    """Test cases for model robustness and stability."""
    
    def test_lstm_model_with_nan_input(self):
        """Test LSTMModel behavior with NaN input."""
        model = LSTMModel(input_size=5)
        input_tensor = torch.randn(2, 10, 5)
        input_tensor[0, 0, 0] = float('nan')
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # Model should handle NaN gracefully - output shape should be (batch_size, num_classes)
        self.assertEqual(output.shape, (2, 3))
        
    def test_lstm_model_with_inf_input(self):
        """Test LSTMModel behavior with infinite input."""
        model = LSTMModel(input_size=5)
        input_tensor = torch.randn(2, 10, 5)
        input_tensor[0, 0, 0] = float('inf')
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (2, 3))
        
    def test_models_with_zero_input(self):
        """Test all models with zero input."""
        models = [
            LSTMModel(input_size=5),
            LSTMAttentionModel(input_size=5),
            CNNLSTMAttentionModel(input_size=5, look_back=10)
        ]
        
        zero_input = torch.zeros(2, 10, 5)
        
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(zero_input)
            
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())


class TestModelArchitectureDetails(unittest.TestCase):
    """Test cases for specific architectural details."""
    
    def test_lstm_progressive_dimension_reduction(self):
        """Test LSTM models properly reduce dimensions progressively."""
        model = LSTMModel(input_size=20, hidden_size=128)
        
        # Check LSTM layer dimensions
        self.assertEqual(model.lstm1.input_size, 20)
        self.assertEqual(model.lstm1.hidden_size, 128)
        self.assertEqual(model.lstm2.input_size, 128)
        self.assertEqual(model.lstm2.hidden_size, 32)
        self.assertEqual(model.lstm3.input_size, 32)
        self.assertEqual(model.lstm3.hidden_size, 16)
        
    def test_cnn_channel_distribution(self):
        """Test CNN1DExtractor properly distributes channels."""
        extractor = CNN1DExtractor(
            input_channels=5,
            cnn_features=64,
            kernel_sizes=[3, 5, 7, 9]  # 4 scales
        )
        
        # With 64 features and 4 scales: 16, 16, 16, 16
        expected_channels = [16, 16, 16, 16]
        
        for i, conv_layer in enumerate(extractor.conv_layers):
            conv = conv_layer[0]  # First layer is Conv1d
            self.assertEqual(conv.out_channels, expected_channels[i])
            
    def test_attention_dimension_compatibility(self):
        """Test attention dimensions are properly handled."""
        # Test case where attention_dim is not divisible by num_heads
        # attention_dim = 16, so valid divisors are 1, 2, 4, 8, 16
        model = LSTMAttentionModel(
            input_size=5,
            num_heads=7  # 16 % 7 != 0, should be adjusted to 4 (largest valid divisor <= 7)
        )
        
        # Model should automatically adjust num_heads and initialize without error
        input_tensor = torch.randn(1, 10, 5)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (1, 3))
        
    def test_cnn_lstm_attention_head_compatibility(self):
        """Test CNNLSTMAttentionModel attention head adjustment."""
        # With lstm_hidden=8, final_features = 4, so valid divisors are 1, 2, 4
        model = CNNLSTMAttentionModel(
            input_size=5,
            look_back=10,
            lstm_hidden=8,  # This will make attention_dim = 4
            num_heads=8,    # Should be adjusted to 4 (largest valid divisor <= 8)
            use_attention=True
        )
        
        input_tensor = torch.randn(1, 10, 5)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (1, 3))

        with torch.no_grad():
            output2 = model2(input_tensor)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
        
    def test_output_range_consistency(self):
        """Test output values are within expected ranges."""
        models_and_ranges = [
            (LSTMModel(input_size=5), (0.0, 1.0)),  # Softmax output
            (LSTMAttentionModel(input_size=5), (0.0, 1.0)),  # Softmax output
            (CNNLSTMAttentionModel(input_size=5, look_back=10, output_mode='regression'), (-1.0, 1.0)),  # Tanh output
        ]
        
        input_tensor = torch.randn(2, 10, 5)
        
        for model, (min_val, max_val) in models_and_ranges:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
            
            self.assertTrue(torch.all(output >= min_val - 1e-6))
            self.assertTrue(torch.all(output <= max_val + 1e-6))
            
    def test_softmax_probability_sum(self):
        """Test softmax outputs sum to 1.0."""
        classification_models = [
            LSTMModel(input_size=5),
            LSTMAttentionModel(input_size=5),
            CNNLSTMAttentionModel(input_size=5, look_back=10, output_mode='classification')
        ]
        
        input_tensor = torch.randn(3, 10, 5)
        
        for model in classification_models:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
            
            prob_sums = torch.sum(output, dim=1)
            self.assertTrue(torch.allclose(prob_sums, torch.ones(3), atol=1e-5))


class TestModelArchitectureDetails(unittest.TestCase):
    """Test cases for specific architectural details."""
    
    def test_lstm_progressive_dimension_reduction(self):
        """Test LSTM models properly reduce dimensions progressively."""
        model = LSTMModel(input_size=20, hidden_size=128)
        
        # Check LSTM layer dimensions
        self.assertEqual(model.lstm1.input_size, 20)
        self.assertEqual(model.lstm1.hidden_size, 128)
        self.assertEqual(model.lstm2.input_size, 128)
        self.assertEqual(model.lstm2.hidden_size, 32)
        self.assertEqual(model.lstm3.input_size, 32)
        self.assertEqual(model.lstm3.hidden_size, 16)
        
    def test_cnn_channel_distribution(self):
        """Test CNN1DExtractor properly distributes channels."""
        extractor = CNN1DExtractor(
            input_channels=5,
            cnn_features=64,
            kernel_sizes=[3, 5, 7, 9]  # 4 scales
        )
        
        # With 64 features and 4 scales: 16, 16, 16, 16
        expected_channels = [16, 16, 16, 16]
        
        for i, conv_layer in enumerate(extractor.conv_layers):
            conv = conv_layer[0]  # First layer is Conv1d
            self.assertEqual(conv.out_channels, expected_channels[i])
            
    def test_attention_dimension_compatibility(self):
        """Test attention dimensions are properly handled."""
        # Test case where attention_dim is not divisible by num_heads
        model = LSTMAttentionModel(
            input_size=5,
            num_heads=7  # 16 (attention_dim) % 7 != 0
        )
        
        # Model should automatically adjust num_heads
        # Check that model initializes without error
        input_tensor = torch.randn(1, 10, 5)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (1, 3))


if __name__ == '__main__':
    unittest.main()
