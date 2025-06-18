import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent.parent)) if str(current_dir.parent.parent.parent) not in sys.path else None

from signals._components.LSTM__class__FeedForward import FeedForward

class TestFeedForward(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.d_model = 512
        self.d_ff = 2048
        self.dropout_rate = 0.1
        self.batch_size = 32
        self.seq_len = 100
        
        self.feed_forward = FeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout_rate
        )
        
        # Create sample input tensor
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_init_default_parameters(self):
        """Test FeedForward initialization with default parameters"""
        ff = FeedForward(d_model=256, d_ff=1024)
        
        # Check layer dimensions
        self.assertEqual(ff.linear1.in_features, 256)
        self.assertEqual(ff.linear1.out_features, 1024)
        self.assertEqual(ff.linear2.in_features, 1024)
        self.assertEqual(ff.linear2.out_features, 256)
        
        # Check activation and dropout
        self.assertIsInstance(ff.activation, nn.ReLU)
        self.assertIsInstance(ff.dropout, nn.Dropout)
        self.assertEqual(ff.dropout.p, 0.1)  # Default dropout rate
    
    def test_init_custom_parameters(self):
        """Test FeedForward initialization with custom parameters"""
        d_model, d_ff, dropout = 128, 512, 0.3
        ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        self.assertEqual(ff.linear1.in_features, d_model)
        self.assertEqual(ff.linear1.out_features, d_ff)
        self.assertEqual(ff.linear2.in_features, d_ff)
        self.assertEqual(ff.linear2.out_features, d_model)
        self.assertEqual(ff.dropout.p, dropout)
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape"""
        output = self.feed_forward(self.sample_input)
        
        # Output shape should match input shape
        self.assertEqual(output.shape, self.sample_input.shape)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes"""
        batch_sizes = [1, 16, 64, 128]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(batch_size, self.seq_len, self.d_model)
                output = self.feed_forward(input_tensor)
                
                self.assertEqual(output.shape, (batch_size, self.seq_len, self.d_model))
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
    
    def test_forward_pass_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths"""
        seq_lengths = [1, 10, 50, 200]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                input_tensor = torch.randn(self.batch_size, seq_len, self.d_model)
                output = self.feed_forward(input_tensor)
                
                self.assertEqual(output.shape, (self.batch_size, seq_len, self.d_model))
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
    
    def test_forward_pass_values(self):
        """Test forward pass produces reasonable values"""
        output = self.feed_forward(self.sample_input)
        
        # Check output is not all zeros or ones
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        self.assertFalse(torch.allclose(output, torch.ones_like(output)))
        
        # Check output has reasonable range (not too extreme)
        self.assertTrue(output.abs().max() < 100)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_training_vs_eval_mode(self):
        """Test difference between training and evaluation modes"""
        input_tensor = torch.randn(10, 20, self.d_model)
        
        # Training mode
        self.feed_forward.train()
        output_train = self.feed_forward(input_tensor)
        
        # Evaluation mode
        self.feed_forward.eval()
        output_eval = self.feed_forward(input_tensor)
        
        # Outputs should be different due to dropout in training mode
        # Note: This test might occasionally fail due to randomness
        with torch.no_grad():
            # Run multiple times to increase chance of detecting difference
            differences = []
            for _ in range(10):
                self.feed_forward.train()
                out_train = self.feed_forward(input_tensor)
                self.feed_forward.eval()
                out_eval = self.feed_forward(input_tensor)
                differences.append(torch.allclose(out_train, out_eval, atol=1e-6))
            
            # At least some outputs should be different
            self.assertFalse(all(differences))
    
    def test_gradient_flow(self):
        """Test gradient flow through the network"""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        output = self.feed_forward(input_tensor)
        
        # Create a simple loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are non-zero
        self.assertIsNotNone(input_tensor.grad)
        grad = input_tensor.grad
        assert grad is not None
        self.assertFalse(torch.allclose(grad, torch.zeros_like(grad)))
        
        # Check gradients for network parameters
        for param in self.feed_forward.parameters():
            self.assertIsNotNone(param.grad)
            grad = param.grad
            assert grad is not None
            self.assertFalse(torch.allclose(grad, torch.zeros_like(grad)))
        """Test total parameter count"""
        total_params = sum(p.numel() for p in self.feed_forward.parameters())
        
        # Calculate expected parameters
        # linear1: d_model * d_ff + d_ff (weights + bias)
        # linear2: d_ff * d_model + d_model (weights + bias)
        expected_params = (self.d_model * self.d_ff + self.d_ff + 
                          self.d_ff * self.d_model + self.d_model)
        
        self.assertEqual(total_params, expected_params)
    
    def test_layer_types(self):
        """Test that layers are of correct types"""
        self.assertIsInstance(self.feed_forward.linear1, nn.Linear)
        self.assertIsInstance(self.feed_forward.linear2, nn.Linear)
        self.assertIsInstance(self.feed_forward.dropout, nn.Dropout)
        self.assertIsInstance(self.feed_forward.activation, nn.ReLU)
    
    def test_zero_dropout(self):
        """Test behavior with zero dropout"""
        ff_no_dropout = FeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=0.0)
        
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Should produce same output in train and eval modes
        ff_no_dropout.train()
        output_train = ff_no_dropout(input_tensor)
        
        ff_no_dropout.eval()
        output_eval = ff_no_dropout(input_tensor)
        
        torch.testing.assert_close(output_train, output_eval, atol=1e-6, rtol=1e-6)
    
    def test_high_dropout(self):
        """Test behavior with high dropout rate"""
        ff_high_dropout = FeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=0.9)
        
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        ff_high_dropout.train()
        output = ff_high_dropout(input_tensor)
        
        # Output should still have reasonable shape and values
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_activation_function(self):
        """Test that ReLU activation is working correctly"""
        # Create input that will definitely produce negative values after first linear layer
        input_tensor = torch.full((1, 1, self.d_model), -10.0)
        
        # Forward through first linear layer manually
        intermediate = self.feed_forward.linear1(input_tensor)
        activated = self.feed_forward.activation(intermediate)
        
        # All negative values should be clipped to 0
        self.assertTrue((activated >= 0).all())
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        torch.manual_seed(42)
        ff1 = FeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=0.1)
        
        torch.manual_seed(42)
        ff2 = FeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=0.1)
        
        input_tensor = torch.randn(10, 20, self.d_model)
        
        # Set to eval mode to eliminate dropout randomness
        ff1.eval()
        ff2.eval()
        
        output1 = ff1(input_tensor)
        output2 = ff2(input_tensor)
        
        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6)
    
    def test_different_model_dimensions(self):
        """Test with various d_model and d_ff combinations"""
        test_cases = [
            (64, 256),
            (128, 512),
            (256, 1024),
            (512, 2048),
            (1024, 4096)
        ]
        
        for d_model, d_ff in test_cases:
            with self.subTest(d_model=d_model, d_ff=d_ff):
                ff = FeedForward(d_model=d_model, d_ff=d_ff)
                input_tensor = torch.randn(8, 16, d_model)
                output = ff(input_tensor)
                
                self.assertEqual(output.shape, (8, 16, d_model))
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
    
    def test_edge_case_single_dimension(self):
        """Test edge case with minimal dimensions"""
        ff = FeedForward(d_model=1, d_ff=2)
        input_tensor = torch.randn(1, 1, 1)
        output = ff(input_tensor)
        
        self.assertEqual(output.shape, (1, 1, 1))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


if __name__ == '__main__':
    unittest.main()
