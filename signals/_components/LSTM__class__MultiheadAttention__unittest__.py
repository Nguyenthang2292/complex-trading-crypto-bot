import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None
from signals._components.LSTM__class__MultiheadAttention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for MultiHeadAttention class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 4
        self.seq_len = 10
        self.d_model = 64
        self.num_heads = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sample input data
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_initialization_default_params(self):
        """Test MultiHeadAttention initialization with default parameters."""
        attention = MultiHeadAttention(d_model=self.d_model)
        
        self.assertEqual(attention.d_model, self.d_model)
        self.assertEqual(attention.num_heads, 8)  # Default value
        self.assertEqual(attention.d_k, self.d_model // 8)
        self.assertIsInstance(attention.W_q, nn.Linear)
        self.assertIsInstance(attention.W_k, nn.Linear)
        self.assertIsInstance(attention.W_v, nn.Linear)
        self.assertIsInstance(attention.W_o, nn.Linear)
        
    def test_initialization_custom_params(self):
        """Test MultiHeadAttention initialization with custom parameters."""
        d_model = 128
        num_heads = 4
        dropout = 0.2
        
        attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.assertEqual(attention.d_model, d_model)
        self.assertEqual(attention.num_heads, num_heads)
        self.assertEqual(attention.d_k, d_model // num_heads)
        
    def test_invalid_d_model_num_heads_combination(self):
        """Test MultiHeadAttention with d_model not divisible by num_heads."""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=63, num_heads=8)  # 63 % 8 != 0
            
    def test_forward_pass_self_attention(self):
        """Test forward pass with self-attention (Q=K=V)."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        attention.eval()
        
        with torch.no_grad():
            output = attention(self.sample_input, self.sample_input, self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertIsInstance(output, torch.Tensor)
        
    def test_forward_pass_cross_attention(self):
        """Test forward pass with cross-attention (different Q, K, V)."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        attention.eval()
        
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # Key and value should have same sequence length for cross-attention
        key_value_seq_len = self.seq_len + 2
        key = torch.randn(self.batch_size, key_value_seq_len, self.d_model)
        value = torch.randn(self.batch_size, key_value_seq_len, self.d_model)
        
        with torch.no_grad():
            output = attention(query, key, value)
        
        # Output should have same shape as query
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_forward_pass_with_mask(self):
        """Test forward pass with attention mask."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        attention.eval()
        
        # Create a mask that masks out the last 3 positions
        mask = torch.ones(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        mask[:, :, :, -3:] = 0  # Mask last 3 positions
        
        with torch.no_grad():
            output = attention(self.sample_input, self.sample_input, self.sample_input, mask=mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_attention_mechanism_internal(self):
        """Test the internal attention mechanism directly."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        
        # Create query, key, value tensors in the expected shape for attention method
        query = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        key = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        value = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        
        with torch.no_grad():
            attn_output = attention.attention(query, key, value, mask=None, dropout=None)
        
        self.assertEqual(attn_output.shape, (self.batch_size, self.num_heads, self.seq_len, attention.d_k))
        
    def test_attention_with_mask_internal(self):
        """Test internal attention mechanism with mask."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        
        query = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        key = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        value = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        
        # Create mask
        mask = torch.ones(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        mask[:, :, :, -2:] = 0  # Mask last 2 positions
        
        with torch.no_grad():
            attn_output = attention.attention(query, key, value, mask=mask, dropout=None)
        
        self.assertEqual(attn_output.shape, (self.batch_size, self.num_heads, self.seq_len, attention.d_k))
        
    def test_different_batch_sizes(self):
        """Test MultiHeadAttention with different batch sizes."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        attention.eval()
        
        for batch_size in [1, 2, 8, 16]:
            input_tensor = torch.randn(batch_size, self.seq_len, self.d_model)
            with torch.no_grad():
                output = attention(input_tensor, input_tensor, input_tensor)
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], self.seq_len)
            self.assertEqual(output.shape[2], self.d_model)
            
    def test_different_sequence_lengths(self):
        """Test MultiHeadAttention with different sequence lengths."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        attention.eval()
        
        for seq_len in [5, 10, 20, 50]:
            input_tensor = torch.randn(self.batch_size, seq_len, self.d_model)
            with torch.no_grad():
                output = attention(input_tensor, input_tensor, input_tensor)
            self.assertEqual(output.shape, (self.batch_size, seq_len, self.d_model))
            
    def test_different_num_heads(self):
        """Test MultiHeadAttention with different number of heads."""
        for num_heads in [1, 2, 4, 8, 16]:
            if self.d_model % num_heads == 0:  # Only test valid combinations
                attention = MultiHeadAttention(d_model=self.d_model, num_heads=num_heads)
                attention.eval()
                
                with torch.no_grad():
                    output = attention(self.sample_input, self.sample_input, self.sample_input)
                
                self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
                
    def test_gradient_computation(self):
        """Test gradient computation through MultiHeadAttention."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        
        output = attention(input_tensor, input_tensor, input_tensor)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        
    def test_device_consistency(self):
        """Test that scale tensor is moved to correct device."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        
        # Test on CPU
        input_cpu = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output_cpu = attention(input_cpu, input_cpu, input_cpu)
        self.assertEqual(output_cpu.device, input_cpu.device)
        
        # Test on GPU if available
        if torch.cuda.is_available():
            attention_gpu = attention.cuda()
            input_gpu = input_cpu.cuda()
            output_gpu = attention_gpu(input_gpu, input_gpu, input_gpu)
            self.assertEqual(output_gpu.device, input_gpu.device)
            
    def test_dropout_effect(self):
        """Test that dropout has effect during training."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, dropout=0.5)
        
        # Training mode - outputs should be different due to dropout
        attention.train()
        output1 = attention(self.sample_input, self.sample_input, self.sample_input)
        output2 = attention(self.sample_input, self.sample_input, self.sample_input)
        
        # Outputs should be different due to dropout randomness
        self.assertFalse(torch.allclose(output1, output2, atol=1e-6))
        
        # Eval mode - outputs should be identical
        attention.eval()
        with torch.no_grad():
            output3 = attention(self.sample_input, self.sample_input, self.sample_input)
            output4 = attention(self.sample_input, self.sample_input, self.sample_input)
        
        self.assertTrue(torch.allclose(output3, output4, atol=1e-6))
        
    def test_linear_layer_dimensions(self):
        """Test that linear layers have correct dimensions."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        
        # Check input and output dimensions of linear layers
        self.assertEqual(attention.W_q.in_features, self.d_model)
        self.assertEqual(attention.W_q.out_features, self.d_model)
        self.assertEqual(attention.W_k.in_features, self.d_model)
        self.assertEqual(attention.W_k.out_features, self.d_model)
        self.assertEqual(attention.W_v.in_features, self.d_model)
        self.assertEqual(attention.W_v.out_features, self.d_model)
        self.assertEqual(attention.W_o.in_features, self.d_model)
        self.assertEqual(attention.W_o.out_features, self.d_model)
        
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 (implicitly through softmax)."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        
        # Create simple query, key, value for testing
        query = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        key = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        value = torch.randn(self.batch_size, self.num_heads, self.seq_len, attention.d_k)
        
        # Manually compute attention scores to verify softmax behavior
        with torch.no_grad():
            scores = torch.matmul(query, key.transpose(-2, -1)) / attention.scale.to(query.device)
            attention_weights = torch.softmax(scores, dim=-1)
            
            # Check that weights sum to 1 along the last dimension
            weight_sums = torch.sum(attention_weights, dim=-1)
            expected_sums = torch.ones_like(weight_sums)
            self.assertTrue(torch.allclose(weight_sums, expected_sums, atol=1e-6))
            
    def test_single_head_attention(self):
        """Test MultiHeadAttention with single head."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=1)
        attention.eval()
        
        with torch.no_grad():
            output = attention(self.sample_input, self.sample_input, self.sample_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(attention.d_k, self.d_model)  # With 1 head, d_k should equal d_model
        
    def test_zero_input(self):
        """Test MultiHeadAttention with zero input."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        attention.eval()
        
        zero_input = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        
        with torch.no_grad():
            output = attention(zero_input, zero_input, zero_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        # Output should not be all zeros due to linear transformations and biases
        
    def test_parameter_count(self):
        """Test that the model has reasonable parameter count."""
        attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        
        param_count = sum(p.numel() for p in attention.parameters())
        expected_approx = 4 * self.d_model * self.d_model  # Rough estimate for 4 linear layers
        
        self.assertGreater(param_count, 0)
        self.assertLess(param_count, expected_approx * 2)  # Should be in reasonable range


if __name__ == '__main__':
    unittest.main()
