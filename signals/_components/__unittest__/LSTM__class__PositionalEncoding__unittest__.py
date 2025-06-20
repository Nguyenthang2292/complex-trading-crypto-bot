import unittest
import torch
import sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from signals._components.LSTM__class__PositionalEncoding import PositionalEncoding

class TestPositionalEncoding(unittest.TestCase):
    """Test cases for the PositionalEncoding class."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        d_model = 64
        max_seq_length = 100
        pos_encoder = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)
        
        # Check that pe buffer was created correctly
        self.assertTrue(hasattr(pos_encoder, 'pe'))
        self.assertEqual(pos_encoder.pe.shape, (1, max_seq_length, d_model))
    
    def test_init_invalid_d_model(self):
        """Test initialization with invalid d_model (should raise ValueError)."""
        with self.assertRaises(ValueError):
            PositionalEncoding(d_model=0)
        
        with self.assertRaises(ValueError):
            PositionalEncoding(d_model=-10)
    
    def test_init_invalid_max_seq_length(self):
        """Test initialization with invalid max_seq_length (should raise ValueError)."""
        with self.assertRaises(ValueError):
            PositionalEncoding(d_model=64, max_seq_length=0)
        
        with self.assertRaises(ValueError):
            PositionalEncoding(d_model=64, max_seq_length=-5)
    
    def test_forward_valid_input(self):
        """Test forward pass with valid input."""
        d_model = 64
        max_seq_length = 100
        batch_size = 8
        seq_len = 50
        
        pos_encoder = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_encoder(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        # Check that output is different from input (positional encoding was added)
        self.assertFalse(torch.allclose(output, x))
    
    def test_forward_invalid_input_dimensions(self):
        """Test forward pass with invalid input dimensions (should raise ValueError)."""
        d_model = 64
        pos_encoder = PositionalEncoding(d_model=d_model)
        
        # Test with 2D input
        x_2d = torch.randn(10, d_model)
        with self.assertRaises(ValueError):
            pos_encoder(x_2d)
        
        # Test with 4D input
        x_4d = torch.randn(8, 10, d_model, 2)
        with self.assertRaises(ValueError):
            pos_encoder(x_4d)
    
    def test_odd_even_d_model(self):
        """Test both odd and even d_model values."""
        # Test even d_model
        d_model_even = 64
        pos_encoder_even = PositionalEncoding(d_model=d_model_even)
        x_even = torch.randn(8, 10, d_model_even)
        output_even = pos_encoder_even(x_even)
        self.assertEqual(output_even.shape, (8, 10, d_model_even))
        
        # Test odd d_model
        d_model_odd = 65
        pos_encoder_odd = PositionalEncoding(d_model=d_model_odd)
        x_odd = torch.randn(8, 10, d_model_odd)
        output_odd = pos_encoder_odd(x_odd)
        self.assertEqual(output_odd.shape, (8, 10, d_model_odd))
    
    def test_exceed_max_seq_length(self):
        """Test sequence length exceeding the pre-computed positions."""
        d_model = 64
        max_seq_length = 50
        batch_size = 8
        longer_seq_len = 100  # Longer than max_seq_length
        
        pos_encoder = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)
        x = torch.randn(batch_size, longer_seq_len, d_model)
        output = pos_encoder(x)
        
        # Check output shape matches input shape despite longer sequence
        self.assertEqual(output.shape, (batch_size, longer_seq_len, d_model))
    
    def test_positional_encoding_values(self):
        """Test that positional encoding values follow the expected pattern."""
        d_model = 16
        max_seq_length = 20
        pos_encoder = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)
        
        # Extract the positional encoding matrix
        pe = pos_encoder.pe.squeeze(0).cpu().numpy()
        
        # Check that different positions have different encodings
        for i in range(1, max_seq_length):
            self.assertFalse(np.allclose(pe[i], pe[i-1]))
        
        # Check that the same position in different examples would get the same encoding
        batch_size = 3
        seq_len = 10
        x1 = torch.zeros(batch_size, seq_len, d_model)
        x2 = torch.zeros(batch_size, seq_len, d_model)
        
        out1 = pos_encoder(x1)
        out2 = pos_encoder(x2)
        
        # Since inputs are zero, outputs should be identical (just the positional encoding)
        self.assertTrue(torch.allclose(out1, out2))


if __name__ == '__main__':
    unittest.main()
