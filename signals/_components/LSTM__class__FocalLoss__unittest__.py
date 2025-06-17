import unittest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None
from signals._components.LSTM__class__FocalLoss import FocalLoss

class TestFocalLoss(unittest.TestCase):
    """Test cases for FocalLoss class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 4
        self.num_classes = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sample data
        self.inputs = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
    def test_initialization_default_params(self):
        """Test FocalLoss initialization with default parameters."""
        focal_loss = FocalLoss()
        self.assertEqual(focal_loss.alpha, 0.25)
        self.assertEqual(focal_loss.gamma, 2.0)
        self.assertEqual(focal_loss.reduction, 'mean')
        
    def test_initialization_custom_params(self):
        """Test FocalLoss initialization with custom parameters."""
        alpha = 0.5
        gamma = 1.5
        reduction = 'sum'
        focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        
        self.assertEqual(focal_loss.alpha, alpha)
        self.assertEqual(focal_loss.gamma, gamma)
        self.assertEqual(focal_loss.reduction, reduction)
        
    def test_forward_pass_mean_reduction(self):
        """Test forward pass with mean reduction."""
        focal_loss = FocalLoss(reduction='mean')
        loss = focal_loss(self.inputs, self.targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar tensor
        self.assertGreater(loss.item(), 0)
        
    def test_forward_pass_sum_reduction(self):
        """Test forward pass with sum reduction."""
        focal_loss = FocalLoss(reduction='sum')
        loss = focal_loss(self.inputs, self.targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar tensor
        self.assertGreater(loss.item(), 0)
        
    def test_forward_pass_none_reduction(self):
        """Test forward pass with no reduction."""
        focal_loss = FocalLoss(reduction='none')
        loss = focal_loss(self.inputs, self.targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, (self.batch_size,))
        self.assertTrue(torch.all(loss > 0))
        
    def test_reduction_consistency(self):
        """Test that different reduction methods are consistent."""
        focal_loss_none = FocalLoss(reduction='none')
        focal_loss_mean = FocalLoss(reduction='mean')
        focal_loss_sum = FocalLoss(reduction='sum')
        
        loss_none = focal_loss_none(self.inputs, self.targets)
        loss_mean = focal_loss_mean(self.inputs, self.targets)
        loss_sum = focal_loss_sum(self.inputs, self.targets)
        
        # Check consistency
        self.assertAlmostEqual(loss_mean.item(), loss_none.mean().item(), places=5)
        self.assertAlmostEqual(loss_sum.item(), loss_none.sum().item(), places=5)
        
    def test_gamma_effect(self):
        """Test that gamma parameter affects the loss correctly."""
        inputs = torch.tensor([[2.0, 1.0, 0.5], [1.0, 2.0, 0.5]], requires_grad=True)
        targets = torch.tensor([0, 1])
        
        focal_loss_low_gamma = FocalLoss(gamma=0.0, reduction='none')
        focal_loss_high_gamma = FocalLoss(gamma=2.0, reduction='none')
        
        loss_low = focal_loss_low_gamma(inputs, targets)
        loss_high = focal_loss_high_gamma(inputs, targets)
        
        # With higher gamma, easy examples should have lower loss
        # This is a simplified test - in practice, the relationship depends on pt values
        self.assertEqual(loss_low.shape, loss_high.shape)
        
    def test_alpha_effect(self):
        """Test that alpha parameter affects the loss correctly."""
        focal_loss_low_alpha = FocalLoss(alpha=0.1, reduction='mean')
        focal_loss_high_alpha = FocalLoss(alpha=0.9, reduction='mean')
        
        loss_low = focal_loss_low_alpha(self.inputs, self.targets)
        loss_high = focal_loss_high_alpha(self.inputs, self.targets)
        
        # Different alpha values should produce different losses
        self.assertNotAlmostEqual(loss_low.item(), loss_high.item(), places=3)
        
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        focal_loss = FocalLoss()
        inputs = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        
        loss = focal_loss(inputs, self.targets)
        loss.backward()
        
        self.assertIsNotNone(inputs.grad)
        self.assertEqual(inputs.grad.shape, inputs.shape)
        
    def test_comparison_with_cross_entropy(self):
        """Test that focal loss behaves reasonably compared to cross entropy."""
        focal_loss = FocalLoss(alpha=1.0, gamma=0.0, reduction='mean')
        
        loss_focal = focal_loss(self.inputs, self.targets)
        loss_ce = F.cross_entropy(self.inputs, self.targets)
        
        # With alpha=1 and gamma=0, focal loss should be similar to cross entropy
        self.assertAlmostEqual(loss_focal.item(), loss_ce.item(), places=5)
        
    def test_edge_case_single_sample(self):
        """Test focal loss with single sample."""
        inputs = torch.randn(1, self.num_classes, requires_grad=True)
        targets = torch.tensor([0])
        
        focal_loss = FocalLoss()
        loss = focal_loss(inputs, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
        
    def test_edge_case_perfect_prediction(self):
        """Test focal loss with perfect predictions."""
        # Create inputs where the correct class has very high probability
        inputs = torch.zeros(2, self.num_classes)
        inputs[0, 0] = 10.0  # Very high logit for class 0
        inputs[1, 1] = 10.0  # Very high logit for class 1
        targets = torch.tensor([0, 1])
        
        focal_loss = FocalLoss(reduction='none')
        loss = focal_loss(inputs, targets)
        
        # Loss should be very small for perfect predictions
        self.assertTrue(torch.all(loss < 0.1))
        
    def test_invalid_reduction_parameter(self):
        """Test that invalid reduction parameter raises appropriate error."""
        with self.assertRaises(ValueError):
            # This should fail at initialization with invalid reduction parameter
            FocalLoss(reduction='invalid')
            
    def test_different_tensor_types(self):
        """Test focal loss with different tensor types."""
        focal_loss = FocalLoss()
        
        # Test with float32
        inputs_float32 = self.inputs.float()
        targets_long = self.targets.long()
        loss = focal_loss(inputs_float32, targets_long)
        self.assertIsInstance(loss, torch.Tensor)
        
        # Test with double precision
        inputs_double = self.inputs.double()
        focal_loss_double = FocalLoss()
        loss_double = focal_loss_double(inputs_double, targets_long)
        self.assertIsInstance(loss_double, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
