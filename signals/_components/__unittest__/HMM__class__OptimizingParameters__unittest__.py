import unittest
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from signals._components.HMM__class__OptimizingParameters import OptimizingParameters

class TestOptimizingParametersHMM(unittest.TestCase):
    """Test cases for OptimizingParametersHMM dataclass."""
    
    def test_default_initialization(self):
        """Test initialization with default values."""
        params = OptimizingParameters()
        
        # Test HIGH_ORDER_HMM strategy parameters
        self.assertEqual(params.orders_argrelextrema, 5)
        self.assertFalse(params.strict_mode)
        
        # Test HMM_KAMA strategy parameters
        self.assertEqual(params.fast_kama, 2)
        self.assertEqual(params.slow_kama, 30)
        self.assertEqual(params.window_kama, 10)
        
        # Test shared parameters
        self.assertEqual(params.window_size, 200)
        self.assertEqual(params.lot_size_ratio, 1e-6)
        self.assertEqual(params.take_profit_pct, 0.0001)
        self.assertEqual(params.stop_loss_pct, 0.0001)
    
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        params = OptimizingParameters(
            orders_argrelextrema=10,
            strict_mode=True,
            fast_kama=5,
            slow_kama=50,
            window_kama=20,
            window_size=300,
            lot_size_ratio=2e-6,
            take_profit_pct=0.0002,
            stop_loss_pct=0.0003
        )
        
        self.assertEqual(params.orders_argrelextrema, 10)
        self.assertTrue(params.strict_mode)
        self.assertEqual(params.fast_kama, 5)
        self.assertEqual(params.slow_kama, 50)
        self.assertEqual(params.window_kama, 20)
        self.assertEqual(params.window_size, 300)
        self.assertEqual(params.lot_size_ratio, 2e-6)
        self.assertEqual(params.take_profit_pct, 0.0002)
        self.assertEqual(params.stop_loss_pct, 0.0003)
    
    def test_dataclass_features(self):
        """Test dataclass automatic features."""
        params1 = OptimizingParameters()
        params2 = OptimizingParameters()
        params3 = OptimizingParameters(orders_argrelextrema=10)
        
        # Test equality
        self.assertEqual(params1, params2)
        self.assertNotEqual(params1, params3)
        
        # Test string representation
        repr_str = repr(params1)
        self.assertIn("OptimizingParametersHMM", repr_str)
        self.assertIn("orders_argrelextrema=5", repr_str)
        self.assertIn("strict_mode=False", repr_str)
    
    def test_type_annotations(self):
        """Test that type annotations are preserved."""
        params = OptimizingParameters()
        annotations = OptimizingParameters.__annotations__
        
        self.assertEqual(annotations['orders_argrelextrema'], int)
        self.assertEqual(annotations['strict_mode'], bool)
        self.assertEqual(annotations['fast_kama'], int)
        self.assertEqual(annotations['slow_kama'], int)
        self.assertEqual(annotations['window_kama'], int)
        self.assertEqual(annotations['window_size'], int)
        self.assertEqual(annotations['lot_size_ratio'], float)
        self.assertEqual(annotations['take_profit_pct'], float)
        self.assertEqual(annotations['stop_loss_pct'], float)
    
    def test_validation_orders_argrelextrema(self):
        """Test validation for orders_argrelextrema parameter."""
        # Valid values
        OptimizingParameters(orders_argrelextrema=1)
        OptimizingParameters(orders_argrelextrema=10)
        
        # Invalid values
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(orders_argrelextrema=0)
        self.assertIn("orders_argrelextrema must be >= 1", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(orders_argrelextrema=-1)
        self.assertIn("orders_argrelextrema must be >= 1", str(context.exception))
    
    def test_validation_kama_parameters(self):
        """Test validation for KAMA parameters."""
        # Valid values
        OptimizingParameters(fast_kama=1, slow_kama=2)
        OptimizingParameters(fast_kama=5, slow_kama=30)
        
        # Invalid fast_kama
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(fast_kama=0, slow_kama=30)
        self.assertIn("KAMA parameters must be >= 1", str(context.exception))
        
        # Invalid slow_kama
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(fast_kama=2, slow_kama=0)
        self.assertIn("KAMA parameters must be >= 1", str(context.exception))
        
        # fast_kama >= slow_kama
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(fast_kama=30, slow_kama=30)
        self.assertIn("fast_kama must be < slow_kama", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(fast_kama=35, slow_kama=30)
        self.assertIn("fast_kama must be < slow_kama", str(context.exception))
    
    def test_validation_window_sizes(self):
        """Test validation for window size parameters."""
        # Valid values
        OptimizingParameters(window_kama=1, window_size=1)
        OptimizingParameters(window_kama=50, window_size=500)
        
        # Invalid window_kama
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(window_kama=0)
        self.assertIn("Window sizes must be >= 1", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(window_kama=-5)
        self.assertIn("Window sizes must be >= 1", str(context.exception))
        
        # Invalid window_size
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(window_size=0)
        self.assertIn("Window sizes must be >= 1", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(window_size=-10)
        self.assertIn("Window sizes must be >= 1", str(context.exception))
    
    def test_validation_lot_size_ratio(self):
        """Test validation for lot_size_ratio parameter."""
        # Valid values
        OptimizingParameters(lot_size_ratio=1e-8)
        OptimizingParameters(lot_size_ratio=0.1)
        OptimizingParameters(lot_size_ratio=1.0)
        
        # Invalid values
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(lot_size_ratio=0.0)
        self.assertIn("lot_size_ratio must be > 0", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(lot_size_ratio=-0.1)
        self.assertIn("lot_size_ratio must be > 0", str(context.exception))
    
    def test_validation_profit_loss_percentages(self):
        """Test validation for profit and loss percentage parameters."""
        # Valid values
        OptimizingParameters(take_profit_pct=1e-6, stop_loss_pct=1e-6)
        OptimizingParameters(take_profit_pct=0.01, stop_loss_pct=0.02)
        
        # Invalid take_profit_pct
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(take_profit_pct=0.0)
        self.assertIn("Profit/loss percentages must be > 0", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(take_profit_pct=-0.01)
        self.assertIn("Profit/loss percentages must be > 0", str(context.exception))
        
        # Invalid stop_loss_pct
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(stop_loss_pct=0.0)
        self.assertIn("Profit/loss percentages must be > 0", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(stop_loss_pct=-0.02)
        self.assertIn("Profit/loss percentages must be > 0", str(context.exception))
    
    def test_edge_cases_valid_values(self):
        """Test edge cases with valid minimum values."""
        params = OptimizingParameters(
            orders_argrelextrema=1,
            fast_kama=1,
            slow_kama=2,
            window_kama=1,
            window_size=1,
            lot_size_ratio=1e-10,
            take_profit_pct=1e-10,
            stop_loss_pct=1e-10
        )
        
        self.assertEqual(params.orders_argrelextrema, 1)
        self.assertEqual(params.fast_kama, 1)
        self.assertEqual(params.slow_kama, 2)
        self.assertEqual(params.window_kama, 1)
        self.assertEqual(params.window_size, 1)
        self.assertEqual(params.lot_size_ratio, 1e-10)
        self.assertEqual(params.take_profit_pct, 1e-10)
        self.assertEqual(params.stop_loss_pct, 1e-10)
    
    def test_multiple_validation_errors(self):
        """Test that validation catches first error when multiple errors exist."""
        # This should catch the first validation error (orders_argrelextrema)
        with self.assertRaises(ValueError) as context:
            OptimizingParameters(
                orders_argrelextrema=0,
                fast_kama=0,
                slow_kama=0
            )
        self.assertIn("orders_argrelextrema must be >= 1", str(context.exception))
    
    def test_strategy_parameter_grouping(self):
        """Test that parameters are correctly grouped by strategy."""
        params = OptimizingParameters()
        
        # HIGH_ORDER_HMM strategy parameters
        high_order_params = ['orders_argrelextrema', 'strict_mode']
        for param in high_order_params:
            self.assertTrue(hasattr(params, param))
        
        # HMM_KAMA strategy parameters
        hmm_kama_params = ['fast_kama', 'slow_kama', 'window_kama']
        for param in hmm_kama_params:
            self.assertTrue(hasattr(params, param))
        
        # Shared parameters
        shared_params = ['window_size', 'lot_size_ratio', 'take_profit_pct', 'stop_loss_pct']
        for param in shared_params:
            self.assertTrue(hasattr(params, param))
    
    def test_realistic_trading_values(self):
        """Test with realistic trading parameter values."""
        realistic_params = OptimizingParameters(
            orders_argrelextrema=3,
            strict_mode=True,
            fast_kama=2,
            slow_kama=20,
            window_kama=14,
            window_size=100,
            lot_size_ratio=0.001,
            take_profit_pct=0.02,
            stop_loss_pct=0.01
        )
        
        # Verify all values are set correctly
        self.assertEqual(realistic_params.orders_argrelextrema, 3)
        self.assertTrue(realistic_params.strict_mode)
        self.assertEqual(realistic_params.fast_kama, 2)
        self.assertEqual(realistic_params.slow_kama, 20)
        self.assertEqual(realistic_params.window_kama, 14)
        self.assertEqual(realistic_params.window_size, 100)
        self.assertEqual(realistic_params.lot_size_ratio, 0.001)
        self.assertEqual(realistic_params.take_profit_pct, 0.02)
        self.assertEqual(realistic_params.stop_loss_pct, 0.01)
    
    def test_immutability_after_creation(self):
        """Test that parameters can be modified after creation (dataclass is mutable by default)."""
        params = OptimizingParameters()
        original_value = params.orders_argrelextrema
        
        # Modify parameter
        params.orders_argrelextrema = 10
        self.assertEqual(params.orders_argrelextrema, 10)
        self.assertNotEqual(params.orders_argrelextrema, original_value)
        
        # Note: If immutability is desired, use @dataclass(frozen=True)


if __name__ == '__main__':
    unittest.main()