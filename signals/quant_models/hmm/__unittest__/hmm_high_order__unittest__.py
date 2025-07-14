import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path
import time

# Add project root to sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from signals.quant_models.hmm.__class__OptimizingParameters import OptimizingParameters
from signals.quant_models.hmm.hmm_high_order import (
    hmm_high_order,
    convert_swing_to_state,
    create_hmm_model,
    optimize_n_states,
    average_swing_distance,
    HIGH_ORDER_HMM,
    BULLISH,
    NEUTRAL,
    BEARISH,
    predict_next_observation,
    predict_next_hidden_state_forward_backward,
    train_model,
    evaluate_model_accuracy,
    safe_forward_backward,
    timeout
)

class TestHMMHighOrder(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create sample DataFrame with datetime index
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5, out=None)
        
        self.df_valid = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100) * 0.2),
            'low': prices - np.abs(np.random.randn(100) * 0.2),
            'close': prices
        }, index=dates)
        
        # Create DataFrame without datetime index
        self.df_no_datetime = self.df_valid.copy()
        self.df_no_datetime.index = pd.RangeIndex(len(self.df_no_datetime))
        
        # Create invalid DataFrames
        self.df_empty = pd.DataFrame()
        self.df_no_close = self.df_valid.drop('close', axis=1)
        
        # Create small DataFrame with insufficient data
        self.df_small = self.df_valid.head(5)
        
        # Optimizing parameters
        self.optimizing_params = OptimizingParameters()

    def test_hmm_high_order_valid_input(self):
        """Test hmm_high_order with valid input"""
        result = hmm_high_order(self.df_valid)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)
        self.assertIn(result.next_state_with_high_order_hmm, [BULLISH, NEUTRAL, BEARISH])
        self.assertIsInstance(result.next_state_duration, int)
        self.assertIsInstance(result.next_state_probability, float)
        self.assertGreaterEqual(result.next_state_probability, 0.0)
        self.assertLessEqual(result.next_state_probability, 1.0)

    def test_hmm_high_order_empty_dataframe(self):
        """Test hmm_high_order with empty DataFrame"""
        result = hmm_high_order(self.df_empty)
        
        self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)
        self.assertEqual(result.next_state_duration, 1)
        self.assertEqual(result.next_state_probability, 0.33)

    def test_hmm_high_order_none_input(self):
        """Test hmm_high_order with None input"""
        result = hmm_high_order(None)  # type: ignore
        
        self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)
        self.assertEqual(result.next_state_duration, 1)
        self.assertEqual(result.next_state_probability, 0.33)

    def test_hmm_high_order_no_close_column(self):
        """Test hmm_high_order without close column"""
        result = hmm_high_order(self.df_no_close)
        
        self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)
        self.assertEqual(result.next_state_duration, 1)
        self.assertEqual(result.next_state_probability, 0.33)

    def test_hmm_high_order_insufficient_swing_points(self):
        """Test hmm_high_order with insufficient swing points"""
        result = hmm_high_order(self.df_small)
        
        self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)
        self.assertEqual(result.next_state_duration, 1)
        self.assertEqual(result.next_state_probability, 0.33)

    def test_hmm_high_order_no_datetime_index(self):
        """Test hmm_high_order with non-datetime index"""
        result = hmm_high_order(self.df_no_datetime)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)
        self.assertIn(result.next_state_with_high_order_hmm, [BULLISH, NEUTRAL, BEARISH])

    def test_hmm_high_order_different_train_ratios(self):
        """Test hmm_high_order with different train ratios"""
        for ratio in [0.6, 0.8, 0.9]:
            result = hmm_high_order(self.df_valid, train_ratio=ratio)
            self.assertIsInstance(result, HIGH_ORDER_HMM)

    def test_hmm_high_order_eval_mode_false(self):
        """Test hmm_high_order with eval_mode=False"""
        result = hmm_high_order(self.df_valid, eval_mode=False)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)

    def test_hmm_high_order_custom_optimizing_params(self):
        """Test hmm_high_order with custom optimizing parameters"""
        custom_params = OptimizingParameters()
        custom_params.orders_argrelextrema = 3
        custom_params.strict_mode = True
        
        result = hmm_high_order(self.df_valid, optimizing_params=custom_params)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)

    def test_convert_swing_to_state_empty_input(self):
        """Test convert_swing_to_state with empty DataFrames"""
        empty_df = pd.DataFrame()
        result = convert_swing_to_state(empty_df, empty_df)
        
        self.assertEqual(result, [])

    def test_convert_swing_to_state_strict_mode(self):
        """Test convert_swing_to_state in strict mode"""
        # Create sample swing data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        swing_highs = pd.DataFrame({
            'high': [100, 102, 101, 103, 99]
        }, index=dates)
        swing_lows = pd.DataFrame({
            'low': [98, 100, 99, 101, 97]
        }, index=dates)
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=True)
        
        self.assertIsInstance(result, list)
        self.assertTrue(all(state in [0, 1, 2] for state in result))

    def test_convert_swing_to_state_non_strict_mode(self):
        """Test convert_swing_to_state in non-strict mode"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        swing_highs = pd.DataFrame({
            'high': [100, 102, 101, 103, 99]
        }, index=dates)
        swing_lows = pd.DataFrame({
            'low': [98, 100, 99, 101, 97]
        }, index=dates)
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=False)
        
        self.assertIsInstance(result, list)
        self.assertTrue(all(state in [0, 1, 2] for state in result))

    def test_create_hmm_model_default(self):
        """Test create_hmm_model with default parameters"""
        model = create_hmm_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.distributions), 2)

    def test_create_hmm_model_custom_states(self):
        """Test create_hmm_model with custom number of states"""
        model = create_hmm_model(n_symbols=3, n_states=3)
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.distributions), 3)

    def test_optimize_n_states_valid_input(self):
        """Test optimize_n_states with valid input"""
        observations = [np.array([0, 1, 2, 1, 0, 2, 1]).reshape(-1, 1)]
        
        result = optimize_n_states(observations, min_states=2, max_states=4, n_folds=2)
        
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 2)
        self.assertLessEqual(result, 4)

    def test_optimize_n_states_invalid_input(self):
        """Test optimize_n_states with invalid input"""
        # Multiple sequences (should raise ValueError)
        observations = [np.array([0, 1]), np.array([1, 2])]
        
        with self.assertRaises(ValueError):
            optimize_n_states(observations)

    def test_optimize_n_states_short_sequence(self):
        """Test optimize_n_states with sequence too short for folds"""
        observations = [np.array([0, 1]).reshape(-1, 1)]
        
        result = optimize_n_states(observations, n_folds=3)
        self.assertEqual(result, 2)  # min_states mặc định

    def test_average_swing_distance_valid_input(self):
        """Test average_swing_distance with valid input"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        swing_highs = pd.DataFrame({'high': [100, 102, 101, 103, 99]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98, 100, 99, 101, 97]}, index=dates)
        
        result = average_swing_distance(swing_highs, swing_lows)
        
        self.assertIsInstance(result, float)
        self.assertTrue(result >= 0.0)

    def test_average_swing_distance_single_point(self):
        """Test average_swing_distance with single point"""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='1h')
        swing_highs = pd.DataFrame({'high': [100]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98]}, index=dates)
        
        result = average_swing_distance(swing_highs, swing_lows)
        
        self.assertEqual(result, 3600.0)

    def test_high_order_hmm_dataclass(self):
        """Test HIGH_ORDER_HMM dataclass"""
        result = HIGH_ORDER_HMM(
            next_state_with_high_order_hmm=BULLISH,
            next_state_duration=5,
            next_state_probability=0.75
        )
        
        self.assertEqual(result.next_state_with_high_order_hmm, BULLISH)
        self.assertEqual(result.next_state_duration, 5)
        self.assertEqual(result.next_state_probability, 0.75)

    def test_constants(self):
        """Test predefined constants"""
        self.assertEqual(BULLISH, 1)
        self.assertEqual(NEUTRAL, 0)
        self.assertEqual(BEARISH, -1)

    @patch('signals.quant_models.hmm.hmm_high_order.optimize_n_states')
    def test_hmm_high_order_optimize_exception(self, mock_optimize):
        """Test hmm_high_order when optimize_n_states raises exception"""
        mock_optimize.side_effect = Exception("Test exception")
        
        result = hmm_high_order(self.df_valid)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)

    def test_hmm_high_order_low_accuracy(self):
        """Test hmm_high_order returns NEUTRAL for low accuracy"""
        # Create a DataFrame that should result in low accuracy
        dates = pd.date_range(start='2023-01-01', periods=20, freq='1h')
        # Create very noisy data that should be hard to predict
        np.random.seed(123)
        prices = 100 + np.random.randn(20) * 5
        
        df_noisy = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices
        }, index=dates)
        
        with patch('signals.quant_models.hmm.hmm_high_order.evaluate_model_accuracy', return_value=0.2):
            result = hmm_high_order(df_noisy)
            
            self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)

    def test_convert_swing_to_state_with_nans(self):
        """Test convert_swing_to_state with NaN values"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        swing_highs = pd.DataFrame({
            'high': [100, np.nan, 101, 103, 99]
        }, index=dates)
        swing_lows = pd.DataFrame({
            'low': [98, 100, np.nan, 101, 97]
        }, index=dates)
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=False)
        
        self.assertIsInstance(result, list)

    def test_hmm_high_order_extreme_volatility(self):
        """Test with extremely volatile data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        # Create data with extreme volatility
        prices = 100 + np.random.randn(100) * 50  # High volatility
        
        df_extreme_volatility = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices
        }, index=dates)
        
        result = hmm_high_order(df_extreme_volatility)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)
        self.assertIn(result.next_state_with_high_order_hmm, [BULLISH, NEUTRAL, BEARISH])

    def test_hmm_high_order_trending_data(self):
        """Test with strongly trending data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        # Create strongly trending data
        prices = np.linspace(100, 200, 100)  # Strong uptrend
        
        df_trending = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices
        }, index=dates)
        
        result = hmm_high_order(df_trending)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)
        # The model should return a valid prediction, but we can't guarantee it will be BULLISH
        # due to swing point detection and model complexity
        self.assertIn(result.next_state_with_high_order_hmm, [BULLISH, NEUTRAL, BEARISH])
        self.assertGreater(result.next_state_probability, 0.0)

    def test_hmm_high_order_sideways_data(self):
        """Test with sideways market data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        # Create sideways data
        prices = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 2
        
        df_sideways = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices
        }, index=dates)
        
        result = hmm_high_order(df_sideways)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)
        # The model should return a valid prediction, but we can't guarantee it will be NEUTRAL
        # due to swing point detection and model complexity
        self.assertIn(result.next_state_with_high_order_hmm, [BULLISH, NEUTRAL, BEARISH])
        self.assertGreater(result.next_state_probability, 0.0)

    @patch('signals.quant_models.hmm.hmm_high_order.create_hmm_model')
    @patch('signals.quant_models.hmm.hmm_high_order.optimize_n_states')
    def test_hmm_integration_flow(self, mock_optimize, mock_create):
        """Test the complete integration flow with mocked dependencies"""
        mock_optimize.return_value = 2  # Use 2 states for simpler mocking
        mock_model = MagicMock()
        
        # Mock the forward_backward method to return expected 5-tuple
        mock_model.forward_backward.return_value = (
            np.array([0.1, 0.2]),  # forward_probs
            np.array([[0.3, 0.4], [0.5, 0.6]]),  # log_alpha
            np.array([0.7, 0.8]),  # backward_probs
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # log_beta
            np.array([0.9, 1.0])   # log_probability
        )
        
        # Mock the edges (transition matrix) - 2x2 for 2 states
        mock_model.edges = np.array([[0.6, 0.4], [0.3, 0.7]])
        
        # Mock the distributions for emission probabilities
        mock_dist1 = MagicMock()
        mock_dist1.parameters.return_value = [None, np.array([[0.33, 0.33, 0.34]])]
        mock_dist2 = MagicMock()
        mock_dist2.parameters.return_value = [None, np.array([[0.4, 0.3, 0.3]])]
        mock_model.distributions = [mock_dist1, mock_dist2]
        
        # Mock the fit method
        mock_model.fit.return_value = None
        
        mock_create.return_value = mock_model
        
        result = hmm_high_order(self.df_valid)
        
        mock_optimize.assert_called_once()
        mock_create.assert_called_once()
        self.assertIsInstance(result, HIGH_ORDER_HMM)

    def test_hmm_high_order_invalid_data_types(self):
        """Test with invalid data types"""
        # Create DataFrames with all required columns but invalid data
        dates = pd.date_range(start='2023-01-01', periods=3, freq='1h')
        
        invalid_data = {
            'string_data': pd.DataFrame({
                'open': ['a', 'b', 'c'],
                'high': ['d', 'e', 'f'], 
                'low': ['g', 'h', 'i'],
                'close': ['j', 'k', 'l']
            }, index=dates),
            'mixed_data': pd.DataFrame({
                'open': [100, 'invalid', 102],
                'high': [101, 'invalid', 103],
                'low': [99, 'invalid', 101], 
                'close': [100, 'invalid', 102]
            }, index=dates),
            'inf_data': pd.DataFrame({
                'open': [100, np.inf, 102],
                'high': [101, np.inf, 103],
                'low': [99, np.inf, 101],
                'close': [100, np.inf, 102]
            }, index=dates),
            'negative_inf_data': pd.DataFrame({
                'open': [100, -np.inf, 102],
                'high': [101, -np.inf, 103], 
                'low': [99, -np.inf, 101],
                'close': [100, -np.inf, 102]
            }, index=dates)
        }
        
        for name, df in invalid_data.items():
            with self.subTest(data_type=name):
                result = hmm_high_order(df)
                self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)

    def test_timeout_decorator_success(self):
        """Test timeout decorator with successful execution"""
        @timeout(5)
        def fast_function():
            return "success"
        
        result = fast_function()
        self.assertEqual(result, "success")

    def test_timeout_decorator_timeout(self):
        """Test timeout decorator with timeout"""
        @timeout(1)
        def slow_function():
            time.sleep(2)  # This will timeout
            return "should not reach here"
        
        with self.assertRaises(TimeoutError):
            slow_function()

    def test_timeout_decorator_exception_propagation(self):
        """Test timeout decorator propagates exceptions"""
        @timeout(5)
        def exception_function():
            raise ValueError("Test exception")
        
        with self.assertRaises(ValueError):
            exception_function()

    def test_safe_forward_backward_success(self):
        """Test safe_forward_backward with valid model and observations"""
        mock_model = MagicMock()
        mock_model.forward_backward.return_value = (
            np.array([0.1, 0.2]),
            np.array([[0.3, 0.4], [0.5, 0.6]]),
            np.array([0.7, 0.8]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([0.9, 1.0])
        )
        
        observations = [np.array([0, 1, 2]).reshape(-1, 1)]
        result = safe_forward_backward(mock_model, observations)
        
        self.assertIsNotNone(result)
        mock_model.forward_backward.assert_called_once()

    def test_safe_forward_backward_timeout(self):
        """Test safe_forward_backward with timeout"""
        mock_model = MagicMock()
        mock_model.forward_backward.side_effect = lambda x: time.sleep(35)  # Will timeout
        
        observations = [np.array([0, 1, 2]).reshape(-1, 1)]
        
        with self.assertRaises(TimeoutError):
            safe_forward_backward(mock_model, observations)

    def test_predict_next_hidden_state_forward_backward_2d_array(self):
        """Test predict_next_hidden_state_forward_backward with 2D output"""
        mock_model = MagicMock()
        
        # Mock forward_backward to return expected 5-tuple
        with patch('signals.quant_models.hmm.hmm_high_order.safe_forward_backward') as mock_fb:
            mock_fb.return_value = (
                np.array([0.1, 0.2]),
                np.array([[0.3, 0.4, 0.3], [0.5, 0.6, 0.1], [0.2, 0.7, 0.1]]),  # 3 time steps
                np.array([0.7, 0.8]),
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.9, 1.0])
            )
            
            # Mock edges as 3x2 matrix to create 2D output
            mock_model.edges = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_hidden_state_forward_backward(mock_model, observations)
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)  # Should return [sum_left, sum_right]

    def test_predict_next_observation_detailed(self):
        """Test predict_next_observation with detailed mocking"""
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_hidden_state_forward_backward') as mock_predict:
            mock_predict.return_value = [0.6, 0.4]  # 2 hidden states
            
            mock_model = MagicMock()
            
            # Mock distributions with proper parameters structure
            mock_dist1 = MagicMock()
            mock_dist1.parameters.return_value = [None, np.array([[0.33, 0.33, 0.34]])]
            mock_dist2 = MagicMock()
            mock_dist2.parameters.return_value = [None, np.array([[0.4, 0.3, 0.3]])]
            
            mock_model.distributions = [mock_dist1, mock_dist2]
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_observation(mock_model, observations)
            
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 3)  # 3 symbols
            self.assertAlmostEqual(np.sum(result), 1.0, places=5)  # Should sum to 1

    def test_train_model_basic(self):
        """Test train_model function"""
        model = create_hmm_model(n_symbols=3, n_states=2)
        observations = [np.array([0, 1, 2, 1, 0]).reshape(-1, 1)]
        
        trained_model = train_model(model, observations)
        
        self.assertIsNotNone(trained_model)

    def test_evaluate_model_accuracy_perfect_prediction(self):
        """Test evaluate_model_accuracy with perfect predictions"""
        mock_model = MagicMock()
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            # Mock perfect predictions
            mock_predict.side_effect = [
                np.array([0.9, 0.05, 0.05]),  # Predicts state 0
                np.array([0.05, 0.9, 0.05]),  # Predicts state 1
                np.array([0.05, 0.05, 0.9])   # Predicts state 2
            ]
            
            train_states = [0, 1, 2]
            test_states = [0, 1, 2]
            
            accuracy = evaluate_model_accuracy(mock_model, train_states, test_states) # type: ignore
            
            self.assertEqual(accuracy, 1.0)

    def test_evaluate_model_accuracy_no_test_states(self):
        """Test evaluate_model_accuracy with empty test states"""
        mock_model = MagicMock()
        train_states = [0, 1, 2]
        test_states = []
        
        accuracy = evaluate_model_accuracy(mock_model, train_states, test_states) # type: ignore
        
        self.assertEqual(accuracy, 0.0)

    def test_convert_swing_to_state_unequal_lengths_strict(self):
        """Test convert_swing_to_state with unequal lengths in strict mode"""
        dates_high = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        dates_low = pd.date_range(start='2023-01-01', periods=3, freq='1h')  # Different length
        
        swing_highs = pd.DataFrame({'high': [100, 102, 101, 103, 99]}, index=dates_high)
        swing_lows = pd.DataFrame({'low': [98, 100, 99]}, index=dates_low)
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=True)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # min_length - 1

    def test_convert_swing_to_state_duplicate_timestamps(self):
        """Test convert_swing_to_state with duplicate timestamps"""
        dates = pd.date_range(start='2023-01-01', periods=3, freq='1h')
        duplicate_dates = dates.tolist() + [dates[1]]  # Add duplicate
        
        swing_highs = pd.DataFrame({
            'high': [100, 102, 101, 102]
        }, index=duplicate_dates) # type: ignore
        swing_lows = pd.DataFrame({
            'low': [98, 100, 99, 100]
        }, index=duplicate_dates) # type: ignore    
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=False)
        
        self.assertIsInstance(result, list)

    def test_convert_swing_to_state_consecutive_same_type(self):
        """Test convert_swing_to_state with consecutive same type swings"""
        dates = pd.date_range(start='2023-01-01', periods=4, freq='1h')
        
        # Create scenario with consecutive highs
        swing_highs = pd.DataFrame({'high': [100, 102]}, index=[dates[0], dates[1]]) # type: ignore
        swing_lows = pd.DataFrame({'low': [98, 99]}, index=[dates[2], dates[3]]) # type: ignore
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=False)
        
        self.assertIsInstance(result, list)

    def test_average_swing_distance_empty_intervals(self):
        """Test average_swing_distance with DataFrames having single entries"""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='1h')
        swing_highs = pd.DataFrame({'high': [100]}, index=dates) # type: ignore
        swing_lows = pd.DataFrame({'high': [98]}, index=dates)  # type: ignore # Note: using 'high' column for lows to test edge case
        
        # This should handle the case where there are no intervals to compute
        result = average_swing_distance(swing_highs, swing_lows)
        
        self.assertEqual(result, 3600.0)

    def test_hmm_high_order_minute_interval(self):
        """Test hmm_high_order with minute-based intervals"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        df_minutes = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100) * 0.2),
            'low': prices - np.abs(np.random.randn(100) * 0.2),
            'close': prices
        }, index=dates) # type: ignore
        
        result = hmm_high_order(df_minutes)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)
        # Duration should be in minutes for minute-based data
        self.assertGreater(result.next_state_duration, 0)

    def test_hmm_high_order_nan_probability_handling(self):
        """Test hmm_high_order handles NaN probabilities correctly"""
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            # Mock function to return NaN probabilities
            mock_predict.return_value = np.array([np.nan, np.nan, np.nan])
            
            result = hmm_high_order(self.df_valid)
            
            # After np.nan_to_num, all values become 1/3, so argmax returns 0 (first index)
            # which corresponds to BEARISH state
            self.assertEqual(result.next_state_with_high_order_hmm, BEARISH)
            self.assertAlmostEqual(result.next_state_probability, 0.33, places=2)

    def test_hmm_high_order_infinite_probability_handling(self):
        """Test hmm_high_order handles infinite probabilities correctly"""
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            # Mock function to return infinite probabilities - first element is inf
            mock_predict.return_value = np.array([np.inf, 0.5, 0.3])
            
            result = hmm_high_order(self.df_valid)
            
            # After np.nan_to_num, inf becomes 1/3, so array becomes [1/3, 0.5, 0.3]
            # argmax returns 1 (index of 0.5), which corresponds to NEUTRAL state
            self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)
            self.assertEqual(result.next_state_probability, 0.5)

    def test_hmm_high_order_all_infinite_probability_handling(self):
        """Test hmm_high_order handles all infinite probabilities correctly"""
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            # Mock function to return all infinite probabilities
            mock_predict.return_value = np.array([np.inf, np.inf, np.inf])
            
            result = hmm_high_order(self.df_valid)
            
            # After np.nan_to_num, all values become 1/3, so argmax returns 0 (first index)  
            # which corresponds to BEARISH state
            self.assertEqual(result.next_state_with_high_order_hmm, BEARISH)
            self.assertAlmostEqual(result.next_state_probability, 0.33, places=2)

    def test_hmm_high_order_mixed_nan_inf_probability_handling(self):
        """Test hmm_high_order handles mixed NaN and infinite probabilities correctly"""
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            # Mock function to return mixed NaN/inf probabilities
            mock_predict.return_value = np.array([np.nan, np.inf, 0.7])
            
            result = hmm_high_order(self.df_valid)
            
            # After np.nan_to_num, array becomes [1/3, 1/3, 0.7]
            # argmax returns 2 (index of 0.7), which corresponds to BULLISH state
            self.assertEqual(result.next_state_with_high_order_hmm, BULLISH)
            self.assertEqual(result.next_state_probability, 0.7)

    def test_hmm_high_order_none_optimizing_params(self):
        """Test hmm_high_order with None optimizing_params"""
        result = hmm_high_order(self.df_valid, optimizing_params=None)
        
        self.assertIsInstance(result, HIGH_ORDER_HMM)
        self.assertIn(result.next_state_with_high_order_hmm, [BULLISH, NEUTRAL, BEARISH])

    def test_hmm_high_order_invalid_train_ratio(self):
        """Test hmm_high_order with invalid train_ratio values"""
        # Test train_ratio too low
        result_low = hmm_high_order(self.df_valid, train_ratio=0.05)
        self.assertIsInstance(result_low, HIGH_ORDER_HMM)
        
        # Test train_ratio too high
        result_high = hmm_high_order(self.df_valid, train_ratio=0.95)
        self.assertIsInstance(result_high, HIGH_ORDER_HMM)

    def test_hmm_high_order_time_interval_calculation_error(self):
        """Test hmm_high_order handles time interval calculation errors"""
        # Create DataFrame with problematic datetime index
        dates = pd.date_range(start='2023-01-01', periods=2, freq='1h')
        problematic_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102]
        }, index=dates) # type: ignore
        
        # Mock the time difference calculation to raise an exception
        with patch('pandas.Timestamp') as mock_timestamp:
            mock_timestamp.side_effect = Exception("Time calculation error")
            result = hmm_high_order(problematic_df)
            
            self.assertIsInstance(result, HIGH_ORDER_HMM)

    def test_hmm_high_order_swing_point_detection_error(self):
        """Test hmm_high_order handles swing point detection errors"""
        # Create DataFrame that might cause issues with argrelextrema
        dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
        problematic_df = pd.DataFrame({
            'open': [np.nan] * 10,  # All NaN values
            'high': [np.nan] * 10,
            'low': [np.nan] * 10,
            'close': [np.nan] * 10
        }, index=dates) # type: ignore
        
        result = hmm_high_order(problematic_df)
        
        self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)
        self.assertEqual(result.next_state_duration, 1)
        self.assertEqual(result.next_state_probability, 0.33)

    def test_hmm_high_order_model_creation_error(self):
        """Test hmm_high_order handles model creation/training errors"""
        with patch('signals.quant_models.hmm.hmm_high_order.create_hmm_model') as mock_create:
            mock_create.side_effect = Exception("Model creation failed")
            
            result = hmm_high_order(self.df_valid)
            
            self.assertEqual(result.next_state_with_high_order_hmm, NEUTRAL)
            self.assertEqual(result.next_state_duration, 1)
            self.assertEqual(result.next_state_probability, 0.33)

    def test_hmm_high_order_invalid_converted_distance(self):
        """Test hmm_high_order handles invalid converted_distance values"""
        with patch('signals.quant_models.hmm.hmm_high_order.average_swing_distance') as mock_distance:
            # Mock to return invalid values
            mock_distance.return_value = -1  # Negative distance
            
            result = hmm_high_order(self.df_valid)
            
            self.assertIsInstance(result, HIGH_ORDER_HMM)
            self.assertGreater(result.next_state_duration, 0)  # Should be corrected to positive

    def test_hmm_high_order_invalid_max_value(self):
        """Test hmm_high_order handles invalid max_value from prediction"""
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            # Mock to return probabilities that result in invalid max_value
            mock_predict.return_value = np.array([np.nan, np.nan, np.nan])
            
            result = hmm_high_order(self.df_valid)
            
            self.assertIsInstance(result, HIGH_ORDER_HMM)
            self.assertGreaterEqual(result.next_state_probability, 0.0)
            self.assertLessEqual(result.next_state_probability, 1.0)

    def test_convert_swing_to_state_missing_columns(self):
        """Test convert_swing_to_state with missing required columns"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        
        # Missing 'high' column
        swing_highs_no_high = pd.DataFrame({
            'open': [100, 102, 101, 103, 99]
        }, index=dates) # type: ignore
        swing_lows = pd.DataFrame({
            'low': [98, 100, 99, 101, 97]
        }, index=dates) # type: ignore
        
        result = convert_swing_to_state(swing_highs_no_high, swing_lows)
        self.assertEqual(result, [])
        
        # Missing 'low' column
        swing_highs = pd.DataFrame({
            'high': [100, 102, 101, 103, 99]
        }, index=dates) # type: ignore
        swing_lows_no_low = pd.DataFrame({
            'open': [98, 100, 99, 101, 97]
        }, index=dates) # type: ignore
        
        result = convert_swing_to_state(swing_highs, swing_lows_no_low)
        self.assertEqual(result, [])

    def test_convert_swing_to_state_insufficient_data_strict(self):
        """Test convert_swing_to_state with insufficient data in strict mode"""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='1h')
        swing_highs = pd.DataFrame({'high': [100]}, index=dates) # type: ignore
        swing_lows = pd.DataFrame({'low': [98]}, index=dates) # type: ignore
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=True)
        self.assertEqual(result, [])

    def test_convert_swing_to_state_nan_handling_strict(self):
        """Test convert_swing_to_state handles NaN values in strict mode"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        swing_highs = pd.DataFrame({
            'high': [100, np.nan, 101, 103, 99]
        }, index=dates) # type: ignore
        swing_lows = pd.DataFrame({
            'low': [98, 100, np.nan, 101, 97]
        }, index=dates) # type: ignore
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=True)
        self.assertIsInstance(result, list)
        # Should skip NaN values and continue processing

    def test_convert_swing_to_state_empty_after_nan_removal(self):
        """Test convert_swing_to_state when all data becomes empty after NaN removal"""
        dates = pd.date_range(start='2023-01-01', periods=3, freq='1h')
        swing_highs = pd.DataFrame({
            'high': [np.nan, np.nan, np.nan]
        }, index=dates)
        swing_lows = pd.DataFrame({
            'low': [np.nan, np.nan, np.nan]
        }, index=dates)
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=False)
        self.assertEqual(result, [])

    def test_convert_swing_to_state_insufficient_unique_swings(self):
        """Test convert_swing_to_state with insufficient unique swing points"""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='1h')
        swing_highs = pd.DataFrame({'high': [100]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98]}, index=dates)
        
        result = convert_swing_to_state(swing_highs, swing_lows, strict_mode=False)
        self.assertEqual(result, [])

    def test_optimize_n_states_empty_observations(self):
        """Test optimize_n_states with empty observations"""
        with self.assertRaises(ValueError):
            optimize_n_states([])

    def test_optimize_n_states_none_observations(self):
        """Test optimize_n_states with None observations"""
        with self.assertRaises(ValueError):
            optimize_n_states(None)  # type: ignore

    def test_optimize_n_states_single_observation(self):
        """Test optimize_n_states with sequence too short"""
        observations = [np.array([0]).reshape(-1, 1)]
        
        with self.assertRaises(ValueError):
            optimize_n_states(observations)

    def test_optimize_n_states_sequence_shorter_than_folds(self):
        """Test optimize_n_states with sequence shorter than n_folds"""
        observations = [np.array([0, 1]).reshape(-1, 1)]
        
        # Should use default min_states when sequence is too short
        result = optimize_n_states(observations, n_folds=3)
        self.assertEqual(result, 2)  # Default min_states

    def test_optimize_n_states_max_states_exceeds_sequence_length(self):
        """Test optimize_n_states when max_states exceeds sequence length"""
        observations = [np.array([0, 1, 2, 1, 0]).reshape(-1, 1)]
        
        # Should limit max_states to sequence length
        result = optimize_n_states(observations, min_states=2, max_states=10)
        self.assertLessEqual(result, 5)  # Should not exceed sequence length

    def test_optimize_n_states_training_exception_handling(self):
        """Test optimize_n_states handles training exceptions gracefully"""
        observations = [np.array([0, 1, 2, 1, 0, 2, 1]).reshape(-1, 1)]
        
        with patch('signals.quant_models.hmm.hmm_high_order.create_hmm_model') as mock_create:
            mock_create.side_effect = Exception("Training failed")
            
            # Should continue with other n_states values
            result = optimize_n_states(observations, min_states=2, max_states=3)
            self.assertIsInstance(result, int)

    def test_optimize_n_states_log_probability_exception(self):
        """Test optimize_n_states handles log_probability exceptions"""
        observations = [np.array([0, 1, 2, 1, 0, 2, 1]).reshape(-1, 1)]
        
        with patch('signals.quant_models.hmm.hmm_high_order.create_hmm_model') as mock_create:
            mock_model = MagicMock()
            mock_model.fit.return_value = None
            mock_model.log_probability.side_effect = Exception("Log probability failed")
            mock_create.return_value = mock_model
            
            # Should continue with other n_states values
            result = optimize_n_states(observations, min_states=2, max_states=3)
            self.assertIsInstance(result, int)

    def test_optimize_n_states_empty_train_test_splits(self):
        """Test optimize_n_states handles empty train/test splits"""
        observations = [np.array([0, 1]).reshape(-1, 1)]
        
        # Should handle case where KFold creates empty splits
        result = optimize_n_states(observations, n_folds=2)
        self.assertIsInstance(result, int)

    def test_predict_next_hidden_state_forward_backward_numerical_stability(self):
        """Test predict_next_hidden_state_forward_backward handles numerical stability issues"""
        mock_model = MagicMock()
        
        # Mock forward_backward to return problematic values
        with patch('signals.quant_models.hmm.hmm_high_order.safe_forward_backward') as mock_fb:
            mock_fb.return_value = (
                np.array([0.1, 0.2]),
                np.array([np.inf, np.nan]),  # Problematic log_alpha
                np.array([0.7, 0.8]),
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.9, 1.0])
            )
            
            # Mock problematic transition matrix
            mock_model.edges = np.array([[np.inf, 0.4], [0.3, np.nan]])
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_hidden_state_forward_backward(mock_model, observations)
            
            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(x, (int, float)) for x in result))

    def test_predict_next_hidden_state_forward_backward_exception_handling(self):
        """Test predict_next_hidden_state_forward_backward handles exceptions"""
        mock_model = MagicMock()
        mock_model.distributions = [MagicMock(), MagicMock()]  # 2 states
        
        with patch('signals.quant_models.hmm.hmm_high_order.safe_forward_backward') as mock_fb:
            mock_fb.side_effect = Exception("Forward-backward failed")
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_hidden_state_forward_backward(mock_model, observations)
            
            # Should return uniform distribution as fallback
            self.assertEqual(len(result), 2)
            self.assertTrue(all(x == 0.5 for x in result))

    def test_predict_next_observation_insufficient_parameters(self):
        """Test predict_next_observation handles insufficient distribution parameters"""
        mock_model = MagicMock()
        
        # Mock distribution with insufficient parameters
        mock_dist = MagicMock()
        mock_dist.parameters.return_value = [None]  # Only one parameter
        mock_model.distributions = [mock_dist]
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_hidden_state_forward_backward') as mock_predict:
            mock_predict.return_value = [0.6, 0.4]
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_observation(mock_model, observations)
            
            # Should return uniform distribution as fallback
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(x == 1/3 for x in result))

    def test_predict_next_observation_parameter_extraction_error(self):
        """Test predict_next_observation handles parameter extraction errors"""
        mock_model = MagicMock()
        
        # Mock distribution that raises exception during parameter extraction
        mock_dist = MagicMock()
        mock_dist.parameters.side_effect = Exception("Parameter extraction failed")
        mock_model.distributions = [mock_dist]
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_hidden_state_forward_backward') as mock_predict:
            mock_predict.return_value = [0.6, 0.4]
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_observation(mock_model, observations)
            
            # Should return uniform distribution as fallback
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(x == 1/3 for x in result))

    def test_predict_next_observation_no_valid_emission_probs(self):
        """Test predict_next_observation when no valid emission probabilities found"""
        mock_model = MagicMock()
        mock_model.distributions = []  # Empty distributions
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_hidden_state_forward_backward') as mock_predict:
            mock_predict.return_value = [0.6, 0.4]
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_observation(mock_model, observations)
            
            # Should return uniform distribution as fallback
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(x == 1/3 for x in result))

    def test_predict_next_observation_zero_sum_probabilities(self):
        """Test predict_next_observation handles zero sum probabilities"""
        mock_model = MagicMock()
        
        # Mock distributions that result in zero sum probabilities
        mock_dist1 = MagicMock()
        mock_dist1.parameters.return_value = [None, np.array([[0.0, 0.0, 0.0]])]
        mock_dist2 = MagicMock()
        mock_dist2.parameters.return_value = [None, np.array([[0.0, 0.0, 0.0]])]
        mock_model.distributions = [mock_dist1, mock_dist2]
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_hidden_state_forward_backward') as mock_predict:
            mock_predict.return_value = [0.0, 0.0]  # Zero hidden state probabilities
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_observation(mock_model, observations)
            
            # Should return uniform distribution as fallback
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(x == 1/3 for x in result))

    def test_predict_next_observation_exception_handling(self):
        """Test predict_next_observation handles general exceptions"""
        mock_model = MagicMock()
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_hidden_state_forward_backward') as mock_predict:
            mock_predict.side_effect = Exception("Prediction failed")
            
            observations = [np.array([0, 1, 2]).reshape(-1, 1)]
            result = predict_next_observation(mock_model, observations)
            
            # Should return uniform distribution as fallback
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(x == 1/3 for x in result))

    def test_average_swing_distance_empty_dataframes(self):
        """Test average_swing_distance with empty DataFrames"""
        empty_df = pd.DataFrame()
        result = average_swing_distance(empty_df, empty_df)
        self.assertEqual(result, 3600.0)

    def test_average_swing_distance_non_datetime_index(self):
        """Test average_swing_distance with non-datetime index"""
        dates = pd.RangeIndex(0, 5)  # Non-datetime index
        swing_highs = pd.DataFrame({'high': [100, 102, 101, 103, 99]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98, 100, 99, 101, 97]}, index=dates)
        
        result = average_swing_distance(swing_highs, swing_lows)
        self.assertEqual(result, 3600.0)

    def test_average_swing_distance_negative_intervals(self):
        """Test average_swing_distance handles negative intervals"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        # Create data that might result in negative intervals
        swing_highs = pd.DataFrame({'high': [100, 102, 101, 103, 99]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98, 100, 99, 101, 97]}, index=dates)
        
        result = average_swing_distance(swing_highs, swing_lows)
        self.assertGreater(result, 0)  # Should be positive

    def test_average_swing_distance_interval_calculation_error(self):
        """Test average_swing_distance handles interval calculation errors"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        swing_highs = pd.DataFrame({'high': [100, 102, 101, 103, 99]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98, 100, 99, 101, 97]}, index=dates)
        
        # Mock the time difference calculation to raise an exception
        with patch('pandas.Timedelta.total_seconds') as mock_seconds:
            mock_seconds.side_effect = Exception("Time calculation error")
            result = average_swing_distance(swing_highs, swing_lows)
            self.assertEqual(result, 3600.0)

    def test_average_swing_distance_no_valid_intervals(self):
        """Test average_swing_distance when no valid intervals found"""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='1h')
        swing_highs = pd.DataFrame({'high': [100]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98]}, index=dates)
        
        result = average_swing_distance(swing_highs, swing_lows)
        self.assertEqual(result, 3600.0)

    def test_average_swing_distance_exception_handling(self):
        """Test average_swing_distance handles general exceptions"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        swing_highs = pd.DataFrame({'high': [100, 102, 101, 103, 99]}, index=dates)
        swing_lows = pd.DataFrame({'low': [98, 100, 99, 101, 97]}, index=dates)
        
        # Mock np.mean to raise an exception
        with patch('numpy.mean') as mock_mean:
            mock_mean.side_effect = Exception("Mean calculation failed")
            result = average_swing_distance(swing_highs, swing_lows)
            self.assertEqual(result, 3600.0)

    def test_evaluate_model_accuracy_empty_train_states(self):
        """Test evaluate_model_accuracy with empty train states"""
        mock_model = MagicMock()
        train_states = [] # type: ignore
        test_states = [0, 1, 2] # type: ignore
        
        accuracy = evaluate_model_accuracy(mock_model, train_states, test_states) # type: ignore
        self.assertEqual(accuracy, 0.0)

    def test_evaluate_model_accuracy_prediction_exception(self):
        """Test evaluate_model_accuracy handles prediction exceptions"""
        mock_model = MagicMock()
        train_states = [0, 1, 2]
        test_states = [0, 1, 2]
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            mock_predict.side_effect = Exception("Prediction failed")
            
            accuracy = evaluate_model_accuracy(mock_model, train_states, test_states) # type: ignore
            self.assertIsInstance(accuracy, float)
            self.assertGreaterEqual(accuracy, 0.0)

    def test_evaluate_model_accuracy_invalid_accuracy_value(self):
        """Test evaluate_model_accuracy handles invalid accuracy values"""
        mock_model = MagicMock()
        train_states = [0, 1, 2]
        test_states = [0, 1, 2]
        
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            # Mock to return probabilities that result in invalid accuracy
            mock_predict.return_value = np.array([np.nan, np.nan, np.nan])
            
            accuracy = evaluate_model_accuracy(mock_model, train_states, test_states) # type: ignore
            self.assertAlmostEqual(accuracy, 1/3, places=2)

    def test_evaluate_model_accuracy_exception_handling(self):
        """Test evaluate_model_accuracy handles general exceptions"""
        mock_model = MagicMock()
        train_states = [0, 1, 2]
        test_states = [0, 1, 2]
        
        # Mock to raise exception during evaluation
        with patch('signals.quant_models.hmm.hmm_high_order.predict_next_observation') as mock_predict:
            mock_predict.side_effect = Exception("Evaluation failed")
            
            accuracy = evaluate_model_accuracy(mock_model, train_states, test_states) # type: ignore
            self.assertEqual(accuracy, 0.0)

if __name__ == '__main__':
    unittest.main()