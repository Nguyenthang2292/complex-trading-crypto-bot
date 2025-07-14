import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

# Import the functions to test
from signals.quant_models.hmm.hmm_kama import (
    hmm_kama, calculate_kama, prepare_observations, train_hmm, 
    apply_hmm_model, compute_state_using_standard_deviation,
    compute_state_using_hmm, compute_state_using_association_rule_mining,
    compute_state_using_k_means, calculate_all_state_durations,
    HMM_KAMA, calculate_composite_scores_association_rule_mining,
)

# Mock OptimizingParameters class for testing
@dataclass
class MockOptimizingParameters:
    window_kama: int = 10
    fast_kama: int = 2
    slow_kama: int = 30
    window_size: int = 20

class TestHMMKama(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        warnings.filterwarnings('ignore')
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'close': np.random.normal(100, 10, 100),
            'high': np.random.normal(105, 10, 100),
            'low': np.random.normal(95, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        
        self.optimizing_params = MockOptimizingParameters()
        
        # Small dataset for edge cases
        self.small_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        # Constant price data
        self.constant_data = pd.DataFrame({
            'close': [100] * 50
        })

        # Data with extreme values for testing robustness
        self.extreme_data = pd.DataFrame({
            'close': [1e-10, 1e10, 100, -1e5, 1e8] * 20,
            'high': [1e-10, 1e10, 105, -1e5, 1e8] * 20,
            'low': [1e-10, 1e10, 95, -1e5, 1e8] * 20
        })

        # Data with mixed invalid values
        self.mixed_invalid_data = pd.DataFrame({
            'close': [100, np.nan, 102, np.inf, 104, -np.inf, 106] * 15,
            'high': [105, np.nan, 107, np.inf, 109, -np.inf, 111] * 15,
            'low': [95, np.nan, 97, np.inf, 99, -np.inf, 101] * 15
        })

    def test_calculate_kama_normal_case(self):
        """Test KAMA calculation with normal data"""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        result = calculate_kama(prices, window=5, fast=2, slow=10)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())
        self.assertIsInstance(result, np.ndarray)

    def test_calculate_kama_insufficient_data(self):
        """Test KAMA with insufficient data"""
        prices = [100, 101]
        result = calculate_kama(prices, window=5)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_calculate_kama_with_nans(self):
        """Test KAMA with NaN values"""
        prices = [100, np.nan, 102, np.inf, 104]
        result = calculate_kama(prices, window=3)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_calculate_kama_zero_volatility(self):
        """Test KAMA with zero volatility (all same values)"""
        prices = [100] * 20
        result = calculate_kama(prices, window=5)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())
        # Should handle zero volatility gracefully

    def test_calculate_kama_extreme_values(self):
        """Test KAMA with extreme price values"""
        prices = [1e-10, 1e10, 100, -1e5, 1e8]
        result = calculate_kama(prices, window=3)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_calculate_kama_negative_prices(self):
        """Test KAMA with negative prices"""
        prices = [-100, -101, -99, -102, -98]
        result = calculate_kama(prices, window=3)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_prepare_observations_normal_case(self):
        """Test prepare_observations with normal data"""
        result = prepare_observations(self.sample_data, self.optimizing_params)  # type: ignore
        
        self.assertEqual(result.shape[0], len(self.sample_data))
        self.assertEqual(result.shape[1], 3)  # returns, kama, volatility
        self.assertTrue(np.isfinite(result).all())

    def test_prepare_observations_invalid_data(self):
        """Test prepare_observations with invalid data"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            prepare_observations(empty_df, self.optimizing_params)  # type: ignore

    def test_prepare_observations_no_close_column(self):
        """Test prepare_observations without close column"""
        invalid_df = pd.DataFrame({'high': [100, 101, 102]})
        
        with self.assertRaises(ValueError):
            prepare_observations(invalid_df, self.optimizing_params)  # type: ignore

    def test_prepare_observations_constant_prices(self):
        """Test prepare_observations with constant prices"""
        result = prepare_observations(self.constant_data, self.optimizing_params)  # type: ignore
        
        self.assertEqual(result.shape[0], len(self.constant_data))
        self.assertEqual(result.shape[1], 3)
        self.assertTrue(np.isfinite(result).all())

    def test_prepare_observations_with_inf_values(self):
        """Test prepare_observations with infinite values"""
        df_with_inf = self.sample_data.copy()
        df_with_inf.loc[10:15, 'close'] = np.inf
        df_with_inf.loc[20:25, 'close'] = -np.inf
        
        result = prepare_observations(df_with_inf, self.optimizing_params)  # type: ignore
        
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.shape[1], 3)

    def test_prepare_observations_single_unique_price(self):
        """Test prepare_observations with only one unique price"""
        df_single = pd.DataFrame({'close': [100.0] * 50})
        
        result = prepare_observations(df_single, self.optimizing_params)  # type: ignore
        
        self.assertEqual(result.shape[0], 50)
        self.assertTrue(np.isfinite(result).all())
        # Should add synthetic variance

    def test_prepare_observations_very_small_window(self):
        """Test prepare_observations with very small window"""
        params = MockOptimizingParameters()
        params.window_kama = 1
        
        result = prepare_observations(self.sample_data, params)  # type: ignore
        
        self.assertTrue(np.isfinite(result).all())

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_train_hmm_normal_case(self, mock_hmm):
        """Test HMM training with normal observations"""
        mock_model = Mock()
        mock_model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
        mock_model.means_ = np.array([[0, 1], [1, 0]])
        mock_model.fit.return_value = None  # fit method returns None
        mock_hmm.return_value = mock_model
        
        observations = np.random.normal(0, 1, (50, 2))
        result = train_hmm(observations, n_components=2)
        
        self.assertIsNotNone(result)
        # Check that GaussianHMM was called at least once (could be twice due to exception handling)
        self.assertGreaterEqual(mock_hmm.call_count, 1)

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_train_hmm_with_exception(self, mock_hmm):
        """Test HMM training when fit method raises exception"""
        mock_model = Mock()
        mock_model.fit.side_effect = Exception("Fitting failed")
        mock_hmm.return_value = mock_model
        
        observations = np.random.normal(0, 1, (50, 2))
        result = train_hmm(observations, n_components=2)
        
        self.assertIsNotNone(result)
        # Should be called twice: once for normal fit, once for fallback
        self.assertEqual(mock_hmm.call_count, 2)

    def test_train_hmm_empty_observations(self):
        """Test HMM training with empty observations"""
        empty_obs = np.array([])
        
        with self.assertRaises(ValueError):
            train_hmm(empty_obs)

    def test_train_hmm_invalid_observations(self):
        """Test HMM training with invalid observations"""
        invalid_obs = np.array([[np.inf, np.nan], [1, 2]])
        
        # Should handle invalid values gracefully
        result = train_hmm(invalid_obs)
        self.assertIsNotNone(result)

    def test_train_hmm_with_nans_in_observations(self):
        """Test HMM training with NaN values in observations"""
        observations = np.random.normal(0, 1, (50, 2))
        observations[10:15, 0] = np.nan
        observations[20:25, 1] = np.inf
        
        result = train_hmm(observations)
        self.assertIsNotNone(result)

    def test_train_hmm_single_component(self):
        """Test HMM training with single component"""
        observations = np.random.normal(0, 1, (20, 2))
        result = train_hmm(observations, n_components=1)
        
        self.assertIsNotNone(result)

    def test_train_hmm_large_observations(self):
        """Test HMM training with very large observation values"""
        observations = np.random.normal(0, 1e8, (50, 2))
        result = train_hmm(observations)
        
        self.assertIsNotNone(result)

    # New test cases for improved error handling
    def test_train_hmm_1d_observations(self):
        """Test HMM training with 1D observations array"""
        observations = np.random.normal(0, 1, 50)
        result = train_hmm(observations, n_components=2)
        
        self.assertIsNotNone(result)

    def test_train_hmm_with_low_variance(self):
        """Test HMM training with low variance observations"""
        observations = np.ones((50, 2)) * 0.1  # Very low variance
        result = train_hmm(observations)
        
        self.assertIsNotNone(result)

    def test_train_hmm_with_extreme_values(self):
        """Test HMM training with extreme values that need scaling"""
        observations = np.random.normal(0, 1e10, (50, 2))
        result = train_hmm(observations)
        
        self.assertIsNotNone(result)

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_apply_hmm_model(self, mock_hmm):
        """Test applying HMM model to data"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 3] * 25)
        mock_model.transmat_ = np.array([[0.25, 0.25, 0.25, 0.25]] * 4)
        
        observations = np.random.normal(0, 1, (100, 3))
        data, next_state = apply_hmm_model(mock_model, self.sample_data, observations)
        
        self.assertIn('state', data.columns)
        self.assertEqual(len(data), len(self.sample_data))
        self.assertIsInstance(next_state, int)
        self.assertIn(next_state, [0, 1, 2, 3])

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_apply_hmm_model_mismatched_lengths(self, mock_hmm):
        """Test apply_hmm_model with mismatched prediction and data lengths"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])  # Shorter than data
        mock_model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        observations = np.random.normal(0, 1, (100, 3))
        data, next_state = apply_hmm_model(mock_model, self.sample_data, observations)
        
        self.assertEqual(len(data), len(self.sample_data))

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_apply_hmm_model_longer_predictions(self, mock_hmm):
        """Test apply_hmm_model with predictions longer than data"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 3] * 50)  # Longer than data
        mock_model.transmat_ = np.array([[0.25, 0.25, 0.25, 0.25]] * 4)
        
        observations = np.random.normal(0, 1, (100, 3))
        data, next_state = apply_hmm_model(mock_model, self.sample_data, observations)
        
        self.assertEqual(len(data), len(self.sample_data))

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_apply_hmm_model_no_transmat(self, mock_hmm):
        """Test apply_hmm_model when model has no transition matrix"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 3] * 25)
        del mock_model.transmat_  # Remove transmat_ attribute
        
        observations = np.random.normal(0, 1, (100, 3))
        data, next_state = apply_hmm_model(mock_model, self.sample_data, observations)
        
        self.assertEqual(next_state, 0)  # Should default to 0

    def test_compute_state_using_standard_deviation(self):
        """Test state computation using standard deviation"""
        durations_df = pd.DataFrame({
            'duration': [10, 20, 30, 25, 15],
            'state': ['bullish weak'] * 5
        })
        
        result = compute_state_using_standard_deviation(durations_df)
        self.assertIn(result, [0, 1])

    def test_compute_state_using_standard_deviation_edge_cases(self):
        """Test standard deviation computation with edge cases"""
        # Single duration
        single_duration = pd.DataFrame({'duration': [10], 'state': ['bullish weak']})
        result = compute_state_using_standard_deviation(single_duration)
        self.assertIn(result, [0, 1])
        
        # Zero std
        zero_std = pd.DataFrame({'duration': [10, 10, 10], 'state': ['bullish weak'] * 3})
        result = compute_state_using_standard_deviation(zero_std)
        self.assertIn(result, [0, 1])

    def test_compute_state_using_hmm_insufficient_data(self):
        """Test HMM state computation with insufficient data"""
        small_durations = pd.DataFrame({
            'duration': [10],
            'state': ['bullish weak']
        })
        
        result_df, state = compute_state_using_hmm(small_durations)
        self.assertEqual(state, 0)
        self.assertIn('hidden_state', result_df.columns)

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_compute_state_using_hmm_normal_case(self, mock_hmm):
        """Test HMM state computation with normal data"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 1])
        mock_hmm.return_value = mock_model
        
        durations_df = pd.DataFrame({
            'duration': [10, 20, 30, 25, 15],
            'state': ['bullish weak'] * 5
        })
        
        result_df, state = compute_state_using_hmm(durations_df)
        self.assertIn(state, [0, 1])
        self.assertIn('hidden_state', result_df.columns)

    @patch('signals.quant_models.hmm.hmm_kama.GaussianHMM')
    def test_compute_state_using_hmm_exception_handling(self, mock_hmm):
        """Test HMM state computation exception handling"""
        mock_model = Mock()
        mock_model.fit.side_effect = Exception("HMM fit failed")
        mock_hmm.return_value = mock_model
        
        durations_df = pd.DataFrame({
            'duration': [10, 20, 30],
            'state': ['bullish weak'] * 3
        })
        
        result_df, state = compute_state_using_hmm(durations_df)
        self.assertEqual(state, 0)  # Should default to 0 on exception

    def test_compute_state_using_k_means_insufficient_data(self):
        """Test K-means with insufficient data"""
        small_durations = pd.DataFrame({
            'duration': [10, 20],
            'state': ['bullish weak'] * 2
        })
        
        result = compute_state_using_k_means(small_durations)
        self.assertEqual(result, 0)

    @patch('signals.quant_models.hmm.hmm_kama.KMeans')
    def test_compute_state_using_k_means_normal_case(self, mock_kmeans):
        """Test K-means with normal data"""
        mock_model = Mock()
        mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 1])
        mock_kmeans.return_value = mock_model
        
        durations_df = pd.DataFrame({
            'duration': [10, 20, 30, 25, 15],
            'state': ['bullish weak'] * 5
        })
        
        result = compute_state_using_k_means(durations_df)
        self.assertIn(result, [0, 1])

    @patch('signals.quant_models.hmm.hmm_kama.KMeans')
    def test_compute_state_using_k_means_exception_handling(self, mock_kmeans):
        """Test K-means exception handling"""
        mock_kmeans.side_effect = Exception("KMeans failed")
        
        durations_df = pd.DataFrame({
            'duration': [10, 20, 30, 25, 15],
            'state': ['bullish weak'] * 5
        })
        
        result = compute_state_using_k_means(durations_df)
        self.assertEqual(result, 0)  # Should default to 0 on exception

    def test_calculate_all_state_durations(self):
        """Test calculation of state durations"""
        data_with_states = pd.DataFrame({
            'state': ['bullish weak', 'bullish weak', 'bearish strong', 'bearish strong', 'bullish weak'],
            'close': [100, 101, 99, 98, 102]
        })
        
        result = calculate_all_state_durations(data_with_states)
        
        self.assertIn('duration', result.columns)
        self.assertIn('state', result.columns)
        self.assertEqual(len(result), 3)  # 3 distinct state segments

    def test_calculate_all_state_durations_single_state(self):
        """Test state duration calculation with single state"""
        single_state_data = pd.DataFrame({
            'state': ['bullish weak'] * 10,
            'close': np.random.normal(100, 5, 10)
        })
        
        result = calculate_all_state_durations(single_state_data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['duration'], 10)

    def test_calculate_all_state_durations_alternating_states(self):
        """Test state duration calculation with alternating states"""
        alternating_data = pd.DataFrame({
            'state': ['bullish weak', 'bearish strong'] * 10,
            'close': np.random.normal(100, 5, 20)
        })
        
        result = calculate_all_state_durations(alternating_data)
        self.assertEqual(len(result), 20)  # Each state appears once per row

    def test_calculate_composite_scores_association_rule_mining(self):
        """Test composite score calculation for association rules"""
        rules_df = pd.DataFrame({
            'support': [0.1, 0.2, 0.3],
            'confidence': [0.8, 0.9, 0.7],
            'lift': [1.5, 1.2, 1.8]
        })
        
        result = calculate_composite_scores_association_rule_mining(rules_df)
        
        self.assertIn('composite_score', result.columns)
        self.assertEqual(len(result), len(rules_df))

    def test_calculate_composite_scores_empty_rules(self):
        """Test composite score calculation with empty rules"""
        empty_rules = pd.DataFrame()
        result = calculate_composite_scores_association_rule_mining(empty_rules)
        self.assertTrue(result.empty)

    def test_calculate_composite_scores_with_inf_values(self):
        """Test composite score calculation with infinite values"""
        rules_with_inf = pd.DataFrame({
            'support': [0.1, np.inf, 0.3],
            'confidence': [0.8, 0.9, -np.inf],
            'lift': [1.5, 1.2, 1.8]
        })
        
        result = calculate_composite_scores_association_rule_mining(rules_with_inf)
        
        self.assertTrue(np.isfinite(result['composite_score']).all())

    def test_calculate_composite_scores_no_metrics(self):
        """Test composite score calculation with no valid metrics"""
        rules_no_metrics = pd.DataFrame({
            'invalid_col1': [1, 2, 3],
            'invalid_col2': ['a', 'b', 'c']
        })
        
        result = calculate_composite_scores_association_rule_mining(rules_no_metrics)
        self.assertTrue((result['composite_score'] == 0.0).all())

    @patch('signals.quant_models.hmm.hmm_kama.apriori')
    @patch('signals.quant_models.hmm.hmm_kama.fpgrowth')
    def test_compute_state_using_association_rule_mining(self, mock_fpgrowth, mock_apriori):
        """Test association rule mining state computation"""
        mock_apriori.return_value = pd.DataFrame({
            'itemsets': [frozenset(['bullish weak'])],
            'support': [0.5]
        })
        mock_fpgrowth.return_value = pd.DataFrame({
            'itemsets': [frozenset(['bearish strong'])],
            'support': [0.4]
        })
        
        durations_df = pd.DataFrame({
            'duration': [10, 25, 35, 20, 15],
            'state': ['bullish weak', 'bearish strong', 'bullish weak', 'bearish strong', 'bullish weak']
        })
        
        apriori_result, fpgrowth_result = compute_state_using_association_rule_mining(durations_df)
        
        self.assertIn(apriori_result, [0, 1, 2, 3])
        self.assertIn(fpgrowth_result, [0, 1, 2, 3])

    @patch('signals.quant_models.hmm.hmm_kama.apriori')
    @patch('signals.quant_models.hmm.hmm_kama.fpgrowth')
    def test_compute_state_using_association_rule_mining_no_frequent_itemsets(self, mock_fpgrowth, mock_apriori):
        """Test association rule mining when no frequent itemsets found"""
        mock_apriori.return_value = pd.DataFrame()  # Empty DataFrame
        mock_fpgrowth.return_value = pd.DataFrame()  # Empty DataFrame
        
        durations_df = pd.DataFrame({
            'duration': [10, 25, 35],
            'state': ['bullish weak', 'bearish strong', 'bullish weak']
        })
        
        apriori_result, fpgrowth_result = compute_state_using_association_rule_mining(durations_df)
        
        self.assertEqual(apriori_result, 0)
        self.assertEqual(fpgrowth_result, 0)

    # New test cases for improved error handling
    def test_compute_state_using_association_rule_mining_missing_columns(self):
        """Test association rule mining with missing required columns"""
        durations_df = pd.DataFrame({
            'duration': [10, 25, 35]  # Missing 'state' column
        })
        
        apriori_result, fpgrowth_result = compute_state_using_association_rule_mining(durations_df)
        
        self.assertEqual(apriori_result, 0)
        self.assertEqual(fpgrowth_result, 0)

    def test_compute_state_using_association_rule_mining_empty_dataframe(self):
        """Test association rule mining with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        apriori_result, fpgrowth_result = compute_state_using_association_rule_mining(empty_df)
        
        self.assertEqual(apriori_result, 0)
        self.assertEqual(fpgrowth_result, 0)

    def test_compute_state_using_association_rule_mining_single_row(self):
        """Test association rule mining with single row data"""
        single_row_df = pd.DataFrame({
            'duration': [10],
            'state': ['bullish weak']
        })
        
        apriori_result, fpgrowth_result = compute_state_using_association_rule_mining(single_row_df)
        
        self.assertIn(apriori_result, [0, 1, 2, 3])
        self.assertIn(fpgrowth_result, [0, 1, 2, 3])

    def test_hmm_kama_normal_case(self):
        """Test main hmm_kama function with normal data"""
        result = hmm_kama(self.sample_data, self.optimizing_params)
        
        self.assertIsInstance(result, HMM_KAMA)
        self.assertIn(result.next_state_with_hmm_kama, [0, 1, 2, 3])
        self.assertIn(result.current_state_of_state_using_std, [0, 1])
        self.assertIn(result.current_state_of_state_using_hmm, [0, 1])
        self.assertIn(result.state_high_probabilities_using_arm_apriori, [0, 1, 2, 3])
        self.assertIn(result.state_high_probabilities_using_arm_fpgrowth, [0, 1, 2, 3])
        self.assertIn(result.current_state_of_state_using_kmeans, [0, 1])

    def test_hmm_kama_invalid_data(self):
        """Test hmm_kama with invalid data"""
        # Test with None
        result = hmm_kama(None, self.optimizing_params)
        self.assertEqual(result, HMM_KAMA(0, 0, 0, 0, 0, 0))
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = hmm_kama(empty_df, self.optimizing_params)
        self.assertEqual(result, HMM_KAMA(0, 0, 0, 0, 0, 0))

    def test_hmm_kama_insufficient_data(self):
        """Test hmm_kama with insufficient data"""
        small_df = pd.DataFrame({'close': [100, 101, 102]})
        result = hmm_kama(small_df, self.optimizing_params)
        self.assertEqual(result, HMM_KAMA(0, 0, 0, 0, 0, 0))

    def test_hmm_kama_no_variance(self):
        """Test hmm_kama with zero variance data"""
        result = hmm_kama(self.constant_data, self.optimizing_params)
        self.assertEqual(result, HMM_KAMA(0, 0, 0, 0, 0, 0))

    def test_hmm_kama_with_missing_close(self):
        """Test hmm_kama without close column"""
        df_no_close = pd.DataFrame({'high': [100, 101, 102] * 20})
        result = hmm_kama(df_no_close, self.optimizing_params)
        self.assertEqual(result, HMM_KAMA(0, 0, 0, 0, 0, 0))

    @patch('signals.quant_models.hmm.hmm_kama.timeout_context')
    def test_hmm_kama_timeout(self, mock_timeout):
        """Test hmm_kama timeout handling"""
        mock_timeout.side_effect = TimeoutError("Operation timed out")
        
        result = hmm_kama(self.sample_data, self.optimizing_params)
        self.assertEqual(result, HMM_KAMA(0, 0, 0, 0, 0, 0))

    def test_hmm_kama_dataclass_structure(self):
        """Test HMM_KAMA dataclass structure"""
        result = HMM_KAMA(1, 0, 1, 2, 3, 0)
        
        self.assertEqual(result.next_state_with_hmm_kama, 1)
        self.assertEqual(result.current_state_of_state_using_std, 0)
        self.assertEqual(result.current_state_of_state_using_hmm, 1)
        self.assertEqual(result.state_high_probabilities_using_arm_apriori, 2)
        self.assertEqual(result.state_high_probabilities_using_arm_fpgrowth, 3)
        self.assertEqual(result.current_state_of_state_using_kmeans, 0)

    def test_hmm_kama_with_extreme_parameters(self):
        """Test hmm_kama with extreme parameter values"""
        extreme_params = MockOptimizingParameters()
        extreme_params.window_kama = 1000  # Very large window
        extreme_params.fast_kama = 100
        extreme_params.slow_kama = 1
        
        result = hmm_kama(self.sample_data, extreme_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_with_data_having_nans(self):
        """Test hmm_kama with DataFrame containing NaN values"""
        df_with_nans = self.sample_data.copy()
        df_with_nans.loc[10:20, 'close'] = np.nan
        
        result = hmm_kama(df_with_nans, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_very_large_dataset(self):
        """Test hmm_kama with very large dataset"""
        large_data = pd.DataFrame({
            'close': np.random.normal(100, 10, 10000),
            'high': np.random.normal(105, 10, 10000),
            'low': np.random.normal(95, 10, 10000)
        })
        
        result = hmm_kama(large_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_prevent_infinite_loop_decorator(self):
        """Test the prevent_infinite_loop decorator functionality"""
        # This is a bit tricky to test directly, but we can test multiple rapid calls
        results = []
        for _ in range(5):
            result = hmm_kama(self.sample_data, self.optimizing_params)
            results.append(result)
        
        # All results should be valid HMM_KAMA objects
        for result in results:
            self.assertIsInstance(result, HMM_KAMA)

    def test_timeout_context_manager(self):
        """Test timeout context manager with quick operation"""
        from signals.quant_models.hmm.hmm_kama import timeout_context
        
        # Test successful operation within timeout
        with timeout_context(5):
            result = 1 + 1
            self.assertEqual(result, 2)

    def test_hmm_kama_with_missing_columns(self):
        """Test hmm_kama with DataFrame missing expected columns"""
        minimal_df = pd.DataFrame({
            'close': np.random.normal(100, 10, 50)
        })
        
        result = hmm_kama(minimal_df, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_with_string_data(self):
        """Test hmm_kama with non-numeric data that should be filtered out"""
        mixed_df = self.sample_data.copy()
        mixed_df['text_column'] = ['text'] * len(mixed_df)
        mixed_df['mixed_column'] = [1, 'text', 2, 'more_text'] * (len(mixed_df) // 4)
        
        result = hmm_kama(mixed_df, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_state_mapping_coverage(self):
        """Test that all state mappings are properly handled"""
        from signals.quant_models.hmm.hmm_kama import STATE_MAPPING
        
        # Ensure all expected states are in mapping
        expected_states = ["bearish weak", "bullish weak", "bearish strong", "bullish strong"]
        for state in expected_states:
            self.assertIn(state, STATE_MAPPING)
            self.assertIn(STATE_MAPPING[state], [0, 1, 2, 3])

    # New test cases for improved error handling
    def test_hmm_kama_label_encoder_failure(self):
        """Test hmm_kama when LabelEncoder fails"""
        # Create data that might cause LabelEncoder issues
        problematic_data = pd.DataFrame({
            'close': np.random.normal(100, 10, 50),
            'high': np.random.normal(105, 10, 50),
            'low': np.random.normal(95, 10, 50)
        })
        
        result = hmm_kama(problematic_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_with_zero_price_range(self):
        """Test hmm_kama with data that has zero price range"""
        zero_range_data = pd.DataFrame({
            'close': [100.0] * 50,
            'high': [100.0] * 50,
            'low': [100.0] * 50
        })
        
        result = hmm_kama(zero_range_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_with_very_small_price_range(self):
        """Test hmm_kama with data that has very small price range"""
        small_range_data = pd.DataFrame({
            'close': [100.0 + i * 0.0001 for i in range(50)],
            'high': [100.1 + i * 0.0001 for i in range(50)],
            'low': [99.9 + i * 0.0001 for i in range(50)]
        })
        
        result = hmm_kama(small_range_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_with_extreme_outliers(self):
        """Test hmm_kama with extreme outlier values"""
        outlier_data = self.sample_data.copy()
        outlier_data.loc[0, 'close'] = 1e15  # Extreme outlier
        outlier_data.loc[1, 'close'] = -1e15  # Extreme negative outlier
        
        result = hmm_kama(outlier_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    # New test cases for improved KAMA calculation
    def test_calculate_kama_with_extreme_volatility(self):
        """Test KAMA with extreme volatility values"""
        prices = [100, 200, 50, 300, 25, 400, 10]
        result = calculate_kama(prices, window=3)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_calculate_kama_with_very_small_prices(self):
        """Test KAMA with very small price values"""
        prices = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        result = calculate_kama(prices, window=2)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_calculate_kama_with_mixed_invalid_values(self):
        """Test KAMA with mixed invalid values (NaN, Inf, normal)"""
        prices = [100, np.nan, 102, np.inf, 104, -np.inf, 106]
        result = calculate_kama(prices, window=3)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_calculate_kama_with_overflow_risk(self):
        """Test KAMA with values that could cause overflow"""
        prices = [1e10, 1e11, 1e12, 1e13, 1e14]
        result = calculate_kama(prices, window=2)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    def test_calculate_kama_with_underflow_risk(self):
        """Test KAMA with values that could cause underflow"""
        prices = [1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
        result = calculate_kama(prices, window=2)
        
        self.assertEqual(len(result), len(prices))
        self.assertTrue(np.isfinite(result).all())

    # Additional test cases for new improvements
    def test_hmm_kama_with_extreme_data(self):
        """Test hmm_kama with extreme data values"""
        result = hmm_kama(self.extreme_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_with_mixed_invalid_data(self):
        """Test hmm_kama with mixed invalid data"""
        result = hmm_kama(self.mixed_invalid_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def test_timeout_context_manager_success(self):
        """Test timeout context manager with successful operation"""
        from signals.quant_models.hmm.hmm_kama import timeout_context
        
        with timeout_context(5):
            result = 1 + 1
            self.assertEqual(result, 2)

    def test_timeout_context_manager_timeout(self):
        """Test timeout context manager with timeout"""
        from signals.quant_models.hmm.hmm_kama import timeout_context
        import time
        
        with self.assertRaises(TimeoutError):
            with timeout_context(0.1):  # Very short timeout
                time.sleep(0.2)  # Sleep longer than timeout

    def test_prevent_infinite_loop_decorator(self):
        """Test the prevent_infinite_loop decorator"""
        # This tests the decorator functionality indirectly
        # by calling hmm_kama multiple times rapidly
        results = []
        for _ in range(3):
            result = hmm_kama(self.sample_data, self.optimizing_params)
            results.append(result)
        
        # All results should be valid
        for result in results:
            self.assertIsInstance(result, HMM_KAMA)

    def test_hmm_kama_with_all_edge_cases_combined(self):
        """Test hmm_kama with all edge cases combined"""
        # Create data with multiple problematic characteristics
        problematic_data = pd.DataFrame({
            'close': [100] * 25 + [np.nan] * 25 + [np.inf] * 25 + [-np.inf] * 25,
            'high': [105] * 25 + [np.nan] * 25 + [np.inf] * 25 + [-np.inf] * 25,
            'low': [95] * 25 + [np.nan] * 25 + [np.inf] * 25 + [-np.inf] * 25
        })
        
        result = hmm_kama(problematic_data, self.optimizing_params)
        self.assertIsInstance(result, HMM_KAMA)

    def tearDown(self):
        """Clean up after each test"""
        # Clean up any temporary files or resources if needed
        pass

if __name__ == '__main__':
    unittest.main()