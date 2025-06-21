import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from signals._components.BestPerformanceSymbols__class__PerformanceAnalyzer import PerformanceAnalyzer

class TestPerformanceAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = PerformanceAnalyzer()
        
        # Create sample DataFrame with OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        self.sample_df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.001),
            'high': prices * (1 + abs(np.random.randn(100)) * 0.002),
            'low': prices * (1 - abs(np.random.randn(100)) * 0.002),
            'close': prices,
            'volume': 1000000 + np.random.randn(100) * 100000
        }, index=dates)
        
        # Ensure high >= close >= low and high >= open >= low
        self.sample_df['high'] = np.maximum(self.sample_df[['open', 'close']].max(axis=1), self.sample_df['high'])
        self.sample_df['low'] = np.minimum(self.sample_df[['open', 'close']].min(axis=1), self.sample_df['low'])
    
    def test_init(self):
        """Test PerformanceAnalyzer initialization"""
        self.assertIsInstance(self.analyzer.long_weights, dict)
        self.assertIsInstance(self.analyzer.short_weights, dict)
        self.assertIsInstance(self.analyzer.timeframe_weights, dict)
        
        # Check that weights sum to 1.0 (approximately)
        self.assertAlmostEqual(sum(self.analyzer.long_weights.values()), 1.0, places=2)
        self.assertAlmostEqual(sum(self.analyzer.short_weights.values()), 1.0, places=2)
        self.assertAlmostEqual(sum(self.analyzer.timeframe_weights.values()), 1.0, places=2)
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation"""
        period = 30
        metrics = self.analyzer.calculate_basic_metrics(self.sample_df, period)
        
        # Check all required keys are present
        required_keys = [
            'start_price', 'end_price', 'high_price', 'low_price',
            'total_return', 'volatility', 'avg_volume_usdt',
            'momentum_short', 'momentum_long', 'max_drawdown',
            'sharpe_ratio', 'rsi', 'volume_trend', 'distance_from_high'
        ]
        for key in required_keys:
            self.assertIn(key, metrics)
        
        # Check data types and reasonable values
        self.assertIsInstance(metrics['total_return'], (int, float))
        self.assertIsInstance(metrics['volatility'], (int, float))
        self.assertGreaterEqual(metrics['volatility'], 0)
        self.assertGreaterEqual(metrics['avg_volume_usdt'], 0)
        self.assertGreaterEqual(metrics['rsi'], 0)
        self.assertLessEqual(metrics['rsi'], 100)
        self.assertGreaterEqual(metrics['distance_from_high'], 0)
    
    def test_calculate_basic_metrics_insufficient_data(self):
        """Test basic metrics with insufficient data"""
        small_df = self.sample_df.head(5)
        period = 30
        metrics = self.analyzer.calculate_basic_metrics(small_df, period)
        
        # Should still return metrics but with limited data
        self.assertIn('total_return', metrics)
        self.assertIn('volatility', metrics)
    
    def test_calculate_long_score(self):
        """Test long score calculation"""
        # Create sample metrics
        sample_metrics = {
            'total_return': 0.1,
            'sharpe_ratio': 1.5,
            'momentum_short': 0.05,
            'momentum_long': 0.03,
            'volume_trend': 0.02,
            'volatility': 0.2,
            'max_drawdown': -0.05
        }
        
        score = self.analyzer.calculate_long_score(sample_metrics)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_calculate_long_score_extreme_values(self):
        """Test long score with extreme values"""
        # Test with very positive metrics
        positive_metrics = {
            'total_return': 2.0,
            'sharpe_ratio': 5.0,
            'momentum_short': 1.0,
            'momentum_long': 1.0,
            'volume_trend': 2.0,
            'volatility': 0.01,
            'max_drawdown': 0.0
        }
        
        positive_score = self.analyzer.calculate_long_score(positive_metrics)
        self.assertGreaterEqual(positive_score, 0)
        self.assertLessEqual(positive_score, 1)
        
        # Test with very negative metrics
        negative_metrics = {
            'total_return': -1.0,
            'sharpe_ratio': -3.0,
            'momentum_short': -1.0,
            'momentum_long': -1.0,
            'volume_trend': -2.0,
            'volatility': 5.0,
            'max_drawdown': -0.5
        }
        
        negative_score = self.analyzer.calculate_long_score(negative_metrics)
        self.assertGreaterEqual(negative_score, 0)
        self.assertLessEqual(negative_score, 1)
        self.assertLess(negative_score, positive_score)
    
    def test_calculate_short_score(self):
        """Test short score calculation"""
        sample_metrics = {
            'total_return': -0.1,
            'momentum_short': -0.05,
            'momentum_long': -0.03,
            'rsi': 75,
            'volume_trend': 0.1,
            'distance_from_high': 0.15,
            'volatility': 0.3
        }
        
        score = self.analyzer.calculate_short_score(sample_metrics)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_calculate_short_score_extreme_values(self):
        """Test short score with extreme values"""
        # Test with values favorable for short signals
        bearish_metrics = {
            'total_return': -0.5,
            'momentum_short': -0.3,
            'momentum_long': -0.2,
            'rsi': 90,
            'volume_trend': 1.0,
            'distance_from_high': 0.3,
            'volatility': 2.0
        }
        
        bearish_score = self.analyzer.calculate_short_score(bearish_metrics)
        
        # Test with values unfavorable for short signals
        bullish_metrics = {
            'total_return': 0.5,
            'momentum_short': 0.3,
            'momentum_long': 0.2,
            'rsi': 20,
            'volume_trend': -0.5,
            'distance_from_high': 0.0,
            'volatility': 0.1
        }
        
        bullish_score = self.analyzer.calculate_short_score(bullish_metrics)
        self.assertGreater(bearish_score, bullish_score)
    
    def test_calculate_performance_metrics(self):
        """Test comprehensive performance metrics calculation"""
        symbol = "BTCUSDT"
        timeframe = "1h"
        period = 50
        
        metrics = self.analyzer.calculate_performance_metrics(
            self.sample_df, symbol, timeframe, period
        )
        
        # Check required keys
        required_keys = [
            'symbol', 'timeframe', 'composite_score', 'short_composite_score',
            'total_return', 'volatility', 'sharpe_ratio', 'max_drawdown',
            'momentum_short', 'momentum_long', 'bearish_momentum_short',
            'bearish_momentum_long', 'volume_trend', 'avg_volume_usdt',
            'rsi', 'distance_from_high', 'price_range'
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics)
        
        # Check values
        self.assertEqual(metrics['symbol'], symbol)
        self.assertEqual(metrics['timeframe'], timeframe)
        self.assertIsInstance(metrics['composite_score'], (int, float))
        self.assertIsInstance(metrics['short_composite_score'], (int, float))
        self.assertIsInstance(metrics['price_range'], dict)
        
        # Check price range structure
        price_range_keys = ['start', 'end', 'high', 'low']
        for key in price_range_keys:
            self.assertIn(key, metrics['price_range'])
    
    def test_calculate_performance_metrics_insufficient_data(self):
        """Test performance metrics with insufficient data"""
        small_df = self.sample_df.head(5)
        symbol = "TESTCOIN"
        timeframe = "1h"
        period = 30
        
        metrics = self.analyzer.calculate_performance_metrics(
            small_df, symbol, timeframe, period
        )
        
        # Should return default values for error cases
        self.assertEqual(metrics['symbol'], symbol)
        self.assertEqual(metrics['timeframe'], timeframe)
        self.assertEqual(metrics['composite_score'], 0)
        self.assertEqual(metrics['short_composite_score'], 0)
        self.assertEqual(metrics['volatility'], 999)
    
    def test_calculate_overall_scores(self):
        """Test overall scores calculation across timeframes"""
        symbol_scores = {
            'BTCUSDT': {'1h': 0.8, '4h': 0.7, '1d': 0.6},
            'ETHUSDT': {'1h': 0.6, '4h': 0.8, '1d': 0.7},
            'ADAUSDT': {'1h': 0.5, '4h': 0.5}  # Missing 1d data
        }
        timeframes = ['1h', '4h', '1d']
        
        results = self.analyzer.calculate_overall_scores(symbol_scores, timeframes)
        
        self.assertIsInstance(results, list)
        
        # Should only include symbols with complete data
        complete_symbols = [r['symbol'] for r in results]
        self.assertIn('BTCUSDT', complete_symbols)
        self.assertIn('ETHUSDT', complete_symbols)
        self.assertNotIn('ADAUSDT', complete_symbols)
        
        # Check result structure
        for result in results:
            required_keys = ['symbol', 'composite_score', 'timeframe_scores', 'score_consistency']
            for key in required_keys:
                self.assertIn(key, result)
            
            self.assertIsInstance(result['composite_score'], (int, float))
            self.assertIsInstance(result['timeframe_scores'], dict)
            self.assertIsInstance(result['score_consistency'], (int, float))
        
        # Results should be sorted by composite score (descending)
        scores = [r['composite_score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_calculate_overall_scores_empty_input(self):
        """Test overall scores with empty input"""
        empty_scores = {}
        timeframes = ['1h', '4h', '1d']
        
        results = self.analyzer.calculate_overall_scores(empty_scores, timeframes)
        self.assertEqual(results, [])
    
    def test_calculate_momentum(self):
        """Test momentum calculation static method"""
        # Create test series with clear trend
        upward_trend = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        downward_trend = pd.Series([110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100])
        
        upward_momentum = PerformanceAnalyzer._calculate_momentum(upward_trend, 3, 6)
        downward_momentum = PerformanceAnalyzer._calculate_momentum(downward_trend, 3, 6)
        
        self.assertGreater(upward_momentum, 0)
        self.assertLess(downward_momentum, 0)
        
        # Test with insufficient data
        short_series = pd.Series([100, 101])
        momentum = PerformanceAnalyzer._calculate_momentum(short_series, 5, 10)
        self.assertEqual(momentum, 0)
    
    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation static method"""
        # Test with declining prices
        declining_prices = pd.Series([100, 90, 80, 70, 85, 75, 60])
        drawdown = PerformanceAnalyzer._calculate_max_drawdown(declining_prices)
        
        self.assertLessEqual(drawdown, 0)  # Drawdown should be negative or zero
        
        # Test with constantly rising prices
        rising_prices = pd.Series([100, 105, 110, 115, 120])
        drawdown_rising = PerformanceAnalyzer._calculate_max_drawdown(rising_prices)
        
        self.assertGreaterEqual(drawdown_rising, -0.1)  # Should be close to 0
        
        # Test with single price
        single_price = pd.Series([100])
        drawdown_single = PerformanceAnalyzer._calculate_max_drawdown(single_price)
        self.assertEqual(drawdown_single, 0.0)
        
        # Test with empty series
        empty_series = pd.Series([])
        drawdown_empty = PerformanceAnalyzer._calculate_max_drawdown(empty_series)
        self.assertEqual(drawdown_empty, 0.0)
    
    def test_calculate_simple_rsi(self):
        """Test RSI calculation static method"""
        # Create test series with known pattern
        test_prices = pd.Series([44, 44.5, 44, 43.5, 44, 44.5, 44, 43.5, 44, 44.5, 
                                44, 43.5, 44, 44.5, 44, 44.5, 44, 43.5, 44, 44.5])
        
        rsi = PerformanceAnalyzer._calculate_simple_rsi(test_prices, period=14)
        
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        self.assertIsInstance(rsi, (int, float))
        
        # Test with insufficient data
        short_prices = pd.Series([100, 101, 102])
        rsi_short = PerformanceAnalyzer._calculate_simple_rsi(short_prices, period=14)
        self.assertEqual(rsi_short, 50)  # Default value
        
        # Test with constant prices (should handle division by zero)
        constant_prices = pd.Series([100] * 20)
        rsi_constant = PerformanceAnalyzer._calculate_simple_rsi(constant_prices, period=14)
        self.assertEqual(rsi_constant, 50)  # Default for no change
    
    @patch('signals._components.BestPerformanceSymbols__class__PerformanceAnalyzer.logger')
    def test_error_handling_in_calculate_overall_scores(self, mock_logger):
        """Test error handling in calculate_overall_scores"""
        # Create valid input data but mock numpy.std to raise an exception
        valid_scores = {
            'BTCUSDT': {'1h': 0.8, '4h': 0.7, '1d': 0.6},
            'ETHUSDT': {'1h': 0.6, '4h': 0.8, '1d': 0.7}
        }
        timeframes = ['1h', '4h', '1d']
        
        # Patch numpy.std to raise an exception during score consistency calculation
        with patch('numpy.std', side_effect=Exception("Test error")):
            results = self.analyzer.calculate_overall_scores(valid_scores, timeframes)
            self.assertEqual(results, [])
            mock_logger.error.assert_called_once()
    
    def test_error_handling_in_calculate_performance_metrics(self):
        """Test error handling in calculate_performance_metrics"""
        # Test with invalid DataFrame that will cause an error in basic metrics calculation
        invalid_df = pd.DataFrame({'invalid_column': [1, 2, 3]})
        symbol = "TESTCOIN"
        timeframe = "1h"
        period = 30
        
        with patch('signals._components.BestPerformanceSymbols__class__PerformanceAnalyzer.logger') as mock_logger:
            metrics = self.analyzer.calculate_performance_metrics(
                invalid_df, symbol, timeframe, period
            )
            
            # Should return default error values
            self.assertEqual(metrics['symbol'], symbol)
            self.assertEqual(metrics['timeframe'], timeframe)
            self.assertEqual(metrics['composite_score'], 0)
            self.assertEqual(metrics['short_composite_score'], 0)
            self.assertEqual(metrics['volatility'], 999)
            mock_logger.debug.assert_called_once()
    
    def test_weights_consistency(self):
        """Test that all weight dictionaries are properly structured"""
        # Check long weights
        self.assertIn('return', self.analyzer.long_weights)
        self.assertIn('sharpe', self.analyzer.long_weights)
        self.assertIn('momentum_short', self.analyzer.long_weights)
        
        # Check short weights
        self.assertIn('negative_return', self.analyzer.short_weights)
        self.assertIn('bearish_momentum_short', self.analyzer.short_weights)
        self.assertIn('high_rsi', self.analyzer.short_weights)
        
        # Check timeframe weights
        self.assertIn('1h', self.analyzer.timeframe_weights)
        self.assertIn('4h', self.analyzer.timeframe_weights)
        self.assertIn('1d', self.analyzer.timeframe_weights)
        
        # All weights should be positive
        for weight in self.analyzer.long_weights.values():
            self.assertGreater(weight, 0)
        for weight in self.analyzer.short_weights.values():
            self.assertGreater(weight, 0)
        for weight in self.analyzer.timeframe_weights.values():
            self.assertGreater(weight, 0)


if __name__ == '__main__':
    unittest.main()
