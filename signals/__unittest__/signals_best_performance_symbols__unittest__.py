import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timezone
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from signals.signals_best_performance_symbols import (
    signal_best_performance_symbols,
    get_top_performers_by_timeframe,
    get_worst_performers_by_timeframe,
    get_short_signal_candidates,
    _prepare_symbols,
    _analyze_timeframe_performance,
    _select_performers,
    _create_result_dict,
    _calculate_timeframe_statistics,
    _log_timeframe_stats,
    logging_performance_summary
)
from signals._components.BestPerformanceSymbols__class__PerformanceAnalyzer import PerformanceAnalyzer

class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for PerformanceAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = PerformanceAnalyzer()
        
        # Create sample DataFrame with realistic OHLCV data
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data with trend
        base_price = 100.0
        price_changes = np.random.normal(0.001, 0.02, 50)  # Small random changes
        prices: list[float] = [base_price]
        for change in price_changes[1:]:
            prices.append(float(prices[-1] * (1 + change)))
        
        self.sample_df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 50)  
        }, index=dates)
        
    def test_init_weights(self):
        """Test analyzer initialization with correct weights"""
        self.assertIsInstance(self.analyzer.long_weights, dict)
        self.assertIsInstance(self.analyzer.short_weights, dict)
        self.assertIsInstance(self.analyzer.timeframe_weights, dict)
        
        # Check that weights sum to approximately 1.0
        self.assertAlmostEqual(sum(self.analyzer.long_weights.values()), 1.0, places=2)
        self.assertAlmostEqual(sum(self.analyzer.short_weights.values()), 1.0, places=2)
        
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation"""
        metrics = self.analyzer.calculate_basic_metrics(self.sample_df, 20)
        
        # Check all required metrics are present
        required_keys = [
            'start_price', 'end_price', 'high_price', 'low_price',
            'total_return', 'volatility', 'avg_volume_usdt',
            'momentum_short', 'momentum_long', 'max_drawdown',
            'sharpe_ratio', 'rsi', 'volume_trend', 'distance_from_high'
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float, np.number))
        
        # Test data consistency
        self.assertGreaterEqual(metrics['high_price'], metrics['end_price'])
        self.assertLessEqual(metrics['low_price'], metrics['end_price'])
        self.assertGreaterEqual(metrics['rsi'], 0)
        self.assertLessEqual(metrics['rsi'], 100)
        
    def test_calculate_basic_metrics_insufficient_data(self):
        """Test basic metrics with insufficient data"""
        small_df = self.sample_df.head(5)
        metrics = self.analyzer.calculate_basic_metrics(small_df, 20)
        
        # Should still return metrics without crashing
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_return', metrics)
        
    def test_calculate_long_score(self):
        """Test LONG score calculation"""
        metrics = self.analyzer.calculate_basic_metrics(self.sample_df, 20)
        long_score = self.analyzer.calculate_long_score(metrics)
        
        self.assertIsInstance(long_score, (float, np.number))
        self.assertGreaterEqual(long_score, 0)
        self.assertLessEqual(long_score, 1)
        
    def test_calculate_short_score(self):
        """Test SHORT score calculation"""
        metrics = self.analyzer.calculate_basic_metrics(self.sample_df, 20)
        short_score = self.analyzer.calculate_short_score(metrics)
        
        self.assertIsInstance(short_score, (float, np.number))
        self.assertGreaterEqual(short_score, 0)
        self.assertLessEqual(short_score, 1)
        
    def test_calculate_performance_metrics(self):
        """Test comprehensive performance metrics calculation"""
        # Use a DataFrame with sufficient data and proper volume column
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')  # More data points
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100.0
        prices = [base_price]
        for i in range(99):
            change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        test_df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, 100)  # Higher volume
        }, index=dates)
        
        metrics = self.analyzer.calculate_performance_metrics(
            test_df, 'BTCUSDT', '1h', 20
        )
        
        # Check required fields
        required_fields = [
            'symbol', 'timeframe', 'composite_score', 'short_composite_score',
            'total_return', 'volatility', 'max_drawdown',
            'momentum_short', 'momentum_long', 'volume_trend', 'avg_volume_usdt',
            'rsi', 'distance_from_high', 'price_range'
        ]
        
        for field in required_fields:
            self.assertIn(field, metrics)
        
        self.assertEqual(metrics['symbol'], 'BTCUSDT')
        self.assertEqual(metrics['timeframe'], '1h')
        self.assertIsInstance(metrics['price_range'], dict)
        # Should have actual values, not defaults
        self.assertNotEqual(metrics['composite_score'], 0)
        self.assertNotEqual(metrics['volatility'], 999)

    def test_calculate_performance_metrics_insufficient_data(self):
        """Test performance metrics with insufficient data"""
        small_df = self.sample_df.head(5)
        metrics = self.analyzer.calculate_performance_metrics(
            small_df, 'BTCUSDT', '1h', 20
        )
        
        # Should return default values without crashing
        self.assertEqual(metrics['composite_score'], 0)
        self.assertEqual(metrics['short_composite_score'], 0)
        self.assertEqual(metrics['volatility'], 999)
        
    def test_calculate_overall_scores(self):
        """Test overall scores calculation across timeframes"""
        symbol_scores = {
            'BTCUSDT': {'1h': 0.8, '4h': 0.7, '1d': 0.6},
            'ETHUSDT': {'1h': 0.6, '4h': 0.8, '1d': 0.7},
            'ADAUSDT': {'1h': 0.5, '4h': 0.4, '1d': 0.3}
        }
        
        timeframes = ['1h', '4h', '1d']
        results = self.analyzer.calculate_overall_scores(symbol_scores, timeframes)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for result in results:
            self.assertIn('symbol', result)
            self.assertIn('composite_score', result)
            self.assertIn('timeframe_scores', result)
            self.assertIn('score_consistency', result)
        
        # Results should be sorted by composite score (descending)
        scores = [r['composite_score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
    def test_momentum_calculation(self):
        """Test momentum calculation static method"""
        prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])
        momentum = PerformanceAnalyzer._calculate_momentum(prices, 3, 6)
        
        self.assertIsInstance(momentum, (float, np.number))
        
        # Test with insufficient data
        short_prices = pd.Series([100, 102])
        momentum_short = PerformanceAnalyzer._calculate_momentum(short_prices, 3, 6)
        self.assertEqual(momentum_short, 0)
        
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Test with declining prices
        declining_prices = pd.Series([100, 95, 90, 85, 80, 75])
        drawdown = PerformanceAnalyzer._calculate_max_drawdown(declining_prices)
        self.assertLess(drawdown, 0)  # Should be negative
        
        # Test with single price
        single_price = pd.Series([100])
        drawdown_single = PerformanceAnalyzer._calculate_max_drawdown(single_price)
        self.assertEqual(drawdown_single, 0.0)
        
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        # Test with trending up prices
        up_prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118] * 2)
        rsi_up = PerformanceAnalyzer._calculate_simple_rsi(up_prices, 14)
        self.assertGreater(rsi_up, 50)  # Should be above 50 for uptrend
        
        # Test with insufficient data
        short_prices = pd.Series([100, 102])
        rsi_short = PerformanceAnalyzer._calculate_simple_rsi(short_prices, 14)
        self.assertEqual(rsi_short, 50)  # Default value

class TestMainFunction(unittest.TestCase):
    """Test cases for main signal_best_performance_pairs function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_processor = Mock()
        self.mock_processor.get_symbols_list_by_quote_usdt.return_value = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT'
        ]
        
        # Create sample preloaded data
        self.sample_preloaded_data = self._create_sample_preloaded_data()
        
    def _create_sample_preloaded_data(self):
        """Create sample preloaded data for testing"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        timeframes = ['1h', '4h', '1d']
        
        data = {}
        np.random.seed(42)
        
        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                # Generate different trends for different symbols
                if symbol == 'BTCUSDT':
                    trend = 0.002  # Uptrend
                elif symbol == 'ETHUSDT':
                    trend = 0.001  # Mild uptrend
                else:
                    trend = -0.001  # Downtrend
                
                dates = pd.date_range('2024-01-01', periods=50, freq='1h')
                base_price = 100.0  # Initialize as float
                prices: list[float] = [base_price] # Explicit type hint for clarity
                
                for i in range(49):
                    change = np.random.normal(trend, 0.02)
                    prices.append(prices[-1] * (1 + change))
                
                data[symbol][tf] = pd.DataFrame({
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                    'close': prices,
                    'volume': np.random.uniform(10000000, 50000000, 50)  
                }, index=dates)
        
        return data
    
    @patch('signals.signals_best_performance_symbols.logger')
    def test_signal_best_performance_pairs_success(self, mock_logger):
        """Test successful execution of main function"""
        result = signal_best_performance_symbols(
            processor=self.mock_processor,
            symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            timeframes=['1h', '4h'],
            performance_period=20,
            top_percentage=0.5,
            include_short_signals=True,
            preloaded_data=self.sample_preloaded_data
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('best_performers', result)
        self.assertIn('worst_performers', result)
        self.assertIn('timeframe_analysis', result)
        self.assertIn('summary', result)
        
        # Check summary fields
        summary = result['summary']
        required_summary_fields = [
            'total_symbols_analyzed', 'timeframes_analyzed',
            'top_performers_count', 'worst_performers_count',
            'analysis_timestamp', 'include_short_signals'
        ]
        
        for field in required_summary_fields:
            self.assertIn(field, summary)
    
    def test_signal_best_performance_pairs_no_preloaded_data(self):
        """Test function with no preloaded data"""
        result = signal_best_performance_symbols(
            processor=self.mock_processor,
            preloaded_data=None
        )
        
        self.assertEqual(result, {})
    
    def test_signal_best_performance_pairs_empty_symbols(self):
        """Test function with empty symbols list"""
        self.mock_processor.get_symbols_list_by_quote_usdt.return_value = []
        
        result = signal_best_performance_symbols(
            processor=self.mock_processor,
            preloaded_data=self.sample_preloaded_data
        )
        
        self.assertEqual(result, {})

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_analysis_result = {
            'best_performers': [
                {'symbol': 'BTCUSDT', 'composite_score': 0.8},
                {'symbol': 'ETHUSDT', 'composite_score': 0.7}
            ],
            'worst_performers': [
                {'symbol': 'ADAUSDT', 'composite_score': 0.3, 'avg_short_score': 0.7},
                {'symbol': 'DOTUSDT', 'composite_score': 0.2, 'avg_short_score': 0.8}
            ],
            'timeframe_analysis': {
                '1h': {
                    'symbol_metrics': [
                        {'symbol': 'BTCUSDT', 'composite_score': 0.9},
                        {'symbol': 'ETHUSDT', 'composite_score': 0.8},
                        {'symbol': 'ADAUSDT', 'composite_score': 0.3}
                    ]
                }
            }
        }
    
    def test_get_top_performers_by_timeframe(self):
        """Test getting top performers for specific timeframe"""
        top_performers = get_top_performers_by_timeframe(
            self.sample_analysis_result, '1h', top_n=2
        )
        
        self.assertEqual(len(top_performers), 2)
        self.assertEqual(top_performers[0]['symbol'], 'BTCUSDT')
        self.assertEqual(top_performers[1]['symbol'], 'ETHUSDT')
    
    def test_get_top_performers_invalid_timeframe(self):
        """Test getting top performers for invalid timeframe"""
        top_performers = get_top_performers_by_timeframe(
            self.sample_analysis_result, '5m', top_n=2
        )
        
        self.assertEqual(top_performers, [])
    
    def test_get_worst_performers_by_timeframe(self):
        """Test getting worst performers for specific timeframe"""
        worst_performers = get_worst_performers_by_timeframe(
            self.sample_analysis_result, '1h', top_n=1
        )
        
        self.assertEqual(len(worst_performers), 1)
        self.assertEqual(worst_performers[0]['symbol'], 'ADAUSDT')
    
    def test_get_short_signal_candidates(self):
        """Test getting SHORT signal candidates"""
        # Add timeframe_scores to worst_performers for testing
        analysis_result = self.sample_analysis_result.copy()
        analysis_result['worst_performers'][0]['timeframe_scores'] = {
            '1h': {'short_composite_score': 0.7},
            '4h': {'short_composite_score': 0.8}
        }
        
        candidates = get_short_signal_candidates(analysis_result, min_short_score=0.6)
        
        self.assertIsInstance(candidates, list)
        # Should contain candidates with avg_short_score >= 0.6
        
    def test_prepare_symbols_with_stable_coin_filtering(self):
        """Test symbol preparation with stablecoin filtering"""
        mock_processor = Mock()
        mock_processor.get_symbols_list_by_quote_usdt.return_value = [
            'BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BUSDUSDT', 'ADAUSDT'
        ]
        
        filtered_symbols = _prepare_symbols(mock_processor, None, exclude_stable_coins=True)
        
        # Should exclude USDCUSDT and BUSDUSDT
        expected_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        self.assertEqual(set(filtered_symbols), set(expected_symbols))
    
    def test_prepare_symbols_without_filtering(self):
        """Test symbol preparation without stablecoin filtering"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'USDCUSDT']
        
        result = _prepare_symbols(None, symbols, exclude_stable_coins=False)
        
        self.assertEqual(result, symbols)
    
    def test_select_performers(self):
        """Test performer selection"""
        overall_scores = [
            {'symbol': 'BTCUSDT', 'composite_score': 0.9},
            {'symbol': 'ETHUSDT', 'composite_score': 0.8},
            {'symbol': 'ADAUSDT', 'composite_score': 0.7},
            {'symbol': 'DOTUSDT', 'composite_score': 0.6},
            {'symbol': 'LINKUSDT', 'composite_score': 0.5}
        ]
        
        best, worst = _select_performers(
            overall_scores, top_percentage=0.4, worst_percentage=0.4, include_short_signals=True
        )
        
        # Should select top 40% (2 symbols) and bottom 40% (2 symbols)
        self.assertEqual(len(best), 2)
        self.assertEqual(len(worst), 2)
        self.assertEqual(best[0]['symbol'], 'BTCUSDT')
        self.assertEqual(worst[0]['symbol'], 'LINKUSDT')  # Worst first after reverse

class TestPrivateFunctions(unittest.TestCase):
    """Test cases for private helper functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.symbol_metrics = [
            {
                'symbol': 'BTCUSDT',
                'composite_score': 0.85,
                'total_return': 0.15,
                'volatility': 0.05,
                'short_composite_score': 0.35
            },
            {
                'symbol': 'ETHUSDT',
                'composite_score': 0.75,
                'total_return': 0.10,
                'volatility': 0.06,
                'short_composite_score': 0.45
            },
            {
                'symbol': 'SOLUSDT',
                'composite_score': 0.65,
                'total_return': 0.08,
                'volatility': 0.07,
                'short_composite_score': 0.55
            }
        ]
        
    def test_calculate_timeframe_statistics(self):
        """Test calculation of timeframe statistics"""
        stats = _calculate_timeframe_statistics(self.symbol_metrics, True)
        
        # Check presence of required statistics
        required_keys = [
            'symbols_processed', 'avg_score', 'median_score', 'avg_return', 
            'avg_volatility', 'avg_short_score', 'top_performer', 'worst_performer'
        ]
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check specific values
        self.assertEqual(stats['symbols_processed'], 3)
        self.assertAlmostEqual(stats['avg_score'], 0.75, places=2)
        self.assertAlmostEqual(stats['median_score'], 0.75, places=2)
        self.assertAlmostEqual(stats['avg_return'], 0.11, places=2)
        self.assertAlmostEqual(stats['avg_volatility'], 0.06, places=2)
        self.assertAlmostEqual(stats['avg_short_score'], 0.45, places=2)
        
        # Check top and worst performers
        self.assertEqual(stats['top_performer']['symbol'], 'BTCUSDT')
        self.assertEqual(stats['worst_performer']['symbol'], 'SOLUSDT')
    
    def test_calculate_timeframe_statistics_empty(self):
        """Test calculation of timeframe statistics with empty input"""
        stats = _calculate_timeframe_statistics([], True)
        
        self.assertEqual(stats['symbols_processed'], 0)
        self.assertEqual(stats['avg_score'], 0)
        self.assertEqual(stats['median_score'], 0)
        self.assertEqual(stats['avg_return'], 0)
        self.assertEqual(stats['avg_volatility'], 0)
        self.assertEqual(stats['avg_short_score'], 0)
        self.assertIsNone(stats['top_performer'])
        self.assertIsNone(stats['worst_performer'])
    
    def test_log_timeframe_stats(self):
        """Test logging of timeframe statistics"""
        with patch('signals.signals_best_performance_symbols.logger') as mock_logger:
            stats = {
                'processed': 10,
                'no_timeframe': 2,
                'insufficient_data': 3,
                'low_volume': 1
            }
            
            _log_timeframe_stats('1h', stats)
            
            # Verify logger calls - should be 5 calls total (1 header + 4 stats)
            self.assertEqual(mock_logger.performance.call_count, 5)
            mock_logger.performance.assert_any_call("Timeframe 1h results:")
            mock_logger.performance.assert_any_call("  - Processed successfully: 10")
            mock_logger.performance.assert_any_call("  - Skipped (no timeframe data): 2")
            mock_logger.performance.assert_any_call("  - Skipped (insufficient data): 3")
            mock_logger.performance.assert_any_call("  - Skipped (low volume): 1")

    def test_create_result_dict(self):
        """Test creation of the result dictionary"""
        best_performers = [{'symbol': 'BTCUSDT', 'composite_score': 0.85}]
        worst_performers = [{'symbol': 'ADAUSDT', 'composite_score': 0.25}]
        timeframe_results = {'1h': {'symbol_metrics': []}}
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        timeframes = ['1h', '4h']
        
        result = _create_result_dict(
            best_performers, worst_performers, timeframe_results,
            symbols, timeframes, 0.3, 0.3, 24, 1000000, True
        )
        
        # Check result structure
        self.assertIn('best_performers', result)
        self.assertIn('worst_performers', result)
        self.assertIn('timeframe_analysis', result)
        self.assertIn('summary', result)
        
        # Check summary fields
        summary = result['summary']
        self.assertEqual(summary['total_symbols_analyzed'], 3)
        self.assertEqual(summary['timeframes_analyzed'], ['1h', '4h'])
        self.assertEqual(summary['top_performers_count'], 1)
        self.assertEqual(summary['worst_performers_count'], 1)
        self.assertEqual(summary['top_percentage'], 0.3)
        self.assertEqual(summary['short_percentage'], 0.3)
        self.assertEqual(summary['performance_period'], 24)
        self.assertEqual(summary['min_volume_usdt'], 1000000)
        self.assertEqual(summary['include_short_signals'], True)

class TestAnalyzeTimeframePerformance(unittest.TestCase):
    """Test cases for _analyze_timeframe_performance function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = PerformanceAnalyzer()
        
        # Create sample symbol data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        
        # Sample data for BTC (uptrend)
        btc_prices = [40000.0]
        for i in range(99):
            btc_prices.append(btc_prices[-1] * (1 + np.random.normal(0.002, 0.01)))
        
        # Sample data for ETH (downtrend)
        eth_prices = [3000.0]
        for i in range(99):
            eth_prices.append(eth_prices[-1] * (1 + np.random.normal(-0.001, 0.01)))
        
        self.symbol_data = {
            'BTCUSDT': {
                '1h': pd.DataFrame({
                    'open': btc_prices,
                    'high': [p * 1.01 for p in btc_prices],
                    'low': [p * 0.99 for p in btc_prices],
                    'close': btc_prices,
                    'volume': [np.random.uniform(5000000, 10000000) for _ in range(100)]
                }, index=dates)
            },
            'ETHUSDT': {
                '1h': pd.DataFrame({
                    'open': eth_prices,
                    'high': [p * 1.01 for p in eth_prices],
                    'low': [p * 0.99 for p in eth_prices],
                    'close': eth_prices,
                    'volume': [np.random.uniform(3000000, 8000000) for _ in range(100)]
                }, index=dates)
            },
            'LOWVOLUME': {
                '1h': pd.DataFrame({
                    'open': [100] * 100,
                    'high': [101] * 100,
                    'low': [99] * 100,
                    'close': [100] * 100,
                    'volume': [5000] * 100
                }, index=dates)
            },
            'INSUFFICIENTDATA': {
                '1h': pd.DataFrame({
                    'open': [100, 101],
                    'high': [101, 102],
                    'low': [99, 100],
                    'close': [100, 101],
                    'volume': [1000000, 1000000]
                }, index=dates[:2])
            }
        }
    
    @patch('signals.signals_best_performance_symbols.logger')
    @patch('signals.signals_best_performance_symbols.tqdm')
    def test_analyze_timeframe_performance(self, mock_tqdm, mock_logger):
        """Test analysis of timeframe performance"""
        # Mock tqdm to avoid progressbar in tests
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm.return_value
        
        result = _analyze_timeframe_performance(
            self.analyzer,
            self.symbol_data,
            '1h',
            performance_period=20,
            min_volume_usdt=1000000,
            analyze_for_short=True
        )
        
        # Check result structure
        self.assertIn('timeframe', result)
        self.assertIn('symbol_metrics', result)
        self.assertIn('statistics', result)
        
        self.assertEqual(result['timeframe'], '1h')
        
        # Check what symbols were actually processed
        processed_symbols = [m['symbol'] for m in result['symbol_metrics']]
        
        # Should process BTCUSDT and ETHUSDT (high volume)
        self.assertIn('BTCUSDT', processed_symbols)
        self.assertIn('ETHUSDT', processed_symbols)
        
        # INSUFFICIENTDATA should be filtered out due to insufficient data
        self.assertNotIn('INSUFFICIENTDATA', processed_symbols)
        
        # LOWVOLUME should be filtered out due to low volume
        self.assertNotIn('LOWVOLUME', processed_symbols)
        
        # Check statistics
        stats = result['statistics']
        self.assertEqual(stats['symbols_processed'], 2)
        self.assertIsNotNone(stats['avg_score'])
        self.assertIsNotNone(stats['top_performer'])

class TestPrintPerformanceSummary(unittest.TestCase):
    """Test cases for print_performance_summary function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analysis_result = {
            'best_performers': [
                {'symbol': 'BTCUSDT', 'composite_score': 0.85, 'timeframe_scores': {'1h': 0.8, '4h': 0.9}},
                {'symbol': 'ETHUSDT', 'composite_score': 0.75, 'timeframe_scores': {'1h': 0.7, '4h': 0.8}}
            ],
            'worst_performers': [
                {'symbol': 'ADAUSDT', 'composite_score': 0.25, 'timeframe_scores': {'1h': 0.3, '4h': 0.2}}
            ],
            'summary': {
                'total_symbols_analyzed': 10,
                'timeframes_analyzed': ['1h', '4h', '1d'],
                'top_performers_count': 2,
                'worst_performers_count': 1,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
    
    @patch('signals.signals_best_performance_symbols.logger')
    def test_print_performance_summary(self, mock_logger):
        """Test performance summary printing"""
        logging_performance_summary(self.analysis_result)
        
        # Verify logger was called multiple times for analysis output
        self.assertGreater(mock_logger.analysis.call_count, 5)
        
        # Test with empty results
        mock_logger.reset_mock()
        logging_performance_summary({})
        
        # Should log a warning and not crash
        mock_logger.warning.assert_called_once()
    
    @patch('signals.signals_best_performance_symbols.logger')
    def test_print_performance_section_calls(self, mock_logger):
        """Test that performer sections are printed correctly"""
        logging_performance_summary(self.analysis_result)
        
        # Verify logger.analysis was called for section headers
        # The actual messages include newlines and specific formatting
        mock_logger.analysis.assert_any_call("\n🟢 TOP 2 PERFORMERS (LONG SIGNALS):")
        mock_logger.analysis.assert_any_call("\n🔴 BOTTOM 1 PERFORMERS (SHORT SIGNALS):")

if __name__ == '__main__':
    # Configure unittest to run with verbose output
    unittest.main(verbosity=2, buffer=True)
