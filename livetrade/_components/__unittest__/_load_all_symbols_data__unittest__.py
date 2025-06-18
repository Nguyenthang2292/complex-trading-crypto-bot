import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import threading
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
# Import the function to test
from livetrade._components._load_all_symbols_data import (
    load_symbol_data,
    load_all_symbols_data,
    _create_progress_bar,
    _wait_for_data_with_progress,
    _retry_with_backoff,
    _calculate_memory_usage
)

class MockProcessor:
    """Mock data processor for testing."""
    
    def __init__(self, data_to_return=None, delay=0, fail_symbols=None):
        self.df_cache = {}
        self.data_to_return = data_to_return or {}
        self.delay = delay
        self.fail_symbols = fail_symbols or []
        self.call_count = 0
        self.lock = threading.Lock()
        
    def get_historic_data_by_symbol(self, symbol, timeframe):
        """Mock implementation that simulates getting historic data."""
        self.call_count += 1
        
        # Simulate network delay
        if self.delay > 0:
            time.sleep(self.delay)
            
        # Simulate failures for specific symbols
        if symbol in self.fail_symbols:
            raise Exception(f"Simulated failure for {symbol}")
            
        # Generate cache key
        cache_key = (symbol, timeframe)
        
        # If data exists in the data_to_return dictionary, use it
        if symbol in self.data_to_return and timeframe in self.data_to_return[symbol]:
            with self.lock:
                self.df_cache[cache_key] = self.data_to_return[symbol][timeframe]
        # Otherwise create empty DataFrame
        else:
            with self.lock:
                # Create sample data with basic OHLCV structure
                dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
                self.df_cache[cache_key] = pd.DataFrame({
                    'open': [100 + i for i in range(100)],
                    'high': [105 + i for i in range(100)],
                    'low': [95 + i for i in range(100)],
                    'close': [101 + i for i in range(100)],
                    'volume': [1000 + i*10 for i in range(100)]
                }, index=dates)
                
        return True

class TestLoadSymbolData(unittest.TestCase):
    """Test cases for load_symbol_data function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        self.sample_df = pd.DataFrame({
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [95 + i for i in range(100)],
            'close': [101 + i for i in range(100)],
            'volume': [1000 + i*10 for i in range(100)]
        }, index=dates)
        
        # Create sample data for multiple timeframes
        self.multi_tf_data = {
            'BTCUSDT': {
                '1m': self.sample_df.copy(),
                '5m': self.sample_df.copy(),
                '1h': self.sample_df.copy()
            },
            'ETHUSDT': {
                '1m': self.sample_df.copy(),
                '5m': self.sample_df.copy(),
                '1h': self.sample_df.copy()
            }
        }
        
    @patch('livetrade._components._load_all_symbols_data._wait_for_data_with_progress')
    def test_load_symbol_data_single_timeframe(self, mock_wait):
        """Test loading data for a single timeframe."""
        # Setup
        mock_wait.return_value = self.sample_df.copy()
        processor = MockProcessor(data_to_return=self.multi_tf_data)
        
        # Execute
        result = load_symbol_data(processor, "BTCUSDT", ["1h"], load_multi_timeframes=False)
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsNotNone(result)
        if result is not None and isinstance(result, pd.DataFrame):
            self.assertEqual(len(result), 100)
            self.assertTrue(all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']))
    
    @patch('livetrade._components._load_all_symbols_data._wait_for_data_with_progress')
    def test_load_symbol_data_multi_timeframe(self, mock_wait):
        """Test loading data for multiple timeframes."""
        # Setup
        mock_wait.return_value = self.sample_df.copy()
        processor = MockProcessor(data_to_return=self.multi_tf_data)
        
        # Execute
        result = load_symbol_data(processor, "BTCUSDT", ["1m", "5m", "1h"], load_multi_timeframes=True)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(len(result), 3)
            self.assertTrue(all(tf in result for tf in ["1m", "5m", "1h"]))
            for tf, df in result.items():
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(len(df), 100)
    
    def test_load_symbol_data_invalid_inputs(self):
        """Test loading data with invalid inputs."""
        processor = MockProcessor()
        
        # Test with empty symbol
        result = load_symbol_data(processor, "", ["1h"])
        self.assertIsNone(result)
        
        # Test with invalid processor
        result = load_symbol_data({}, "BTCUSDT", ["1h"])
        self.assertIsNone(result)
        
        # Test with invalid timeframes
        result = load_symbol_data(processor, "BTCUSDT", None)
        self.assertIsNotNone(result)  # Should use default timeframes
        
        result = load_symbol_data(processor, "BTCUSDT", [])
        self.assertIsNone(result)
    
    @patch('livetrade._components._load_all_symbols_data._wait_for_data_with_progress')
    def test_load_symbol_data_error_handling(self, mock_wait):
        """Test error handling in load_symbol_data."""
        # Setup
        mock_wait.side_effect = Exception("Test error")
        processor = MockProcessor()
        
        # Execute
        result = load_symbol_data(processor, "BTCUSDT", ["1h"], load_multi_timeframes=False)
        
        # Assert
        self.assertIsNone(result)


class TestLoadAllSymbolsData(unittest.TestCase):
    """Test cases for load_all_symbols_data function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        self.sample_df = pd.DataFrame({
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [95 + i for i in range(100)],
            'close': [101 + i for i in range(100)],
            'volume': [1000 + i*10 for i in range(100)]
        }, index=dates)
        
        # Create sample data for multiple timeframes
        self.multi_tf_data = {
            'BTCUSDT': {
                '1m': self.sample_df.copy(),
                '5m': self.sample_df.copy(),
                '1h': self.sample_df.copy()
            },
            'ETHUSDT': {
                '1m': self.sample_df.copy(),
                '5m': self.sample_df.copy(),
                '1h': self.sample_df.copy()
            }
        }
    
    @patch('livetrade._components._load_all_symbols_data.load_symbol_data')
    def test_load_all_symbols_data_success(self, mock_load_symbol):
        """Test successful loading of data for multiple symbols."""
        # Setup
        mock_load_symbol.return_value = self.sample_df.copy()
        processor = MockProcessor()
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Execute
        result = load_all_symbols_data(processor, symbols, load_multi_timeframes=False)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(symbol in result for symbol in symbols))
        for symbol, df in result.items():
            self.assertIsInstance(df, pd.DataFrame)
    
    @patch('livetrade._components._load_all_symbols_data.load_symbol_data')
    def test_load_all_symbols_data_partial_failures(self, mock_load_symbol):
        """Test loading with some failures."""
        # Setup
        def side_effect(proc, sym, tfs, multi_tf):
            if sym == "ETHUSDT":
                return None
            return self.sample_df.copy()
            
        mock_load_symbol.side_effect = side_effect
        processor = MockProcessor()
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Execute
        result = load_all_symbols_data(processor, symbols)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result["BTCUSDT"])
        self.assertIsNone(result["ETHUSDT"])
        self.assertIsNotNone(result["XRPUSDT"])
    
    def test_load_all_symbols_data_invalid_inputs(self):
        """Test with invalid inputs."""
        processor = MockProcessor()
        
        # Test with empty symbols list
        result = load_all_symbols_data(processor, [])
        self.assertEqual(result, {})
        
        # Test with invalid processor
        result = load_all_symbols_data({}, ["BTCUSDT"])
        self.assertEqual(result, {})
        
        # Test with mixed valid/invalid symbols
        symbols = ["BTCUSDT", None, "", "ETHUSDT"]
        valid_symbols = [s for s in symbols if s and isinstance(s, str)]
        result = load_all_symbols_data(processor, valid_symbols)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Only valid symbols should be processed
    
    @patch('livetrade._components._load_all_symbols_data.load_symbol_data')
    def test_load_all_symbols_data_with_retries(self, mock_load_symbol):
        """Test retry functionality."""
        # Setup - fail once then succeed
        call_count = 0
        def side_effect(proc, sym, tfs, multi_tf):
            nonlocal call_count
            if sym == "ETHUSDT" and call_count == 0:
                call_count += 1
                return None
            return self.sample_df.copy()
            
        mock_load_symbol.side_effect = side_effect
        processor = MockProcessor()
        
        # Execute with max_retries=1
        with patch('livetrade._components._load_all_symbols_data._retry_with_backoff'):
            result = load_all_symbols_data(processor, ["ETHUSDT"], max_retries=1)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result["ETHUSDT"])
        self.assertEqual(mock_load_symbol.call_count, 2)  # Initial call + 1 retry


class TestHelperFunctions(unittest.TestCase):
    """Test cases for helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        self.sample_df = pd.DataFrame({
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [95 + i for i in range(100)],
            'close': [101 + i for i in range(100)],
            'volume': [1000 + i*10 for i in range(100)]
        }, index=dates)
    
    @patch('livetrade._components._load_all_symbols_data.tqdm')
    def test_create_progress_bar(self, mock_tqdm):
        """Test creating progress bar."""
        _create_progress_bar(100, "Test", "#FF8C00")
        mock_tqdm.assert_called_once()
    
    @patch('livetrade._components._load_all_symbols_data._create_progress_bar')
    @patch('livetrade._components._load_all_symbols_data.sleep')
    def test_wait_for_data_with_progress(self, mock_sleep, mock_create_bar):
        """Test waiting for data with progress."""
        # Setup
        mock_bar = MagicMock()
        mock_create_bar.return_value = mock_bar
        processor = MockProcessor()
        
        # Pre-populate cache
        cache_key = ("BTCUSDT", "1h")
        processor.df_cache[cache_key] = self.sample_df.copy()
        
        # Execute
        result = _wait_for_data_with_progress(processor, cache_key, "BTCUSDT", "1h")
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 100)
        mock_create_bar.assert_called_once()
        mock_bar.close.assert_called_once()
    
    @patch('livetrade._components._load_all_symbols_data._create_progress_bar')
    @patch('livetrade._components._load_all_symbols_data.time.sleep')
    def test_retry_with_backoff(self, mock_sleep, mock_create_bar):
        """Test retry with backoff."""
        # Setup
        mock_bar = MagicMock()
        mock_create_bar.return_value = mock_bar
        
        # Execute
        _retry_with_backoff("BTCUSDT", 0.5, "Retrying")
        
        # Assert
        mock_create_bar.assert_called_once()
        mock_bar.update.assert_called()
        mock_bar.close.assert_called_once()
    
    @patch('livetrade._components._load_all_symbols_data._create_progress_bar')
    def test_calculate_memory_usage(self, mock_create_bar):
        """Test memory usage calculation."""
        # Setup
        mock_bar = MagicMock()
        mock_create_bar.return_value = mock_bar
        
        # Create test data
        test_data = {
            "BTCUSDT": {
                "1h": self.sample_df.copy(),
                "5m": self.sample_df.copy()
            },
            "ETHUSDT": self.sample_df.copy(),
            "XRPUSDT": None
        }
        
        # Execute
        result = _calculate_memory_usage(test_data)
        
        # Assert
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Should have some memory usage
        mock_create_bar.assert_called_once()
        mock_bar.update.assert_called()
        mock_bar.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()