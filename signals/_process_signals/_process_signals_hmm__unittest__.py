import logging
import os
import pandas as pd
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from signals._process_signals._process_signals_hmm import (
    _get_cpu_count,
    _generate_signal_hmm,
    _process_symbol_worker,
    process_signals_hmm,
    reload_timeframes_for_symbols,
    DATAFRAME_COLUMNS,
    _LOG_LOCK
)


class TestGetCpuCount(unittest.TestCase):
    """Test cases for _get_cpu_count function."""
    
    @patch('signals._process_signals._process_signals_hmm._HAS_PSUTIL', True)
    @patch('signals._process_signals._process_signals_hmm.psutil')
    def test_with_psutil_available(self, mock_psutil):
        """Test CPU count detection with psutil available."""
        mock_psutil.cpu_count.return_value = 8
        result = _get_cpu_count()
        self.assertEqual(result, 8)
        mock_psutil.cpu_count.assert_called_once()
    
    @patch('signals._process_signals._process_signals_hmm._HAS_PSUTIL', False)
    @patch('os.cpu_count')
    def test_without_psutil_with_os_cpu_count(self, mock_os_cpu_count):
        """Test CPU count detection without psutil but with os.cpu_count."""
        mock_os_cpu_count.return_value = 4
        result = _get_cpu_count()
        self.assertEqual(result, 4)
        mock_os_cpu_count.assert_called_once()
    
    @patch('signals._process_signals._process_signals_hmm._HAS_PSUTIL', False)
    @patch('os.cpu_count')
    def test_without_psutil_no_os_cpu_count(self, mock_os_cpu_count):
        """Test CPU count detection fallback to default."""
        mock_os_cpu_count.return_value = None
        result = _get_cpu_count()
        self.assertEqual(result, 4)
    
    @patch('signals._process_signals._process_signals_hmm._HAS_PSUTIL', True)
    @patch('signals._process_signals._process_signals_hmm.psutil')
    def test_exception_handling(self, mock_psutil):
        """Test exception handling in CPU count detection."""
        mock_psutil.cpu_count.side_effect = Exception("Test error")
        result = _get_cpu_count()
        self.assertEqual(result, 4)


class TestGenerateSignalHmm(unittest.TestCase):
    """Test cases for _generate_signal_hmm function."""
    
    def setUp(self):
        """Set up test data."""
        self.test_pair = "BTCUSDT"
        self.valid_df = pd.DataFrame({
            'High': [50000, 51000, 52000, 53000, 54000] * 20,
            'Low': [49000, 50000, 51000, 52000, 53000] * 20,
            'close': [49500, 50500, 51500, 52500, 53500] * 20
        })
        self.insufficient_df = pd.DataFrame({
            'High': [50000, 51000],
            'Low': [49000, 50000],
            'close': [49500, 50500]
        })
    
    def test_missing_close_column(self):
        """Test handling of missing close column."""
        df_no_close = pd.DataFrame({
            'High': [50000, 51000, 52000],
            'Low': [49000, 50000, 51000]
        })
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, df_no_close)
            
        self.assertEqual(result, (self.test_pair, None, None))
    
    @patch('pandas_ta.rsi')
    def test_rsi_calculation_failure(self, mock_rsi):
        """Test RSI calculation failure handling."""
        mock_rsi.return_value = pd.Series([None] * len(self.valid_df))
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, self.valid_df)
            
        self.assertEqual(result, (self.test_pair, None, None))
    
    @patch('pandas_ta.rsi')
    def test_insufficient_data_for_hmm(self, mock_rsi):
        """Test insufficient data for HMM analysis."""
        mock_rsi.return_value = pd.Series([70.0, 65.0])
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, self.insufficient_df)
            
        self.assertEqual(result, (self.test_pair, None, None))
    
    @patch('signals._process_signals._process_signals_hmm.hmm_signals')
    @patch('pandas_ta.rsi')
    def test_long_signal_generation(self, mock_rsi, mock_hmm_signals):
        """Test LONG signal generation."""
        # Setup RSI for LONG signal (> 60)
        mock_rsi.return_value = pd.Series([65.0] * len(self.valid_df))
        # Setup HMM for bullish signal
        mock_hmm_signals.return_value = (1, 1)
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, self.valid_df)
            
        pair, signal, price = result
        self.assertEqual(pair, self.test_pair)
        self.assertEqual(signal, 1)  # LONG signal
        self.assertEqual(price, 53500)  # Last close price
    
    @patch('signals._process_signals._process_signals_hmm.hmm_signals')
    @patch('pandas_ta.rsi')
    def test_short_signal_generation(self, mock_rsi, mock_hmm_signals):
        """Test SHORT signal generation."""
        # Setup RSI for SHORT signal (< 40)
        mock_rsi.return_value = pd.Series([35.0] * len(self.valid_df))
        # Setup HMM for bearish signal
        mock_hmm_signals.return_value = (-1, -1)
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, self.valid_df)
            
        pair, signal, price = result
        self.assertEqual(pair, self.test_pair)
        self.assertEqual(signal, -1)  # SHORT signal
        self.assertEqual(price, 53500)
    
    @patch('signals._process_signals._process_signals_hmm.hmm_signals')
    @patch('pandas_ta.rsi')
    def test_no_signal_generation(self, mock_rsi, mock_hmm_signals):
        """Test no signal generation."""
        # Setup RSI for neutral (between 40-60)
        mock_rsi.return_value = pd.Series([50.0] * len(self.valid_df))
        # Setup HMM for neutral signal
        mock_hmm_signals.return_value = (0, 0)
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, self.valid_df)
            
        pair, signal, price = result
        self.assertEqual(pair, self.test_pair)
        self.assertIsNone(signal)  # No signal
        self.assertEqual(price, 53500)
    
    @patch('signals._process_signals._process_signals_hmm.hmm_signals')
    @patch('pandas_ta.rsi')
    def test_invalid_hmm_signals(self, mock_rsi, mock_hmm_signals):
        """Test handling of invalid HMM signals."""
        mock_rsi.return_value = pd.Series([65.0] * len(self.valid_df))
        mock_hmm_signals.return_value = (None, None)
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, self.valid_df)
            
        pair, signal, price = result
        self.assertEqual(pair, self.test_pair)
        self.assertIsNone(signal)
        self.assertEqual(price, 53500)
    
    @patch('signals._process_signals._process_signals_hmm.hmm_signals')
    @patch('pandas_ta.rsi')
    def test_strict_mode_parameter(self, mock_rsi, mock_hmm_signals):
        """Test strict mode parameter passing."""
        mock_rsi.return_value = pd.Series([65.0] * len(self.valid_df))
        mock_hmm_signals.return_value = (1, 0)
        
        with patch('signals._process_signals._process_signals_hmm.logger'), \
             patch('signals._process_signals._process_signals_hmm.OptimizingParameters') as mock_params:
            
            mock_params_instance = Mock()
            mock_params.return_value = mock_params_instance
            
            _generate_signal_hmm(self.test_pair, self.valid_df, strict_mode=True)
            
            # Verify strict_mode was set
            self.assertTrue(mock_params_instance.strict_mode)
    
    def test_exception_handling(self):
        """Test exception handling in signal generation."""
        # Pass invalid dataframe to trigger exception
        invalid_df = "not_a_dataframe"
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _generate_signal_hmm(self.test_pair, invalid_df)
            
        self.assertEqual(result, (self.test_pair, None, None))


class TestProcessSymbolWorker(unittest.TestCase):
    """Test cases for _process_symbol_worker function."""
    
    def setUp(self):
        """Set up test data."""
        self.test_symbol = "BTCUSDT"
        self.test_timeframes = ['1h', '4h', '1d']
        self.valid_df = pd.DataFrame({
            'High': [50000] * 60,
            'Low': [49000] * 60,
            'close': [49500] * 60
        })
        self.symbol_data = {
            '1h': self.valid_df.copy(),
            '4h': self.valid_df.copy(),
            '1d': self.valid_df.copy()
        }
    
    def test_no_data_provided(self):
        """Test handling when no data is provided."""
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _process_symbol_worker(
                self.test_symbol, None, self.test_timeframes, False
            )
        self.assertIsNone(result)
    
    def test_invalid_data_format(self):
        """Test handling of invalid data format."""
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _process_symbol_worker(
                self.test_symbol, "invalid_format", self.test_timeframes, False
            )
        self.assertIsNone(result)
    
    def test_insufficient_data_all_timeframes(self):
        """Test when all timeframes have insufficient data."""
        insufficient_data = {
            '1h': pd.DataFrame({'close': [1, 2, 3]}),  # < 50 rows
            '4h': pd.DataFrame(),  # Empty
            '1d': None  # None
        }
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _process_symbol_worker(
                self.test_symbol, insufficient_data, self.test_timeframes, False
            )
        self.assertIsNone(result)
    
    @patch('signals._process_signals._process_signals_hmm._generate_signal_hmm')
    def test_long_signal_found(self, mock_generate_signal):
        """Test LONG signal detection."""
        mock_generate_signal.return_value = (self.test_symbol, 1, 50000.0)
        
        result = _process_symbol_worker(
            self.test_symbol, self.symbol_data, self.test_timeframes, False,
            include_long_signals=True, include_short_signals=True
        )
        
        expected = {
            'Symbol': self.test_symbol,
            'SignalTimeframe': '1h',
            'SignalType': 'LONG'
        }
        self.assertEqual(result, expected)
    
    @patch('signals._process_signals._process_signals_hmm._generate_signal_hmm')
    def test_short_signal_found(self, mock_generate_signal):
        """Test SHORT signal detection."""
        mock_generate_signal.return_value = (self.test_symbol, -1, 50000.0)
        
        result = _process_symbol_worker(
            self.test_symbol, self.symbol_data, self.test_timeframes, False,
            include_long_signals=True, include_short_signals=True
        )
        
        expected = {
            'Symbol': self.test_symbol,
            'SignalTimeframe': '1h',
            'SignalType': 'SHORT'
        }
        self.assertEqual(result, expected)
    
    @patch('signals._process_signals._process_signals_hmm._generate_signal_hmm')
    def test_long_signal_disabled(self, mock_generate_signal):
        """Test LONG signal when LONG signals are disabled."""
        mock_generate_signal.return_value = (self.test_symbol, 1, 50000.0)
        
        result = _process_symbol_worker(
            self.test_symbol, self.symbol_data, self.test_timeframes, False,
            include_long_signals=False, include_short_signals=True
        )
        
        self.assertIsNone(result)
    
    @patch('signals._process_signals._process_signals_hmm._generate_signal_hmm')
    def test_short_signal_disabled(self, mock_generate_signal):
        """Test SHORT signal when SHORT signals are disabled."""
        mock_generate_signal.return_value = (self.test_symbol, -1, 50000.0)
        
        result = _process_symbol_worker(
            self.test_symbol, self.symbol_data, self.test_timeframes, False,
            include_long_signals=True, include_short_signals=False
        )
        
        self.assertIsNone(result)
    
    @patch('signals._process_signals._process_signals_hmm._generate_signal_hmm')
    def test_no_signal_generated(self, mock_generate_signal):
        """Test when no signal is generated."""
        mock_generate_signal.return_value = (self.test_symbol, None, 50000.0)
        
        result = _process_symbol_worker(
            self.test_symbol, self.symbol_data, self.test_timeframes, False,
            include_long_signals=True, include_short_signals=True
        )
        
        self.assertIsNone(result)
    
    @patch('signals._process_signals._process_signals_hmm._generate_signal_hmm')
    def test_timeframe_priority(self, mock_generate_signal):
        """Test that timeframes are processed in priority order."""
        # First call returns no signal, second call returns LONG signal
        mock_generate_signal.side_effect = [
            (self.test_symbol, None, 50000.0),  # 1h: no signal
            (self.test_symbol, 1, 50000.0),     # 4h: LONG signal
            (self.test_symbol, -1, 50000.0)     # 1d: SHORT signal (should not be reached)
        ]
        
        result = _process_symbol_worker(
            self.test_symbol, self.symbol_data, self.test_timeframes, False,
            include_long_signals=True, include_short_signals=True
        )
        
        expected = {
            'Symbol': self.test_symbol,
            'SignalTimeframe': '4h',  # Should be from second timeframe
            'SignalType': 'LONG'
        }
        self.assertEqual(result, expected)
        # Should only call twice (1h and 4h, not 1d)
        self.assertEqual(mock_generate_signal.call_count, 2)
    
    @patch('signals._process_signals._process_signals_hmm._generate_signal_hmm')
    def test_exception_handling_in_signal_generation(self, mock_generate_signal):
        """Test exception handling during signal generation."""
        mock_generate_signal.side_effect = Exception("Signal generation error")
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _process_symbol_worker(
                self.test_symbol, self.symbol_data, self.test_timeframes, False
            )
        
        self.assertIsNone(result)
    
    def test_exception_handling_general(self):
        """Test general exception handling in worker function."""
        # Trigger exception by passing invalid symbol type
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = _process_symbol_worker(
                None, self.symbol_data, self.test_timeframes, False
            )
        
        self.assertIsNone(result)


class TestProcessSignalsHmm(unittest.TestCase):
    """Test cases for process_signals_hmm function."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_df = pd.DataFrame({
            'High': [50000] * 60,
            'Low': [49000] * 60,
            'close': [49500] * 60
        })
        self.preloaded_data = {
            'BTCUSDT': {
                '1h': self.valid_df.copy(),
                '4h': self.valid_df.copy(),
                '1d': self.valid_df.copy()
            },
            'ETHUSDT': {
                '1h': self.valid_df.copy(),
                '4h': self.valid_df.copy(),
                '1d': self.valid_df.copy()
            }
        }
        self.test_timeframes = ['1h', '4h', '1d']
    
    def test_both_signals_disabled(self):
        """Test when both LONG and SHORT signals are disabled."""
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = process_signals_hmm(
                self.preloaded_data, self.test_timeframes,
                include_long_signals=False, include_short_signals=False
            )
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), DATAFRAME_COLUMNS)
    
    def test_invalid_preloaded_data(self):
        """Test handling of invalid preloaded data."""
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = process_signals_hmm(
                None, self.test_timeframes
            )
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), DATAFRAME_COLUMNS)
    
    def test_empty_preloaded_data(self):
        """Test handling of empty preloaded data."""
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = process_signals_hmm(
                {}, self.test_timeframes
            )
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), DATAFRAME_COLUMNS)
    
    def test_empty_timeframes(self):
        """Test handling of empty timeframes list."""
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = process_signals_hmm(
                self.preloaded_data, []
            )
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), DATAFRAME_COLUMNS)
    
    def test_none_timeframes_uses_default(self):
        """Test that None timeframes uses default values."""
        with patch('signals._process_signals._process_signals_hmm.logger'), \
             patch('signals._process_signals._process_signals_hmm._process_symbol_worker') as mock_worker:
            
            mock_worker.return_value = None
            
            process_signals_hmm(self.preloaded_data, None)
            
            # Check that default timeframes were used
            calls = mock_worker.call_args_list
            for call in calls:
                self.assertEqual(call[0][2], ['1h', '4h', '1d'])  # Default timeframes
    
    @patch('signals._process_signals._process_signals_hmm._get_cpu_count')
    def test_auto_max_workers_calculation(self, mock_get_cpu_count):
        """Test automatic max_workers calculation."""
        mock_get_cpu_count.return_value = 10
        
        with patch('signals._process_signals._process_signals_hmm.logger'), \
             patch('signals._process_signals._process_signals_hmm.ThreadPoolExecutor') as mock_executor:
            
            mock_executor.return_value.__enter__.return_value.submit.return_value = Mock()
            
            process_signals_hmm(
                self.preloaded_data, self.test_timeframes, max_workers=0
            )
            
            # Should use 80% of CPU count = 8
            mock_executor.assert_called_with(max_workers=8)
    
    @patch('signals._process_signals._process_signals_hmm._process_symbol_worker')
    @patch('signals._process_signals._process_signals_hmm.tqdm')
    def test_successful_signal_processing(self, mock_tqdm, mock_worker):
        """Test successful signal processing with results."""
        # Mock worker to return different signals for different symbols
        mock_worker.side_effect = [
            {'Symbol': 'BTCUSDT', 'SignalTimeframe': '1h', 'SignalType': 'LONG'},
            {'Symbol': 'ETHUSDT', 'SignalTimeframe': '4h', 'SignalType': 'SHORT'}
        ]
        
        # Mock tqdm to return futures directly
        mock_future1 = Mock()
        mock_future1.result.return_value = {'Symbol': 'BTCUSDT', 'SignalTimeframe': '1h', 'SignalType': 'LONG'}
        mock_future2 = Mock()
        mock_future2.result.return_value = {'Symbol': 'ETHUSDT', 'SignalTimeframe': '4h', 'SignalType': 'SHORT'}
        
        mock_tqdm.return_value = [mock_future1, mock_future2]
        
        with patch('signals._process_signals._process_signals_hmm.logger'), \
             patch('signals._process_signals._process_signals_hmm.ThreadPoolExecutor') as mock_executor:
            
            mock_executor_instance = Mock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
            
            result = process_signals_hmm(
                self.preloaded_data, self.test_timeframes
            )
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['Symbol'], 'BTCUSDT')
        self.assertEqual(result.iloc[0]['SignalType'], 'LONG')
        self.assertEqual(result.iloc[1]['Symbol'], 'ETHUSDT')
        self.assertEqual(result.iloc[1]['SignalType'], 'SHORT')
    
    @patch('signals._process_signals._process_signals_hmm._process_symbol_worker')
    def test_no_signals_found(self, mock_worker):
        """Test when no signals are found."""
        mock_worker.return_value = None
        
        with patch('signals._process_signals._process_signals_hmm.logger'), \
             patch('signals._process_signals._process_signals_hmm.tqdm') as mock_tqdm:
            
            mock_future = Mock()
            mock_future.result.return_value = None
            mock_tqdm.return_value = [mock_future, mock_future]
            
            with patch('signals._process_signals._process_signals_hmm.ThreadPoolExecutor') as mock_executor:
                mock_executor_instance = Mock()
                mock_executor.return_value.__enter__.return_value = mock_executor_instance
                mock_executor_instance.submit.return_value = mock_future
                
                result = process_signals_hmm(
                    self.preloaded_data, self.test_timeframes
                )
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), DATAFRAME_COLUMNS)
    
    def test_strict_mode_parameter_passing(self):
        """Test that strict_mode parameter is passed correctly."""
        with patch('signals._process_signals._process_signals_hmm.logger'), \
             patch('signals._process_signals._process_signals_hmm._process_symbol_worker') as mock_worker:
            
            mock_worker.return_value = None
            
            process_signals_hmm(
                self.preloaded_data, self.test_timeframes, strict_mode=True
            )
            
            # Check that strict_mode=True was passed to worker
            calls = mock_worker.call_args_list
            for call in calls:
                self.assertTrue(call[0][3])  # strict_mode parameter
    
    def test_signal_type_filtering(self):
        """Test signal type filtering parameters."""
        with patch('signals._process_signals._process_signals_hmm.logger'), \
             patch('signals._process_signals._process_signals_hmm._process_symbol_worker') as mock_worker:
            
            mock_worker.return_value = None
            
            process_signals_hmm(
                self.preloaded_data, self.test_timeframes,
                include_long_signals=False, include_short_signals=True
            )
            
            # Check that signal filtering parameters were passed
            calls = mock_worker.call_args_list
            for call in calls:
                self.assertFalse(call[1]['include_long_signals'])
                self.assertTrue(call[1]['include_short_signals'])


class TestReloadTimeframesForSymbols(unittest.TestCase):
    """Test cases for reload_timeframes_for_symbols function."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_processor = Mock()
        self.test_symbols = ['BTCUSDT', 'ETHUSDT']
        self.test_timeframes = ['1h', '4h', '1d']
        self.valid_df = pd.DataFrame({
            'High': [50000] * 100,
            'Low': [49000] * 100,
            'close': [49500] * 100
        })
    
    @patch('signals._process_signals._process_signals_hmm.load_symbol_data')
    def test_successful_data_reload(self, mock_load_symbol_data):
        """Test successful data reloading."""
        # Mock successful data loading
        mock_load_symbol_data.return_value = {
            '1h': self.valid_df.copy(),
            '4h': self.valid_df.copy(),
            '1d': self.valid_df.copy()
        }
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = reload_timeframes_for_symbols(
                self.mock_processor, self.test_symbols, self.test_timeframes
            )
        
        self.assertEqual(len(result), 2)  # Two symbols
        self.assertIn('BTCUSDT', result)
        self.assertIn('ETHUSDT', result)
        
        for symbol in self.test_symbols:
            self.assertEqual(len(result[symbol]), 3)  # Three timeframes
            for tf in self.test_timeframes:
                self.assertIn(tf, result[symbol])
                self.assertFalse(result[symbol][tf].empty)
    
    @patch('signals._process_signals._process_signals_hmm.load_symbol_data')
    def test_partial_data_loading(self, mock_load_symbol_data):
        """Test partial data loading (some timeframes missing)."""
        # Mock partial data loading
        mock_load_symbol_data.return_value = {
            '1h': self.valid_df.copy(),
            '4h': None,  # Missing data
            '1d': pd.DataFrame()  # Empty dataframe
        }
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = reload_timeframes_for_symbols(
                self.mock_processor, self.test_symbols, self.test_timeframes
            )
        
        # Should still include symbols but only with valid timeframes
        self.assertEqual(len(result), 2)
        for symbol in self.test_symbols:
            self.assertEqual(len(result[symbol]), 1)  # Only 1h timeframe
            self.assertIn('1h', result[symbol])
            self.assertNotIn('4h', result[symbol])
            self.assertNotIn('1d', result[symbol])
    
    @patch('signals._process_signals._process_signals_hmm.load_symbol_data')
    def test_no_valid_data(self, mock_load_symbol_data):
        """Test when no valid data is available."""
        # Mock no valid data
        mock_load_symbol_data.return_value = {
            '1h': None,
            '4h': pd.DataFrame(),
            '1d': None
        }
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = reload_timeframes_for_symbols(
                self.mock_processor, self.test_symbols, self.test_timeframes
            )
        
        self.assertEqual(len(result), 0)  # No symbols with valid data
    
    @patch('signals._process_signals._process_signals_hmm.load_symbol_data')
    def test_load_data_failure(self, mock_load_symbol_data):
        """Test handling of data loading failures."""
        # Mock data loading failure
        mock_load_symbol_data.return_value = None
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = reload_timeframes_for_symbols(
                self.mock_processor, self.test_symbols, self.test_timeframes
            )
        
        self.assertEqual(len(result), 0)  # No symbols loaded
    
    @patch('signals._process_signals._process_signals_hmm.load_symbol_data')
    def test_exception_handling(self, mock_load_symbol_data):
        """Test exception handling during data loading."""
        # Mock exception during data loading
        mock_load_symbol_data.side_effect = Exception("Loading error")
        
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = reload_timeframes_for_symbols(
                self.mock_processor, self.test_symbols, self.test_timeframes
            )
        
        self.assertEqual(len(result), 0)  # No symbols loaded due to exceptions
    
    def test_empty_symbols_list(self):
        """Test with empty symbols list."""
        with patch('signals._process_signals._process_signals_hmm.logger'):
            result = reload_timeframes_for_symbols(
                self.mock_processor, [], self.test_timeframes
            )
        
        self.assertEqual(len(result), 0)
    
    def test_empty_timeframes_list(self):
        """Test with empty timeframes list."""
        with patch('signals._process_signals._process_signals_hmm.load_symbol_data') as mock_load_symbol_data:
            mock_load_symbol_data.return_value = {}
            
            with patch('signals._process_signals._process_signals_hmm.logger'):
                result = reload_timeframes_for_symbols(
                    self.mock_processor, self.test_symbols, []
                )
        
        self.assertEqual(len(result), 0)


class TestModuleConstants(unittest.TestCase):
    """Test cases for module constants and globals."""
    
    def test_dataframe_columns(self):
        """Test DATAFRAME_COLUMNS constant."""
        expected_columns = ['Symbol', 'SignalTimeframe', 'SignalType']
        self.assertEqual(DATAFRAME_COLUMNS, expected_columns)
    
    def test_log_lock_type(self):
        """Test _LOG_LOCK is a threading.Lock."""
        self.assertIsInstance(_LOG_LOCK, threading.Lock)


if __name__ == '__main__':
    unittest.main()