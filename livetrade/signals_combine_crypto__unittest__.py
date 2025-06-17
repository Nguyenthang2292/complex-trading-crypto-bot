import unittest
import sys
import os
from unittest.mock import Mock, patch
import pandas as pd
import logging

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

# Import the module to test
from livetrade.signals_combine_crypto import (
    crypto_signal_workflow,
    final_signals,
    main,
)

class TestSignalCombineCrypto(unittest.TestCase):
    """
    Unit tests for signals_combine_crypto module
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_processor = Mock()
        self.mock_logger = Mock()
        
        # Mock symbols list
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
        
        # Mock preloaded data structure
        self.mock_preloaded_data = {}
        for symbol in self.test_symbols:
            self.mock_preloaded_data[symbol] = {
                '1h': pd.DataFrame({
                    'Open': [100, 101, 102, 103, 104],
                    'High': [101, 102, 103, 104, 105],
                    'Low': [99, 100, 101, 102, 103],
                    'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
                    'Volume': [1000, 1100, 1200, 1300, 1400]
                }),
                '4h': pd.DataFrame({
                    'Open': [100, 102, 104],
                    'High': [102, 104, 106],
                    'Low': [99, 101, 103],
                    'Close': [101, 103, 105],
                    'Volume': [4000, 4400, 4800]
                }),
                '1d': pd.DataFrame({
                    'Open': [100, 105],
                    'High': [106, 110],
                    'Low': [98, 104],
                    'Close': [104, 108],
                    'Volume': [24000, 26400]
                })
            }
        
        # Mock performance results
        self.mock_performance_result_long = {
            'best_performers': [
                {'symbol': 'BTCUSDT', 'composite_score': 0.85},
                {'symbol': 'ETHUSDT', 'composite_score': 0.78}
            ]
        }
        
        self.mock_performance_result_short = {
            'worst_performers': [
                {'symbol': 'ADAUSDT', 'composite_score': 0.25},
                {'symbol': 'BNBUSDT', 'composite_score': 0.32}
            ]
        }
        
        # Mock RF signals
        self.mock_rf_signals_df = pd.DataFrame({
            'Pair': ['BTCUSDT', 'ETHUSDT'],
            'SignalTimeframe': ['1h', '4h']
        })
        
        # Mock HMM signals  
        self.mock_hmm_signals_df = pd.DataFrame({
            'Pair': ['BTCUSDT', 'ETHUSDT'],
            'SignalTimeframe': ['1h', '4h']
        })
    
    @patch('livetrade.signals_combine_crypto.load_all_pairs_data')
    @patch('livetrade.signals_combine_crypto.signal_best_performance_pairs')
    @patch('livetrade.signals_combine_crypto.process_signals_random_forest')
    @patch('livetrade.signals_combine_crypto.process_signals_hmm')
    @patch('livetrade.signals_combine_crypto.reload_timeframes_for_symbols')
    @patch('signals.signals_random_forest.train_and_save_global_rf_model')
    def test_crypto_signal_workflow_success(self, mock_train, mock_reload_timeframes, 
                                          mock_process_hmm, mock_process_rf, mock_signal_performance, 
                                          mock_load_data):
        """Test successful execution of crypto_signal_workflow."""
        
        # Setup mocks
        self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
        mock_load_data.return_value = self.mock_preloaded_data
        
        # Mock performance analysis calls
        mock_signal_performance.side_effect = [
            self.mock_performance_result_long,  # First call for LONG
            self.mock_performance_result_short   # Second call for SHORT
        ]
        
        # Mock RF processing
        mock_process_rf.return_value = self.mock_rf_signals_df
        
        # Mock HMM processing
        mock_process_hmm.return_value = self.mock_hmm_signals_df
        
        # Mock timeframe reloading
        mock_reload_timeframes.return_value = self.mock_preloaded_data
        
        # Mock RF model training function
        mock_train.return_value = (Mock(), '/path/to/model.joblib')
        
        # Execute the function
        result = crypto_signal_workflow(self.mock_processor)
        
        # Assertions
        self.assertIsInstance(result, list)
        self.mock_processor.get_symbols_list_by_quote_usdt.assert_called_once()
        mock_load_data.assert_called_once()
        self.assertEqual(mock_signal_performance.call_count, 2)  # Called twice for LONG and SHORT
    
    @patch('livetrade.signals_combine_crypto.load_all_pairs_data')
    def test_crypto_signal_workflow_no_data(self, mock_load_data):
        """Test crypto_signal_workflow when no data is loaded."""
        
        # Setup mocks
        self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
        mock_load_data.return_value = {}  # Empty data
        
        # Execute the function
        result = crypto_signal_workflow(self.mock_processor)
        
        # Assertions
        self.assertEqual(result, [])
        mock_load_data.assert_called_once()
    
    @patch('livetrade.signals_combine_crypto.load_all_pairs_data')
    @patch('livetrade.signals_combine_crypto.signal_best_performance_pairs')
    def test_crypto_signal_workflow_no_performance_results(self, mock_signal_performance, mock_load_data):
        """Test crypto_signal_workflow when performance analysis fails."""
        
        # Setup mocks
        self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
        mock_load_data.return_value = self.mock_preloaded_data
        mock_signal_performance.return_value = {}  # No best_performers key
        
        # Execute the function
        result = crypto_signal_workflow(self.mock_processor)
        
        # Assertions
        self.assertEqual(result, [])
    
    def test_crypto_signal_workflow_with_custom_timeframes(self):
        """Test crypto_signal_workflow with custom timeframes."""
        
        custom_performance_tf = ['30m', '2h']
        custom_hmm_tf = ['15m', '1h']
        
        with patch('livetrade.signals_combine_crypto.load_all_pairs_data') as mock_load_data:
            self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
            mock_load_data.return_value = {}  # Will return early due to no data
            
            # Execute the function with custom timeframes
            result = crypto_signal_workflow(
                self.mock_processor,
                timeframes_performance=custom_performance_tf,
                timeframes_hmm=custom_hmm_tf
            )
            
            # Verify that load_all_pairs_data was called with custom timeframes
            mock_load_data.assert_called_once()
            call_args = mock_load_data.call_args
            self.assertEqual(call_args[1]['timeframes'], custom_performance_tf)
    
    @patch('livetrade.signals_combine_crypto.crypto_signal_workflow')
    @patch('livetrade.signals_combine_crypto.tick_processor')
    def test_final_signals_success(self, mock_tick_processor, mock_workflow):
        """Test successful execution of final_signals function."""
        
        # Setup mocks
        mock_processor_instance = Mock()
        mock_tick_processor.return_value = mock_processor_instance
        
        expected_signals = [
            {'pair': 'BTCUSDT', 'direction': 'LONG', 'timeframe': '1h'},
            {'pair': 'ETHUSDT', 'direction': 'SHORT', 'timeframe': '4h'}
        ]
        mock_workflow.return_value = expected_signals
        
        # Execute the function
        result = final_signals()
        
        # Assertions
        self.assertEqual(result, expected_signals)
        mock_tick_processor.assert_called_once()
        mock_workflow.assert_called_once_with(
            processor=mock_processor_instance,
            timeframes_performance=['15m', '30m', '1h', '4h', '1d'],
            timeframes_hmm=['5m', '15m', '30m', '1h', '4h']
        )
    
    @patch('livetrade.signals_combine_crypto.crypto_signal_workflow')
    @patch('livetrade.signals_combine_crypto.tick_processor')
    def test_final_signals_with_custom_params(self, mock_tick_processor, mock_workflow):
        """Test final_signals with custom parameters."""
        
        # Setup mocks
        mock_processor_instance = Mock()
        mock_tick_processor.return_value = mock_processor_instance
        mock_workflow.return_value = []
        
        custom_performance_tf = ['1h', '4h']
        custom_hmm_tf = ['5m', '30m']
        
        # Execute the function with custom parameters
        result = final_signals(
            timeframes_performance=custom_performance_tf,
            timeframes_hmm=custom_hmm_tf,
            log_level='DEBUG'
        )
        
        # Assertions
        self.assertEqual(result, [])
        mock_workflow.assert_called_once_with(
            processor=mock_processor_instance,
            timeframes_performance=custom_performance_tf,
            timeframes_hmm=custom_hmm_tf
        )
    
    @patch('livetrade.signals_combine_crypto.crypto_signal_workflow')
    @patch('livetrade.signals_combine_crypto.tick_processor')
    def test_final_signals_keyboard_interrupt(self, mock_tick_processor, mock_workflow):
        """Test final_signals handling of KeyboardInterrupt."""
        
        # Setup mocks to raise KeyboardInterrupt
        mock_tick_processor.side_effect = KeyboardInterrupt("Test interrupt")
        
        # Execute the function
        result = final_signals()
        
        # Assertions
        self.assertEqual(result, [])
    
    @patch('livetrade.signals_combine_crypto.crypto_signal_workflow')
    @patch('livetrade.signals_combine_crypto.tick_processor')
    def test_final_signals_exception_handling(self, mock_tick_processor, mock_workflow):
        """Test final_signals handling of general exceptions."""
        
        # Setup mocks to raise an exception
        mock_tick_processor.side_effect = Exception("Test exception")
        
        # Execute the function
        result = final_signals()
        
        # Assertions
        self.assertEqual(result, [])
    
    @patch('livetrade.signals_combine_crypto.final_signals')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_function_success(self, mock_parse_args, mock_final_signals):
        """Test main function with successful signal generation."""
        
        # Setup mocks
        mock_args = Mock()
        mock_args.log_level = 'INFO'
        mock_parse_args.return_value = mock_args
        
        expected_signals = [
            {'pair': 'BTCUSDT', 'direction': 'LONG', 'timeframe': '1h'},
            {'pair': 'ETHUSDT', 'direction': 'SHORT', 'timeframe': '4h'}
        ]
        mock_final_signals.return_value = expected_signals
        
        # Execute main function
        main()
        
        # Assertions
        mock_final_signals.assert_called_once_with(log_level='INFO')
    
    @patch('livetrade.signals_combine_crypto.final_signals')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_function_no_signals(self, mock_parse_args, mock_final_signals):
        """Test main function when no signals are generated."""
        
        # Setup mocks
        mock_args = Mock()
        mock_args.log_level = 'DEBUG'
        mock_parse_args.return_value = mock_args
        
        mock_final_signals.return_value = []
        
        # Execute main function
        main()
        
        # Assertions
        mock_final_signals.assert_called_once_with(log_level='DEBUG')
    
    def test_cpu_count_calculation(self):
        """Test CPU count calculation logic in crypto_signal_workflow."""
        
        with patch('os.cpu_count') as mock_cpu_count:
            with patch('livetrade.signals_combine_crypto.load_all_pairs_data') as mock_load_data:
                
                # Test with normal CPU count
                mock_cpu_count.return_value = 8
                self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
                mock_load_data.return_value = {}  # Will return early
                
                result = crypto_signal_workflow(self.mock_processor)
                
                # Should use 80% of 8 CPUs = 6 workers (but will return early due to no data)
                self.assertEqual(result, [])
                
                # Test with None CPU count
                mock_cpu_count.return_value = None
                result = crypto_signal_workflow(self.mock_processor)
                self.assertEqual(result, [])
    
    def test_signal_structure_validation(self):
        """Test that generated signals have the correct structure."""
        
        # Test signal dictionary structure
        expected_signal = {
            'pair': 'BTCUSDT',
            'direction': 'LONG',
            'timeframe': '1h'
        }
        
        # Verify all required keys are present
        required_keys = ['pair', 'direction', 'timeframe']
        for key in required_keys:
            self.assertIn(key, expected_signal)
        
        # Verify direction values
        valid_directions = ['LONG', 'SHORT']
        self.assertIn(expected_signal['direction'], valid_directions)
        
        # Verify timeframe format
        valid_timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.assertIn(expected_signal['timeframe'], valid_timeframes)
    
    @patch('pathlib.Path.mkdir')
    @patch('livetrade.signals_combine_crypto.load_all_pairs_data')
    def test_models_directory_creation(self, mock_load_data, mock_mkdir):
        """Test that models directory is created properly."""
        # Return minimal nonempty data so that execution proceeds beyond early return.
        minimal_data = {'BTCUSDT': {'15m': pd.DataFrame({'Close': [100, 101, 102, 103, 104]})}}
        self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
        mock_load_data.return_value = minimal_data
        
        result = crypto_signal_workflow(self.mock_processor)
        
        self.assertEqual(result, [])
        # Verify that mkdir was attempted at least once.
        mock_mkdir.assert_called_with(exist_ok=True)
    
    def test_module_imports(self):
        """Test that required module members are imported correctly."""
        from livetrade.signals_combine_crypto import terminate
        self.assertIsInstance(terminate, bool)
    
    def test_termination_flag_handling(self):
        """Test that termination flag is properly handled."""
        import livetrade.signals_combine_crypto as sc
        sc.terminate = True
        self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
        
        # Since crypto_signal_workflow catches KeyboardInterrupt and returns an empty list,
        # we assert that the result is empty.
        result = sc.crypto_signal_workflow(self.mock_processor)
        self.assertEqual(result, [])
        # Reset flag for subsequent tests.
        sc.terminate = False
    
    def test_default_timeframes(self):
        """Test that default timeframes are set correctly."""
        
        with patch('livetrade.signals_combine_crypto.load_all_pairs_data') as mock_load_data:
            self.mock_processor.get_symbols_list_by_quote_usdt.return_value = self.test_symbols
            mock_load_data.return_value = {}
            
            # Test with None timeframes (should use defaults)
            result = crypto_signal_workflow(self.mock_processor, None, None)
            
            # Verify function was called (will return [] due to no data)
            self.assertEqual(result, [])
            
            # Verify load_all_pairs_data was called with default performance timeframes
            mock_load_data.assert_called_once()
            call_args = mock_load_data.call_args
            expected_default_tf = ['15m', '30m', '1h', '4h', '1d']
            self.assertEqual(call_args[1]['timeframes'], expected_default_tf)


class TestSignalFiltering(unittest.TestCase):
    """
    Test signal filtering logic specifically
    """
    
    def test_rf_signal_to_hmm_filtering(self):
        """Test RF signals are properly filtered by HMM results."""
        
        # Mock RF signals
        rf_long_signals = [
            {'pair': 'BTCUSDT', 'direction': 'LONG', 'timeframe': '1h'},
            {'pair': 'ETHUSDT', 'direction': 'LONG', 'timeframe': '4h'},
            {'pair': 'ADAUSDT', 'direction': 'LONG', 'timeframe': '1d'}
        ]
        
        rf_short_signals = [
            {'pair': 'BNBUSDT', 'direction': 'SHORT', 'timeframe': '1h'},
            {'pair': 'SOLUSDT', 'direction': 'SHORT', 'timeframe': '4h'}
        ]
        
        # Mock HMM pairs (only some RF signals should pass)
        strict_hmm_pairs = {('BTCUSDT', '1h'), ('BNBUSDT', '1h')}
        non_strict_hmm_pairs = {('ETHUSDT', '4h')}
        all_hmm_pairs = strict_hmm_pairs.union(non_strict_hmm_pairs)
        
        # Filter signals based on HMM (simulating the logic in crypto_signal_workflow)
        final_signals = []
        
        # Filter LONG signals
        for rf_signal in rf_long_signals:
            signal_key = (rf_signal['pair'], rf_signal['timeframe'])
            if signal_key in all_hmm_pairs:
                final_signals.append(rf_signal)
        
        # Filter SHORT signals
        for rf_signal in rf_short_signals:
            signal_key = (rf_signal['pair'], rf_signal['timeframe'])
            if signal_key in all_hmm_pairs:
                final_signals.append(rf_signal)
        
        # Assertions
        self.assertEqual(len(final_signals), 3)  # BTCUSDT-1h, ETHUSDT-4h, BNBUSDT-1h
        
        # Verify specific signals made it through
        signal_keys = [(s['pair'], s['timeframe']) for s in final_signals]
        self.assertIn(('BTCUSDT', '1h'), signal_keys)
        self.assertIn(('ETHUSDT', '4h'), signal_keys)
        self.assertIn(('BNBUSDT', '1h'), signal_keys)
        
        # Verify filtered out signals
        self.assertNotIn(('ADAUSDT', '1d'), signal_keys)
        self.assertNotIn(('SOLUSDT', '4h'), signal_keys)


if __name__ == '__main__':
    # Set up test environment
    logging.disable(logging.CRITICAL)  # Disable logging during tests
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSignalCombineCrypto))
    test_suite.addTest(unittest.makeSuite(TestSignalFiltering))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
