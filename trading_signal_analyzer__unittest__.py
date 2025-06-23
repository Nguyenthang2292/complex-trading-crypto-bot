#!/usr/bin/env python3
"""
Unit Tests for Trading Signal Analyzer
Test coverage for all methods and functionality in trading_signal_analyzer.py

This test suite covers:
- Initialization and setup
- Symbol validation and normalization
- Model management (clear, reload)
- Data preparation and processing
- Signal analysis from all models
- Performance analysis
- Error handling and edge cases
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trading_signal_analyzer import TradingSignalAnalyzer
from components.config import (
    DEFAULT_TIMEFRAMES, DEFAULT_CRYPTO_SYMBOLS, MODELS_DIR,
    SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NEUTRAL
)


class TestTradingSignalAnalyzer(unittest.TestCase):
    """Test cases for TradingSignalAnalyzer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.original_models_dir = MODELS_DIR
        
        # Mock MODELS_DIR to use temp directory
        self.models_dir_patcher = patch('trading_signal_analyzer.MODELS_DIR', Path(self.temp_dir))
        self.mock_models_dir = self.models_dir_patcher.start()
        
        # Create models directory
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Mock logger to avoid actual logging during tests
        self.logger_patcher = patch('trading_signal_analyzer.logger')
        self.mock_logger = self.logger_patcher.start()
        
        # Mock TickProcessor
        self.tick_processor_patcher = patch('trading_signal_analyzer.TickProcessor')
        self.mock_tick_processor_class = self.tick_processor_patcher.start()
        self.mock_tick_processor = Mock()
        self.mock_tick_processor_class.return_value = self.mock_tick_processor
        
        # Mock symbol list
        self.mock_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        self.mock_tick_processor.get_symbols_list_by_quote_usdt.return_value = self.mock_symbols
        
        # Initialize analyzer
        self.analyzer = TradingSignalAnalyzer()

    def tearDown(self):
        """Clean up after each test method."""
        # Stop all patches
        self.models_dir_patcher.stop()
        self.logger_patcher.stop()
        self.tick_processor_patcher.stop()
        
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_success(self):
        """Test successful initialization of TradingSignalAnalyzer."""
        # Test that analyzer was initialized correctly
        self.assertEqual(self.analyzer.valid_timeframes, DEFAULT_TIMEFRAMES)
        self.assertEqual(self.analyzer.valid_symbols, self.mock_symbols)
        self.assertIsNotNone(self.analyzer.processor)
        
        # Verify logger was called
        self.mock_logger.config.assert_called()

    def test_initialization_fallback_to_default_symbols(self):
        """Test initialization falls back to default symbols when exchange fails."""
        # Mock exchange failure
        self.mock_tick_processor.get_symbols_list_by_quote_usdt.side_effect = ConnectionError("Connection failed")
        
        # Create new analyzer with failed connection
        analyzer = TradingSignalAnalyzer()
        
        # Should fall back to default symbols
        self.assertEqual(analyzer.valid_symbols, DEFAULT_CRYPTO_SYMBOLS)
        self.mock_logger.warning.assert_called()

    def test_validate_symbol_valid_formats(self):
        """Test symbol validation with various valid formats."""
        valid_symbols = [
            'BTCUSDT',
            'ETHUSDT', 
            'BTC-USDT',
            'ETH-USDT',
            'btcusdt',  # lowercase
            'eth-usdt'   # lowercase with dash
        ]
        
        for symbol in valid_symbols:
            with self.subTest(symbol=symbol):
                result = self.analyzer.validate_symbol(symbol)
                self.assertTrue(result, f"Symbol {symbol} should be valid")

    def test_validate_symbol_invalid_formats(self):
        """Test symbol validation with invalid formats."""
        invalid_symbols = [
            'INVALID',
            'BTC-ETH',
            'USDT-BTC',
            'BTC',
            'USDT',
            '',
            'BTC-USDT-INVALID'
        ]
        
        for symbol in invalid_symbols:
            with self.subTest(symbol=symbol):
                result = self.analyzer.validate_symbol(symbol)
                self.assertFalse(result, f"Symbol {symbol} should be invalid")

    def test_normalize_symbol(self):
        """Test symbol normalization to standard format."""
        test_cases = [
            ('BTC-USDT', 'BTCUSDT'),
            ('ETH-USDT', 'ETHUSDT'),
            ('BTCUSDT', 'BTCUSDT'),
            ('btc-usdt', 'BTCUSDT'),
            ('btcusdt', 'BTCUSDT')
        ]
        
        for input_symbol, expected_output in test_cases:
            with self.subTest(input=input_symbol):
                result = self.analyzer.normalize_symbol(input_symbol)
                self.assertEqual(result, expected_output)

    def test_clear_models_directory_success(self):
        """Test successful clearing of models directory."""
        # Create some dummy files in models directory
        dummy_files = ['model1.pth', 'model2.joblib', 'config.json']
        for filename in dummy_files:
            (Path(self.temp_dir) / filename).touch()
        
        # Verify files exist
        self.assertEqual(len(list(Path(self.temp_dir).glob('*'))), 3)
        
        # Clear directory
        self.analyzer.clear_models_directory()
        
        # Verify directory is empty but exists
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(len(list(Path(self.temp_dir).glob('*'))), 0)
        
        # Verify logger calls
        self.mock_logger.process.assert_called()
        self.mock_logger.success.assert_called()

    def test_clear_models_directory_failure(self):
        """Test clearing models directory when it doesn't exist."""
        # Remove directory
        shutil.rmtree(self.temp_dir)
        
        # Should not raise exception, should create directory
        self.analyzer.clear_models_directory()
        
        # Verify directory was created
        self.assertTrue(Path(self.temp_dir).exists())

    def test_clear_models_directory_permission_error(self):
        """Test clearing models directory with permission error."""
        # Mock shutil.rmtree to raise permission error
        with patch('shutil.rmtree', side_effect=PermissionError("Permission denied")):
            with self.assertRaises(PermissionError):
                self.analyzer.clear_models_directory()

    def test_prepare_combined_dataframe_success(self):
        """Test successful preparation of combined dataframe."""
        # Create mock symbol data
        mock_symbol_data = {
            'BTCUSDT': {
                '1h': pd.DataFrame({
                    'open': [100, 101, 102],
                    'close': [101, 102, 103],
                    'volume': [1000, 1100, 1200]
                }),
                '4h': pd.DataFrame({
                    'open': [100, 102],
                    'close': [102, 104],
                    'volume': [1000, 1200]
                })
            },
            'ETHUSDT': {
                '1h': pd.DataFrame({
                    'open': [200, 201, 202],
                    'close': [201, 202, 203],
                    'volume': [2000, 2100, 2200]
                })
            }
        }
        
        # Mock combine_all_dataframes
        with patch('trading_signal_analyzer.combine_all_dataframes') as mock_combine:
            mock_combined_df = pd.DataFrame({
                'symbol': ['BTCUSDT', 'BTCUSDT', 'ETHUSDT'],
                'timeframe': ['1h', '4h', '1h'],
                'open': [100, 100, 200],
                'close': [101, 102, 201]
            })
            mock_combine.return_value = mock_combined_df
            
            result = self.analyzer._prepare_combined_dataframe(mock_symbol_data)
            
            # Verify result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 3)
            mock_combine.assert_called_once()

    def test_prepare_combined_dataframe_empty_data(self):
        """Test preparation of combined dataframe with empty data."""
        # Empty symbol data
        empty_symbol_data = {}
        
        with patch('trading_signal_analyzer.combine_all_dataframes') as mock_combine:
            mock_combine.return_value = pd.DataFrame()
            
            result = self.analyzer._prepare_combined_dataframe(empty_symbol_data)
            
            # Should return empty dataframe
            self.assertTrue(result.empty)

    def test_prepare_combined_dataframe_mixed_data_types(self):
        """Test preparation with mixed data types (DataFrame and dict)."""
        mock_symbol_data = {
            'BTCUSDT': {
                '1h': pd.DataFrame({'open': [100], 'close': [101]})
            },
            'ETHUSDT': pd.DataFrame({'open': [200], 'close': [201]}),  # Direct DataFrame
            'ADAUSDT': None  # None data
        }
        
        with patch('trading_signal_analyzer.combine_all_dataframes') as mock_combine:
            mock_combine.return_value = pd.DataFrame({'symbol': ['BTCUSDT'], 'open': [100]})
            
            result = self.analyzer._prepare_combined_dataframe(mock_symbol_data)
            
            # Should handle mixed types gracefully
            self.assertIsInstance(result, pd.DataFrame)
            mock_combine.assert_called_once()

    def test_train_all_models_from_combined_data_success(self):
        """Test successful training of all models from combined data."""
        # Create mock combined dataframe
        mock_combined_df = pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT'] * 10,
            'timeframe': ['1h', '4h'] * 10,
            'open': np.random.randn(20),
            'close': np.random.randn(20),
            'volume': np.random.randn(20)
        })
        
        # Mock training functions
        with patch.object(self.analyzer, '_train_non_lstm_models') as mock_non_lstm:
            with patch.object(self.analyzer, '_train_lstm_models') as mock_lstm:
                self.analyzer._train_all_models_from_combined_data(mock_combined_df)
                
                # Verify both training methods were called
                mock_non_lstm.assert_called_once_with(mock_combined_df)
                mock_lstm.assert_called_once_with(mock_combined_df)
                
                # Verify logger calls
                self.mock_logger.model.assert_called()
                self.mock_logger.success.assert_called()

    def test_train_all_models_from_combined_data_empty_dataframe(self):
        """Test training with empty dataframe."""
        empty_df = pd.DataFrame()
        
        self.analyzer._train_all_models_from_combined_data(empty_df)
        
        # Should log error and return early
        self.mock_logger.error.assert_called()

    def test_train_non_lstm_models_success(self):
        """Test successful training of non-LSTM models."""
        mock_combined_df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 10,
            'open': np.random.randn(10),
            'close': np.random.randn(10)
        })
        
        # Mock training functions
        with patch('trading_signal_analyzer.train_and_save_global_rf_model') as mock_rf:
            with patch('trading_signal_analyzer.train_and_save_transformer_model') as mock_transformer:
                mock_rf.return_value = (Mock(), '/path/to/rf_model.joblib')
                mock_transformer.return_value = (Mock(), '/path/to/transformer_model.pth')
                
                self.analyzer._train_non_lstm_models(mock_combined_df)
                
                # Verify both models were trained
                mock_rf.assert_called_once_with(mock_combined_df, model_filename="rf_model_global.joblib")
                mock_transformer.assert_called_once_with(mock_combined_df, model_filename="transformer_model_global.pth")

    def test_train_non_lstm_models_with_exceptions(self):
        """Test training of non-LSTM models with exceptions."""
        mock_combined_df = pd.DataFrame({'symbol': ['BTCUSDT'] * 5})
        
        # Mock training functions to raise exceptions
        with patch('trading_signal_analyzer.train_and_save_global_rf_model', side_effect=Exception("RF failed")):
            with patch('trading_signal_analyzer.train_and_save_transformer_model', side_effect=Exception("Transformer failed")):
                # Should not raise exception, should log errors
                self.analyzer._train_non_lstm_models(mock_combined_df)
                
                # Verify error logging
                self.assertEqual(self.mock_logger.error.call_count, 2)

    def test_train_lstm_models_success(self):
        """Test successful training of LSTM models."""
        mock_combined_df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 20,
            'open': np.random.randn(20),
            'close': np.random.randn(20)
        })
        
        # Mock LSTM training function
        with patch('trading_signal_analyzer.train_cnn_lstm_attention_model') as mock_lstm_train:
            mock_lstm_train.return_value = (Mock(), '/path/to/lstm_model.pth')
            
            self.analyzer._train_lstm_models(mock_combined_df)
            
            # Should be called 12 times (4 base configs × 3 output modes)
            self.assertEqual(mock_lstm_train.call_count, 12)

    def test_train_single_lstm_model_success(self):
        """Test successful training of a single LSTM model."""
        mock_combined_df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 10,
            'open': np.random.randn(10),
            'close': np.random.randn(10)
        })
        
        with patch('trading_signal_analyzer.train_cnn_lstm_attention_model') as mock_train:
            mock_train.return_value = (Mock(), '/path/to/model.pth')
            
            self.analyzer._train_single_lstm_model(
                mock_combined_df, 'LSTM', False, False, 'classification'
            )
            
            # Verify training was called with correct parameters
            mock_train.assert_called_once()
            call_args = mock_train.call_args
            self.assertEqual(call_args[1]['use_cnn'], False)
            self.assertEqual(call_args[1]['use_attention'], False)
            self.assertEqual(call_args[1]['output_mode'], 'classification')

    def test_train_single_lstm_model_failure(self):
        """Test LSTM model training failure."""
        mock_combined_df = pd.DataFrame({'symbol': ['BTCUSDT'] * 5})
        
        with patch('trading_signal_analyzer.train_cnn_lstm_attention_model', side_effect=Exception("Training failed")):
            # Should not raise exception, should log error
            self.analyzer._train_single_lstm_model(
                mock_combined_df, 'LSTM', False, False, 'classification'
            )
            
            self.mock_logger.error.assert_called()

    def test_reload_all_models_success(self):
        """Test successful reload of all models."""
        # Mock load_all_symbols_data
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {
                'BTCUSDT': {
                    '1h': pd.DataFrame({
                        'open': [100, 101, 102],
                        'close': [101, 102, 103],
                        'volume': [1000, 1100, 1200]
                    })
                }
            }
            
            # Mock combine_all_dataframes
            with patch('trading_signal_analyzer.combine_all_dataframes') as mock_combine:
                mock_combine.return_value = pd.DataFrame({
                    'symbol': ['BTCUSDT'] * 3,
                    'open': [100, 101, 102],
                    'close': [101, 102, 103]
                })
                
                # Mock training methods
                with patch.object(self.analyzer, '_train_all_models_from_combined_data'):
                    self.analyzer.reload_all_models()
                    
                    # Verify all steps were called
                    mock_load_data.assert_called_once()
                    mock_combine.assert_called_once()
                    self.mock_logger.success.assert_called()

    def test_reload_all_models_no_data(self):
        """Test reload when no data is available."""
        with patch('trading_signal_analyzer.load_all_symbols_data', return_value=None):
            with self.assertRaises(RuntimeError):
                self.analyzer.reload_all_models()

    def test_reload_all_models_empty_combined_data(self):
        """Test reload when combined data is empty."""
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {'BTCUSDT': {'1h': pd.DataFrame()}}
            
            with patch('trading_signal_analyzer.combine_all_dataframes', return_value=pd.DataFrame()):
                with self.assertRaises(RuntimeError):
                    self.analyzer.reload_all_models()

    def test_analyze_best_performance_signals_success(self):
        """Test successful analysis of best performance signals."""
        # Mock load_all_symbols_data
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {
                'BTCUSDT': {'1h': pd.DataFrame({'close': [100, 101, 102]})},
                'ETHUSDT': {'1h': pd.DataFrame({'close': [200, 201, 202]})}
            }
            
            # Mock signal_best_performance_symbols
            with patch('trading_signal_analyzer.signal_best_performance_symbols') as mock_signal:
                mock_signal.return_value = {
                    'best_performers': [
                        {'symbol': 'BTCUSDT', 'score': 0.8},
                        {'symbol': 'ETHUSDT', 'score': 0.7}
                    ]
                }
                
                # Mock get_short_signal_candidates
                with patch('trading_signal_analyzer.get_short_signal_candidates') as mock_short:
                    mock_short.return_value = [
                        {'symbol': 'ADAUSDT', 'score': -0.6}
                    ]
                    
                    long_candidates, short_candidates = self.analyzer.analyze_best_performance_signals()
                    
                    # Verify results
                    self.assertEqual(len(long_candidates), 2)
                    self.assertEqual(len(short_candidates), 1)
                    self.assertEqual(long_candidates[0]['symbol'], 'BTCUSDT')
                    self.assertEqual(short_candidates[0]['symbol'], 'ADAUSDT')

    def test_analyze_best_performance_signals_no_data(self):
        """Test performance analysis when no data is available."""
        with patch('trading_signal_analyzer.load_all_symbols_data', return_value=None):
            long_candidates, short_candidates = self.analyzer.analyze_best_performance_signals()
            
            # Should return empty lists
            self.assertEqual(long_candidates, [])
            self.assertEqual(short_candidates, [])
            self.mock_logger.error.assert_called()

    def test_analyze_best_performance_signals_no_results(self):
        """Test performance analysis when no results are returned."""
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {'BTCUSDT': {'1h': pd.DataFrame({'close': [100]})}}
            
            with patch('trading_signal_analyzer.signal_best_performance_symbols', return_value=None):
                long_candidates, short_candidates = self.analyzer.analyze_best_performance_signals()
                
                # Should return empty lists
                self.assertEqual(long_candidates, [])
                self.assertEqual(short_candidates, [])
                self.mock_logger.error.assert_called()

    def test_filter_symbol_data(self):
        """Test filtering of symbol data."""
        # Test data with mixed types
        symbol_data = {
            'BTCUSDT': {
                '1h': pd.DataFrame({'close': [100, 101]}),
                '4h': pd.DataFrame({'close': [100, 102]})
            },
            'ETHUSDT': pd.DataFrame({'close': [200, 201]}),  # Direct DataFrame
            'ADAUSDT': None,  # None data
            'DOTUSDT': 'invalid'  # Invalid type
        }
        
        result = self.analyzer._filter_symbol_data(symbol_data)
        
        # Should only include valid DataFrame data
        self.assertIn('BTCUSDT', result)
        self.assertNotIn('ETHUSDT', result)  # Direct DataFrame not included
        self.assertNotIn('ADAUSDT', result)  # None data not included
        self.assertNotIn('DOTUSDT', result)  # Invalid type not included

    def test_check_symbol_in_performance_list_long_match(self):
        """Test checking symbol in performance list for LONG signal."""
        long_candidates = [
            {
                'symbol': 'BTCUSDT',
                'timeframe_scores': {'1h': 0.8, '4h': 0.7}
            }
        ]
        short_candidates = []
        
        result = self.analyzer.check_symbol_in_performance_list(
            'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
        )
        
        self.assertEqual(result, 1)  # Should match

    def test_check_symbol_in_performance_list_long_conflict(self):
        """Test checking symbol in performance list for LONG signal with conflict."""
        long_candidates = []
        short_candidates = [
            {
                'symbol': 'BTCUSDT',
                'timeframe_scores': {'1h': -0.6}
            }
        ]
        
        result = self.analyzer.check_symbol_in_performance_list(
            'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
        )
        
        self.assertEqual(result, -1)  # Should conflict

    def test_check_symbol_in_performance_list_short_match(self):
        """Test checking symbol in performance list for SHORT signal."""
        long_candidates = []
        short_candidates = [
            {
                'symbol': 'BTCUSDT',
                'timeframe_scores': {'1h': -0.6}
            }
        ]
        
        result = self.analyzer.check_symbol_in_performance_list(
            'BTCUSDT', '1h', 'SHORT', long_candidates, short_candidates
        )
        
        self.assertEqual(result, 1)  # Should match

    def test_check_symbol_in_performance_list_no_match(self):
        """Test checking symbol in performance list with no match."""
        long_candidates = []
        short_candidates = []
        
        result = self.analyzer.check_symbol_in_performance_list(
            'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
        )
        
        self.assertEqual(result, 0)  # No match

    def test_check_symbol_in_performance_list_exception_handling(self):
        """Test exception handling in performance list checking."""
        long_candidates = [{'invalid': 'data'}]  # Invalid data structure
        
        result = self.analyzer.check_symbol_in_performance_list(
            'BTCUSDT', '1h', 'LONG', long_candidates, []
        )
        
        self.assertEqual(result, 0)  # Should return 0 on error
        self.mock_logger.error.assert_called()

    def test_get_market_data_for_symbol_success(self):
        """Test successful retrieval of market data for symbol."""
        mock_market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {
                'BTCUSDT': {
                    '1h': mock_market_data
                }
            }
            
            result = self.analyzer._get_market_data_for_symbol('BTCUSDT', '1h')
            
            self.assertIsInstance(result, pd.DataFrame)
            if result is not None:  # Type guard
                self.assertEqual(len(result), 3)
                self.assertEqual(list(result.columns), ['open', 'close', 'volume'])

    def test_get_market_data_for_symbol_not_found(self):
        """Test market data retrieval when symbol not found."""
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {}  # Empty data
            
            result = self.analyzer._get_market_data_for_symbol('INVALID', '1h')
            
            self.assertIsNone(result)
            self.mock_logger.warning.assert_called()

    def test_get_market_data_for_symbol_none_data(self):
        """Test market data retrieval when symbol data is None."""
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {'BTCUSDT': None}
            
            result = self.analyzer._get_market_data_for_symbol('BTCUSDT', '1h')
            
            self.assertIsNone(result)

    def test_get_market_data_for_symbol_direct_dataframe(self):
        """Test market data retrieval when symbol data is direct DataFrame."""
        mock_market_data = pd.DataFrame({
            'open': [100, 101],
            'close': [101, 102]
        })
        
        with patch('trading_signal_analyzer.load_all_symbols_data') as mock_load_data:
            mock_load_data.return_value = {
                'BTCUSDT': mock_market_data  # Direct DataFrame
            }
            
            result = self.analyzer._get_market_data_for_symbol('BTCUSDT', '1h')
            
            self.assertIsInstance(result, pd.DataFrame)
            if result is not None:  # Type guard
                self.assertEqual(len(result), 2)

    def test_get_market_data_for_symbol_exception_handling(self):
        """Test exception handling in market data retrieval."""
        with patch('trading_signal_analyzer.load_all_symbols_data', side_effect=Exception("Data error")):
            result = self.analyzer._get_market_data_for_symbol('BTCUSDT', '1h')
            
            self.assertIsNone(result)
            self.mock_logger.error.assert_called()

    def test_calculate_signal_match_score_match(self):
        """Test signal match score calculation for matching signals."""
        result = self.analyzer._calculate_signal_match_score('LONG', 'LONG')
        self.assertEqual(result, 1)
        
        result = self.analyzer._calculate_signal_match_score('SHORT', 'SHORT')
        self.assertEqual(result, 1)

    def test_calculate_signal_match_score_conflict(self):
        """Test signal match score calculation for conflicting signals."""
        result = self.analyzer._calculate_signal_match_score('LONG', 'SHORT')
        self.assertEqual(result, -1)
        
        result = self.analyzer._calculate_signal_match_score('SHORT', 'LONG')
        self.assertEqual(result, -1)

    def test_calculate_signal_match_score_neutral(self):
        """Test signal match score calculation for neutral signals."""
        result = self.analyzer._calculate_signal_match_score('NEUTRAL', 'LONG')
        self.assertEqual(result, 0)
        
        result = self.analyzer._calculate_signal_match_score('', 'LONG')
        self.assertEqual(result, 0)

    def test_calculate_signal_match_score_case_insensitive(self):
        """Test signal match score calculation is case insensitive."""
        result = self.analyzer._calculate_signal_match_score('long', 'LONG')
        self.assertEqual(result, 1)
        
        result = self.analyzer._calculate_signal_match_score('LONG', 'long')
        self.assertEqual(result, 1)

    def test_calculate_final_threshold_positive_score(self):
        """Test final threshold calculation with positive score."""
        result = self.analyzer.calculate_final_threshold(5, 10)
        self.assertEqual(result, 0.75)  # (5 + 10) / (2 * 10) = 0.75

    def test_calculate_final_threshold_negative_score(self):
        """Test final threshold calculation with negative score."""
        result = self.analyzer.calculate_final_threshold(-3, 10)
        self.assertEqual(result, 0.35)  # (-3 + 10) / (2 * 10) = 0.35

    def test_calculate_final_threshold_zero_max_score(self):
        """Test final threshold calculation with zero max score."""
        result = self.analyzer.calculate_final_threshold(5, 0)
        self.assertEqual(result, 0.0)

    def test_calculate_final_threshold_boundary_values(self):
        """Test final threshold calculation with boundary values."""
        # Maximum positive score
        result = self.analyzer.calculate_final_threshold(10, 10)
        self.assertEqual(result, 1.0)
        
        # Maximum negative score
        result = self.analyzer.calculate_final_threshold(-10, 10)
        self.assertEqual(result, 0.0)
        
        # Zero score
        result = self.analyzer.calculate_final_threshold(0, 10)
        self.assertEqual(result, 0.5)

    def test_calculate_max_possible_score(self):
        """Test calculation of maximum possible score."""
        result = self.analyzer._calculate_max_possible_score()
        
        # Should be sum of all model max scores
        expected = 1 + 1 + 2 + 1 + 12  # performance + rf + hmm + transformer + lstm
        self.assertEqual(result, expected)

    def test_get_random_forest_signal_score_success(self):
        """Test successful Random Forest signal scoring."""
        mock_market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        # Mock model file existence
        rf_model_path = Path(self.temp_dir) / "rf_model_global.joblib"
        rf_model_path.touch()
        
        # Mock load_random_forest_model
        with patch('trading_signal_analyzer.load_random_forest_model') as mock_load_rf:
            mock_model = Mock()
            mock_load_rf.return_value = mock_model
            
            # Mock get_latest_random_forest_signal
            with patch('trading_signal_analyzer.get_latest_random_forest_signal') as mock_get_signal:
                mock_get_signal.return_value = ('LONG', 0.8)
                
                result = self.analyzer.get_random_forest_signal_score(mock_market_data, 'LONG')
                
                # Should return 1 for matching signal
                self.assertEqual(result, 1)
                mock_load_rf.assert_called_once()
                mock_get_signal.assert_called_once_with(mock_market_data, mock_model)

    def test_get_random_forest_signal_score_model_not_found(self):
        """Test Random Forest signal scoring when model file doesn't exist."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        result = self.analyzer.get_random_forest_signal_score(mock_market_data, 'LONG')
        
        # Should return 0 when model doesn't exist
        self.assertEqual(result, 0)

    def test_get_random_forest_signal_score_load_failure(self):
        """Test Random Forest signal scoring when model loading fails."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create model file but mock loading failure
        rf_model_path = Path(self.temp_dir) / "rf_model_global.joblib"
        rf_model_path.touch()
        
        with patch('trading_signal_analyzer.load_random_forest_model', return_value=None):
            result = self.analyzer.get_random_forest_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 when loading fails
            self.assertEqual(result, 0)

    def test_get_random_forest_signal_score_exception_handling(self):
        """Test Random Forest signal scoring exception handling."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create model file
        rf_model_path = Path(self.temp_dir) / "rf_model_global.joblib"
        rf_model_path.touch()
        
        with patch('trading_signal_analyzer.load_random_forest_model', side_effect=Exception("Load error")):
            result = self.analyzer.get_random_forest_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 on exception
            self.assertEqual(result, 0)
            self.mock_logger.error.assert_called()

    def test_get_hmm_signal_score_success(self):
        """Test successful HMM signal scoring."""
        mock_market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        # Mock hmm_signals
        with patch('trading_signal_analyzer.hmm_signals') as mock_hmm:
            mock_hmm.return_value = [1, -1]  # strict=1 (LONG), non_strict=-1 (SHORT)
            
            result = self.analyzer.get_hmm_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 (strict LONG matches = +1, non-strict SHORT conflicts = -1)
            self.assertEqual(result, 0)
            mock_hmm.assert_called_once_with(mock_market_data)

    def test_get_hmm_signal_score_both_match(self):
        """Test HMM signal scoring when both signals match."""
        mock_market_data = pd.DataFrame({'symbol': ['BTCUSDT'] * 3, 'close': [100, 101, 102]})
        
        with patch('trading_signal_analyzer.hmm_signals') as mock_hmm:
            mock_hmm.return_value = [1, 1]  # Both strict and non_strict are LONG
            
            result = self.analyzer.get_hmm_signal_score(mock_market_data, 'LONG')
            
            # Should return 2 (both match)
            self.assertEqual(result, 2)

    def test_get_hmm_signal_score_insufficient_signals(self):
        """Test HMM signal scoring with insufficient signals."""
        mock_market_data = pd.DataFrame({'symbol': ['BTCUSDT'] * 3, 'close': [100, 101, 102]})
        
        with patch('trading_signal_analyzer.hmm_signals') as mock_hmm:
            mock_hmm.return_value = [1]  # Only one signal
            
            result = self.analyzer.get_hmm_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 when insufficient signals
            self.assertEqual(result, 0)

    def test_get_hmm_signal_score_exception_handling(self):
        """Test HMM signal scoring exception handling."""
        mock_market_data = pd.DataFrame({'symbol': ['BTCUSDT'] * 3, 'close': [100, 101, 102]})
        
        with patch('trading_signal_analyzer.hmm_signals', side_effect=Exception("HMM error")):
            result = self.analyzer.get_hmm_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 on exception
            self.assertEqual(result, 0)
            self.mock_logger.error.assert_called()

    def test_get_transformer_signal_score_success(self):
        """Test successful Transformer signal scoring."""
        mock_market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        # Mock model file existence
        transformer_model_path = Path(self.temp_dir) / "transformer_model_global.pth"
        transformer_model_path.touch()
        
        # Mock load_transformer_model
        with patch('trading_signal_analyzer.load_transformer_model') as mock_load_transformer:
            mock_model_data = (Mock(), Mock(), ['feature1', 'feature2'], 0)
            mock_load_transformer.return_value = mock_model_data
            
            # Mock get_latest_transformer_signal
            with patch('trading_signal_analyzer.get_latest_transformer_signal') as mock_get_signal:
                mock_get_signal.return_value = 'LONG'
                
                result = self.analyzer.get_transformer_signal_score(mock_market_data, 'LONG')
                
                # Should return 1 for matching signal
                self.assertEqual(result, 1)
                mock_load_transformer.assert_called_once()
                mock_get_signal.assert_called_once()

    def test_get_transformer_signal_score_model_not_found(self):
        """Test Transformer signal scoring when model file doesn't exist."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        result = self.analyzer.get_transformer_signal_score(mock_market_data, 'LONG')
        
        # Should return 0 when model doesn't exist
        self.assertEqual(result, 0)

    def test_get_transformer_signal_score_load_failure(self):
        """Test Transformer signal scoring when model loading fails."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create model file but mock loading failure
        transformer_model_path = Path(self.temp_dir) / "transformer_model_global.pth"
        transformer_model_path.touch()
        
        with patch('trading_signal_analyzer.load_transformer_model', return_value=None):
            result = self.analyzer.get_transformer_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 when loading fails
            self.assertEqual(result, 0)

    def test_get_transformer_signal_score_invalid_model_data(self):
        """Test Transformer signal scoring with invalid model data."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create model file
        transformer_model_path = Path(self.temp_dir) / "transformer_model_global.pth"
        transformer_model_path.touch()
        
        with patch('trading_signal_analyzer.load_transformer_model') as mock_load_transformer:
            # Return invalid model data (None values)
            mock_load_transformer.return_value = (None, None, None, None)
            
            result = self.analyzer.get_transformer_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 when model data is invalid
            self.assertEqual(result, 0)

    def test_get_transformer_signal_score_exception_handling(self):
        """Test Transformer signal scoring exception handling."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create model file
        transformer_model_path = Path(self.temp_dir) / "transformer_model_global.pth"
        transformer_model_path.touch()
        
        with patch('trading_signal_analyzer.load_transformer_model', side_effect=Exception("Load error")):
            result = self.analyzer.get_transformer_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 on exception
            self.assertEqual(result, 0)
            self.mock_logger.error.assert_called()

    def test_get_lstm_signal_score_success(self):
        """Test successful LSTM signal scoring."""
        mock_market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        # Create mock model files for some LSTM variants
        lstm_files = [
            "lstm_classification_model_global.pth",
            "lstm_attention_regression_model_global.pth",
            "cnn_lstm_classification_advanced_model_global.pth"
        ]
        for filename in lstm_files:
            (Path(self.temp_dir) / filename).touch()
        
        # Mock load_cnn_lstm_attention_model
        with patch('trading_signal_analyzer.load_cnn_lstm_attention_model') as mock_load_lstm:
            mock_model_data = (Mock(), {'output_mode': 'classification'}, {}, {})
            mock_load_lstm.return_value = mock_model_data
            
            # Mock get_latest_cnn_lstm_attention_signal
            with patch('trading_signal_analyzer.get_latest_cnn_lstm_attention_signal') as mock_get_signal:
                mock_get_signal.return_value = 'LONG'
                
                result = self.analyzer.get_lstm_signal_score(mock_market_data, 'LONG')
                
                # Should return positive score for successful predictions
                self.assertGreater(result, 0)
                self.mock_logger.signal.assert_called()

    def test_get_lstm_signal_score_no_models(self):
        """Test LSTM signal scoring when no model files exist."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        result = self.analyzer.get_lstm_signal_score(mock_market_data, 'LONG')
        
        # Should return 0 when no models exist
        self.assertEqual(result, 0)

    def test_get_lstm_signal_score_load_failures(self):
        """Test LSTM signal scoring with model loading failures."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create mock model files
        lstm_files = [
            "lstm_classification_model_global.pth",
            "lstm_attention_regression_model_global.pth"
        ]
        for filename in lstm_files:
            (Path(self.temp_dir) / filename).touch()
        
        # Mock load_cnn_lstm_attention_model to return None
        with patch('trading_signal_analyzer.load_cnn_lstm_attention_model', return_value=None):
            result = self.analyzer.get_lstm_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 when all models fail to load
            self.assertEqual(result, 0)

    def test_get_lstm_signal_score_mixed_success_failure(self):
        """Test LSTM signal scoring with mixed success and failure."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create mock model files
        lstm_files = [
            "lstm_classification_model_global.pth",
            "lstm_attention_regression_model_global.pth"
        ]
        for filename in lstm_files:
            (Path(self.temp_dir) / filename).touch()
        
        # Mock load_cnn_lstm_attention_model to fail for first, succeed for second
        with patch('trading_signal_analyzer.load_cnn_lstm_attention_model') as mock_load_lstm:
            mock_load_lstm.side_effect = [None, (Mock(), {'output_mode': 'classification'}, {}, {})]
            
            with patch('trading_signal_analyzer.get_latest_cnn_lstm_attention_signal') as mock_get_signal:
                mock_get_signal.return_value = 'LONG'
                
                result = self.analyzer.get_lstm_signal_score(mock_market_data, 'LONG')
                
                # Should return score for successful prediction
                self.assertEqual(result, 1)
                self.mock_logger.signal.assert_called()

    def test_get_lstm_signal_score_exception_handling(self):
        """Test LSTM signal scoring exception handling."""
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        # Create mock model file
        (Path(self.temp_dir) / "lstm_classification_model_global.pth").touch()
        
        with patch('trading_signal_analyzer.load_cnn_lstm_attention_model', side_effect=Exception("LSTM error")):
            result = self.analyzer.get_lstm_signal_score(mock_market_data, 'LONG')
            
            # Should return 0 on exception
            self.assertEqual(result, 0)

    def test_analyze_symbol_signal_success(self):
        """Test successful comprehensive symbol signal analysis."""
        # Mock performance analysis
        long_candidates = [{'symbol': 'BTCUSDT', 'timeframe_scores': {'1h': 0.8}}]
        short_candidates = []
        
        # Mock market data
        mock_market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        # Mock _get_market_data_for_symbol
        with patch.object(self.analyzer, '_get_market_data_for_symbol', return_value=mock_market_data):
            # Mock all signal scoring methods
            with patch.object(self.analyzer, 'get_random_forest_signal_score', return_value=1):
                with patch.object(self.analyzer, 'get_hmm_signal_score', return_value=2):
                    with patch.object(self.analyzer, 'get_transformer_signal_score', return_value=1):
                        with patch.object(self.analyzer, 'get_lstm_signal_score', return_value=6):
                            result = self.analyzer.analyze_symbol_signal(
                                'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
                            )
                            
                            # Verify result structure
                            self.assertIn('symbol', result)
                            self.assertIn('timeframe', result)
                            self.assertIn('signal', result)
                            self.assertIn('scores', result)
                            self.assertIn('total_score', result)
                            self.assertIn('threshold', result)
                            self.assertIn('recommendation', result)
                            
                            # Verify values
                            self.assertEqual(result['symbol'], 'BTCUSDT')
                            self.assertEqual(result['timeframe'], '1h')
                            self.assertEqual(result['signal'], 'LONG')
                            scores = result['scores']
                            self.assertEqual(scores['performance'], 1)  #  type: ignore 
                            self.assertEqual(scores['random_forest'], 1) # type: ignore
                            self.assertEqual(scores['hmm'], 2) # type: ignore
                            self.assertEqual(scores['transformer'], 1) # type: ignore
                            self.assertEqual(scores['lstm'], 6) # type: ignore
                            self.assertEqual(result['total_score'], 11)  # 1+1+2+1+6

    def test_analyze_symbol_signal_no_market_data(self):
        """Test symbol signal analysis when no market data is available."""
        long_candidates = []
        short_candidates = []
        
        # Mock _get_market_data_for_symbol to return None
        with patch.object(self.analyzer, '_get_market_data_for_symbol', return_value=None):
            result = self.analyzer.analyze_symbol_signal(
                'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
            )
            
            # Should have zero scores for model-based signals
            scores = result['scores']
            self.assertEqual(scores['random_forest'], 0) # type: ignore
            self.assertEqual(scores['hmm'], 0) # type: ignore
            self.assertEqual(scores['transformer'], 0) # type: ignore
            self.assertEqual(scores['lstm'], 0) # type: ignore
            self.mock_logger.warning.assert_called()

    def test_analyze_symbol_signal_empty_market_data(self):
        """Test symbol signal analysis with empty market data."""
        long_candidates = []
        short_candidates = []
        
        # Mock _get_market_data_for_symbol to return empty DataFrame
        with patch.object(self.analyzer, '_get_market_data_for_symbol', return_value=pd.DataFrame()):
            result = self.analyzer.analyze_symbol_signal(
                'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
            )
            
            # Should have zero scores for model-based signals
            scores = result['scores']
            self.assertEqual(scores['random_forest'], 0) # type: ignore
            self.assertEqual(scores['hmm'], 0) # type: ignore
            self.assertEqual(scores['transformer'], 0) # type: ignore
            self.assertEqual(scores['lstm'], 0) # type: ignore
            self.mock_logger.warning.assert_called()

    def test_analyze_symbol_signal_threshold_calculation(self):
        """Test threshold calculation in symbol signal analysis."""
        long_candidates = []
        short_candidates = []
        
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        with patch.object(self.analyzer, '_get_market_data_for_symbol', return_value=mock_market_data):
            # Mock all signal scoring methods to return 0
            with patch.object(self.analyzer, 'get_random_forest_signal_score', return_value=0):
                with patch.object(self.analyzer, 'get_hmm_signal_score', return_value=0):
                    with patch.object(self.analyzer, 'get_transformer_signal_score', return_value=0):
                        with patch.object(self.analyzer, 'get_lstm_signal_score', return_value=0):
                            result = self.analyzer.analyze_symbol_signal(
                                'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
                            )
                            
                            # Should have threshold of 0.5 (neutral) when all scores are 0
                            self.assertEqual(result['threshold'], 0.5)
                            self.assertEqual(result['recommendation'], 'WAIT')

    def test_analyze_symbol_signal_high_confidence(self):
        """Test symbol signal analysis with high confidence (should recommend ENTER)."""
        long_candidates = []
        short_candidates = []
        
        mock_market_data = pd.DataFrame({'close': [100, 101, 102]})
        
        with patch.object(self.analyzer, '_get_market_data_for_symbol', return_value=mock_market_data):
            # Mock high scores to achieve high confidence
            with patch.object(self.analyzer, 'get_random_forest_signal_score', return_value=1):
                with patch.object(self.analyzer, 'get_hmm_signal_score', return_value=2):
                    with patch.object(self.analyzer, 'get_transformer_signal_score', return_value=1):
                        with patch.object(self.analyzer, 'get_lstm_signal_score', return_value=12):
                            result = self.analyzer.analyze_symbol_signal(
                                'BTCUSDT', '1h', 'LONG', long_candidates, short_candidates
                            )
                            
                            # Should have high threshold and recommend ENTER
                            threshold = result['threshold']
                            self.assertGreater(threshold, 0.7) # type: ignore
                            self.assertEqual(result['recommendation'], 'ENTER')

    def test_run_analysis_success(self):
        """Test successful run_analysis with valid inputs."""
        # Mock analyze_best_performance_signals
        with patch.object(self.analyzer, 'analyze_best_performance_signals') as mock_performance:
            mock_performance.return_value = ([], [])  # Empty candidates
            
            # Mock analyze_symbol_signal
            with patch.object(self.analyzer, 'analyze_symbol_signal') as mock_analyze:
                mock_result = {
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'signal': 'LONG',
                    'scores': {'performance': 1, 'random_forest': 1, 'hmm': 2, 'transformer': 1, 'lstm': 6},
                    'total_score': 11,
                    'threshold': 0.75,
                    'recommendation': 'ENTER'
                }
                mock_analyze.return_value = mock_result
                
                result = self.analyzer.run_analysis(
                    reload_model=False,
                    symbol='BTCUSDT',
                    timeframe='1h',
                    signal='LONG'
                )
                
                # Should return the analysis result
                self.assertEqual(result, mock_result)
                mock_performance.assert_called_once()
                mock_analyze.assert_called_once()

    def test_run_analysis_with_model_reload(self):
        """Test run_analysis with model reload enabled."""
        # Mock reload_all_models
        with patch.object(self.analyzer, 'reload_all_models') as mock_reload:
            # Mock analyze_best_performance_signals
            with patch.object(self.analyzer, 'analyze_best_performance_signals') as mock_performance:
                mock_performance.return_value = ([], [])
                
                # Mock analyze_symbol_signal
                with patch.object(self.analyzer, 'analyze_symbol_signal') as mock_analyze:
                    mock_result = {
                        'symbol': 'BTCUSDT',
                        'timeframe': '1h',
                        'signal': 'LONG',
                        'scores': {'performance': 0, 'random_forest': 0, 'hmm': 0, 'transformer': 0, 'lstm': 0},
                        'total_score': 0,
                        'threshold': 0.5,
                        'recommendation': 'WAIT'
                    }
                    mock_analyze.return_value = mock_result
                    
                    result = self.analyzer.run_analysis(
                        reload_model=True,
                        symbol='BTCUSDT',
                        timeframe='1h',
                        signal='LONG'
                    )
                    
                    # Should reload models first
                    mock_reload.assert_called_once()
                    self.assertEqual(result, mock_result)

    def test_run_analysis_invalid_symbol(self):
        """Test run_analysis with invalid symbol."""
        result = self.analyzer.run_analysis(
            reload_model=False,
            symbol='INVALID',
            timeframe='1h',
            signal='LONG'
        )
        
        # Should return error
        self.assertIn('error', result)
        self.assertIn('Invalid symbol', result['error'])

    def test_run_analysis_invalid_timeframe(self):
        """Test run_analysis with invalid timeframe."""
        result = self.analyzer.run_analysis(
            reload_model=False,
            symbol='BTCUSDT',
            timeframe='invalid',
            signal='LONG'
        )
        
        # Should return error
        self.assertIn('error', result)
        self.assertIn('Invalid timeframe', result['error'])

    def test_run_analysis_invalid_signal(self):
        """Test run_analysis with invalid signal."""
        result = self.analyzer.run_analysis(
            reload_model=False,
            symbol='BTCUSDT',
            timeframe='1h',
            signal='INVALID'
        )
        
        # Should return error
        self.assertIn('error', result)
        self.assertIn('Invalid signal', result['error'])

    def test_run_analysis_exception_handling(self):
        """Test run_analysis exception handling."""
        # Mock analyze_best_performance_signals to raise exception
        with patch.object(self.analyzer, 'analyze_best_performance_signals', side_effect=Exception("Analysis failed")):
            result = self.analyzer.run_analysis(
                reload_model=False,
                symbol='BTCUSDT',
                timeframe='1h',
                signal='LONG'
            )
            
            # Should return error
            self.assertIn('error', result)
            self.assertEqual(result['error'], 'Analysis failed')
            self.mock_logger.error.assert_called()

    def test_print_analysis_result_success(self):
        """Test print_analysis_result with successful result."""
        from trading_signal_analyzer import print_analysis_result
        
        result = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'signal': 'LONG',
            'scores': {
                'performance': 1, 'random_forest': 1, 'hmm': 2,
                'transformer': 1, 'lstm': 6
            },
            'total_score': 11,
            'max_possible_score': 17,
            'threshold': 0.75,
            'recommendation': 'ENTER'
        }
        
        print_analysis_result(result)
        
        # Check that various logger methods are called
        self.assertGreater(self.mock_logger.analysis.call_count, 2)
        self.mock_logger.data.assert_called()
        self.mock_logger.signal.assert_called()
        self.mock_logger.success.assert_called_once()
        
        # Verify content of a specific call
        self.mock_logger.success.assert_called_with(
            "  • Recommendation:      ✅ ENTER (Confidence ≥ 0.7)"
        )

    def test_print_analysis_result_error(self):
        """Test print_analysis_result with error result."""
        from trading_signal_analyzer import print_analysis_result
        
        result = {'error': 'Invalid symbol'}
        print_analysis_result(result)
        
        # Should call logger.error once
        self.mock_logger.error.assert_called_once_with("Analysis failed: Invalid symbol")

    def test_print_analysis_result_zero_scores(self):
        """Test print_analysis_result with zero scores."""
        from trading_signal_analyzer import print_analysis_result
        
        result = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'signal': 'LONG',
            'scores': {
                'performance': 0, 'random_forest': 0, 'hmm': 0,
                'transformer': 0, 'lstm': 0
            },
            'total_score': 0,
            'max_possible_score': 17,
            'threshold': 0.5,
            'recommendation': 'WAIT'
        }
        
        print_analysis_result(result)
        
        # Should call logger.warning for WAIT recommendation
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args[0][0]
        self.assertIn('WAIT', call_args)

    def test_main_function_import(self):
        """Test that main function can be imported."""
        from trading_signal_analyzer import main
        
        # Should be callable
        self.assertTrue(callable(main))

    def test_analyzer_class_methods_exist(self):
        """Test that all expected methods exist in TradingSignalAnalyzer."""
        expected_methods = [
            'validate_symbol',
            'normalize_symbol',
            'clear_models_directory',
            'reload_all_models',
            'analyze_best_performance_signals',
            'check_symbol_in_performance_list',
            'get_random_forest_signal_score',
            'get_hmm_signal_score',
            'get_transformer_signal_score',
            'get_lstm_signal_score',
            'calculate_final_threshold',
            'analyze_symbol_signal',
            'run_analysis'
        ]
        
        for method_name in expected_methods:
            with self.subTest(method=method_name):
                self.assertTrue(hasattr(self.analyzer, method_name))
                self.assertTrue(callable(getattr(self.analyzer, method_name)))

    def test_analyzer_class_attributes_exist(self):
        """Test that all expected attributes exist in TradingSignalAnalyzer."""
        expected_attributes = [
            'valid_timeframes',
            'valid_symbols',
            'processor'
        ]
        
        for attr_name in expected_attributes:
            with self.subTest(attribute=attr_name):
                self.assertTrue(hasattr(self.analyzer, attr_name))


if __name__ == '__main__':
    unittest.main() 