import argparse
import logging
import os
import pandas as pd
import sys
from pathlib import Path
from time import sleep
from typing import List, Optional, Dict, Union

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from livetrade._components._load_all_pairs_data import load_all_pairs_data
from livetrade._components._tick_processor import tick_processor
from signals._components._process_signals_hmm import process_signals_hmm
from signals._components._process_signals_random_forest import process_signals_random_forest
from signals.signals_best_performance_pairs import signal_best_performance_pairs
from livetrade._components._combine_all_dataframes import combine_all_dataframes 
from signals.signals_random_forest import train_and_save_global_rf_model

from utilities._logger import setup_logging
logger = setup_logging(module_name="signal_combine_crypto", log_level=logging.INFO)

terminate = False

def _prepare_combined_dataframe_for_rf(data_source: Dict, 
                                       existing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Prepares a combined and normalized DataFrame for RF model training.
    Uses existing_df if provided and valid, otherwise creates from data_source.
    """
    if existing_df is not None and not existing_df.empty:
        logger.info("Using existing pre-combined DataFrame for RF training.")
        # Ensure normalization, though ideally already done by caller if providing existing_df
        if 'Open' not in existing_df.columns and 'open' in existing_df.columns:
            existing_df['Open'] = existing_df['open']
        if 'Close' not in existing_df.columns and 'close' in existing_df.columns:
            existing_df['Close'] = existing_df['close']
        return existing_df

    logger.info("Creating new combined DataFrame for RF training from data_source.")
    combined_df = combine_all_dataframes(data_source)
    
    if combined_df.empty:
        logger.warning("Failed to create a combined DataFrame from data_source.")
        return pd.DataFrame() # Return empty DataFrame
    
    logger.info(f"Combined DataFrame created, shape: {combined_df.shape}")
    # Normalize columns
    if 'Open' not in combined_df.columns and 'open' in combined_df.columns:
        combined_df['Open'] = combined_df['open']
    if 'Close' not in combined_df.columns and 'close' in combined_df.columns:
        combined_df['Close'] = combined_df['close']
    logger.info("Column normalization performed on the newly combined DataFrame.")
    return combined_df

def _debug_rf_hmm_chain(rf_long_signals, rf_short_signals, 
                        strict_hmm_long_pairs, strict_hmm_short_pairs,
                        non_strict_hmm_long_pairs, 
                        non_strict_hmm_short_pairs,
                        final_long_signals, final_short_signals, logger):
    """✅ FIXED: Debug logging với direction phân biệt"""
    
    all_hmm_long_pairs = strict_hmm_long_pairs.union(non_strict_hmm_long_pairs)
    all_hmm_short_pairs = strict_hmm_short_pairs.union(non_strict_hmm_short_pairs)
    
    logger.info("\n" + "="*60)
    logger.info("🔍 DEBUG: RF_HMM_CHAIN ANALYSIS (FIXED)")
    logger.info("="*60)
    
    logger.info(f"\n📊 RF SIGNAL BREAKDOWN:")
    logger.info(f"  • RF LONG signals: {len(rf_long_signals)}")
    logger.info(f"  • RF SHORT signals: {len(rf_short_signals)}")
    
    logger.info(f"\n🎯 HMM PAIRS BREAKDOWN BY DIRECTION:")
    logger.info(f"  • Strict HMM LONG pairs: {len(strict_hmm_long_pairs)}")
    logger.info(f"  • Strict HMM SHORT pairs: {len(strict_hmm_short_pairs)}")
    logger.info(f"  • Non-strict HMM LONG pairs: {len(non_strict_hmm_long_pairs)}")
    logger.info(f"  • Non-strict HMM SHORT pairs: {len(non_strict_hmm_short_pairs)}")
    logger.info(f"  • Combined HMM LONG pairs: {len(all_hmm_long_pairs)}")
    logger.info(f"  • Combined HMM SHORT pairs: {len(all_hmm_short_pairs)}")
    
    logger.info(f"\n✅ FINAL RESULTS:")
    logger.info(f"  • Final LONG signals: {len(final_long_signals)}")
    logger.info(f"  • Final SHORT signals: {len(final_short_signals)}")
    
    logger.info(f"\n🔍 DIRECTION-SPECIFIC MATCHING:")
    
    # RF LONG vs HMM LONG matching
    matched_long = 0
    filtered_long = 0
    for rf_signal in rf_long_signals:
        signal_key = (rf_signal['pair'], rf_signal['timeframe'])
        if signal_key in all_hmm_long_pairs:
            matched_long += 1
        else:
            filtered_long += 1
    
    # RF SHORT vs HMM SHORT matching
    matched_short = 0
    filtered_short = 0
    for rf_signal in rf_short_signals:
        signal_key = (rf_signal['pair'], rf_signal['timeframe'])
        if signal_key in all_hmm_short_pairs:
            matched_short += 1
        else:
            filtered_short += 1
    
    logger.info(f"\n📊 DIRECTION MATCHING SUMMARY:")
    logger.info(f"  • LONG: {matched_long} matched, {filtered_long} filtered")
    logger.info(f"  • SHORT: {matched_short} matched, {filtered_short} filtered")
    
    logger.info("="*60)
    logger.info("🔍 END RF_HMM_CHAIN DEBUG (FIXED)")
    logger.info("="*60)

def _debug_signal_filtering(performance_long_signals, performance_short_signals, rf_hmm_long_signals, rf_hmm_short_signals, logger):
    """Debug logging for comprehensive signal filtering analysis"""
    logger.info("\n" + "="*80)
    logger.info("🔍 DEBUG: COMPREHENSIVE SIGNAL FILTERING ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\n📊 PERFORMANCE ANALYSIS RESULTS:")
    logger.info(f"  • Performance LONG signals: {len(performance_long_signals)}")
    logger.info(f"  • Performance SHORT signals: {len(performance_short_signals)}")
    
    logger.info(f"\n🎯 RF_HMM_CHAIN RESULTS:")
    logger.info(f"  • RF_HMM LONG signals: {len(rf_hmm_long_signals)}")
    logger.info(f"  • RF_HMM SHORT signals: {len(rf_hmm_short_signals)}")
    
    perf_long_symbols = set([sig['pair'] for sig in performance_long_signals])
    perf_short_symbols = set([sig['pair'] for sig in performance_short_signals])
    rf_hmm_long_symbols = set([sig['pair'] for sig in rf_hmm_long_signals])
    rf_hmm_short_symbols = set([sig['pair'] for sig in rf_hmm_short_signals])
    
    logger.info(f"\n🔍 SYMBOL ANALYSIS:")
    logger.info(f"  • Performance LONG symbols: {len(perf_long_symbols)}")
    logger.info(f"  • Performance SHORT symbols: {len(perf_short_symbols)}")
    logger.info(f"  • RF_HMM LONG symbols: {len(rf_hmm_long_symbols)}")
    logger.info(f"  • RF_HMM SHORT symbols: {len(rf_hmm_short_symbols)}")
    
    common_long_symbols = perf_long_symbols.intersection(rf_hmm_long_symbols)
    common_short_symbols = perf_short_symbols.intersection(rf_hmm_short_symbols)
    
    logger.info(f"  • Common LONG symbols: {len(common_long_symbols)}")
    logger.info(f"  • Common SHORT symbols: {len(common_short_symbols)}")
    
    if common_long_symbols:
        logger.info(f"    Common LONG symbols: {', '.join(sorted(common_long_symbols))}")
    
    if common_short_symbols:
        logger.info(f"    Common SHORT symbols: {', '.join(sorted(common_short_symbols))}")
    
    logger.info("="*80)
    logger.info("🔍 END COMPREHENSIVE DEBUG ANALYSIS")
    logger.info("="*80)

def crypto_signal_workflow(processor, timeframes_performance: Optional[List[str]] = None, timeframes_hmm: Optional[List[str]] = None):
    """
    Comprehensive crypto signal generation workflow combining performance analysis with RF_HMM_chain filtering.
    
    Args:
        processor: The tick processor instance
        timeframes_performance: Timeframes for performance analysis and RF
        timeframes_hmm: Timeframes for HMM analysis
    
    Returns:
        List of signal dictionaries with keys: pair, direction, timeframe
    """
    global terminate
    
    cpu_cores = os.cpu_count()
    max_workers_cpu = max(1, int(cpu_cores * 0.8)) if cpu_cores else 2
    logger.info(f"Using {max_workers_cpu} workers")

    timeframes_performance = timeframes_performance or ['15m', '30m', '1h', '4h', '1d']
    timeframes_hmm = timeframes_hmm or ['5m', '15m', '30m', '1h', '4h']
    
    # Combine all unique timeframes to load data only once
    all_timeframes = list(set(timeframes_performance + timeframes_hmm))
    
    logger.info("="*80)
    logger.info("STARTING CRYPTO SIGNAL WORKFLOW")
    logger.info("="*80)
    
    try:
        logger.info("\nSTEP 1: Loading all pairs data...")
        all_symbols = processor.get_symbols_list_by_quote_usdt()
        logger.info(f"Found {len(all_symbols)} available crypto symbols")
        
        if terminate:
            raise KeyboardInterrupt("Termination requested during data loading")
        
        # Load data for ALL required timeframes at once
        preloaded_data = load_all_pairs_data(
            processor=processor,
            symbols=all_symbols,
            load_multi_timeframes=True,
            timeframes=all_timeframes  # Use combined timeframes
        )
        
        if not preloaded_data:
            logger.error("Failed to load data")
            return []
        
        logger.info(f"Successfully loaded data for {len(preloaded_data)} symbols")

        # Prepare combined DataFrame for RF training once
        logger.info("\nSTEP 1: Preparing combined DataFrame for RF model training...")
        combined_df_for_rf_training = _prepare_combined_dataframe_for_rf(preloaded_data)
        
        if combined_df_for_rf_training.empty:
            logger.error("Failed to create a combined DataFrame from preloaded_data for RF training. Workflow might be impacted.")
            # Allow continuation, RF_HMM_chain will handle empty df if passed

        logger.info("\nSTEP 2: Analyzing performance for LONG and SHORT signal generation...")
        if terminate:
            raise KeyboardInterrupt("Termination requested during performance analysis")
        
        performance_result_long = signal_best_performance_pairs(
            processor=processor,
            symbols=list(preloaded_data.keys()),
            timeframes=timeframes_performance,
            performance_period=48, 
            top_percentage=0.4,
            min_volume_usdt=50000,  
            exclude_stable_coins=True,
            include_short_signals=False,
            preloaded_data=preloaded_data # type: ignore
        )
        
        if not performance_result_long or 'best_performers' not in performance_result_long:
            logger.error("Failed to get performance analysis results for LONG signals")
            return []
        
        performance_result_short = signal_best_performance_pairs(
            processor=processor,
            symbols=list(preloaded_data.keys()),
            timeframes=timeframes_performance,
            performance_period=48, 
            top_percentage=0.0,
            include_short_signals=True,
            worst_percentage=0.4,
            min_volume_usdt=50000,  
            exclude_stable_coins=True,
            preloaded_data=preloaded_data # type: ignore
        )
        
        if not performance_result_short:
            logger.error("Failed to get performance analysis results for SHORT signals")
            return []
        
        best_performers_long = performance_result_long['best_performers']
        best_performer_symbols_long = [performer['symbol'] for performer in best_performers_long]
        logger.info(f"Found {len(best_performers_long)} top performing symbols for LONG signals")

        worst_performers_short = performance_result_short.get('worst_performers', [])
        if 'worst_performers' not in performance_result_short:
            logger.warning("No worst_performers found in performance_result_short")
        
        worst_performer_symbols_short = [performer['symbol'] for performer in worst_performers_short]
        logger.info(f"Found {len(worst_performers_short)} worst performing symbols for SHORT signals")
        
        # Create filtered performance data for optimized RF_HMM_chain processing
        logger.info("\nSTEP 2.5: Creating filtered performance data for optimized processing...")
        
        best_performance_data = {}
        for symbol in best_performer_symbols_long:
            if symbol in preloaded_data:
                best_performance_data[symbol] = preloaded_data[symbol]
        
        worst_performance_data = {}
        for symbol in worst_performer_symbols_short:
            if symbol in preloaded_data:
                worst_performance_data[symbol] = preloaded_data[symbol]
        
        logger.info(f"Created filtered data: {len(best_performance_data)} best performers, {len(worst_performance_data)} worst performers")
        
        logger.info("\nSTEP 3: Using optimized RF_HMM_chain for LONG and SHORT signal processing...")
        if terminate:
            raise KeyboardInterrupt("Termination requested during RF_HMM_chain analysis")

        logger.info("Running optimized RF_HMM_chain with performance-filtered data...")
        rf_hmm_result = RF_HMM_chain(
            processor=processor,
            pairs_input=preloaded_data,                             # Full data for RF model training
            best_performance_data=best_performance_data,            # Filtered best performers
            worst_performance_data=worst_performance_data,          # Filtered worst performers
            timeframes_rf=timeframes_performance,
            timeframes_hmm=timeframes_hmm,
            skip_data_loading=True,                                 # Data is already loaded
            pre_combined_df_for_rf_training=combined_df_for_rf_training
        )
        
        if not rf_hmm_result:
            logger.error("RF_HMM_chain returned empty results")
            return []
        
        # Get final signals from optimized RF_HMM_chain
        rf_hmm_long_signals = rf_hmm_result.get('long_signals', [])
        rf_hmm_short_signals = rf_hmm_result.get('short_signals', [])
        
        logger.info(f"Optimized RF_HMM_chain generated {len(rf_hmm_long_signals)} LONG and {len(rf_hmm_short_signals)} SHORT signals")
        
        # Filter LONG signals based on best performers
        final_long_signals = []
        for signal in rf_hmm_long_signals:
            if signal['pair'] in best_performer_symbols_long:
                final_long_signals.append(signal)
        
        # Filter SHORT signals based on worst performers  
        final_short_signals = []
        for signal in rf_hmm_short_signals:
            if signal['pair'] in worst_performer_symbols_short:
                final_short_signals.append(signal)
        
        logger.info(f"After performance filtering: {len(final_long_signals)} LONG signals, {len(final_short_signals)} SHORT signals")
        
        final_signals = final_long_signals + final_short_signals
        
        long_count = len(final_long_signals)
        short_count = len(final_short_signals)
        
        clean_final_signals = []
        for signal in final_signals:
            clean_signal = {
                'pair': signal['pair'],
                'direction': signal['direction'],
                'timeframe': signal['timeframe']
            }
            clean_final_signals.append(clean_signal)
        
        logger.info("="*80)
        logger.info("COMPREHENSIVE WORKFLOW RESULTS:")
        logger.info(f"  • Performance Analysis - Top performers (LONG): {len(best_performer_symbols_long)}")
        logger.info(f"  • Performance Analysis - Worst performers (SHORT): {len(worst_performer_symbols_short)}")
        logger.info(f"  • RF_HMM_chain - Final LONG signals: {len(final_long_signals)}")
        logger.info(f"  • RF_HMM_chain - Final SHORT signals: {len(final_short_signals)}")
        logger.info(f"  • Total final signals: {len(clean_final_signals)}")
        logger.info("="*80)
        
        if clean_final_signals:
            logger.info("DETAILED TRADING SIGNALS:")
            
            if long_count > 0:
                logger.info("\n🟢 LONG SIGNALS:")
                for idx, signal in enumerate(sorted([s for s in clean_final_signals if s['direction'] == 'LONG'], key=lambda x: x['pair'])):
                    logger.info(f"  L{idx+1}. {signal['pair']} on {signal['timeframe']}")
            
            if short_count > 0:
                logger.info("\n🔴 SHORT SIGNALS:")
                for idx, signal in enumerate(sorted([s for s in clean_final_signals if s['direction'] == 'SHORT'], key=lambda x: x['pair'])):
                    logger.info(f"  S{idx+1}. {signal['pair']} on {signal['timeframe']}")
            
            logger.info("\n📊 TRADING SUMMARY (All Signals):")
            for idx, signal in enumerate(sorted(clean_final_signals, key=lambda x: (x['direction'], x['pair']))):
                symbol = signal['pair']
                direction = signal['direction']
                timeframe = signal['timeframe']
                
                direction_fmt = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                
                extra_info = ""
                if direction == "LONG" and symbol in best_performer_symbols_long:
                    performer = next((p for p in best_performers_long if p['symbol'] == symbol), None)
                    if performer:
                        score = performer.get('composite_score', 'N/A')
                        extra_info = f"(Performance Score: {score:.4f})"
                elif direction == "SHORT" and symbol in worst_performer_symbols_short:
                    performer = next((p for p in worst_performers_short if p['symbol'] == symbol), None)
                    if performer:
                        score = performer.get('composite_score', 'N/A')
                        extra_info = f"(Performance Score: {score:.4f})"
                
                logger.info(f"  {idx+1}. {symbol} - {direction_fmt} on {timeframe} {extra_info}")
        
        logger.info("\n🔍 RUNNING COMPREHENSIVE DEBUG ANALYSIS...")
        _debug_signal_filtering(
            performance_long_signals=[{'pair': symbol} for symbol in best_performer_symbols_long],
            performance_short_signals=[{'pair': symbol} for symbol in worst_performer_symbols_short],
            rf_hmm_long_signals=final_long_signals,        # Use final signals for debug
            rf_hmm_short_signals=final_short_signals,      # Use final signals for debug
            logger=logger
        )
        
        logger.info(f"Comprehensive workflow completed: {len(clean_final_signals)} total signals ({long_count} LONG, {short_count} SHORT)")
        return clean_final_signals
        
    except KeyboardInterrupt:
        logger.warning("Crypto workflow interrupted by user or termination signal")
        return []
    except Exception as e:
        logger.error(f"Error in crypto workflow: {e}", exc_info=True)
        return []

def final_signals(timeframes_performance: Optional[List[str]] = None, timeframes_hmm: Optional[List[str]] = None, log_level: str = 'INFO'):
    """
    Comprehensive crypto signal generation using performance analysis combined with RF_HMM_chain filtering.
    
    Args:
        timeframes_performance: Timeframes for performance analysis and RF
        timeframes_hmm: Timeframes for HMM analysis
        log_level: Logging level

    Returns:
        List of signal dictionaries containing: pair, direction, timeframe
    """
    global terminate
    terminate = False
    processor = None 

    def trade_open_callback_handler(trade_data=None, command=None, signal_data=None):
        logger.info(f"Trade open event: trade_data={trade_data}, command={command}, signal_data={signal_data}")

    def trade_close_callback_handler(trade_data=None, command=None, signal_data=None):
        logger.info(f"Trade close event: trade_data={trade_data}, command={command}, signal_data={signal_data}")

    timeframes_performance = timeframes_performance or ['15m', '30m', '1h', '4h', '1d']
    timeframes_hmm = timeframes_hmm or ['5m', '15m', '30m', '1h', '4h']

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    try:
        logger.info("Initializing processor with Binance API...")
        processor = tick_processor(trade_open_callback=trade_open_callback_handler, trade_close_callback=trade_close_callback_handler)
        sleep(3)

        if terminate:
            logger.warning("Termination requested during processor initialization.")
            raise KeyboardInterrupt("Termination requested during processor initialization")

        final_signals_result = crypto_signal_workflow(
            processor=processor,
            timeframes_performance=timeframes_performance,
            timeframes_hmm=timeframes_hmm
        )

        logger.info(f"Crypto workflow completed. Generated {len(final_signals_result)} signals.")
        return final_signals_result

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user or termination signal")
        return []
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        return []
    finally:
        logger.info("Analysis complete. Exiting.")
        try:
            if processor is not None:
                processor.stop()
                logger.info("Processor stopped properly")
        except Exception as e:
            logger.error(f"Error stopping processor: {e}")

def RF_HMM_chain(processor, 
                 pairs_input: Union[Dict, List], 
                 best_performance_data: Optional[Dict] = None,
                 worst_performance_data: Optional[Dict] = None,
                 timeframes_rf: Optional[List[str]] = None, 
                 timeframes_hmm: Optional[List[str]] = None, 
                 skip_data_loading: bool = False, 
                 pre_combined_df_for_rf_training: Optional[pd.DataFrame] = None):
    """
    Optimized Random Forest and HMM filtering pipeline using performance-filtered data.
    Enhanced to combine signals by direction regardless of timeframe matching.
    
    Args:
        processor: Tick processor instance
        pairs_input: Full dataset for RF model training
        best_performance_data: Filtered data for best performing pairs (RF LONG + HMM)
        worst_performance_data: Filtered data for worst performing pairs (RF SHORT + HMM)
        timeframes_rf: Timeframes for RF analysis
        timeframes_hmm: Timeframes for HMM analysis
        skip_data_loading: Whether to skip data loading
        pre_combined_df_for_rf_training: Pre-combined DataFrame for RF training
    """
    global terminate
    
    timeframes_rf = timeframes_rf or ['15m', '30m', '1h', '4h', '1d']
    timeframes_hmm = timeframes_hmm or ['5m', '15m', '30m', '1h', '4h']
    
    cpu_cores = os.cpu_count()
    max_workers_cpu = max(1, int(cpu_cores * 0.8)) if cpu_cores else 2
    
    logger.info("="*60)
    logger.info("STARTING ENHANCED RF_HMM_CHAIN PROCESSING")
    logger.info("="*60)
    
    try:
        # Process full dataset for RF model training
        if isinstance(pairs_input, dict):
            preloaded_data = pairs_input
            symbols_list = list(pairs_input.keys())
            logger.info(f"Using preloaded data for RF training: {len(symbols_list)} symbols")
        elif isinstance(pairs_input, list) and not skip_data_loading:
            symbols_list = pairs_input
            logger.info(f"Loading data for RF training: {len(symbols_list)} symbols...")
            all_timeframes = list(set(timeframes_rf + timeframes_hmm))
            preloaded_data = load_all_pairs_data(
                processor=processor,
                symbols=symbols_list,
                load_multi_timeframes=True,
                timeframes=all_timeframes
            )
            if not preloaded_data:
                logger.error("Failed to load data for symbols")
                return {'long_signals': [], 'short_signals': []}
        else:
            logger.error("Invalid pairs_input configuration")
            return {'long_signals': [], 'short_signals': []}
        
        # Use filtered performance data or fallback to full data
        best_perf_data = best_performance_data or preloaded_data
        worst_perf_data = worst_performance_data or preloaded_data
        
        logger.info(f"Performance filtering: Best={len(best_perf_data)} pairs, Worst={len(worst_perf_data)} pairs")
        
        logger.info("Checking and cleaning models folder...")
        models_dir = Path(main_dir) / "models"
        models_dir.mkdir(exist_ok=True)
        
        existing_models = list(models_dir.glob("rf_model_*.joblib"))
        logger.info(f"Found {len(existing_models)} existing RF models")
        
        if existing_models:
            logger.info("Deleting existing RF models to force creation of new model...")
            for model_file in existing_models:
                try:
                    model_file.unlink()
                    logger.info(f"Removed model: {model_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove model {model_file.name}: {e}")
        
        # Train RF model on FULL dataset for better quality
        logger.info("Training RF model on full dataset...")
        rf_model = None
        rf_model_path = None
        
        try:
            rf_training_source_df = None
            if pre_combined_df_for_rf_training is not None and not pre_combined_df_for_rf_training.empty:
                rf_training_source_df = _prepare_combined_dataframe_for_rf(
                    data_source={},
                    existing_df=pre_combined_df_for_rf_training
                )
                logger.info("Using pre-combined DataFrame for RF training on full dataset.")
            elif isinstance(preloaded_data, dict) and preloaded_data:
                logger.info("Generating combined DataFrame from full preloaded_data for RF training.")
                rf_training_source_df = _prepare_combined_dataframe_for_rf(preloaded_data)
            else:
                logger.error("No data source available for RF model training.")
                return {'long_signals': [], 'short_signals': []}

            if rf_training_source_df is None or rf_training_source_df.empty:
                logger.error("❌ No data available for training RF model.")
                return {'long_signals': [], 'short_signals': []}
            
            rf_model, rf_model_path = train_and_save_global_rf_model(
                combined_df=rf_training_source_df,
                model_filename=None 
            )
            
            if rf_model is not None and rf_model_path:
                logger.info(f"✅ Successfully trained RF model on full dataset: {rf_model_path}")
            else:
                logger.error("❌ Failed to train RF model")
                return {'long_signals': [], 'short_signals': []}
                
        except Exception as e:
            logger.error(f"Error during RF model training: {e}", exc_info=True)
            return {'long_signals': [], 'short_signals': []}
        
        # Generate RF signals on FILTERED performance data
        logger.info("Generating RF signals on performance-filtered data...")
        rf_long_signals = []
        rf_short_signals = []
        
        # Type-safe data filtering for RF processing
        def filter_valid_data(data_dict: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
            """Filter out invalid data entries and ensure proper nested structure"""
            valid_data = {}
            for symbol, timeframe_data in data_dict.items():
                if isinstance(timeframe_data, dict):
                    valid_timeframes = {}
                    for tf, df in timeframe_data.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            valid_timeframes[tf] = df
                    if valid_timeframes:
                        valid_data[symbol] = valid_timeframes
            return valid_data
        
        if rf_model is not None:
            try:
                # Filter and validate best performance data
                filtered_best_perf_data = filter_valid_data(best_perf_data)
                
                # RF LONG signals from best_performance_data only
                logger.info(f"Processing RF LONG signals from {len(filtered_best_perf_data)} best performing pairs...")
                rf_long_df = process_signals_random_forest(
                    preloaded_data=filtered_best_perf_data,
                    timeframes_to_scan=timeframes_rf,
                    trained_model=rf_model,
                    model_path=rf_model_path,
                    auto_train_if_missing=False
                )
                
                if not rf_long_df.empty:
                    for _, row in rf_long_df.iterrows():
                        rf_long_signals.append({
                            'pair': row['Pair'],
                            'direction': 'LONG',
                            'timeframe': row['SignalTimeframe']
                        })
                
                # Filter and validate worst performance data
                filtered_worst_perf_data = filter_valid_data(worst_perf_data)
                
                # RF SHORT signals from worst_performance_data only
                logger.info(f"Processing RF SHORT signals from {len(filtered_worst_perf_data)} worst performing pairs...")
                rf_short_df = process_signals_random_forest(
                    preloaded_data=filtered_worst_perf_data,
                    timeframes_to_scan=timeframes_rf,
                    trained_model=rf_model,
                    model_path=rf_model_path,
                    auto_train_if_missing=False
                )
                
                if not rf_short_df.empty:
                    for _, row in rf_short_df.iterrows():
                        rf_short_signals.append({
                            'pair': row['Pair'],
                            'direction': 'SHORT',
                            'timeframe': row['SignalTimeframe']
                        })
                
                logger.info(f"RF signals generated: {len(rf_long_signals)} LONG, {len(rf_short_signals)} SHORT")
                    
            except Exception as e:
                logger.error(f"Error during RF signal generation: {e}")
                return {'long_signals': [], 'short_signals': []}
        
        # Combine performance data for HMM analysis
        logger.info("Preparing combined performance data for HMM analysis...")
        combined_performance_data = {}
        
        # Add best performance data
        for symbol, data in best_perf_data.items():
            if symbol not in combined_performance_data:
                combined_performance_data[symbol] = {}
            for tf in timeframes_hmm:
                if data is not None and isinstance(data, dict) and tf in data:
                    combined_performance_data[symbol][tf] = data[tf]
        
        # Add worst performance data
        for symbol, data in worst_perf_data.items():
            if symbol not in combined_performance_data:
                combined_performance_data[symbol] = {}
            for tf in timeframes_hmm:
                if data is not None and isinstance(data, dict) and tf in data:
                    combined_performance_data[symbol][tf] = data[tf]
        
        logger.info(f"Combined performance data for HMM: {len(combined_performance_data)} symbols")
        
        if not combined_performance_data:
            logger.error("No combined performance data available for HMM analysis")
            return {'long_signals': [], 'short_signals': []}
        
        # Quality filter for HMM data
        quality_symbols = []
        for symbol in combined_performance_data.keys():
            try:
                if symbol in combined_performance_data:
                    sample_tf = list(combined_performance_data[symbol].keys())[0]
                    sample_data = combined_performance_data[symbol][sample_tf]
                    
                    if len(sample_data) < 50:
                        logger.warning(f"Skipping {symbol}: insufficient data ({len(sample_data)} rows)")
                        continue
                        
                    price_range = sample_data['Close'].max() - sample_data['Close'].min()
                    if price_range == 0:
                        logger.warning(f"Skipping {symbol}: zero price variance")
                        continue
                        
                    quality_symbols.append(symbol)
            except Exception as e:
                logger.warning(f"Skipping {symbol} due to data error: {e}")
                continue
        
        logger.info(f"HMM quality filtering: {len(quality_symbols)}/{len(combined_performance_data)} symbols passed")
        
        filtered_hmm_data = {symbol: combined_performance_data[symbol] for symbol in quality_symbols}
        
        # Run HMM analysis on combined performance data
        logger.info("Running HMM analysis on performance-filtered data...")
        
        strict_hmm_long_pairs = set()
        strict_hmm_short_pairs = set()
        non_strict_hmm_long_pairs = set() 
        non_strict_hmm_short_pairs = set()
        
        # Process HMM Strict Mode - CHỈ GỌI 1 LẦN
        if terminate:
            return {'long_signals': [], 'short_signals': []}
        
        logger.info("Processing HMM signals (strict mode) on performance data...")
        try:
            hmm_strict_results_df = process_signals_hmm(
                preloaded_data=filtered_hmm_data,
                timeframes_to_scan=timeframes_hmm,
                strict_mode=True,
                max_workers=max_workers_cpu
            )
        except Exception as e:
            logger.error(f"Error during HMM STRICT MODE processing: {e}")
            hmm_strict_results_df = pd.DataFrame()  # Empty DataFrame on error
        
        # Lưu HMM strict signals theo direction
        for _, row in hmm_strict_results_df.iterrows():
            pair_tf_key = (row['Pair'], row['SignalTimeframe'])
            
            if 'Direction' in row and row['Direction']:
                if row['Direction'].upper() == 'LONG':
                    strict_hmm_long_pairs.add(pair_tf_key)
                elif row['Direction'].upper() == 'SHORT':
                    strict_hmm_short_pairs.add(pair_tf_key)
            else:
                strict_hmm_long_pairs.add(pair_tf_key)
                strict_hmm_short_pairs.add(pair_tf_key)
        
        # Process HMM Non-Strict Mode - CHỈ GỌI 1 LẦN
        if terminate:
            return {'long_signals': [], 'short_signals': []}
            
        logger.info("Processing HMM signals (non-strict mode) on performance data...")
        try:
            hmm_non_strict_results_df = process_signals_hmm(
                preloaded_data=filtered_hmm_data,
                timeframes_to_scan=timeframes_hmm,
                strict_mode=False,
                max_workers=max_workers_cpu
            )
        except Exception as e:
            logger.error(f"Error during HMM NON-STRICT MODE processing: {e}")
            hmm_non_strict_results_df = pd.DataFrame()  # Empty DataFrame on error
        
        # Lưu HMM non-strict signals theo direction
        for _, row in hmm_non_strict_results_df.iterrows():
            pair_tf_key = (row['Pair'], row['SignalTimeframe'])
            
            if 'Direction' in row and row['Direction']:
                if row['Direction'].upper() == 'LONG':
                    non_strict_hmm_long_pairs.add(pair_tf_key)
                elif row['Direction'].upper() == 'SHORT':
                    non_strict_hmm_short_pairs.add(pair_tf_key)
            else:
                non_strict_hmm_long_pairs.add(pair_tf_key)
                non_strict_hmm_short_pairs.add(pair_tf_key)
        
        # Enhanced signal combination logic - combine by direction regardless of timeframe
        logger.info("🔄 ENHANCED: Combining RF and HMM signals by direction (timeframe-independent)...")
        
        # Extract unique pairs from RF signals by direction
        rf_long_pairs = set([signal['pair'] for signal in rf_long_signals])
        rf_short_pairs = set([signal['pair'] for signal in rf_short_signals])
        
        # Extract unique pairs from HMM signals by direction (ignoring timeframe)
        hmm_long_pairs = set()
        hmm_short_pairs = set()
        
        # Combine strict and non-strict HMM pairs
        all_hmm_long_pairs = strict_hmm_long_pairs.union(non_strict_hmm_long_pairs)
        all_hmm_short_pairs = strict_hmm_short_pairs.union(non_strict_hmm_short_pairs)
        
        # Extract unique pairs only (ignore timeframes)
        for pair, _ in all_hmm_long_pairs:
            hmm_long_pairs.add(pair)
        for pair, _ in all_hmm_short_pairs:
            hmm_short_pairs.add(pair)
        
        logger.info(f"RF pairs: {len(rf_long_pairs)} LONG, {len(rf_short_pairs)} SHORT")
        logger.info(f"HMM pairs: {len(hmm_long_pairs)} LONG, {len(hmm_short_pairs)} SHORT")
        
        # Enhanced logic: If both RF and HMM agree on direction for a pair, include it
        final_long_signals = []
        final_short_signals = []
        
        best_performance_symbols = set(best_perf_data.keys())
        worst_performance_symbols = set(worst_perf_data.keys())
        
        # LONG signals: pairs that have both RF LONG and HMM LONG (any timeframe)
        common_long_pairs = rf_long_pairs.intersection(hmm_long_pairs)
        for pair in common_long_pairs:
            if pair in best_performance_symbols:
                # Find the best timeframe from RF signals for this pair
                rf_timeframes = [s['timeframe'] for s in rf_long_signals if s['pair'] == pair]
                selected_timeframe = rf_timeframes[0] if rf_timeframes else '1h'  # Default fallback
                
                final_long_signals.append({
                    'pair': pair,
                    'direction': 'LONG',
                    'timeframe': selected_timeframe
                })
        
        # SHORT signals: pairs that have both RF SHORT and HMM SHORT (any timeframe)
        common_short_pairs = rf_short_pairs.intersection(hmm_short_pairs)
        for pair in common_short_pairs:
            if pair in worst_performance_symbols:
                # Find the best timeframe from RF signals for this pair
                rf_timeframes = [s['timeframe'] for s in rf_short_signals if s['pair'] == pair]
                selected_timeframe = rf_timeframes[0] if rf_timeframes else '1h'  # Default fallback
                
                final_short_signals.append({
                    'pair': pair,
                    'direction': 'SHORT',
                    'timeframe': selected_timeframe
                })
        
        logger.info(f"🎯 ENHANCED RESULTS: {len(final_long_signals)} LONG signals, {len(final_short_signals)} SHORT signals")
        logger.info(f"   • LONG pairs with both RF+HMM agreement: {len(common_long_pairs)}")
        logger.info(f"   • SHORT pairs with both RF+HMM agreement: {len(common_short_pairs)}")
        
        # Enhanced debug logging
        _debug_rf_hmm_chain(
            rf_long_signals=rf_long_signals,
            rf_short_signals=rf_short_signals,
            strict_hmm_long_pairs=strict_hmm_long_pairs,
            strict_hmm_short_pairs=strict_hmm_short_pairs,
            non_strict_hmm_long_pairs=non_strict_hmm_long_pairs,
            non_strict_hmm_short_pairs=non_strict_hmm_short_pairs,
            final_long_signals=final_long_signals,
            final_short_signals=final_short_signals,
            logger=logger
        )
        
        logger.info("="*60)
        logger.info("ENHANCED RF_HMM_CHAIN COMPLETED")
        logger.info(f"  • RF model trained on: {len(preloaded_data)} symbols")
        logger.info(f"  • RF LONG processed: {len(best_perf_data)} best performance symbols")
        logger.info(f"  • RF SHORT processed: {len(worst_perf_data)} worst performance symbols")
        logger.info(f"  • HMM processed: {len(combined_performance_data)} combined performance symbols")
        logger.info(f"  • Final LONG signals (RF+HMM agreement): {len(final_long_signals)}")
        logger.info(f"  • Final SHORT signals (RF+HMM agreement): {len(final_short_signals)}")
        logger.info("="*60)
        
        return {
            'long_signals': final_long_signals,
            'short_signals': final_short_signals
        }
        
    except KeyboardInterrupt:
        logger.info("Enhanced RF_HMM_chain interrupted by user")
        return {'long_signals': [], 'short_signals': []}
    except Exception as e:
        logger.error(f"Error in enhanced RF_HMM_chain: {str(e)}")
        return {'long_signals': [], 'short_signals': []}

def main():
    """Main function for command line usage with argument parsing."""
    parser = argparse.ArgumentParser(description='Comprehensive crypto signal analysis using performance, RF and HMM for LONG/SHORT signals')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')
    args = parser.parse_args()
    
    final_signals_list = final_signals(log_level=args.log_level)
    logger.info("Comprehensive crypto signal analysis completed successfully.")
    
    if final_signals_list:
        long_signals = [s for s in final_signals_list if s['direction'] == 'LONG']
        short_signals = [s for s in final_signals_list if s['direction'] == 'SHORT']
        
        logger.info(f"Final Results: {len(long_signals)} LONG signals, {len(short_signals)} SHORT signals")
        
        if long_signals:
            logger.info("\n▶️ LONG Trading Signals:")
            for i, signal in enumerate(long_signals):
                logger.info(f"  {i+1}. BUY {signal['pair']} (Timeframe: {signal['timeframe']})")
        
        if short_signals:
            logger.info("\n▶️ SHORT Trading Signals:")
            for i, signal in enumerate(short_signals):
                logger.info(f"  {i+1}. SELL {signal['pair']} (Timeframe: {signal['timeframe']})")
    else:
        logger.info("No signals generated.")

if __name__ == "__main__":
    main()
