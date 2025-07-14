import logging
import os
import multiprocessing
from typing import Tuple, Literal, Optional
import pandas as pd
import ray

from signals.quant_models.hmm.HMM__class__OptimizingParameters import OptimizingParameters, HMMKamaResult, HMMHighOrderResult
from config.config import (
    SIGNAL_LONG_HMM as LONG,
    SIGNAL_HOLD_HMM as HOLD,
    SIGNAL_SHORT_HMM as SHORT,
    HMM_PROBABILITY_THRESHOLD,
    MAX_CPU_MEMORY_FRACTION
)
from signals.quant_models.hmm.hmm_kama import hmm_kama
from signals.quant_models.hmm.hmm_high_order import hmm_high_order
from utilities._logger import setup_logging

logger = setup_logging(module_name="signals_hmm", log_level=logging.DEBUG)

num_cpus = int(multiprocessing.cpu_count() * MAX_CPU_MEMORY_FRACTION)
runtime_env = {"env_vars": {"PYTHONPATH": os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}}

Signal = Literal[-1, 0, 1]

def initialize_ray():
    """Initializes a local Ray cluster if not already running."""
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, runtime_env=runtime_env)
        logger.info(f"Ray initialized with {num_cpus} CPUs.")


def _get_high_order_hmm_signal(result: Optional[HMMHighOrderResult]) -> Tuple[Signal, float]:
    """Processes the HMMHighOrderResult to generate a trading signal."""
    if not isinstance(result, HMMHighOrderResult):
        logger.error("Invalid HMMHighOrderResult object received.")
        return HOLD, 0.0

    next_state = result.next_state_with_high_order_hmm
    probability = result.next_state_probability

    if not isinstance(probability, (float, int)) or not pd.notna(probability):
        logger.warning(f"Invalid probability value: {probability}. Defaulting to HOLD.")
        return HOLD, 0.0
    
    if probability < HMM_PROBABILITY_THRESHOLD:
        return HOLD, probability

    if next_state == 1:
        return LONG, probability
    if next_state == -1:
        return SHORT, probability
    
    return HOLD, probability

def _get_hmm_kama_signal(result: Optional[HMMKamaResult]) -> Signal:
    """Processes the HMMKamaResult to generate a trading signal based on a scoring system."""
    if not isinstance(result, HMMKamaResult):
        logger.error("Invalid HMMKamaResult object received.")
        return HOLD

    score_long = 0
    score_short = 0

    # Primary signal (Weight: 2)
    if result.next_state_with_hmm_kama in {1, 3}: # States indicating upward trend
        score_long += 2
    elif result.next_state_with_hmm_kama in {0, 2}: # States indicating downward trend
        score_short += 2

    # Transition state indicators (Weight: 1 each)
    transition_states = [
        result.current_state_of_state_using_std,
        result.current_state_of_state_using_hmm,
        result.current_state_of_state_using_kmeans
    ]
    score_long += sum(1 for state in transition_states if state == 1) # Trending up
    score_short += sum(1 for state in transition_states if state == 0) # Trending down

    # ARM-based state confirmation (Weight: 1 each)
    arm_states = [
        result.state_high_probabilities_using_arm_apriori,
        result.state_high_probabilities_using_arm_fpgrowth
    ]
    for state in arm_states:
        if state in {1, 3}:
            score_long += 1
        elif state in {0, 2}:
            score_short += 1
    
    # Signal decision with a minimum confidence threshold
    MIN_SCORE_THRESHOLD = 3
    if score_long >= MIN_SCORE_THRESHOLD and score_long > score_short:
        return LONG
    if score_short >= MIN_SCORE_THRESHOLD and score_short > score_long:
        return SHORT
        
    return HOLD

def hmm_signals(
    df: pd.DataFrame, 
    optimizing_params: OptimizingParameters = OptimizingParameters()
) -> Tuple[Signal, Signal]:
    """
    Generates trading signals from High-Order HMM and HMM-KAMA models.

    This function runs two separate Hidden Markov Model-based analyses on the
    input data to produce two independent trading signals.

    Args:
        df: A pandas DataFrame containing OHLCV data, indexed by timestamp.
        optimizing_params: Configuration parameters for the HMM models.

    Returns:
        A tuple containing the trading signals from the high-order HMM
        and HMM-KAMA models, respectively. Each signal is one of
        LONG (1), SHORT (-1), or HOLD (0).
    """
    try:
        hmm_kama_result = hmm_kama(df, optimizing_params)
        high_order_hmm_result = hmm_high_order(df, eval_mode=True, optimizing_params=optimizing_params)
    except Exception as e:
        logger.error(f"Error executing HMM models: {e}", exc_info=True)
        return HOLD, HOLD

    # Ensure type compatibility for result arguments
    signal_high_order_hmm, probability = _get_high_order_hmm_signal(
        high_order_hmm_result if isinstance(high_order_hmm_result, HMMHighOrderResult) or high_order_hmm_result is None else None
    )
    signal_hmm_kama = _get_hmm_kama_signal(
        hmm_kama_result if isinstance(hmm_kama_result, HMMKamaResult) or hmm_kama_result is None else None
    )

    if signal_high_order_hmm != HOLD or signal_hmm_kama != HOLD:
        signal_map = {LONG: "LONG", HOLD: "HOLD", SHORT: "SHORT"}
        logger.info(
            f"HMM Signals - High Order: {signal_map[signal_high_order_hmm]} "
            f"(prob: {probability:.3f}), KAMA: {signal_map[signal_hmm_kama]}"
        )
        
    return signal_high_order_hmm, signal_hmm_kama


