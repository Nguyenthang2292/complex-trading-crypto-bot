import logging
import os
import multiprocessing
from typing import Tuple, Literal
import pandas as pd
import ray

main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
python_path = [main_dir]
runtime_env = {"env_vars": {"PYTHONPATH": os.pathsep.join(python_path)}}

from signals._components.HMM__class__OptimizingParameters import OptimizingParameters
from livetrade.config import (
    SIGNAL_LONG_HMM as LONG,
    SIGNAL_HOLD_HMM as HOLD,
    SIGNAL_SHORT_HMM as SHORT,
    HMM_PROBABILITY_THRESHOLD,
    MAX_CPU_USAGE_FRACTION
)
from signals._quant_models.hmm_kama import hmm_kama
from signals._quant_models.hmm_high_order import hmm_high_order
from utilities._logger import setup_logging
logger = setup_logging(module_name="signals_hmm", log_level=logging.DEBUG)

num_cpus = int(multiprocessing.cpu_count() * MAX_CPU_USAGE_FRACTION)

def initialize_ray():
    """Safely initialize Ray if not already initialized"""
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, runtime_env=runtime_env)

Signal = Literal[-1, 0, 1] 

def hmm_signals(df: pd.DataFrame, optimizing_params: OptimizingParameters = OptimizingParameters()) -> Tuple[Signal, Signal]:
    """
    Generate HMM trading signals from high-order HMM and HMM-KAMA models.
    
    Args:
        df: DataFrame containing OHLCV data
        optimizing_params: Parameters for model optimization
        
    Returns:
        Tuple of (high_order_hmm_signal, hmm_kama_signal)
    """
    try:
        hmm_kama_result = hmm_kama(df, optimizing_params)
        high_order_hmm_result = hmm_high_order(df, eval_mode=True, optimizing_params=optimizing_params)
    except Exception as e:
        logger.error(f"Error in HMM model: {str(e)}")
        return HOLD, HOLD

    # High-Order HMM signal evaluation
    next_state: int = high_order_hmm_result.next_state_with_high_order_hmm
    probability: float = high_order_hmm_result.next_state_probability
    
    signal_high_order_hmm: Signal = (
        SHORT if next_state == -1 and probability >= HMM_PROBABILITY_THRESHOLD else
        LONG if next_state == 1 and probability >= HMM_PROBABILITY_THRESHOLD else
        HOLD
    )

    # HMM-KAMA scoring system
    score_long: int = 0
    score_short: int = 0
    
    # Primary signal scoring (weight: 2)
    if hmm_kama_result.next_state_with_hmm_kama in {1, 3}:
        score_long += 2
    elif hmm_kama_result.next_state_with_hmm_kama in {0, 2}:
        score_short += 2
    
    # Transition state indicators (weight: 1 each, both directions)
    transition_states: list[int] = [
        hmm_kama_result.current_state_of_state_using_std,
        hmm_kama_result.current_state_of_state_using_hmm,
        hmm_kama_result.current_state_of_state_using_kmeans
    ]
    transition_bonus: int = sum(1 for state in transition_states if state == 1)
    score_long += transition_bonus
    score_short += transition_bonus
    
    # ARM-based state scoring (weight: 1 each)
    arm_states: list[int] = [
        hmm_kama_result.state_high_probabilities_using_arm_apriori,
        hmm_kama_result.state_high_probabilities_using_arm_fpgrowth
    ]
    for state in arm_states:
        if state in {1, 3}:
            score_long += 1
        elif state in {0, 2}:
            score_short += 1
    
    # Signal decision with minimum threshold of 3
    signal_hmm_kama: Signal = (
        LONG if score_long >= 3 and score_long > score_short else
        SHORT if score_short >= 3 and score_short > score_long else
        HOLD
    )
    
    # Log active signals
    if signal_high_order_hmm != HOLD or signal_hmm_kama != HOLD:
        signal_map: dict[Signal, str] = {LONG: "LONG", HOLD: "HOLD", SHORT: "SHORT"}
        logger.info(f"HMM Signals - High Order: {signal_map[signal_high_order_hmm]} "
                   f"(prob: {probability:.3f}), KAMA: {signal_map[signal_hmm_kama]} "
                   f"(scores L:{score_long}/S:{score_short})")
    
    return signal_high_order_hmm, signal_hmm_kama


