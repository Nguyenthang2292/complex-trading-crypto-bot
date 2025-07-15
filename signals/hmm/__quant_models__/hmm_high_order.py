import logging
import numpy as np
import pandas as pd
import sys
import threading
import warnings
from dataclasses import dataclass
from functools import wraps
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
from pathlib import Path
from scipy.signal import argrelextrema
from sklearn.model_selection import KFold
from typing import Any, List, Literal
from colorama import init
init(autoreset=True)

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent)) if str(current_dir.parent.parent) not in sys.path else None

from signals.hmm.__components__.__class__OptimizingParameters import OptimizingParameters
from utilities.logger import setup_logging

logger = setup_logging('hmm_high_order', log_level=logging.DEBUG)

@dataclass
class HIGH_ORDER_HMM:
    next_state_with_high_order_hmm: Literal[-1, 0, 1]
    next_state_duration: int
    next_state_probability: float

BULLISH, NEUTRAL, BEARISH = 1, 0, -1

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result: List[Any] = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        
        return wrapper
    return decorator

@timeout(30)
def safe_forward_backward(model, observations):
    return model.forward_backward(observations)

def convert_swing_to_state(swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame, strict_mode: bool = False) -> List[float]:
    """
    Convert swing high and low points to market state sequence.
    
    States:
    - 0: Downtrend
    - 1: Sideways/Consolidation
    - 2: Uptrend
    
    Methods:
    - strict_mode=True: Compares consecutive swing values requiring equal counts
    - strict_mode=False: Uses chronological transitions between highs and lows
    
    Parameters
    ----------
    swing_highs_info : DataFrame
        Swing highs with 'high' column and datetime index
    swing_lows_info : DataFrame
        Swing lows with 'low' column and datetime index
    strict_mode : bool, default=False
        Whether to use strict comparison mode
        
    Returns
    -------
    List[float]
        Market state values (0, 1, or 2)
    """
    if swing_highs_info.empty or swing_lows_info.empty:
        logger.warning("One of the swing DataFrames is empty. Returning empty list.")
        return []
    
    if strict_mode:
        states = []
        min_length = min(len(swing_highs_info), len(swing_lows_info))
        
        if min_length < len(swing_highs_info) or min_length < len(swing_lows_info):
            logger.warning(f"Warning: Trimming data - using {min_length} points (highs: {len(swing_highs_info)}, lows: {len(swing_lows_info)})")
        
        for i in range(1, min_length):
            current_high, previous_high = swing_highs_info['high'].iloc[i], swing_highs_info['high'].iloc[i - 1]
            current_low, previous_low = swing_lows_info['low'].iloc[i], swing_lows_info['low'].iloc[i - 1]
            
            if current_high < previous_high and current_low < previous_low:
                state = 0
            elif current_high > previous_high and current_low > previous_low:
                state = 2
            else:
                state = 1
            states.append(state)
    
        return states
    else:
        # Remove rows with NaN values
        swing_highs_info = swing_highs_info.dropna(subset=['high'])
        swing_lows_info = swing_lows_info.dropna(subset=['low'])
        
        # Combine high and low swing points
        swings = []
        for idx in swing_highs_info.index:
            swings.append({'time': idx, 'type': 'high', 'value': swing_highs_info.loc[idx, 'high']})
        for idx in swing_lows_info.index:
            swings.append({'time': idx, 'type': 'low', 'value': swing_lows_info.loc[idx, 'low']})
        
        # Sort and remove duplicates
        swings.sort(key=lambda x: x['time'])
        unique_swings, prev_time = [], None
        for swing in swings:
            if swing['time'] != prev_time:
                unique_swings.append(swing)
                prev_time = swing['time']
        
        # Determine states
        states, prev_swing = [], None
        for swing in unique_swings:
            if prev_swing is None:
                prev_swing = swing
                continue
            
            if prev_swing['type'] == 'low' and swing['type'] == 'high':
                state = 2  # price increase
            elif prev_swing['type'] == 'high' and swing['type'] == 'low':
                state = 0  # price decrease
            else:
                state = 1  # unchanged or mixed
            
            states.append(state)
            prev_swing = swing
        
        return states

def optimize_n_states(observations, min_states=2, max_states=10, n_folds=3):
    """
    Automatically optimize the number of hidden states using KFold cross-validation.
    
    Parameters:
    - observations: A list containing a single observation sequence (2D array).
    - min_states: Minimum number of hidden states.
    - max_states: Maximum number of hidden states.
    - n_folds: Number of folds in cross-validation.
    
    Returns:
    - The best number of hidden states.
    """
    
    # Check if observations are in the correct format
    if len(observations) != 1:
        raise ValueError("Expected a single observation sequence.")
    
    seq = observations[0]  # Get the single observation sequence
    seq_length = len(seq)  # Length of the sequence
    
    if seq_length < n_folds:
        raise ValueError(f"Sequence length ({seq_length}) too short for {n_folds} folds.")
    
    best_n_states, best_score = min_states, -np.inf
    
    # Try each number of hidden states
    for n_states in range(min_states, max_states + 1):
        scores = []
        
        # Use KFold for cross-validation
        kf = KFold(n_splits=n_folds, shuffle=False)
        
        for train_idx, test_idx in kf.split(np.arange(seq_length).reshape(-1, 1)):
            # Create train and test sequences
            train_seq, test_seq = seq[train_idx], seq[test_idx]
            
            if len(train_seq) == 0 or len(test_seq) == 0:
                continue
            
            # Create and train model
            model = create_hmm_model(n_symbols=3, n_states=n_states)
            model = train_model(model, [train_seq])
            
            # Evaluate on test sequence (log probability calculation)
            try:
                score = model.log_probability([test_seq])
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error in log_probability: {type(e).__name__}: {e}")
        
        # Compute the average score and update the best number of states
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score, best_n_states = avg_score, n_states
    
    return best_n_states

def create_hmm_model(n_symbols=3, n_states=2):
    """
    Create an optimally configured HMM model.
    
    Args:
        n_symbols (int): Number of observable symbols (0, 1, 2)
        n_states (int): Number of hidden states
        
    Returns:
        DenseHMM: The configured HMM model.
    """
    if n_states == 2:
        # Optimized configuration for 2 hidden states
        distributions = [
            Categorical([[0.25, 0.25, 0.50]]),  # Mixed trend, biased toward increase
            Categorical([[0.50, 0.25, 0.25]])   # Mixed trend, biased toward decrease
        ]
        edges = [[0.85, 0.15], [0.15, 0.85]]
        starts = [0.5, 0.5]
        ends = [0.01, 0.01]
    else:
        # Configuration for custom number of hidden states
        distributions = [Categorical([[1/n_symbols] * n_symbols]) for _ in range(n_states)]
        edges = np.ones((n_states, n_states), dtype=np.float32) / n_states
        starts = np.ones(n_states, dtype=np.float32) / n_states
        ends = np.ones(n_states, dtype=np.float32) * 0.01
    
    return DenseHMM(distributions, edges=edges, starts=starts, ends=ends, verbose=False)

def train_model(model, observations):
    """
    Train the HMM model with observation data.
    
    Args:
        model (DenseHMM): The HMM model to be trained.
        observations (list): List of observation arrays.
        
    Returns:
        DenseHMM: The trained HMM model.
    """
    model.fit(observations)
    return model

def predict_next_hidden_state_forward_backward(model: DenseHMM, observations: list) -> List[float]:
    """
    Compute the hidden state distribution for step T+1 given T observations.
    
    Args:
        model (DenseHMM): The trained HMM model.
        observations (list): List of observations.
        
    Returns:
        list: The probability distribution of the hidden state at step T+1.

    Explanation of the alpha and beta variables (from the forward-backward algorithm):

    - Alpha (forward variable):
    Represents the probability of observing the sequence from the beginning up to time t 
    and being in a specific state at time t.
    
    - Beta (backward variable):
    Represents the probability of observing the sequence from time t+1 to the end 
    given that the system is in a specific state at time t.
    
    Combining alpha and beta allows computation of the posterior probabilities of the states.
    """
    # Get forward probabilities
    _, log_alpha, _, _, _ = safe_forward_backward(model, observations)
    log_alpha_last = log_alpha[-1]
    
    # Convert to standard probabilities
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        alpha_last = np.exp(log_alpha_last)
    
    alpha_last /= alpha_last.sum()
    transition_matrix = model.edges
    
    # Compute distribution for step T+1
    next_hidden_proba = alpha_last @ transition_matrix
    
    # Handle both 1D and 2D arrays
    if next_hidden_proba.ndim == 1:
        # For 2-state models, return the probabilities directly
        return next_hidden_proba.tolist()
    else:
        # For multi-state models, sum the probabilities per column
        sum_left = next_hidden_proba[:, 0].sum()
        sum_right = next_hidden_proba[:, 1].sum()
        return [sum_left, sum_right]

def predict_next_observation(model, observations):
    """
    Return an array (n_symbols,) representing P( O_{T+1} = i ), for i=0..n_symbols-1.
    """
    next_hidden_proba = predict_next_hidden_state_forward_backward(model, observations)
    distributions = model.distributions
    
    # Get emission distributions
    params = list(distributions[0].parameters())
    n_symbols = params[1].shape[1]
    next_obs_proba = np.zeros(n_symbols)
    
    emission_probs_list = []
    for dist in distributions:
        params = list(dist.parameters())
        emission_tensor = params[1]
        emission_probs_list.append(emission_tensor.flatten())
    
    # Calculate next observation probability
    for o in range(n_symbols):
        for z in range(len(next_hidden_proba)):
            next_obs_proba[o] += next_hidden_proba[z] * emission_probs_list[z][o]
    
    return next_obs_proba / next_obs_proba.sum()

def average_swing_distance(swing_highs_info, swing_lows_info):
    """
    Calculate the average time interval (in seconds) between consecutive swing highs and swing lows,
    and return the overall average.

    Args:
        swing_highs_info (pd.DataFrame): DataFrame containing swing high information with datetime index.
        swing_lows_info (pd.DataFrame): DataFrame containing swing low information with datetime index.

    Returns:
        float: The average time distance between swing points in seconds.
    """
    # Calculate high intervals
    swing_high_times = swing_highs_info.index
    intervals_seconds_high = [(swing_high_times[i] - swing_high_times[i - 1]).total_seconds() 
                            for i in range(1, len(swing_high_times))]
    avg_distance_high = np.mean(intervals_seconds_high) if intervals_seconds_high else 0

    # Calculate low intervals
    swing_low_times = swing_lows_info.index
    intervals_seconds_low = [(swing_low_times[i] - swing_low_times[i - 1]).total_seconds() 
                            for i in range(1, len(swing_low_times))]
    avg_distance_low = np.mean(intervals_seconds_low) if intervals_seconds_low else 0

    # Return average
    if avg_distance_high and avg_distance_low:
        return (avg_distance_high + avg_distance_low) / 2
    return avg_distance_high or avg_distance_low

def evaluate_model_accuracy(model, train_states, test_states):
    """
    Evaluate the accuracy of the HMM model on the test set.
    
    Parameters:
        model: The trained HMM model.
        train_states: Sequence of states used for training.
        test_states: Sequence of states used for testing.
        
    Returns:
        float: The accuracy of the model.
    """
    correct_predictions = 0
    
    # Predict each state in the test set
    for i in range(len(test_states)):
        # Create an observation sequence using training data and known test states so far
        current_states = train_states + test_states[:i]
        observations = [np.array(current_states).reshape(-1, 1)]
        
        # Predict the next state
        next_obs_proba = predict_next_observation(model, observations)
        predicted_state = np.argmax(next_obs_proba)
        
        # Compare with the actual state
        if predicted_state == test_states[i]:
            correct_predictions += 1
    
    # Calculate accuracy
    return correct_predictions / len(test_states) if test_states else 0.0

def hmm_high_order(df: pd.DataFrame, train_ratio: float = 0.8, eval_mode: bool = True, optimizing_params: OptimizingParameters = OptimizingParameters()) -> HIGH_ORDER_HMM:
    """
    Generates and trains a Hidden Markov Model (HMM) using swing points extracted from market price data.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price data with at least the columns 'open', 'high', 'low', 'close'.
        train_ratio (float): The ratio of data to use for training (default: 0.8).
        eval_mode (bool): If True, evaluates model performance on the test set.
    
    Returns:
        HIGH_ORDER_HMM: Instance containing the predicted market state.
    """
    # Check for valid input
    required_columns = ['open', 'high', 'low', 'close']
    if df is None or df.empty or not all(col in df.columns for col in required_columns):
        logger.error("Invalid dataframe provided - missing required columns")
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
    
    # Check if data is numeric
    try:
        for col in required_columns:
            pd.to_numeric(df[col], errors='raise')
    except (ValueError, TypeError):
        logger.error("Invalid dataframe provided - non-numeric data detected")
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
    
    # Determine data interval
    interval_str = ""
    if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
        time_diff = df.index[1] - df.index[0]
        total_minutes = int(time_diff.total_seconds() / 60)
        interval_str = f"h{total_minutes // 60}" if total_minutes % 60 == 0 else f"m{total_minutes}"
    elif len(df) > 1:
        # For non-datetime index, use default interval
        logger.warning("DataFrame index is not DatetimeIndex. Using default interval.")
        interval_str = "h1"  # Default to 1 hour
    
    # Find swing points
    swing_highs = argrelextrema(df['high'].values, np.greater, order=optimizing_params.orders_argrelextrema)[0]
    swing_lows = argrelextrema(df['low'].values, np.less, order=optimizing_params.orders_argrelextrema)[0]
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        logger.warning("Not enough swing points detected for reliable prediction")
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
    
    swing_highs_info = df.iloc[swing_highs][['open', 'high', 'low', 'close']]
    swing_lows_info = df.iloc[swing_lows][['open', 'high', 'low', 'close']]

    # Convert swing points to states
    states = convert_swing_to_state(swing_highs_info, swing_lows_info, strict_mode=optimizing_params.strict_mode)
    if not states:
        logger.warning("No states detected from swing points")
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
    
    # Split data into training and testing sets
    train_size = int(len(states) * train_ratio)
    if train_size < 2:
        train_states, test_states = states, []
    else:
        train_states, test_states = states[:train_size], states[train_size:]
    
    # Build and train HMM model
    train_observations = [np.array(train_states).reshape(-1, 1)]
    try:
        optimal_n_states = optimize_n_states(train_observations)
    except Exception:
        optimal_n_states = 2
    
    model = create_hmm_model(n_symbols=3, n_states=optimal_n_states)
    model = train_model(model, train_observations)
    
    # Evaluate model if requested
    accuracy = evaluate_model_accuracy(model, train_states, test_states) if eval_mode and test_states else 0.0
    
    # Calculate average time between swing points - only for datetime index
    if isinstance(swing_highs_info.index, pd.DatetimeIndex) and isinstance(swing_lows_info.index, pd.DatetimeIndex):
        average_distance = average_swing_distance(swing_highs_info, swing_lows_info) or 3600
    else:
        # For non-datetime index, use default distance
        logger.warning("Non-datetime index detected. Using default swing distance.")
        average_distance = 3600  # Default to 1 hour in seconds
    
    # Convert time units
    if interval_str.startswith("h"):
        converted_distance = average_distance / 3600  # to hours
    elif interval_str.startswith("m"):
        converted_distance = average_distance / 60    # to minutes
    else:
        converted_distance = average_distance
    
    # Return NEUTRAL if accuracy is too low
    if accuracy <= 0.3:
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, 
                                next_state_duration=int(converted_distance), 
                                next_state_probability=max(accuracy, 0.33))
    
    # Predict next state
    full_observations = [np.array(states).reshape(-1, 1)]
    next_obs_proba = predict_next_observation(model, full_observations)
    next_obs_proba = np.nan_to_num(next_obs_proba, nan=1/3, posinf=1/3, neginf=1/3)
    
    if not np.isfinite(next_obs_proba).all() or np.sum(next_obs_proba) == 0:
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, 
                                next_state_duration=int(converted_distance), 
                                next_state_probability=0.33)
        
    max_index = int(np.argmax(next_obs_proba))
    max_value = round(next_obs_proba[max_index], 2)
    
    # Return appropriate state based on prediction
    if max_index == 0:
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=BEARISH, 
                                next_state_duration=int(converted_distance), 
                                next_state_probability=max_value)
    elif max_index == 1:
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, 
                                next_state_duration=int(converted_distance), 
                                next_state_probability=max_value)
    else:
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=BULLISH, 
                                next_state_duration=int(converted_distance), 
                                next_state_probability=max_value)

