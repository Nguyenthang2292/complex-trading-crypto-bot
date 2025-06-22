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
from typing import Any, List, Literal, Optional, Union, Callable, Tuple
from colorama import init
init(autoreset=True)

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent)) if str(current_dir.parent.parent) not in sys.path else None

from signals._components.HMM__class__OptimizingParameters import OptimizingParameters
from utilities._logger import setup_logging

logger = setup_logging('hmm_high_order', log_level=logging.DEBUG)

@dataclass
class HIGH_ORDER_HMM:
    next_state_with_high_order_hmm: Literal[-1, 0, 1]
    next_state_duration: int
    next_state_probability: float

BULLISH, NEUTRAL, BEARISH = 1, 0, -1

def timeout(seconds: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: List[Any] = [None]
            thread = None
            
            def target() -> None:
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                # Note: We can't actually kill the thread, but we can raise an exception
                # The thread will continue running in the background
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        
        return wrapper
    return decorator

@timeout(30)
def safe_forward_backward(model: DenseHMM, observations: List[np.ndarray]) -> Tuple[Any, Any, Any, Any, Any]:
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
    
    # Validate required columns exist
    if 'high' not in swing_highs_info.columns or 'low' not in swing_lows_info.columns:
        logger.warning("Missing required columns 'high' or 'low'. Returning empty list.")
        return []
    
    if strict_mode:
        states = []
        min_length = min(len(swing_highs_info), len(swing_lows_info))
        
        if min_length < 2:
            logger.warning("Insufficient data for strict mode comparison. Need at least 2 points.")
            return []
        
        if min_length < len(swing_highs_info) or min_length < len(swing_lows_info):
            logger.warning(f"Warning: Trimming data - using {min_length} points (highs: {len(swing_highs_info)}, lows: {len(swing_lows_info)})")
        
        for i in range(1, min_length):
            try:
                current_high, previous_high = swing_highs_info['high'].iloc[i], swing_highs_info['high'].iloc[i - 1]
                current_low, previous_low = swing_lows_info['low'].iloc[i], swing_lows_info['low'].iloc[i - 1]
                
                # Check for NaN values
                if pd.isna(current_high) or pd.isna(previous_high) or pd.isna(current_low) or pd.isna(previous_low):
                    continue
                
                if current_high < previous_high and current_low < previous_low:
                    state = 0
                elif current_high > previous_high and current_low > previous_low:
                    state = 2
                else:
                    state = 1
                states.append(state)
            except (IndexError, KeyError) as e:
                logger.warning(f"Error accessing swing data at index {i}: {e}")
                continue
    
        return states
    else:
        # Remove rows with NaN values
        swing_highs_info = swing_highs_info.dropna(subset=['high'])
        swing_lows_info = swing_lows_info.dropna(subset=['low'])
        
        if swing_highs_info.empty or swing_lows_info.empty:
            logger.warning("No valid swing points after removing NaN values.")
            return []
        
        # Combine high and low swing points
        swings = []
        for idx in swing_highs_info.index:
            try:
                swings.append({'time': idx, 'type': 'high', 'value': swing_highs_info.loc[idx, 'high']})
            except KeyError:
                continue
        for idx in swing_lows_info.index:
            try:
                swings.append({'time': idx, 'type': 'low', 'value': swing_lows_info.loc[idx, 'low']})
            except KeyError:
                continue
        
        if not swings:
            logger.warning("No valid swing points found.")
            return []
        
        # Sort and remove duplicates
        try:
            swings.sort(key=lambda x: x['time'])
        except (TypeError, AttributeError):
            logger.warning("Cannot sort swings by time. Using original order.")
        
        unique_swings, prev_time = [], None
        for swing in swings:
            if swing['time'] != prev_time:
                unique_swings.append(swing)
                prev_time = swing['time']
        
        if len(unique_swings) < 2:
            logger.warning("Insufficient unique swing points for state determination.")
            return []
        
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

def optimize_n_states(observations: List[np.ndarray], min_states: int = 2, max_states: int = 10, n_folds: int = 3) -> int:
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
    if not observations or len(observations) != 1:
        raise ValueError("Expected a single observation sequence.")
    
    seq = observations[0]  # Get the single observation sequence
    
    # Ensure seq is a numpy array
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    
    seq_length = len(seq)  # Length of the sequence
    
    if seq_length < 2:
        raise ValueError(f"Sequence too short: {seq_length}. Need at least 2 observations.")
    
    if seq_length < n_folds:
        logger.warning(f"Sequence length ({seq_length}) too short for {n_folds} folds. Using 2 folds.")
        n_folds = 2
    
    if seq_length < n_folds:
        logger.warning(f"Sequence length ({seq_length}) still too short. Using default {min_states} states.")
        return min_states
    
    best_n_states, best_score = min_states, -np.inf
    
    # Try each number of hidden states
    for n_states in range(min_states, min(max_states + 1, seq_length)):
        scores = []
        
        # Use KFold for cross-validation
        kf = KFold(n_splits=n_folds, shuffle=False)
        
        for train_idx, test_idx in kf.split(np.arange(seq_length).reshape(-1, 1)):
            # Create train and test sequences
            train_seq, test_seq = seq[train_idx], seq[test_idx]
            
            if len(train_seq) == 0 or len(test_seq) == 0:
                continue
            
            # Create and train model
            try:
                model = create_hmm_model(n_symbols=3, n_states=n_states)
                model = train_model(model, [train_seq])
                
                # Evaluate on test sequence (log probability calculation)
                score = model.log_probability([test_seq])
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error in log_probability: {type(e).__name__}: {e}")
                continue
        
        # Compute the average score and update the best number of states
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score, best_n_states = avg_score, n_states
    
    return best_n_states

def create_hmm_model(n_symbols: int = 3, n_states: int = 2) -> DenseHMM:
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

def train_model(model: DenseHMM, observations: List[np.ndarray]) -> DenseHMM:
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

def predict_next_hidden_state_forward_backward(model: DenseHMM, observations: List[np.ndarray]) -> List[float]:
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
    try:
        # Get forward probabilities
        _, log_alpha, _, _, _ = safe_forward_backward(model, observations)
        log_alpha_last = log_alpha[-1]
        
        # Convert to standard probabilities with numerical stability
        with np.errstate(over='ignore', under='ignore'):
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            alpha_last = np.exp(log_alpha_last)
        
        # Handle numerical issues
        if not np.isfinite(alpha_last).all():
            logger.warning("Non-finite values in alpha_last. Using uniform distribution.")
            if hasattr(alpha_last, '__len__'):
                alpha_last = np.ones_like(alpha_last) / len(alpha_last)
            else:
                # Handle scalar case
                alpha_last = np.array([0.5, 0.5])  # Default 2-state uniform distribution
        
        alpha_last /= alpha_last.sum()
        transition_matrix = model.edges
        
        # Validate transition matrix
        if not np.isfinite(transition_matrix).all():  # type: ignore
            logger.warning("Non-finite values in transition matrix. Using uniform transitions.")
            n_states = len(alpha_last)
            transition_matrix = np.ones((n_states, n_states)) / n_states
        
        # Compute distribution for step T+1
        next_hidden_proba = alpha_last @ transition_matrix
        
        # Handle numerical issues in result
        if not np.isfinite(next_hidden_proba).all():
            logger.warning("Non-finite values in next_hidden_proba. Using uniform distribution.")
            n_states = len(next_hidden_proba)
            next_hidden_proba = np.ones(n_states) / n_states
        
        # Normalize probabilities
        proba_sum = next_hidden_proba.sum()
        if proba_sum > 0:
            next_hidden_proba /= proba_sum
        else:
            logger.warning("Zero sum in next_hidden_proba. Using uniform distribution.")
            n_states = len(next_hidden_proba)
            next_hidden_proba = np.ones(n_states) / n_states
        
        # Handle both 1D and 2D arrays
        if next_hidden_proba.ndim == 1:
            # For 2-state models, return the probabilities directly
            return next_hidden_proba.tolist()
        else:
            # For multi-state models, sum the probabilities per column
            sum_left = next_hidden_proba[:, 0].sum()
            sum_right = next_hidden_proba[:, 1].sum()
            return [sum_left, sum_right]
    
    except Exception as e:
        logger.error(f"Error in predict_next_hidden_state_forward_backward: {e}")
        # Return uniform distribution as fallback
        n_states = len(model.distributions) if model.distributions else 2
        return [1.0 / n_states] * n_states

def predict_next_observation(model: DenseHMM, observations: List[np.ndarray]) -> np.ndarray:
    """
    Return an array (n_symbols,) representing P( O_{T+1} = i ), for i=0..n_symbols-1.
    """
    try:
        next_hidden_proba = predict_next_hidden_state_forward_backward(model, observations)
        distributions = model.distributions
        
        # Get emission distributions
        emission_probs_list = []
        for dist in distributions:
            try:
                params = list(dist.parameters())
                if len(params) >= 2:
                    emission_tensor = params[1]
                    if hasattr(emission_tensor, 'flatten'):
                        emission_probs_list.append(emission_tensor.flatten())  # type: ignore
                    else:
                        # Handle case where emission_tensor is not a numpy array
                        emission_probs_list.append(np.array(emission_tensor).flatten())  # type: ignore
                else:
                    logger.warning(f"Distribution has insufficient parameters: {len(params)}")
                    continue
            except Exception as e:
                logger.warning(f"Error extracting distribution parameters: {e}")
                continue
        
        if not emission_probs_list:
            logger.warning("No valid emission probabilities found. Using uniform distribution.")
            n_symbols = 3  # Default for our use case
            return np.ones(n_symbols) / n_symbols
        
        # Determine number of symbols from first valid distribution
        n_symbols = len(emission_probs_list[0])
        next_obs_proba = np.zeros(n_symbols)
        
        # Calculate next observation probability
        for o in range(n_symbols):
            for z in range(len(next_hidden_proba)):
                if z < len(emission_probs_list) and o < len(emission_probs_list[z]):
                    next_obs_proba[o] += next_hidden_proba[z] * emission_probs_list[z][o]
        
        # Normalize probabilities
        proba_sum = next_obs_proba.sum()
        if proba_sum > 0:
            return next_obs_proba / proba_sum
        else:
            logger.warning("Zero sum in next_obs_proba. Using uniform distribution.")
            return np.ones(n_symbols) / n_symbols
    
    except Exception as e:
        logger.error(f"Error in predict_next_observation: {e}")
        # Return uniform distribution as fallback
        return np.array([1/3, 1/3, 1/3])

def average_swing_distance(swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame) -> float:  # type: ignore
    """
    Calculate the average time interval (in seconds) between consecutive swing highs and swing lows,
    and return the overall average.

    Args:
        swing_highs_info (pd.DataFrame): DataFrame containing swing high information with datetime index.
        swing_lows_info (pd.DataFrame): DataFrame containing swing low information with datetime index.

    Returns:
        float: The average time distance between swing points in seconds.
    """
    try:
        # Validate inputs
        if swing_highs_info.empty or swing_lows_info.empty:
            logger.warning("Empty swing DataFrames provided. Using default distance.")
            return 3600.0  # Default 1 hour
        
        # Check if indices are datetime
        if not isinstance(swing_highs_info.index, pd.DatetimeIndex) or not isinstance(swing_lows_info.index, pd.DatetimeIndex):
            logger.warning("Non-datetime indices detected. Using default distance.")
            return 3600.0  # Default 1 hour
        
        # Calculate high intervals
        swing_high_times = swing_highs_info.index
        intervals_seconds_high = []
        for i in range(1, len(swing_high_times)):
            try:
                interval = (swing_high_times[i] - swing_high_times[i - 1]).total_seconds()
                if interval > 0:  # Only include positive intervals
                    intervals_seconds_high.append(interval)
            except Exception as e:
                logger.warning(f"Error calculating high interval: {e}")
                continue
        
        avg_distance_high = np.mean(intervals_seconds_high) if intervals_seconds_high else 0  # type: ignore

        # Calculate low intervals
        swing_low_times = swing_lows_info.index
        intervals_seconds_low = []
        for i in range(1, len(swing_low_times)):
            try:
                interval = (swing_low_times[i] - swing_low_times[i - 1]).total_seconds()
                if interval > 0:  # Only include positive intervals
                    intervals_seconds_low.append(interval)
            except Exception as e:
                logger.warning(f"Error calculating low interval: {e}")
                continue
        
        avg_distance_low = np.mean(intervals_seconds_low) if intervals_seconds_low else 0  # type: ignore

        # Return average
        if avg_distance_high > 0 and avg_distance_low > 0:
            return (avg_distance_high + avg_distance_low) / 2  # type: ignore
        elif avg_distance_high > 0:
            return avg_distance_high  # type: ignore
        elif avg_distance_low > 0:
            return avg_distance_low  # type: ignore
        else:
            logger.warning("No valid intervals found. Using default distance.")
            return 3600.0  # Default 1 hour
    
    except Exception as e:
        logger.error(f"Error in average_swing_distance: {e}")
        return 3600.0  # Default 1 hour

def evaluate_model_accuracy(model: DenseHMM, train_states: List[float], test_states: List[float]) -> float:
    """
    Evaluate the accuracy of the HMM model on the test set.
    
    Parameters:
        model: The trained HMM model.
        train_states: Sequence of states used for training.
        test_states: Sequence of states used for testing.
        
    Returns:
        float: The accuracy of the model.
    """
    try:
        if not test_states:
            logger.warning("No test states provided. Returning 0.0 accuracy.")
            return 0.0
        
        if not train_states:
            logger.warning("No train states provided. Returning 0.0 accuracy.")
            return 0.0
        
        correct_predictions = 0
        
        # Predict each state in the test set
        for i in range(len(test_states)):
            try:
                # Create an observation sequence using training data and known test states so far
                current_states = train_states + test_states[:i]
                if not current_states:
                    continue
                
                observations = [np.array(current_states).reshape(-1, 1)]
                
                # Predict the next state
                next_obs_proba = predict_next_observation(model, observations)
                predicted_state = np.argmax(next_obs_proba)
                
                # Compare with the actual state
                if predicted_state == test_states[i]:
                    correct_predictions += 1
            except Exception as e:
                logger.warning(f"Error predicting state {i}: {e}")
                continue
        
        # Calculate accuracy
        accuracy = correct_predictions / len(test_states) if test_states else 0.0
        
        # Validate accuracy is in valid range
        if not np.isfinite(accuracy) or accuracy < 0 or accuracy > 1:
            logger.warning(f"Invalid accuracy value: {accuracy}. Returning 0.0.")
            return 0.0
        
        return accuracy
    
    except Exception as e:
        logger.error(f"Error in evaluate_model_accuracy: {e}")
        return 0.0

def hmm_high_order(df: pd.DataFrame, train_ratio: float = 0.8, eval_mode: bool = True, optimizing_params: Optional[OptimizingParameters] = None) -> HIGH_ORDER_HMM:
    """
    Generates and trains a Hidden Markov Model (HMM) using swing points extracted from market price data.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price data with at least the columns 'open', 'high', 'low', 'close'.
        train_ratio (float): The ratio of data to use for training (default: 0.8).
        eval_mode (bool): If True, evaluates model performance on the test set.
    
    Returns:
        HIGH_ORDER_HMM: Instance containing the predicted market state.
    """
    try:
        # Set default optimizing_params if None
        if optimizing_params is None:
            optimizing_params = OptimizingParameters()
        
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
        
        # Validate train_ratio
        if not (0.1 <= train_ratio <= 0.9):
            logger.warning(f"Invalid train_ratio: {train_ratio}. Using default 0.8.")
            train_ratio = 0.8
        
        # Determine data interval
        interval_str = ""
        if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
            try:
                time_diff = df.index[1] - df.index[0]
                total_minutes = int(time_diff.total_seconds() / 60)
                interval_str = f"h{total_minutes // 60}" if total_minutes % 60 == 0 else f"m{total_minutes}"
            except Exception as e:
                logger.warning(f"Error calculating time interval: {e}. Using default interval.")
                interval_str = "h1"
        elif len(df) > 1:
            # For non-datetime index, use default interval
            logger.warning("DataFrame index is not DatetimeIndex. Using default interval.")
            interval_str = "h1"  # Default to 1 hour
        
        # Find swing points
        try:
            swing_highs = argrelextrema(df['high'].values, np.greater, order=optimizing_params.orders_argrelextrema)[0]
            swing_lows = argrelextrema(df['low'].values, np.less, order=optimizing_params.orders_argrelextrema)[0]
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
        
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
        except Exception as e:
            logger.warning(f"Error optimizing n_states: {e}. Using default 2 states.")
            optimal_n_states = 2
        
        try:
            model = create_hmm_model(n_symbols=3, n_states=optimal_n_states)
            model = train_model(model, train_observations)
        except Exception as e:
            logger.error(f"Error creating/training model: {e}")
            return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
        
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
        
        # Ensure converted_distance is positive and finite
        if not np.isfinite(converted_distance) or converted_distance <= 0:
            logger.warning(f"Invalid converted_distance: {converted_distance}. Using default 1.")
            converted_distance = 1
        
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
        
        # Validate max_value
        if not np.isfinite(max_value) or max_value < 0 or max_value > 1:
            logger.warning(f"Invalid max_value: {max_value}. Using default 0.33.")
            max_value = 0.33
        
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
    
    except Exception as e:
        logger.error(f"Error in hmm_high_order: {e}")
        return HIGH_ORDER_HMM(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)

