import functools
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from hmmlearn.hmm import GaussianHMM
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from typing import Literal, Tuple, cast

# Fix KMeans memory leak on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent.parent)) if str(current_dir.parent.parent) not in sys.path else None

from signals.hmm.__components__.__class__OptimizingParameters import OptimizingParameters
from utilities.logger import setup_logging
logger = setup_logging('hmm_kama', log_level=logging.DEBUG)
    
@dataclass
class HMM_KAMA:
    next_state_with_hmm_kama: Literal[0, 1, 2, 3]
    current_state_of_state_using_std: Literal[0, 1]
    current_state_of_state_using_hmm: Literal[0, 1]
    state_high_probabilities_using_arm_apriori: Literal[0, 1, 2, 3]
    state_high_probabilities_using_arm_fpgrowth: Literal[0, 1, 2, 3]
    current_state_of_state_using_kmeans: Literal[0, 1]

def calculate_kama(prices, window: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Function to calculate KAMA (Kaufman's Adaptive Moving Average) with robust error handling"""
    prices_array = np.asarray(prices, dtype=np.float64)
    
    if len(prices_array) < window:
        return np.full_like(prices_array, float(prices_array.flat[0])) if len(prices_array) > 0 else np.array([0.0])
    
    kama = np.zeros_like(prices_array, dtype=np.float64)
    first_valid_idx = next((i for i, p in enumerate(prices_array) if not (np.isnan(p) or np.isinf(p))), 0)
    initial_value = float(prices_array[first_valid_idx]) if first_valid_idx < len(prices_array) else float(np.nanmean(prices_array[:window]))
    kama[:window] = initial_value
    
    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
    
    try:
        price_series = pd.Series(prices)
        changes = price_series.diff(window).abs()
        volatility = price_series.rolling(window).apply(
            lambda x: np.sum(np.abs(np.diff(x))) if len(x) > 1 else 1e-10, raw=False
        ).fillna(1e-10)
        
        volatility = np.where(np.logical_or(volatility == 0, np.isinf(volatility)), 1e-10, volatility)
        
        er = np.clip((changes / volatility).fillna(0).replace([np.inf, -np.inf], 0), 0, 1)
        
        for i in range(window, len(prices_array)):
            if np.isnan(prices_array[i]) or np.isinf(prices_array[i]):
                kama[i] = kama[i - 1]
                continue
                
            er_value = float(er.iloc[i] if isinstance(er, pd.Series) else er[i])
            if np.isnan(er_value) or np.isinf(er_value):
                kama[i] = kama[i - 1]
                continue
            
            sc = np.clip((er_value * (fast_sc - slow_sc) + slow_sc) ** 2, 1e-10, 1.0)
            price_diff = np.clip(prices_array[i] - kama[i - 1], -1e10, 1e10) if abs(prices_array[i] - kama[i - 1]) > 1e10 else prices_array[i] - kama[i - 1]
            kama[i] = kama[i - 1] + sc * price_diff
            
            if np.isnan(kama[i]) or np.isinf(kama[i]):
                kama[i] = kama[i - 1]
                
    except Exception as e:
        logger.warning(f"Error in KAMA calculation: {e}. Using simple moving average fallback.")
        kama = pd.Series(prices).rolling(window=window, min_periods=1).mean().ffill().values
    
    kama_array = np.asarray(kama, dtype=np.float64)
    return np.where(~np.isfinite(kama_array), initial_value, kama_array).astype(np.float64)

def prepare_observations(data: pd.DataFrame, optimizing_params: OptimizingParameters) -> np.ndarray:
    """Generate observation features optimized specifically for crypto assets."""
    if data.empty or 'close' not in data.columns or len(data) < 10:
        raise ValueError(f"Invalid data: empty={data.empty}, has close={'close' in data.columns}, len={len(data)}")
    
    close_prices = data["close"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if close_prices.isna().any(): # type: ignore    
        close_prices = close_prices.fillna(close_prices.median())

    price_range = close_prices.max() - close_prices.min()
    unique_prices = close_prices.nunique()
    
    if price_range == 0 or unique_prices < 3:
        logger.data(f"Problematic price data: range={price_range}, unique_prices={unique_prices}")
        close_prices = pd.Series(np.linspace(close_prices.mean() * 0.95, close_prices.mean() * 1.05, len(close_prices)))
        price_range = close_prices.max() - close_prices.min()

    close_prices = ((close_prices - close_prices.min()) / price_range * 100) if price_range > 0 else pd.Series(np.linspace(45, 55, len(close_prices)))
    close_prices_array = close_prices.values.astype(np.float64)

    try:
        window = max(2, min(optimizing_params.window_kama, len(close_prices_array)//2))
        fast = optimizing_params.fast_kama
        slow = max(optimizing_params.slow_kama, fast + 5)
        
        kama_values = calculate_kama(close_prices_array, window=window, fast=fast, slow=slow)
        
        if np.max(kama_values) - np.min(kama_values) < 1e-10:
            logger.data("KAMA has zero variance. Adding gradient.")
            kama_values = np.linspace(kama_values[0] - 0.5, kama_values[0] + 0.5, len(kama_values))
            
    except Exception as e:
        logger.error(f"KAMA calculation failed: {e}. Using EMA fallback.")
        kama_values = pd.Series(close_prices_array).ewm(alpha=2.0/(optimizing_params.window_kama+1), adjust=False).mean().values

    returns = np.diff(close_prices_array, prepend=close_prices_array[0])
    if np.std(returns) < 1e-10:
        logger.data("Returns have zero variance. Adding synthetic returns.")
        returns = np.random.RandomState(42).normal(0, 0.1, len(returns))
    
    volatility = np.abs(np.diff(np.array(kama_values), prepend=kama_values[0]))
    if np.std(volatility) < 1e-10:
        logger.data("Volatility has zero variance. Adding synthetic volatility.")
        volatility = np.random.RandomState(42).exponential(0.1, len(volatility))
    
    rolling_vol = pd.Series(returns).rolling(window=5, min_periods=1).std().fillna(0.01).values
    volatility = (volatility + np.asarray(rolling_vol)) / 2

    def _clean_crypto_array(arr, name="array", default_val=0.0):
        arr = np.where(np.isfinite(arr), arr, default_val)
        valid_values = arr[arr != default_val]
        q_range = max(float(abs(np.percentile(valid_values, 95))), float(abs(np.percentile(valid_values, 5)))) * 1.5 if np.any(valid_values) else 100
        return np.clip(arr, -q_range, q_range).astype(np.float64)

    returns, kama_values, volatility = [_clean_crypto_array(arr, name, val) for arr, name, val in 
                                    [(returns, "returns", 0.0), (kama_values, "kama_values", close_prices_array.mean()), (volatility, "volatility", 0.01)]]
    
    # Final variance check
    if np.std(returns) == 0: returns[0] = 0.01
    if np.std(kama_values) == 0: kama_values[-1] = kama_values[-1] + 0.01
    if np.std(volatility) == 0: volatility[0], volatility[-1] = 0.005, 0.015

    feature_matrix = np.column_stack([returns, kama_values, volatility])
    
    if not np.isfinite(feature_matrix).all():
        logger.error("Feature matrix contains invalid values. Creating safe fallback.")
        n_rows = len(returns)
        feature_matrix = np.column_stack([
            np.random.RandomState(42).normal(0, 0.01, n_rows),
            np.linspace(close_prices_array.mean()-1, close_prices_array.mean()+1, n_rows),
            np.random.RandomState(42).exponential(0.01, n_rows)
        ])
    
    logger.analysis(f"Crypto-optimized features - Shape: {feature_matrix.shape}, "
                f"Returns range: [{returns.min():.6f}, {returns.max():.6f}], "
                f"KAMA range: [{kama_values.min():.6f}, {kama_values.max():.6f}], "
                f"Volatility range: [{volatility.min():.6f}, {volatility.max():.6f}]")
    
    return feature_matrix

def train_hmm(observations: np.ndarray, n_components: int = 4, n_iter: int = 10, random_state: int = 36) -> GaussianHMM:
    """Train HMM with robust error handling and data validation"""
    if observations.size == 0:
        raise ValueError("Empty observations array")
        
    n_features = observations.shape[1] if len(observations.shape) > 1 else 1
        
    if not np.isfinite(observations).all():
        logger.warning("Observations contain invalid values. Cleaning...")
        for col in range(observations.shape[1]): # type: ignore
            col_data = observations[:, col]
            finite_mask = np.isfinite(col_data)
            if finite_mask.any():
                observations[:, col] = np.where(finite_mask, col_data, np.median(col_data[finite_mask]))
            else:
                observations[:, col] = 0.0
    
    variances = np.var(observations, axis=0)
    low_var_mask = variances < 1e-12
    
    if low_var_mask.any():
        logger.data(f"Low variance detected in columns {np.where(low_var_mask)[0]}. Adding noise.")
        noise = np.random.RandomState(random_state).normal(0, 1e-6, observations.shape)
        observations[:, low_var_mask] += noise[:, low_var_mask]
    
    obs_max = np.max(np.abs(observations))
    if obs_max > 1e6:
        scale_factor = 1e6 / obs_max
        observations = observations * scale_factor
        logger.data(f"Scaled observations by factor {scale_factor} to prevent overflow")
    
    logger.model(f"Training HMM - Observations shape: {observations.shape}, "
                f"Range: [{np.min(observations):.2e}, {np.max(observations):.2e}], "
                f"Finite values: {np.isfinite(observations).sum()}/{observations.size}")
    
    try:
        model = GaussianHMM(n_components=n_components, 
                            covariance_type="diag", 
                            n_iter=min(n_iter, 50), 
                            random_state=random_state, tol=1e-3)
        
        with np.errstate(all='ignore'): 
            model.fit(observations)
        
        if (hasattr(model, 'transmat_') and not np.isfinite(model.transmat_).all() or 
            hasattr(model, 'means_') and not np.isfinite(model.means_).all()):
            raise ValueError("Invalid transition matrix or means after fitting")
            
        if hasattr(model, 'covars_'):
            try:
                covars = model.covars_
                if covars is not None and not np.isfinite(covars).all():
                    raise ValueError("Invalid covariances after fitting")
            except AttributeError:
                pass
            
        logger.success("HMM training completed successfully")
        
    except Exception as e:
        logger.error(f"HMM training failed: {str(e)}. Creating default model.")
        
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1, random_state=random_state)
        
        model.startprob_ = np.ones(n_components, dtype=np.float64) / n_components
        model.transmat_ = np.eye(n_components, dtype=np.float64) * 0.7 + np.ones((n_components, n_components), dtype=np.float64) * 0.3 / n_components
        
        model.means_ = np.zeros((n_components, n_features), dtype=np.float64)
        for i in range(n_components):
            model.means_[i] = np.quantile(observations, (i + 1) / (n_components + 1), axis=0)
        
        try:
            model.covars_ = np.ones((n_components, n_features), dtype=np.float64) * np.var(observations, axis=0)
        except Exception:
            pass
        
        for attr in ['startprob_', 'transmat_', 'means_']:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                attr_array = np.asarray(getattr(model, attr))
                setattr(model, attr, np.where(np.isfinite(attr_array), attr_array, 1.0/n_components if attr != 'means_' else 0.0))
        
        if hasattr(model, 'covars_'):
            try:
                if model.covars_ is not None:
                    model.covars_ = np.where(np.isfinite(np.asarray(model.covars_)), model.covars_, 1.0)
            except Exception:
                pass
    
    return model

@contextmanager
def timeout_context(seconds):
    """Cross-platform timeout context manager"""
    timeout_occurred = threading.Event()
    timer = threading.Timer(seconds, timeout_occurred.set)
    timer.start()
    
    try:
        yield
        if timeout_occurred.is_set():
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
    finally:
        timer.cancel()
        
def apply_hmm_model(model: GaussianHMM, data: pd.DataFrame, observations: np.ndarray) -> Tuple[pd.DataFrame, int]:
    """Apply the trained HMM model to the data and predict hidden states."""
    predicted_states = model.predict(observations)
    
    if len(predicted_states) != len(data):
        if len(predicted_states) < len(data):
            last_state = predicted_states[-1] if len(predicted_states) > 0 else 0
            predicted_states = np.concatenate([predicted_states, np.full(len(data) - len(predicted_states), last_state)])
        else:
            predicted_states = predicted_states[:len(data)]
    
    state_mapping = {0: "bearish weak", 1: "bullish weak", 2: "bearish strong", 3: "bullish strong"}
    
    data = data.copy()
    data['state'] = [state_mapping.get(s, f"State {s}") for s in predicted_states]
    
    last_state = predicted_states[-1] if len(predicted_states) > 0 else 0
    
    try:
        next_state_probs = model.transmat_[last_state]
        next_state_probs = np.ones(len(next_state_probs)) / len(next_state_probs) if np.isnan(next_state_probs).any() or np.isinf(next_state_probs).any() else next_state_probs
        next_state = int(np.argmax(next_state_probs))
    except (AttributeError, IndexError):
        next_state = 0
    
    return data, next_state

def compute_state_using_standard_deviation(durations: pd.DataFrame) -> int:
    """Calculate indicator based on mean and standard deviation of durations."""
    mean_duration, std_duration = durations['duration'].mean(), durations['duration'].std()
    last_duration = durations.iloc[-1]['duration']
    return 0 if (mean_duration - std_duration <= last_duration <= mean_duration + std_duration) else 1
    
def compute_state_using_hmm(durations: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Computes hidden states from duration data using a Gaussian HMM."""
    if len(durations) < 2:
        durations_copy = durations.copy()
        durations_copy['hidden_state'] = 0
        return durations_copy, 0
    
    try:
        model = GaussianHMM(n_components=min(2, len(durations)), covariance_type="diag", n_iter=10, random_state=36)
        model.fit(durations[['duration']].values)
        hidden_states = model.predict(durations[['duration']].values)
        durations_copy = durations.copy()
        durations_copy['hidden_state'] = hidden_states
        return durations_copy, int(hidden_states[-1])
    
    except Exception as e:
        logger.model(f"HMM fitting failed: {e}. Using default state assignment.")
        durations_copy = durations.copy()
        durations_copy['hidden_state'] = 0
        return durations_copy, 0

def calculate_composite_scores_association_rule_mining(rules: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite score with better infinity handling"""
    if rules.empty:
        return rules
    
    numeric_cols = rules.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in rules.columns:
            rules[col] = rules[col].replace([np.inf, -np.inf], np.nan)
            fill_value = rules[col].median() if rules[col].notna().any() else 0.0 # type: ignore
            rules[col] = rules[col].fillna(fill_value)
    
    metrics = [m for m in ['antecedent support', 'consequent support', 'support', 'confidence',
                        'lift', 'representativity', 'leverage', 'conviction',
                        'zhangs_metric', 'jaccard', 'certainty', 'kulczynski'] if m in rules.columns]
    
    if not metrics:
        rules['composite_score'] = 0.0
        return rules
    
    rules_normalized = rules.copy()
    
    if len(rules_normalized) > 0:
        try:            
            for metric in metrics:
                values = np.where(np.isfinite(rules_normalized[metric].values.astype(np.float64)), rules_normalized[metric].values.astype(np.float64), 0.0)
                
                mean_val, std_val = np.mean(values), np.std(values)
                
                if std_val > 0 and np.isfinite(std_val):
                    rules_normalized[metric] = np.clip((values - mean_val) / std_val, -5, 5)
                else:
                    rules_normalized[metric] = 0.0
                    
        except Exception as e:
            logger.data(f"Manual normalization failed: {e}. Using raw values.")
            pass
    
    rules_normalized['composite_score'] = rules_normalized[metrics].mean(axis=1) if metrics else 0.0
    
    return rules_normalized.sort_values(by='composite_score', ascending=False)

def compute_state_using_association_rule_mining(durations: pd.DataFrame) -> Tuple[int, int]:
    """Perform association rule mining on the durations DataFrame."""
    bins, labels = [0, 15, 30, 100], ['state_1', 'state_2', 'state_3']
    durations['duration_bin'] = pd.cut(durations['duration'], bins=bins, labels=labels, right=False)
    durations['transaction'] = durations[['state', 'duration_bin']].apply(
        lambda x: [str(x['state']), str(x['duration_bin'])], axis=1
    )

    te = TransactionEncoder()
    te_ary = te.fit(durations['transaction']).transform(durations['transaction'])
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_) # type: ignore

    frequent_itemsets_apriori = pd.DataFrame()
    for min_support_val in [0.2, 0.15, 0.1, 0.05]:
        try:
            frequent_itemsets_apriori = apriori(df_transactions, min_support=min_support_val, use_colnames=True)
            if not frequent_itemsets_apriori.empty:
                break
        except Exception as e:
            logger.analysis(f"Error with Apriori min_support={min_support_val}: {e}")
    
    rules_apriori = pd.DataFrame()
    if not frequent_itemsets_apriori.empty:
        try:
            rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.6)
        except Exception as e:
            logger.warning(f"Error generating association rules for Apriori: {e}")

    rules_apriori_sorted = calculate_composite_scores_association_rule_mining(rules_apriori)
    top_antecedents_apriori = rules_apriori_sorted.iloc[0]['antecedents'] if not rules_apriori_sorted.empty else frozenset()

    frequent_itemsets_fpgrowth = pd.DataFrame()
    for min_support_val_fp in [0.2, 0.15, 0.1, 0.05]:
        try:
            frequent_itemsets_fpgrowth = fpgrowth(df_transactions, min_support=min_support_val_fp, use_colnames=True)
            if not frequent_itemsets_fpgrowth.empty:
                break
        except Exception as e:
            logger.analysis(f"Error with FP-Growth min_support={min_support_val_fp}: {e}")

    rules_fpgrowth = pd.DataFrame()
    if not frequent_itemsets_fpgrowth.empty:
        try:
            rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.6) # type: ignore
        except Exception as e:
            logger.warning(f"Error generating association rules for FP-Growth: {e}")
            
    rules_fpgrowth_sorted = calculate_composite_scores_association_rule_mining(rules_fpgrowth)
    top_antecedents_fpgrowth = rules_fpgrowth_sorted.iloc[0]['antecedents'] if not rules_fpgrowth_sorted.empty else frozenset()
    
    top_apriori, top_fpgrowth = 0, 0
    STATE_MAPPING = {"bearish weak": 0, "bullish weak": 1, "bearish strong": 2, "bullish strong": 3}

    for item in top_antecedents_apriori:
        if item in STATE_MAPPING:
            top_apriori = STATE_MAPPING[item]
            break

    for item in top_antecedents_fpgrowth:
        if item in STATE_MAPPING:
            top_fpgrowth = STATE_MAPPING[item]
            break
    
    return top_apriori, top_fpgrowth

def compute_state_using_k_means(durations: pd.DataFrame) -> int:
    """Compute the trading state using K-Means clustering on the duration feature."""
    if len(durations) < 3:
        return 0
    
    try:
        kmeans = KMeans(n_clusters=2, random_state=42, max_iter=300)
        durations['cluster'] = kmeans.fit_predict(durations[['duration']])
    except Exception as e:
        logger.model(f"K-Means clustering failed: {e}. Using default cluster 0.")
        durations['cluster'] = 0

    return int(durations.iloc[-1]['cluster'])
            
def calculate_all_state_durations(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the duration of all consecutive state segments."""
    df = data.copy()
    df['group'] = (df['state'] != df['state'].shift()).cumsum()
    
    return df.groupby('group').agg(     # type: ignore
        state=('state', 'first'),
        start_time=('state', lambda s: s.index[0]),
        duration=('state', 'size')
    ).reset_index(drop=True)
    
STATE_MAPPING = {
    "bearish weak": 0,
    "bullish weak": 1,
    "bearish strong": 2,
    "bullish strong": 3
}

_thread_local = threading.local()

def prevent_infinite_loop(max_calls=3):
    """Decorator to prevent infinite loops in function calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(_thread_local, 'call_counts'):
                _thread_local.call_counts = {}
            
            func_name = func.__name__
            if func_name not in _thread_local.call_counts:
                _thread_local.call_counts[func_name] = 0
            
            _thread_local.call_counts[func_name] += 1
            
            try:
                if _thread_local.call_counts[func_name] > 1:
                    logger.warning(f"Multiple calls detected for {func_name} ({_thread_local.call_counts[func_name]}). Possible infinite loop.")
                    if _thread_local.call_counts[func_name] > max_calls:
                        logger.error(f"Too many recursive calls for {func_name}. Breaking to prevent infinite loop.")
                        return HMM_KAMA(0, 0, 0, 0, 0, 0)
                
                return func(*args, **kwargs)
            finally:
                _thread_local.call_counts[func_name] = 0
        
        return wrapper
    return decorator

@prevent_infinite_loop(max_calls=3)
def hmm_kama(df, optimizing_params):
    try:
        with timeout_context(30):
            if df is None or df.empty or 'close' not in df.columns or len(df) < 20:
                raise ValueError(f"Invalid DataFrame: empty={df.empty if df is not None else True}, has close={'close' in df.columns if df is not None else False}, len={len(df) if df is not None else 0}")
            
            if df['close'].std() == 0 or pd.isna(df['close'].std()):
                raise ValueError("Price data has no variance")
                
            min_required = max(optimizing_params.window_kama, optimizing_params.window_size, 10)
            if len(df) < min_required:
                raise ValueError(f"Insufficient data: got {len(df)}, need at least {min_required}")
            
            df_clean = df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                if pd.notna(df_clean[col].quantile(0.99)) and pd.notna(df_clean[col].quantile(0.01)):
                    df_clean[col] = df_clean[col].clip(lower=df_clean[col].quantile(0.01)*10, upper=df_clean[col].quantile(0.99)*10)
            
            df_clean = df_clean.ffill().bfill()
            
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            
            hmm_kama_result = HMM_KAMA(0, 0, 0, 0, 0, 0)
            
            observations = prepare_observations(df_clean, optimizing_params)
            model = train_hmm(observations, n_components=4)
            data, next_state = apply_hmm_model(model, df_clean, observations)
            hmm_kama_result.next_state_with_hmm_kama = cast(Literal[0, 1, 2, 3], next_state)
            
            all_duration = calculate_all_state_durations(data)
            
            hmm_kama_result.current_state_of_state_using_std = cast(Literal[0, 1], compute_state_using_standard_deviation(all_duration))
            
            if all_duration['state'].nunique() <= 1:
                all_duration['state_encoded'] = 0            
            else:
                all_duration['state_encoded'] = LabelEncoder().fit_transform(all_duration['state'])
                
            all_duration, last_hidden_state = compute_state_using_hmm(all_duration)
            hmm_kama_result.current_state_of_state_using_hmm = cast(Literal[0, 1], min(1, max(0, last_hidden_state)))
            
            top_apriori, top_fpgrowth = compute_state_using_association_rule_mining(all_duration)
            hmm_kama_result.state_high_probabilities_using_arm_apriori = cast(Literal[0, 1, 2, 3], top_apriori)
            hmm_kama_result.state_high_probabilities_using_arm_fpgrowth = cast(Literal[0, 1, 2, 3], top_fpgrowth)
            
            hmm_kama_result.current_state_of_state_using_kmeans = cast(Literal[0, 1], compute_state_using_k_means(all_duration))
            
            return hmm_kama_result
        
    except Exception as e:
        logger.error(f"Error in hmm_kama: {str(e)}")
        return HMM_KAMA(0, 0, 0, 0, 0, 0)
