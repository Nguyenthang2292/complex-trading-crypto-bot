from dataclasses import dataclass
from typing import NamedTuple

@dataclass
class OptimizingParameters:
    """
    Configuration parameters for HMM-based trading strategies.
    
    Stores optimized parameters for two main Hidden Markov Model trading strategies:
    - HIGH_ORDER_HMM: Higher-order HMM for complex pattern recognition
    - HMM_KAMA: HMM combined with Kaufman's Adaptive Moving Average
    
    Attributes:
        orders_argrelextrema: Order parameter for relative extrema detection (HIGH_ORDER_HMM)
        strict_mode: Enable strict filtering mode for signal validation (HIGH_ORDER_HMM)
        fast_kama: Fast smoothing constant for KAMA calculation (HMM_KAMA)
        slow_kama: Slow smoothing constant for KAMA calculation (HMM_KAMA)
        window_kama: Window size for KAMA smoothing (HMM_KAMA)
        window_size: Rolling window size for both strategies
        lot_size_ratio: Position sizing ratio relative to account balance
        take_profit_pct: Take profit threshold as percentage
        stop_loss_pct: Stop loss threshold as percentage
    """
    orders_argrelextrema: int = 5
    strict_mode: bool = False
    fast_kama: int = 2
    slow_kama: int = 30
    window_kama: int = 10
    window_size: int = 200
    lot_size_ratio: float = 1e-6
    take_profit_pct: float = 0.0001
    stop_loss_pct: float = 0.0001
    
    def __post_init__(self) -> None:
        """
        Validate parameter values after initialization.
        
        Raises:
            ValueError: If any parameter is outside acceptable range
        """
        if self.orders_argrelextrema < 1:
            raise ValueError("orders_argrelextrema must be >= 1")
        if self.fast_kama < 1 or self.slow_kama < 1:
            raise ValueError("KAMA parameters must be >= 1")
        if self.fast_kama >= self.slow_kama:
            raise ValueError("fast_kama must be < slow_kama")
        if self.window_kama < 1 or self.window_size < 1:
            raise ValueError("Window sizes must be >= 1")
        if self.lot_size_ratio <= 0:
            raise ValueError("lot_size_ratio must be > 0")
        if self.take_profit_pct <= 0 or self.stop_loss_pct <= 0:
            raise ValueError("Profit/loss percentages must be > 0")
        

class HMMHighOrderResult(NamedTuple):
    """
    Represents the output of the high-order Hidden Markov Model analysis.

    Attributes:
        next_state_with_high_order_hmm: The predicted next market state.
        next_state_probability: The probability associated with the prediction.
    """
    next_state_with_high_order_hmm: int
    next_state_probability: float

class HMMKamaResult(NamedTuple):
    """
    Represents the output of the HMM-KAMA model analysis.

    Attributes:
        next_state_with_hmm_kama: The primary predicted next state from HMM-KAMA.
        current_state_of_state_using_std: Market state derived from volatility.
        current_state_of_state_using_hmm: Market state derived from HMM.
        current_state_of_state_using_kmeans: Market state derived from clustering.
        state_high_probabilities_using_arm_apriori: State confirmed by Apriori.
        state_high_probabilities_using_arm_fpgrowth: State confirmed by FP-Growth.
    """
    next_state_with_hmm_kama: int
    current_state_of_state_using_std: int
    current_state_of_state_using_hmm: int
    current_state_of_state_using_kmeans: int
    state_high_probabilities_using_arm_apriori: int
    state_high_probabilities_using_arm_fpgrowth: int
        
