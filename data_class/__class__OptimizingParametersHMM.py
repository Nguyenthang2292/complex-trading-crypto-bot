from dataclasses import dataclass

@dataclass
class OptimizingParametersHMM:
    """
    OptimizingParameters class stores configuration parameters for different trading strategies.
    The class contains parameters for two main strategies:
    - HIGH_ORDER_HMM strategy: Uses higher-order Hidden Markov Models for trading signals
    - HMM_KAMA strategy: Combines Hidden Markov Models with Kaufman's Adaptive Moving Average
    Attributes:
        orders_argrelextrema (int): Order parameter for finding relative extrema in the HIGH_ORDER_HMM strategy
        strict_mode (bool): Whether to use strict mode in the HIGH_ORDER_HMM strategy
        fast_kama (int): Fast coefficient for KAMA calculation in the HMM_KAMA strategy
        slow_kama (int): Slow coefficient for KAMA calculation in the HMM_KAMA strategy
        window_kama (int): Window size for KAMA calculation in the HMM_KAMA strategy
        window_size (int): Rolling window size used in both strategies
        lot_size (float): Size of trading lot
        take_profit (float): Take profit threshold percentage
        stop_loss (float): Stop loss threshold percentage
    """
    def __init__(self):
        # FOR HIGH_ORDER_HMM STRATEGY
        self.orders_argrelextrema: int = 5
        self.strict_mode: bool = False
        
        # FOR HMM_KAMA STRATEGY
        self.fast_kama: int = 2
        self.slow_kama: int = 30
        self.window_kama: int = 10
        
        # FOR BOTH STRATEGY
        self.window_size: int = 200
        self.lot_size_ratio: float = 1e-6
        self.take_profit_pct:float = 0.0001 #take profit percentage  
        self.stop_loss_pct:float = 0.0001 #stop-loss percentage
        
