from typing import Optional, List, Dict
from datetime import datetime
from dataclasses import dataclass, field
import logging

# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class Candle:
    """
    A class representing a candlestick in financial market data.

    Attributes:
        timestamp (Optional[datetime]): The timestamp of the candlestick, representing when this price data was recorded.
        open (float): The opening price of the asset during this candlestick period.
        high (float): The highest price of the asset during this candlestick period.
        low (float): The lowest price of the asset during this candlestick period.
        close (float): The closing price of the asset during this candlestick period.
        volume (float): The trading volume of the asset during this candlestick period.
    """
    timestamp: Optional[datetime] = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    # For backward compatibility
    index: Optional[datetime] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate the candle data after initialization"""
        # For backward compatibility: map index to timestamp if provided
        if self.index is not None and self.timestamp is None:
            self.timestamp = self.index
            
        # Validate price values
        if self.high < self.low:
            logger.warning(f"Candle has high ({self.high}) < low ({self.low}). Swapping values.")
            self.high, self.low = self.low, self.high
            
        # Ensure high includes the highest of open/close
        self.high = max(self.high, self.open, self.close)
        
        # Ensure low includes the lowest of open/close
        self.low = min(self.low, self.open, self.close)
        
    def __str__(self) -> str:
        """String representation of the candle"""
        candle_type = "Bullish" if self.is_bullish else "Bearish"
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M") if self.timestamp else "No timestamp"
        return f"{candle_type} Candle at {time_str}: O:{self.open:.5f} H:{self.high:.5f} L:{self.low:.5f} C:{self.close:.5f} V:{self.volume:.1f}"

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish (green) candle"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish (red) candle"""
        return self.close < self.open
        
    @property
    def is_doji(self) -> bool:
        """Check if this is a doji candle (open ≈ close)"""
        # A doji has body size < 10% of the candle range
        return self.body_size / (self.high - self.low) < 0.1 if self.high != self.low else True
    
    @property
    def body_size(self) -> float:
        """Get the absolute size of the candle body"""
        return abs(self.close - self.open)
    
    @property
    def body_percent(self) -> float:
        """Get the body size as percentage of candle range"""
        range_size = self.high - self.low
        if range_size == 0:
            return 0.0
        return (self.body_size / range_size) * 100
    
    @property
    def upper_wick(self) -> float:
        """Get the size of the upper wick/shadow"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Get the size of the lower wick/shadow"""
        return min(self.open, self.close) - self.low
    
    @property
    def range(self) -> float:
        """Get the full range of the candle (high to low)"""
        return self.high - self.low
    
    @property
    def midpoint(self) -> float:
        """Get the midpoint of the candle range"""
        return (self.high + self.low) / 2
    
    @property
    def typical_price(self) -> float:
        """Get the typical price (high + low + close) / 3"""
        return (self.high + self.low + self.close) / 3
    
    def movement_percent(self) -> float:
        """Calculate the percentage movement from open to close"""
        if self.open == 0:
            return 0.0
        return ((self.close - self.open) / self.open) * 100
    
    def contains_price(self, price: float) -> bool:
        """Check if a price is within the candle's range"""
        return self.low <= price <= self.high
    
    def body_contains_price(self, price: float) -> bool:
        """Check if a price is within the candle's body"""
        body_low = min(self.open, self.close)
        body_high = max(self.open, self.close)
        return body_low <= price <= body_high
    
    # Pattern detection methods
    
    def is_hammer(self) -> bool:
        """
        Check if this candle forms a hammer pattern.
        
        A hammer has:
        - Small body at the top of the candle
        - Very small or no upper wick
        - Long lower wick (at least 2x body size)
        """
        if self.body_size == 0:
            return False
            
        # Body in upper third of candle
        body_low = min(self.open, self.close)
        body_position = (body_low - self.low) / self.range if self.range > 0 else 0
        
        # Lower wick at least 2x body size
        lower_to_body_ratio = self.lower_wick / self.body_size if self.body_size > 0 else 0
        
        # Upper wick small relative to lower wick
        upper_to_lower_ratio = self.upper_wick / self.lower_wick if self.lower_wick > 0 else float('inf')
        
        return (body_position > 0.6 and  # Body in top third
                lower_to_body_ratio > 2.0 and  # Long lower wick
                upper_to_lower_ratio < 0.3)  # Small upper wick
    
    def is_shooting_star(self) -> bool:
        """
        Check if this candle forms a shooting star pattern.
        
        A shooting star has:
        - Small body at the bottom of the candle
        - Long upper wick (at least 2x body size)
        - Very small or no lower wick
        """
        if self.body_size == 0:
            return False
            
        # Body in lower third of candle
        body_high = max(self.open, self.close)
        body_position = (self.high - body_high) / self.range if self.range > 0 else 0
        
        # Upper wick at least 2x body size
        upper_to_body_ratio = self.upper_wick / self.body_size if self.body_size > 0 else 0
        
        # Lower wick small relative to upper wick
        lower_to_upper_ratio = self.lower_wick / self.upper_wick if self.upper_wick > 0 else float('inf')
        
        return (body_position > 0.6 and  # Body in bottom third
                upper_to_body_ratio > 2.0 and  # Long upper wick
                lower_to_upper_ratio < 0.3)  # Small lower wick
    
    def is_engulfing(self, previous_candle: 'Candle') -> bool:
        """
        Check if this candle engulfs the previous one (bullish or bearish engulfing).
        
        Args:
            previous_candle (Candle): The previous candlestick
            
        Returns:
            bool: True if this candle engulfs the previous one
        """
        if not previous_candle:
            return False
            
        # Different trends (one bullish, one bearish)
        if self.is_bullish == previous_candle.is_bullish:
            return False
            
        # Current candle body fully contains previous candle body
        prev_body_low = min(previous_candle.open, previous_candle.close)
        prev_body_high = max(previous_candle.open, previous_candle.close)
        
        curr_body_low = min(self.open, self.close)
        curr_body_high = max(self.open, self.close)
        
        return curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
    
    @staticmethod
    def detect_patterns(candles: List['Candle'], lookback: int = 5) -> Dict[str, List[int]]:
        """
        Detect common candlestick patterns in a list of candles.
        
        Args:
            candles (List[Candle]): List of candlesticks to analyze
            lookback (int): Number of candles to look back for pattern detection
            
        Returns:
            Dict[str, List[int]]: Dictionary mapping pattern names to lists of indices
                                where those patterns were found
        """
        if len(candles) < lookback:
            return {}
            
        patterns = {
            "hammer": [],
            "shooting_star": [],
            "engulfing": [],
            "doji": []
        }
        
        for i in range(lookback - 1, len(candles)):
            curr = candles[i]
            
            # Individual candle patterns
            if curr.is_hammer():
                patterns["hammer"].append(i)
                
            if curr.is_shooting_star():
                patterns["shooting_star"].append(i)
                
            if curr.is_doji:
                patterns["doji"].append(i)
                
            # Multi-candle patterns
            if i > 0 and curr.is_engulfing(candles[i-1]):
                patterns["engulfing"].append(i)
                
        return patterns
        
    @staticmethod
    def from_dict(data: dict) -> 'Candle':
        """
        Create a Candle from a dictionary, handling different key name variations.
        
        Args:
            data (dict): Dictionary containing candle data
        
        Returns:
            Candle: A new candle instance
        """
        # Handle different possible key names
        timestamp = data.get('timestamp') or data.get('time') or data.get('date')
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.warning(f"Could not parse timestamp: {timestamp}")
                    timestamp = None
        
        return Candle(
            timestamp=timestamp,
            open=float(data.get('open', 0.0)),
            high=float(data.get('high', 0.0)),
            low=float(data.get('low', 0.0)),
            close=float(data.get('close', 0.0)),
            volume=float(data.get('volume', 0.0))
        )

@dataclass
class OptimizingParameters:
    """Parameters for strategy optimization"""
    take_profit: float = 0.01  # Take profit percentage
    stop_loss: float = 0.01    # Stop loss percentage
    window_rolling: int = 200  # Rolling window size
