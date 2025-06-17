from datetime import datetime
from dataclasses import dataclass
import random

@dataclass
class Order:
    """
    Represents a trading order placed by a strategy.
    The Order class encapsulates all information about a trading position, including
    its entry conditions, size, direction, and profit/loss calculations.
    Attributes:
        ticket (int): Unique identifier for the order, randomly generated.
        strategy_name (str): Name of the strategy that generated this order.
        strategy_type (str): Type classification of the strategy.
        symbol (str): Trading instrument symbol (e.g., "EURUSD").
        entry_time (datetime): Timestamp when the order was entered.
        entry_price (float): Price at which the order was executed.
        lot_size (float): Size of the position in lots.
        side (str): Direction of the trade, either 'LONG' or 'SHORT'.
        realized_pnl (float): Realized profit and loss, initialized to 0.
    Methods:
        update_unrealized_pnl(current_price, pip_size, pip_value): 
            Calculates the current unrealized profit/loss based on market price.
    """
    def __init__(self, strategy_name, strategy_type,
                symbol, entry_time, entry_price, lot_size, side):
        self.ticket: int = random.randint(100000000, 999999999)
        self.strategy_name: str = strategy_name
        self.strategy_type: str = strategy_type
        self.symbol: str = symbol
        self.entry_time: datetime = entry_time
        self.entry_price: float = entry_price
        self.lot_size: float = lot_size
        self.side: str = side
        self.realized_pnl: float = 0.00  # Realized profit

    def update_unrealized_pnl(self, current_price, pip_size, pip_value):
        if self.side == 'LONG':
            return ((current_price - self.entry_price) / pip_size) * pip_value * self.lot_size
        elif self.side == 'SHORT':
            return ((self.entry_price - current_price) / pip_size) * pip_value * self.lot_size
        return 0.00

