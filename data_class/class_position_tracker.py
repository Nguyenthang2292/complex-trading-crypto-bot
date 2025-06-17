from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from .class_order import Order

# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class PositionTracker:
    """
    PositionTracker class manages trading positions including opening, closing, and tracking P&L.
    
    The class maintains a list of open positions and calculates both realized and unrealized profits
    and losses across all tracked positions.
    
    Attributes:
        positions (List[Order]): List of current open positions.
        total_realized_pnl (float): Cumulative realized profit/loss from closed positions.
        trade_history (List[Dict]): History of all closed trades
    """
    def __init__(self):
        self.positions: List[Order] = []  # List of current positions
        self.total_realized_pnl: float = 0.00  # Total realized PnL
        self.trade_history: List[Dict[str, Any]] = []  # History of all trades
    
    def open_position(self, position: Order) -> bool:
        """
        Add a new position to the tracker.
        
        Args:
            position (Order): The order to track
            
        Returns:
            bool: True if position was added successfully
        
        Raises:
            ValueError: If position is invalid
        """
        if position is None:
            logger.error("Cannot add None position")
            return False
            
        if not isinstance(position, Order):
            logger.error(f"Expected Order type, got {type(position)}")
            raise ValueError(f"Position must be an Order object, not {type(position)}")
        
        self.positions.append(position)
        logger.info(f"Added position: {position.side} {position.lot_size} @ {position.entry_price} at {position.entry_time}")
        return True

    def close_position(self, position: Order, exit_price: float, 
                    exit_time: datetime, pip_size: float, pip_value: float) -> Dict[str, Any]:
        """
        Close a position and calculate profit.
        
        Args:
            position (Order): The position to close
            exit_price (float): The price at which the position is closed
            exit_time (datetime): When the position was closed
            pip_size (float): The size of a pip for this instrument
            pip_value (float): The value of a pip in account currency
            
        Returns:
            Dict[str, Any]: Trade record with details of the closed position
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if position is None:
            logger.error("Cannot close None position")
            raise ValueError("Position cannot be None")
            
        if exit_price <= 0:
            logger.error(f"Invalid exit price: {exit_price}")
            raise ValueError(f"Exit price must be positive: {exit_price}")
            
        if pip_size <= 0 or pip_value <= 0:
            logger.error(f"Invalid pip parameters: size={pip_size}, value={pip_value}")
            raise ValueError(f"Pip size and value must be positive")
        
        # Calculate PnL
        unrealized = position.update_unrealized_pnl(exit_price, pip_size, pip_value)
        self.total_realized_pnl += unrealized
        
        # Try to remove the position from our list
        if position in self.positions:
            self.positions.remove(position)
        else:
            logger.warning(f"Attempted to close position not in tracker: {position.ticket}")
        
        # Create trade record
        trade_record = {
            'strategy_name': position.strategy_name,
            'strategy_type': position.strategy_type,
            'ticket': position.ticket,
            'symbol': position.symbol,
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'lot_size': position.lot_size,
            'side': position.side,
            'pnl': round(unrealized, 2),
            'pips': self._calculate_pips(position, exit_price)
        }
        
        # Store in history and log
        self.trade_history.append(trade_record)
        logger.info(f"Closed position {position.ticket}: {trade_record['side']} {trade_record['pnl']:.2f}")
        
        return trade_record

    def get_total_unrealized_pnl(self, current_price: float, pip_size: float, pip_value: float) -> float:
        """
        Calculate total unrealized P&L for all open positions.
        
        Args:
            current_price (float): Current market price
            pip_size (float): The size of a pip for this instrument
            pip_value (float): The value of a pip in account currency
            
        Returns:
            float: Total unrealized profit/loss
        """
        if not self.positions:
            return 0.0
            
        try:
            total = sum([pos.update_unrealized_pnl(current_price, pip_size, pip_value) for pos in self.positions])
            return total
        except Exception as e:
            logger.error(f"Error calculating unrealized PnL: {e}")
            return 0.0
    
    def get_position_count(self) -> int:
        """
        Get the number of currently open positions.
        
        Returns:
            int: Number of open positions
        """
        return len(self.positions)
        
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for all closed trades.
        
        Returns:
            Dict[str, Any]: Dictionary with trade statistics
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0
            }
            
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in self.trade_history if trade['pnl'] < 0)
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average profit and loss
        profits = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
        losses = [trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Profit factor (sum of profits / sum of losses)
        profit_factor = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": self.total_realized_pnl
        }
    
    def _calculate_pips(self, position: Order, exit_price: float) -> float:
        """
        Calculate the number of pips moved between entry and exit.
        
        Args:
            position (Order): The position
            exit_price (float): The exit price
            
        Returns:
            float: Number of pips moved (positive for profit, negative for loss)
        """
        direction = 1 if position.side == "LONG" else -1
        pips = direction * (exit_price - position.entry_price) * 10000
        return round(pips, 1)

