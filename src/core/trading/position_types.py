"""
Position Types Module
Định nghĩa các loại position và cấu trúc dữ liệu
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import MetaTrader5 as mt5


class PositionType(Enum):
    """Loại position"""
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(Enum):
    """Trạng thái position"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


@dataclass
class Position:
    """Position data structure"""
    position_id: str
    ticket: Optional[int] = None  # MT5 position ticket
    symbol: str = ""
    position_type: PositionType = PositionType.BUY
    volume: float = 0.0
    open_price: float = 0.0
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    
    # Timing
    open_time: datetime = None
    close_time: Optional[datetime] = None
    
    # Financial data
    realized_profit: float = 0.0
    unrealized_profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    
    # Metadata
    comment: str = ""
    magic_number: int = 0
    
    # Partial close tracking
    original_volume: float = 0.0
    closed_volume: float = 0.0
    remaining_volume: float = 0.0
    
    # Risk metrics
    risk_amount: float = 0.0
    risk_percentage: float = 0.0
    
    def __post_init__(self):
        if self.open_time is None:
            self.open_time = datetime.now()
        if self.original_volume == 0.0:
            self.original_volume = self.volume
        if self.remaining_volume == 0.0:
            self.remaining_volume = self.volume
    
    def update_current_price(self, price: float):
        """Update current price and calculate unrealized profit"""
        self.current_price = price
        self.unrealized_profit = self.calculate_unrealized_profit()
    
    def calculate_unrealized_profit(self) -> float:
        """Calculate unrealized profit based on current price"""
        if self.current_price == 0.0:
            return 0.0
        
        if self.position_type == PositionType.BUY:
            return (self.current_price - self.open_price) * self.remaining_volume
        else:
            return (self.open_price - self.current_price) * self.remaining_volume
    
    def calculate_realized_profit(self, close_price: float, close_volume: float) -> float:
        """Calculate realized profit for partial/full close"""
        if self.position_type == PositionType.BUY:
            return (close_price - self.open_price) * close_volume
        else:
            return (self.open_price - close_price) * close_volume
    
    def partial_close(self, close_volume: float, close_price: float) -> float:
        """Partially close position"""
        if close_volume > self.remaining_volume:
            close_volume = self.remaining_volume
        
        # Calculate realized profit for this partial close
        realized_profit = self.calculate_realized_profit(close_price, close_volume)
        
        # Update volumes
        self.closed_volume += close_volume
        self.remaining_volume -= close_volume
        self.volume = self.remaining_volume
        
        # Update profit
        self.realized_profit += realized_profit
        
        # Update status
        if self.remaining_volume <= 0.001:  # Essentially closed
            self.status = PositionStatus.CLOSED
            self.close_time = datetime.now()
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED
        
        return realized_profit
    
    def full_close(self, close_price: float) -> float:
        """Fully close position"""
        realized_profit = self.partial_close(self.remaining_volume, close_price)
        self.status = PositionStatus.CLOSED
        self.close_time = datetime.now()
        return realized_profit
    
    def update_stop_loss(self, new_stop_loss: float):
        """Update stop loss level"""
        self.stop_loss = new_stop_loss
    
    def update_take_profit(self, new_take_profit: float):
        """Update take profit level"""
        self.take_profit = new_take_profit
    
    def get_duration(self) -> float:
        """Get position duration in hours"""
        end_time = self.close_time if self.close_time else datetime.now()
        return (end_time - self.open_time).total_seconds() / 3600
    
    def get_return_percentage(self) -> float:
        """Get return percentage based on open price"""
        if self.open_price == 0:
            return 0.0
        
        if self.status == PositionStatus.CLOSED:
            return (self.realized_profit / (self.open_price * self.original_volume)) * 100
        else:
            return (self.unrealized_profit / (self.open_price * self.remaining_volume)) * 100
    
    def is_profitable(self) -> bool:
        """Check if position is profitable"""
        if self.status == PositionStatus.CLOSED:
            return self.realized_profit > 0
        else:
            return self.unrealized_profit > 0
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio"""
        if not self.stop_loss or not self.take_profit:
            return None
        
        if self.position_type == PositionType.BUY:
            risk = abs(self.open_price - self.stop_loss)
            reward = abs(self.take_profit - self.open_price)
        else:
            risk = abs(self.stop_loss - self.open_price)
            reward = abs(self.open_price - self.take_profit)
        
        return reward / risk if risk > 0 else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'ticket': self.ticket,
            'symbol': self.symbol,
            'position_type': self.position_type.value,
            'volume': self.volume,
            'open_price': self.open_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'status': self.status.value,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'close_time': self.close_time.isoformat() if self.close_time else None,
            'realized_profit': self.realized_profit,
            'unrealized_profit': self.unrealized_profit,
            'commission': self.commission,
            'swap': self.swap,
            'comment': self.comment,
            'magic_number': self.magic_number,
            'original_volume': self.original_volume,
            'closed_volume': self.closed_volume,
            'remaining_volume': self.remaining_volume,
            'risk_amount': self.risk_amount,
            'risk_percentage': self.risk_percentage,
            'duration_hours': self.get_duration(),
            'return_percentage': self.get_return_percentage(),
            'is_profitable': self.is_profitable(),
            'risk_reward_ratio': self.get_risk_reward_ratio()
        }


@dataclass
class PositionModifyRequest:
    """Request to modify position"""
    position_id: str
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    comment: str = ""
    
    def validate(self) -> tuple[bool, str]:
        """Validate modification request"""
        if not self.position_id:
            return False, "Position ID is required"
        
        if self.new_stop_loss is not None and self.new_stop_loss <= 0:
            return False, "Stop loss must be positive"
        
        if self.new_take_profit is not None and self.new_take_profit <= 0:
            return False, "Take profit must be positive"
        
        if self.new_stop_loss is None and self.new_take_profit is None:
            return False, "At least one modification parameter is required"
        
        return True, "Valid modification request"


@dataclass
class PositionCloseRequest:
    """Request to close position"""
    position_id: str
    volume: Optional[float] = None  # None means full close
    comment: str = ""
    
    def validate(self) -> tuple[bool, str]:
        """Validate close request"""
        if not self.position_id:
            return False, "Position ID is required"
        
        if self.volume is not None and self.volume <= 0:
            return False, "Close volume must be positive"
        
        return True, "Valid close request"


class PositionSummary:
    """Position portfolio summary"""
    
    def __init__(self, positions: List[Position]):
        self.positions = positions
        self.open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
        self.closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
        self.partially_closed_positions = [p for p in positions if p.status == PositionStatus.PARTIALLY_CLOSED]
        
        # Summary properties
        self.total_positions = len(positions)
        self.symbols = list(set(p.symbol for p in positions))
        self.total_volume = sum(p.original_volume for p in positions)
        self.total_profit = self.get_total_profit()
        
    def get_total_unrealized_profit(self) -> float:
        """Get total unrealized profit"""
        return sum(p.unrealized_profit for p in self.open_positions)
    
    def get_total_realized_profit(self) -> float:
        """Get total realized profit"""
        return sum(p.realized_profit for p in self.closed_positions)
    
    def get_total_profit(self) -> float:
        """Get total profit (realized + unrealized)"""
        return self.get_total_realized_profit() + self.get_total_unrealized_profit()
    
    def get_open_volume_by_symbol(self) -> Dict[str, float]:
        """Get open volume by symbol"""
        volume_by_symbol = {}
        for position in self.open_positions:
            symbol = position.symbol
            if symbol not in volume_by_symbol:
                volume_by_symbol[symbol] = 0.0
            volume_by_symbol[symbol] += position.remaining_volume
        return volume_by_symbol
    
    def get_net_exposure_by_symbol(self) -> Dict[str, float]:
        """Get net exposure by symbol (long - short)"""
        exposure_by_symbol = {}
        for position in self.open_positions:
            symbol = position.symbol
            if symbol not in exposure_by_symbol:
                exposure_by_symbol[symbol] = 0.0
            
            volume = position.remaining_volume
            if position.position_type == PositionType.SELL:
                volume = -volume
            
            exposure_by_symbol[symbol] += volume
        
        return exposure_by_symbol
    
    def get_win_rate(self) -> float:
        """Get win rate percentage"""
        if not self.closed_positions:
            return 0.0
        
        winning_positions = len([p for p in self.closed_positions if p.realized_profit > 0])
        return (winning_positions / len(self.closed_positions)) * 100
    
    def get_average_profit(self) -> float:
        """Get average profit per closed position"""
        if not self.closed_positions:
            return 0.0
        
        return sum(p.realized_profit for p in self.closed_positions) / len(self.closed_positions)
    
    def get_largest_win(self) -> float:
        """Get largest winning trade"""
        if not self.closed_positions:
            return 0.0
        
        return max(p.realized_profit for p in self.closed_positions)
    
    def get_largest_loss(self) -> float:
        """Get largest losing trade"""
        if not self.closed_positions:
            return 0.0
        
        return min(p.realized_profit for p in self.closed_positions)
    
    def get_profit_factor(self) -> float:
        """Get profit factor (gross profit / gross loss)"""
        gross_profit = sum(p.realized_profit for p in self.closed_positions if p.realized_profit > 0)
        gross_loss = abs(sum(p.realized_profit for p in self.closed_positions if p.realized_profit < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary"""
        return {
            'total_positions': len(self.positions),
            'open_positions': len(self.open_positions),
            'closed_positions': len(self.closed_positions),
            'total_unrealized_profit': self.get_total_unrealized_profit(),
            'total_realized_profit': self.get_total_realized_profit(),
            'total_profit': self.get_total_profit(),
            'open_volume_by_symbol': self.get_open_volume_by_symbol(),
            'net_exposure_by_symbol': self.get_net_exposure_by_symbol(),
            'win_rate': self.get_win_rate(),
            'average_profit': self.get_average_profit(),
            'largest_win': self.get_largest_win(),
            'largest_loss': self.get_largest_loss(),
            'profit_factor': self.get_profit_factor()
        } 