"""
Order Types Module
Định nghĩa các loại lệnh giao dịch và cấu trúc dữ liệu
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import MetaTrader5 as mt5


class OrderType(Enum):
    """Các loại lệnh giao dịch"""
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    STOP_BUY = "STOP_BUY"
    STOP_SELL = "STOP_SELL"
    STOP_LIMIT_BUY = "STOP_LIMIT_BUY"
    STOP_LIMIT_SELL = "STOP_LIMIT_SELL"


class OrderStatus(Enum):
    """Trạng thái lệnh"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderTimeInForce(Enum):
    """Thời gian hiệu lực lệnh"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Day order


@dataclass
class OrderRequest:
    """Yêu cầu đặt lệnh"""
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_price: Optional[float] = None  # For stop orders
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    expiration: Optional[datetime] = None
    comment: str = ""
    magic_number: int = 0
    
    def validate(self) -> tuple[bool, str]:
        """Validate order request"""
        try:
            # Basic validation
            if not self.symbol:
                return False, "Symbol is required"
            
            if self.volume <= 0:
                return False, "Volume must be positive"
            
            if self.volume > 100:  # Max volume check
                return False, "Volume exceeds maximum allowed"
            
            # Price validation for limit orders
            if self.order_type in [OrderType.LIMIT_BUY, OrderType.LIMIT_SELL]:
                if self.price is None or self.price <= 0:
                    return False, "Price is required for limit orders"
            
            # Stop price validation for stop orders
            if self.order_type in [OrderType.STOP_BUY, OrderType.STOP_SELL, 
                                 OrderType.STOP_LIMIT_BUY, OrderType.STOP_LIMIT_SELL]:
                if self.stop_price is None or self.stop_price <= 0:
                    return False, "Stop price is required for stop orders"
            
            # Stop limit validation
            if self.order_type in [OrderType.STOP_LIMIT_BUY, OrderType.STOP_LIMIT_SELL]:
                if self.price is None or self.price <= 0:
                    return False, "Limit price is required for stop-limit orders"
            
            # Stop loss validation
            if self.stop_loss is not None and self.stop_loss <= 0:
                return False, "Stop loss must be positive"
            
            # Take profit validation
            if self.take_profit is not None and self.take_profit <= 0:
                return False, "Take profit must be positive"
            
            return True, "Valid order request"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class Order:
    """Lệnh giao dịch"""
    order_id: str
    ticket: Optional[int] = None  # MT5 ticket
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET_BUY
    volume: float = 0.0
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: float = 0.0
    remaining_volume: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    time_created: datetime = None
    time_filled: Optional[datetime] = None
    time_cancelled: Optional[datetime] = None
    comment: str = ""
    magic_number: int = 0
    error_code: Optional[int] = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.time_created is None:
            self.time_created = datetime.now()
        if self.remaining_volume == 0.0:
            self.remaining_volume = self.volume
    
    def update_fill(self, fill_volume: float, fill_price: float):
        """Cập nhật thông tin fill"""
        self.filled_volume += fill_volume
        self.remaining_volume = max(0, self.volume - self.filled_volume)
        
        # Update average fill price
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            total_filled_value = (self.filled_volume - fill_volume) * self.average_fill_price + fill_volume * fill_price
            self.average_fill_price = total_filled_value / self.filled_volume
        
        # Update status
        if self.remaining_volume == 0:
            self.status = OrderStatus.FILLED
            self.time_filled = datetime.now()
        elif self.filled_volume > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self, reason: str = ""):
        """Hủy lệnh"""
        self.status = OrderStatus.CANCELLED
        self.time_cancelled = datetime.now()
        if reason:
            self.comment += f" | Cancelled: {reason}"
    
    def reject(self, error_code: int, error_message: str):
        """Từ chối lệnh"""
        self.status = OrderStatus.REJECTED
        self.error_code = error_code
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'ticket': self.ticket,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'volume': self.volume,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_volume': self.filled_volume,
            'remaining_volume': self.remaining_volume,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'swap': self.swap,
            'profit': self.profit,
            'time_created': self.time_created.isoformat() if self.time_created else None,
            'time_filled': self.time_filled.isoformat() if self.time_filled else None,
            'time_cancelled': self.time_cancelled.isoformat() if self.time_cancelled else None,
            'comment': self.comment,
            'magic_number': self.magic_number,
            'error_code': self.error_code,
            'error_message': self.error_message
        }


def mt5_order_type_mapping(order_type: OrderType) -> int:
    """Map OrderType to MT5 order type"""
    mapping = {
        OrderType.MARKET_BUY: mt5.ORDER_TYPE_BUY,
        OrderType.MARKET_SELL: mt5.ORDER_TYPE_SELL,
        OrderType.LIMIT_BUY: mt5.ORDER_TYPE_BUY_LIMIT,
        OrderType.LIMIT_SELL: mt5.ORDER_TYPE_SELL_LIMIT,
        OrderType.STOP_BUY: mt5.ORDER_TYPE_BUY_STOP,
        OrderType.STOP_SELL: mt5.ORDER_TYPE_SELL_STOP,
        OrderType.STOP_LIMIT_BUY: mt5.ORDER_TYPE_BUY_STOP_LIMIT,
        OrderType.STOP_LIMIT_SELL: mt5.ORDER_TYPE_SELL_STOP_LIMIT
    }
    return mapping.get(order_type, mt5.ORDER_TYPE_BUY)


def mt5_time_in_force_mapping(tif: OrderTimeInForce) -> int:
    """Map TimeInForce to MT5 time in force"""
    try:
        mapping = {
            OrderTimeInForce.GTC: getattr(mt5, 'ORDER_TIME_GTC', 0),
            OrderTimeInForce.IOC: getattr(mt5, 'ORDER_TIME_IOC', 1),
            OrderTimeInForce.FOK: getattr(mt5, 'ORDER_TIME_FOK', 2),
            OrderTimeInForce.DAY: getattr(mt5, 'ORDER_TIME_DAY', 3)
        }
        return mapping.get(tif, getattr(mt5, 'ORDER_TIME_GTC', 0))
    except:
        # Fallback if MT5 constants not available
        return 0 