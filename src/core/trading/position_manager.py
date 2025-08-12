"""
Position Manager Module
Core position management system với MT5 integration
"""

import uuid
import time
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import json

from .position_types import (
    Position, PositionType, PositionStatus, PositionModifyRequest, 
    PositionCloseRequest, PositionSummary
)

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base system class for inheritance"""
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        self.last_update = datetime.now()
        
    def start(self):
        self.is_active = True
        
    def stop(self):
        self.is_active = False
        
    def get_status(self) -> Dict:
        return {
            'name': self.name,
            'active': self.is_active,
            'last_update': self.last_update.isoformat()
        }


class PositionManager(BaseSystem):
    """Professional Position Management System"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("PositionManager")
        
        # Configuration
        self.config = config or {}
        self.update_interval = self.config.get('update_interval', 1)  # seconds
        self.max_positions = self.config.get('max_positions', 50)
        self.auto_sync = self.config.get('auto_sync', True)
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        
        # MT5 ticket to position ID mapping
        self.ticket_to_position: Dict[int, str] = {}
        
        # Event callbacks
        self.position_callbacks: Dict[str, List[Callable]] = {
            'position_opened': [],
            'position_closed': [],
            'position_modified': [],
            'position_updated': []
        }
        
        # Statistics
        self.stats = {
            'total_positions': 0,
            'open_positions': 0,
            'closed_positions': 0,
            'winning_positions': 0,
            'losing_positions': 0,
            'total_profit': 0.0,
            'total_volume': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.is_monitoring = False
        
        logger.info("PositionManager initialized")
    
    def start(self):
        """Start position manager"""
        super().start()
        
        # Initialize MT5 connection
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")
        
        # Sync existing positions
        if self.auto_sync:
            self._sync_positions_from_mt5()
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_positions, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("PositionManager started")
    
    def stop(self):
        """Stop position manager"""
        super().stop()
        
        # Stop monitoring
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Shutdown MT5
        mt5.shutdown()
        
        logger.info("PositionManager stopped")
    
    def add_position_from_order(self, ticket: int, symbol: str, position_type: PositionType, 
                               volume: float, open_price: float, stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None, comment: str = "",
                               magic_number: int = 0) -> str:
        """Add position from executed order"""
        try:
            with self.lock:
                position_id = str(uuid.uuid4())
                
                position = Position(
                    position_id=position_id,
                    ticket=ticket,
                    symbol=symbol,
                    position_type=position_type,
                    volume=volume,
                    open_price=open_price,
                    current_price=open_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    comment=comment,
                    magic_number=magic_number
                )
                
                # Store position
                self.positions[position_id] = position
                self.ticket_to_position[ticket] = position_id
                
                # Update statistics
                self.stats['total_positions'] += 1
                self.stats['open_positions'] += 1
                self.stats['total_volume'] += volume
                
                # Trigger callbacks
                self._trigger_callbacks('position_opened', position)
                
                logger.info(f"Position {position_id} added from order ticket {ticket}")
                return position_id
                
        except Exception as e:
            logger.error(f"Error adding position from order: {e}")
            return ""
    
    def close_position(self, close_request: PositionCloseRequest) -> Tuple[bool, str]:
        """Close position (full or partial)"""
        try:
            # Validate request
            is_valid, message = close_request.validate()
            if not is_valid:
                return False, f"Invalid close request: {message}"
            
            with self.lock:
                position = self.positions.get(close_request.position_id)
                if not position:
                    return False, "Position not found"
                
                if position.status != PositionStatus.OPEN:
                    return False, f"Cannot close position in status: {position.status.value}"
                
                # Determine close volume
                close_volume = close_request.volume if close_request.volume else position.remaining_volume
                
                if close_volume > position.remaining_volume:
                    close_volume = position.remaining_volume
                
                # Execute close with MT5
                if position.ticket:
                    success, close_price = self._execute_close_with_mt5(position, close_volume, close_request.comment)
                    if not success:
                        return False, f"Failed to close position with MT5: {close_price}"
                else:
                    # Mock close for testing
                    close_price = position.current_price
                
                # Update position
                if close_volume >= position.remaining_volume:
                    # Full close
                    realized_profit = position.full_close(close_price)
                    
                    # Move to history
                    self.position_history.append(position)
                    del self.positions[close_request.position_id]
                    if position.ticket in self.ticket_to_position:
                        del self.ticket_to_position[position.ticket]
                    
                    # Update statistics
                    self.stats['open_positions'] -= 1
                    self.stats['closed_positions'] += 1
                    self.stats['total_profit'] += realized_profit
                    
                    if realized_profit > 0:
                        self.stats['winning_positions'] += 1
                    else:
                        self.stats['losing_positions'] += 1
                    
                    logger.info(f"Position {close_request.position_id} fully closed. Profit: {realized_profit}")
                else:
                    # Partial close
                    realized_profit = position.partial_close(close_volume, close_price)
                    self.stats['total_profit'] += realized_profit
                    
                    logger.info(f"Position {close_request.position_id} partially closed. Volume: {close_volume}, Profit: {realized_profit}")
                
                # Trigger callbacks
                self._trigger_callbacks('position_closed', position)
                
                return True, f"Position closed successfully. Profit: {realized_profit:.2f}"
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False, f"Close error: {str(e)}"
    
    def modify_position(self, modify_request: PositionModifyRequest) -> Tuple[bool, str]:
        """Modify position stop loss and/or take profit"""
        try:
            # Validate request
            is_valid, message = modify_request.validate()
            if not is_valid:
                return False, f"Invalid modify request: {message}"
            
            with self.lock:
                position = self.positions.get(modify_request.position_id)
                if not position:
                    return False, "Position not found"
                
                if position.status != PositionStatus.OPEN:
                    return False, f"Cannot modify position in status: {position.status.value}"
                
                # Execute modification with MT5
                if position.ticket:
                    success, error_msg = self._execute_modify_with_mt5(position, modify_request)
                    if not success:
                        return False, f"Failed to modify position with MT5: {error_msg}"
                
                # Update position locally
                if modify_request.new_stop_loss is not None:
                    position.update_stop_loss(modify_request.new_stop_loss)
                
                if modify_request.new_take_profit is not None:
                    position.update_take_profit(modify_request.new_take_profit)
                
                # Trigger callbacks
                self._trigger_callbacks('position_modified', position)
                
                logger.info(f"Position {modify_request.position_id} modified successfully")
                return True, "Position modified successfully"
                
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False, f"Modify error: {str(e)}"
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        with self.lock:
            return self.positions.get(position_id)
    
    def get_position_by_ticket(self, ticket: int) -> Optional[Position]:
        """Get position by MT5 ticket"""
        with self.lock:
            position_id = self.ticket_to_position.get(ticket)
            return self.positions.get(position_id) if position_id else None
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        with self.lock:
            return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions by symbol"""
        with self.lock:
            return [p for p in self.positions.values() if p.symbol == symbol]
    
    def get_position_history(self, limit: int = 100) -> List[Position]:
        """Get position history"""
        with self.lock:
            return self.position_history[-limit:] if limit > 0 else self.position_history
    
    def get_position_summary(self) -> PositionSummary:
        """Get position portfolio summary"""
        with self.lock:
            all_positions = list(self.positions.values()) + self.position_history
            return PositionSummary(all_positions)
    
    def _execute_close_with_mt5(self, position: Position, volume: float, comment: str) -> Tuple[bool, float]:
        """Execute position close with MT5"""
        try:
            # Determine order type for closing
            if position.position_type == PositionType.BUY:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                order_type = mt5.ORDER_TYPE_BUY
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": volume,
                "type": order_type,
                "position": position.ticket,
                "magic": position.magic_number,
                "comment": comment or f"Close position {position.position_id}"
            }
            
            # Send close request
            result = mt5.order_send(request)
            
            if result is None:
                error_code = mt5.last_error()
                return False, f"MT5 error: {error_code}"
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return True, result.price
            else:
                return False, f"Close failed: {result.retcode} - {result.comment}"
                
        except Exception as e:
            return False, f"Close execution error: {str(e)}"
    
    def _execute_modify_with_mt5(self, position: Position, modify_request: PositionModifyRequest) -> Tuple[bool, str]:
        """Execute position modification with MT5"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "magic": position.magic_number
            }
            
            # Add stop loss if specified
            if modify_request.new_stop_loss is not None:
                request["sl"] = modify_request.new_stop_loss
            else:
                request["sl"] = position.stop_loss or 0.0
            
            # Add take profit if specified
            if modify_request.new_take_profit is not None:
                request["tp"] = modify_request.new_take_profit
            else:
                request["tp"] = position.take_profit or 0.0
            
            # Send modification request
            result = mt5.order_send(request)
            
            if result is None:
                error_code = mt5.last_error()
                return False, f"MT5 error: {error_code}"
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return True, "Modification successful"
            else:
                return False, f"Modify failed: {result.retcode} - {result.comment}"
                
        except Exception as e:
            return False, f"Modify execution error: {str(e)}"
    
    def _sync_positions_from_mt5(self):
        """Sync positions from MT5"""
        try:
            mt5_positions = mt5.positions_get()
            if mt5_positions is None:
                logger.warning("No positions found in MT5")
                return
            
            with self.lock:
                for mt5_pos in mt5_positions:
                    # Check if position already exists
                    if mt5_pos.ticket in self.ticket_to_position:
                        continue
                    
                    # Create position from MT5 data
                    position_id = str(uuid.uuid4())
                    position_type = PositionType.LONG if mt5_pos.type == mt5.POSITION_TYPE_BUY else PositionType.SHORT
                    
                    position = Position(
                        position_id=position_id,
                        ticket=mt5_pos.ticket,
                        symbol=mt5_pos.symbol,
                        position_type=position_type,
                        volume=mt5_pos.volume,
                        open_price=mt5_pos.price_open,
                        current_price=mt5_pos.price_current,
                        stop_loss=mt5_pos.sl if mt5_pos.sl > 0 else None,
                        take_profit=mt5_pos.tp if mt5_pos.tp > 0 else None,
                        time_open=datetime.fromtimestamp(mt5_pos.time),
                        profit=mt5_pos.profit,
                        commission=mt5_pos.commission,
                        swap=mt5_pos.swap,
                        comment=mt5_pos.comment,
                        magic_number=mt5_pos.magic
                    )
                    
                    # Update current price and unrealized profit
                    position.update_current_price(mt5_pos.price_current)
                    
                    # Store position
                    self.positions[position_id] = position
                    self.ticket_to_position[mt5_pos.ticket] = position_id
                    
                    logger.info(f"Synced position {position_id} from MT5 ticket {mt5_pos.ticket}")
                
                # Update statistics
                self.stats['open_positions'] = len(self.positions)
                
        except Exception as e:
            logger.error(f"Error syncing positions from MT5: {e}")
    
    def _monitor_positions(self):
        """Monitor positions for updates"""
        while self.is_monitoring:
            try:
                # Update position prices and profits
                self._update_position_prices()
                
                # Sync with MT5 if enabled
                if self.auto_sync:
                    self._check_mt5_position_changes()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Sleep before next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(5)
    
    def _update_position_prices(self):
        """Update current prices for all open positions"""
        try:
            with self.lock:
                symbols_to_update = set()
                for position in self.positions.values():
                    if position.status == PositionStatus.OPEN:
                        symbols_to_update.add(position.symbol)
                
                # Get current prices for all symbols
                for symbol in symbols_to_update:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        current_price = tick.bid  # Use bid for current price
                        
                        # Update all positions for this symbol
                        for position in self.positions.values():
                            if position.symbol == symbol and position.status == PositionStatus.OPEN:
                                old_profit = position.unrealized_profit
                                position.update_current_price(current_price)
                                
                                # Trigger callback if profit changed significantly
                                if abs(position.unrealized_profit - old_profit) > 1.0:  # $1 threshold
                                    self._trigger_callbacks('position_updated', position)
                
        except Exception as e:
            logger.error(f"Error updating position prices: {e}")
    
    def _check_mt5_position_changes(self):
        """Check for position changes in MT5"""
        try:
            mt5_positions = mt5.positions_get()
            mt5_tickets = set()
            
            if mt5_positions:
                mt5_tickets = {pos.ticket for pos in mt5_positions}
            
            with self.lock:
                # Check for closed positions
                for ticket, position_id in list(self.ticket_to_position.items()):
                    if ticket not in mt5_tickets:
                        # Position was closed in MT5
                        position = self.positions.get(position_id)
                        if position and position.status == PositionStatus.OPEN:
                            # Get close info from history
                            history = mt5.history_deals_get(ticket=ticket)
                            if history:
                                close_deal = history[-1]  # Last deal should be the close
                                close_price = close_deal.price
                                
                                # Close position
                                realized_profit = position.full_close(close_price)
                                
                                # Move to history
                                self.position_history.append(position)
                                del self.positions[position_id]
                                del self.ticket_to_position[ticket]
                                
                                # Update statistics
                                self.stats['open_positions'] -= 1
                                self.stats['closed_positions'] += 1
                                self.stats['total_profit'] += realized_profit
                                
                                if realized_profit > 0:
                                    self.stats['winning_positions'] += 1
                                else:
                                    self.stats['losing_positions'] += 1
                                
                                # Trigger callbacks
                                self._trigger_callbacks('position_closed', position)
                                
                                logger.info(f"Position {position_id} closed by MT5. Profit: {realized_profit}")
                
        except Exception as e:
            logger.error(f"Error checking MT5 position changes: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event in self.position_callbacks:
            self.position_callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, position: Position):
        """Trigger event callbacks"""
        try:
            for callback in self.position_callbacks.get(event, []):
                callback(position)
        except Exception as e:
            logger.error(f"Error triggering callback for {event}: {e}")
    
    def get_statistics(self) -> Dict:
        """Get position statistics"""
        with self.lock:
            summary = self.get_position_summary()
            return {
                **self.stats,
                'summary': summary.to_dict()
            }
    
    def export_positions(self, filename: str = None) -> str:
        """Export positions to JSON"""
        if filename is None:
            filename = f"positions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'open_positions': [pos.to_dict() for pos in self.get_open_positions()],
            'position_history': [pos.to_dict() for pos in self.get_position_history()]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Positions exported to {filename}")
        return filename 