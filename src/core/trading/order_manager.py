"""
Order Manager Module - Core order management system với MT5 integration
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

from .order_types import (
    Order, OrderRequest, OrderType, OrderStatus, OrderTimeInForce,
    mt5_order_type_mapping, mt5_time_in_force_mapping
)
from .order_validator import OrderValidator

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


class OrderManager(BaseSystem):
    """Professional Order Management System"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("OrderManager")
        
        # Configuration
        self.config = config or {}
        self.max_concurrent_orders = self.config.get('max_concurrent_orders', 10)
        self.order_timeout = self.config.get('order_timeout', 30)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        
        # Core components
        self.validator = OrderValidator()
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_orders)
        
        # Order storage
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.pending_requests: Dict[str, OrderRequest] = {}
        
        # Event callbacks
        self.order_callbacks: Dict[str, List[Callable]] = {
            'order_created': [],
            'order_filled': [],
            'order_cancelled': [],
            'order_rejected': [],
            'order_modified': []
        }
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0,
            'total_volume': 0.0,
            'total_profit': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.is_monitoring = False
        
        logger.info("OrderManager initialized")
    
    def start(self):
        """Start order manager"""
        super().start()
        
        # Initialize MT5 connection
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_orders, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("OrderManager started")
    
    def stop(self):
        """Stop order manager"""
        super().stop()
        
        # Stop monitoring
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Cancel all pending orders
        self.cancel_all_orders("System shutdown")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Shutdown MT5
        mt5.shutdown()
        
        logger.info("OrderManager stopped")
    
    def submit_order(self, order_request: OrderRequest) -> Tuple[bool, str, Optional[str]]:
        """Submit order for execution"""
        try:
            with self.lock:
                # Generate order ID
                order_id = str(uuid.uuid4())
                
                # Validate order
                is_valid, validation_message = self.validator.validate_order(order_request)
                if not is_valid:
                    logger.warning(f"Order validation failed: {validation_message}")
                    return False, f"Validation failed: {validation_message}", None
                
                # Create order object
                order = Order(
                    order_id=order_id,
                    symbol=order_request.symbol,
                    order_type=order_request.order_type,
                    volume=order_request.volume,
                    price=order_request.price,
                    stop_loss=order_request.stop_loss,
                    take_profit=order_request.take_profit,
                    stop_price=order_request.stop_price,
                    comment=order_request.comment,
                    magic_number=order_request.magic_number
                )
                
                # Store order and request
                self.active_orders[order_id] = order
                self.pending_requests[order_id] = order_request
                
                # Submit for execution
                future = self.executor.submit(self._execute_order, order_id)
                
                # Update statistics
                self.stats['total_orders'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('order_created', order)
                
                logger.info(f"Order {order_id} submitted for execution")
                return True, "Order submitted successfully", order_id
                
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return False, f"Submission error: {str(e)}", None
    
    def _execute_order(self, order_id: str) -> bool:
        """Execute order with MT5"""
        try:
            with self.lock:
                order = self.active_orders.get(order_id)
                order_request = self.pending_requests.get(order_id)
                
                if not order or not order_request:
                    logger.error(f"Order {order_id} not found for execution")
                    return False
            
            # Prepare MT5 request
            mt5_request = self._prepare_mt5_request(order_request)
            
            # Execute with retries
            for attempt in range(self.retry_attempts):
                try:
                    # Send order to MT5
                    result = mt5.order_send(mt5_request)
                    
                    if result is None:
                        error_code = mt5.last_error()
                        logger.error(f"MT5 order_send failed: {error_code}")
                        continue
                    
                    # Process result
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Order successful
                        with self.lock:
                            order.ticket = result.order
                            order.status = OrderStatus.FILLED
                            order.filled_volume = result.volume
                            order.remaining_volume = 0.0
                            order.average_fill_price = result.price
                            order.time_filled = datetime.now()
                            
                            # Update statistics
                            self.stats['successful_orders'] += 1
                            self.stats['total_volume'] += result.volume
                            
                            # Update validator counters
                            self.validator.update_daily_counters(trade_executed=True)
                            
                            # Move to history
                            self.order_history.append(order)
                            del self.active_orders[order_id]
                            del self.pending_requests[order_id]
                            
                            # Trigger callbacks
                            self._trigger_callbacks('order_filled', order)
                        
                        logger.info(f"Order {order_id} executed successfully. Ticket: {result.order}")
                        return True
                    
                    elif result.retcode == mt5.TRADE_RETCODE_PLACED:
                        # Order placed (pending)
                        with self.lock:
                            order.ticket = result.order
                            order.status = OrderStatus.PENDING
                        
                        logger.info(f"Order {order_id} placed as pending. Ticket: {result.order}")
                        return True
                    
                    else:
                        # Order failed
                        error_msg = f"MT5 error: {result.retcode} - {result.comment}"
                        logger.warning(f"Order {order_id} failed: {error_msg}")
                        
                        if attempt == self.retry_attempts - 1:
                            # Final attempt failed
                            with self.lock:
                                order.reject(result.retcode, error_msg)
                                self.stats['failed_orders'] += 1
                                
                                # Move to history
                                self.order_history.append(order)
                                del self.active_orders[order_id]
                                del self.pending_requests[order_id]
                                
                                # Trigger callbacks
                                self._trigger_callbacks('order_rejected', order)
                            
                            return False
                        
                        # Wait before retry
                        time.sleep(self.retry_delay)
                
                except Exception as e:
                    logger.error(f"Error executing order {order_id}, attempt {attempt + 1}: {e}")
                    if attempt == self.retry_attempts - 1:
                        with self.lock:
                            order.reject(-1, f"Execution error: {str(e)}")
                            self.stats['failed_orders'] += 1
                            
                            # Move to history
                            self.order_history.append(order)
                            del self.active_orders[order_id]
                            del self.pending_requests[order_id]
                            
                            # Trigger callbacks
                            self._trigger_callbacks('order_rejected', order)
                        
                        return False
                    
                    time.sleep(self.retry_delay)
            
            return False
            
        except Exception as e:
            logger.error(f"Critical error executing order {order_id}: {e}")
            return False
    
    def _prepare_mt5_request(self, order_request: OrderRequest) -> Dict:
        """Prepare MT5 order request"""
        request = {
            "action": mt5.TRADE_ACTION_DEAL if order_request.order_type in [OrderType.MARKET_BUY, OrderType.MARKET_SELL] else mt5.TRADE_ACTION_PENDING,
            "symbol": order_request.symbol,
            "volume": order_request.volume,
            "type": mt5_order_type_mapping(order_request.order_type),
            "magic": order_request.magic_number,
            "comment": order_request.comment,
            "type_time": mt5_time_in_force_mapping(order_request.time_in_force)
        }
        
        # Add price for limit/stop orders
        if order_request.price is not None:
            request["price"] = order_request.price
        
        # Add stop price for stop orders
        if order_request.stop_price is not None:
            request["stoplimit"] = order_request.stop_price
        
        # Add stop loss
        if order_request.stop_loss is not None:
            request["sl"] = order_request.stop_loss
        
        # Add take profit
        if order_request.take_profit is not None:
            request["tp"] = order_request.take_profit
        
        # Add expiration for time-limited orders
        if order_request.expiration is not None:
            request["expiration"] = int(order_request.expiration.timestamp())
        
        return request
    
    def cancel_order(self, order_id: str, reason: str = "") -> Tuple[bool, str]:
        """Cancel specific order"""
        try:
            with self.lock:
                order = self.active_orders.get(order_id)
                if not order:
                    return False, "Order not found"
                
                if order.status not in [OrderStatus.PENDING]:
                    return False, f"Cannot cancel order in status: {order.status.value}"
                
                # Cancel with MT5 if ticket exists
                if order.ticket:
                    request = {
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": order.ticket
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        order.cancel(reason)
                        self.stats['cancelled_orders'] += 1
                        
                        # Move to history
                        self.order_history.append(order)
                        del self.active_orders[order_id]
                        if order_id in self.pending_requests:
                            del self.pending_requests[order_id]
                        
                        # Trigger callbacks
                        self._trigger_callbacks('order_cancelled', order)
                        
                        logger.info(f"Order {order_id} cancelled successfully")
                        return True, "Order cancelled"
                    else:
                        error_msg = f"MT5 cancel failed: {result.retcode if result else 'Unknown error'}"
                        logger.error(error_msg)
                        return False, error_msg
                else:
                    # Cancel local order (not yet sent to MT5)
                    order.cancel(reason)
                    self.stats['cancelled_orders'] += 1
                    
                    # Move to history
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    if order_id in self.pending_requests:
                        del self.pending_requests[order_id]
                    
                    # Trigger callbacks
                    self._trigger_callbacks('order_cancelled', order)
                    
                    return True, "Order cancelled"
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False, f"Cancel error: {str(e)}"
    
    def cancel_all_orders(self, reason: str = "Bulk cancellation") -> int:
        """Cancel all active orders"""
        cancelled_count = 0
        
        with self.lock:
            order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            success, _ = self.cancel_order(order_id, reason)
            if success:
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        with self.lock:
            return self.active_orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        with self.lock:
            return list(self.active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history"""
        with self.lock:
            return self.order_history[-limit:] if limit > 0 else self.order_history
    
    def _monitor_orders(self):
        """Monitor orders for updates"""
        while self.is_monitoring:
            try:
                # Check for order updates from MT5
                self._sync_with_mt5()
                
                # Check for expired orders
                self._check_expired_orders()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Sleep before next check
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                time.sleep(5)
    
    def _sync_with_mt5(self):
        """Sync order status with MT5"""
        try:
            with self.lock:
                for order_id, order in list(self.active_orders.items()):
                    if order.ticket and order.status == OrderStatus.PENDING:
                        # Check if order still exists in MT5
                        mt5_order = mt5.orders_get(ticket=order.ticket)
                        
                        if not mt5_order:
                            # Order might be filled or cancelled
                            # Check in history
                            history = mt5.history_orders_get(ticket=order.ticket)
                            if history:
                                mt5_order_info = history[0]
                                if mt5_order_info.state == mt5.ORDER_STATE_FILLED:
                                    # Order was filled
                                    order.status = OrderStatus.FILLED
                                    order.time_filled = datetime.fromtimestamp(mt5_order_info.time_done)
                                    
                                    # Move to history
                                    self.order_history.append(order)
                                    del self.active_orders[order_id]
                                    
                                    # Trigger callbacks
                                    self._trigger_callbacks('order_filled', order)
                                
                                elif mt5_order_info.state == mt5.ORDER_STATE_CANCELED:
                                    # Order was cancelled
                                    order.cancel("Cancelled by broker")
                                    
                                    # Move to history
                                    self.order_history.append(order)
                                    del self.active_orders[order_id]
                                    
                                    # Trigger callbacks
                                    self._trigger_callbacks('order_cancelled', order)
                        
        except Exception as e:
            logger.error(f"Error syncing with MT5: {e}")
    
    def _check_expired_orders(self):
        """Check for expired orders"""
        try:
            current_time = datetime.now()
            
            with self.lock:
                for order_id, order in list(self.active_orders.items()):
                    # Check timeout
                    if (current_time - order.time_created).total_seconds() > self.order_timeout:
                        if order.status == OrderStatus.PENDING and not order.ticket:
                            # Order hasn't been sent to MT5 yet and timed out
                            order.status = OrderStatus.EXPIRED
                            order.comment += " | Expired (timeout)"
                            
                            # Move to history
                            self.order_history.append(order)
                            del self.active_orders[order_id]
                            if order_id in self.pending_requests:
                                del self.pending_requests[order_id]
                            
                            logger.warning(f"Order {order_id} expired due to timeout")
                        
        except Exception as e:
            logger.error(f"Error checking expired orders: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event in self.order_callbacks:
            self.order_callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, order: Order):
        """Trigger event callbacks"""
        try:
            for callback in self.order_callbacks.get(event, []):
                callback(order)
        except Exception as e:
            logger.error(f"Error triggering callback for {event}: {e}")
    
    def get_statistics(self) -> Dict:
        """Get order statistics"""
        with self.lock:
            return {
                **self.stats,
                'active_orders': len(self.active_orders),
                'pending_requests': len(self.pending_requests),
                'success_rate': (self.stats['successful_orders'] / max(1, self.stats['total_orders'])) * 100,
                'validator_summary': self.validator.get_validation_summary()
            }
    
    def export_orders(self, filename: str = None) -> str:
        """Export orders to JSON"""
        if filename is None:
            filename = f"orders_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'active_orders': [order.to_dict() for order in self.get_active_orders()],
            'order_history': [order.to_dict() for order in self.get_order_history()]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Orders exported to {filename}")
        return filename 