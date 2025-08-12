"""
Smart Order Routing System
Ultimate XAU Super System V4.0
"""

from typing import Dict, List, Optional
from enum import Enum
import time

class BrokerType(Enum):
    MT5 = "metatrader5"
    IB = "interactive_brokers"

class SmartRouter:
    """Intelligent order routing system"""
    
    def __init__(self):
        self.brokers = {}
        self.routing_rules = {}
        
    def add_broker(self, broker_type: BrokerType, connector):
        """Add broker connector"""
        self.brokers[broker_type] = connector
        
    def route_order(self, order: Dict) -> Dict:
        """Route order to best broker"""
        symbol = order.get('symbol')
        volume = order.get('volume')
        order_type = order.get('type')
        
        # Get best broker for this order
        best_broker = self.select_best_broker(symbol, volume)
        
        if not best_broker:
            return {'success': False, 'error': 'No available broker'}
            
        # Execute order
        result = self.execute_order(best_broker, order)
        
        return {
            'success': result.get('result', False),
            'broker': best_broker.name,
            'order_id': result.get('order_id'),
            'execution_time': time.time()
        }
        
    def select_best_broker(self, symbol: str, volume: float) -> Optional[BrokerType]:
        """Select best broker based on criteria"""
        available_brokers = []
        
        for broker_type, connector in self.brokers.items():
            if self.is_broker_available(connector):
                score = self.calculate_broker_score(broker_type, symbol, volume)
                available_brokers.append((broker_type, score))
                
        if not available_brokers:
            return None
            
        # Return broker with highest score
        return max(available_brokers, key=lambda x: x[1])[0]
        
    def calculate_broker_score(self, broker_type: BrokerType, symbol: str, volume: float) -> float:
        """Calculate broker suitability score"""
        score = 0.0
        
        # Base scores
        if broker_type == BrokerType.MT5:
            score += 8.0  # Good for forex/commodities
        elif broker_type == BrokerType.IB:
            score += 9.0  # Excellent execution
            
        # Symbol-specific scoring
        if 'XAU' in symbol:
            if broker_type == BrokerType.MT5:
                score += 2.0  # MT5 good for gold
                
        # Volume considerations
        if volume > 10.0:
            if broker_type == BrokerType.IB:
                score += 1.0  # IB better for large orders
                
        return score
        
    def is_broker_available(self, connector) -> bool:
        """Check if broker is available"""
        return getattr(connector, 'connected', False)
        
    def execute_order(self, broker_type: BrokerType, order: Dict) -> Dict:
        """Execute order on selected broker"""
        connector = self.brokers.get(broker_type)
        
        if not connector:
            return {'result': False, 'error': 'Broker not found'}
            
        try:
            if broker_type == BrokerType.MT5:
                return connector.place_order(
                    symbol=order['symbol'],
                    order_type=order['type'], 
                    volume=order['volume'],
                    price=order.get('price'),
                    sl=order.get('sl'),
                    tp=order.get('tp')
                )
            elif broker_type == BrokerType.IB:
                order_id = connector.place_market_order(
                    symbol=order['symbol'],
                    quantity=order['volume'],
                    action=order['type']
                )
                return {'result': True, 'order_id': order_id}
                
        except Exception as e:
            return {'result': False, 'error': str(e)}

# Global smart router
smart_router = SmartRouter()
