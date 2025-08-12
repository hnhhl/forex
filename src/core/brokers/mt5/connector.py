"""
MetaTrader 5 Integration
Ultimate XAU Super System V4.0
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class MT5Connector:
    """MetaTrader 5 Broker Connection"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        
    def connect(self, login: int, password: str, server: str) -> bool:
        """Connect to MT5 terminal"""
        try:
            if not mt5.initialize():
                return False
                
            if not mt5.login(login, password, server):
                return False
                
            self.connected = True
            self.account_info = mt5.account_info()
            return True
            
        except Exception as e:
            print(f"MT5 connection error: {e}")
            return False
            
    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.connected:
            return {}
            
        info = mt5.account_info()
        return {
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'free_margin': info.margin_free,
            'profit': info.profit
        }
        
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None) -> Dict:
        """Place trading order"""
        if not self.connected:
            return {'result': False, 'error': 'Not connected'}
            
        try:
            # Order request structure
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': volume,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': price or mt5.symbol_info_tick(symbol).ask,
                'sl': sl,
                'tp': tp,
                'magic': 123456,
                'comment': 'XAU System Order',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            return {
                'result': result.retcode == mt5.TRADE_RETCODE_DONE,
                'order_id': result.order,
                'deal_id': result.deal,
                'volume': result.volume,
                'price': result.price
            }
            
        except Exception as e:
            return {'result': False, 'error': str(e)}
            
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        if not self.connected:
            return []
            
        positions = mt5.positions_get()
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': pos.type,
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit
            }
            for pos in positions
        ]

# Global MT5 connector
mt5_connector = MT5Connector()
