"""
Interactive Brokers API Integration
Ultimate XAU Super System V4.0
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time

class IBConnector(EWrapper, EClient):
    """Interactive Brokers API Connection"""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.positions = {}
        self.orders = {}
        
    def connect_ib(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """Connect to IB Gateway/TWS"""
        try:
            self.connect(host, port, client_id)
            
            # Start message processing thread
            thread = threading.Thread(target=self.run)
            thread.daemon = True
            thread.start()
            
            time.sleep(2)  # Wait for connection
            return self.isConnected()
            
        except Exception as e:
            print(f"IB connection error: {e}")
            return False
            
    def nextValidId(self, order_id: int):
        """Callback for next valid order ID"""
        self.next_order_id = order_id
        
    def error(self, req_id: int, error_code: int, error_string: str):
        """Error callback"""
        print(f"IB Error {error_code}: {error_string}")
        
    def create_contract(self, symbol: str, sec_type: str = "CASH", 
                       exchange: str = "IDEALPRO", currency: str = "USD") -> Contract:
        """Create trading contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract
        
    def place_market_order(self, symbol: str, quantity: int, action: str) -> int:
        """Place market order"""
        contract = self.create_contract(symbol)
        
        order = Order()
        order.action = action  # "BUY" or "SELL"
        order.orderType = "MKT"
        order.totalQuantity = abs(quantity)
        
        order_id = self.next_order_id
        self.placeOrder(order_id, contract, order)
        self.next_order_id += 1
        
        return order_id
        
    def get_account_summary(self):
        """Request account summary"""
        self.reqAccountSummary(1, "All", "TotalCashValue,NetLiquidation")

# Global IB connector
ib_connector = IBConnector()
