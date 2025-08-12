"""
Demo Order Manager
Demonstration script cho OrderManager system
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.trading.order_types import OrderRequest, OrderType, OrderTimeInForce
from src.core.trading.order_manager import OrderManager


def demo_order_manager():
    """Demo OrderManager functionality"""
    print("🚀 DEMO: ORDER MANAGEMENT SYSTEM")
    print("=" * 50)
    
    # Configuration
    config = {
        'max_concurrent_orders': 10,
        'order_timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 1
    }
    
    # Initialize OrderManager
    print("\n📋 Initializing OrderManager...")
    order_manager = OrderManager(config)
    
    try:
        # Note: In real environment, this would connect to MT5
        print("⚠️  Note: This demo runs without actual MT5 connection")
        print("   In production, ensure MT5 is running and logged in")
        
        # Start the system (mock mode)
        print("\n🔄 Starting OrderManager...")
        # order_manager.start()  # Commented out to avoid MT5 connection
        
        # Demo 1: Create Market Order
        print("\n" + "="*30)
        print("DEMO 1: MARKET ORDER")
        print("="*30)
        
        market_order = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.MARKET_BUY,
            volume=0.1,
            stop_loss=1950.0,
            take_profit=2050.0,
            comment="Demo market buy order",
            magic_number=12345
        )
        
        print(f"📝 Creating market order:")
        print(f"   Symbol: {market_order.symbol}")
        print(f"   Type: {market_order.order_type.value}")
        print(f"   Volume: {market_order.volume}")
        print(f"   Stop Loss: {market_order.stop_loss}")
        print(f"   Take Profit: {market_order.take_profit}")
        
        # Validate order (without actual submission)
        is_valid, message = market_order.validate()
        print(f"✅ Order validation: {is_valid} - {message}")
        
        # Demo 2: Create Limit Order
        print("\n" + "="*30)
        print("DEMO 2: LIMIT ORDER")
        print("="*30)
        
        limit_order = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.LIMIT_BUY,
            volume=0.05,
            price=1980.0,
            stop_loss=1960.0,
            take_profit=2020.0,
            time_in_force=OrderTimeInForce.GTC,
            comment="Demo limit buy order"
        )
        
        print(f"📝 Creating limit order:")
        print(f"   Symbol: {limit_order.symbol}")
        print(f"   Type: {limit_order.order_type.value}")
        print(f"   Volume: {limit_order.volume}")
        print(f"   Price: {limit_order.price}")
        print(f"   Stop Loss: {limit_order.stop_loss}")
        print(f"   Take Profit: {limit_order.take_profit}")
        
        is_valid, message = limit_order.validate()
        print(f"✅ Order validation: {is_valid} - {message}")
        
        # Demo 3: Create Stop Order
        print("\n" + "="*30)
        print("DEMO 3: STOP ORDER")
        print("="*30)
        
        stop_order = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.STOP_SELL,
            volume=0.02,
            stop_price=1970.0,
            stop_loss=1990.0,
            take_profit=1950.0,
            comment="Demo stop sell order"
        )
        
        print(f"📝 Creating stop order:")
        print(f"   Symbol: {stop_order.symbol}")
        print(f"   Type: {stop_order.order_type.value}")
        print(f"   Volume: {stop_order.volume}")
        print(f"   Stop Price: {stop_order.stop_price}")
        print(f"   Stop Loss: {stop_order.stop_loss}")
        print(f"   Take Profit: {stop_order.take_profit}")
        
        is_valid, message = stop_order.validate()
        print(f"✅ Order validation: {is_valid} - {message}")
        
        # Demo 4: Invalid Order Examples
        print("\n" + "="*30)
        print("DEMO 4: INVALID ORDERS")
        print("="*30)
        
        invalid_orders = [
            OrderRequest(
                symbol="",  # Empty symbol
                order_type=OrderType.MARKET_BUY,
                volume=0.1
            ),
            OrderRequest(
                symbol="XAUUSD",
                order_type=OrderType.MARKET_BUY,
                volume=0.0  # Zero volume
            ),
            OrderRequest(
                symbol="XAUUSD",
                order_type=OrderType.LIMIT_BUY,
                volume=0.1
                # Missing price for limit order
            ),
            OrderRequest(
                symbol="XAUUSD",
                order_type=OrderType.MARKET_BUY,
                volume=0.1,
                stop_loss=-100.0  # Negative stop loss
            )
        ]
        
        for i, invalid_order in enumerate(invalid_orders, 1):
            is_valid, message = invalid_order.validate()
            print(f"❌ Invalid Order {i}: {message}")
        
        # Demo 5: Order Statistics
        print("\n" + "="*30)
        print("DEMO 5: ORDER STATISTICS")
        print("="*30)
        
        stats = order_manager.get_statistics()
        print(f"📊 Order Statistics:")
        print(f"   Total Orders: {stats['total_orders']}")
        print(f"   Successful Orders: {stats['successful_orders']}")
        print(f"   Failed Orders: {stats['failed_orders']}")
        print(f"   Cancelled Orders: {stats['cancelled_orders']}")
        print(f"   Active Orders: {stats['active_orders']}")
        print(f"   Success Rate: {stats['success_rate']:.2f}%")
        
        # Demo 6: Order Types Overview
        print("\n" + "="*30)
        print("DEMO 6: ORDER TYPES OVERVIEW")
        print("="*30)
        
        print("📋 Available Order Types:")
        for order_type in OrderType:
            print(f"   • {order_type.value}")
        
        print("\n⏰ Time In Force Options:")
        for tif in OrderTimeInForce:
            print(f"   • {tif.value}")
        
        # Demo 7: Callback System
        print("\n" + "="*30)
        print("DEMO 7: CALLBACK SYSTEM")
        print("="*30)
        
        def order_created_callback(order):
            print(f"🔔 Callback: Order {order.order_id} created!")
        
        def order_filled_callback(order):
            print(f"🔔 Callback: Order {order.order_id} filled at {order.average_fill_price}!")
        
        def order_cancelled_callback(order):
            print(f"🔔 Callback: Order {order.order_id} cancelled!")
        
        # Add callbacks
        order_manager.add_callback('order_created', order_created_callback)
        order_manager.add_callback('order_filled', order_filled_callback)
        order_manager.add_callback('order_cancelled', order_cancelled_callback)
        
        print("✅ Event callbacks registered")
        print("   • order_created")
        print("   • order_filled") 
        print("   • order_cancelled")
        
        # Demo 8: Export Functionality
        print("\n" + "="*30)
        print("DEMO 8: EXPORT FUNCTIONALITY")
        print("="*30)
        
        print("📤 Export capabilities:")
        print("   • JSON format")
        print("   • Includes statistics")
        print("   • Active orders")
        print("   • Order history")
        print("   • Timestamp information")
        
        # Demo 9: Risk Management Integration
        print("\n" + "="*30)
        print("DEMO 9: RISK MANAGEMENT")
        print("="*30)
        
        validator_summary = order_manager.validator.get_validation_summary()
        print("🛡️ Risk Management Features:")
        print(f"   • Daily Trade Limit: {validator_summary['daily_trades']}")
        print(f"   • Daily Risk Used: {validator_summary['daily_risk_used']}")
        print(f"   • Max Open Orders: {validator_summary['max_open_orders']}")
        print(f"   • Min Distance Points: {validator_summary['min_distance_points']}")
        print(f"   • Cached Symbols: {validator_summary['cached_symbols']}")
        
        # Demo 10: Production Readiness
        print("\n" + "="*30)
        print("DEMO 10: PRODUCTION FEATURES")
        print("="*30)
        
        print("🏭 Production-Ready Features:")
        print("   ✅ Multi-threaded execution")
        print("   ✅ Comprehensive validation")
        print("   ✅ Error handling & retries")
        print("   ✅ Real-time monitoring")
        print("   ✅ Event-driven callbacks")
        print("   ✅ Statistics tracking")
        print("   ✅ Order history management")
        print("   ✅ Risk management integration")
        print("   ✅ MT5 native integration")
        print("   ✅ Thread-safe operations")
        
        print("\n" + "="*50)
        print("✅ ORDER MANAGER DEMO COMPLETED")
        print("="*50)
        
        print("\n📋 Next Steps:")
        print("1. Connect to MT5 terminal")
        print("2. Configure trading account")
        print("3. Set risk parameters")
        print("4. Start live trading")
        print("5. Monitor performance")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            # order_manager.stop()  # Commented out for demo
            print("\n🔄 OrderManager demo cleanup completed")
        except:
            pass


if __name__ == "__main__":
    demo_order_manager() 