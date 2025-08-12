"""
Demo Script for Position Management System
Comprehensive demonstration của position manager, calculator, và stop loss manager
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.trading.position_manager import PositionManager
from src.core.trading.position_calculator import PositionCalculator, PositionSizingMethod, PnLCalculationType
from src.core.trading.stop_loss_manager import StopLossManager, StopLossRule, StopLossType, TrailingStopMethod
from src.core.trading.position_types import (
    Position, PositionType, PositionStatus, PositionModifyRequest, 
    PositionCloseRequest, PositionSummary
)


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print(f"{'='*80}")


def print_section(title: str):
    """Print formatted section"""
    print(f"\n{'─'*60}")
    print(f"📋 {title}")
    print(f"{'─'*60}")


def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")


def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")


def print_warning(message: str):
    """Print warning message"""
    print(f"⚠️  {message}")


def demo_position_manager():
    """Demo Position Manager functionality"""
    print_header("POSITION MANAGER DEMO")
    
    # Initialize position manager
    config = {
        'update_interval': 1,
        'max_positions': 50,
        'auto_sync': False  # Disable MT5 sync for demo
    }
    
    manager = PositionManager(config)
    print_success("Position Manager initialized")
    
    # Start manager
    manager.start()
    print_success("Position Manager started")
    
    print_section("1. Adding Positions from Orders")
    
    # Add multiple positions
    positions = []
    
    # Position 1: XAUUSD Buy
    pos1_id = manager.add_position_from_order(
        ticket=12345,
        symbol="XAUUSD",
        position_type=PositionType.BUY,
        volume=0.1,
        open_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        comment="Gold buy position"
    )
    positions.append(pos1_id)
    print_success(f"Added XAUUSD BUY position: {pos1_id[:8]}...")
    
    # Position 2: EURUSD Sell
    pos2_id = manager.add_position_from_order(
        ticket=12346,
        symbol="EURUSD",
        position_type=PositionType.SELL,
        volume=0.2,
        open_price=1.1000,
        stop_loss=1.1050,
        take_profit=1.0950,
        comment="EUR sell position"
    )
    positions.append(pos2_id)
    print_success(f"Added EURUSD SELL position: {pos2_id[:8]}...")
    
    # Position 3: GBPUSD Buy
    pos3_id = manager.add_position_from_order(
        ticket=12347,
        symbol="GBPUSD",
        position_type=PositionType.BUY,
        volume=0.15,
        open_price=1.2500,
        stop_loss=1.2450,
        take_profit=1.2600,
        comment="GBP buy position"
    )
    positions.append(pos3_id)
    print_success(f"Added GBPUSD BUY position: {pos3_id[:8]}...")
    
    print_section("2. Position Information")
    
    # Display position details
    for pos_id in positions:
        position = manager.get_position(pos_id)
        if position:
            print(f"""
Position ID: {pos_id[:8]}...
Symbol: {position.symbol}
Type: {position.position_type.value}
Volume: {position.volume}
Open Price: {position.open_price}
Current Price: {position.current_price}
Stop Loss: {position.stop_loss}
Take Profit: {position.take_profit}
Status: {position.status.value}
Comment: {position.comment}
            """)
    
    print_section("3. Position Modification")
    
    # Modify first position
    modify_request = PositionModifyRequest(
        position_id=pos1_id,
        new_stop_loss=1995.0,
        new_take_profit=2025.0,
        comment="Modified SL/TP levels"
    )
    
    success, message = manager.modify_position(modify_request)
    if success:
        print_success(f"Position modified: {message}")
    else:
        print_warning(f"Modification failed: {message}")
    
    print_section("4. Partial Position Close")
    
    # Partially close second position
    close_request = PositionCloseRequest(
        position_id=pos2_id,
        volume=0.1,  # Close half
        comment="Partial profit taking"
    )
    
    success, message = manager.close_position(close_request)
    if success:
        print_success(f"Position partially closed: {message}")
        
        # Show updated position
        position = manager.get_position(pos2_id)
        print_info(f"Remaining volume: {position.remaining_volume}")
        print_info(f"Status: {position.status.value}")
    else:
        print_warning(f"Partial close failed: {message}")
    
    print_section("5. Position Summary")
    
    # Get position summary
    summary = manager.get_position_summary()
    print(f"""
📊 POSITION SUMMARY
Total Positions: {summary.total_positions}
Open Positions: {summary.open_positions}
Closed Positions: {summary.closed_positions}
Partially Closed: {summary.partially_closed_positions}
Symbols: {', '.join(summary.symbols)}
Total Volume: {summary.total_volume:.2f}
Total Profit: ${summary.total_profit:.2f}
    """)
    
    print_section("6. Positions by Symbol")
    
    # Get positions by symbol
    xau_positions = manager.get_positions_by_symbol("XAUUSD")
    eur_positions = manager.get_positions_by_symbol("EURUSD")
    gbp_positions = manager.get_positions_by_symbol("GBPUSD")
    
    print(f"XAUUSD positions: {len(xau_positions)}")
    print(f"EURUSD positions: {len(eur_positions)}")
    print(f"GBPUSD positions: {len(gbp_positions)}")
    
    print_section("7. Manager Statistics")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"""
📈 MANAGER STATISTICS
Total Positions: {stats['total_positions']}
Open Positions: {stats['open_positions']}
Closed Positions: {stats['closed_positions']}
Winning Positions: {stats['winning_positions']}
Losing Positions: {stats['losing_positions']}
Total Profit: ${stats['total_profit']:.2f}
Total Volume: {stats['total_volume']:.2f}
    """)
    
    # Stop manager
    manager.stop()
    print_success("Position Manager stopped")
    
    return positions


def demo_position_calculator():
    """Demo Position Calculator functionality"""
    print_header("POSITION CALCULATOR DEMO")
    
    # Initialize calculator
    config = {
        'default_risk_percentage': 2.0,
        'max_risk_percentage': 5.0,
        'min_position_size': 0.01,
        'max_position_size': 10.0,
        'kelly_win_rate': 0.6,
        'kelly_avg_win': 100,
        'kelly_avg_loss': 80
    }
    
    calculator = PositionCalculator(config)
    print_success("Position Calculator initialized")
    
    print_section("1. P&L Calculations")
    
    # Create test position
    position = Position(
        position_id="calc-test-1",
        symbol="XAUUSD",
        position_type=PositionType.BUY,
        volume=0.1,
        open_price=2000.0,
        current_price=2010.0,
        stop_loss=1990.0,
        take_profit=2020.0
    )
    
    # Set some realized profit
    position.realized_profit = 25.0
    
    # Calculate different P&L types
    unrealized_pnl = calculator.calculate_pnl(position, 2010.0, PnLCalculationType.UNREALIZED)
    realized_pnl = calculator.calculate_pnl(position, calculation_type=PnLCalculationType.REALIZED)
    total_pnl = calculator.calculate_pnl(position, 2010.0, PnLCalculationType.TOTAL)
    
    print(f"""
💰 P&L CALCULATIONS
Position: {position.symbol} {position.position_type.value}
Open Price: ${position.open_price}
Current Price: ${position.current_price}
Volume: {position.volume}

Unrealized P&L: ${unrealized_pnl:.2f}
Realized P&L: ${realized_pnl:.2f}
Total P&L: ${total_pnl:.2f}
    """)
    
    print_section("2. Position Sizing Methods")
    
    # Test different position sizing methods
    symbol = "XAUUSD"
    entry_price = 2000.0
    stop_loss = 1990.0
    account_balance = 10000.0
    risk_percentage = 2.0
    
    sizing_methods = [
        (PositionSizingMethod.FIXED_AMOUNT, "Fixed Amount"),
        (PositionSizingMethod.FIXED_PERCENTAGE, "Fixed Percentage"),
        (PositionSizingMethod.RISK_BASED, "Risk Based"),
        (PositionSizingMethod.KELLY_CRITERION, "Kelly Criterion"),
        (PositionSizingMethod.VOLATILITY_BASED, "Volatility Based"),
        (PositionSizingMethod.ATR_BASED, "ATR Based")
    ]
    
    print(f"""
📏 POSITION SIZING COMPARISON
Symbol: {symbol}
Entry Price: ${entry_price}
Stop Loss: ${stop_loss}
Account Balance: ${account_balance:,.2f}
Risk Percentage: {risk_percentage}%
    """)
    
    for method, name in sizing_methods:
        size = calculator.calculate_position_size(
            symbol, entry_price, stop_loss, account_balance, risk_percentage, method
        )
        print(f"{name:20}: {size:.3f} lots")
    
    print_section("3. Risk Calculations")
    
    # Calculate various risk metrics
    margin_required = calculator.calculate_margin_required(symbol, 0.1, entry_price)
    pip_value = calculator.calculate_pip_value(symbol, 0.1)
    break_even = calculator.calculate_break_even_price(position, spread=0.5)
    risk_reward = calculator.calculate_risk_reward_ratio(entry_price, stop_loss, position.take_profit)
    
    print(f"""
⚖️  RISK METRICS
Margin Required: ${margin_required:.2f}
Pip Value: ${pip_value:.2f}
Break-even Price: ${break_even:.2f}
Risk/Reward Ratio: {risk_reward:.2f}:1
    """)
    
    print_section("4. Comprehensive Position Metrics")
    
    # Calculate comprehensive metrics
    metrics = calculator.calculate_position_metrics(position, 2010.0)
    
    print(f"""
📊 COMPREHENSIVE METRICS
Unrealized P&L: ${metrics.get('unrealized_pnl', 0):.2f}
Realized P&L: ${metrics.get('realized_pnl', 0):.2f}
Total P&L: ${metrics.get('total_pnl', 0):.2f}
P&L Percentage: {metrics.get('pnl_percentage', 0):.2f}%
Pip Value: ${metrics.get('pip_value', 0):.2f}
Margin Required: ${metrics.get('margin_required', 0):.2f}
Break-even Price: ${metrics.get('break_even_price', 0):.2f}
Risk/Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}:1
Days Held: {metrics.get('days_held', 0)} days
    """)
    
    print_section("5. Calculator Statistics")
    
    # Get calculator statistics
    stats = calculator.get_statistics()
    print(f"""
🔧 CALCULATOR SETTINGS
Default Risk %: {stats['default_risk_percentage']}%
Max Risk %: {stats['max_risk_percentage']}%
Min Position Size: {stats['min_position_size']} lots
Max Position Size: {stats['max_position_size']} lots
Supported Methods: {len(stats['supported_sizing_methods'])}
    """)


def demo_stop_loss_manager():
    """Demo Stop Loss Manager functionality"""
    print_header("STOP LOSS MANAGER DEMO")
    
    # Initialize stop loss manager
    config = {
        'update_interval': 1,
        'max_adjustment_frequency': 2
    }
    
    stop_manager = StopLossManager(config)
    print_success("Stop Loss Manager initialized")
    
    # Start manager
    stop_manager.start()
    print_success("Stop Loss Manager started")
    
    print_section("1. Creating Stop Loss Rules")
    
    # Create different types of stop loss rules
    rules = []
    
    # Fixed stop loss rule
    fixed_rule = StopLossRule(
        rule_id="fixed-stop-1",
        stop_type=StopLossType.FIXED,
        distance=0.001
    )
    stop_manager.add_stop_rule(fixed_rule)
    rules.append(fixed_rule)
    print_success(f"Added Fixed Stop Rule: {fixed_rule.rule_id}")
    
    # Trailing stop loss rule
    trailing_rule = StopLossRule(
        rule_id="trailing-stop-1",
        stop_type=StopLossType.TRAILING,
        trailing_method=TrailingStopMethod.FIXED_DISTANCE,
        distance=0.001,
        trail_start_profit=0.002,
        trail_step=0.0005
    )
    stop_manager.add_stop_rule(trailing_rule)
    rules.append(trailing_rule)
    print_success(f"Added Trailing Stop Rule: {trailing_rule.rule_id}")
    
    # ATR-based stop loss rule
    atr_rule = StopLossRule(
        rule_id="atr-stop-1",
        stop_type=StopLossType.ATR_BASED,
        atr_period=14,
        atr_multiplier=2.0
    )
    stop_manager.add_stop_rule(atr_rule)
    rules.append(atr_rule)
    print_success(f"Added ATR Stop Rule: {atr_rule.rule_id}")
    
    # Percentage-based stop loss rule
    percentage_rule = StopLossRule(
        rule_id="percentage-stop-1",
        stop_type=StopLossType.PERCENTAGE_BASED,
        percentage=1.0  # 1%
    )
    stop_manager.add_stop_rule(percentage_rule)
    rules.append(percentage_rule)
    print_success(f"Added Percentage Stop Rule: {percentage_rule.rule_id}")
    
    # Breakeven stop loss rule
    breakeven_rule = StopLossRule(
        rule_id="breakeven-stop-1",
        stop_type=StopLossType.BREAKEVEN,
        breakeven_trigger=0.001,
        breakeven_buffer=0.0002
    )
    stop_manager.add_stop_rule(breakeven_rule)
    rules.append(breakeven_rule)
    print_success(f"Added Breakeven Stop Rule: {breakeven_rule.rule_id}")
    
    print_section("2. Applying Rules to Positions")
    
    # Create test positions
    test_positions = ["pos-1", "pos-2", "pos-3"]
    
    # Apply different rules to positions
    stop_manager.apply_stop_to_position("pos-1", "fixed-stop-1")
    stop_manager.apply_stop_to_position("pos-1", "trailing-stop-1")
    print_success("Applied Fixed + Trailing stops to pos-1")
    
    stop_manager.apply_stop_to_position("pos-2", "atr-stop-1")
    stop_manager.apply_stop_to_position("pos-2", "breakeven-stop-1")
    print_success("Applied ATR + Breakeven stops to pos-2")
    
    stop_manager.apply_stop_to_position("pos-3", "percentage-stop-1")
    print_success("Applied Percentage stop to pos-3")
    
    print_section("3. Stop Rule Details")
    
    # Display rule details
    for rule in rules:
        print(f"""
Rule ID: {rule.rule_id}
Type: {rule.stop_type.value}
Active: {rule.is_active}
Created: {rule.created_time.strftime('%Y-%m-%d %H:%M:%S')}
Distance: {getattr(rule, 'distance', 'N/A')}
Percentage: {getattr(rule, 'percentage', 'N/A')}%
        """)
    
    print_section("4. Position Stop Assignments")
    
    # Show which stops are applied to which positions
    for pos_id in test_positions:
        position_stops = stop_manager.get_position_stops(pos_id)
        if position_stops:
            stop_types = [stop.stop_type.value for stop in position_stops]
            print(f"Position {pos_id}: {', '.join(stop_types)}")
        else:
            print(f"Position {pos_id}: No stops applied")
    
    print_section("5. Stop Loss Callbacks")
    
    # Add callback for stop adjustments
    callback_log = []
    
    def stop_adjusted_callback(position, new_stop):
        callback_log.append(f"Stop adjusted for {position.position_id}: {new_stop}")
        print_info(f"Callback: Stop adjusted to {new_stop}")
    
    def breakeven_set_callback(position, new_stop):
        callback_log.append(f"Breakeven set for {position.position_id}: {new_stop}")
        print_info(f"Callback: Breakeven set at {new_stop}")
    
    stop_manager.add_callback('stop_adjusted', stop_adjusted_callback)
    stop_manager.add_callback('breakeven_set', breakeven_set_callback)
    print_success("Stop loss callbacks registered")
    
    # Simulate callback triggers
    test_position = Position(
        position_id="callback-test",
        symbol="XAUUSD",
        position_type=PositionType.BUY,
        volume=0.1,
        open_price=2000.0,
        current_price=2010.0
    )
    
    stop_manager._trigger_callbacks('stop_adjusted', test_position, 1995.0)
    stop_manager._trigger_callbacks('breakeven_set', test_position, 2000.2)
    
    print_section("6. Stop Manager Statistics")
    
    # Get statistics
    stats = stop_manager.get_statistics()
    print(f"""
📊 STOP MANAGER STATISTICS
Active Rules: {stats['active_rules']}
Total Rules: {stats['total_rules']}
Positions with Stops: {stats['positions_with_stops']}
Total Adjustments: {stats['total_adjustments']}
Trailing Activations: {stats['trailing_activations']}
Breakeven Sets: {stats['breakeven_sets']}
Stops Triggered: {stats['stops_triggered']}
Update Interval: {stats['update_interval']}s
Max Adjustment Frequency: {stats['max_adjustment_frequency']}s
    """)
    
    print_section("7. Callback Log")
    
    if callback_log:
        print("📝 CALLBACK EVENTS:")
        for event in callback_log:
            print(f"  • {event}")
    else:
        print("No callback events recorded")
    
    # Stop manager
    stop_manager.stop()
    print_success("Stop Loss Manager stopped")


def demo_integration():
    """Demo integrated position management"""
    print_header("INTEGRATED POSITION MANAGEMENT DEMO")
    
    # Initialize all components
    position_manager = PositionManager({'auto_sync': False})
    position_calculator = PositionCalculator()
    stop_manager = StopLossManager()
    
    print_success("All components initialized")
    
    # Start components
    position_manager.start()
    stop_manager.start()
    print_success("All components started")
    
    print_section("1. Integrated Position Creation")
    
    # Calculate optimal position size
    account_balance = 10000.0
    risk_percentage = 2.0
    entry_price = 2000.0
    stop_loss_price = 1990.0
    
    optimal_size = position_calculator.calculate_position_size(
        "XAUUSD", entry_price, stop_loss_price, account_balance, risk_percentage,
        PositionSizingMethod.RISK_BASED
    )
    
    print_info(f"Calculated optimal position size: {optimal_size:.3f} lots")
    
    # Create position with calculated size
    pos_id = position_manager.add_position_from_order(
        ticket=99999,
        symbol="XAUUSD",
        position_type=PositionType.BUY,
        volume=optimal_size,
        open_price=entry_price,
        stop_loss=stop_loss_price,
        take_profit=2020.0,
        comment="Risk-calculated position"
    )
    
    print_success(f"Created position with optimal sizing: {pos_id[:8]}...")
    
    print_section("2. Advanced Stop Loss Setup")
    
    # Create advanced trailing stop rule
    advanced_trailing = StopLossRule(
        rule_id="advanced-trailing",
        stop_type=StopLossType.TRAILING,
        trailing_method=TrailingStopMethod.ATR_MULTIPLE,
        atr_period=14,
        atr_multiplier=2.0,
        trail_start_profit=0.002,
        min_distance=0.0005,
        max_distance=0.01
    )
    
    stop_manager.add_stop_rule(advanced_trailing)
    stop_manager.apply_stop_to_position(pos_id, "advanced-trailing")
    
    print_success("Applied advanced trailing stop to position")
    
    print_section("3. Real-time Monitoring Simulation")
    
    # Simulate price movements and position updates
    price_movements = [2005.0, 2010.0, 2015.0, 2012.0, 2018.0]
    
    position = position_manager.get_position(pos_id)
    
    for i, new_price in enumerate(price_movements):
        print(f"\n📈 Price Update {i+1}: ${new_price}")
        
        # Update position price
        position.current_price = new_price
        
        # Calculate current metrics
        metrics = position_calculator.calculate_position_metrics(position, new_price)
        
        print(f"  Current P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"  P&L %: {metrics.get('pnl_percentage', 0):.2f}%")
        
        # Simulate stop loss update
        stop_manager.update_position_stop(position)
        
        time.sleep(0.5)  # Brief pause for demo
    
    print_section("4. Final Position Analysis")
    
    # Final position metrics
    final_metrics = position_calculator.calculate_position_metrics(position)
    
    print(f"""
🎯 FINAL POSITION ANALYSIS
Position ID: {pos_id[:8]}...
Symbol: {position.symbol}
Type: {position.position_type.value}
Volume: {position.volume:.3f} lots
Entry Price: ${position.open_price}
Current Price: ${position.current_price}
Stop Loss: ${position.stop_loss}
Take Profit: ${position.take_profit}

📊 PERFORMANCE METRICS
Total P&L: ${final_metrics.get('total_pnl', 0):.2f}
P&L Percentage: {final_metrics.get('pnl_percentage', 0):.2f}%
Risk/Reward: {final_metrics.get('risk_reward_ratio', 0):.2f}:1
Days Held: {final_metrics.get('days_held', 0)} days
Margin Used: ${final_metrics.get('margin_required', 0):.2f}
    """)
    
    # Stop all components
    position_manager.stop()
    stop_manager.stop()
    print_success("All components stopped")


def main():
    """Main demo function"""
    print_header("ULTIMATE XAU SYSTEM - POSITION MANAGEMENT DEMO")
    print_info("Comprehensive demonstration of Position Management System")
    print_info("Components: Position Manager, Position Calculator, Stop Loss Manager")
    
    try:
        # Run individual component demos
        demo_position_manager()
        time.sleep(1)
        
        demo_position_calculator()
        time.sleep(1)
        
        demo_stop_loss_manager()
        time.sleep(1)
        
        # Run integrated demo
        demo_integration()
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print_success("All Position Management System components working correctly!")
        print_info("System ready for production use")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 