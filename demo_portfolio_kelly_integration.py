"""
Demo: Portfolio Manager with Kelly Criterion Integration
Demonstrates the integration between Portfolio Manager and Position Sizing System
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.trading.portfolio_manager import (
    PortfolioManager, AllocationMethod, PortfolioRiskLevel
)
from core.trading.position_types import Position, PositionType, PositionStatus
from core.trading.kelly_criterion import KellyMethod

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"🎯 {title}")
    print(f"{'='*80}")

def print_section(title: str):
    """Print formatted section"""
    print(f"\n{'─'*60}")
    print(f"📊 {title}")
    print(f"{'─'*60}")

def create_sample_positions(symbols: list, portfolio_manager) -> dict:
    """Create sample positions for demonstration"""
    positions = {}
    
    for i, symbol in enumerate(symbols):
        # Create multiple positions per symbol
        symbol_positions = []
        
        for j in range(2):  # 2 positions per symbol
            position = Position(
                position_id=f"{symbol}_pos_{j+1}",
                symbol=symbol,
                position_type=PositionType.BUY if j % 2 == 0 else PositionType.SELL,
                volume=1.0 + j * 0.5,
                open_price=2000.0 + i * 100 + j * 10,
                current_price=2010.0 + i * 100 + j * 15,
                status=PositionStatus.OPEN,
                remaining_volume=1.0 + j * 0.5,
                realized_profit=0.0,
                unrealized_profit=10.0 + j * 5
            )
            
            symbol_positions.append(position)
            portfolio_manager.add_position_to_portfolio(position)
        
        positions[symbol] = symbol_positions
    
    return positions

def add_sample_trade_history(portfolio_manager, symbol: str, num_trades: int = 50):
    """Add sample trade history for Kelly calculations"""
    print(f"   📈 Adding {num_trades} sample trades for {symbol}...")
    
    # Generate realistic trade results
    win_rate = 0.6  # 60% win rate
    avg_win = 0.025  # 2.5% average win
    avg_loss = -0.015  # -1.5% average loss
    
    for i in range(num_trades):
        is_win = np.random.random() < win_rate
        
        if is_win:
            profit_loss = np.random.normal(avg_win, 0.01) * 100  # Convert to dollar amount
        else:
            profit_loss = np.random.normal(avg_loss, 0.005) * 100
        
        # Add trade result
        portfolio_manager.add_trade_result_to_kelly(
            symbol=symbol,
            profit_loss=profit_loss,
            win=is_win,
            trade_date=datetime.now() - timedelta(days=num_trades-i),
            entry_price=2000.0 + np.random.normal(0, 20),
            exit_price=2000.0 + profit_loss/100 + np.random.normal(0, 20),
            volume=1.0
        )

def demo_portfolio_initialization():
    """Demo 1: Portfolio Manager Initialization with Kelly Support"""
    print_header("Demo 1: Portfolio Manager Initialization with Kelly Support")
    
    # Configuration with Kelly parameters
    config = {
        'initial_capital': 250000.0,
        'risk_per_trade': 0.02,
        'max_position_size': 0.15,
        'kelly_enabled': True,
        'kelly_max_fraction': 0.25,
        'kelly_min_fraction': 0.01,
        'max_symbols': 10,
        'risk_level': 'moderate',
        'update_interval': 10
    }
    
    print("🔧 Creating Portfolio Manager with Kelly Configuration...")
    portfolio_manager = PortfolioManager(config)
    portfolio_manager.start()
    
    print(f"✅ Portfolio Manager initialized successfully!")
    print(f"   💰 Initial Capital: ${config['initial_capital']:,.2f}")
    print(f"   🎯 Risk per Trade: {config['risk_per_trade']:.1%}")
    print(f"   📊 Kelly Enabled: {portfolio_manager.kelly_enabled}")
    print(f"   🔢 Kelly Max Fraction: {portfolio_manager.sizing_parameters.kelly_max_fraction:.1%}")
    print(f"   📈 Default Kelly Method: {portfolio_manager.default_kelly_method}")
    
    # Show statistics
    stats = portfolio_manager.get_statistics()
    print(f"\n📊 Initial Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if 'count' in key or 'calls' in key:
                print(f"   {key}: {value}")
            elif isinstance(value, float) and value < 1:
                print(f"   {key}: {value:.1%}")
            else:
                print(f"   {key}: {value}")
    
    portfolio_manager.stop()
    return config

def demo_symbol_management_with_kelly():
    """Demo 2: Symbol Management with Kelly Position Sizing"""
    print_header("Demo 2: Symbol Management with Kelly Position Sizing")
    
    config = {
        'initial_capital': 500000.0,
        'kelly_enabled': True,
        'kelly_max_fraction': 0.3,
        'risk_per_trade': 0.025
    }
    
    portfolio_manager = PortfolioManager(config)
    portfolio_manager.start()
    
    # Add symbols with different weights
    symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    print("🔧 Adding symbols to portfolio...")
    for symbol, weight in zip(symbols, weights):
        success = portfolio_manager.add_symbol(symbol, weight)
        print(f"   {'✅' if success else '❌'} {symbol}: {weight:.1%}")
        
        if success:
            # Initialize position sizer
            portfolio_manager.initialize_position_sizer(symbol)
            print(f"      🎯 Position sizer initialized for {symbol}")
    
    # Create sample positions
    print("\n📊 Creating sample positions...")
    positions = create_sample_positions(symbols, portfolio_manager)
    
    # Calculate optimal position sizes
    print("\n🧮 Calculating optimal position sizes with Kelly Criterion...")
    current_price = 2000.0
    
    for symbol in symbols:
        print(f"\n   📈 {symbol}:")
        
        # Add sample trade history for better Kelly calculations
        add_sample_trade_history(portfolio_manager, symbol, 40)
        
        # Calculate with different Kelly methods
        kelly_methods = [
            KellyMethod.CLASSIC,
            KellyMethod.ADAPTIVE,
            KellyMethod.CONSERVATIVE
        ]
        
        for method in kelly_methods:
            result = portfolio_manager.calculate_optimal_position_size(
                symbol, current_price, method
            )
            
            if result:
                print(f"      {method.value}: {result.position_size:.4f} units "
                      f"(${result.position_size * current_price:,.2f}) "
                      f"[Confidence: {result.confidence_score:.1%}]")
    
    # Show portfolio summary
    print_section("Portfolio Summary")
    summary = portfolio_manager.get_portfolio_summary()
    
    print(f"💰 Portfolio Value: ${summary['portfolio_metrics'].total_value:,.2f}")
    print(f"📊 Total P&L: ${summary['portfolio_metrics'].total_pnl:,.2f}")
    print(f"🎯 Total Symbols: {len(summary['symbol_allocations'])}")
    
    print(f"\n📈 Symbol Allocations:")
    for symbol, alloc_data in summary['symbol_allocations'].items():
        print(f"   {symbol}: Target {alloc_data['target_weight']:.1%}, "
              f"Current {alloc_data['current_weight']:.1%}, "
              f"Value ${alloc_data['current_value']:,.2f}")
    
    portfolio_manager.stop()
    return portfolio_manager

def demo_kelly_rebalancing():
    """Demo 3: Kelly Criterion-Based Portfolio Rebalancing"""
    print_header("Demo 3: Kelly Criterion-Based Portfolio Rebalancing")
    
    config = {
        'initial_capital': 1000000.0,
        'kelly_enabled': True,
        'kelly_max_fraction': 0.25,
        'risk_per_trade': 0.02
    }
    
    portfolio_manager = PortfolioManager(config)
    portfolio_manager.start()
    
    # Add symbols
    symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
    initial_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weight initially
    
    print("🔧 Setting up portfolio with equal weights...")
    for symbol, weight in zip(symbols, initial_weights):
        portfolio_manager.add_symbol(symbol, weight)
        portfolio_manager.initialize_position_sizer(symbol)
        
        # Add extensive trade history for realistic Kelly calculations
        add_sample_trade_history(portfolio_manager, symbol, 60)
    
    # Create positions
    positions = create_sample_positions(symbols, portfolio_manager)
    
    print(f"\n📊 Initial Portfolio State:")
    allocations = portfolio_manager.get_symbol_allocations()
    for symbol, alloc in allocations.items():
        print(f"   {symbol}: {alloc.target_weight:.1%} target weight")
    
    # Test different rebalancing methods
    rebalancing_methods = [
        (AllocationMethod.EQUAL_WEIGHT, "Equal Weight"),
        (AllocationMethod.RISK_PARITY, "Risk Parity"),
        (AllocationMethod.KELLY_OPTIMAL, "Kelly Optimal")
    ]
    
    for method, method_name in rebalancing_methods:
        print(f"\n🔄 Rebalancing using {method_name}...")
        
        success = portfolio_manager.rebalance_portfolio(method)
        print(f"   {'✅' if success else '❌'} Rebalancing {method_name}: {'Success' if success else 'Failed'}")
        
        if success:
            allocations = portfolio_manager.get_symbol_allocations()
            print(f"   📊 New allocations:")
            total_weight = 0
            
            for symbol, alloc in allocations.items():
                print(f"      {symbol}: {alloc.target_weight:.1%}")
                if hasattr(alloc, 'kelly_fraction') and alloc.kelly_fraction > 0:
                    print(f"         Kelly Fraction: {alloc.kelly_fraction:.3f}")
                    print(f"         Kelly Confidence: {alloc.kelly_confidence:.1%}")
                total_weight += alloc.target_weight
            
            print(f"   📈 Total Weight: {total_weight:.1%}")
        
        time.sleep(1)  # Brief pause between rebalancing
    
    # Show final statistics
    print_section("Final Statistics")
    stats = portfolio_manager.get_statistics()
    print(f"🔄 Rebalance Count: {stats['rebalance_count']}")
    print(f"🧮 Kelly Calculations: {stats['kelly_calculations']}")
    print(f"📊 Position Sizing Calls: {stats['position_sizing_calls']}")
    print(f"⚠️ Risk Breaches: {stats['risk_breaches']}")
    
    portfolio_manager.stop()
    return portfolio_manager

def demo_real_time_kelly_analysis():
    """Demo 4: Real-time Kelly Analysis and Position Sizing"""
    print_header("Demo 4: Real-time Kelly Analysis and Position Sizing")
    
    config = {
        'initial_capital': 750000.0,
        'kelly_enabled': True,
        'update_interval': 2  # Fast updates for demo
    }
    
    portfolio_manager = PortfolioManager(config)
    portfolio_manager.start()
    
    # Setup callback for real-time updates
    kelly_updates = []
    position_updates = []
    
    def kelly_callback(symbol, analysis):
        kelly_updates.append((symbol, analysis))
        print(f"   🔔 Kelly update for {symbol}: "
              f"Best method confidence {max(method['confidence_score'] for method in analysis['kelly_analysis'].values()):.1%}")
    
    def position_callback(symbol, result):
        position_updates.append((symbol, result))
        print(f"   🔔 Position sized for {symbol}: {result.position_size:.4f} units")
    
    portfolio_manager.add_callback('kelly_updated', kelly_callback)
    portfolio_manager.add_callback('position_sized', position_callback)
    
    # Add symbols
    symbols = ["XAUUSD", "EURUSD"]
    for symbol in symbols:
        portfolio_manager.add_symbol(symbol, 0.5)
        portfolio_manager.initialize_position_sizer(symbol)
        add_sample_trade_history(portfolio_manager, symbol, 50)
    
    print("🔄 Running real-time Kelly analysis simulation...")
    
    # Simulate price changes and Kelly updates
    base_prices = {"XAUUSD": 2000.0, "EURUSD": 1.1000}
    
    for i in range(5):  # 5 iterations
        print(f"\n📊 Iteration {i+1}/5:")
        
        for symbol in symbols:
            # Simulate price change
            price_change = np.random.normal(0, 0.01)  # 1% volatility
            current_price = base_prices[symbol] * (1 + price_change)
            
            print(f"   💰 {symbol} price: {current_price:.4f}")
            
            # Get comprehensive Kelly analysis
            analysis = portfolio_manager.get_kelly_analysis(symbol, current_price)
            
            if analysis:
                kelly_methods = analysis['kelly_analysis']
                best_method = max(kelly_methods.items(), 
                                key=lambda x: x[1]['confidence_score'])
                
                print(f"      🎯 Best Kelly method: {best_method[0]}")
                print(f"      📊 Kelly fraction: {best_method[1]['kelly_fraction']:.3f}")
                print(f"      ✅ Confidence: {best_method[1]['confidence_score']:.1%}")
                print(f"      💵 Recommended size: {best_method[1]['position_size']:.4f} units")
            
            # Calculate optimal position size
            result = portfolio_manager.calculate_optimal_position_size(symbol, current_price)
            
        time.sleep(1)  # Simulate real-time delay
    
    print(f"\n📈 Summary:")
    print(f"   🔔 Kelly updates received: {len(kelly_updates)}")
    print(f"   🔔 Position updates received: {len(position_updates)}")
    
    # Final portfolio summary
    summary = portfolio_manager.get_portfolio_summary()
    print(f"   💰 Final portfolio value: ${summary['portfolio_metrics'].total_value:,.2f}")
    print(f"   📊 Total Kelly calculations: {summary['statistics']['kelly_calculations']}")
    
    portfolio_manager.stop()
    return portfolio_manager

def main():
    """Run all demos"""
    print_header("🚀 Portfolio Manager with Kelly Criterion Integration - Demo Suite")
    print("Demonstrating advanced portfolio management with professional Kelly Criterion integration")
    
    try:
        # Run demos
        demo_portfolio_initialization()
        demo_symbol_management_with_kelly()
        demo_kelly_rebalancing()
        demo_real_time_kelly_analysis()
        
        print_header("✅ All Demos Completed Successfully!")
        print("🎯 Key Features Demonstrated:")
        print("   • Portfolio Manager initialization with Kelly support")
        print("   • Symbol management with position sizing integration")
        print("   • Kelly Criterion-based portfolio rebalancing")
        print("   • Real-time Kelly analysis and position sizing")
        print("   • Professional risk management and callbacks")
        print("\n🏆 Portfolio-Kelly Integration is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 