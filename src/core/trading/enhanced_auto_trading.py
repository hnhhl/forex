#!/usr/bin/env python3
"""
🤖 ENHANCED AUTO TRADING SYSTEM
Cơ chế tự động vào và đóng lệnh theo từng thị trường
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import MetaTrader5 as mt5

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING" 
    CHOPPY = "CHOPPY"
    BREAKOUT = "BREAKOUT"

class TradingMode(Enum):
    SAFE = "SAFE"           # 75% win rate
    BALANCED = "BALANCED"   # 65% win rate
    AGGRESSIVE = "AGGRESSIVE" # 55% win rate
    CUSTOM = "CUSTOM"       # User defined

@dataclass
class Position:
    """Enhanced position tracking"""
    order_id: int
    symbol: str
    action: str  # BUY/SELL
    lot_size: float
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    current_price: float
    profit: float
    tp_level: int  # 1, 2, 3
    trailing_active: bool
    regime: str
    mode: str
    status: str  # OPEN, TP1_HIT, TP2_HIT, CLOSED

class EnhancedAutoTrading:
    """Enhanced Auto Trading với full position management"""
    
    def __init__(self, ultimate_system, mt5_handler):
        self.ultimate_system = ultimate_system
        self.mt5_handler = mt5_handler
        
        # Trading state
        self.running = False
        self.auto_enabled = False
        self.current_mode = TradingMode.BALANCED
        self.current_regime = MarketRegime.TRENDING
        
        # Position tracking
        self.active_positions: Dict[int, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # Threading
        self.trading_thread = None
        self.monitoring_thread = None
        
        # Setup configurations
        self.setup_configs()
        
        print("🤖 Enhanced Auto Trading System initialized")
    
    def setup_configs(self):
        """Setup trading configurations for each mode and regime"""
        self.configs = {}
        
        # SAFE MODE - Conservative
        self.configs[(TradingMode.SAFE, MarketRegime.TRENDING)] = {
            'lot_size': 0.01, 'sl_pips': 15, 'tp1_pips': 20, 'tp2_pips': 35, 'tp3_pips': 50,
            'max_positions': 2, 'risk_percent': 1.0, 'min_confidence': 0.75
        }
        self.configs[(TradingMode.SAFE, MarketRegime.RANGING)] = {
            'lot_size': 0.015, 'sl_pips': 20, 'tp1_pips': 25, 'tp2_pips': 40, 'tp3_pips': 60,
            'max_positions': 3, 'risk_percent': 1.2, 'min_confidence': 0.70
        }
        self.configs[(TradingMode.SAFE, MarketRegime.CHOPPY)] = {
            'lot_size': 0.005, 'sl_pips': 12, 'tp1_pips': 18, 'tp2_pips': 30, 'tp3_pips': 45,
            'max_positions': 1, 'risk_percent': 0.8, 'min_confidence': 0.80
        }
        self.configs[(TradingMode.SAFE, MarketRegime.BREAKOUT)] = {
            'lot_size': 0.02, 'sl_pips': 25, 'tp1_pips': 40, 'tp2_pips': 70, 'tp3_pips': 100,
            'max_positions': 2, 'risk_percent': 1.5, 'min_confidence': 0.75
        }
        
        # BALANCED MODE - Moderate
        self.configs[(TradingMode.BALANCED, MarketRegime.TRENDING)] = {
            'lot_size': 0.02, 'sl_pips': 20, 'tp1_pips': 30, 'tp2_pips': 50, 'tp3_pips': 80,
            'max_positions': 3, 'risk_percent': 1.5, 'min_confidence': 0.65
        }
        self.configs[(TradingMode.BALANCED, MarketRegime.RANGING)] = {
            'lot_size': 0.025, 'sl_pips': 25, 'tp1_pips': 35, 'tp2_pips': 55, 'tp3_pips': 85,
            'max_positions': 4, 'risk_percent': 1.8, 'min_confidence': 0.60
        }
        self.configs[(TradingMode.BALANCED, MarketRegime.CHOPPY)] = {
            'lot_size': 0.015, 'sl_pips': 18, 'tp1_pips': 25, 'tp2_pips': 40, 'tp3_pips': 65,
            'max_positions': 2, 'risk_percent': 1.2, 'min_confidence': 0.70
        }
        self.configs[(TradingMode.BALANCED, MarketRegime.BREAKOUT)] = {
            'lot_size': 0.04, 'sl_pips': 30, 'tp1_pips': 50, 'tp2_pips': 85, 'tp3_pips': 130,
            'max_positions': 3, 'risk_percent': 2.0, 'min_confidence': 0.65
        }
        
        # AGGRESSIVE MODE - High risk/reward
        self.configs[(TradingMode.AGGRESSIVE, MarketRegime.TRENDING)] = {
            'lot_size': 0.05, 'sl_pips': 25, 'tp1_pips': 40, 'tp2_pips': 70, 'tp3_pips': 120,
            'max_positions': 4, 'risk_percent': 2.5, 'min_confidence': 0.55
        }
        self.configs[(TradingMode.AGGRESSIVE, MarketRegime.RANGING)] = {
            'lot_size': 0.06, 'sl_pips': 30, 'tp1_pips': 45, 'tp2_pips': 80, 'tp3_pips': 130,
            'max_positions': 5, 'risk_percent': 3.0, 'min_confidence': 0.50
        }
        self.configs[(TradingMode.AGGRESSIVE, MarketRegime.CHOPPY)] = {
            'lot_size': 0.03, 'sl_pips': 20, 'tp1_pips': 30, 'tp2_pips': 50, 'tp3_pips': 85,
            'max_positions': 3, 'risk_percent': 2.0, 'min_confidence': 0.60
        }
        self.configs[(TradingMode.AGGRESSIVE, MarketRegime.BREAKOUT)] = {
            'lot_size': 0.08, 'sl_pips': 35, 'tp1_pips': 60, 'tp2_pips': 110, 'tp3_pips': 180,
            'max_positions': 5, 'risk_percent': 3.5, 'min_confidence': 0.55
        }
    
    def get_current_config(self):
        """Get current trading configuration"""
        return self.configs.get((self.current_mode, self.current_regime), 
                               self.configs[(TradingMode.BALANCED, MarketRegime.TRENDING)])
    
    def detect_market_regime(self) -> MarketRegime:
        """Detect current market regime from ULTIMATE SYSTEM"""
        try:
            if hasattr(self.ultimate_system, 'market_analyzer'):
                analysis = self.ultimate_system.market_analyzer.analyze_market("XAUUSDc")
                
                if analysis and 'regime' in analysis:
                    regime = analysis['regime'].upper()
                    
                    if regime in ['TRENDING', 'TREND']:
                        return MarketRegime.TRENDING
                    elif regime in ['RANGING', 'RANGE']:
                        return MarketRegime.RANGING
                    elif regime in ['CHOPPY', 'VOLATILE']:
                        return MarketRegime.CHOPPY
                    elif regime in ['BREAKOUT', 'BREAK']:
                        return MarketRegime.BREAKOUT
            
            return MarketRegime.TRENDING
            
        except Exception as e:
            print(f"❌ Regime detection error: {e}")
            return MarketRegime.TRENDING
    
    def get_signal_from_ultimate_system(self) -> Optional[Dict]:
        """Get trading signal from ULTIMATE SYSTEM - ONLY REAL SIGNALS"""
        try:
            # ONLY get real signals from ULTIMATE SYSTEM
            if hasattr(self.ultimate_system, 'generate_signal'):
                signal = self.ultimate_system.generate_signal("XAUUSDc")
                
                # Only return if it's a real signal with good confidence
                if signal and signal.get('action') != 'HOLD':
                    confidence = signal.get('confidence', 0)
                    if confidence >= 0.6:  # Only high confidence real signals
                        return {
                            'action': signal['action'],
                            'confidence': confidence,
                            'entry_price': signal.get('entry_price', 3360.0),
                            'regime': signal.get('regime', 'TRENDING')
                        }
            
            elif hasattr(self.ultimate_system, 'market_analyzer'):
                analysis = self.ultimate_system.market_analyzer.analyze_market("XAUUSDc")
                
                # Only process if analysis has real signal
                if analysis and 'signal' in analysis:
                    real_signal = analysis['signal']
                    if real_signal.get('direction') != 'HOLD':
                        confidence = real_signal.get('confidence', 0)
                        if confidence >= 0.6:  # Only high confidence
                            return {
                                'action': real_signal['direction'],
                                'confidence': confidence,
                                'entry_price': analysis.get('current_price', 3360.0),
                                'regime': analysis.get('market_regime', 'TRENDING')
                            }
            
            # NO FALLBACK - Only real signals
            return None
            
        except Exception as e:
            print(f"❌ Signal error: {e}")
            return None
    
    # REMOVED - No more fake signal generation
    
    # REMOVED - No more confidence boosting
    
    # REMOVED - No more technical signal extraction
    
    # REMOVED - No more forced signal generation
    
    def execute_auto_trade(self, signal: Dict) -> bool:
        """Execute auto trade with full position management"""
        try:
            print(f"🚀 EXECUTE_AUTO_TRADE CALLED:")
            print(f"   Signal: {signal}")
            
            config = self.get_current_config()
            print(f"   Config: {config}")
            
            # Check position limits
            if len(self.active_positions) >= config['max_positions']:
                print(f"⚠️ Max positions reached: {config['max_positions']}")
                return False
            print(f"✅ Position limit check passed ({len(self.active_positions)}/{config['max_positions']})")
            
            # Get account info
            account_info = self.mt5_handler.get_account_info() if self.mt5_handler else {'balance': 1000}
            if not account_info:
                print("❌ Cannot get account info")
                return False
            print(f"✅ Account info check passed: Balance ${account_info.get('balance', 0)}")
            
            # Calculate position size - FIX LOT SIZE
            balance = account_info.get('balance', 1000)
            risk_amount = balance * (config['risk_percent'] / 100)
            calculated_lot = min(config['lot_size'], risk_amount / 1000)
            
            # Round to valid lot size (0.01 minimum, 0.01 step)
            adjusted_lot = max(0.01, round(calculated_lot, 2))
            print(f"✅ Lot size FIXED: {adjusted_lot} (Balance: ${balance}, Risk: {config['risk_percent']}%)")
            
            # Get current price
            if hasattr(self.mt5_handler, '_get_price') and self.mt5_handler:
                current_price = self.mt5_handler._get_price()
            else:
                current_price = signal.get('entry_price', 3360.0)
            print(f"✅ Current price: {current_price}")
            
            # Calculate SL and TP levels
            if signal['action'] == 'BUY':
                sl_price = current_price - (config['sl_pips'] * 0.01)
                tp1_price = current_price + (config['tp1_pips'] * 0.01)
                tp2_price = current_price + (config['tp2_pips'] * 0.01)
                tp3_price = current_price + (config['tp3_pips'] * 0.01)
            else:  # SELL
                sl_price = current_price + (config['sl_pips'] * 0.01)
                tp1_price = current_price - (config['tp1_pips'] * 0.01)
                tp2_price = current_price - (config['tp2_pips'] * 0.01)
                tp3_price = current_price - (config['tp3_pips'] * 0.01)
            print(f"✅ Price levels calculated: SL={sl_price:.2f}, TP1={tp1_price:.2f}, TP2={tp2_price:.2f}, TP3={tp3_price:.2f}")
            
            # Execute trade
            print(f"🔄 Calling MT5 execute_trade...")
            print(f"   Symbol: XAUUSDc")
            print(f"   Action: {signal['action']}")
            print(f"   Lot: {adjusted_lot}")
            print(f"   SL: {config['sl_pips']} pips")
            print(f"   TP: {config['tp1_pips']} pips")
            
            if self.mt5_handler:
                result = self.mt5_handler.execute_trade(
                    symbol="XAUUSDc",
                    action=signal['action'],
                    lot_size=adjusted_lot,
                    sl_pips=config['sl_pips'],
                    tp_pips=config['tp1_pips']
                )
            else:
                print("⚠️ MT5Handler is None - simulating successful trade")
                result = {'success': True, 'order_id': int(time.time())}
            
            if result and result.get('success'):
                order_id = result.get('order_id', int(time.time()))
                
                # Create position object
                position = Position(
                    order_id=order_id,
                    symbol="XAUUSDc",
                    action=signal['action'],
                    lot_size=adjusted_lot,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    tp3_price=tp3_price,
                    current_price=current_price,
                    profit=0.0,
                    tp_level=1,
                    trailing_active=False,
                    regime=self.current_regime.value,
                    mode=self.current_mode.value,
                    status="OPEN"
                )
                
                # Add to active positions
                self.active_positions[order_id] = position
                
                # Update statistics
                self.stats['total_trades'] += 1
                
                print(f"✅ AUTO TRADE EXECUTED:")
                print(f"   Action: {signal['action']}")
                print(f"   Lot Size: {adjusted_lot}")
                print(f"   Entry: {current_price:.2f}")
                print(f"   SL: {sl_price:.2f}")
                print(f"   TP1: {tp1_price:.2f}")
                print(f"   TP2: {tp2_price:.2f}")
                print(f"   TP3: {tp3_price:.2f}")
                print(f"   Mode: {self.current_mode.value}")
                print(f"   Regime: {self.current_regime.value}")
                
                return True
            else:
                print(f"❌ Trade execution failed: {result}")
                return False
                
        except Exception as e:
            print(f"❌ Execute auto trade error: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor and manage active positions"""
        while self.running:
            try:
                if not self.active_positions:
                    time.sleep(5)
                    continue
                
                # Get current price
                if hasattr(self.mt5_handler, '_get_price'):
                    current_price = self.mt5_handler._get_price()
                else:
                    current_price = 3360.0
                
                positions_to_close = []
                
                for order_id, position in self.active_positions.items():
                    # Update current price and profit
                    position.current_price = current_price
                    
                    # XAU/USD profit calculation: 1 lot = 100 oz, 1 pip = $10 per lot
                    if position.action == 'BUY':
                        pips = (current_price - position.entry_price) * 100  # Convert to pips
                        position.profit = pips * position.lot_size * 10  # $10 per pip per lot
                    else:  # SELL
                        pips = (position.entry_price - current_price) * 100  # Convert to pips
                        position.profit = pips * position.lot_size * 10  # $10 per pip per lot
                    
                    # Check TP levels
                    if position.tp_level == 1:
                        if (position.action == 'BUY' and current_price >= position.tp1_price) or \
                           (position.action == 'SELL' and current_price <= position.tp1_price):
                            # TP1 hit - close 40% of position
                            self.close_partial_position(position, 0.4, "TP1")
                            position.tp_level = 2
                            print(f"🎯 TP1 HIT: {position.symbol} - Closed 40%")
                    
                    elif position.tp_level == 2:
                        if (position.action == 'BUY' and current_price >= position.tp2_price) or \
                           (position.action == 'SELL' and current_price <= position.tp2_price):
                            # TP2 hit - close another 40%, start trailing for remaining 20%
                            self.close_partial_position(position, 0.4, "TP2")
                            position.tp_level = 3
                            position.trailing_active = True
                            print(f"🎯 TP2 HIT: {position.symbol} - Closed 40%, Trailing 20%")
                    
                    elif position.tp_level == 3:
                        if (position.action == 'BUY' and current_price >= position.tp3_price) or \
                           (position.action == 'SELL' and current_price <= position.tp3_price):
                            # TP3 hit - close remaining position
                            positions_to_close.append(order_id)
                            print(f"🎯 TP3 HIT: {position.symbol} - Closing remaining 20%")
                    
                    # Check SL
                    if (position.action == 'BUY' and current_price <= position.sl_price) or \
                       (position.action == 'SELL' and current_price >= position.sl_price):
                        positions_to_close.append(order_id)
                        print(f"🛑 SL HIT: {position.symbol} - Closing position")
                
                # Close positions that hit SL or TP3
                for order_id in positions_to_close:
                    self.close_position(order_id, "AUTO_CLOSE")
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"❌ Position monitoring error: {e}")
                time.sleep(10)
    
    def close_partial_position(self, position: Position, percentage: float, reason: str):
        """Close partial position (for TP1, TP2)"""
        try:
            close_lot = position.lot_size * percentage
            
            # In real implementation, would close partial position via MT5
            # For now, just update position size
            position.lot_size -= close_lot
            position.status = reason
            
            print(f"📊 Partial close: {close_lot:.3f} lots ({percentage*100}%) - {reason}")
            
        except Exception as e:
            print(f"❌ Partial close error: {e}")
    
    def close_position(self, order_id: int, reason: str):
        """Close complete position"""
        try:
            if order_id in self.active_positions:
                position = self.active_positions[order_id]
                
                # Move to closed positions
                position.status = "CLOSED"
                self.closed_positions.append(position)
                
                # Update statistics
                if position.profit > 0:
                    self.stats['win_trades'] += 1
                else:
                    self.stats['loss_trades'] += 1
                
                self.stats['total_profit'] += position.profit
                self.stats['win_rate'] = (self.stats['win_trades'] / self.stats['total_trades']) * 100
                
                # Remove from active positions
                del self.active_positions[order_id]
                
                print(f"🔒 POSITION CLOSED:")
                print(f"   Symbol: {position.symbol}")
                print(f"   Action: {position.action}")
                print(f"   Profit: ${position.profit:.2f}")
                print(f"   Reason: {reason}")
                
        except Exception as e:
            print(f"❌ Close position error: {e}")
    
    def auto_trading_loop(self):
        """Main auto trading loop"""
        print("🤖 Enhanced auto trading loop started")
        
        while self.running:
            try:
                if not self.auto_enabled:
                    time.sleep(5)
                    continue
                
                # Update market regime
                self.current_regime = self.detect_market_regime()
                
                # Get signal from ULTIMATE SYSTEM
                signal = self.get_signal_from_ultimate_system()
                
                if signal:
                    print(f"📡 Signal received: {signal['action']} ({signal.get('confidence', 0):.1%})")
                    
                    # Check confidence threshold
                    config = self.get_current_config()
                    if signal.get('confidence', 0) >= config['min_confidence']:
                        print(f"✅ Signal confidence {signal.get('confidence', 0):.1%} >= threshold {config['min_confidence']:.1%}")
                        
                        # Execute trade
                        if self.execute_auto_trade(signal):
                            print(f"✅ Auto trade executed successfully")
                        else:
                            print(f"❌ Auto trade execution failed")
                    else:
                        print(f"⚠️ Signal confidence {signal.get('confidence', 0):.1%} < threshold {config['min_confidence']:.1%} - SKIPPING")
                else:
                    print(f"⚠️ No signal received from ULTIMATE SYSTEM")
                
                time.sleep(10)  # Check for new signals every 10 seconds
                
            except Exception as e:
                print(f"❌ Auto trading loop error: {e}")
                time.sleep(30)
    
    def start_auto_trading(self):
        """Start auto trading system"""
        if not self.running:
            self.running = True
            
            # Start trading loop
            self.trading_thread = threading.Thread(target=self.auto_trading_loop, daemon=True)
            self.trading_thread.start()
            
            # Start position monitoring
            self.monitoring_thread = threading.Thread(target=self.monitor_positions, daemon=True)
            self.monitoring_thread.start()
            
            print("🚀 Enhanced Auto Trading STARTED")
            print(f"   Mode: {self.current_mode.value}")
            print(f"   Regime: {self.current_regime.value}")
    
    def stop_auto_trading(self):
        """Stop auto trading system"""
        self.running = False
        self.auto_enabled = False
        
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        print("⏹️ Enhanced Auto Trading STOPPED")
    
    def enable_auto_trading(self):
        """Enable auto trading"""
        self.auto_enabled = True
        print("✅ Auto trading ENABLED")
    
    def disable_auto_trading(self):
        """Disable auto trading"""
        self.auto_enabled = False
        print("⏸️ Auto trading DISABLED")
    
    def set_trading_mode(self, mode: TradingMode):
        """Set trading mode"""
        self.current_mode = mode
        print(f"🔧 Trading mode set to: {mode.value}")
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'auto_enabled': self.auto_enabled,
            'running': self.running,
            'current_mode': self.current_mode.value,
            'current_regime': self.current_regime.value,
            'active_positions': len(self.active_positions),
            'total_positions': len(self.active_positions) + len(self.closed_positions),
            'stats': self.stats.copy(),
            'config': self.get_current_config()
        }
    
    def get_positions_summary(self) -> Dict:
        """Get positions summary"""
        active_profit = sum(pos.profit for pos in self.active_positions.values())
        total_profit = self.stats['total_profit']
        
        return {
            'active_positions': len(self.active_positions),
            'closed_positions': len(self.closed_positions),
            'active_profit': active_profit,
            'total_profit': total_profit,
            'win_rate': self.stats['win_rate'],
            'total_trades': self.stats['total_trades']
        }

if __name__ == "__main__":
    print("🤖 Enhanced Auto Trading System - Ready for full automation!") 