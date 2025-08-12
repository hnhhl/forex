#!/usr/bin/env python3
"""
Kelly Criterion System for Ultimate XAU System V4.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base_system import BaseSystem

logger = logging.getLogger(__name__)


class KellyCriterionSystem(BaseSystem):
    """Kelly Criterion System for optimal position sizing"""
    
    def __init__(self, config):
        super().__init__(config, "KellyCriterionSystem")
        self.trade_history = []
        self.portfolio_value = 100000.0
        self.kelly_performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'kelly_fraction': 0.0,
            'position_size_usd': 0.0
        }
    
    def initialize(self) -> bool:
        try:
            self.is_active = True
            logger.info("✅ Kelly Criterion System initialized")
            return True
        except Exception as e:
            logger.error(f"Kelly Criterion System initialization error: {e}")
            return False
    
    def process(self, data: Any) -> Dict:
        try:
            if isinstance(data, pd.DataFrame):
                market_data = data
            elif isinstance(data, dict) and 'market_data' in data:
                market_data = data['market_data']
            else:
                return {'error': 'Invalid market data format'}
            
            # Calculate Kelly position size
            kelly_results = self._calculate_kelly_position_size(market_data)
            
            return {
                'kelly_calculation': kelly_results,
                'system_status': 'ACTIVE',
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.kelly_performance.copy(),
                'prediction': kelly_results.get('confidence', 0.5),
                'confidence': kelly_results.get('confidence', 0.5)
            }
        except Exception as e:
            logger.error(f"Kelly Criterion processing error: {e}")
            return {'error': str(e)}
    
    def _calculate_kelly_position_size(self, market_data: pd.DataFrame) -> Dict:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            # Extract parameters from trade history
            params = self._extract_kelly_parameters()
            
            # Kelly formula: f* = (bp - q) / b
            b = abs(params['average_win'] / params['average_loss']) if params['average_loss'] != 0 else 2.0
            p = params['win_rate']
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Use getattr to safely access config attributes
            kelly_max_fraction = getattr(self.config, 'kelly_max_fraction', 0.25)
            kelly_safety_factor = getattr(self.config, 'kelly_safety_factor', 0.5)
            kelly_min_fraction = getattr(self.config, 'kelly_min_fraction', 0.01)
            kelly_method = getattr(self.config, 'kelly_method', 'adaptive')
            
            kelly_fraction = max(min(kelly_fraction, kelly_max_fraction), 0.0)
            
            # Apply safety factor
            safe_kelly_fraction = kelly_fraction * kelly_safety_factor
            safe_kelly_fraction = max(min(safe_kelly_fraction, kelly_max_fraction), kelly_min_fraction)
            
            position_size_usd = self.portfolio_value * safe_kelly_fraction
            
            # Update performance tracking
            self.kelly_performance['kelly_fraction'] = safe_kelly_fraction
            self.kelly_performance['position_size_usd'] = position_size_usd
            
            return {
                'kelly_fraction': kelly_fraction,
                'safe_kelly_fraction': safe_kelly_fraction,
                'position_size_usd': position_size_usd,
                'confidence': 0.7 if params['total_trades'] > 30 else 0.5,
                'method': kelly_method,
                'parameters': params,
                'recommendation': 'BUY' if safe_kelly_fraction > 0.1 else 'HOLD'
            }
        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return {'error': str(e)}
    
    def _extract_kelly_parameters(self) -> Dict:
        """Extract Kelly parameters from trading history"""
        if len(self.trade_history) < 10:
            return {
                'win_rate': 0.6,
                'average_win': 0.02,
                'average_loss': -0.01,
                'profit_factor': 2.0,
                'total_trades': len(self.trade_history)
            }
        
        wins = [t for t in self.trade_history if t.get('profit_loss', 0) > 0]
        losses = [t for t in self.trade_history if t.get('profit_loss', 0) < 0]
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean([t['profit_loss'] for t in wins]) if wins else 0.02
        avg_loss = np.mean([t['profit_loss'] for t in losses]) if losses else -0.01
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 2.0
        
        return {
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.trade_history)
        }
    
    def add_trade_result(self, trade_result: Dict):
        """Add trade result to history"""
        try:
            self.trade_history.append(trade_result)
            
            # Use getattr to safely access config attributes
            lookback_period = getattr(self.config, 'kelly_lookback_period', 100)
            
            if len(self.trade_history) > lookback_period:
                self.trade_history = self.trade_history[-lookback_period:]
            
            if trade_result.get('profit_loss', 0) > 0:
                self.kelly_performance['winning_trades'] += 1
            
            self.kelly_performance['total_trades'] += 1
            
            logger.info(f"Trade result added to Kelly system: P&L={trade_result.get('profit_loss', 0)}")
        except Exception as e:
            logger.error(f"Error adding trade result: {e}")
    
    def cleanup(self) -> bool:
        try:
            self.trade_history.clear()
            return True
        except:
            return False
    
    def get_statistics(self) -> Dict:
        """Get Kelly system statistics"""
        try:
            params = self._extract_kelly_parameters()
            return {
                'system_name': 'KellyCriterionSystem',
                'total_trades': params['total_trades'],
                'win_rate': params['win_rate'],
                'profit_factor': params['profit_factor'],
                'kelly_fraction': self.kelly_performance.get('kelly_fraction', 0),
                'position_size_usd': self.kelly_performance.get('position_size_usd', 0),
                'is_active': self.is_active,
                'last_update': getattr(self, 'last_update', datetime.now()).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)} 