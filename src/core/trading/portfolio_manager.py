"""
Portfolio Manager Module
Advanced portfolio management với multi-symbol position tracking và risk analysis
Enhanced with Professional Position Sizing System and Kelly Criterion
"""

import uuid
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import dataclass
from enum import Enum

from .position_types import Position, PositionType, PositionStatus, PositionSummary
from .position_manager import PositionManager
from .position_calculator import PositionCalculator

# Import Position Sizing System with Kelly Criterion
try:
    from ..risk.position_sizer import (
        PositionSizer, SizingMethod, SizingParameters, SizingResult, RiskLevel
    )
    from .kelly_criterion import KellyMethod, TradeResult
    POSITION_SIZING_AVAILABLE = True
except ImportError:
    POSITION_SIZING_AVAILABLE = False
    print("⚠️ Position Sizing System not available")

logger = logging.getLogger(__name__)


class PortfolioRiskLevel(Enum):
    """Portfolio risk levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


class AllocationMethod(Enum):
    """Portfolio allocation methods"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    MINIMUM_VARIANCE = "minimum_variance"
    KELLY_OPTIMAL = "kelly_optimal"  # New Kelly-based allocation


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_return: float = 0.0
    daily_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0


@dataclass
class SymbolAllocation:
    """Symbol allocation in portfolio"""
    symbol: str
    target_weight: float
    current_weight: float
    current_value: float
    positions: List[Position]
    pnl: float
    risk_contribution: float
    correlation_avg: float
    # Enhanced with Kelly metrics
    kelly_fraction: float = 0.0
    kelly_confidence: float = 0.0
    recommended_size: float = 0.0


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics"""
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    portfolio_volatility: float = 0.0
    concentration_risk: float = 0.0
    correlation_risk: float = 0.0
    leverage_ratio: float = 0.0
    exposure_ratio: float = 0.0
    risk_budget_utilization: float = 0.0


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


class PortfolioManager(BaseSystem):
    """Advanced Portfolio Management System with Kelly Criterion Integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("PortfolioManager")
        
        # Configuration
        self.config = config or {}
        self.update_interval = self.config.get('update_interval', 5)  # seconds
        self.max_symbols = self.config.get('max_symbols', 20)
        self.risk_level = PortfolioRiskLevel(self.config.get('risk_level', 'moderate'))
        self.base_currency = self.config.get('base_currency', 'USD')
        
        # Portfolio data
        self.symbols: Dict[str, SymbolAllocation] = {}
        self.positions: Dict[str, List[Position]] = {}  # symbol -> positions
        self.portfolio_history: List[Dict] = []
        self.benchmark_data: Dict[str, float] = {}
        
        # Risk management
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)  # 5%
        self.max_symbol_weight = self.config.get('max_symbol_weight', 0.3)  # 30%
        self.min_symbol_weight = self.config.get('min_symbol_weight', 0.01)  # 1%
        self.max_correlation = self.config.get('max_correlation', 0.8)  # 80%
        
        # Components
        self.position_manager = None
        self.position_calculator = None
        
        # Position Sizing Integration
        self.position_sizers: Dict[str, PositionSizer] = {}  # symbol -> sizer
        self.kelly_enabled = self.config.get('kelly_enabled', True)
        self.default_kelly_method = KellyMethod.ADAPTIVE if POSITION_SIZING_AVAILABLE else None
        
        # Position sizing parameters
        self.sizing_parameters = SizingParameters(
            risk_per_trade=self.config.get('risk_per_trade', 0.02),
            max_position_size=self.config.get('max_position_size', 0.1),
            min_position_size=self.config.get('min_position_size', 0.01),
            kelly_max_fraction=self.config.get('kelly_max_fraction', 0.25),
            kelly_min_fraction=self.config.get('kelly_min_fraction', 0.01)
        ) if POSITION_SIZING_AVAILABLE else None
        
        # Event callbacks
        self.portfolio_callbacks: Dict[str, List[Callable]] = {
            'portfolio_rebalanced': [],
            'risk_limit_breached': [],
            'allocation_changed': [],
            'performance_updated': [],
            'kelly_updated': [],  # New Kelly callback
            'position_sized': []   # New position sizing callback
        }
        
        # Performance tracking
        self.performance_history: List[PortfolioMetrics] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Statistics
        self.stats = {
            'total_symbols': 0,
            'total_positions': 0,
            'total_value': 0.0,
            'total_pnl': 0.0,
            'rebalance_count': 0,
            'risk_breaches': 0,
            'last_rebalance': None,
            'kelly_calculations': 0,  # New Kelly stats
            'position_sizing_calls': 0
        }
        
        if POSITION_SIZING_AVAILABLE:
            logger.info("✅ PortfolioManager initialized with Position Sizing & Kelly Criterion")
        else:
            logger.warning("⚠️ PortfolioManager initialized without Position Sizing")
    
    def start(self):
        """Start portfolio manager"""
        super().start()
        
        # Initialize components
        if not self.position_manager:
            self.position_manager = PositionManager({'auto_sync': False})
            self.position_manager.start()
        
        if not self.position_calculator:
            self.position_calculator = PositionCalculator()
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_portfolio, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("PortfolioManager started")
    
    def initialize_position_sizer(self, symbol: str, price_data: pd.DataFrame = None) -> bool:
        """Initialize position sizer for a symbol"""
        try:
            if not POSITION_SIZING_AVAILABLE:
                logger.warning(f"Position sizing not available for {symbol}")
                return False
            
            # Create position sizer for symbol
            sizer = PositionSizer()
            
            # Set portfolio value (current portfolio value)
            portfolio_value = self.get_portfolio_value()
            if portfolio_value <= 0:
                portfolio_value = self.config.get('initial_capital', 100000.0)
            
            # Set price data if provided
            if price_data is not None:
                sizer.set_data(price_data, portfolio_value)
            
            # Set performance metrics from symbol history
            win_rate, avg_win, avg_loss = self._get_symbol_performance_metrics(symbol)
            sizer.set_performance_metrics(win_rate, avg_win, avg_loss)
            
            # Store sizer
            self.position_sizers[symbol] = sizer
            
            logger.info(f"✅ Position sizer initialized for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing position sizer for {symbol}: {e}")
            return False
    
    def calculate_optimal_position_size(self, symbol: str, current_price: float, 
                                      kelly_method: KellyMethod = None) -> Optional[SizingResult]:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            if not POSITION_SIZING_AVAILABLE:
                return None
            
            # Get or create position sizer
            if symbol not in self.position_sizers:
                if not self.initialize_position_sizer(symbol):
                    return None
            
            sizer = self.position_sizers[symbol]
            kelly_method = kelly_method or self.default_kelly_method
            
            # Calculate position size
            if kelly_method:
                result = sizer.calculate_kelly_criterion_size(
                    current_price, self.sizing_parameters, kelly_method
                )
            else:
                result = sizer.calculate_optimal_size(current_price, parameters=self.sizing_parameters)
            
            # Update statistics
            self.stats['kelly_calculations'] += 1
            self.stats['position_sizing_calls'] += 1
            
            # Update symbol allocation with Kelly metrics
            if symbol in self.symbols:
                allocation = self.symbols[symbol]
                allocation.kelly_fraction = result.additional_metrics.get('kelly_fraction', 0)
                allocation.kelly_confidence = result.confidence_score
                allocation.recommended_size = result.position_size
            
            # Trigger callbacks
            self._trigger_callbacks('position_sized', symbol, result)
            
            logger.debug(f"Position size calculated for {symbol}: {result.position_size:.4f} units")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return None
    
    def get_kelly_analysis(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Get comprehensive Kelly analysis for a symbol"""
        try:
            if not POSITION_SIZING_AVAILABLE or symbol not in self.position_sizers:
                return None
            
            sizer = self.position_sizers[symbol]
            analysis = sizer.get_kelly_analysis(current_price)
            
            # Trigger Kelly callback
            self._trigger_callbacks('kelly_updated', symbol, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting Kelly analysis for {symbol}: {e}")
            return None
    
    def add_trade_result_to_kelly(self, symbol: str, profit_loss: float, win: bool, 
                                 trade_date: datetime = None, entry_price: float = 0.0,
                                 exit_price: float = 0.0, volume: float = 0.0) -> bool:
        """Add trade result to Kelly Calculator for a symbol"""
        try:
            if not POSITION_SIZING_AVAILABLE:
                return False
            
            # Get or create position sizer
            if symbol not in self.position_sizers:
                if not self.initialize_position_sizer(symbol):
                    return False
            
            sizer = self.position_sizers[symbol]
            
            # Add trade result
            sizer.add_trade_result(
                profit_loss=profit_loss,
                win=win,
                trade_date=trade_date or datetime.now(),
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                volume=volume,
                duration_minutes=60  # Default 1 hour
            )
            
            logger.debug(f"Trade result added to Kelly for {symbol}: P&L={profit_loss:.4f}, Win={win}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding trade result to Kelly for {symbol}: {e}")
            return False
    
    def _get_symbol_performance_metrics(self, symbol: str) -> Tuple[float, float, float]:
        """Get performance metrics for a symbol from position history"""
        try:
            if symbol not in self.positions:
                # Default metrics for new symbols
                return 0.6, 0.02, -0.015  # 60% WR, 2% avg win, -1.5% avg loss
            
            positions = self.positions[symbol]
            if not positions:
                return 0.6, 0.02, -0.015
            
            # Calculate from closed positions
            closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
            if len(closed_positions) < 5:  # Need minimum trades
                return 0.6, 0.02, -0.015
            
            wins = [p for p in closed_positions if p.realized_pnl > 0]
            losses = [p for p in closed_positions if p.realized_pnl <= 0]
            
            win_rate = len(wins) / len(closed_positions)
            avg_win = np.mean([p.realized_pnl for p in wins]) if wins else 0.02
            avg_loss = np.mean([p.realized_pnl for p in losses]) if losses else -0.015
            
            # Convert to percentage
            avg_win = avg_win / 100.0 if avg_win > 1 else avg_win
            avg_loss = avg_loss / 100.0 if avg_loss < -1 else avg_loss
            
            return win_rate, avg_win, avg_loss
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics for {symbol}: {e}")
            return 0.6, 0.02, -0.015
    
    def stop(self):
        """Stop portfolio manager"""
        super().stop()
        
        # Stop monitoring
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Stop components
        if self.position_manager:
            self.position_manager.stop()
        
        logger.info("PortfolioManager stopped")
    
    def add_symbol(self, symbol: str, target_weight: float, 
                   max_positions: int = 5) -> bool:
        """Add symbol to portfolio"""
        try:
            with self.lock:
                if len(self.symbols) >= self.max_symbols:
                    logger.error(f"Maximum symbols ({self.max_symbols}) reached")
                    return False
                
                if symbol in self.symbols:
                    logger.warning(f"Symbol {symbol} already in portfolio")
                    return False
                
                # Validate weight (allow flexibility for testing)
                if target_weight < 0 or target_weight > 1.0:
                    logger.error(f"Invalid weight {target_weight} for {symbol} (must be 0-1.0)")
                    return False
                
                # Check total weight
                total_weight = sum(s.target_weight for s in self.symbols.values()) + target_weight
                if total_weight > 1.0:
                    logger.error(f"Total weight would exceed 100%: {total_weight}")
                    return False
                
                # Create symbol allocation
                allocation = SymbolAllocation(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=0.0,
                    current_value=0.0,
                    positions=[],
                    pnl=0.0,
                    risk_contribution=0.0,
                    correlation_avg=0.0
                )
                
                self.symbols[symbol] = allocation
                self.positions[symbol] = []
                
                # Update statistics
                self.stats['total_symbols'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('allocation_changed', symbol, allocation)
                
                logger.info(f"Symbol {symbol} added with weight {target_weight}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False
    
    def remove_symbol(self, symbol: str, close_positions: bool = True) -> bool:
        """Remove symbol from portfolio"""
        try:
            with self.lock:
                if symbol not in self.symbols:
                    logger.error(f"Symbol {symbol} not in portfolio")
                    return False
                
                # Close positions if requested
                if close_positions and self.positions[symbol]:
                    for position in self.positions[symbol]:
                        if position.status == PositionStatus.OPEN:
                            # Close position logic would go here
                            logger.info(f"Closing position {position.position_id}")
                
                # Remove symbol
                del self.symbols[symbol]
                del self.positions[symbol]
                
                # Update statistics
                self.stats['total_symbols'] -= 1
                
                logger.info(f"Symbol {symbol} removed from portfolio")
                return True
                
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False
    
    def add_position_to_portfolio(self, position: Position) -> bool:
        """Add position to portfolio tracking"""
        try:
            with self.lock:
                symbol = position.symbol
                
                # Add symbol if not exists
                if symbol not in self.symbols:
                    # Auto-add with equal weight
                    remaining_weight = 1.0 - sum(s.target_weight for s in self.symbols.values())
                    auto_weight = min(remaining_weight, 0.1)  # Max 10% auto weight
                    
                    if auto_weight >= self.min_symbol_weight:
                        self.add_symbol(symbol, auto_weight)
                    else:
                        logger.error(f"Cannot auto-add {symbol}: insufficient weight available")
                        return False
                
                # Add position to symbol
                self.positions[symbol].append(position)
                self.symbols[symbol].positions.append(position)
                
                # Update statistics
                self.stats['total_positions'] += 1
                
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                logger.info(f"Position {position.position_id} added to portfolio")
                return True
                
        except Exception as e:
            logger.error(f"Error adding position to portfolio: {e}")
            return False
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        try:
            with self.lock:
                total_value = 0.0
                
                for symbol, positions in self.positions.items():
                    for position in positions:
                        if position.status == PositionStatus.OPEN:
                            # Calculate position value
                            position_value = position.remaining_volume * position.current_price
                            total_value += position_value
                
                return total_value
                
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def get_portfolio_pnl(self) -> Tuple[float, float, float]:
        """Get portfolio P&L (total, realized, unrealized)"""
        try:
            with self.lock:
                total_realized = 0.0
                total_unrealized = 0.0
                
                for symbol, positions in self.positions.items():
                    for position in positions:
                        total_realized += position.realized_profit
                        if position.status == PositionStatus.OPEN:
                            # Calculate unrealized P&L
                            if self.position_calculator:
                                unrealized = self.position_calculator.calculate_pnl(
                                    position, position.current_price
                                )
                                total_unrealized += unrealized
                
                total_pnl = total_realized + total_unrealized
                return total_pnl, total_realized, total_unrealized
                
        except Exception as e:
            logger.error(f"Error calculating portfolio P&L: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            with self.lock:
                metrics = PortfolioMetrics()
                
                # Basic metrics
                metrics.total_value = self.get_portfolio_value()
                total_pnl, realized_pnl, unrealized_pnl = self.get_portfolio_pnl()
                metrics.total_pnl = total_pnl
                metrics.realized_pnl = realized_pnl
                metrics.unrealized_pnl = unrealized_pnl
                
                # Return calculations
                if metrics.total_value > 0:
                    metrics.total_return = (metrics.total_pnl / metrics.total_value) * 100
                
                # Daily return
                if len(self.daily_returns) > 0:
                    metrics.daily_return = self.daily_returns[-1]
                
                # Volatility (annualized)
                if len(self.daily_returns) > 1:
                    returns_array = np.array(self.daily_returns)
                    metrics.volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
                
                # Sharpe ratio (assuming risk-free rate = 2%)
                risk_free_rate = 0.02
                if metrics.volatility > 0:
                    excess_return = (metrics.total_return / 100) - risk_free_rate
                    metrics.sharpe_ratio = excess_return / metrics.volatility
                
                # Maximum drawdown
                metrics.max_drawdown = self._calculate_max_drawdown()
                
                # VaR calculations
                if len(self.daily_returns) > 30:
                    returns_array = np.array(self.daily_returns)
                    metrics.var_95 = np.percentile(returns_array, 5)
                    metrics.var_99 = np.percentile(returns_array, 1)
                
                # Beta and Alpha (vs benchmark)
                if len(self.daily_returns) > 30 and len(self.benchmark_returns) > 30:
                    portfolio_returns = np.array(self.daily_returns[-30:])
                    benchmark_returns = np.array(self.benchmark_returns[-30:])
                    
                    if len(portfolio_returns) == len(benchmark_returns):
                        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
                        benchmark_variance = np.var(benchmark_returns)
                        
                        if benchmark_variance > 0:
                            metrics.beta = covariance / benchmark_variance
                            
                            # Alpha calculation
                            portfolio_mean = np.mean(portfolio_returns)
                            benchmark_mean = np.mean(benchmark_returns)
                            metrics.alpha = portfolio_mean - (metrics.beta * benchmark_mean)
                
                # Sortino ratio (downside deviation)
                if len(self.daily_returns) > 1:
                    returns_array = np.array(self.daily_returns)
                    negative_returns = returns_array[returns_array < 0]
                    if len(negative_returns) > 0:
                        downside_deviation = np.std(negative_returns) * np.sqrt(252)
                        if downside_deviation > 0:
                            excess_return = (metrics.total_return / 100) - risk_free_rate
                            metrics.sortino_ratio = excess_return / downside_deviation
                
                # Calmar ratio
                if metrics.max_drawdown != 0:
                    metrics.calmar_ratio = (metrics.total_return / 100) / abs(metrics.max_drawdown)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics()
    
    def calculate_portfolio_risk(self) -> PortfolioRiskMetrics:
        """Calculate portfolio risk metrics"""
        try:
            with self.lock:
                risk_metrics = PortfolioRiskMetrics()
                
                # Portfolio volatility
                if len(self.daily_returns) > 1:
                    returns_array = np.array(self.daily_returns)
                    risk_metrics.portfolio_volatility = np.std(returns_array) * np.sqrt(252)
                
                # Portfolio VaR and CVaR
                if len(self.daily_returns) > 30:
                    returns_array = np.array(self.daily_returns)
                    risk_metrics.portfolio_var = np.percentile(returns_array, 5)
                    
                    # CVaR (Expected Shortfall)
                    var_threshold = risk_metrics.portfolio_var
                    tail_returns = returns_array[returns_array <= var_threshold]
                    if len(tail_returns) > 0:
                        risk_metrics.portfolio_cvar = np.mean(tail_returns)
                
                # Concentration risk (Herfindahl index)
                total_value = self.get_portfolio_value()
                if total_value > 0:
                    weights_squared = 0.0
                    for symbol, allocation in self.symbols.items():
                        weight = allocation.current_value / total_value
                        weights_squared += weight ** 2
                    risk_metrics.concentration_risk = weights_squared
                
                # Correlation risk (average correlation)
                correlations = []
                symbols = list(self.symbols.keys())
                for i in range(len(symbols)):
                    for j in range(i + 1, len(symbols)):
                        # Mock correlation calculation
                        correlation = self._calculate_symbol_correlation(symbols[i], symbols[j])
                        correlations.append(abs(correlation))
                
                if correlations:
                    risk_metrics.correlation_risk = np.mean(correlations)
                
                # Leverage ratio
                total_exposure = sum(
                    sum(pos.remaining_volume * pos.current_price for pos in positions)
                    for positions in self.positions.values()
                )
                if total_value > 0:
                    risk_metrics.leverage_ratio = total_exposure / total_value
                
                # Exposure ratio
                risk_metrics.exposure_ratio = min(risk_metrics.leverage_ratio, 1.0)
                
                # Risk budget utilization
                current_risk = risk_metrics.portfolio_volatility
                risk_metrics.risk_budget_utilization = current_risk / self.max_portfolio_risk
                
                return risk_metrics
                
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return PortfolioRiskMetrics()
    
    def rebalance_portfolio(self, method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT) -> bool:
        """Rebalance portfolio to target allocations"""
        try:
            with self.lock:
                logger.info(f"Starting portfolio rebalance using {method.value}")
                
                # Calculate current allocations
                total_value = self.get_portfolio_value()
                if total_value <= 0:
                    logger.warning("Cannot rebalance: portfolio value is zero")
                    return False
                
                # Update current weights
                for symbol, allocation in self.symbols.items():
                    symbol_value = sum(
                        pos.remaining_volume * pos.current_price 
                        for pos in self.positions[symbol] 
                        if pos.status == PositionStatus.OPEN
                    )
                    allocation.current_value = symbol_value
                    allocation.current_weight = symbol_value / total_value if total_value > 0 else 0.0
                
                # Calculate target allocations based on method
                if method == AllocationMethod.EQUAL_WEIGHT:
                    self._rebalance_equal_weight()
                elif method == AllocationMethod.RISK_PARITY:
                    self._rebalance_risk_parity()
                elif method == AllocationMethod.MINIMUM_VARIANCE:
                    self._rebalance_minimum_variance()
                elif method == AllocationMethod.KELLY_OPTIMAL:
                    self._rebalance_kelly_optimal()
                else:
                    logger.warning(f"Rebalance method {method.value} not implemented")
                    return False
                
                # Update statistics
                self.stats['rebalance_count'] += 1
                self.stats['last_rebalance'] = datetime.now()
                
                # Trigger callbacks
                self._trigger_callbacks('portfolio_rebalanced', method)
                
                logger.info("Portfolio rebalance completed")
                return True
                
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return False
    
    def _rebalance_equal_weight(self):
        """Rebalance to equal weights"""
        num_symbols = len(self.symbols)
        if num_symbols == 0:
            return
        
        target_weight = 1.0 / num_symbols
        for allocation in self.symbols.values():
            allocation.target_weight = target_weight
    
    def _rebalance_risk_parity(self):
        """Rebalance using risk parity"""
        # Simplified risk parity - inverse volatility weighting
        volatilities = {}
        
        for symbol in self.symbols.keys():
            # Calculate symbol volatility (mock implementation)
            volatility = self._calculate_symbol_volatility(symbol)
            volatilities[symbol] = max(volatility, 0.01)  # Minimum volatility
        
        # Inverse volatility weights
        inv_vol_sum = sum(1.0 / vol for vol in volatilities.values())
        
        for symbol, allocation in self.symbols.items():
            weight = (1.0 / volatilities[symbol]) / inv_vol_sum
            allocation.target_weight = weight
    
    def _rebalance_minimum_variance(self):
        """Rebalance using minimum variance optimization"""
        # Simplified minimum variance - equal weight for now
        # In production, this would use covariance matrix optimization
        self._rebalance_equal_weight()
    
    def _rebalance_kelly_optimal(self):
        """Rebalance using Kelly Criterion optimal allocation"""
        try:
            if not POSITION_SIZING_AVAILABLE:
                logger.warning("Kelly rebalancing not available - falling back to equal weight")
                self._rebalance_equal_weight()
                return
            
            total_kelly_fraction = 0.0
            kelly_fractions = {}
            
            # Calculate Kelly fractions for each symbol
            for symbol in self.symbols:
                if symbol not in self.position_sizers:
                    self.initialize_position_sizer(symbol)
                
                sizer = self.position_sizers[symbol]
                
                # Get current price (simplified - would get from market data)
                current_price = 2000.0  # Default XAU price
                
                try:
                    result = sizer.calculate_kelly_adaptive_size(current_price)
                    kelly_fraction = result.additional_metrics.get('kelly_fraction', 0)
                    kelly_fractions[symbol] = max(0, kelly_fraction)  # Ensure non-negative
                    total_kelly_fraction += kelly_fractions[symbol]
                except Exception as e:
                    logger.warning(f"Kelly calculation failed for {symbol}: {e}")
                    kelly_fractions[symbol] = 0.0
            
            # Normalize Kelly fractions to portfolio weights
            if total_kelly_fraction > 0:
                for symbol in self.symbols:
                    normalized_weight = kelly_fractions[symbol] / total_kelly_fraction
                    
                    # Apply portfolio constraints
                    normalized_weight = max(self.min_symbol_weight, 
                                          min(self.max_symbol_weight, normalized_weight))
                    
                    self.symbols[symbol].target_weight = normalized_weight
                    
                    logger.info(f"Kelly rebalance - {symbol}: {normalized_weight:.2%} "
                              f"(Kelly: {kelly_fractions[symbol]:.3f})")
            else:
                # Fallback to equal weight if no valid Kelly fractions
                logger.warning("No valid Kelly fractions - falling back to equal weight")
                self._rebalance_equal_weight()
            
        except Exception as e:
            logger.error(f"Error in Kelly rebalancing: {e}")
            self._rebalance_equal_weight()
    
    def _calculate_symbol_volatility(self, symbol: str) -> float:
        """Calculate symbol volatility (mock implementation)"""
        # In production, this would calculate from historical price data
        volatility_map = {
            'XAUUSD': 0.15,
            'EURUSD': 0.08,
            'GBPUSD': 0.12,
            'USDJPY': 0.10,
            'AUDUSD': 0.14
        }
        return volatility_map.get(symbol, 0.12)
    
    def _calculate_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between symbols (mock implementation)"""
        # In production, this would calculate from historical price data
        if symbol1 == symbol2:
            return 1.0
        
        # Mock correlations
        correlation_map = {
            ('XAUUSD', 'EURUSD'): 0.3,
            ('XAUUSD', 'GBPUSD'): 0.25,
            ('EURUSD', 'GBPUSD'): 0.7,
            ('USDJPY', 'EURUSD'): -0.4
        }
        
        key = tuple(sorted([symbol1, symbol2]))
        return correlation_map.get(key, 0.1)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.performance_history) < 2:
            return 0.0
        
        values = [metrics.total_value for metrics in self.performance_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _update_portfolio_metrics(self):
        """Update portfolio metrics and history"""
        try:
            # Calculate current metrics
            metrics = self.calculate_portfolio_metrics()
            
            # Add to history
            self.performance_history.append(metrics)
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Update daily returns
            if len(self.performance_history) > 1:
                prev_value = self.performance_history[-2].total_value
                curr_value = metrics.total_value
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    self.daily_returns.append(daily_return)
                    
                    # Keep only last 252 days (1 year)
                    if len(self.daily_returns) > 252:
                        self.daily_returns = self.daily_returns[-252:]
            
            # Update statistics
            self.stats['total_value'] = metrics.total_value
            self.stats['total_pnl'] = metrics.total_pnl
            
            # Trigger callbacks
            self._trigger_callbacks('performance_updated', metrics)
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def _monitor_portfolio(self):
        """Monitor portfolio continuously"""
        while self.is_monitoring:
            try:
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _check_risk_limits(self):
        """Check portfolio risk limits"""
        try:
            risk_metrics = self.calculate_portfolio_risk()
            
            # Check risk budget utilization
            if risk_metrics.risk_budget_utilization > 1.0:
                self.stats['risk_breaches'] += 1
                self._trigger_callbacks('risk_limit_breached', 'risk_budget', risk_metrics)
                logger.warning(f"Risk budget exceeded: {risk_metrics.risk_budget_utilization:.2%}")
            
            # Check concentration risk
            if risk_metrics.concentration_risk > 0.5:  # 50% concentration threshold
                self._trigger_callbacks('risk_limit_breached', 'concentration', risk_metrics)
                logger.warning(f"High concentration risk: {risk_metrics.concentration_risk:.2%}")
            
            # Check correlation risk
            if risk_metrics.correlation_risk > self.max_correlation:
                self._trigger_callbacks('risk_limit_breached', 'correlation', risk_metrics)
                logger.warning(f"High correlation risk: {risk_metrics.correlation_risk:.2%}")
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event in self.portfolio_callbacks:
            self.portfolio_callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args):
        """Trigger event callbacks"""
        try:
            for callback in self.portfolio_callbacks.get(event, []):
                callback(*args)
        except Exception as e:
            logger.error(f"Error triggering callbacks: {e}")
    
    def get_symbol_allocations(self) -> Dict[str, SymbolAllocation]:
        """Get current symbol allocations"""
        with self.lock:
            return self.symbols.copy()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            with self.lock:
                metrics = self.calculate_portfolio_metrics()
                risk_metrics = self.calculate_portfolio_risk()
                
                return {
                    'portfolio_metrics': metrics,
                    'risk_metrics': risk_metrics,
                    'symbol_allocations': {
                        symbol: {
                            'target_weight': alloc.target_weight,
                            'current_weight': alloc.current_weight,
                            'current_value': alloc.current_value,
                            'pnl': alloc.pnl,
                            'position_count': len(alloc.positions)
                        }
                        for symbol, alloc in self.symbols.items()
                    },
                    'statistics': self.stats,
                    'last_update': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def export_portfolio_data(self, filename: str = None) -> str:
        """Export portfolio data to JSON"""
        try:
            if not filename:
                filename = f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            export_data = {
                'portfolio_summary': self.get_portfolio_summary(),
                'performance_history': [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'total_value': metrics.total_value,
                        'total_pnl': metrics.total_pnl,
                        'total_return': metrics.total_return,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown': metrics.max_drawdown
                    }
                    for metrics in self.performance_history[-100:]  # Last 100 records
                ],
                'daily_returns': self.daily_returns[-30:],  # Last 30 days
                'export_time': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Portfolio data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting portfolio data: {e}")
            return ""
    
    def get_statistics(self) -> Dict:
        """Get portfolio manager statistics"""
        with self.lock:
            return {
                **self.stats,
                'symbols_count': len(self.symbols),
                'positions_count': sum(len(positions) for positions in self.positions.values()),
                'monitoring_active': self.is_monitoring,
                'update_interval': self.update_interval,
                'risk_level': self.risk_level.value,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_symbol_weight': self.max_symbol_weight
            }