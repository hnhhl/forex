"""
Risk Limits System
Risk limit enforcement and management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import warnings
warnings.filterwarnings('ignore')

from ..base_system import BaseSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of risk limits"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_VAR = "portfolio_var"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    LEVERAGE = "leverage"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    EXPOSURE = "exposure"
    SECTOR_LIMIT = "sector_limit"
    COUNTRY_LIMIT = "country_limit"
    CURRENCY_LIMIT = "currency_limit"


class LimitScope(Enum):
    """Scope of risk limits"""
    GLOBAL = "global"           # Portfolio-wide
    SECTOR = "sector"           # Sector-specific
    ASSET_CLASS = "asset_class" # Asset class specific
    SYMBOL = "symbol"           # Individual symbol
    STRATEGY = "strategy"       # Strategy-specific
    TRADER = "trader"           # Trader-specific


class LimitStatus(Enum):
    """Status of risk limits"""
    ACTIVE = "active"
    BREACHED = "breached"
    WARNING = "warning"
    DISABLED = "disabled"
    EXPIRED = "expired"


class ActionType(Enum):
    """Actions when limits are breached"""
    ALERT_ONLY = "alert_only"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    BLOCK_NEW_TRADES = "block_new_trades"
    EMERGENCY_STOP = "emergency_stop"
    NOTIFY_MANAGER = "notify_manager"


@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_id: str
    name: str
    limit_type: LimitType
    scope: LimitScope
    
    # Limit values
    soft_limit: float           # Warning threshold
    hard_limit: float           # Breach threshold
    emergency_limit: Optional[float] = None  # Emergency threshold
    
    # Scope specification
    scope_filter: Optional[str] = None  # e.g., "EURUSD", "Technology", "Equity"
    
    # Time constraints
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    trading_hours_only: bool = True
    
    # Actions
    soft_action: ActionType = ActionType.ALERT_ONLY
    hard_action: ActionType = ActionType.REDUCE_POSITION
    emergency_action: ActionType = ActionType.EMERGENCY_STOP
    
    # Status
    status: LimitStatus = LimitStatus.ACTIVE
    is_percentage: bool = False
    is_absolute: bool = True
    
    # Monitoring
    check_frequency: int = 60   # seconds
    consecutive_breaches: int = 1
    
    # Metadata
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'limit_id': self.limit_id,
            'name': self.name,
            'limit_type': self.limit_type.value,
            'scope': self.scope.value,
            'soft_limit': self.soft_limit,
            'hard_limit': self.hard_limit,
            'emergency_limit': self.emergency_limit,
            'scope_filter': self.scope_filter,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'trading_hours_only': self.trading_hours_only,
            'soft_action': self.soft_action.value,
            'hard_action': self.hard_action.value,
            'emergency_action': self.emergency_action.value,
            'status': self.status.value,
            'is_percentage': self.is_percentage,
            'is_absolute': self.is_absolute,
            'check_frequency': self.check_frequency,
            'consecutive_breaches': self.consecutive_breaches,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_checked': self.last_checked.isoformat() if self.last_checked else None
        }


@dataclass
class LimitBreach:
    """Risk limit breach information"""
    breach_id: str
    limit_id: str
    timestamp: datetime
    limit_name: str
    limit_type: LimitType
    scope: LimitScope
    
    # Breach details
    current_value: float
    limit_value: float
    breach_magnitude: float
    breach_percentage: float
    
    # Actions taken
    action_taken: ActionType
    action_successful: bool
    action_details: Optional[str] = None
    
    # Status
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'breach_id': self.breach_id,
            'limit_id': self.limit_id,
            'timestamp': self.timestamp.isoformat(),
            'limit_name': self.limit_name,
            'limit_type': self.limit_type.value,
            'scope': self.scope.value,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'breach_magnitude': self.breach_magnitude,
            'breach_percentage': self.breach_percentage,
            'action_taken': self.action_taken.value,
            'action_successful': self.action_successful,
            'action_details': self.action_details,
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_method': self.resolution_method
        }


class RiskLimitManager(BaseSystem):
    """
    Risk Limit Management System
    
    Features:
    - Multiple limit types and scopes
    - Real-time limit monitoring
    - Automatic breach detection
    - Configurable actions on breach
    - Limit hierarchy and escalation
    - Historical breach tracking
    - Performance impact analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Risk Limit Manager"""
        super().__init__("RiskLimitManager", config)
        
        # Configuration
        self.monitoring_interval = config.get('monitoring_interval', 30) if config else 30  # seconds
        self.enable_enforcement = config.get('enable_enforcement', True) if config else True
        self.max_breach_history = config.get('max_breach_history', 1000) if config else 1000
        
        # Data storage
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.breach_history: List[LimitBreach] = []
        self.active_breaches: Dict[str, LimitBreach] = {}
        
        # Current market data
        self.current_positions: Dict[str, float] = {}
        self.current_portfolio_value: float = 0.0
        self.current_daily_pnl: float = 0.0
        self.current_var: float = 0.0
        self.current_drawdown: float = 0.0
        
        # Monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_lock = threading.RLock()
        
        # Callbacks
        self.breach_callbacks: List[Callable] = []
        self.action_callbacks: List[Callable] = []
        
        # Setup default limits
        self.setup_default_limits()
        
        logger.info("RiskLimitManager initialized")
    
    def setup_default_limits(self):
        """Setup default risk limits"""
        default_limits = [
            # Portfolio VaR limit
            RiskLimit(
                limit_id="portfolio_var_95",
                name="Portfolio VaR 95%",
                limit_type=LimitType.PORTFOLIO_VAR,
                scope=LimitScope.GLOBAL,
                soft_limit=0.02,        # 2% of portfolio
                hard_limit=0.05,        # 5% of portfolio
                emergency_limit=0.10,   # 10% of portfolio
                is_percentage=True,
                soft_action=ActionType.ALERT_ONLY,
                hard_action=ActionType.REDUCE_POSITION,
                emergency_action=ActionType.EMERGENCY_STOP
            ),
            
            # Daily loss limit
            RiskLimit(
                limit_id="daily_loss_limit",
                name="Daily Loss Limit",
                limit_type=LimitType.DAILY_LOSS,
                scope=LimitScope.GLOBAL,
                soft_limit=0.01,        # 1% daily loss
                hard_limit=0.02,        # 2% daily loss
                emergency_limit=0.05,   # 5% daily loss
                is_percentage=True,
                soft_action=ActionType.ALERT_ONLY,
                hard_action=ActionType.BLOCK_NEW_TRADES,
                emergency_action=ActionType.EMERGENCY_STOP
            ),
            
            # Drawdown limit
            RiskLimit(
                limit_id="max_drawdown",
                name="Maximum Drawdown",
                limit_type=LimitType.DRAWDOWN,
                scope=LimitScope.GLOBAL,
                soft_limit=0.05,        # 5% drawdown
                hard_limit=0.10,        # 10% drawdown
                emergency_limit=0.20,   # 20% drawdown
                is_percentage=True,
                soft_action=ActionType.ALERT_ONLY,
                hard_action=ActionType.REDUCE_POSITION,
                emergency_action=ActionType.EMERGENCY_STOP
            ),
            
            # Leverage limit
            RiskLimit(
                limit_id="max_leverage",
                name="Maximum Leverage",
                limit_type=LimitType.LEVERAGE,
                scope=LimitScope.GLOBAL,
                soft_limit=2.0,         # 2x leverage
                hard_limit=5.0,         # 5x leverage
                emergency_limit=10.0,   # 10x leverage
                is_absolute=True,
                soft_action=ActionType.ALERT_ONLY,
                hard_action=ActionType.REDUCE_POSITION,
                emergency_action=ActionType.CLOSE_POSITION
            ),
            
            # Concentration limit
            RiskLimit(
                limit_id="max_concentration",
                name="Maximum Concentration",
                limit_type=LimitType.CONCENTRATION,
                scope=LimitScope.GLOBAL,
                soft_limit=0.20,        # 20% in single position
                hard_limit=0.30,        # 30% in single position
                emergency_limit=0.50,   # 50% in single position
                is_percentage=True,
                soft_action=ActionType.ALERT_ONLY,
                hard_action=ActionType.REDUCE_POSITION,
                emergency_action=ActionType.CLOSE_POSITION
            ),
            
            # Individual position size limit
            RiskLimit(
                limit_id="max_position_size",
                name="Maximum Position Size",
                limit_type=LimitType.POSITION_SIZE,
                scope=LimitScope.SYMBOL,
                soft_limit=0.05,        # 5% of portfolio per position
                hard_limit=0.10,        # 10% of portfolio per position
                emergency_limit=0.20,   # 20% of portfolio per position
                is_percentage=True,
                soft_action=ActionType.ALERT_ONLY,
                hard_action=ActionType.REDUCE_POSITION,
                emergency_action=ActionType.CLOSE_POSITION
            )
        ]
        
        for limit in default_limits:
            limit.created_at = datetime.now()
            limit.created_by = "system"
            self.risk_limits[limit.limit_id] = limit
        
        logger.info(f"Setup {len(default_limits)} default risk limits")
    
    def add_risk_limit(self, limit: RiskLimit) -> bool:
        """Add a new risk limit"""
        try:
            if limit.limit_id in self.risk_limits:
                logger.warning(f"Risk limit {limit.limit_id} already exists")
                return False
            
            limit.created_at = datetime.now()
            self.risk_limits[limit.limit_id] = limit
            
            logger.info(f"Added risk limit: {limit.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding risk limit: {e}")
            return False
    
    def update_risk_limit(self, limit_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing risk limit"""
        try:
            if limit_id not in self.risk_limits:
                logger.error(f"Risk limit {limit_id} not found")
                return False
            
            limit = self.risk_limits[limit_id]
            
            for key, value in updates.items():
                if hasattr(limit, key):
                    setattr(limit, key, value)
                else:
                    logger.warning(f"Unknown limit attribute: {key}")
            
            logger.info(f"Updated risk limit: {limit_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk limit: {e}")
            return False
    
    def remove_risk_limit(self, limit_id: str) -> bool:
        """Remove a risk limit"""
        try:
            if limit_id not in self.risk_limits:
                logger.error(f"Risk limit {limit_id} not found")
                return False
            
            del self.risk_limits[limit_id]
            
            # Remove any active breaches for this limit
            if limit_id in self.active_breaches:
                del self.active_breaches[limit_id]
            
            logger.info(f"Removed risk limit: {limit_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing risk limit: {e}")
            return False
    
    def update_market_data(self, positions: Dict[str, float], portfolio_value: float,
                          daily_pnl: float, var: float, drawdown: float):
        """Update current market data"""
        with self.monitoring_lock:
            self.current_positions = positions.copy()
            self.current_portfolio_value = portfolio_value
            self.current_daily_pnl = daily_pnl
            self.current_var = var
            self.current_drawdown = drawdown
            
            logger.debug("Updated market data for risk limit monitoring")
    
    def check_all_limits(self) -> List[LimitBreach]:
        """Check all active risk limits"""
        try:
            new_breaches = []
            
            with self.monitoring_lock:
                for limit_id, limit in self.risk_limits.items():
                    if limit.status != LimitStatus.ACTIVE:
                        continue
                    
                    # Check time constraints
                    if not self._is_limit_active_now(limit):
                        continue
                    
                    # Calculate current value for this limit
                    current_value = self._calculate_limit_value(limit)
                    if current_value is None:
                        continue
                    
                    # Check for breaches
                    breach = self._check_limit_breach(limit, current_value)
                    if breach:
                        new_breaches.append(breach)
                        
                        # Store breach
                        self.active_breaches[limit_id] = breach
                        self.breach_history.append(breach)
                        
                        # Trim history if needed
                        if len(self.breach_history) > self.max_breach_history:
                            self.breach_history = self.breach_history[-self.max_breach_history:]
                        
                        # Execute action
                        if self.enable_enforcement:
                            self._execute_breach_action(breach)
                        
                        # Trigger callbacks
                        for callback in self.breach_callbacks:
                            try:
                                callback(breach)
                            except Exception as e:
                                logger.error(f"Error in breach callback: {e}")
                    
                    # Update last checked time
                    limit.last_checked = datetime.now()
            
            # Check for resolved breaches
            self._check_resolved_breaches()
            
            if new_breaches:
                logger.warning(f"Detected {len(new_breaches)} new limit breaches")
            
            return new_breaches
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return []
    
    def _is_limit_active_now(self, limit: RiskLimit) -> bool:
        """Check if limit is active at current time"""
        now = datetime.now()
        
        # Check time window
        if limit.start_time and now < limit.start_time:
            return False
        
        if limit.end_time and now > limit.end_time:
            limit.status = LimitStatus.EXPIRED
            return False
        
        # Check trading hours (simplified - would need market calendar)
        if limit.trading_hours_only:
            hour = now.hour
            if hour < 9 or hour > 17:  # Outside 9 AM - 5 PM
                return False
        
        return True
    
    def _calculate_limit_value(self, limit: RiskLimit) -> Optional[float]:
        """Calculate current value for a risk limit"""
        try:
            if limit.limit_type == LimitType.PORTFOLIO_VAR:
                return abs(self.current_var / self.current_portfolio_value) if self.current_portfolio_value > 0 else 0
            
            elif limit.limit_type == LimitType.DAILY_LOSS:
                if self.current_daily_pnl < 0:  # Only consider losses
                    return abs(self.current_daily_pnl / self.current_portfolio_value) if self.current_portfolio_value > 0 else 0
                return 0.0
            
            elif limit.limit_type == LimitType.DRAWDOWN:
                return self.current_drawdown
            
            elif limit.limit_type == LimitType.LEVERAGE:
                if self.current_positions and self.current_portfolio_value > 0:
                    total_exposure = sum(abs(pos) for pos in self.current_positions.values())
                    return total_exposure / self.current_portfolio_value
                return 1.0
            
            elif limit.limit_type == LimitType.CONCENTRATION:
                if self.current_positions:
                    max_position = max(abs(pos) for pos in self.current_positions.values())
                    total_value = sum(abs(pos) for pos in self.current_positions.values())
                    return max_position / total_value if total_value > 0 else 0
                return 0.0
            
            elif limit.limit_type == LimitType.POSITION_SIZE:
                if limit.scope_filter and limit.scope_filter in self.current_positions:
                    position_value = abs(self.current_positions[limit.scope_filter])
                    return position_value / self.current_portfolio_value if self.current_portfolio_value > 0 else 0
                elif not limit.scope_filter and self.current_positions:
                    # Check maximum position size
                    max_position = max(abs(pos) for pos in self.current_positions.values())
                    return max_position / self.current_portfolio_value if self.current_portfolio_value > 0 else 0
                return 0.0
            
            elif limit.limit_type == LimitType.EXPOSURE:
                if self.current_positions:
                    total_exposure = sum(abs(pos) for pos in self.current_positions.values())
                    return total_exposure / self.current_portfolio_value if self.current_portfolio_value > 0 else 0
                return 0.0
            
            else:
                logger.warning(f"Unsupported limit type: {limit.limit_type}")
                return None
            
        except Exception as e:
            logger.error(f"Error calculating limit value for {limit.limit_id}: {e}")
            return None
    
    def _check_limit_breach(self, limit: RiskLimit, current_value: float) -> Optional[LimitBreach]:
        """Check if a limit is breached"""
        try:
            breach_level = None
            limit_value = None
            action = None
            
            # Check emergency limit first
            if limit.emergency_limit is not None and current_value >= limit.emergency_limit:
                breach_level = "emergency"
                limit_value = limit.emergency_limit
                action = limit.emergency_action
            
            # Check hard limit
            elif current_value >= limit.hard_limit:
                breach_level = "hard"
                limit_value = limit.hard_limit
                action = limit.hard_action
            
            # Check soft limit
            elif current_value >= limit.soft_limit:
                breach_level = "soft"
                limit_value = limit.soft_limit
                action = limit.soft_action
            
            if breach_level is None:
                return None
            
            # Check if this is a new breach or escalation
            existing_breach = self.active_breaches.get(limit.limit_id)
            if existing_breach and not existing_breach.is_resolved:
                # Check if this is an escalation
                if (breach_level == "emergency" and existing_breach.action_taken != ActionType.EMERGENCY_STOP) or \
                   (breach_level == "hard" and existing_breach.action_taken == ActionType.ALERT_ONLY):
                    # This is an escalation, create new breach
                    pass
                else:
                    # Same level breach, don't create duplicate
                    return None
            
            # Create breach record
            breach_id = f"{limit.limit_id}_{breach_level}_{int(datetime.now().timestamp())}"
            breach_magnitude = current_value - limit_value
            breach_percentage = (breach_magnitude / limit_value) * 100 if limit_value > 0 else 0
            
            breach = LimitBreach(
                breach_id=breach_id,
                limit_id=limit.limit_id,
                timestamp=datetime.now(),
                limit_name=limit.name,
                limit_type=limit.limit_type,
                scope=limit.scope,
                current_value=current_value,
                limit_value=limit_value,
                breach_magnitude=breach_magnitude,
                breach_percentage=breach_percentage,
                action_taken=action,
                action_successful=False  # Will be updated after action execution
            )
            
            return breach
            
        except Exception as e:
            logger.error(f"Error checking limit breach: {e}")
            return None
    
    def _execute_breach_action(self, breach: LimitBreach):
        """Execute action for limit breach"""
        try:
            action_successful = False
            action_details = None
            
            if breach.action_taken == ActionType.ALERT_ONLY:
                # Just log the alert
                logger.warning(f"RISK LIMIT BREACH: {breach.limit_name} - Current: {breach.current_value:.4f}, Limit: {breach.limit_value:.4f}")
                action_successful = True
                action_details = "Alert logged"
            
            elif breach.action_taken == ActionType.REDUCE_POSITION:
                # Reduce positions (would need integration with order management)
                logger.critical(f"REDUCING POSITIONS due to {breach.limit_name} breach")
                action_details = "Position reduction initiated"
                action_successful = True  # Assume success for demo
            
            elif breach.action_taken == ActionType.CLOSE_POSITION:
                # Close positions (would need integration with order management)
                logger.critical(f"CLOSING POSITIONS due to {breach.limit_name} breach")
                action_details = "Position closure initiated"
                action_successful = True  # Assume success for demo
            
            elif breach.action_taken == ActionType.BLOCK_NEW_TRADES:
                # Block new trades (would need integration with order management)
                logger.critical(f"BLOCKING NEW TRADES due to {breach.limit_name} breach")
                action_details = "New trades blocked"
                action_successful = True  # Assume success for demo
            
            elif breach.action_taken == ActionType.EMERGENCY_STOP:
                # Emergency stop all trading
                logger.critical(f"EMERGENCY STOP triggered by {breach.limit_name} breach")
                action_details = "Emergency stop activated"
                action_successful = True  # Assume success for demo
            
            elif breach.action_taken == ActionType.NOTIFY_MANAGER:
                # Notify risk manager
                logger.critical(f"NOTIFYING MANAGER of {breach.limit_name} breach")
                action_details = "Manager notification sent"
                action_successful = True  # Assume success for demo
            
            # Update breach record
            breach.action_successful = action_successful
            breach.action_details = action_details
            
            # Trigger action callbacks
            for callback in self.action_callbacks:
                try:
                    callback(breach)
                except Exception as e:
                    logger.error(f"Error in action callback: {e}")
            
        except Exception as e:
            logger.error(f"Error executing breach action: {e}")
            breach.action_successful = False
            breach.action_details = f"Action failed: {str(e)}"
    
    def _check_resolved_breaches(self):
        """Check for resolved breaches"""
        try:
            resolved_breaches = []
            
            for limit_id, breach in self.active_breaches.items():
                if breach.is_resolved:
                    continue
                
                limit = self.risk_limits.get(limit_id)
                if not limit:
                    continue
                
                # Calculate current value
                current_value = self._calculate_limit_value(limit)
                if current_value is None:
                    continue
                
                # Check if breach is resolved
                is_resolved = False
                resolution_method = None
                
                if current_value < limit.soft_limit:
                    is_resolved = True
                    resolution_method = "Natural recovery below soft limit"
                elif breach.action_taken in [ActionType.REDUCE_POSITION, ActionType.CLOSE_POSITION] and \
                     current_value < breach.limit_value:
                    is_resolved = True
                    resolution_method = f"Resolved by {breach.action_taken.value}"
                
                if is_resolved:
                    breach.is_resolved = True
                    breach.resolved_at = datetime.now()
                    breach.resolution_method = resolution_method
                    resolved_breaches.append(limit_id)
                    
                    logger.info(f"Resolved breach: {breach.breach_id} - {resolution_method}")
            
            # Remove resolved breaches from active list
            for limit_id in resolved_breaches:
                del self.active_breaches[limit_id]
            
        except Exception as e:
            logger.error(f"Error checking resolved breaches: {e}")
    
    def start_monitoring(self) -> bool:
        """Start risk limit monitoring"""
        try:
            if self.is_monitoring:
                logger.warning("Risk limit monitoring already running")
                return True
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Risk limit monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting risk limit monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop risk limit monitoring"""
        try:
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Risk limit monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping risk limit monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Risk limit monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Check all limits
                breaches = self.check_all_limits()
                
                # Update status
                self.update_status()
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        logger.info("Risk limit monitoring loop stopped")
    
    def add_breach_callback(self, callback: Callable[[LimitBreach], None]):
        """Add breach callback function"""
        self.breach_callbacks.append(callback)
        logger.info("Added breach callback")
    
    def add_action_callback(self, callback: Callable[[LimitBreach], None]):
        """Add action callback function"""
        self.action_callbacks.append(callback)
        logger.info("Added action callback")
    
    def get_limit_status_report(self) -> Dict[str, Any]:
        """Get comprehensive limit status report"""
        try:
            active_limits = [limit for limit in self.risk_limits.values() if limit.status == LimitStatus.ACTIVE]
            breached_limits = len(self.active_breaches)
            
            # Limit utilization
            limit_utilization = {}
            for limit_id, limit in self.risk_limits.items():
                if limit.status != LimitStatus.ACTIVE:
                    continue
                
                current_value = self._calculate_limit_value(limit)
                if current_value is not None:
                    utilization = (current_value / limit.soft_limit) * 100 if limit.soft_limit > 0 else 0
                    limit_utilization[limit_id] = {
                        'name': limit.name,
                        'current_value': current_value,
                        'soft_limit': limit.soft_limit,
                        'hard_limit': limit.hard_limit,
                        'utilization_pct': utilization,
                        'status': 'breached' if limit_id in self.active_breaches else 'normal'
                    }
            
            # Recent breaches
            recent_breaches = []
            for breach in self.breach_history[-10:]:  # Last 10 breaches
                recent_breaches.append(breach.to_dict())
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_limits': len(self.risk_limits),
                    'active_limits': len(active_limits),
                    'breached_limits': breached_limits,
                    'total_breaches_today': len([b for b in self.breach_history if b.timestamp.date() == datetime.now().date()]),
                    'monitoring_status': self.is_monitoring
                },
                'limit_utilization': limit_utilization,
                'active_breaches': [breach.to_dict() for breach in self.active_breaches.values()],
                'recent_breaches': recent_breaches,
                'configuration': {
                    'monitoring_interval': self.monitoring_interval,
                    'enable_enforcement': self.enable_enforcement,
                    'max_breach_history': self.max_breach_history
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating limit status report: {e}")
            return {'error': str(e)}
    
    def export_data(self, filepath: str) -> bool:
        """Export risk limit data"""
        try:
            export_data = {
                'risk_limits': [limit.to_dict() for limit in self.risk_limits.values()],
                'breach_history': [breach.to_dict() for breach in self.breach_history],
                'active_breaches': [breach.to_dict() for breach in self.active_breaches.values()],
                'configuration': {
                    'monitoring_interval': self.monitoring_interval,
                    'enable_enforcement': self.enable_enforcement,
                    'max_breach_history': self.max_breach_history
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Risk limit data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting risk limit data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk limit manager statistics"""
        return {
            'total_limits': len(self.risk_limits),
            'active_limits': len([l for l in self.risk_limits.values() if l.status == LimitStatus.ACTIVE]),
            'breached_limits': len(self.active_breaches),
            'total_breaches': len(self.breach_history),
            'breaches_today': len([b for b in self.breach_history if b.timestamp.date() == datetime.now().date()]),
            'is_monitoring': self.is_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'enable_enforcement': self.enable_enforcement,
            'breach_callbacks': len(self.breach_callbacks),
            'action_callbacks': len(self.action_callbacks),
            'last_update': self.last_update.isoformat() if self.last_update else None
        } 