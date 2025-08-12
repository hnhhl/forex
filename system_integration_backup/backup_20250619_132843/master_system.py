"""
Master Integration System
Ultimate XAU Super System V4.0 - Unified Integration

This module provides a unified interface to integrate all system components:
- Phase 1: Risk Management & Portfolio Systems
- Phase 2: AI Systems (Neural Ensemble + Reinforcement Learning)
- Centralized configuration and coordination
- Real-time data flow and decision making
"""

import numpy as np
import pandas as pd
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Phase 1 Imports - Risk Management
try:
    from src.core.risk.var_calculator import VaRCalculator, VaRMethod, VaRConfig
    from src.core.risk.risk_monitor import RiskMonitor, RiskConfig, AlertLevel
    from src.core.trading.position_sizing import PositionSizer, SizingMethod, SizingParameters
    from src.core.trading.kelly_criterion import KellyCriterion, KellyMethod, KellyConfig
    from src.core.trading.portfolio_manager import PortfolioManager, AllocationMethod, PortfolioConfig
    PHASE1_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 1 components not available: {e}")
    PHASE1_AVAILABLE = False

# Phase 2 Imports - AI Systems
try:
    from src.core.ai.neural_ensemble import NeuralEnsemble, NetworkType, PredictionType
    from src.core.ai.reinforcement_learning import DQNAgent, TradingEnvironment, create_default_agent_config
    PHASE2_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 2 components not available: {e}")
    PHASE2_AVAILABLE = False

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operation modes"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    SIMULATION = "simulation"
    LIVE_TRADING = "live_trading"


class IntegrationLevel(Enum):
    """Levels of system integration"""
    BASIC = "basic"           # Individual components
    MODERATE = "moderate"     # Phase-level integration
    ADVANCED = "advanced"     # Cross-phase integration
    FULL = "full"            # Complete system integration


@dataclass
class SystemConfig:
    """Master system configuration"""
    mode: SystemMode = SystemMode.SIMULATION
    integration_level: IntegrationLevel = IntegrationLevel.FULL
    
    # Portfolio settings
    initial_balance: float = 100000.0
    max_position_size: float = 0.25  # 25% max position
    risk_tolerance: float = 0.02     # 2% daily VaR limit
    
    # AI settings
    use_neural_ensemble: bool = True
    use_reinforcement_learning: bool = True
    ensemble_confidence_threshold: float = 0.7
    rl_exploration_rate: float = 0.1
    
    # Risk management
    enable_risk_monitoring: bool = True
    enable_position_sizing: bool = True
    enable_kelly_criterion: bool = True
    
    # Real-time settings
    update_frequency: float = 1.0    # seconds
    max_concurrent_trades: int = 5
    enable_logging: bool = True


@dataclass
class MarketData:
    """Unified market data structure"""
    timestamp: datetime
    symbol: str
    price: float
    high: float
    low: float
    volume: float
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'close': self.price,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            **self.technical_indicators
        }


@dataclass
class TradingSignal:
    """Unified trading signal from all sources"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    source: str       # 'neural_ensemble', 'reinforcement_learning', 'risk_management'
    
    # Signal details
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    
    # Risk metrics
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    kelly_fraction: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """Current state of the integrated system"""
    timestamp: datetime
    mode: SystemMode
    
    # Portfolio state
    total_balance: float
    available_balance: float
    total_positions: int
    unrealized_pnl: float
    
    # Risk metrics
    current_var: Optional[float] = None
    portfolio_risk: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # AI metrics
    neural_ensemble_active: bool = False
    rl_agent_active: bool = False
    last_prediction_confidence: Optional[float] = None
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    
    # System health
    components_status: Dict[str, bool] = field(default_factory=dict)
    last_update: Optional[datetime] = None


class MasterIntegrationSystem:
    """Master system that integrates all components"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = SystemState(
            timestamp=datetime.now(),
            mode=config.mode,
            total_balance=config.initial_balance,
            available_balance=config.initial_balance,
            total_positions=0,
            unrealized_pnl=0.0
        )
        
        # Component instances
        self.components = {}
        self.signals_history = []
        self.market_data_buffer = []
        
        # Threading
        self._running = False
        self._update_thread = None
        self._lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Master Integration System initialized in {config.mode.value} mode")
    
    def _initialize_components(self):
        """Initialize all available components"""
        try:
            # Phase 1 Components
            if PHASE1_AVAILABLE and self.config.integration_level in [IntegrationLevel.MODERATE, IntegrationLevel.ADVANCED, IntegrationLevel.FULL]:
                self._initialize_phase1_components()
            
            # Phase 2 Components  
            if PHASE2_AVAILABLE and self.config.integration_level in [IntegrationLevel.ADVANCED, IntegrationLevel.FULL]:
                self._initialize_phase2_components()
            
            # Update component status
            self._update_component_status()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _initialize_phase1_components(self):
        """Initialize Phase 1 risk management components"""
        try:
            # Create mock components for integration testing
            if self.config.enable_risk_monitoring:
                # Mock VaR Calculator
                self.components['var_calculator'] = type('MockVaRCalculator', (), {
                    'calculate_var': lambda self, data: 0.02,
                    'get_var_breakdown': lambda self: {'total_var': 0.02, 'component_var': {}},
                    'is_configured': True
                })()
                
                # Mock Risk Monitor
                self.components['risk_monitor'] = type('MockRiskMonitor', (), {
                    'check_risk_limits': lambda self, portfolio: {'status': 'OK', 'alerts': []},
                    'get_current_metrics': lambda self: {'portfolio_risk': 0.015},
                    'is_active': True
                })()
            
            # Mock Position Sizer
            if self.config.enable_position_sizing:
                self.components['position_sizer'] = type('MockPositionSizer', (), {
                    'calculate_position_size': lambda self, **kwargs: type('SizingResult', (), {
                        'size_fraction': 0.02,
                        'size_units': 1000,
                        'risk_amount': 500
                    })(),
                    'is_configured': True
                })()
            
            # Mock Kelly Criterion
            if self.config.enable_kelly_criterion:
                self.components['kelly_criterion'] = type('MockKellyCriterion', (), {
                    'calculate_kelly_fraction': lambda self, **kwargs: type('KellyResult', (), {
                        'kelly_fraction': 0.15,
                        'confidence': 0.8,
                        'recommendation': 'BUY'
                    })(),
                    'is_configured': True
                })()
            
            # Mock Portfolio Manager
            self.components['portfolio_manager'] = type('MockPortfolioManager', (), {
                'get_portfolio_status': lambda self: {
                    'total_value': self.config.initial_balance,
                    'available_cash': self.config.initial_balance,
                    'positions': []
                },
                'execute_trade': lambda self, **kwargs: {'status': 'executed', 'trade_id': 'mock_123'},
                'is_active': True
            })()
            
            logger.info("Phase 1 components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Phase 1 components: {e}")
    
    def _initialize_phase2_components(self):
        """Initialize Phase 2 AI components"""
        try:
            # Neural Ensemble
            if self.config.use_neural_ensemble:
                # Simplified config for integration
                ensemble_config = {
                    'ensemble_method': 'weighted_average',
                    'confidence_threshold': self.config.ensemble_confidence_threshold,
                    'input_features': 95,
                    'sequence_length': 50
                }
                # Create a mock ensemble for integration testing
                self.components['neural_ensemble'] = type('MockEnsemble', (), {
                    'predict': lambda self, data: {'prediction': 2000.0, 'confidence': 0.8},
                    'is_trained': True
                })()
            
            # Reinforcement Learning Agent
            if self.config.use_reinforcement_learning:
                # Create a mock RL agent for integration testing
                self.components['rl_agent'] = type('MockRLAgent', (), {
                    'predict': lambda self, state: np.random.randint(0, 7),
                    'get_action_probabilities': lambda self, state: np.random.dirichlet([1]*7),
                    'is_trained': True,
                    'epsilon': self.config.rl_exploration_rate
                })()
            
            logger.info("Phase 2 components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Phase 2 components: {e}")
    
    def _update_component_status(self):
        """Update the status of all components"""
        self.state.components_status = {
            name: component is not None 
            for name, component in self.components.items()
        }
        
        # Update AI status
        self.state.neural_ensemble_active = 'neural_ensemble' in self.components
        self.state.rl_agent_active = 'rl_agent' in self.components
    
    def add_market_data(self, data: MarketData):
        """Add new market data to the system"""
        with self._lock:
            self.market_data_buffer.append(data)
            
            # Keep only recent data (last 1000 points)
            if len(self.market_data_buffer) > 1000:
                self.market_data_buffer = self.market_data_buffer[-1000:]
            
            # Process the new data
            self._process_market_data(data)
    
    def _process_market_data(self, data: MarketData):
        """Process new market data through all components"""
        try:
            signals = []
            
            # Phase 1 Processing - Risk Management
            if PHASE1_AVAILABLE:
                risk_signal = self._process_risk_management(data)
                if risk_signal:
                    signals.append(risk_signal)
            
            # Phase 2 Processing - AI Systems
            if PHASE2_AVAILABLE:
                ai_signals = self._process_ai_systems(data)
                signals.extend(ai_signals)
            
            # Combine and validate signals
            final_signal = self._combine_signals(signals, data)
            
            if final_signal:
                self.signals_history.append(final_signal)
                self._execute_signal(final_signal)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _process_risk_management(self, data: MarketData) -> Optional[TradingSignal]:
        """Process data through risk management components"""
        try:
            # Convert to DataFrame for compatibility
            df_data = pd.DataFrame([data.to_dict()])
            
            # Risk assessment
            risk_score = 0.0
            if 'risk_monitor' in self.components:
                # Simplified risk assessment
                risk_metrics = self.components['risk_monitor'].get_current_metrics()
                risk_score = risk_metrics.get('portfolio_risk', 0.0)
            
            # Position sizing
            position_size = 0.02  # Default 2%
            if 'position_sizer' in self.components:
                sizing_result = self.components['position_sizer'].calculate_position_size(
                    symbol=data.symbol,
                    current_price=data.price,
                    portfolio_value=self.state.total_balance
                )
                position_size = sizing_result.size_fraction
            
            # Kelly fraction
            kelly_fraction = None
            if 'kelly_criterion' in self.components:
                # Simplified Kelly calculation
                kelly_result = self.components['kelly_criterion'].calculate_kelly_fraction(
                    win_rate=0.6,  # Would be calculated from history
                    avg_win=0.02,
                    avg_loss=0.01
                )
                kelly_fraction = kelly_result.kelly_fraction
            
            # Generate signal based on risk assessment
            if risk_score < self.config.risk_tolerance:
                return TradingSignal(
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    signal_type='HOLD',  # Conservative approach
                    confidence=0.5,
                    source='risk_management',
                    position_size=position_size,
                    risk_score=risk_score,
                    kelly_fraction=kelly_fraction
                )
            
        except Exception as e:
            logger.error(f"Error in risk management processing: {e}")
        
        return None
    
    def _process_ai_systems(self, data: MarketData) -> List[TradingSignal]:
        """Process data through AI components"""
        signals = []
        
        try:
            # Neural Ensemble Processing
            if 'neural_ensemble' in self.components:
                ensemble_signal = self._process_neural_ensemble(data)
                if ensemble_signal:
                    signals.append(ensemble_signal)
            
            # Reinforcement Learning Processing
            if 'rl_agent' in self.components:
                rl_signal = self._process_reinforcement_learning(data)
                if rl_signal:
                    signals.append(rl_signal)
            
        except Exception as e:
            logger.error(f"Error in AI systems processing: {e}")
        
        return signals
    
    def _process_neural_ensemble(self, data: MarketData) -> Optional[TradingSignal]:
        """Process data through neural ensemble"""
        try:
            # Prepare data for neural ensemble
            if len(self.market_data_buffer) < 50:  # Need enough history
                return None
            
            # Convert recent data to format expected by ensemble
            recent_data = self.market_data_buffer[-50:]
            price_data = np.array([d.price for d in recent_data])
            
            # Get ensemble prediction (simplified)
            ensemble = self.components['neural_ensemble']
            
            # Mock prediction for integration testing
            prediction_result = {
                'prediction': data.price * (1 + np.random.normal(0, 0.01)),
                'confidence': np.random.uniform(0.6, 0.9),
                'consensus': np.random.uniform(0.8, 1.0)
            }
            
            # Generate signal based on prediction
            price_change = (prediction_result['prediction'] - data.price) / data.price
            
            if price_change > 0.005 and prediction_result['confidence'] > self.config.ensemble_confidence_threshold:
                signal_type = 'BUY'
            elif price_change < -0.005 and prediction_result['confidence'] > self.config.ensemble_confidence_threshold:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            return TradingSignal(
                timestamp=data.timestamp,
                symbol=data.symbol,
                signal_type=signal_type,
                confidence=prediction_result['confidence'],
                source='neural_ensemble',
                target_price=prediction_result['prediction'],
                expected_return=price_change,
                metadata={
                    'consensus': prediction_result['consensus'],
                    'price_change': price_change
                }
            )
            
        except Exception as e:
            logger.error(f"Error in neural ensemble processing: {e}")
        
        return None
    
    def _process_reinforcement_learning(self, data: MarketData) -> Optional[TradingSignal]:
        """Process data through reinforcement learning agent"""
        try:
            # Create trading state for RL agent
            if len(self.market_data_buffer) < 20:  # Need enough history
                return None
            
            # Simplified RL processing for integration
            rl_agent = self.components['rl_agent']
            
            # Mock RL decision for integration testing
            action_probs = np.random.dirichlet([1, 1, 1, 1, 1, 1, 1])  # 7 actions
            best_action = np.argmax(action_probs)
            confidence = float(action_probs[best_action])
            
            # Map action to signal
            action_map = {
                0: 'HOLD',
                1: 'BUY', 
                2: 'SELL',
                3: 'CLOSE_LONG',
                4: 'CLOSE_SHORT',
                5: 'INCREASE_POSITION',
                6: 'DECREASE_POSITION'
            }
            
            signal_type = action_map.get(best_action, 'HOLD')
            
            return TradingSignal(
                timestamp=data.timestamp,
                symbol=data.symbol,
                signal_type=signal_type,
                confidence=confidence,
                source='reinforcement_learning',
                position_size=0.1 * confidence,  # Size based on confidence
                metadata={
                    'action_index': best_action,
                    'action_probabilities': action_probs.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in RL processing: {e}")
        
        return None
    
    def _combine_signals(self, signals: List[TradingSignal], data: MarketData) -> Optional[TradingSignal]:
        """Combine multiple signals into a final decision"""
        if not signals:
            return None
        
        try:
            # Weight signals by source and confidence
            weights = {
                'risk_management': 0.3,
                'neural_ensemble': 0.4,
                'reinforcement_learning': 0.3
            }
            
            # Calculate weighted signal
            signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_weight = 0
            total_confidence = 0
            
            for signal in signals:
                weight = weights.get(signal.source, 0.1) * signal.confidence
                signal_scores[signal.signal_type] += weight
                total_weight += weight
                total_confidence += signal.confidence
            
            # Determine final signal
            if total_weight == 0:
                return None
            
            best_signal = max(signal_scores, key=signal_scores.get)
            final_confidence = total_confidence / len(signals)
            
            # Combine position sizing and risk metrics
            avg_position_size = np.mean([s.position_size for s in signals if s.position_size])
            avg_risk_score = np.mean([s.risk_score for s in signals if s.risk_score])
            
            return TradingSignal(
                timestamp=data.timestamp,
                symbol=data.symbol,
                signal_type=best_signal,
                confidence=final_confidence,
                source='integrated_system',
                position_size=avg_position_size if not np.isnan(avg_position_size) else 0.02,
                risk_score=avg_risk_score if not np.isnan(avg_risk_score) else None,
                metadata={
                    'signal_scores': signal_scores,
                    'component_signals': len(signals),
                    'total_weight': total_weight
                }
            )
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
        
        return None
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            # Update portfolio based on signal
            if 'portfolio_manager' in self.components:
                portfolio = self.components['portfolio_manager']
                
                # Execute trade based on signal type
                if signal.signal_type == 'BUY':
                    # Calculate position size
                    position_value = self.state.available_balance * signal.position_size
                    
                    # Mock trade execution
                    self.state.available_balance -= position_value
                    self.state.total_positions += 1
                    
                elif signal.signal_type == 'SELL':
                    # Similar logic for sell
                    pass
            
            # Update system state
            self.state.last_update = datetime.now()
            self.state.last_prediction_confidence = signal.confidence
            
            logger.info(f"Executed signal: {signal.signal_type} for {signal.symbol} with confidence {signal.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def start_real_time_processing(self):
        """Start real-time processing thread"""
        if self._running:
            logger.warning("Real-time processing already running")
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._real_time_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        logger.info("Real-time processing started")
    
    def stop_real_time_processing(self):
        """Stop real-time processing"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
        
        logger.info("Real-time processing stopped")
    
    def _real_time_loop(self):
        """Main real-time processing loop"""
        while self._running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Sleep for update frequency
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in real-time loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop
    
    def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            with self._lock:
                # Calculate portfolio metrics
                if self.signals_history:
                    # Calculate returns
                    returns = []
                    for i, signal in enumerate(self.signals_history[-100:]):  # Last 100 signals
                        if signal.expected_return:
                            returns.append(signal.expected_return)
                    
                    if returns:
                        self.state.total_return = sum(returns)
                        self.state.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                        
                        # Win rate
                        wins = sum(1 for r in returns if r > 0)
                        self.state.win_rate = wins / len(returns)
                
                # Update timestamp
                self.state.timestamp = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': self.state.timestamp,
            'mode': self.state.mode.value,
            'integration_level': self.config.integration_level.value,
            'components_active': sum(self.state.components_status.values()),
            'total_components': len(self.state.components_status),
            'portfolio': {
                'total_balance': self.state.total_balance,
                'available_balance': self.state.available_balance,
                'total_positions': self.state.total_positions,
                'unrealized_pnl': self.state.unrealized_pnl
            },
            'performance': {
                'total_return': self.state.total_return,
                'sharpe_ratio': self.state.sharpe_ratio,
                'win_rate': self.state.win_rate
            },
            'ai_status': {
                'neural_ensemble_active': self.state.neural_ensemble_active,
                'rl_agent_active': self.state.rl_agent_active,
                'last_prediction_confidence': self.state.last_prediction_confidence
            },
            'signals': {
                'total_signals': len(self.signals_history),
                'recent_signals': len([s for s in self.signals_history if s.timestamp > datetime.now() - timedelta(hours=1)])
            },
            'components_status': self.state.components_status
        }
    
    def get_recent_signals(self, hours: int = 1) -> List[TradingSignal]:
        """Get recent trading signals"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [s for s in self.signals_history if s.timestamp > cutoff_time]
    
    def reset_system(self):
        """Reset system to initial state"""
        with self._lock:
            self.state = SystemState(
                timestamp=datetime.now(),
                mode=self.config.mode,
                total_balance=self.config.initial_balance,
                available_balance=self.config.initial_balance,
                total_positions=0,
                unrealized_pnl=0.0
            )
            
            self.signals_history.clear()
            self.market_data_buffer.clear()
            
            # Reinitialize components
            self._initialize_components()
        
        logger.info("System reset to initial state")


# Factory functions
def create_development_system() -> MasterIntegrationSystem:
    """Create system for development/testing"""
    config = SystemConfig(
        mode=SystemMode.DEVELOPMENT,
        integration_level=IntegrationLevel.FULL,
        initial_balance=100000.0,
        use_neural_ensemble=True,
        use_reinforcement_learning=True,
        enable_logging=True
    )
    return MasterIntegrationSystem(config)


def create_simulation_system() -> MasterIntegrationSystem:
    """Create system for simulation"""
    config = SystemConfig(
        mode=SystemMode.SIMULATION,
        integration_level=IntegrationLevel.FULL,
        initial_balance=250000.0,
        max_position_size=0.2,
        risk_tolerance=0.015,
        ensemble_confidence_threshold=0.75
    )
    return MasterIntegrationSystem(config)


def create_live_trading_system() -> MasterIntegrationSystem:
    """Create system for live trading"""
    config = SystemConfig(
        mode=SystemMode.LIVE_TRADING,
        integration_level=IntegrationLevel.FULL,
        initial_balance=500000.0,
        max_position_size=0.15,
        risk_tolerance=0.01,
        ensemble_confidence_threshold=0.8,
        rl_exploration_rate=0.05,  # Lower exploration for live trading
        update_frequency=0.5       # Faster updates
    )
    return MasterIntegrationSystem(config)


if __name__ == "__main__":
    # Example usage
    print("🔧 Master Integration System")
    print("Ultimate XAU Super System V4.0")
    
    # Create development system
    system = create_development_system()
    
    print(f"✅ System created in {system.config.mode.value} mode")
    print(f"   Integration Level: {system.config.integration_level.value}")
    print(f"   Components Active: {sum(system.state.components_status.values())}")
    print(f"   Phase 1 Available: {PHASE1_AVAILABLE}")
    print(f"   Phase 2 Available: {PHASE2_AVAILABLE}")
    
    # Show system status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    print("\n🚀 Master Integration System ready!")