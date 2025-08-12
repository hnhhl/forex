"""
Advanced Portfolio Backtesting Module for Ultimate XAU Super System V4.0
Day 31: Advanced Portfolio Backtesting

Triển khai hệ thống backtesting portfolio tiên tiến:
- Multi-strategy backtesting engine
- AI-driven portfolio optimization
- Deep learning integration
- Advanced performance analytics
- Risk-adjusted backtesting
- Real-time strategy evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import các modules khác
try:
    from .deep_learning_neural_networks import DeepLearningNeuralNetworks, NetworkConfig, NetworkType
    from .ml_enhanced_trading_signals import MLEnhancedTradingSignals
    from .advanced_risk_management import AdvancedRiskManagement
    from .risk_adjusted_portfolio_optimization import RiskAdjustedPortfolioOptimization
except ImportError:
    # Fallback cho testing
    pass

# Khởi tạo logger
logger = logging.getLogger(__name__)

class BacktestingStrategy(Enum):
    """Các chiến lược backtesting"""
    BUY_AND_HOLD = "buy_and_hold"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ML_SIGNALS = "ml_signals"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE_AI = "ensemble_ai"
    ADAPTIVE = "adaptive"

class PerformanceMetric(Enum):
    """Các chỉ số đánh giá hiệu suất"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    ALPHA = "alpha"
    BETA = "beta"

class RebalanceFrequency(Enum):
    """Tần suất rebalance"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ADAPTIVE = "adaptive"

@dataclass
class BacktestingConfig:
    """Cấu hình backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    strategy: BacktestingStrategy = BacktestingStrategy.ML_SIGNALS
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 0.1  # 10% max per position
    risk_free_rate: float = 0.02  # 2% annual
    benchmark: str = "XAU"
    # AI/ML settings
    use_ai_signals: bool = True
    use_deep_learning: bool = True
    ai_confidence_threshold: float = 0.6
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class TradeResult:
    """Kết quả giao dịch"""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    quantity: float
    price: float
    value: float
    commission: float
    slippage_cost: float
    portfolio_value: float
    position_size: float
    signal_source: str  # "ML", "DL", "ENSEMBLE", etc.
    confidence: float
    
@dataclass
class PortfolioSnapshot:
    """Snapshot portfolio tại một thời điểm"""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]
    weights: Dict[str, float]
    daily_return: float
    cumulative_return: float
    drawdown: float
    risk_metrics: Dict[str, float]

@dataclass
class BacktestingResult:
    """Kết quả backtesting tổng thể"""
    config: BacktestingConfig
    portfolio_history: List[PortfolioSnapshot]
    trade_history: List[TradeResult]
    performance_metrics: Dict[PerformanceMetric, float]
    risk_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, float]
    ai_performance: Dict[str, float]
    execution_time: float
    total_trades: int
    winning_trades: int
    losing_trades: int

class SignalGenerator:
    """Tạo tín hiệu trading từ nhiều nguồn"""
    
    def __init__(self, config: BacktestingConfig):
        self.config = config
        self.ml_signals = None
        self.dl_networks = None
        self.risk_manager = None
        
        # Khởi tạo AI components nếu được bật
        if config.use_ai_signals:
            try:
                self.ml_signals = MLEnhancedTradingSignals()
            except:
                logger.warning("ML Enhanced Trading Signals not available")
                
        if config.use_deep_learning:
            try:
                self.dl_networks = DeepLearningNeuralNetworks()
            except:
                logger.warning("Deep Learning Neural Networks not available")
                
        try:
            self.risk_manager = AdvancedRiskManagement()
        except:
            logger.warning("Advanced Risk Management not available")
    
    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Tạo tín hiệu từ tất cả các nguồn"""
        signals = {}
        
        # Traditional technical signals
        signals.update(self._generate_technical_signals(data))
        
        # ML signals
        if self.ml_signals and self.config.use_ai_signals:
            try:
                ml_signal = self._generate_ml_signals(data)
                signals.update(ml_signal)
            except Exception as e:
                logger.warning(f"ML signal generation failed: {e}")
        
        # Deep learning signals
        if self.dl_networks and self.config.use_deep_learning:
            try:
                dl_signal = self._generate_dl_signals(data)
                signals.update(dl_signal)
            except Exception as e:
                logger.warning(f"DL signal generation failed: {e}")
        
        # Risk-adjusted signals
        if self.risk_manager:
            try:
                risk_adjusted = self._apply_risk_adjustment(signals, data)
                signals.update(risk_adjusted)
            except Exception as e:
                logger.warning(f"Risk adjustment failed: {e}")
        
        # Ensemble signal
        ensemble_signal = self._create_ensemble_signal(signals)
        signals['ensemble'] = ensemble_signal
        
        return signals
    
    def _generate_technical_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Tạo tín hiệu technical analysis truyền thống"""
        if len(data) < 20:
            return {'technical': 0.0}
            
        # Moving averages
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9).mean()
        macd_signal = macd.iloc[-1] - signal_line.iloc[-1]
        
        # Combine signals
        ma_signal = 1.0 if current_price > sma_5 > sma_20 else -1.0 if current_price < sma_5 < sma_20 else 0.0
        rsi_signal = -1.0 if current_rsi > 70 else 1.0 if current_rsi < 30 else 0.0
        macd_signal = 1.0 if macd_signal > 0 else -1.0 if macd_signal < 0 else 0.0
        
        # Weighted combination
        technical_signal = (ma_signal * 0.5 + rsi_signal * 0.3 + macd_signal * 0.2)
        
        return {
            'technical': technical_signal,
            'ma_signal': ma_signal,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal
        }
    
    def _generate_ml_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Tạo tín hiệu từ ML models"""
        try:
            # Placeholder for ML signal generation
            # In reality, this would use the trained ML models
            ml_prediction = np.random.uniform(-1, 1)  # Simulated ML prediction
            confidence = np.random.uniform(0.5, 0.95)
            
            return {
                'ml_signal': ml_prediction,
                'ml_confidence': confidence
            }
        except Exception as e:
            logger.warning(f"ML signal generation error: {e}")
            return {'ml_signal': 0.0, 'ml_confidence': 0.0}
    
    def _generate_dl_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Tạo tín hiệu từ Deep Learning models"""
        try:
            # Placeholder for DL signal generation
            # In reality, this would use the trained neural networks
            dl_prediction = np.random.uniform(-1, 1)  # Simulated DL prediction
            confidence = np.random.uniform(0.6, 0.98)
            
            return {
                'dl_signal': dl_prediction,
                'dl_confidence': confidence
            }
        except Exception as e:
            logger.warning(f"DL signal generation error: {e}")
            return {'dl_signal': 0.0, 'dl_confidence': 0.0}
    
    def _apply_risk_adjustment(self, signals: Dict[str, float], data: pd.DataFrame) -> Dict[str, float]:
        """Áp dụng risk adjustment cho signals"""
        try:
            # Calculate volatility adjustment
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            vol_adjustment = max(0.5, min(1.5, 1.0 / (volatility * 100))) if not np.isnan(volatility) else 1.0
            
            # Apply risk adjustment to all signals
            risk_adjusted = {}
            for key, value in signals.items():
                if 'signal' in key:
                    risk_adjusted[f'{key}_risk_adj'] = value * vol_adjustment
            
            risk_adjusted['volatility'] = volatility
            risk_adjusted['vol_adjustment'] = vol_adjustment
            
            return risk_adjusted
        except Exception as e:
            logger.warning(f"Risk adjustment error: {e}")
            return {}
    
    def _create_ensemble_signal(self, signals: Dict[str, float]) -> float:
        """Tạo ensemble signal từ tất cả signals"""
        try:
            # Default weights
            weights = {
                'technical': 0.3,
                'ml_signal': 0.35,
                'dl_signal': 0.35
            }
            
            # Override with config weights if available
            if self.config.ensemble_weights:
                weights.update(self.config.ensemble_weights)
            
            # Calculate weighted ensemble
            ensemble_signal = 0.0
            total_weight = 0.0
            
            for signal_name, weight in weights.items():
                if signal_name in signals:
                    ensemble_signal += signals[signal_name] * weight
                    total_weight += weight
            
            # Normalize
            if total_weight > 0:
                ensemble_signal /= total_weight
            
            # Apply confidence threshold
            confidence = signals.get('ml_confidence', 0.5) * signals.get('dl_confidence', 0.5)
            if confidence < self.config.ai_confidence_threshold:
                ensemble_signal *= confidence / self.config.ai_confidence_threshold
            
            return np.clip(ensemble_signal, -1.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Ensemble signal creation error: {e}")
            return 0.0

class PortfolioManager:
    """Quản lý portfolio trong backtesting"""
    
    def __init__(self, config: BacktestingConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions = {}  # symbol -> quantity
        self.portfolio_history = []
        self.trade_history = []
        
    def execute_trade(self, symbol: str, signal: float, price: float, 
                     timestamp: datetime, confidence: float = 0.5,
                     signal_source: str = "ENSEMBLE") -> Optional[TradeResult]:
        """Thực hiện giao dịch dựa trên signal"""
        
        try:
            # Determine action based on signal
            if abs(signal) < 0.1:  # Threshold for no action
                action = "HOLD"
                quantity = 0
            elif signal > 0:
                action = "BUY"
                # Calculate position size based on signal strength and confidence
                target_value = self.get_total_value(price) * min(signal * confidence * self.config.max_position_size, self.config.max_position_size)
                quantity = target_value / price
            else:
                action = "SELL"
                # Sell all or partial position
                current_position = self.positions.get(symbol, 0)
                if current_position > 0:
                    quantity = min(abs(signal) * current_position, current_position)
                else:
                    quantity = 0
            
            if quantity == 0:
                return None
            
            # Calculate costs
            trade_value = quantity * price
            commission = trade_value * self.config.transaction_cost
            slippage_cost = trade_value * self.config.slippage
            total_cost = commission + slippage_cost
            
            # Check if we have enough cash for buy orders
            if action == "BUY":
                total_required = trade_value + total_cost
                if total_required > self.cash:
                    # Adjust quantity to available cash
                    available_for_trade = self.cash * 0.95  # Keep 5% cash buffer
                    quantity = available_for_trade / (price * (1 + self.config.transaction_cost + self.config.slippage))
                    trade_value = quantity * price
                    commission = trade_value * self.config.transaction_cost
                    slippage_cost = trade_value * self.config.slippage
                    total_cost = commission + slippage_cost
                
                if quantity < 0.001:  # Minimum trade size
                    return None
            
            # Execute trade
            if action == "BUY":
                self.cash -= (trade_value + total_cost)
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            elif action == "SELL":
                self.cash += (trade_value - total_cost)
                self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
            
            # Calculate portfolio metrics
            portfolio_value = self.get_total_value(price)
            position_size = (quantity * price) / portfolio_value if portfolio_value > 0 else 0
            
            # Create trade result
            trade_result = TradeResult(
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                value=trade_value,
                commission=commission,
                slippage_cost=slippage_cost,
                portfolio_value=portfolio_value,
                position_size=position_size,
                signal_source=signal_source,
                confidence=confidence
            )
            
            self.trade_history.append(trade_result)
            return trade_result
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def get_total_value(self, current_price: float) -> float:
        """Tính tổng giá trị portfolio"""
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            total_value += quantity * current_price  # Assuming single asset for now
        return total_value
    
    def create_snapshot(self, timestamp: datetime, current_price: float, 
                       benchmark_return: float = 0.0) -> PortfolioSnapshot:
        """Tạo snapshot của portfolio"""
        
        total_value = self.get_total_value(current_price)
        
        # Calculate positions and weights
        positions = dict(self.positions)
        weights = {}
        for symbol, quantity in positions.items():
            weights[symbol] = (quantity * current_price) / total_value if total_value > 0 else 0
        
        # Calculate returns
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1].total_value
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            daily_return = 0
        
        cumulative_return = (total_value - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate drawdown
        if len(self.portfolio_history) > 0:
            peak_value = max([snap.total_value for snap in self.portfolio_history] + [total_value])
            drawdown = (peak_value - total_value) / peak_value if peak_value > 0 else 0
        else:
            drawdown = 0
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=self.cash,
            positions=positions,
            weights=weights,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
            risk_metrics=risk_metrics
        )
        
        self.portfolio_history.append(snapshot)
        return snapshot
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Tính toán các chỉ số rủi ro"""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = [snap.daily_return for snap in self.portfolio_history[-30:]]  # Last 30 days
        
        try:
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0  # Annualized
            sharpe = (np.mean(returns) * 252 - self.config.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Downside deviation
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 1 else 0
            sortino = (np.mean(returns) * 252 - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'var_95': np.percentile(returns, 5) if len(returns) > 1 else 0
            }
        except:
            return {}

class PerformanceAnalyzer:
    """Phân tích hiệu suất backtesting"""
    
    def __init__(self, config: BacktestingConfig):
        self.config = config
    
    def analyze_performance(self, portfolio_history: List[PortfolioSnapshot],
                          trade_history: List[TradeResult]) -> Dict[PerformanceMetric, float]:
        """Phân tích toàn diện hiệu suất"""
        
        if not portfolio_history:
            return {}
        
        # Calculate returns
        returns = [snap.daily_return for snap in portfolio_history[1:]]
        
        # Basic metrics
        total_return = portfolio_history[-1].cumulative_return
        
        # Calculate annualized return
        days = len(portfolio_history)
        years = days / 252 if days > 0 else 1
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 1 else 0
            sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Drawdown metrics
        max_drawdown = max([snap.drawdown for snap in portfolio_history]) if portfolio_history else 0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade metrics
        if trade_history:
            winning_trades = len([t for t in trade_history if t.action in ["SELL"] and t.value > 0])
            total_trades = len([t for t in trade_history if t.action != "HOLD"])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            profits = sum([t.value for t in trade_history if t.action == "SELL" and t.value > 0])
            losses = abs(sum([t.value for t in trade_history if t.action == "SELL" and t.value < 0]))
            profit_factor = profits / losses if losses > 0 else float('inf') if profits > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            PerformanceMetric.TOTAL_RETURN: total_return,
            PerformanceMetric.ANNUALIZED_RETURN: annualized_return,
            PerformanceMetric.SHARPE_RATIO: sharpe_ratio,
            PerformanceMetric.SORTINO_RATIO: sortino_ratio,
            PerformanceMetric.MAX_DRAWDOWN: max_drawdown,
            PerformanceMetric.CALMAR_RATIO: calmar_ratio,
            PerformanceMetric.WIN_RATE: win_rate,
            PerformanceMetric.PROFIT_FACTOR: profit_factor
        }
    
    def compare_with_benchmark(self, portfolio_history: List[PortfolioSnapshot],
                             benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """So sánh với benchmark"""
        
        if not portfolio_history or benchmark_data.empty:
            return {}
        
        # Calculate portfolio returns
        portfolio_returns = [snap.daily_return for snap in portfolio_history[1:]]
        
        # Calculate benchmark returns
        benchmark_returns = benchmark_data['close'].pct_change().dropna().tolist()
        
        # Align lengths
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        if min_length < 2:
            return {}
        
        try:
            # Calculate beta and alpha
            portfolio_var = np.var(portfolio_returns)
            benchmark_var = np.var(benchmark_returns)
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            
            beta = covariance / benchmark_var if benchmark_var > 0 else 0
            
            portfolio_mean_return = np.mean(portfolio_returns) * 252
            benchmark_mean_return = np.mean(benchmark_returns) * 252
            alpha = portfolio_mean_return - (self.config.risk_free_rate + beta * (benchmark_mean_return - self.config.risk_free_rate))
            
            # Information ratio
            excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
            tracking_error = np.std(excess_returns) * np.sqrt(252)
            information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
            
            return {
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0][1]
            }
        except:
            return {}

class AdvancedPortfolioBacktesting:
    """
    Main class cho Advanced Portfolio Backtesting system
    Tích hợp AI, ML, Deep Learning và advanced analytics
    """
    
    def __init__(self, config: BacktestingConfig):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.portfolio_manager = PortfolioManager(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        
    def run_backtest(self, data: pd.DataFrame, 
                    benchmark_data: Optional[pd.DataFrame] = None) -> BacktestingResult:
        """Chạy backtesting hoàn chỉnh"""
        
        start_time = datetime.now()
        logger.info(f"Starting advanced portfolio backtesting from {self.config.start_date} to {self.config.end_date}")
        
        # Filter data by date range
        mask = (data.index >= self.config.start_date) & (data.index <= self.config.end_date)
        backtest_data = data.loc[mask].copy()
        
        if backtest_data.empty:
            raise ValueError("No data available for the specified date range")
        
        # Ensure required columns
        if not all(col in backtest_data.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Data must contain OHLC columns")
        
        # Add volume if missing
        if 'volume' not in backtest_data.columns:
            backtest_data['volume'] = 1000
        
        # Main backtesting loop
        for i, (timestamp, row) in enumerate(backtest_data.iterrows()):
            try:
                # Get historical data up to current point
                historical_data = backtest_data.iloc[:i+1]
                
                if len(historical_data) < 20:  # Need minimum data for indicators
                    continue
                
                # Generate signals
                signals = self.signal_generator.generate_signals(historical_data, timestamp)
                
                # Get ensemble signal
                ensemble_signal = signals.get('ensemble', 0.0)
                confidence = signals.get('ml_confidence', 0.5) * signals.get('dl_confidence', 0.5)
                
                # Execute trade based on ensemble signal
                trade_result = self.portfolio_manager.execute_trade(
                    symbol="XAU",
                    signal=ensemble_signal,
                    price=row['close'],
                    timestamp=timestamp,
                    confidence=confidence,
                    signal_source="ENSEMBLE"
                )
                
                # Create portfolio snapshot
                benchmark_return = 0.0
                if benchmark_data is not None and timestamp in benchmark_data.index:
                    if i > 0:
                        prev_benchmark = benchmark_data.loc[backtest_data.index[i-1], 'close']
                        current_benchmark = benchmark_data.loc[timestamp, 'close']
                        benchmark_return = (current_benchmark - prev_benchmark) / prev_benchmark
                
                snapshot = self.portfolio_manager.create_snapshot(
                    timestamp=timestamp,
                    current_price=row['close'],
                    benchmark_return=benchmark_return
                )
                
                # Log progress
                if i % 100 == 0:
                    logger.info(f"Processed {i+1}/{len(backtest_data)} days, Portfolio Value: ${snapshot.total_value:,.2f}")
                    
            except Exception as e:
                logger.warning(f"Error processing {timestamp}: {e}")
                continue
        
        # Calculate performance metrics
        performance_metrics = self.performance_analyzer.analyze_performance(
            self.portfolio_manager.portfolio_history,
            self.portfolio_manager.trade_history
        )
        
        # Benchmark comparison
        benchmark_comparison = {}
        if benchmark_data is not None:
            benchmark_comparison = self.performance_analyzer.compare_with_benchmark(
                self.portfolio_manager.portfolio_history,
                benchmark_data
            )
        
        # AI performance analysis
        ai_performance = self._analyze_ai_performance()
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = BacktestingResult(
            config=self.config,
            portfolio_history=self.portfolio_manager.portfolio_history,
            trade_history=self.portfolio_manager.trade_history,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            benchmark_comparison=benchmark_comparison,
            ai_performance=ai_performance,
            execution_time=execution_time,
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades']
        )
        
        logger.info(f"Backtesting completed in {execution_time:.2f}s")
        return result
    
    def _analyze_ai_performance(self) -> Dict[str, float]:
        """Phân tích hiệu suất AI components"""
        
        ai_trades = [t for t in self.portfolio_manager.trade_history if 'ML' in t.signal_source or 'DL' in t.signal_source or 'ENSEMBLE' in t.signal_source]
        
        if not ai_trades:
            return {}
        
        # AI signal accuracy
        profitable_ai_trades = len([t for t in ai_trades if t.action == "SELL" and t.value > 0])
        ai_accuracy = profitable_ai_trades / len(ai_trades) if ai_trades else 0
        
        # Average confidence
        avg_confidence = np.mean([t.confidence for t in ai_trades])
        
        # High confidence trades performance
        high_conf_trades = [t for t in ai_trades if t.confidence > 0.8]
        high_conf_accuracy = len([t for t in high_conf_trades if t.action == "SELL" and t.value > 0]) / len(high_conf_trades) if high_conf_trades else 0
        
        return {
            'ai_trade_count': len(ai_trades),
            'ai_accuracy': ai_accuracy,
            'average_confidence': avg_confidence,
            'high_confidence_accuracy': high_conf_accuracy,
            'ai_trade_ratio': len(ai_trades) / len(self.portfolio_manager.trade_history) if self.portfolio_manager.trade_history else 0
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Tính toán risk metrics chi tiết"""
        
        if len(self.portfolio_manager.portfolio_history) < 2:
            return {}
        
        returns = [snap.daily_return for snap in self.portfolio_manager.portfolio_history[1:]]
        
        try:
            # VaR calculations
            var_95 = np.percentile(returns, 5) if returns else 0
            var_99 = np.percentile(returns, 1) if returns else 0
            
            # Expected Shortfall (CVaR)
            var_95_threshold = var_95
            tail_losses = [r for r in returns if r <= var_95_threshold]
            cvar_95 = np.mean(tail_losses) if tail_losses else 0
            
            # Maximum consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            for ret in returns:
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            # Ulcer Index
            drawdowns = [snap.drawdown for snap in self.portfolio_manager.portfolio_history]
            ulcer_index = np.sqrt(np.mean([dd**2 for dd in drawdowns])) if drawdowns else 0
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'max_consecutive_losses': max_consecutive_losses,
                'ulcer_index': ulcer_index,
                'daily_volatility': np.std(returns) if returns else 0,
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns)
            }
        except:
            return {}
    
    def _calculate_trade_statistics(self) -> Dict[str, int]:
        """Tính toán thống kê giao dịch"""
        
        trades = self.portfolio_manager.trade_history
        total_trades = len([t for t in trades if t.action != "HOLD"])
        winning_trades = len([t for t in trades if t.action == "SELL" and t.value > 0])
        losing_trades = total_trades - winning_trades
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
    
    def _calculate_skewness(self, returns: List[float]) -> float:
        """Tính skewness của returns"""
        if len(returns) < 3:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        skewness = np.mean([((r - mean_return) / std_return) ** 3 for r in returns])
        return skewness
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Tính kurtosis của returns"""
        if len(returns) < 4:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        kurtosis = np.mean([((r - mean_return) / std_return) ** 4 for r in returns]) - 3
        return kurtosis
    
    def generate_report(self, result: BacktestingResult) -> str:
        """Tạo báo cáo backtesting chi tiết"""
        
        report = []
        report.append("=" * 80)
        report.append("ADVANCED PORTFOLIO BACKTESTING REPORT")
        report.append("=" * 80)
        
        # Configuration
        report.append(f"\n📋 BACKTESTING CONFIGURATION:")
        report.append(f"Period: {result.config.start_date.strftime('%Y-%m-%d')} to {result.config.end_date.strftime('%Y-%m-%d')}")
        report.append(f"Initial Capital: ${result.config.initial_capital:,.2f}")
        report.append(f"Strategy: {result.config.strategy.value}")
        report.append(f"AI Enabled: {result.config.use_ai_signals}")
        report.append(f"Deep Learning: {result.config.use_deep_learning}")
        
        # Performance metrics
        report.append(f"\n📊 PERFORMANCE METRICS:")
        for metric, value in result.performance_metrics.items():
            if metric in [PerformanceMetric.TOTAL_RETURN, PerformanceMetric.ANNUALIZED_RETURN]:
                report.append(f"{metric.value}: {value:.2%}")
            elif metric in [PerformanceMetric.SHARPE_RATIO, PerformanceMetric.SORTINO_RATIO, PerformanceMetric.CALMAR_RATIO]:
                report.append(f"{metric.value}: {value:.3f}")
            elif metric == PerformanceMetric.MAX_DRAWDOWN:
                report.append(f"{metric.value}: {value:.2%}")
            elif metric == PerformanceMetric.WIN_RATE:
                report.append(f"{metric.value}: {value:.1%}")
            else:
                report.append(f"{metric.value}: {value:.3f}")
        
        # Trade statistics
        report.append(f"\n📈 TRADE STATISTICS:")
        report.append(f"Total Trades: {result.total_trades}")
        report.append(f"Winning Trades: {result.winning_trades}")
        report.append(f"Losing Trades: {result.losing_trades}")
        if result.total_trades > 0:
            report.append(f"Win Rate: {result.winning_trades/result.total_trades:.1%}")
        
        # AI Performance
        if result.ai_performance:
            report.append(f"\n🤖 AI PERFORMANCE:")
            for key, value in result.ai_performance.items():
                if 'accuracy' in key or 'ratio' in key:
                    report.append(f"{key}: {value:.1%}")
                else:
                    report.append(f"{key}: {value:.3f}")
        
        # Risk metrics
        if result.risk_metrics:
            report.append(f"\n⚠️  RISK METRICS:")
            for key, value in result.risk_metrics.items():
                if 'var' in key.lower() or 'volatility' in key.lower():
                    report.append(f"{key}: {value:.3%}")
                else:
                    report.append(f"{key}: {value:.3f}")
        
        # Benchmark comparison
        if result.benchmark_comparison:
            report.append(f"\n📊 BENCHMARK COMPARISON:")
            for key, value in result.benchmark_comparison.items():
                if key in ['alpha', 'tracking_error']:
                    report.append(f"{key}: {value:.3%}")
                else:
                    report.append(f"{key}: {value:.3f}")
        
        # Final portfolio value
        if result.portfolio_history:
            final_value = result.portfolio_history[-1].total_value
            report.append(f"\n💰 FINAL RESULTS:")
            report.append(f"Final Portfolio Value: ${final_value:,.2f}")
            report.append(f"Total Return: ${final_value - result.config.initial_capital:,.2f}")
            report.append(f"Execution Time: {result.execution_time:.2f} seconds")
        
        report.append("=" * 80)
        
        return "\n".join(report)

# Utility functions

def create_advanced_portfolio_backtesting(config: BacktestingConfig) -> AdvancedPortfolioBacktesting:
    """Factory function để tạo AdvancedPortfolioBacktesting instance"""
    return AdvancedPortfolioBacktesting(config)

def create_default_config(start_date: datetime, end_date: datetime) -> BacktestingConfig:
    """Tạo config mặc định cho backtesting"""
    return BacktestingConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        strategy=BacktestingStrategy.ENSEMBLE_AI,
        use_ai_signals=True,
        use_deep_learning=True,
        ai_confidence_threshold=0.6,
        ensemble_weights={
            'technical': 0.3,
            'ml_signal': 0.35,
            'dl_signal': 0.35
        }
    )

def analyze_multiple_strategies(data: pd.DataFrame, strategies: List[BacktestingStrategy],
                              start_date: datetime, end_date: datetime) -> Dict[str, BacktestingResult]:
    """So sánh nhiều strategies"""
    
    results = {}
    
    for strategy in strategies:
        config = create_default_config(start_date, end_date)
        config.strategy = strategy
        
        backtester = create_advanced_portfolio_backtesting(config)
        result = backtester.run_backtest(data)
        results[strategy.value] = result
    
    return results

# Export all classes and functions
__all__ = [
    'BacktestingStrategy', 'PerformanceMetric', 'RebalanceFrequency',
    'BacktestingConfig', 'TradeResult', 'PortfolioSnapshot', 'BacktestingResult',
    'SignalGenerator', 'PortfolioManager', 'PerformanceAnalyzer',
    'AdvancedPortfolioBacktesting', 'create_advanced_portfolio_backtesting',
    'create_default_config', 'analyze_multiple_strategies'
]

if __name__ == "__main__":
    # Test the system
    print("🚀 Ultimate XAU Super System V4.0 - Advanced Portfolio Backtesting")
    print("=" * 70)
    
    # Create sample configuration
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    config = create_default_config(start_date, end_date)
    
    # Create sample data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(1900, 2100, len(dates)),
        'high': np.random.uniform(1920, 2120, len(dates)),
        'low': np.random.uniform(1880, 2080, len(dates)),
        'close': np.random.uniform(1900, 2100, len(dates)),
        'volume': np.random.uniform(1000, 5000, len(dates))
    }, index=dates)
    
    # Test backtesting
    backtester = create_advanced_portfolio_backtesting(config)
    print("✅ Advanced Portfolio Backtesting system đã sẵn sàng!")
    print(f"📊 Configuration: {config.strategy.value} strategy với AI/ML integration") 