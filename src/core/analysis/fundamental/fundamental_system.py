"""
Fundamental Analysis System
Ultimate XAU Super System V4.0 - Phase 3 Component

Comprehensive fundamental analysis for XAU trading:
- Economic indicators analysis (GDP, CPI, unemployment, etc.)
- Central bank policy analysis (Fed, ECB, BOJ, etc.)
- Geopolitical risk assessment
- Currency correlations and strength analysis
- Interest rates and inflation analysis
- Supply and demand fundamentals for gold
"""

import numpy as np
import pandas as pd
import logging
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available - using mock data")

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("FRED API not available - using mock data")

logger = logging.getLogger(__name__)


class EconomicIndicator(Enum):
    """Economic indicators relevant to gold"""
    GDP_GROWTH = "gdp_growth"
    CPI_INFLATION = "cpi_inflation"
    UNEMPLOYMENT_RATE = "unemployment_rate"
    INTEREST_RATES = "interest_rates"
    DOLLAR_INDEX = "dollar_index"
    TREASURY_YIELDS = "treasury_yields"
    MONEY_SUPPLY = "money_supply"
    DEBT_TO_GDP = "debt_to_gdp"
    REAL_INTEREST_RATES = "real_interest_rates"
    COMMODITY_PRICES = "commodity_prices"


class GeopoliticalRisk(Enum):
    """Geopolitical risk factors"""
    MILITARY_CONFLICT = "military_conflict"
    TRADE_WAR = "trade_war"
    POLITICAL_INSTABILITY = "political_instability"
    SANCTIONS = "sanctions"
    CURRENCY_CRISIS = "currency_crisis"
    NATURAL_DISASTERS = "natural_disasters"


class CentralBank(Enum):
    """Major central banks"""
    FEDERAL_RESERVE = "fed"
    EUROPEAN_CENTRAL_BANK = "ecb"
    BANK_OF_JAPAN = "boj"
    BANK_OF_ENGLAND = "boe"
    PEOPLE_BANK_OF_CHINA = "pboc"


@dataclass
class EconomicData:
    """Economic data point"""
    indicator: EconomicIndicator
    timestamp: datetime
    value: float
    country: str
    source: str
    impact_on_gold: float  # -1 to 1, negative bearish, positive bullish
    confidence: float      # 0 to 1
    release_date: datetime
    previous_value: Optional[float] = None
    forecast_value: Optional[float] = None


@dataclass
class PolicyData:
    """Central bank policy data"""
    central_bank: CentralBank
    timestamp: datetime
    policy_rate: float
    policy_stance: str  # "dovish", "neutral", "hawkish"
    policy_change: float  # Change in basis points
    forward_guidance: str
    impact_on_gold: float  # -1 to 1
    confidence: float


@dataclass
class GeopoliticalData:
    """Geopolitical risk data"""
    risk_type: GeopoliticalRisk
    timestamp: datetime
    severity: float  # 0 to 1
    description: str
    affected_regions: List[str]
    impact_on_gold: float  # -1 to 1
    duration_days: Optional[int] = None


@dataclass
class FundamentalSignal:
    """Fundamental analysis signal"""
    timestamp: datetime
    signal_strength: float  # -1 to 1, negative bearish, positive bullish
    confidence: float       # 0 to 1
    components: Dict[str, float]  # Individual component contributions
    reasoning: str
    time_horizon: str  # "short", "medium", "long"
    risk_level: float  # 0 to 1


class EconomicDataProvider:
    """Provides economic data from various sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
        logger.info("EconomicDataProvider initialized")
    
    def get_economic_indicators(self) -> List[EconomicData]:
        """Get current economic indicators"""
        # In production, this would connect to real data sources
        # For now, generate realistic mock data
        
        indicators = []
        current_time = datetime.now()
        
        # US Economic Indicators
        us_indicators = [
            (EconomicIndicator.GDP_GROWTH, 2.1, 0.3),  # (indicator, value, gold_impact)
            (EconomicIndicator.CPI_INFLATION, 3.2, 0.6),
            (EconomicIndicator.UNEMPLOYMENT_RATE, 3.7, -0.2),
            (EconomicIndicator.INTEREST_RATES, 5.25, -0.8),
            (EconomicIndicator.DOLLAR_INDEX, 103.5, -0.7),
            (EconomicIndicator.TREASURY_YIELDS, 4.8, -0.6),
            (EconomicIndicator.REAL_INTEREST_RATES, 1.6, -0.9),
        ]
        
        for indicator, value, impact in us_indicators:
            data = EconomicData(
                indicator=indicator,
                timestamp=current_time,
                value=value,
                country="US",
                source="Federal Reserve/BLS",
                impact_on_gold=impact,
                confidence=0.8 + np.random.random() * 0.2,
                release_date=current_time - timedelta(days=np.random.randint(1, 30))
            )
            indicators.append(data)
        
        return indicators
    
    def get_policy_data(self) -> List[PolicyData]:
        """Get central bank policy data"""
        policies = []
        current_time = datetime.now()
        
        # Fed Policy
        fed_policy = PolicyData(
            central_bank=CentralBank.FEDERAL_RESERVE,
            timestamp=current_time,
            policy_rate=5.25,
            policy_stance="neutral",
            policy_change=0,  # No change in last meeting
            forward_guidance="Data-dependent approach, monitoring inflation",
            impact_on_gold=-0.3,  # Neutral to slightly bearish
            confidence=0.9
        )
        policies.append(fed_policy)
        
        # ECB Policy
        ecb_policy = PolicyData(
            central_bank=CentralBank.EUROPEAN_CENTRAL_BANK,
            timestamp=current_time,
            policy_rate=4.0,
            policy_stance="dovish",
            policy_change=-25,  # 25bps cut
            forward_guidance="Committed to supporting recovery",
            impact_on_gold=0.4,  # Bullish
            confidence=0.85
        )
        policies.append(ecb_policy)
        
        return policies
    
    def get_geopolitical_risks(self) -> List[GeopoliticalData]:
        """Get geopolitical risk assessment"""
        risks = []
        current_time = datetime.now()
        
        # Mock geopolitical risks
        risk_events = [
            (GeopoliticalRisk.TRADE_WAR, 0.6, "US-China trade tensions", ["US", "China"], 0.3),
            (GeopoliticalRisk.MILITARY_CONFLICT, 0.4, "Regional conflicts", ["Middle East"], 0.5),
            (GeopoliticalRisk.CURRENCY_CRISIS, 0.3, "Emerging market volatility", ["EM"], 0.2),
        ]
        
        for risk_type, severity, desc, regions, impact in risk_events:
            risk = GeopoliticalData(
                risk_type=risk_type,
                timestamp=current_time,
                severity=severity,
                description=desc,
                affected_regions=regions,
                impact_on_gold=impact,
                duration_days=np.random.randint(30, 180)
            )
            risks.append(risk)
        
        return risks


class FundamentalAnalyzer:
    """Analyzes fundamental data and generates signals"""
    
    def __init__(self):
        self.economic_weights = {
            EconomicIndicator.REAL_INTEREST_RATES: -0.9,  # Most important for gold
            EconomicIndicator.DOLLAR_INDEX: -0.8,
            EconomicIndicator.CPI_INFLATION: 0.7,
            EconomicIndicator.INTEREST_RATES: -0.6,
            EconomicIndicator.TREASURY_YIELDS: -0.5,
            EconomicIndicator.GDP_GROWTH: -0.3,
            EconomicIndicator.UNEMPLOYMENT_RATE: 0.2,
        }
        
        self.policy_weights = {
            CentralBank.FEDERAL_RESERVE: 0.6,  # Most influential for gold
            CentralBank.EUROPEAN_CENTRAL_BANK: 0.2,
            CentralBank.BANK_OF_JAPAN: 0.1,
            CentralBank.BANK_OF_ENGLAND: 0.1,
        }
        
        logger.info("FundamentalAnalyzer initialized")
    
    def analyze_economic_data(self, economic_data: List[EconomicData]) -> float:
        """Analyze economic indicators and return gold impact score"""
        if not economic_data:
            return 0.0
        
        total_impact = 0.0
        total_weight = 0.0
        
        for data in economic_data:
            if data.indicator in self.economic_weights:
                weight = self.economic_weights[data.indicator]
                impact = data.impact_on_gold * abs(weight) * data.confidence
                total_impact += impact
                total_weight += abs(weight)
        
        return total_impact / total_weight if total_weight > 0 else 0.0
    
    def analyze_policy_data(self, policy_data: List[PolicyData]) -> float:
        """Analyze central bank policies and return gold impact score"""
        if not policy_data:
            return 0.0
        
        total_impact = 0.0
        total_weight = 0.0
        
        for policy in policy_data:
            if policy.central_bank in self.policy_weights:
                weight = self.policy_weights[policy.central_bank]
                impact = policy.impact_on_gold * weight * policy.confidence
                total_impact += impact
                total_weight += weight
        
        return total_impact / total_weight if total_weight > 0 else 0.0
    
    def analyze_geopolitical_risks(self, geopolitical_data: List[GeopoliticalData]) -> float:
        """Analyze geopolitical risks and return gold impact score"""
        if not geopolitical_data:
            return 0.0
        
        # Geopolitical risks are generally bullish for gold (safe haven)
        total_impact = 0.0
        total_severity = 0.0
        
        for risk in geopolitical_data:
            # Weight by severity and expected impact
            weighted_impact = risk.impact_on_gold * risk.severity
            total_impact += weighted_impact
            total_severity += risk.severity
        
        return total_impact / len(geopolitical_data) if geopolitical_data else 0.0
    
    def calculate_dollar_strength_impact(self) -> float:
        """Calculate USD strength impact on gold"""
        try:
            if YFINANCE_AVAILABLE:
                # Get DXY (Dollar Index) data
                dxy = yf.Ticker("DX-Y.NYB")
                hist = dxy.history(period="5d")
                if not hist.empty:
                    # Calculate recent change
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[0]
                    change_pct = (current_price - prev_price) / prev_price
                    
                    # Strong dollar is bearish for gold
                    return -change_pct * 2.0  # Amplify impact
            
            # Fallback mock data
            return np.random.uniform(-0.2, 0.2)
            
        except Exception as e:
            logger.error(f"Error calculating dollar strength: {e}")
            return 0.0
    
    def calculate_inflation_expectation_impact(self) -> float:
        """Calculate inflation expectations impact"""
        try:
            # In production, would use TIPS breakeven rates
            # Mock realistic inflation expectation data
            current_inflation_expectation = 2.5
            target_inflation = 2.0
            
            # Higher inflation expectations are bullish for gold
            impact = (current_inflation_expectation - target_inflation) / target_inflation
            return min(max(impact, -0.5), 0.5)  # Cap at ±50%
            
        except Exception as e:
            logger.error(f"Error calculating inflation expectations: {e}")
            return 0.0


class FundamentalAnalysisSystem:
    """Main fundamental analysis system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.data_provider = EconomicDataProvider()
        self.analyzer = FundamentalAnalyzer()
        self.is_active = False
        self.last_update = None
        self.signal_history = []
        
        logger.info("FundamentalAnalysisSystem initialized")
    
    def initialize(self) -> bool:
        """Initialize the fundamental analysis system"""
        try:
            self.is_active = True
            self.last_update = datetime.now()
            logger.info("FundamentalAnalysisSystem started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FundamentalAnalysisSystem: {e}")
            return False
    
    def process(self, data: Any = None) -> Dict[str, Any]:
        """Process fundamental analysis and generate signal"""
        try:
            if not self.is_active:
                return {'error': 'System not active'}
            
            # Collect fundamental data
            economic_data = self.data_provider.get_economic_indicators()
            policy_data = self.data_provider.get_policy_data()
            geopolitical_data = self.data_provider.get_geopolitical_risks()
            
            # Analyze different components
            economic_impact = self.analyzer.analyze_economic_data(economic_data)
            policy_impact = self.analyzer.analyze_policy_data(policy_data)
            geopolitical_impact = self.analyzer.analyze_geopolitical_risks(geopolitical_data)
            dollar_impact = self.analyzer.calculate_dollar_strength_impact()
            inflation_impact = self.analyzer.calculate_inflation_expectation_impact()
            
            # Combine all impacts with weights
            component_weights = {
                'economic': 0.3,
                'policy': 0.25,
                'geopolitical': 0.15,
                'dollar': 0.2,
                'inflation': 0.1
            }
            
            components = {
                'economic': economic_impact,
                'policy': policy_impact,
                'geopolitical': geopolitical_impact,
                'dollar': dollar_impact,
                'inflation': inflation_impact
            }
            
            # Calculate overall signal strength
            signal_strength = sum(
                components[comp] * component_weights[comp] 
                for comp in components
            )
            
            # Calculate confidence based on data availability and consistency
            confidence = self._calculate_confidence(components)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(components)
            
            # Determine time horizon based on signal strength
            time_horizon = self._determine_time_horizon(signal_strength, confidence)
            
            # Create signal
            signal = FundamentalSignal(
                timestamp=datetime.now(),
                signal_strength=signal_strength,
                confidence=confidence,
                components=components,
                reasoning=reasoning,
                time_horizon=time_horizon,
                risk_level=self._calculate_risk_level(components)
            )
            
            # Store in history
            self.signal_history.append(signal)
            if len(self.signal_history) > 100:  # Keep last 100 signals
                self.signal_history.pop(0)
            
            self.last_update = datetime.now()
            
            return {
                'signal_strength': signal.signal_strength,
                'confidence': signal.confidence,
                'components': signal.components,
                'reasoning': signal.reasoning,
                'time_horizon': signal.time_horizon,
                'risk_level': signal.risk_level,
                'timestamp': signal.timestamp.isoformat(),
                'data_points': {
                    'economic_indicators': len(economic_data),
                    'policy_data': len(policy_data),
                    'geopolitical_risks': len(geopolitical_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence(self, components: Dict[str, float]) -> float:
        """Calculate confidence based on component availability and consistency"""
        # Base confidence on data availability
        available_components = sum(1 for v in components.values() if v != 0)
        availability_score = available_components / len(components)
        
        # Check consistency - if all components point in same direction, higher confidence
        positive_components = sum(1 for v in components.values() if v > 0.1)
        negative_components = sum(1 for v in components.values() if v < -0.1)
        
        if positive_components > 0 and negative_components == 0:
            consistency_score = 1.0  # All bullish
        elif negative_components > 0 and positive_components == 0:
            consistency_score = 1.0  # All bearish
        else:
            consistency_score = 0.5  # Mixed signals
        
        return (availability_score * 0.6 + consistency_score * 0.4)
    
    def _generate_reasoning(self, components: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the signal"""
        reasoning_parts = []
        
        for component, impact in components.items():
            if abs(impact) > 0.1:  # Only include significant impacts
                direction = "bullish" if impact > 0 else "bearish"
                strength = "strong" if abs(impact) > 0.3 else "moderate"
                reasoning_parts.append(f"{component.title()} factors are {strength}ly {direction}")
        
        if not reasoning_parts:
            return "Fundamental factors are neutral for gold"
        
        return "; ".join(reasoning_parts)
    
    def _determine_time_horizon(self, signal_strength: float, confidence: float) -> str:
        """Determine appropriate time horizon for the signal"""
        signal_magnitude = abs(signal_strength) * confidence
        
        if signal_magnitude > 0.4:
            return "long"      # Strong signal = long-term view
        elif signal_magnitude > 0.2:
            return "medium"    # Moderate signal = medium-term
        else:
            return "short"     # Weak signal = short-term only
    
    def _calculate_risk_level(self, components: Dict[str, float]) -> float:
        """Calculate risk level based on component volatility"""
        # Higher geopolitical and policy risks increase overall risk
        geopolitical_risk = abs(components.get('geopolitical', 0))
        policy_risk = abs(components.get('policy', 0))
        
        base_risk = 0.3  # Base market risk
        additional_risk = (geopolitical_risk + policy_risk) * 0.5
        
        return min(base_risk + additional_risk, 1.0)
    
    def cleanup(self) -> bool:
        """Cleanup the system"""
        try:
            self.is_active = False
            logger.info("FundamentalAnalysisSystem stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping FundamentalAnalysisSystem: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'system_name': 'FundamentalAnalysisSystem',
            'is_active': self.is_active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'signal_history_count': len(self.signal_history),
            'dependencies': {
                'yfinance': YFINANCE_AVAILABLE,
                'fred_api': FRED_AVAILABLE
            }
        }


def demo_fundamental_analysis():
    """Demo function to test the fundamental analysis system"""
    print("\n🔍 FUNDAMENTAL ANALYSIS SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize system
    system = FundamentalAnalysisSystem()
    
    if not system.initialize():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully")
    
    # Process analysis
    result = system.process()
    
    if 'error' in result:
        print(f"❌ Analysis failed: {result['error']}")
        return
    
    # Display results
    print(f"\n📊 FUNDAMENTAL ANALYSIS RESULTS")
    print(f"Signal Strength: {result['signal_strength']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Time Horizon: {result['time_horizon']}")
    print(f"Risk Level: {result['risk_level']:.3f}")
    print(f"Reasoning: {result['reasoning']}")
    
    print(f"\n🔍 COMPONENT BREAKDOWN:")
    for component, value in result['components'].items():
        direction = "↗️" if value > 0 else "↘️" if value < 0 else "➡️"
        print(f"  {component.title()}: {value:.3f} {direction}")
    
    print(f"\n📈 DATA SUMMARY:")
    for data_type, count in result['data_points'].items():
        print(f"  {data_type.replace('_', ' ').title()}: {count}")
    
    # Cleanup
    system.cleanup()
    print("\n✅ Demo completed successfully")


if __name__ == "__main__":
    demo_fundamental_analysis() 