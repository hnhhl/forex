"""
VaR Risk Specialist
================================================================================
Risk Specialist chuyên về Value at Risk (VaR) và Risk Management
Thuộc Risk Category trong Multi-Perspective Ensemble System
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats

from .base_specialist import BaseSpecialist, SpecialistVote

logger = logging.getLogger(__name__)


class VaRRiskSpecialist(BaseSpecialist):
    """VaR Risk Specialist - Chuyên gia phân tích Value at Risk"""
    
    def __init__(self, confidence_level: float = 0.95, lookback_period: int = 30, max_var_threshold: float = 0.05):
        super().__init__(
            name="VaR_Risk_Specialist",
            category="Risk",
            description=f"VaR analysis với confidence={confidence_level}, lookback={lookback_period}"
        )
        
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.max_var_threshold = max_var_threshold  # 5% daily VaR limit
        self.min_data_points = max(20, lookback_period)
        
        logger.info(f"VaR Risk Specialist initialized: confidence={confidence_level}, lookback={lookback_period}")
    
    def calculate_historical_var(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Historical VaR and CVaR"""
        
        try:
            if len(returns) < 10:
                self.logger.warning("Insufficient data for Historical VaR calculation")
                return 0.0, 0.0
            
            # Sort returns in ascending order
            sorted_returns = returns.sort_values()
            
            # Calculate VaR (percentile)
            var_percentile = (1 - self.confidence_level) * 100
            var_value = np.percentile(sorted_returns, var_percentile)
            
            # Calculate CVaR (Expected Shortfall)
            cvar_returns = sorted_returns[sorted_returns <= var_value]
            cvar_value = cvar_returns.mean() if len(cvar_returns) > 0 else var_value
            
            return abs(var_value), abs(cvar_value)
            
        except Exception as e:
            self.logger.error(f"Error calculating Historical VaR: {e}")
            return 0.0, 0.0
    
    def calculate_parametric_var(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Parametric VaR assuming normal distribution"""
        
        try:
            if len(returns) < 10:
                return 0.0, 0.0
            
            # Calculate mean and standard deviation
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Z-score for confidence level
            z_score = stats.norm.ppf(1 - self.confidence_level)
            
            # Parametric VaR
            var_value = abs(mean_return + z_score * std_return)
            
            # Parametric CVaR (for normal distribution)
            density_at_var = stats.norm.pdf(z_score)
            cvar_value = abs(mean_return - std_return * density_at_var / (1 - self.confidence_level))
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Error calculating Parametric VaR: {e}")
            return 0.0, 0.0
    
    def calculate_drawdown_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        
        try:
            if len(prices) < 2:
                return {'current_drawdown': 0.0, 'max_drawdown': 0.0}
            
            # Calculate returns and cumulative returns
            returns = prices.pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            current_drawdown = drawdowns.iloc[-1]
            max_drawdown = drawdowns.min()
            
            return {
                'current_drawdown': current_drawdown,
                'max_drawdown': max_drawdown,
                'drawdown_duration': self._calculate_drawdown_duration(drawdowns)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
            return {'current_drawdown': 0.0, 'max_drawdown': 0.0}
    
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """Calculate current drawdown duration"""
        
        try:
            # Find last peak (drawdown = 0)
            last_peak_idx = None
            for i in range(len(drawdowns) - 1, -1, -1):
                if drawdowns.iloc[i] >= -0.001:  # Near zero (allowing for small numerical errors)
                    last_peak_idx = i
                    break
            
            if last_peak_idx is not None:
                return len(drawdowns) - 1 - last_peak_idx
            else:
                return len(drawdowns)
                
        except Exception as e:
            self.logger.error(f"Error calculating drawdown duration: {e}")
            return 0
    
    def calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility-based risk metrics"""
        
        try:
            if len(returns) < 2:
                return {'volatility': 0.0, 'annualized_volatility': 0.0}
            
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days
            
            # Downside deviation (volatility of negative returns)
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() if len(negative_returns) > 0 else daily_volatility
            annualized_downside_deviation = downside_deviation * np.sqrt(252)
            
            return {
                'volatility': daily_volatility,
                'annualized_volatility': annualized_volatility,
                'downside_deviation': downside_deviation,
                'annualized_downside_deviation': annualized_downside_deviation
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            return {'volatility': 0.0, 'annualized_volatility': 0.0}
    
    def analyze(self, data: pd.DataFrame, current_price: float, **kwargs) -> SpecialistVote:
        """Analyze VaR và risk metrics để generate vote"""
        
        if not self.enabled:
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning="Specialist is disabled",
                timestamp=datetime.now(),
                technical_data={}
            )
        
        if not self.validate_data(data):
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning="Invalid or insufficient data",
                timestamp=datetime.now(),
                technical_data={}
            )
        
        try:
            # Calculate returns
            prices = data['close']
            returns = prices.pct_change().dropna()
            
            if len(returns) < self.min_data_points:
                return SpecialistVote(
                    specialist_name=self.name,
                    vote="HOLD",
                    confidence=0.0,
                    reasoning="Insufficient data for VaR calculation",
                    timestamp=datetime.now(),
                    technical_data={}
                )
            
            # Use recent data for VaR calculation
            recent_returns = returns.tail(self.lookback_period)
            
            # Calculate VaR metrics
            historical_var, historical_cvar = self.calculate_historical_var(recent_returns)
            parametric_var, parametric_cvar = self.calculate_parametric_var(recent_returns)
            
            # Calculate drawdown metrics
            drawdown_metrics = self.calculate_drawdown_metrics(prices.tail(self.lookback_period))
            
            # Calculate volatility metrics
            volatility_metrics = self.calculate_volatility_metrics(recent_returns)
            
            # Combine all metrics
            analysis_result = {
                'current_price': current_price,
                'historical_var': historical_var,
                'historical_cvar': historical_cvar,
                'parametric_var': parametric_var,
                'parametric_cvar': parametric_cvar,
                'current_drawdown': drawdown_metrics['current_drawdown'],
                'max_drawdown': drawdown_metrics['max_drawdown'],
                'volatility': volatility_metrics['volatility'],
                'annualized_volatility': volatility_metrics['annualized_volatility']
            }
            
            # Generate vote based on risk assessment
            vote = self.generate_risk_vote(analysis_result)
            confidence = self.calculate_confidence(analysis_result)
            reasoning = self.generate_reasoning(analysis_result, vote)
            
            specialist_vote = SpecialistVote(
                specialist_name=self.name,
                vote=vote,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                technical_data=analysis_result
            )
            
            self.vote_history.append({
                'vote': vote,
                'confidence': confidence,
                'var_95': historical_var,
                'max_drawdown': drawdown_metrics['max_drawdown'],
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"VaR Risk Analysis: VaR={historical_var:.4f}, Vote={vote}, Confidence={confidence:.2f}")
            
            return specialist_vote
            
        except Exception as e:
            self.logger.error(f"Error in VaR risk analysis: {e}")
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                timestamp=datetime.now(),
                technical_data={}
            )
    
    def generate_risk_vote(self, analysis_result: Dict[str, Any]) -> str:
        """Generate vote based on risk assessment"""
        
        historical_var = analysis_result['historical_var']
        current_drawdown = abs(analysis_result['current_drawdown'])
        max_drawdown = abs(analysis_result['max_drawdown'])
        volatility = analysis_result['annualized_volatility']
        
        # Risk-based voting logic
        risk_score = 0
        
        # VaR risk assessment
        if historical_var > self.max_var_threshold:
            risk_score += 3  # High risk
        elif historical_var > self.max_var_threshold * 0.7:
            risk_score += 2  # Medium risk
        elif historical_var > self.max_var_threshold * 0.5:
            risk_score += 1  # Low risk
        
        # Drawdown risk assessment
        if current_drawdown > 0.1:  # 10%+ current drawdown
            risk_score += 2
        elif current_drawdown > 0.05:  # 5%+ current drawdown
            risk_score += 1
        
        if max_drawdown < -0.15:  # 15%+ historical max drawdown
            risk_score += 2
        elif max_drawdown < -0.10:  # 10%+ historical max drawdown
            risk_score += 1
        
        # Volatility risk assessment
        if volatility > 0.4:  # 40%+ annualized volatility
            risk_score += 2
        elif volatility > 0.25:  # 25%+ annualized volatility
            risk_score += 1
        
        # Vote based on total risk score
        if risk_score >= 6:
            return "SELL"  # High risk - reduce exposure
        elif risk_score >= 3:
            return "HOLD"  # Medium risk - maintain position
        else:
            return "BUY"   # Low risk - can increase exposure
    
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on risk metrics reliability"""
        
        base_confidence = 0.7  # Base confidence for risk analysis
        
        # Adjust confidence based on data quality
        historical_var = analysis_result['historical_var']
        parametric_var = analysis_result['parametric_var']
        
        # If historical and parametric VaR are similar, increase confidence
        if historical_var > 0 and parametric_var > 0:
            var_similarity = 1 - abs(historical_var - parametric_var) / max(historical_var, parametric_var)
            base_confidence += var_similarity * 0.2
        
        # Adjust for recent accuracy
        recent_accuracy = self.get_recent_accuracy()
        if recent_accuracy > 0.6:
            base_confidence *= 1.1
        elif recent_accuracy < 0.4:
            base_confidence *= 0.8
        
        return min(1.0, max(0.1, base_confidence))
    
    def generate_reasoning(self, analysis_result: Dict[str, Any], vote: str) -> str:
        """Generate reasoning for the risk assessment"""
        
        historical_var = analysis_result['historical_var']
        current_drawdown = analysis_result['current_drawdown']
        max_drawdown = analysis_result['max_drawdown']
        volatility = analysis_result['annualized_volatility']
        
        reasoning = f"VaR Risk Analysis: "
        reasoning += f"VaR {self.confidence_level:.0%} = {historical_var:.3f} ({historical_var/self.max_var_threshold:.1f}x limit), "
        reasoning += f"Current DD = {current_drawdown:.2%}, Max DD = {max_drawdown:.2%}, "
        reasoning += f"Volatility = {volatility:.1%}"
        
        vote_reasoning = {
            "BUY": "Suggesting BUY - low risk environment, acceptable VaR levels",
            "SELL": "Suggesting SELL - high risk detected, VaR limits breached",
            "HOLD": "Suggesting HOLD - moderate risk levels, cautious approach"
        }
        
        return f"{reasoning}. {vote_reasoning.get(vote, '')}"


def create_var_risk_specialist(confidence: float = 0.95, lookback: int = 30, threshold: float = 0.05) -> VaRRiskSpecialist:
    """Factory function to create VaR Risk Specialist"""
    return VaRRiskSpecialist(confidence_level=confidence, lookback_period=lookback, max_var_threshold=threshold) 