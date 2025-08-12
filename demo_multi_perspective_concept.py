"""
Demo Concept: Multi-Perspective Ensemble System
Minh họa cách 18 specialists tạo ra signal consensus
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class SpecialistSignal:
    def __init__(self, name: str, signal: str, confidence: float, reasoning: str):
        self.name = name
        self.signal = signal  # BUY, SELL, HOLD
        self.confidence = confidence  # 0.0 - 1.0
        self.reasoning = reasoning
        self.timestamp = datetime.now()

class MultiPerspectiveEnsemble:
    def __init__(self):
        self.specialists = {
            'technical': ['RSI_Specialist', 'MACD_Specialist', 'Fibonacci_Specialist'],
            'sentiment': ['News_Specialist', 'Social_Specialist', 'Fear_Greed_Specialist'],
            'pattern': ['Chart_Pattern_Specialist', 'Candlestick_Specialist', 'Wave_Specialist'],
            'risk': ['VaR_Specialist', 'Drawdown_Specialist', 'Position_Size_Specialist'],
            'momentum': ['Trend_Specialist', 'Mean_Reversion_Specialist', 'Breakout_Specialist'],
            'volatility': ['ATR_Specialist', 'Bollinger_Specialist', 'Vol_Clustering_Specialist']
        }
        
        # Dynamic weights based on market conditions
        self.base_weights = {
            'technical': 0.25,
            'sentiment': 0.15,
            'pattern': 0.20,
            'risk': 0.15,
            'momentum': 0.15,
            'volatility': 0.10
        }
    
    def simulate_specialist_signals(self, market_data: Dict) -> List[SpecialistSignal]:
        """Simulate signals from all 18 specialists"""
        signals = []
        
        # Technical Specialists
        signals.extend([
            SpecialistSignal("RSI_Specialist", "BUY", 0.72, "RSI=28, oversold condition"),
            SpecialistSignal("MACD_Specialist", "BUY", 0.68, "MACD bullish crossover"),
            SpecialistSignal("Fibonacci_Specialist", "HOLD", 0.55, "Price at 50% retracement")
        ])
        
        # Sentiment Specialists  
        signals.extend([
            SpecialistSignal("News_Specialist", "BUY", 0.78, "Fed dovish comments, USD weakness"),
            SpecialistSignal("Social_Specialist", "BUY", 0.65, "Positive gold sentiment on Twitter"),
            SpecialistSignal("Fear_Greed_Specialist", "BUY", 0.70, "Fear index high, safe haven demand")
        ])
        
        # Pattern Specialists
        signals.extend([
            SpecialistSignal("Chart_Pattern_Specialist", "BUY", 0.75, "Ascending triangle breakout"),
            SpecialistSignal("Candlestick_Specialist", "BUY", 0.82, "Hammer pattern at support"),
            SpecialistSignal("Wave_Specialist", "HOLD", 0.60, "Wave 3 completion expected")
        ])
        
        # Risk Specialists
        signals.extend([
            SpecialistSignal("VaR_Specialist", "BUY", 0.58, "1% VaR acceptable for position"),
            SpecialistSignal("Drawdown_Specialist", "HOLD", 0.45, "High drawdown risk detected"),
            SpecialistSignal("Position_Size_Specialist", "BUY", 0.63, "Optimal size: 2% of portfolio")
        ])
        
        # Momentum Specialists
        signals.extend([
            SpecialistSignal("Trend_Specialist", "BUY", 0.77, "Strong uptrend confirmed"),
            SpecialistSignal("Mean_Reversion_Specialist", "SELL", 0.52, "Price extended from mean"),
            SpecialistSignal("Breakout_Specialist", "BUY", 0.85, "Volume confirmed breakout")
        ])
        
        # Volatility Specialists
        signals.extend([
            SpecialistSignal("ATR_Specialist", "BUY", 0.66, "ATR expansion suggests move"),
            SpecialistSignal("Bollinger_Specialist", "BUY", 0.71, "Price bounced off lower band"),
            SpecialistSignal("Vol_Clustering_Specialist", "HOLD", 0.48, "Low volatility period")
        ])
        
        return signals
    
    def calculate_consensus(self, signals: List[SpecialistSignal]) -> Tuple[str, float, Dict]:
        """Calculate consensus from all specialist signals"""
        
        # Count votes by category
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_sum = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        category_votes = {category: {'BUY': 0, 'SELL': 0, 'HOLD': 0} for category in self.base_weights.keys()}
        
        for signal in signals:
            votes[signal.signal] += 1
            confidence_sum[signal.signal] += signal.confidence
            
            # Determine category
            for category, specialists in self.specialists.items():
                if signal.name in specialists:
                    category_votes[category][signal.signal] += 1
                    break
        
        # Calculate weighted consensus
        weighted_score = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for category, weight in self.base_weights.items():
            category_total = sum(category_votes[category].values())
            if category_total > 0:
                for signal_type in ['BUY', 'SELL', 'HOLD']:
                    category_strength = category_votes[category][signal_type] / category_total
                    weighted_score[signal_type] += weight * category_strength
        
        # Determine final signal
        final_signal = max(weighted_score, key=weighted_score.get)
        
        # Calculate confidence
        total_votes = sum(votes.values())
        signal_votes = votes[final_signal]
        agreement_ratio = signal_votes / total_votes
        avg_confidence = confidence_sum[final_signal] / max(signal_votes, 1)
        
        final_confidence = (agreement_ratio * 0.6 + avg_confidence * 0.4) * 100
        
        # Detailed breakdown
        breakdown = {
            'total_specialists': len(signals),
            'votes': votes,
            'agreement_ratio': agreement_ratio,
            'avg_specialist_confidence': avg_confidence,
            'category_breakdown': category_votes,
            'weighted_scores': weighted_score
        }
        
        return final_signal, final_confidence, breakdown

def demo_multi_perspective_system():
    """Demo the Multi-Perspective Ensemble System"""
    print("🎯 MULTI-PERSPECTIVE ENSEMBLE SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize system
    ensemble = MultiPerspectiveEnsemble()
    
    # Simulate market data
    market_data = {
        'price': 1945.50,
        'volume': 125000,
        'volatility': 0.015,
        'trend': 'BULLISH'
    }
    
    print(f"📊 Market Data: Gold @ ${market_data['price']}")
    print(f"📈 Trend: {market_data['trend']}, Vol: {market_data['volatility']:.3f}")
    print()
    
    # Get specialist signals
    signals = ensemble.simulate_specialist_signals(market_data)
    
    print("🤖 SPECIALIST SIGNALS:")
    print("-" * 50)
    
    categories = ['technical', 'sentiment', 'pattern', 'risk', 'momentum', 'volatility']
    for category in categories:
        print(f"\n📋 {category.upper()} SPECIALISTS:")
        for signal in signals:
            if signal.name in ensemble.specialists[category]:
                print(f"   {signal.name:20} → {signal.signal:4} ({signal.confidence:.2f}) - {signal.reasoning}")
    
    # Calculate consensus
    final_signal, confidence, breakdown = ensemble.calculate_consensus(signals)
    
    print("\n" + "=" * 50)
    print("🎯 CONSENSUS RESULT:")
    print("=" * 50)
    
    print(f"📊 Final Signal: {final_signal}")
    print(f"🎯 Confidence: {confidence:.1f}%")
    print(f"📈 Agreement: {breakdown['agreement_ratio']:.1%} ({breakdown['votes'][final_signal]}/{breakdown['total_specialists']} specialists)")
    
    print(f"\n📋 Vote Breakdown:")
    for signal_type, count in breakdown['votes'].items():
        percentage = count / breakdown['total_specialists'] * 100
        print(f"   {signal_type:4}: {count:2} votes ({percentage:4.1f}%)")
    
    print(f"\n🏆 Category Performance:")
    for category, votes in breakdown['category_breakdown'].items():
        dominant = max(votes, key=votes.get)
        print(f"   {category:12}: {dominant} dominant ({votes[dominant]}/3)")
    
    # Trading recommendation
    print(f"\n💡 TRADING RECOMMENDATION:")
    if confidence >= 70:
        print(f"   ✅ STRONG {final_signal} - High confidence trade")
    elif confidence >= 60:
        print(f"   ⚠️  MODERATE {final_signal} - Proceed with caution")
    else:
        print(f"   ❌ LOW CONFIDENCE - Consider waiting")
    
    print(f"\n📊 Risk Management:")
    print(f"   Stop Loss: ${market_data['price'] * 0.995:.2f}")
    print(f"   Take Profit: ${market_data['price'] * 1.015:.2f}")
    print(f"   Position Size: 2% of portfolio")

if __name__ == "__main__":
    demo_multi_perspective_system() 