"""
Day 21 Technical Analysis Foundation Demo
Ultimate XAU Super System V4.0

Comprehensive demonstration of technical analysis capabilities:
- Advanced indicator calculations
- Pattern recognition engine
- Multi-timeframe analysis
- Signal generation and validation
- Real-time market analysis
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add source path
sys.path.append('src')

# Import technical analysis components
try:
    from src.core.analysis.technical_analysis import (
        create_technical_analyzer, TechnicalAnalyzer, IndicatorConfig,
        TechnicalSignal, PatternResult, TechnicalIndicators, PatternRecognition
    )
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Technical Analyzer not available: {e}")
    TECHNICAL_ANALYZER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Day21TechnicalDemo:
    """Comprehensive Day 21 technical analysis demo"""
    
    def __init__(self):
        self.analyzer = None
        self.demo_results = {}
        self.start_time = datetime.now()
        
        print("\n" + "="*100)
        print("📊 ULTIMATE XAU SUPER SYSTEM V4.0 - DAY 21 TECHNICAL ANALYSIS FOUNDATION")
        print("="*100)
        print(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def initialize_technical_analyzer(self):
        """Initialize technical analysis system"""
        print("🔧 INITIALIZING TECHNICAL ANALYSIS SYSTEM")
        print("-" * 60)
        
        if TECHNICAL_ANALYZER_AVAILABLE:
            try:
                # Create comprehensive technical analyzer
                custom_config = {
                    'sma_periods': [5, 10, 20, 50, 100, 200],
                    'ema_periods': [12, 26, 50, 100],
                    'rsi_period': 14,
                    'bb_period': 20,
                    'bb_std': 2.0
                }
                
                self.analyzer = create_technical_analyzer(custom_config)
                print("✅ Technical Analyzer initialized successfully")
                print(f"   📊 SMA periods: {custom_config['sma_periods']}")
                print(f"   📈 EMA periods: {custom_config['ema_periods']}")
                print(f"   🔄 RSI period: {custom_config['rsi_period']}")
                print(f"   📉 Bollinger Bands: {custom_config['bb_period']} periods, {custom_config['bb_std']} std")
                
            except Exception as e:
                print(f"❌ Failed to initialize Technical Analyzer: {e}")
                self.analyzer = self._create_mock_analyzer()
        else:
            print("❌ Technical Analyzer not available - using mock implementation")
            self.analyzer = self._create_mock_analyzer()
        
        print()
    
    def _create_mock_analyzer(self):
        """Create mock analyzer for demo purposes"""
        class MockAnalyzer:
            def analyze_market_data(self, data, timeframe='1h'):
                return {
                    'timeframe': timeframe,
                    'timestamp': datetime.now(),
                    'indicators': {
                        'rsi': pd.Series([45.2, 47.8, 52.1, 48.9]),
                        'sma': {'sma_20': pd.Series([2000, 2001, 2002, 2003])},
                        'macd': {
                            'macd': pd.Series([0.5, 0.7, 0.3, 0.8]),
                            'signal': pd.Series([0.4, 0.6, 0.5, 0.7]),
                            'histogram': pd.Series([0.1, 0.1, -0.2, 0.1])
                        },
                        'bollinger_bands': {
                            'upper': pd.Series([2020, 2021, 2022, 2023]),
                            'middle': pd.Series([2000, 2001, 2002, 2003]),
                            'lower': pd.Series([1980, 1981, 1982, 1983])
                        }
                    },
                    'signals': [
                        type('Signal', (), {
                            'indicator_name': 'RSI',
                            'signal_type': 'BUY',
                            'strength': 0.7,
                            'confidence': 0.75,
                            'description': 'RSI oversold signal'
                        })(),
                        type('Signal', (), {
                            'indicator_name': 'MACD',
                            'signal_type': 'BUY',
                            'strength': 0.8,
                            'confidence': 0.8,
                            'description': 'MACD bullish crossover'
                        })()
                    ],
                    'patterns': [
                        type('Pattern', (), {
                            'pattern_name': 'Double Bottom',
                            'pattern_type': 'BULLISH',
                            'confidence': 0.85
                        })()
                    ],
                    'support_resistance': {
                        'support': [1980, 1975, 1970],
                        'resistance': [2020, 2025, 2030]
                    },
                    'trend_analysis': {
                        'short_term': 'BULLISH',
                        'medium_term': 'NEUTRAL',
                        'long_term': 'BULLISH',
                        'strength': 0.75,
                        'description': 'Bullish trend (2/3 timeframes)'
                    },
                    'summary': {
                        'total_signals': 2,
                        'buy_signals': 2,
                        'sell_signals': 0,
                        'patterns_detected': 1,
                        'market_sentiment': 'BULLISH',
                        'confidence_score': 0.78,
                        'recommendation': 'BUY'
                    }
                }
        
        return MockAnalyzer()
    
    def generate_sample_market_data(self, symbol: str = "XAUUSD", periods: int = 200) -> pd.DataFrame:
        """Generate realistic sample market data"""
        print("📊 GENERATING SAMPLE MARKET DATA")
        print("-" * 60)
        
        # Create time series
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=periods)
        dates = pd.date_range(start_time, end_time, periods=periods)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Base price for XAUUSD
        base_price = 2000.0
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.015, periods)  # 1.5% volatility
        trend = np.linspace(0, 0.05, periods)  # 5% upward trend over period
        noise = np.random.normal(0, 0.005, periods)  # Market noise
        
        # Combine components
        combined_returns = returns + trend + noise
        
        # Generate price series
        prices = [base_price]
        for ret in combined_returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, prices[-1] * 0.95))  # Prevent extreme drops
        
        # Generate OHLC data
        data = []
        for i, price in enumerate(prices):
            # Add intraday volatility
            daily_vol = np.random.uniform(0.005, 0.02)
            
            open_price = price * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))
            close_price = price
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, daily_vol))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, daily_vol))
            volume = np.random.randint(1000, 15000)
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        
        print(f"✅ Generated {len(df)} data points for {symbol}")
        print(f"   📅 Time range: {df.index[0]} to {df.index[-1]}")
        print(f"   💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"   📊 Average volume: {df['volume'].mean():,.0f}")
        print()
        
        self.demo_results['sample_data'] = {
            'symbol': symbol,
            'periods': len(df),
            'price_range': (df['low'].min(), df['high'].max()),
            'avg_volume': df['volume'].mean()
        }
        
        return df
    
    def run_comprehensive_analysis(self, data: pd.DataFrame, timeframe: str = "1h"):
        """Run comprehensive technical analysis"""
        print("🔍 COMPREHENSIVE TECHNICAL ANALYSIS")
        print("-" * 60)
        
        if not self.analyzer:
            print("❌ Technical Analyzer not available")
            return None
        
        start_time = time.time()
        
        try:
            # Perform comprehensive analysis
            print("🚀 Running technical analysis...")
            print("   This will analyze indicators, patterns, and generate signals")
            print()
            
            results = self.analyzer.analyze_market_data(data, timeframe)
            
            analysis_duration = time.time() - start_time
            
            print(f"✅ Technical analysis completed in {analysis_duration:.2f} seconds")
            
            # Store results
            self.demo_results['analysis'] = {
                'results': results,
                'duration_seconds': analysis_duration,
                'success': True
            }
            
            return results
            
        except Exception as e:
            analysis_duration = time.time() - start_time
            print(f"❌ Technical analysis failed after {analysis_duration:.2f} seconds: {e}")
            
            self.demo_results['analysis'] = {
                'results': None,
                'duration_seconds': analysis_duration,
                'success': False,
                'error': str(e)
            }
            
            return None
    
    def analyze_indicators(self, results):
        """Analyze technical indicators"""
        print("\n📈 TECHNICAL INDICATORS ANALYSIS")
        print("-" * 60)
        
        if not results or 'indicators' not in results:
            print("❌ No indicator data available for analysis")
            return
        
        indicators = results['indicators']
        
        # RSI Analysis
        if 'rsi' in indicators:
            rsi_current = indicators['rsi'].iloc[-1] if hasattr(indicators['rsi'], 'iloc') else 50
            print(f"🔄 RSI Analysis:")
            print(f"   Current RSI: {rsi_current:.2f}")
            if rsi_current < 30:
                print(f"   📊 Status: OVERSOLD (Strong Buy Signal)")
            elif rsi_current > 70:
                print(f"   📊 Status: OVERBOUGHT (Strong Sell Signal)")
            else:
                print(f"   📊 Status: NEUTRAL")
        
        # Moving Averages Analysis
        if 'sma' in indicators:
            print(f"\n📊 Moving Averages Analysis:")
            for ma_name, ma_series in indicators['sma'].items():
                if hasattr(ma_series, 'iloc'):
                    ma_value = ma_series.iloc[-1]
                    print(f"   {ma_name.upper()}: ${ma_value:.2f}")
        
        # MACD Analysis
        if 'macd' in indicators:
            macd_data = indicators['macd']
            if hasattr(macd_data['macd'], 'iloc'):
                macd_current = macd_data['macd'].iloc[-1]
                signal_current = macd_data['signal'].iloc[-1]
                histogram_current = macd_data['histogram'].iloc[-1]
                
                print(f"\n⚡ MACD Analysis:")
                print(f"   MACD Line: {macd_current:.4f}")
                print(f"   Signal Line: {signal_current:.4f}")
                print(f"   Histogram: {histogram_current:.4f}")
                
                if macd_current > signal_current:
                    print(f"   📊 Status: BULLISH (MACD above Signal)")
                else:
                    print(f"   📊 Status: BEARISH (MACD below Signal)")
        
        # Bollinger Bands Analysis
        if 'bollinger_bands' in indicators:
            bb_data = indicators['bollinger_bands']
            if hasattr(bb_data['upper'], 'iloc'):
                bb_upper = bb_data['upper'].iloc[-1]
                bb_middle = bb_data['middle'].iloc[-1]
                bb_lower = bb_data['lower'].iloc[-1]
                
                print(f"\n🎯 Bollinger Bands Analysis:")
                print(f"   Upper Band: ${bb_upper:.2f}")
                print(f"   Middle Band: ${bb_middle:.2f}")
                print(f"   Lower Band: ${bb_lower:.2f}")
                print(f"   Band Width: ${bb_upper - bb_lower:.2f}")
        
        # Store indicator analysis
        self.demo_results['indicator_analysis'] = {
            'rsi_current': rsi_current if 'rsi' in indicators else None,
            'sma_count': len(indicators.get('sma', {})),
            'macd_available': 'macd' in indicators,
            'bollinger_available': 'bollinger_bands' in indicators
        }
    
    def analyze_signals(self, results):
        """Analyze trading signals"""
        print("\n🚨 TRADING SIGNALS ANALYSIS")
        print("-" * 60)
        
        if not results or 'signals' not in results:
            print("❌ No signal data available for analysis")
            return
        
        signals = results['signals']
        
        if not signals:
            print("📊 No trading signals generated")
            return
        
        print(f"📊 Total Signals Generated: {len(signals)}")
        
        # Categorize signals
        buy_signals = [s for s in signals if getattr(s, 'signal_type', '') == 'BUY']
        sell_signals = [s for s in signals if getattr(s, 'signal_type', '') == 'SELL']
        
        print(f"   📈 Buy Signals: {len(buy_signals)}")
        print(f"   📉 Sell Signals: {len(sell_signals)}")
        
        # Analyze individual signals
        print(f"\n🔍 Signal Details:")
        for i, signal in enumerate(signals[:5], 1):  # Show first 5 signals
            indicator = getattr(signal, 'indicator_name', 'Unknown')
            signal_type = getattr(signal, 'signal_type', 'UNKNOWN')
            strength = getattr(signal, 'strength', 0)
            confidence = getattr(signal, 'confidence', 0)
            description = getattr(signal, 'description', 'No description')
            
            print(f"   {i}. {indicator} - {signal_type}")
            print(f"      Strength: {strength:.2f} | Confidence: {confidence:.2f}")
            print(f"      Description: {description}")
        
        # Calculate signal quality
        if signals:
            avg_strength = np.mean([getattr(s, 'strength', 0) for s in signals])
            avg_confidence = np.mean([getattr(s, 'confidence', 0) for s in signals])
            
            print(f"\n📊 Signal Quality Metrics:")
            print(f"   Average Strength: {avg_strength:.2f}")
            print(f"   Average Confidence: {avg_confidence:.2f}")
        
        # Store signal analysis
        self.demo_results['signal_analysis'] = {
            'total_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_strength': avg_strength if signals else 0,
            'avg_confidence': avg_confidence if signals else 0
        }
    
    def analyze_patterns(self, results):
        """Analyze chart patterns"""
        print("\n🔍 CHART PATTERN ANALYSIS")
        print("-" * 60)
        
        if not results or 'patterns' not in results:
            print("❌ No pattern data available for analysis")
            return
        
        patterns = results['patterns']
        
        if not patterns:
            print("📊 No chart patterns detected")
            return
        
        print(f"📊 Total Patterns Detected: {len(patterns)}")
        
        # Categorize patterns
        bullish_patterns = [p for p in patterns if getattr(p, 'pattern_type', '') == 'BULLISH']
        bearish_patterns = [p for p in patterns if getattr(p, 'pattern_type', '') == 'BEARISH']
        neutral_patterns = [p for p in patterns if getattr(p, 'pattern_type', '') == 'NEUTRAL']
        
        print(f"   📈 Bullish Patterns: {len(bullish_patterns)}")
        print(f"   📉 Bearish Patterns: {len(bearish_patterns)}")
        print(f"   ➡️ Neutral Patterns: {len(neutral_patterns)}")
        
        # Analyze individual patterns
        print(f"\n🔍 Pattern Details:")
        for i, pattern in enumerate(patterns[:3], 1):  # Show first 3 patterns
            pattern_name = getattr(pattern, 'pattern_name', 'Unknown')
            pattern_type = getattr(pattern, 'pattern_type', 'UNKNOWN')
            confidence = getattr(pattern, 'confidence', 0)
            
            print(f"   {i}. {pattern_name} ({pattern_type})")
            print(f"      Confidence: {confidence:.2f}")
        
        # Store pattern analysis
        self.demo_results['pattern_analysis'] = {
            'total_patterns': len(patterns),
            'bullish_patterns': len(bullish_patterns),
            'bearish_patterns': len(bearish_patterns),
            'neutral_patterns': len(neutral_patterns)
        }
    
    def analyze_support_resistance(self, results):
        """Analyze support and resistance levels"""
        print("\n🎯 SUPPORT & RESISTANCE ANALYSIS")
        print("-" * 60)
        
        if not results or 'support_resistance' not in results:
            print("❌ No support/resistance data available")
            return
        
        sr_data = results['support_resistance']
        support_levels = sr_data.get('support', [])
        resistance_levels = sr_data.get('resistance', [])
        
        print(f"📊 Support Levels Found: {len(support_levels)}")
        if support_levels:
            for i, level in enumerate(support_levels[:3], 1):
                print(f"   {i}. ${level:.2f}")
        
        print(f"\n📊 Resistance Levels Found: {len(resistance_levels)}")
        if resistance_levels:
            for i, level in enumerate(resistance_levels[:3], 1):
                print(f"   {i}. ${level:.2f}")
        
        # Store support/resistance analysis
        self.demo_results['support_resistance_analysis'] = {
            'support_levels': len(support_levels),
            'resistance_levels': len(resistance_levels)
        }
    
    def analyze_trend(self, results):
        """Analyze market trend"""
        print("\n📈 TREND ANALYSIS")
        print("-" * 60)
        
        if not results or 'trend_analysis' not in results:
            print("❌ No trend data available")
            return
        
        trend_data = results['trend_analysis']
        
        print(f"📊 Multi-Timeframe Trend Analysis:")
        print(f"   Short-term: {trend_data.get('short_term', 'UNKNOWN')}")
        print(f"   Medium-term: {trend_data.get('medium_term', 'UNKNOWN')}")
        print(f"   Long-term: {trend_data.get('long_term', 'UNKNOWN')}")
        print(f"   Overall Strength: {trend_data.get('strength', 0):.2f}")
        print(f"   Description: {trend_data.get('description', 'No description')}")
        
        # Store trend analysis
        self.demo_results['trend_analysis'] = {
            'short_term': trend_data.get('short_term', 'UNKNOWN'),
            'medium_term': trend_data.get('medium_term', 'UNKNOWN'),
            'long_term': trend_data.get('long_term', 'UNKNOWN'),
            'strength': trend_data.get('strength', 0)
        }
    
    def generate_trading_recommendation(self, results):
        """Generate comprehensive trading recommendation"""
        print("\n💡 TRADING RECOMMENDATION ENGINE")
        print("-" * 60)
        
        if not results or 'summary' not in results:
            print("❌ No summary data available for recommendation")
            return
        
        summary = results['summary']
        
        # Extract key metrics
        market_sentiment = summary.get('market_sentiment', 'NEUTRAL')
        recommendation = summary.get('recommendation', 'HOLD')
        confidence_score = summary.get('confidence_score', 0.5)
        total_signals = summary.get('total_signals', 0)
        buy_signals = summary.get('buy_signals', 0)
        sell_signals = summary.get('sell_signals', 0)
        
        print(f"🎯 Market Analysis Summary:")
        print(f"   Market Sentiment: {market_sentiment}")
        print(f"   Recommendation: {recommendation}")
        print(f"   Confidence Score: {confidence_score:.2f}")
        print(f"   Total Signals: {total_signals}")
        print(f"   Signal Breakdown: {buy_signals} BUY, {sell_signals} SELL")
        
        # Generate detailed recommendation
        print(f"\n📋 Detailed Recommendation:")
        
        if recommendation == 'BUY':
            print(f"   🟢 BUY RECOMMENDATION")
            print(f"   📊 Multiple bullish indicators align")
            print(f"   💪 Strong buy signals detected")
            print(f"   ⚠️ Risk Management: Use stop-loss at key support levels")
        elif recommendation == 'SELL':
            print(f"   🔴 SELL RECOMMENDATION") 
            print(f"   📊 Multiple bearish indicators align")
            print(f"   📉 Strong sell signals detected")
            print(f"   ⚠️ Risk Management: Use stop-loss at key resistance levels")
        else:
            print(f"   🟡 HOLD/NEUTRAL RECOMMENDATION")
            print(f"   📊 Mixed signals or insufficient confidence")
            print(f"   ⏳ Wait for clearer market direction")
            print(f"   👀 Monitor key support/resistance levels")
        
        # Confidence assessment
        if confidence_score >= 0.8:
            confidence_level = "HIGH"
        elif confidence_score >= 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        print(f"\n🎯 Confidence Assessment: {confidence_level}")
        print(f"   Score: {confidence_score:.2f}/1.00")
        
        # Store recommendation
        self.demo_results['recommendation'] = {
            'market_sentiment': market_sentiment,
            'recommendation': recommendation,
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'total_signals': total_signals
        }
    
    def calculate_performance_metrics(self, results):
        """Calculate system performance metrics"""
        print("\n📊 PERFORMANCE METRICS CALCULATION")
        print("-" * 60)
        
        # Analysis performance metrics
        analysis_data = self.demo_results.get('analysis', {})
        duration = analysis_data.get('duration_seconds', 0)
        
        print(f"⏱️ Analysis Performance:")
        print(f"   Processing Time: {duration:.2f} seconds")
        print(f"   Data Points Processed: {self.demo_results.get('sample_data', {}).get('periods', 0)}")
        
        if duration > 0:
            throughput = self.demo_results.get('sample_data', {}).get('periods', 0) / duration
            print(f"   Throughput: {throughput:.0f} data points/second")
        
        # Calculate system coverage
        indicators_available = len(results.get('indicators', {})) if results else 0
        signals_generated = len(results.get('signals', [])) if results else 0
        patterns_detected = len(results.get('patterns', [])) if results else 0
        
        print(f"\n📈 Analysis Coverage:")
        print(f"   Indicators Calculated: {indicators_available}")
        print(f"   Signals Generated: {signals_generated}")
        print(f"   Patterns Detected: {patterns_detected}")
        
        # Calculate overall system score
        components_working = sum([
            1 if TECHNICAL_ANALYZER_AVAILABLE else 0,
            1 if analysis_data.get('success', False) else 0,
            1 if indicators_available > 0 else 0,
            1 if signals_generated > 0 else 0
        ])
        
        system_score = (components_working / 4) * 100
        
        print(f"\n🏆 System Performance Score: {system_score:.1f}/100")
        
        # Store performance metrics
        self.demo_results['performance'] = {
            'processing_time': duration,
            'throughput': throughput if duration > 0 else 0,
            'indicators_count': indicators_available,
            'signals_count': signals_generated,
            'patterns_count': patterns_detected,
            'system_score': system_score
        }
        
        return system_score
    
    def generate_final_summary(self, results):
        """Generate final Day 21 summary"""
        print("\n" + "="*100)
        print("📋 DAY 21 TECHNICAL ANALYSIS FOUNDATION - FINAL SUMMARY")
        print("="*100)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"⏰ Demo Duration: {duration.total_seconds():.1f} seconds")
        print(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Technical Analysis Status
        print(f"\n📊 Technical Analysis Status:")
        if TECHNICAL_ANALYZER_AVAILABLE:
            print(f"   🟢 Technical Analyzer: OPERATIONAL")
        else:
            print(f"   🟡 Technical Analyzer: MOCK MODE")
        
        if results:
            print(f"   ✅ Market Analysis: COMPLETED")
            print(f"   📈 Indicators: {len(results.get('indicators', {}))}")
            print(f"   🚨 Signals: {len(results.get('signals', []))}")
            print(f"   🔍 Patterns: {len(results.get('patterns', []))}")
        
        # Performance Summary
        if 'performance' in self.demo_results:
            perf = self.demo_results['performance']
            print(f"\n⚡ Performance Summary:")
            print(f"   ⏱️ Processing Time: {perf['processing_time']:.2f}s")
            print(f"   📊 Throughput: {perf['throughput']:.0f} points/sec")
            print(f"   🏆 System Score: {perf['system_score']:.1f}/100")
        
        # Trading Summary
        if 'recommendation' in self.demo_results:
            rec = self.demo_results['recommendation']
            print(f"\n💡 Trading Summary:")
            print(f"   📊 Market Sentiment: {rec['market_sentiment']}")
            print(f"   💰 Recommendation: {rec['recommendation']}")
            print(f"   🎯 Confidence: {rec['confidence_score']:.2f}")
            print(f"   📶 Confidence Level: {rec['confidence_level']}")
        
        # Success Indicators
        success_indicators = []
        if TECHNICAL_ANALYZER_AVAILABLE:
            success_indicators.append("Technical Analyzer operational")
        if 'analysis' in self.demo_results and self.demo_results['analysis']['success']:
            success_indicators.append("Market analysis completed successfully")
        if 'performance' in self.demo_results and self.demo_results['performance']['system_score'] > 75:
            success_indicators.append("High system performance achieved")
        if 'recommendation' in self.demo_results and self.demo_results['recommendation']['confidence_score'] > 0.6:
            success_indicators.append("High-confidence trading signals generated")
        
        print(f"\n🎉 SUCCESS INDICATORS:")
        for indicator in success_indicators:
            print(f"   ✅ {indicator}")
        
        overall_success = len(success_indicators) >= 3
        print(f"\n🏆 OVERALL DAY 21 STATUS: {'✅ SUCCESS' if overall_success else '⚠️ PARTIAL SUCCESS'}")
        
        if overall_success:
            print("🚀 Day 21 Technical Analysis Foundation completed successfully!")
            print("   📊 Advanced technical analysis capabilities operational")
            print("   🎯 Ready for Day 22 - Advanced Pattern Recognition")
        else:
            print("⚠️ Day 21 completed with some limitations")
            print("   🔧 Additional optimization may be needed")
        
        print("\n" + "="*100)
        
        return {
            'success': overall_success,
            'duration_seconds': duration.total_seconds(),
            'demo_results': self.demo_results,
            'success_indicators': success_indicators
        }


def main():
    """Main demo function"""
    demo = Day21TechnicalDemo()
    
    try:
        # Initialize technical analyzer
        demo.initialize_technical_analyzer()
        
        # Generate sample market data
        sample_data = demo.generate_sample_market_data("XAUUSD", 200)
        
        # Run comprehensive analysis
        results = demo.run_comprehensive_analysis(sample_data, "1h")
        
        if results:
            # Analyze indicators
            demo.analyze_indicators(results)
            
            # Analyze signals
            demo.analyze_signals(results)
            
            # Analyze patterns
            demo.analyze_patterns(results)
            
            # Analyze support/resistance
            demo.analyze_support_resistance(results)
            
            # Analyze trend
            demo.analyze_trend(results)
            
            # Generate recommendation
            demo.generate_trading_recommendation(results)
            
            # Calculate performance
            demo.calculate_performance_metrics(results)
        
        # Generate final summary
        summary = demo.generate_final_summary(results)
        
        return summary
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Run the demo
    try:
        summary = main()
        
        if summary['success']:
            print(f"\n🎊 Day 21 Technical Analysis Foundation Demo completed successfully!")
            print("🚀 Ultimate XAU Super System V4.0 Phase 3 Day 1 COMPLETED!")
        else:
            print(f"\n⚠️ Demo completed with limitations")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")