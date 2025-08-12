#!/usr/bin/env python3
"""
REAL-TIME SIGNAL GENERATION ANALYSIS
Phân tích signal generation thực tế với toàn bộ dữ liệu M1 3 năm
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('src/core')
sys.path.append('src/core/integration')
sys.path.append('src/core/specialists')

def load_full_m1_data():
    """Load toàn bộ dữ liệu M1"""
    print("📊 LOADING FULL M1 DATA (3 YEARS)...")
    print("=" * 60)
    
    m1_file = "data/working_free_data/XAUUSD_M1_realistic.csv"
    
    if not os.path.exists(m1_file):
        print(f"❌ M1 data file not found: {m1_file}")
        return None
    
    try:
        print(f"📈 Loading {m1_file}...")
        df = pd.read_csv(m1_file)
        
        print(f"✅ Loaded {len(df):,} M1 candles")
        print(f"📊 Data range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"📊 Time period: ~{len(df)/1440:.0f} days of data")
        print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading M1 data: {e}")
        return None

def prepare_real_time_features(df, batch_size=10000):
    """Chuẩn bị features theo batch để xử lý dữ liệu lớn"""
    print("\n🔧 PREPARING REAL-TIME FEATURES...")
    print("=" * 60)
    
    # Combine Date and Time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📈 Processing {len(df):,} M1 candles in batches of {batch_size:,}")
    
    # Convert to float32 for memory efficiency
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = df[col].astype('float32')
    
    # Calculate technical indicators
    print("🔧 Calculating technical indicators...")
    
    # Basic price features
    df['returns'] = df['Close'].pct_change().astype('float32')
    df['price_range'] = (df['High'] - df['Low']).astype('float32')
    df['body_size'] = abs(df['Close'] - df['Open']).astype('float32')
    
    # RSI (14 periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = (100 - (100 / (1 + rs))).astype('float32')
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = (exp1 - exp2).astype('float32')
    df['macd_signal'] = df['macd'].ewm(span=9).mean().astype('float32')
    df['macd_histogram'] = (df['macd'] - df['macd_signal']).astype('float32')
    
    # Moving averages
    for period in [20, 50, 200]:
        df[f'sma_{period}'] = df['Close'].rolling(window=period).mean().astype('float32')
        df[f'price_vs_sma_{period}'] = (df['Close'] / df[f'sma_{period}']).astype('float32')
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean().astype('float32')
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = (df['bb_middle'] + (bb_std * 2)).astype('float32')
    df['bb_lower'] = (df['bb_middle'] - (bb_std * 2)).astype('float32')
    df['bb_position'] = ((df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])).astype('float32')
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean().astype('float32')
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(window=20).mean().astype('float32')
    df['volume_ratio'] = (df['Volume'] / df['volume_sma']).astype('float32')
    
    # Time features
    df['hour'] = df['datetime'].dt.hour.astype('int8')
    df['day_of_week'] = df['datetime'].dt.dayofweek.astype('int8')
    df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype('int8')
    df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype('int8')
    
    # Clean data
    df_clean = df.dropna()
    
    print(f"✅ Features prepared: {df_clean.shape[0]:,} samples")
    print(f"✅ Memory usage after processing: {df_clean.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return df_clean

def simulate_18_specialists_real_time(df_clean, signal_interval_minutes=60):
    """Simulate 18 specialists tạo signals real-time"""
    print(f"\n🤖 SIMULATING 18 SPECIALISTS REAL-TIME...")
    print("=" * 60)
    
    # Tạo signals mỗi X phút (mặc định 60 phút = 1 giờ)
    signal_points = df_clean.iloc[::signal_interval_minutes].copy()
    
    print(f"📊 Generating signals every {signal_interval_minutes} minutes")
    print(f"📊 Total signal points: {len(signal_points):,}")
    print(f"📊 Time span: {signal_points['datetime'].iloc[0]} to {signal_points['datetime'].iloc[-1]}")
    
    # Define 18 specialists với logic thực tế
    specialists = {
        'Technical': {
            'RSI_Specialist': lambda row: generate_rsi_signal(row),
            'MACD_Specialist': lambda row: generate_macd_signal(row),
            'Fibonacci_Specialist': lambda row: generate_fibonacci_signal(row)
        },
        'Sentiment': {
            'News_Sentiment_Specialist': lambda row: generate_news_signal(row),
            'Social_Media_Specialist': lambda row: generate_social_signal(row),
            'Fear_Greed_Specialist': lambda row: generate_fear_greed_signal(row)
        },
        'Pattern': {
            'Chart_Pattern_Specialist': lambda row: generate_pattern_signal(row),
            'Candlestick_Specialist': lambda row: generate_candlestick_signal(row),
            'Wave_Specialist': lambda row: generate_wave_signal(row)
        },
        'Risk': {
            'VaR_Risk_Specialist': lambda row: generate_var_signal(row),
            'Drawdown_Specialist': lambda row: generate_drawdown_signal(row),
            'Position_Size_Specialist': lambda row: generate_position_signal(row)
        },
        'Momentum': {
            'Trend_Specialist': lambda row: generate_trend_signal(row),
            'Mean_Reversion_Specialist': lambda row: generate_mean_reversion_signal(row),
            'Breakout_Specialist': lambda row: generate_breakout_signal(row)
        },
        'Volatility': {
            'ATR_Specialist': lambda row: generate_atr_signal(row),
            'Bollinger_Specialist': lambda row: generate_bollinger_signal(row),
            'Volatility_Clustering_Specialist': lambda row: generate_volatility_signal(row)
        }
    }
    
    all_signals = []
    transactions = []
    
    print(f"🔄 Processing {len(signal_points):,} signal points...")
    
    for i, (idx, row) in enumerate(signal_points.iterrows()):
        if i % 1000 == 0:
            print(f"   Processed {i:,}/{len(signal_points):,} signals ({i/len(signal_points)*100:.1f}%)")
        
        # Generate votes from all specialists
        specialist_votes = {}
        vote_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for category, category_specialists in specialists.items():
            for specialist_name, specialist_func in category_specialists.items():
                try:
                    vote = specialist_func(row)
                    specialist_votes[specialist_name] = vote
                    vote_counts[vote['decision']] += 1
                except:
                    # Fallback vote
                    vote = {'decision': 'HOLD', 'confidence': 0.5, 'reasoning': 'Error in calculation'}
                    specialist_votes[specialist_name] = vote
                    vote_counts['HOLD'] += 1
        
        # Determine final decision
        final_decision = max(vote_counts, key=vote_counts.get)
        consensus_strength = vote_counts[final_decision] / 18
        
        # Calculate weighted confidence
        total_confidence = sum(vote['confidence'] for vote in specialist_votes.values())
        weighted_confidence = sum(
            vote['confidence'] for vote in specialist_votes.values() 
            if vote['decision'] == final_decision
        ) / total_confidence if total_confidence > 0 else 0
        
        # Store signal
        signal = {
            'timestamp': row['datetime'],
            'price': row['Close'],
            'final_decision': final_decision,
            'consensus_strength': consensus_strength,
            'weighted_confidence': weighted_confidence,
            'vote_counts': vote_counts,
            'specialist_votes': specialist_votes,
            'market_data': {
                'rsi': row.get('rsi', 50),
                'macd': row.get('macd', 0),
                'bb_position': row.get('bb_position', 0.5),
                'atr': row.get('atr', 0),
                'volume_ratio': row.get('volume_ratio', 1)
            }
        }
        
        all_signals.append(signal)
        
        # Create transaction if not HOLD
        if final_decision != 'HOLD':
            transaction = {
                'transaction_id': len(transactions) + 1,
                'timestamp': row['datetime'],
                'signal_decision': final_decision,
                'price': row['Close'],
                'consensus_strength': consensus_strength,
                'weighted_confidence': weighted_confidence,
                'agree_votes': vote_counts[final_decision],
                'disagree_votes': 18 - vote_counts[final_decision],
                'vote_breakdown': vote_counts
            }
            transactions.append(transaction)
    
    print(f"✅ Generated {len(all_signals):,} signals")
    print(f"✅ Created {len(transactions):,} actionable transactions")
    print(f"✅ Transaction rate: {len(transactions)/len(all_signals)*100:.1f}%")
    
    return all_signals, transactions

# Specialist signal generation functions
def generate_rsi_signal(row):
    """RSI-based signal"""
    rsi = row.get('rsi', 50)
    
    if rsi < 30:
        return {'decision': 'BUY', 'confidence': min(0.9, (30-rsi)/30 + 0.3), 'reasoning': f'RSI oversold: {rsi:.1f}'}
    elif rsi > 70:
        return {'decision': 'SELL', 'confidence': min(0.9, (rsi-70)/30 + 0.3), 'reasoning': f'RSI overbought: {rsi:.1f}'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.5, 'reasoning': f'RSI neutral: {rsi:.1f}'}

def generate_macd_signal(row):
    """MACD-based signal"""
    macd = row.get('macd', 0)
    macd_signal = row.get('macd_signal', 0)
    histogram = row.get('macd_histogram', 0)
    
    if macd > macd_signal and histogram > 0:
        confidence = min(0.9, abs(histogram) * 1000 + 0.3)
        return {'decision': 'BUY', 'confidence': confidence, 'reasoning': 'MACD bullish crossover'}
    elif macd < macd_signal and histogram < 0:
        confidence = min(0.9, abs(histogram) * 1000 + 0.3)
        return {'decision': 'SELL', 'confidence': confidence, 'reasoning': 'MACD bearish crossover'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.4, 'reasoning': 'MACD neutral'}

def generate_fibonacci_signal(row):
    """Fibonacci/Support-Resistance signal"""
    price = row.get('Close', 0)
    sma_20 = row.get('sma_20', price)
    sma_50 = row.get('sma_50', price)
    
    if price > sma_20 > sma_50:
        return {'decision': 'BUY', 'confidence': 0.6, 'reasoning': 'Price above key MAs'}
    elif price < sma_20 < sma_50:
        return {'decision': 'SELL', 'confidence': 0.6, 'reasoning': 'Price below key MAs'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.5, 'reasoning': 'Mixed MA signals'}

def generate_news_signal(row):
    """News sentiment signal (simulated)"""
    # Simulate based on volatility and time
    hour = row.get('hour', 12)
    volume_ratio = row.get('volume_ratio', 1)
    
    # Higher activity during news hours
    if hour in [8, 9, 13, 14, 15] and volume_ratio > 1.2:
        decision = np.random.choice(['BUY', 'SELL'], p=[0.6, 0.4])
        return {'decision': decision, 'confidence': 0.7, 'reasoning': 'High news activity'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.3, 'reasoning': 'Low news impact'}

def generate_social_signal(row):
    """Social media sentiment (simulated)"""
    # Simulate crowd sentiment
    rsi = row.get('rsi', 50)
    
    # Contrarian to RSI (crowd usually wrong at extremes)
    if rsi > 80:
        return {'decision': 'SELL', 'confidence': 0.4, 'reasoning': 'Crowd too bullish'}
    elif rsi < 20:
        return {'decision': 'BUY', 'confidence': 0.4, 'reasoning': 'Crowd too bearish'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.3, 'reasoning': 'Neutral sentiment'}

def generate_fear_greed_signal(row):
    """Fear & Greed signal"""
    bb_position = row.get('bb_position', 0.5)
    
    if bb_position > 0.8:  # Near upper band (greed)
        return {'decision': 'SELL', 'confidence': 0.6, 'reasoning': 'Market greed extreme'}
    elif bb_position < 0.2:  # Near lower band (fear)
        return {'decision': 'BUY', 'confidence': 0.6, 'reasoning': 'Market fear extreme'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.4, 'reasoning': 'Balanced sentiment'}

def generate_pattern_signal(row):
    """Chart pattern signal"""
    price_range = row.get('price_range', 0)
    atr = row.get('atr', 1)
    
    # Breakout pattern
    if price_range > atr * 1.5:
        decision = np.random.choice(['BUY', 'SELL'], p=[0.55, 0.45])
        return {'decision': decision, 'confidence': 0.7, 'reasoning': 'Breakout pattern detected'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.4, 'reasoning': 'No clear pattern'}

def generate_candlestick_signal(row):
    """Candlestick pattern signal"""
    body_size = row.get('body_size', 0)
    price_range = row.get('price_range', 1)
    
    # Doji pattern
    if body_size < price_range * 0.1:
        return {'decision': 'HOLD', 'confidence': 0.8, 'reasoning': 'Doji - indecision'}
    else:
        decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.35, 0.35, 0.3])
        return {'decision': decision, 'confidence': 0.5, 'reasoning': 'Normal candle'}

def generate_wave_signal(row):
    """Elliott Wave signal"""
    sma_20 = row.get('sma_20', 0)
    sma_50 = row.get('sma_50', 0)
    
    if sma_20 > sma_50:
        return {'decision': 'BUY', 'confidence': 0.5, 'reasoning': 'Upward wave'}
    elif sma_20 < sma_50:
        return {'decision': 'SELL', 'confidence': 0.5, 'reasoning': 'Downward wave'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.4, 'reasoning': 'Wave unclear'}

def generate_var_signal(row):
    """VaR risk signal"""
    atr = row.get('atr', 0)
    volume_ratio = row.get('volume_ratio', 1)
    
    # High risk = prefer HOLD
    if atr > 2 and volume_ratio > 1.5:
        return {'decision': 'HOLD', 'confidence': 0.8, 'reasoning': 'High VaR risk'}
    else:
        decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
        return {'decision': decision, 'confidence': 0.4, 'reasoning': 'Acceptable risk'}

def generate_drawdown_signal(row):
    """Drawdown protection signal"""
    returns = row.get('returns', 0)
    
    # Conservative approach
    if abs(returns) > 0.01:  # 1% move
        return {'decision': 'HOLD', 'confidence': 0.7, 'reasoning': 'Drawdown protection'}
    else:
        decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.25, 0.25, 0.5])
        return {'decision': decision, 'confidence': 0.3, 'reasoning': 'Low drawdown risk'}

def generate_position_signal(row):
    """Position sizing signal"""
    volume_ratio = row.get('volume_ratio', 1)
    
    if volume_ratio > 1.3:
        decision = np.random.choice(['BUY', 'SELL'], p=[0.5, 0.5])
        return {'decision': decision, 'confidence': 0.6, 'reasoning': 'Good volume for position'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.5, 'reasoning': 'Insufficient volume'}

def generate_trend_signal(row):
    """Trend following signal"""
    price = row.get('Close', 0)
    sma_200 = row.get('sma_200', price)
    
    if price > sma_200:
        return {'decision': 'BUY', 'confidence': 0.7, 'reasoning': 'Uptrend confirmed'}
    elif price < sma_200:
        return {'decision': 'SELL', 'confidence': 0.7, 'reasoning': 'Downtrend confirmed'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.3, 'reasoning': 'No clear trend'}

def generate_mean_reversion_signal(row):
    """Mean reversion signal"""
    bb_position = row.get('bb_position', 0.5)
    
    if bb_position > 0.8:
        return {'decision': 'SELL', 'confidence': 0.6, 'reasoning': 'Overbought - mean reversion'}
    elif bb_position < 0.2:
        return {'decision': 'BUY', 'confidence': 0.6, 'reasoning': 'Oversold - mean reversion'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.4, 'reasoning': 'Near mean'}

def generate_breakout_signal(row):
    """Breakout signal"""
    price_range = row.get('price_range', 0)
    atr = row.get('atr', 1)
    volume_ratio = row.get('volume_ratio', 1)
    
    if price_range > atr * 1.2 and volume_ratio > 1.1:
        decision = np.random.choice(['BUY', 'SELL'], p=[0.6, 0.4])
        return {'decision': decision, 'confidence': 0.8, 'reasoning': 'Strong breakout'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.3, 'reasoning': 'No breakout'}

def generate_atr_signal(row):
    """ATR volatility signal"""
    atr = row.get('atr', 0)
    price_range = row.get('price_range', 0)
    
    if atr > 0 and price_range < atr * 0.5:
        return {'decision': 'HOLD', 'confidence': 0.6, 'reasoning': 'Low volatility - wait'}
    else:
        decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.35, 0.35, 0.3])
        return {'decision': decision, 'confidence': 0.4, 'reasoning': 'Normal volatility'}

def generate_bollinger_signal(row):
    """Bollinger Bands signal"""
    bb_position = row.get('bb_position', 0.5)
    
    if bb_position > 0.95:
        return {'decision': 'SELL', 'confidence': 0.7, 'reasoning': 'BB upper breach'}
    elif bb_position < 0.05:
        return {'decision': 'BUY', 'confidence': 0.7, 'reasoning': 'BB lower breach'}
    else:
        return {'decision': 'HOLD', 'confidence': 0.4, 'reasoning': 'BB middle range'}

def generate_volatility_signal(row):
    """Volatility clustering signal"""
    atr = row.get('atr', 0)
    volume_ratio = row.get('volume_ratio', 1)
    
    # High volatility clustering
    if atr > 1.5 and volume_ratio > 1.2:
        return {'decision': 'HOLD', 'confidence': 0.7, 'reasoning': 'High volatility cluster'}
    else:
        decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
        return {'decision': decision, 'confidence': 0.3, 'reasoning': 'Normal volatility'}

def analyze_real_time_results(all_signals, transactions):
    """Phân tích kết quả real-time"""
    print(f"\n📊 ANALYZING REAL-TIME RESULTS...")
    print("=" * 60)
    
    # Basic statistics
    total_signals = len(all_signals)
    total_transactions = len(transactions)
    transaction_rate = total_transactions / total_signals * 100
    
    # Decision distribution
    buy_transactions = len([t for t in transactions if t['signal_decision'] == 'BUY'])
    sell_transactions = len([t for t in transactions if t['signal_decision'] == 'SELL'])
    
    # Consensus statistics
    consensus_scores = [s['consensus_strength'] for s in all_signals]
    avg_consensus = np.mean(consensus_scores)
    
    high_consensus_signals = len([s for s in all_signals if s['consensus_strength'] > 0.6])
    low_consensus_signals = len([s for s in all_signals if s['consensus_strength'] < 0.4])
    
    # Transaction consensus
    transaction_consensus = [t['consensus_strength'] for t in transactions]
    avg_transaction_consensus = np.mean(transaction_consensus) if transaction_consensus else 0
    
    print(f"📊 SIGNAL GENERATION OVERVIEW:")
    print(f"   • Total signals analyzed: {total_signals:,}")
    print(f"   • Actionable transactions: {total_transactions:,}")
    print(f"   • Transaction rate: {transaction_rate:.1f}%")
    print(f"   • HOLD signals: {total_signals - total_transactions:,}")
    
    print(f"\n💰 TRANSACTION DISTRIBUTION:")
    print(f"   • BUY transactions: {buy_transactions:,} ({buy_transactions/total_transactions*100:.1f}%)")
    print(f"   • SELL transactions: {sell_transactions:,} ({sell_transactions/total_transactions*100:.1f}%)")
    
    print(f"\n🤝 CONSENSUS ANALYSIS:")
    print(f"   • Average consensus (all signals): {avg_consensus:.1%}")
    print(f"   • Average consensus (transactions): {avg_transaction_consensus:.1%}")
    print(f"   • High consensus signals (>60%): {high_consensus_signals:,} ({high_consensus_signals/total_signals*100:.1f}%)")
    print(f"   • Low consensus signals (<40%): {low_consensus_signals:,} ({low_consensus_signals/total_signals*100:.1f}%)")
    
    # Time-based analysis
    if transactions:
        first_transaction = transactions[0]['timestamp']
        last_transaction = transactions[-1]['timestamp']
        time_span = pd.to_datetime(last_transaction) - pd.to_datetime(first_transaction)
        
        print(f"\n⏰ TIME-BASED ANALYSIS:")
        print(f"   • First transaction: {first_transaction}")
        print(f"   • Last transaction: {last_transaction}")
        print(f"   • Time span: {time_span.days} days")
        print(f"   • Transactions per day: {total_transactions / time_span.days:.1f}")
        print(f"   • Transactions per month: {total_transactions / (time_span.days/30):.1f}")
    
    return {
        'total_signals': total_signals,
        'total_transactions': total_transactions,
        'transaction_rate': transaction_rate,
        'buy_transactions': buy_transactions,
        'sell_transactions': sell_transactions,
        'avg_consensus': avg_consensus,
        'avg_transaction_consensus': avg_transaction_consensus,
        'high_consensus_signals': high_consensus_signals,
        'low_consensus_signals': low_consensus_signals
    }

def save_real_time_results(all_signals, transactions, analysis_results):
    """Save kết quả phân tích real-time"""
    print(f"\n💾 SAVING REAL-TIME RESULTS...")
    print("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save signals (sample first 1000 for file size)
    signals_sample = all_signals[:1000] if len(all_signals) > 1000 else all_signals
    signals_path = f"real_time_signals_sample_{timestamp}.json"
    with open(signals_path, 'w') as f:
        json.dump(signals_sample, f, indent=2, default=str)
    
    # Save all transactions
    transactions_path = f"real_time_transactions_{timestamp}.json"
    with open(transactions_path, 'w') as f:
        json.dump(transactions, f, indent=2, default=str)
    
    # Save analysis results
    analysis_path = f"real_time_analysis_{timestamp}.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"✅ Signals sample saved: {signals_path}")
    print(f"✅ All transactions saved: {transactions_path}")
    print(f"✅ Analysis results saved: {analysis_path}")
    
    return signals_path, transactions_path, analysis_path

def run_real_time_analysis():
    """Chạy phân tích real-time với toàn bộ dữ liệu"""
    print("🚀 REAL-TIME SIGNAL GENERATION ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. Load full data
    df = load_full_m1_data()
    if df is None:
        return
    
    # 2. Prepare features
    df_clean = prepare_real_time_features(df)
    
    # 3. Generate signals (every hour = 60 minutes)
    all_signals, transactions = simulate_18_specialists_real_time(df_clean, signal_interval_minutes=60)
    
    # 4. Analyze results
    analysis_results = analyze_real_time_results(all_signals, transactions)
    
    # 5. Save results
    save_real_time_results(all_signals, transactions, analysis_results)
    
    print(f"\nEnd time: {datetime.now()}")
    print("=" * 80)
    
    return all_signals, transactions, analysis_results

if __name__ == "__main__":
    run_real_time_analysis() 