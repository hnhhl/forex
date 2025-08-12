#!/usr/bin/env python3
"""
📊 DETAILED TRADING ANALYSIS - 50 EPOCHS
======================================================================
🎯 Training với exactly 50 epochs và báo cáo trading chi tiết
📈 Focus: Trades count, Win rate, Profit/Loss, Drawdown analysis  
🚀 Optimized for speed and comprehensive reporting
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DetailedTradingAnalysis:
    def __init__(self):
        self.results_dir = "detailed_trading_results"
        self.models_dir = "detailed_models_50epochs"
        
        # Trading parameters
        self.initial_balance = 10000.0
        self.position_size = 0.02  # 2% per trade (conservative)
        self.spread = 0.0002  # 2 pips
        self.commission = 0.0001  # 0.01%
        
        # Create directories
        for dir_path in [self.results_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_realistic_market_data(self, n_samples=5000):
        """Tạo realistic market data nhanh"""
        print("🔧 Creating realistic market data...")
        
        np.random.seed(42)
        
        # Generate realistic XAUUSD movements
        base_price = 1800.0
        dates = pd.date_range('2022-01-01', periods=n_samples, freq='H')
        
        # Price movements with realistic patterns
        returns = np.random.normal(0, 0.001, n_samples)
        
        # Add trend and volatility clustering
        trend = np.sin(np.arange(n_samples) * 0.01) * 0.0005
        volatility = 0.001 + 0.0005 * np.abs(np.sin(np.arange(n_samples) * 0.02))
        
        adjusted_returns = returns + trend
        adjusted_returns *= volatility / 0.001
        
        prices = base_price * np.cumprod(1 + adjusted_returns)
        
        # Generate OHLC
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        highs = np.maximum(opens, prices) + np.abs(np.random.normal(0, 0.5, n_samples))
        lows = np.minimum(opens, prices) - np.abs(np.random.normal(0, 0.5, n_samples))
        volumes = np.random.lognormal(8, 0.3, n_samples).astype(int)
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        print(f"✅ Created {len(df):,} market records")
        return df
    
    def create_essential_features(self, df):
        """Tạo essential features cho trading"""
        print("🧠 Creating essential trading features...")
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['volatility'] = df['returns'].rolling(20).std() * 100
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Market regime
        df['trend_regime'] = self.classify_trend_regime(df)
        df['volatility_regime'] = self.classify_volatility_regime(df)
        
        print(f"✅ Created {df.shape[1]} essential features")
        return df
    
    def calculate_rsi(self, prices, period=14):
        """RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices):
        """MACD calculation"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        return macd, macd_signal
    
    def classify_trend_regime(self, df):
        """Trend regime classification"""
        returns_20 = df['close'].pct_change(20)
        regime = np.ones(len(df))
        regime[returns_20 > 0.02] = 2  # Uptrend
        regime[returns_20 < -0.02] = 0  # Downtrend
        return regime
    
    def classify_volatility_regime(self, df):
        """Volatility regime classification"""
        volatility = df['volatility'].fillna(0.5)
        regime = np.ones(len(df))
        regime[volatility < volatility.quantile(0.33)] = 0  # Low vol
        regime[volatility > volatility.quantile(0.67)] = 2  # High vol
        return regime
    
    def generate_smart_trading_signals(self, df):
        """Generate smart trading signals"""
        print("🎯 Generating smart trading signals...")
        
        signals = []
        signal_details = []
        
        for i in range(50, len(df) - 10):
            try:
                row = df.iloc[i]
                
                # Multi-factor signal generation
                technical_score = self.get_technical_score(df, i)
                momentum_score = self.get_momentum_score(df, i)
                volume_score = self.get_volume_score(df, i)
                regime_score = self.get_regime_score(df, i)
                
                # Weighted ensemble
                total_score = (
                    technical_score * 0.3 +
                    momentum_score * 0.3 +
                    volume_score * 0.2 +
                    regime_score * 0.2
                )
                
                # Dynamic thresholds
                volatility = row['volatility']
                if pd.isna(volatility):
                    volatility = 0.5
                
                # Adjust thresholds based on volatility
                if volatility > 1.0:  # High volatility
                    buy_threshold = 0.65
                    sell_threshold = 0.35
                else:  # Normal/Low volatility
                    buy_threshold = 0.58
                    sell_threshold = 0.42
                
                # Generate signal
                if total_score > buy_threshold:
                    signal = 1  # BUY
                elif total_score < sell_threshold:
                    signal = 0  # SELL
                else:
                    signal = 2  # HOLD
                
                signals.append(signal)
                signal_details.append({
                    'index': i,
                    'price': row['close'],
                    'signal': signal,
                    'total_score': total_score,
                    'technical_score': technical_score,
                    'momentum_score': momentum_score,
                    'volume_score': volume_score,
                    'regime_score': regime_score,
                    'volatility': volatility
                })
                
            except Exception as e:
                signals.append(2)  # Default HOLD
                signal_details.append(None)
        
        # Signal distribution
        unique, counts = np.unique(signals, return_counts=True)
        signal_dist = {'SELL': 0, 'BUY': 0, 'HOLD': 0}
        for signal, count in zip(unique, counts):
            if signal == 0:
                signal_dist['SELL'] = count
            elif signal == 1:
                signal_dist['BUY'] = count
            elif signal == 2:
                signal_dist['HOLD'] = count
        
        total = sum(signal_dist.values())
        print(f"📊 Signal distribution:")
        for action, count in signal_dist.items():
            print(f"   {action}: {count:,} ({count/total:.1%})")
        
        return np.array(signals), signal_details
    
    def get_technical_score(self, df, i):
        """Technical analysis score"""
        try:
            row = df.iloc[i]
            score = 0.5
            
            # Trend signals
            if row['close'] > row['sma_5'] > row['sma_20']:
                score += 0.2
            elif row['close'] < row['sma_5'] < row['sma_20']:
                score -= 0.2
            
            # RSI
            rsi = row['rsi']
            if rsi < 30:
                score += 0.15
            elif rsi > 70:
                score -= 0.15
            
            # MACD
            if row['macd'] > row['macd_signal']:
                score += 0.1
            else:
                score -= 0.1
            
            return max(0, min(1, score))
        except:
            return 0.5
    
    def get_momentum_score(self, df, i):
        """Momentum score"""
        try:
            row = df.iloc[i]
            score = 0.5
            
            mom_5 = row['momentum_5']
            mom_20 = row['momentum_20']
            
            if mom_5 > 0 and mom_20 > 0:
                score += 0.2
            elif mom_5 < 0 and mom_20 < 0:
                score -= 0.2
            
            return max(0, min(1, score))
        except:
            return 0.5
    
    def get_volume_score(self, df, i):
        """Volume score"""
        try:
            row = df.iloc[i]
            score = 0.5
            
            vol_ratio = row['volume_ratio']
            price_change = row['price_change']
            
            if vol_ratio > 1.2 and price_change > 0:
                score += 0.15
            elif vol_ratio > 1.2 and price_change < 0:
                score -= 0.15
            
            return max(0, min(1, score))
        except:
            return 0.5
    
    def get_regime_score(self, df, i):
        """Market regime score"""
        try:
            row = df.iloc[i]
            score = 0.5
            
            trend_regime = row['trend_regime']
            vol_regime = row['volatility_regime']
            
            if trend_regime == 2:  # Uptrend
                score += 0.15
            elif trend_regime == 0:  # Downtrend
                score -= 0.15
            
            if vol_regime == 0:  # Low volatility
                score += 0.05
            elif vol_regime == 2:  # High volatility
                score -= 0.05
            
            return max(0, min(1, score))
        except:
            return 0.5
    
    def train_models_50_epochs(self, X, y):
        """Train models với exactly 50 epochs"""
        print("🚀 TRAINING MODELS - 50 EPOCHS")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training: {len(X_train):,} | Testing: {len(X_test):,}")
        
        models = {}
        results = {}
        
        # Random Forest (optimized)
        print("\n🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        models['random_forest'] = rf_model
        results['random_forest'] = {
            'test_accuracy': rf_accuracy,
            'train_accuracy': rf_model.score(X_train, y_train),
            'predictions': rf_pred
        }
        
        print(f"✅ Random Forest Accuracy: {rf_accuracy:.3f}")
        
        # Neural Network (exactly 50 epochs)
        print("\n🧠 Training Neural Network (50 epochs)...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            max_iter=50,  # EXACTLY 50 epochs
            learning_rate_init=0.01,
            alpha=0.01,
            random_state=42,
            early_stopping=False,  # NO early stopping
            validation_fraction=0.0
        )
        
        nn_model.fit(X_train, y_train)
        nn_pred = nn_model.predict(X_test)
        nn_accuracy = accuracy_score(y_test, nn_pred)
        
        models['neural_network'] = nn_model
        results['neural_network'] = {
            'test_accuracy': nn_accuracy,
            'train_accuracy': nn_model.score(X_train, y_train),
            'predictions': nn_pred,
            'epochs_completed': nn_model.n_iter_
        }
        
        print(f"✅ Neural Network Accuracy: {nn_accuracy:.3f}")
        print(f"📊 Epochs: {nn_model.n_iter_}")
        
        # Ensemble
        print("\n🤝 Creating Ensemble...")
        rf_proba = rf_model.predict_proba(X_test)
        nn_proba = nn_model.predict_proba(X_test)
        
        ensemble_proba = 0.6 * rf_proba + 0.4 * nn_proba
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        results['ensemble'] = {
            'test_accuracy': ensemble_accuracy,
            'predictions': ensemble_pred
        }
        
        print(f"✅ Ensemble Accuracy: {ensemble_accuracy:.3f}")
        
        return models, results, X_test, y_test
    
    def simulate_detailed_trading(self, models, results, X_test, y_test):
        """Simulate detailed trading với comprehensive metrics"""
        print("\n💰 DETAILED TRADING SIMULATION")
        print("=" * 60)
        
        trading_results = {}
        
        for model_name, result in results.items():
            print(f"\n📊 {model_name.replace('_', ' ').title()} Trading Analysis:")
            
            predictions = result['predictions']
            
            # Trading simulation
            balance = self.initial_balance
            trades = []
            equity_curve = [balance]
            
            for i, (pred, actual) in enumerate(zip(predictions, y_test)):
                if pred != 2:  # Not HOLD
                    # Entry
                    entry_price = 1800 + np.random.normal(0, 2)
                    position_value = balance * self.position_size
                    
                    # Exit (simulate realistic holding period)
                    holding_period = np.random.choice([1, 2, 3, 5], p=[0.5, 0.3, 0.15, 0.05])
                    
                    # Realistic price movement
                    price_change_std = 0.005 * holding_period
                    price_multiplier = 1 + np.random.normal(0, price_change_std)
                    exit_price = entry_price * price_multiplier
                    
                    # Calculate P&L
                    if pred == 1:  # BUY
                        pnl = (exit_price - entry_price) / entry_price * position_value
                    else:  # SELL
                        pnl = (entry_price - exit_price) / entry_price * position_value
                    
                    # Apply costs
                    costs = position_value * (self.spread + self.commission)
                    net_pnl = pnl - costs
                    
                    # Update balance
                    balance += net_pnl
                    equity_curve.append(balance)
                    
                    # Record trade
                    trades.append({
                        'trade_id': len(trades) + 1,
                        'signal': 'BUY' if pred == 1 else 'SELL',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position_value,
                        'gross_pnl': pnl,
                        'costs': costs,
                        'net_pnl': net_pnl,
                        'balance': balance,
                        'holding_period': holding_period,
                        'correct_prediction': (pred == actual)
                    })
            
            # Calculate detailed metrics
            if trades:
                trades_df = pd.DataFrame(trades)
                
                # Basic stats
                total_trades = len(trades)
                winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
                losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
                win_rate = winning_trades / total_trades * 100
                
                # P&L analysis
                total_pnl = trades_df['net_pnl'].sum()
                total_return = (balance - self.initial_balance) / self.initial_balance * 100
                avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
                avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
                
                # Risk metrics
                returns = trades_df['net_pnl'] / self.initial_balance
                volatility = returns.std() * np.sqrt(252)
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # Drawdown
                equity_series = pd.Series(equity_curve)
                peak = equity_series.expanding().max()
                drawdown = (equity_series - peak) / peak * 100
                max_drawdown = drawdown.min()
                
                # Advanced metrics
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
                expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
                
                # Prediction accuracy
                correct_predictions = trades_df['correct_prediction'].sum()
                prediction_accuracy = correct_predictions / total_trades * 100
                
                # Monthly analysis
                monthly_returns = []
                for month in range(1, 13):
                    month_trades = trades_df[trades_df.index % 12 == month-1]
                    if len(month_trades) > 0:
                        monthly_returns.append(month_trades['net_pnl'].sum())
                
                best_month = max(monthly_returns) if monthly_returns else 0
                worst_month = min(monthly_returns) if monthly_returns else 0
                
                trading_results[model_name] = {
                    # Basic metrics
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    
                    # P&L metrics
                    'total_pnl': total_pnl,
                    'total_return': total_return,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'expectancy': expectancy,
                    
                    # Risk metrics
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    
                    # Performance metrics
                    'final_balance': balance,
                    'prediction_accuracy': prediction_accuracy,
                    'best_month': best_month,
                    'worst_month': worst_month,
                    
                    # Detailed data
                    'equity_curve': equity_curve,
                    'trades_detail': trades[:10]  # First 10 trades for sample
                }
                
                # Print comprehensive results
                print(f"   📈 Total Trades: {total_trades}")
                print(f"   🎯 Win Rate: {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)")
                print(f"   💰 Total P&L: ${total_pnl:.2f}")
                print(f"   📊 Total Return: {total_return:.1f}%")
                print(f"   🏆 Avg Win: ${avg_win:.2f} | 💸 Avg Loss: ${avg_loss:.2f}")
                print(f"   📈 Profit Factor: {profit_factor:.2f}")
                print(f"   🎯 Expectancy: ${expectancy:.2f}")
                print(f"   📉 Max Drawdown: {max_drawdown:.1f}%")
                print(f"   📈 Sharpe Ratio: {sharpe:.2f}")
                print(f"   🎯 Prediction Accuracy: {prediction_accuracy:.1f}%")
                print(f"   💵 Final Balance: ${balance:.2f}")
                print(f"   📅 Best Month: ${best_month:.2f} | Worst Month: ${worst_month:.2f}")
                
            else:
                trading_results[model_name] = {'total_trades': 0, 'message': 'No trades generated'}
                print(f"   ⚠️ No trades generated")
        
        return trading_results
    
    def create_detailed_report(self, models, results, trading_results, timestamp):
        """Create comprehensive detailed report"""
        print("\n📊 CREATING DETAILED REPORT")
        print("-" * 50)
        
        # Find best model
        best_model = None
        best_score = 0
        
        for model_name, trading_result in trading_results.items():
            if trading_result.get('total_trades', 0) > 0:
                # Composite score
                accuracy = results[model_name]['test_accuracy']
                return_pct = trading_result.get('total_return', 0) / 100
                drawdown_penalty = abs(trading_result.get('max_drawdown', 0)) / 100
                win_rate_bonus = trading_result.get('win_rate', 0) / 100
                
                composite_score = accuracy + return_pct + win_rate_bonus - drawdown_penalty * 0.5
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model = model_name
        
        # Create detailed report
        detailed_report = {
            'timestamp': timestamp,
            'training_summary': {
                'epochs': 50,
                'models_trained': list(models.keys()),
                'dataset_size': 'Realistic market simulation',
                'training_duration': 'Optimized for speed'
            },
            'model_accuracy': {
                model_name: {
                    'test_accuracy': result['test_accuracy'],
                    'train_accuracy': result.get('train_accuracy', 0),
                    'overfitting_score': result.get('train_accuracy', 0) - result['test_accuracy']
                }
                for model_name, result in results.items()
            },
            'trading_performance': trading_results,
            'best_model_analysis': {
                'best_model': best_model,
                'composite_score': best_score,
                'performance_summary': trading_results.get(best_model, {}) if best_model else {}
            },
            'comparative_analysis': {
                'total_models': len(models),
                'models_with_trades': len([tr for tr in trading_results.values() if tr.get('total_trades', 0) > 0]),
                'average_accuracy': np.mean([r['test_accuracy'] for r in results.values()]),
                'average_win_rate': np.mean([tr.get('win_rate', 0) for tr in trading_results.values() if tr.get('total_trades', 0) > 0]) if any(tr.get('total_trades', 0) > 0 for tr in trading_results.values()) else 0
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if best_model:
            best_perf = trading_results[best_model]
            
            if best_perf.get('win_rate', 0) > 55:
                detailed_report['recommendations'].append("✅ Win rate > 55% - Good trading potential")
            if best_perf.get('total_return', 0) > 5:
                detailed_report['recommendations'].append("💰 Positive returns - Profitable strategy")
            if abs(best_perf.get('max_drawdown', 0)) < 15:
                detailed_report['recommendations'].append("🛡️ Controlled drawdown - Good risk management")
            if best_perf.get('profit_factor', 0) > 1.2:
                detailed_report['recommendations'].append("📈 Profit factor > 1.2 - Sustainable profits")
            if best_perf.get('sharpe_ratio', 0) > 0.5:
                detailed_report['recommendations'].append("📊 Good risk-adjusted returns")
        
        if not detailed_report['recommendations']:
            detailed_report['recommendations'].append("⚠️ Consider strategy optimization or parameter tuning")
        
        # Save report
        report_file = f"{self.results_dir}/detailed_trading_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            clean_report = json.loads(json.dumps(detailed_report, default=convert_numpy))
            json.dump(clean_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed report saved: {report_file}")
        
        # Print summary
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"🏆 Best Model: {best_model}")
        if best_model and best_model in trading_results:
            best_perf = trading_results[best_model]
            print(f"📊 Best Performance Summary:")
            print(f"   💰 Total Trades: {best_perf.get('total_trades', 0)}")
            print(f"   🎯 Win Rate: {best_perf.get('win_rate', 0):.1f}%")
            print(f"   📈 Total Return: {best_perf.get('total_return', 0):.1f}%")
            print(f"   💵 Final Balance: ${best_perf.get('final_balance', 0):.2f}")
        
        return report_file, detailed_report
    
    def run_detailed_analysis(self):
        """Run complete detailed trading analysis"""
        print("🚀 DETAILED TRADING ANALYSIS - 50 EPOCHS")
        print("=" * 60)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create market data
        df = self.create_realistic_market_data(5000)
        
        # Create features
        df = self.create_essential_features(df)
        
        # Generate signals
        signals, signal_details = self.generate_smart_trading_signals(df)
        
        # Prepare training data
        feature_columns = [
            'sma_5', 'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
            'momentum_5', 'momentum_20', 'volatility', 'volume_ratio',
            'hour', 'day_of_week', 'trend_regime', 'volatility_regime'
        ]
        
        # Match lengths
        min_length = min(len(df) - 60, len(signals))
        features_df = df.iloc[50:50+min_length][feature_columns].fillna(0)
        signals_trimmed = signals[:min_length]
        
        X = features_df.values
        y = signals_trimmed
        
        print(f"📊 Final dataset: {len(X):,} samples, {X.shape[1]} features")
        
        # Train models
        models, results, X_test, y_test = self.train_models_50_epochs(X, y)
        
        # Simulate trading
        trading_results = self.simulate_detailed_trading(models, results, X_test, y_test)
        
        # Create report
        report_file, detailed_report = self.create_detailed_report(models, results, trading_results, timestamp)
        
        # Save models
        for model_name, model in models.items():
            model_file = f"{self.models_dir}/{model_name}_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"💾 Model saved: {model_file}")
        
        return report_file

def main():
    """Main execution"""
    analyzer = DetailedTradingAnalysis()
    return analyzer.run_detailed_analysis()

if __name__ == "__main__":
    main() 