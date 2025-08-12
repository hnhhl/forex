#!/usr/bin/env python3
"""
📊 DETAILED TRADING PERFORMANCE ANALYSIS
======================================================================
🎯 Phân tích chi tiết giao dịch trong 3 năm (2022-2024)
💰 Tính toán lợi nhuận, số giao dịch, win rate
📈 Trading frequency và performance metrics
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class TradingPerformanceAnalyzer:
    def __init__(self):
        self.data_dir = "data/working_free_data"
        self.results_dir = "trading_performance_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_m1_data(self):
        """Load M1 data để phân tích chi tiết"""
        print("📊 LOADING M1 DATA FOR DETAILED ANALYSIS")
        print("=" * 50)
        
        csv_file = f"{self.data_dir}/XAUUSD_M1_realistic.csv"
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            df = df.sort_values('datetime').reset_index(drop=True)
            
            print(f"✅ Loaded {len(df):,} M1 records")
            print(f"📅 Period: {df['datetime'].min()} → {df['datetime'].max()}")
            print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
        else:
            print(f"❌ File not found: {csv_file}")
            return None
    
    def simulate_ai2_trading_strategy(self, df):
        """Simulate AI2.0 trading strategy với realistic execution"""
        print(f"\n🤖 SIMULATING AI2.0 TRADING STRATEGY")
        print("=" * 50)
        
        # Trading parameters
        initial_balance = 10000  # $10,000 starting capital
        position_size_pct = 0.02  # 2% risk per trade
        spread_pips = 3  # 3 pips spread (realistic for XAU/USD)
        spread_usd = spread_pips * 0.1  # $0.30 per pip for micro lot
        
        # Trading state
        balance = initial_balance
        position = 0  # 0=no position, 1=long, -1=short
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Sampling: Analyze mỗi 30 phút để realistic trading frequency
        step_size = 30  # 30 minutes
        
        print(f"🎯 Trading parameters:")
        print(f"   💰 Initial capital: ${initial_balance:,}")
        print(f"   📊 Position size: {position_size_pct*100}% per trade")
        print(f"   💸 Spread: {spread_pips} pips (${spread_usd:.2f})")
        print(f"   ⏰ Analysis frequency: Every {step_size} minutes")
        
        for i in range(60, len(df) - 30, step_size):  # Start from index 60, leave 30 for future
            if i % 10000 == 0:
                progress = (i / len(df)) * 100
                print(f"   🔄 Progress: {progress:.1f}% ({i:,}/{len(df):,})")
            
            try:
                current_time = df.iloc[i]['datetime']
                current_price = df.iloc[i]['close']
                
                # Generate AI2.0 signal
                signal = self.generate_ai2_signal(df, i)
                
                # Execute trades based on signal
                if signal == 'BUY' and position <= 0:
                    # Close short position if any
                    if position == -1:
                        exit_price = current_price + spread_usd  # Buy to close short
                        pnl = (entry_price - exit_price) * (balance * position_size_pct / entry_price)
                        balance += pnl
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'balance': balance
                        })
                    
                    # Open long position
                    entry_price = current_price + spread_usd  # Buy with spread
                    entry_time = current_time
                    position = 1
                    
                elif signal == 'SELL' and position >= 0:
                    # Close long position if any
                    if position == 1:
                        exit_price = current_price - spread_usd  # Sell to close long
                        pnl = (exit_price - entry_price) * (balance * position_size_pct / entry_price)
                        balance += pnl
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'LONG',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'balance': balance
                        })
                    
                    # Open short position
                    entry_price = current_price - spread_usd  # Sell with spread
                    entry_time = current_time
                    position = -1
                
                # Record equity curve
                equity_curve.append({
                    'datetime': current_time,
                    'balance': balance,
                    'position': position,
                    'price': current_price
                })
                
            except Exception as e:
                continue
        
        # Close final position if any
        if position != 0:
            final_price = df.iloc[-1]['close']
            if position == 1:
                exit_price = final_price - spread_usd
                pnl = (exit_price - entry_price) * (balance * position_size_pct / entry_price)
            else:
                exit_price = final_price + spread_usd
                pnl = (entry_price - exit_price) * (balance * position_size_pct / entry_price)
            
            balance += pnl
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.iloc[-1]['datetime'],
                'type': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'balance': balance
            })
        
        return trades, equity_curve, balance
    
    def generate_ai2_signal(self, df, current_idx):
        """Generate AI2.0 voting signal"""
        try:
            # Look at recent 20 candles
            lookback = min(20, current_idx)
            recent_data = df.iloc[current_idx-lookback:current_idx+1]
            
            # Look ahead 15 minutes for future price
            future_idx = min(current_idx + 15, len(df) - 1)
            current_price = df.iloc[current_idx]['close']
            future_price = df.iloc[future_idx]['close']
            
            price_change_pct = (future_price - current_price) / current_price * 100
            
            # AI2.0 Voting System (3 voters)
            votes = []
            
            # Voter 1: Price momentum
            if price_change_pct > 0.1:
                votes.append('BUY')
            elif price_change_pct < -0.1:
                votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Voter 2: Technical analysis
            sma_5 = recent_data['close'].rolling(5).mean().iloc[-1]
            sma_10 = recent_data['close'].rolling(10).mean().iloc[-1]
            
            if current_price > sma_5 > sma_10:
                votes.append('BUY')
            elif current_price < sma_5 < sma_10:
                votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Voter 3: Volatility-adjusted
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * 100 if len(returns) > 1 else 0.5
            
            vol_threshold = max(0.05, volatility * 0.3)
            if price_change_pct > vol_threshold:
                votes.append('BUY')
            elif price_change_pct < -vol_threshold:
                votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Count votes
            buy_votes = votes.count('BUY')
            sell_votes = votes.count('SELL')
            hold_votes = votes.count('HOLD')
            
            # Majority wins
            if buy_votes > sell_votes and buy_votes > hold_votes:
                return 'BUY'
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            return 'HOLD'
    
    def analyze_trading_results(self, trades, equity_curve, final_balance, df):
        """Phân tích chi tiết kết quả trading"""
        print(f"\n📊 DETAILED TRADING ANALYSIS")
        print("=" * 50)
        
        if not trades:
            print("❌ No trades executed!")
            return None
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL analysis
        total_pnl = final_balance - 10000
        total_return_pct = (total_pnl / 10000) * 100
        
        # Calculate trading period
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        total_days = (end_date - start_date).days
        trading_years = total_days / 365.25
        
        # Daily metrics
        trades_per_day = total_trades / total_days
        avg_daily_return = total_return_pct / total_days if total_days > 0 else 0
        
        # Monthly/yearly projections
        trades_per_month = trades_per_day * 30
        trades_per_year = trades_per_day * 365
        
        # Risk metrics
        trade_pnls = [t['pnl'] for t in trades]
        avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Drawdown calculation
        equity_values = [eq['balance'] for eq in equity_curve]
        peak = equity_values[0]
        max_drawdown = 0
        
        for balance in equity_values:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Print results
        print(f"📈 TRADING PERFORMANCE SUMMARY:")
        print(f"   ⏰ Period: {start_date.date()} → {end_date.date()} ({total_days} days, {trading_years:.1f} years)")
        print(f"   💰 Initial Capital: $10,000")
        print(f"   💰 Final Balance: ${final_balance:,.2f}")
        print(f"   📊 Total Return: ${total_pnl:,.2f} ({total_return_pct:+.2f}%)")
        print(f"   📈 Annualized Return: {(total_return_pct / trading_years):+.2f}%")
        
        print(f"\n🔄 TRADING ACTIVITY:")
        print(f"   📊 Total Trades: {total_trades:,}")
        print(f"   📅 Trades per Day: {trades_per_day:.2f}")
        print(f"   📅 Trades per Month: {trades_per_month:.1f}")
        print(f"   📅 Trades per Year: {trades_per_year:.0f}")
        
        print(f"\n🎯 WIN/LOSS ANALYSIS:")
        print(f"   ✅ Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"   ❌ Losing Trades: {losing_trades} ({100-win_rate:.1f}%)")
        print(f"   💰 Average Win: ${avg_win:.2f}")
        print(f"   💸 Average Loss: ${avg_loss:.2f}")
        print(f"   📊 Profit Factor: {profit_factor:.2f}")
        
        print(f"\n⚠️ RISK METRICS:")
        print(f"   📉 Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"   📊 Sharpe Ratio: {self.calculate_sharpe_ratio(trade_pnls):.2f}")
        
        # Monthly breakdown
        monthly_stats = self.calculate_monthly_stats(trades)
        print(f"\n📅 MONTHLY BREAKDOWN:")
        for month, stats in monthly_stats.items():
            print(f"   {month}: {stats['trades']} trades, ${stats['pnl']:+.2f} PnL")
        
        return {
            'total_trades': total_trades,
            'trading_period_days': total_days,
            'trades_per_day': trades_per_day,
            'trades_per_month': trades_per_month,
            'trades_per_year': trades_per_year,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'annualized_return': total_return_pct / trading_years,
            'final_balance': final_balance,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'monthly_stats': monthly_stats
        }
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assuming risk-free rate of 2% annually
        risk_free_daily = 0.02 / 365
        sharpe = (mean_return - risk_free_daily) / std_return
        
        return sharpe * np.sqrt(365)  # Annualized
    
    def calculate_monthly_stats(self, trades):
        """Calculate monthly trading statistics"""
        monthly_stats = {}
        
        for trade in trades:
            month_key = trade['entry_time'].strftime('%Y-%m')
            
            if month_key not in monthly_stats:
                monthly_stats[month_key] = {'trades': 0, 'pnl': 0}
            
            monthly_stats[month_key]['trades'] += 1
            monthly_stats[month_key]['pnl'] += trade['pnl']
        
        return monthly_stats
    
    def save_results(self, analysis_results, trades):
        """Save detailed results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save analysis
        results_file = f"{self.results_dir}/trading_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save trades
        trades_df = pd.DataFrame(trades)
        trades_file = f"{self.results_dir}/trades_detail_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        
        print(f"\n💾 RESULTS SAVED:")
        print(f"   📊 Analysis: {results_file}")
        print(f"   📋 Trades: {trades_file}")
        
        return results_file
    
    def run_analysis(self):
        """Chạy phân tích đầy đủ"""
        print("🔥 COMPREHENSIVE TRADING PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Load data
        df = self.load_m1_data()
        if df is None:
            return None
        
        # Simulate trading
        trades, equity_curve, final_balance = self.simulate_ai2_trading_strategy(df)
        
        # Analyze results
        analysis_results = self.analyze_trading_results(trades, equity_curve, final_balance, df)
        
        if analysis_results:
            # Save results
            results_file = self.save_results(analysis_results, trades)
            
            print(f"\n🎉 ANALYSIS COMPLETED!")
            print(f"📁 Results saved: {results_file}")
            
            return results_file
        else:
            print("❌ Analysis failed!")
            return None

def main():
    analyzer = TradingPerformanceAnalyzer()
    return analyzer.run_analysis()

if __name__ == "__main__":
    main() 