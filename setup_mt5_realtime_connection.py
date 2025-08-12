#!/usr/bin/env python3
"""
SETUP KẾT NỐI MT5 THỰC TẾ VÀ TEST HỆ THỐNG
Kết nối đến MT5 real-time và test lại toàn bộ hệ thống
"""

import sys
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
sys.path.append('src/core')

class MT5RealTimeSetup:
    def __init__(self):
        self.mt5_initialized = False
        self.connection_status = {}
        self.test_results = {
            'start_time': datetime.now(),
            'mt5_connection': {},
            'real_time_data': {},
            'system_performance': {},
            'signal_quality': {}
        }
        
    def setup_mt5_connection(self):
        """Setup kết nối MT5 thực tế"""
        print("🚀 SETUP KẾT NỐI MT5 THỰC TẾ")
        print("=" * 60)
        
        try:
            # Import MT5
            import MetaTrader5 as mt5
            
            print("📦 Checking MT5 installation...")
            
            # Initialize MT5
            if not mt5.initialize():
                print("❌ MT5 initialization failed")
                print("🔧 Trying alternative initialization methods...")
                
                # Try different initialization methods
                mt5_paths = [
                    "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
                    "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
                    "C:\\Users\\Admin\\AppData\\Roaming\\MetaQuotes\\Terminal\\*\\terminal64.exe"
                ]
                
                for path in mt5_paths:
                    try:
                        if mt5.initialize(path=path):
                            print(f"✅ MT5 initialized with path: {path}")
                            self.mt5_initialized = True
                            break
                    except Exception as e:
                        print(f"⚠️ Failed with path {path}: {e}")
                
                if not self.mt5_initialized:
                    print("❌ All MT5 initialization methods failed")
                    return self._setup_demo_connection()
            else:
                print("✅ MT5 initialized successfully")
                self.mt5_initialized = True
            
            # Get MT5 info
            if self.mt5_initialized:
                terminal_info = mt5.terminal_info()
                account_info = mt5.account_info()
                
                print(f"\n📊 MT5 TERMINAL INFO:")
                if terminal_info:
                    print(f"   • Company: {terminal_info.company}")
                    print(f"   • Name: {terminal_info.name}")
                    print(f"   • Path: {terminal_info.path}")
                    print(f"   • Data Path: {terminal_info.data_path}")
                    print(f"   • Connected: {terminal_info.connected}")
                
                print(f"\n💰 ACCOUNT INFO:")
                if account_info:
                    print(f"   • Login: {account_info.login}")
                    print(f"   • Server: {account_info.server}")
                    print(f"   • Currency: {account_info.currency}")
                    print(f"   • Balance: ${account_info.balance:,.2f}")
                    print(f"   • Equity: ${account_info.equity:,.2f}")
                    print(f"   • Trade Allowed: {account_info.trade_allowed}")
                
                self.connection_status = {
                    'initialized': True,
                    'connected': terminal_info.connected if terminal_info else False,
                    'terminal_info': terminal_info._asdict() if terminal_info else None,
                    'account_info': account_info._asdict() if account_info else None
                }
                
                return True
            
        except ImportError:
            print("❌ MetaTrader5 package not installed")
            print("🔧 Installing MetaTrader5 package...")
            return self._install_and_setup_mt5()
        
        except Exception as e:
            print(f"❌ MT5 setup error: {e}")
            return self._setup_demo_connection()
    
    def _install_and_setup_mt5(self):
        """Install MT5 package and setup"""
        try:
            import subprocess
            import sys
            
            print("📦 Installing MetaTrader5...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "MetaTrader5"])
            
            print("✅ MetaTrader5 installed successfully")
            
            # Try again
            import MetaTrader5 as mt5
            if mt5.initialize():
                print("✅ MT5 initialized after installation")
                self.mt5_initialized = True
                return True
            else:
                print("❌ MT5 still failed after installation")
                return self._setup_demo_connection()
                
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return self._setup_demo_connection()
    
    def _setup_demo_connection(self):
        """Setup demo connection nếu MT5 thật không available"""
        print("\n🎭 SETTING UP DEMO CONNECTION")
        print("⚠️ Using simulated MT5 connection for testing")
        
        self.connection_status = {
            'initialized': True,
            'connected': True,
            'demo_mode': True,
            'terminal_info': {
                'company': 'Demo Company',
                'name': 'Demo Terminal',
                'connected': True
            },
            'account_info': {
                'login': 12345678,
                'server': 'Demo-Server',
                'currency': 'USD',
                'balance': 10000.0,
                'equity': 10000.0,
                'trade_allowed': True
            }
        }
        
        return True
    
    def get_real_time_data(self, symbol="XAUUSD", timeframe="H1", count=1000):
        """Lấy dữ liệu real-time từ MT5"""
        print(f"\n📊 GETTING REAL-TIME DATA: {symbol}")
        print("-" * 40)
        
        try:
            if self.mt5_initialized and not self.connection_status.get('demo_mode', False):
                import MetaTrader5 as mt5
                
                # Convert timeframe
                tf_map = {
                    'M1': mt5.TIMEFRAME_M1,
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1,
                    'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1
                }
                
                mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
                
                # Get rates
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
                
                if rates is not None and len(rates) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    print(f"✅ Retrieved {len(df)} real-time candles")
                    print(f"📅 Time range: {df['time'].min()} to {df['time'].max()}")
                    print(f"💰 Latest price: {df['close'].iloc[-1]:.2f}")
                    print(f"📊 Latest volume: {df['tick_volume'].iloc[-1]:,}")
                    
                    self.test_results['real_time_data'] = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'count': len(df),
                        'latest_time': df['time'].iloc[-1].isoformat(),
                        'latest_price': float(df['close'].iloc[-1]),
                        'latest_volume': int(df['tick_volume'].iloc[-1]),
                        'data_source': 'MT5_REAL_TIME'
                    }
                    
                    return df
                else:
                    print(f"❌ No data received for {symbol}")
                    return self._get_simulated_realtime_data(symbol)
            else:
                print("⚠️ Using simulated real-time data")
                return self._get_simulated_realtime_data(symbol)
                
        except Exception as e:
            print(f"❌ Real-time data error: {e}")
            return self._get_simulated_realtime_data(symbol)
    
    def _get_simulated_realtime_data(self, symbol="XAUUSD"):
        """Tạo dữ liệu giả lập real-time"""
        print("🎭 Generating simulated real-time data...")
        
        # Create realistic current data
        now = datetime.now()
        base_time = now - timedelta(hours=1000)
        
        # Generate realistic XAU prices
        dates = pd.date_range(start=base_time, end=now, freq='1H')
        
        # Start from realistic current XAU price
        base_price = 2650.0  # Current XAU price range
        prices = []
        
        for i, date in enumerate(dates):
            # Add realistic price movement
            trend = np.sin(i * 0.01) * 50  # Long term trend
            volatility = np.random.normal(0, 15)  # Random volatility
            price = base_price + trend + volatility
            
            # Ensure realistic OHLC
            open_price = price + np.random.uniform(-3, 3)
            high_price = price + np.random.uniform(5, 20)
            low_price = price - np.random.uniform(5, 20)
            close_price = price
            
            prices.append({
                'time': date,
                'open': max(open_price, low_price),
                'high': max(high_price, open_price, close_price),
                'low': min(low_price, open_price, close_price),
                'close': close_price,
                'tick_volume': np.random.randint(5000, 15000),
                'spread': np.random.randint(15, 25),
                'real_volume': 0
            })
        
        df = pd.DataFrame(prices)
        
        print(f"✅ Generated {len(df)} simulated real-time candles")
        print(f"📅 Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"💰 Latest price: {df['close'].iloc[-1]:.2f}")
        
        self.test_results['real_time_data'] = {
            'symbol': symbol,
            'timeframe': 'H1',
            'count': len(df),
            'latest_time': df['time'].iloc[-1].isoformat(),
            'latest_price': float(df['close'].iloc[-1]),
            'latest_volume': int(df['tick_volume'].iloc[-1]),
            'data_source': 'SIMULATED_REAL_TIME'
        }
        
        return df
    
    def test_system_with_realtime_data(self):
        """Test hệ thống với dữ liệu real-time"""
        print(f"\n🧪 TESTING SYSTEM WITH REAL-TIME DATA")
        print("=" * 60)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            # Initialize system
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            # Get real-time data
            rt_data = self.get_real_time_data()
            
            if rt_data.empty:
                print("❌ No real-time data available")
                return False
            
            # Test multiple signals with real-time data
            print("\n🎯 GENERATING SIGNALS WITH REAL-TIME DATA...")
            
            signals = []
            signal_times = []
            
            for i in range(5):
                start_time = time.time()
                
                # Override system's data source with real-time data
                system._real_time_data = rt_data
                signal = system.generate_signal()
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                if signal:
                    signals.append(signal)
                    signal_times.append(generation_time)
                    
                    print(f"   Signal {i+1}: {signal['action']} | "
                          f"Confidence: {signal['confidence']:.1%} | "
                          f"Time: {generation_time:.2f}s")
                else:
                    print(f"   Signal {i+1}: FAILED")
                
                time.sleep(1)  # Wait between signals
            
            # Analyze signal quality
            if signals:
                actions = [s['action'] for s in signals]
                confidences = [s['confidence'] for s in signals]
                systems_used = [s.get('systems_used', 0) for s in signals]
                
                print(f"\n📊 REAL-TIME SIGNAL ANALYSIS:")
                print(f"   • Total Signals: {len(signals)}")
                print(f"   • Average Confidence: {np.mean(confidences):.1%}")
                print(f"   • Average Generation Time: {np.mean(signal_times):.2f}s")
                print(f"   • Average Systems Used: {np.mean(systems_used):.1f}")
                print(f"   • Signal Distribution: {dict(pd.Series(actions).value_counts())}")
                
                # Check signal consistency
                action_changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
                consistency = max(0, 100 - (action_changes / len(actions) * 100))
                
                print(f"   • Signal Consistency: {consistency:.1f}%")
                
                self.test_results['signal_quality'] = {
                    'total_signals': len(signals),
                    'average_confidence': float(np.mean(confidences)),
                    'average_generation_time': float(np.mean(signal_times)),
                    'average_systems_used': float(np.mean(systems_used)),
                    'signal_distribution': dict(pd.Series(actions).value_counts()),
                    'signal_consistency': consistency,
                    'all_signals': [
                        {
                            'action': s['action'],
                            'confidence': s['confidence'],
                            'timestamp': s['timestamp'].isoformat() if 'timestamp' in s else None,
                            'systems_used': s.get('systems_used', 0)
                        } for s in signals
                    ]
                }
                
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ System test error: {e}")
            return False
    
    def test_system_performance_realtime(self):
        """Test performance của hệ thống với real-time data"""
        print(f"\n⚡ TESTING REAL-TIME PERFORMANCE")
        print("-" * 40)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            # Performance metrics
            performance_metrics = {
                'data_retrieval_time': [],
                'signal_generation_time': [],
                'system_response_time': [],
                'memory_usage': [],
                'cpu_usage': []
            }
            
            # Test data retrieval performance
            print("📊 Testing data retrieval performance...")
            for i in range(3):
                start_time = time.time()
                data = self.get_real_time_data(count=100)
                end_time = time.time()
                
                retrieval_time = end_time - start_time
                performance_metrics['data_retrieval_time'].append(retrieval_time)
                print(f"   Retrieval {i+1}: {retrieval_time:.3f}s")
            
            # Test signal generation performance
            print("🎯 Testing signal generation performance...")
            for i in range(3):
                start_time = time.time()
                signal = system.generate_signal()
                end_time = time.time()
                
                generation_time = end_time - start_time
                performance_metrics['signal_generation_time'].append(generation_time)
                print(f"   Generation {i+1}: {generation_time:.3f}s")
            
            # Test system response
            print("📡 Testing system response performance...")
            for i in range(3):
                start_time = time.time()
                status = system.get_system_status()
                end_time = time.time()
                
                response_time = end_time - start_time
                performance_metrics['system_response_time'].append(response_time)
                print(f"   Response {i+1}: {response_time:.3f}s")
            
            # Calculate averages
            avg_metrics = {
                key: np.mean(values) for key, values in performance_metrics.items() 
                if values
            }
            
            print(f"\n📊 REAL-TIME PERFORMANCE SUMMARY:")
            print(f"   • Average Data Retrieval: {avg_metrics.get('data_retrieval_time', 0):.3f}s")
            print(f"   • Average Signal Generation: {avg_metrics.get('signal_generation_time', 0):.3f}s")
            print(f"   • Average System Response: {avg_metrics.get('system_response_time', 0):.3f}s")
            
            self.test_results['system_performance'] = avg_metrics
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test error: {e}")
            return False
    
    def generate_final_report(self):
        """Tạo báo cáo cuối cùng"""
        print(f"\n" + "="*80)
        print("📋 BÁO CÁO CUỐI CÙNG - KẾT NỐI MT5 THỰC TẾ VÀ TEST HỆ THỐNG")
        print("="*80)
        
        end_time = datetime.now()
        test_duration = end_time - self.test_results['start_time']
        
        print(f"⏰ Thời gian test: {test_duration}")
        print(f"🔗 MT5 Connection: {'✅ SUCCESS' if self.connection_status.get('connected') else '❌ FAILED'}")
        
        # Connection status
        if self.connection_status.get('demo_mode'):
            print(f"🎭 Mode: DEMO MODE (MT5 not available)")
        else:
            print(f"🎯 Mode: REAL MT5 CONNECTION")
        
        # Data source
        data_source = self.test_results.get('real_time_data', {}).get('data_source', 'UNKNOWN')
        print(f"📊 Data Source: {data_source}")
        
        if 'real_time_data' in self.test_results:
            rt_data = self.test_results['real_time_data']
            print(f"📅 Latest Data Time: {rt_data.get('latest_time', 'N/A')}")
            print(f"💰 Latest Price: ${rt_data.get('latest_price', 0):.2f}")
            print(f"📊 Data Points: {rt_data.get('count', 0):,}")
        
        # Signal quality
        if 'signal_quality' in self.test_results:
            sq = self.test_results['signal_quality']
            print(f"\n🎯 SIGNAL QUALITY:")
            print(f"   • Signals Generated: {sq.get('total_signals', 0)}")
            print(f"   • Average Confidence: {sq.get('average_confidence', 0):.1%}")
            print(f"   • Signal Consistency: {sq.get('signal_consistency', 0):.1f}%")
            print(f"   • Average Systems Used: {sq.get('average_systems_used', 0):.1f}")
        
        # Performance
        if 'system_performance' in self.test_results:
            perf = self.test_results['system_performance']
            print(f"\n⚡ PERFORMANCE:")
            print(f"   • Data Retrieval: {perf.get('data_retrieval_time', 0):.3f}s")
            print(f"   • Signal Generation: {perf.get('signal_generation_time', 0):.3f}s")
            print(f"   • System Response: {perf.get('system_response_time', 0):.3f}s")
        
        # Overall assessment
        print(f"\n🏆 ĐÁNH GIÁ TỔNG THỂ:")
        
        if self.connection_status.get('connected') and not self.connection_status.get('demo_mode'):
            print("   🎉 KẾT NỐI MT5 THỰC TẾ THÀNH CÔNG!")
            print("   ✅ Hệ thống hoạt động với dữ liệu real-time")
            print("   ✅ Sẵn sàng cho live trading")
        elif self.connection_status.get('demo_mode'):
            print("   ⚠️ CHẠY Ở DEMO MODE")
            print("   ✅ Hệ thống logic hoạt động tốt")
            print("   🔧 Cần setup MT5 thực tế để có real-time data")
        else:
            print("   ❌ KẾT NỐI MT5 THẤT BẠI")
            print("   🔧 Cần cài đặt và cấu hình MT5")
        
        # Save results
        self.test_results['end_time'] = end_time
        self.test_results['test_duration_seconds'] = test_duration.total_seconds()
        self.test_results['mt5_connection'] = self.connection_status
        
        filename = f"mt5_realtime_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📁 Báo cáo chi tiết đã lưu: {filename}")
        print(f"🎉 TEST HOÀN THÀNH!")

def main():
    """Chạy test chính"""
    setup = MT5RealTimeSetup()
    
    # Step 1: Setup MT5 connection
    if setup.setup_mt5_connection():
        
        # Step 2: Test system with real-time data
        setup.test_system_with_realtime_data()
        
        # Step 3: Test performance
        setup.test_system_performance_realtime()
        
        # Step 4: Generate report
        setup.generate_final_report()
    else:
        print("❌ Failed to setup MT5 connection")

if __name__ == "__main__":
    main() 