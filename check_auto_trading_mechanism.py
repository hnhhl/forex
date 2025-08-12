#!/usr/bin/env python3
"""
🔍 CHECK AUTO TRADING MECHANISM - Kiểm tra cơ chế tự động vào lệnh
Phân tích xem hệ thống đã có đầy đủ tính năng auto trading chưa
"""

import sys
import os
import re
import json
from datetime import datetime
from typing import Dict, List, Any

sys.path.append('src')

class AutoTradingAnalyzer:
    """Class phân tích cơ chế auto trading"""
    
    def __init__(self):
        self.system_file = "src/core/ultimate_xau_system.py"
        self.findings = []
        self.auto_trading_components = {}
        
    def analyze_signal_generation(self):
        """Phân tích khả năng tạo signal"""
        print("🔍 ANALYZING SIGNAL GENERATION")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        signal_components = {
            'generate_signal_method': bool(re.search(r'def generate_signal', content)),
            'signal_actions': len(re.findall(r'action.*[\'"](?:BUY|SELL|HOLD)[\'"]', content)),
            'confidence_calculation': bool(re.search(r'confidence.*=', content)),
            'ensemble_logic': bool(re.search(r'ensemble.*signal', content)),
            'risk_filters': bool(re.search(r'risk.*filter', content)),
        }
        
        print("📊 Signal Generation Components:")
        for component, status in signal_components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {status}")
        
        self.auto_trading_components['signal_generation'] = signal_components
        return signal_components
    
    def analyze_order_execution(self):
        """Phân tích khả năng thực thi lệnh"""
        print(f"\n🔍 ANALYZING ORDER EXECUTION")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        execution_components = {
            'place_order_method': bool(re.search(r'def place_order', content)),
            'mt5_integration': bool(re.search(r'mt5\.order_send|MetaTrader5', content)),
            'order_types': len(re.findall(r'ORDER_TYPE_(?:BUY|SELL)', content)),
            'position_management': bool(re.search(r'position.*management|manage.*position', content)),
            'stop_loss_take_profit': bool(re.search(r'stop_loss|take_profit|sl.*tp', content)),
            'volume_calculation': bool(re.search(r'volume.*calculation|calculate.*volume', content)),
        }
        
        print("📊 Order Execution Components:")
        for component, status in execution_components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {status}")
        
        # Check for specific MT5 functions
        mt5_functions = [
            'mt5.initialize', 'mt5.login', 'mt5.order_send', 
            'mt5.positions_get', 'mt5.orders_get', 'mt5.account_info'
        ]
        
        mt5_integration_details = {}
        for func in mt5_functions:
            mt5_integration_details[func] = bool(re.search(func.replace('.', r'\.'), content))
        
        print(f"\n📊 MT5 Integration Details:")
        for func, available in mt5_integration_details.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {func}: {available}")
        
        self.auto_trading_components['order_execution'] = execution_components
        self.auto_trading_components['mt5_integration'] = mt5_integration_details
        return execution_components
    
    def analyze_risk_management(self):
        """Phân tích quản lý rủi ro"""
        print(f"\n🔍 ANALYZING RISK MANAGEMENT")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        risk_components = {
            'position_sizing': bool(re.search(r'position.*siz|kelly.*criterion', content)),
            'risk_per_trade': bool(re.search(r'risk_per_trade|risk.*limit', content)),
            'max_drawdown': bool(re.search(r'max.*drawdown|drawdown.*limit', content)),
            'correlation_check': bool(re.search(r'correlation.*check|max.*correlation', content)),
            'daily_risk_limit': bool(re.search(r'daily.*risk|max_daily', content)),
            'portfolio_risk': bool(re.search(r'portfolio.*risk|risk.*portfolio', content)),
        }
        
        print("📊 Risk Management Components:")
        for component, status in risk_components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {status}")
        
        self.auto_trading_components['risk_management'] = risk_components
        return risk_components
    
    def analyze_automation_logic(self):
        """Phân tích logic tự động hóa"""
        print(f"\n🔍 ANALYZING AUTOMATION LOGIC")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        automation_components = {
            'live_trading_mode': bool(re.search(r'live_trading.*=.*True|live.*mode', content)),
            'continuous_monitoring': bool(re.search(r'continuous.*monitoring|monitor.*continuous', content)),
            'auto_execution': bool(re.search(r'auto.*execution|execute.*auto', content)),
            'scheduler_integration': bool(re.search(r'scheduler|BackgroundScheduler', content)),
            'real_time_data': bool(re.search(r'real.*time.*data|realtime', content)),
            'decision_pipeline': bool(re.search(r'pipeline.*decision|trading.*pipeline', content)),
        }
        
        print("📊 Automation Logic Components:")
        for component, status in automation_components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {status}")
        
        # Check for trading loop/pipeline methods
        trading_methods = [
            'start_trading', 'stop_trading', 'trading_loop', 
            'process_market_data', 'execute_trade', 'monitor_positions'
        ]
        
        trading_method_details = {}
        for method in trading_methods:
            trading_method_details[method] = bool(re.search(f'def {method}', content))
        
        print(f"\n📊 Trading Method Details:")
        for method, available in trading_method_details.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {method}: {available}")
        
        self.auto_trading_components['automation_logic'] = automation_components
        self.auto_trading_components['trading_methods'] = trading_method_details
        return automation_components
    
    def analyze_configuration_settings(self):
        """Phân tích cài đặt cấu hình"""
        print(f"\n🔍 ANALYZING CONFIGURATION SETTINGS")
        print("-" * 40)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        config_components = {
            'live_trading_flag': bool(re.search(r'live_trading.*:.*bool', content)),
            'paper_trading_flag': bool(re.search(r'paper_trading.*:.*bool', content)),
            'auto_rebalancing': bool(re.search(r'auto_rebalancing.*:.*bool', content)),
            'continuous_learning': bool(re.search(r'continuous_learning.*:.*bool', content)),
            'mt5_configuration': bool(re.search(r'mt5_login|mt5_password|mt5_server', content)),
            'risk_parameters': bool(re.search(r'risk_per_trade|max_daily_risk', content)),
        }
        
        print("📊 Configuration Components:")
        for component, status in config_components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {status}")
        
        # Extract actual configuration values
        config_values = {}
        config_patterns = {
            'live_trading': r'live_trading:\s*bool\s*=\s*(\w+)',
            'paper_trading': r'paper_trading:\s*bool\s*=\s*(\w+)',
            'auto_rebalancing': r'auto_rebalancing:\s*bool\s*=\s*(\w+)',
            'max_positions': r'max_positions:\s*int\s*=\s*(\d+)',
            'max_daily_trades': r'max_daily_trades:\s*int\s*=\s*(\d+)',
        }
        
        for key, pattern in config_patterns.items():
            match = re.search(pattern, content)
            if match:
                config_values[key] = match.group(1)
        
        print(f"\n📊 Current Configuration Values:")
        for key, value in config_values.items():
            print(f"   🔧 {key}: {value}")
        
        self.auto_trading_components['configuration'] = config_components
        self.auto_trading_components['config_values'] = config_values
        return config_components
    
    def check_safety_mechanisms(self):
        """Kiểm tra cơ chế an toàn"""
        print(f"\n🔍 CHECKING SAFETY MECHANISMS")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        safety_components = {
            'error_handling': len(re.findall(r'try:|except:', content)),
            'connection_monitoring': bool(re.search(r'connection.*monitor|monitor.*connection', content)),
            'position_limits': bool(re.search(r'max_positions|position.*limit', content)),
            'daily_limits': bool(re.search(r'max_daily|daily.*limit', content)),
            'emergency_stop': bool(re.search(r'emergency.*stop|stop.*emergency', content)),
            'validation_checks': bool(re.search(r'validate|validation', content)),
        }
        
        print("📊 Safety Mechanisms:")
        for component, status in safety_components.items():
            status_icon = "✅" if status else "❌"
            if component == 'error_handling':
                print(f"   {status_icon} {component}: {status} blocks")
            else:
                print(f"   {status_icon} {component}: {status}")
        
        self.auto_trading_components['safety_mechanisms'] = safety_components
        return safety_components
    
    def test_auto_trading_functionality(self):
        """Test chức năng auto trading"""
        print(f"\n🧪 TESTING AUTO TRADING FUNCTIONALITY")
        print("-" * 40)
        
        try:
            from core.ultimate_xau_system import UltimateXAUSystem
            
            # Test system initialization
            print("🔄 Testing system initialization...")
            system = UltimateXAUSystem()
            print("   ✅ System initialized successfully")
            
            # Check auto trading methods
            auto_methods = ['generate_signal', 'start_trading', 'stop_trading']
            method_status = {}
            
            for method in auto_methods:
                if hasattr(system, method):
                    method_status[method] = "✅ Available"
                    print(f"   ✅ {method}: Available")
                else:
                    method_status[method] = "❌ Missing"
                    print(f"   ❌ {method}: Missing")
            
            # Test signal generation
            if hasattr(system, 'generate_signal'):
                print("🔄 Testing signal generation...")
                signal = system.generate_signal()
                if isinstance(signal, dict) and 'action' in signal:
                    print(f"   ✅ Signal generated: {signal.get('action')} (confidence: {signal.get('confidence')})")
                    method_status['signal_generation_test'] = "✅ Working"
                else:
                    print(f"   ❌ Signal generation failed")
                    method_status['signal_generation_test'] = "❌ Failed"
            
            # Check configuration
            if hasattr(system, 'config'):
                config = system.config
                live_trading = getattr(config, 'live_trading', None)
                paper_trading = getattr(config, 'paper_trading', None)
                print(f"   📊 Live trading: {live_trading}")
                print(f"   📊 Paper trading: {paper_trading}")
                method_status['configuration_check'] = "✅ Available"
            else:
                method_status['configuration_check'] = "❌ Missing"
            
            self.auto_trading_components['functionality_test'] = method_status
            return True
            
        except Exception as e:
            print(f"   ❌ Testing failed: {e}")
            self.auto_trading_components['functionality_test'] = {"error": str(e)}
            return False
    
    def generate_auto_trading_assessment(self):
        """Tạo đánh giá tổng thể về auto trading"""
        print(f"\n📋 AUTO TRADING ASSESSMENT")
        print("=" * 35)
        
        # Calculate overall scores
        total_components = 0
        working_components = 0
        
        for category, components in self.auto_trading_components.items():
            if isinstance(components, dict):
                for component, status in components.items():
                    total_components += 1
                    if status == True or status == "✅ Available" or status == "✅ Working":
                        working_components += 1
        
        overall_score = (working_components / total_components * 100) if total_components > 0 else 0
        
        # Determine readiness level
        if overall_score >= 80:
            readiness = "🟢 READY FOR AUTO TRADING"
        elif overall_score >= 60:
            readiness = "🟡 PARTIALLY READY"
        else:
            readiness = "🔴 NOT READY"
        
        print(f"📊 Overall Score: {overall_score:.1f}% ({working_components}/{total_components})")
        print(f"🎯 Readiness Level: {readiness}")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        
        # Check critical components
        critical_components = [
            ('Signal Generation', 'signal_generation'),
            ('Order Execution', 'order_execution'), 
            ('Risk Management', 'risk_management'),
            ('Automation Logic', 'automation_logic'),
        ]
        
        for name, key in critical_components:
            if key in self.auto_trading_components:
                components = self.auto_trading_components[key]
                working = sum(1 for status in components.values() if status)
                total = len(components)
                score = working / total * 100 if total > 0 else 0
                status_icon = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
                print(f"   {status_icon} {name}: {score:.1f}% ({working}/{total})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if overall_score < 80:
            print("   🔧 System needs additional development for full auto trading")
            
            if 'order_execution' in self.auto_trading_components:
                exec_score = sum(1 for s in self.auto_trading_components['order_execution'].values() if s)
                if exec_score < 4:
                    print("   📝 Enhance order execution capabilities")
            
            if 'automation_logic' in self.auto_trading_components:
                auto_score = sum(1 for s in self.auto_trading_components['automation_logic'].values() if s)
                if auto_score < 4:
                    print("   📝 Implement automation pipeline and scheduler")
            
            print("   📝 Add comprehensive testing for live trading")
            print("   📝 Implement additional safety mechanisms")
        else:
            print("   ✅ System appears ready for auto trading")
            print("   📝 Recommend thorough testing in paper trading mode first")
            print("   📝 Monitor performance closely during initial deployment")
        
        # Save assessment
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'readiness_level': readiness,
            'components': self.auto_trading_components,
            'working_components': working_components,
            'total_components': total_components
        }
        
        report_file = f"auto_trading_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(assessment, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Assessment saved: {report_file}")
        
        return assessment
    
    def run_comprehensive_analysis(self):
        """Chạy phân tích toàn diện"""
        print("🔍 COMPREHENSIVE AUTO TRADING ANALYSIS")
        print("=" * 50)
        print("🎯 Objective: Kiểm tra cơ chế tự động vào lệnh")
        print()
        
        # Run all analyses
        self.analyze_signal_generation()
        self.analyze_order_execution()
        self.analyze_risk_management()
        self.analyze_automation_logic()
        self.analyze_configuration_settings()
        self.check_safety_mechanisms()
        self.test_auto_trading_functionality()
        
        # Generate final assessment
        assessment = self.generate_auto_trading_assessment()
        
        return assessment

def main():
    """Main function"""
    analyzer = AutoTradingAnalyzer()
    assessment = analyzer.run_comprehensive_analysis()
    
    print(f"\n✅ AUTO TRADING ANALYSIS COMPLETED!")
    print(f"📊 Overall Score: {assessment['overall_score']:.1f}%")
    print(f"🎯 Readiness: {assessment['readiness_level']}")
    
    return assessment

if __name__ == "__main__":
    main() 