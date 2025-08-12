#!/usr/bin/env python3
"""
CORE SYSTEM DEEP ANALYSIS
Phân tích sâu và demo hoạt động của hệ thống chính
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def analyze_system_architecture():
    """Phân tích kiến trúc hệ thống"""
    print('🏗️ ANALYZING SYSTEM ARCHITECTURE...')
    print('=' * 60)
    
    # Analyze core files
    core_files = {
        'Master Integration': 'src/core/integration/master_system.py',
        'Ultimate XAU System': 'src/core/ultimate_xau_system.py', 
        'AI Master Integration': 'src/core/integration/ai_master_integration.py',
        'Neural Ensemble': 'src/core/ai/neural_ensemble.py',
        'Advanced AI Ensemble': 'src/core/advanced_ai_ensemble.py'
    }
    
    architecture_analysis = {}
    
    for system_name, file_path in core_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            analysis = {
                'exists': True,
                'size_kb': len(content) / 1024,
                'lines': len(content.split('\n')),
                'classes': content.count('class '),
                'functions': content.count('def '),
                'imports': content.count('import '),
                'comments': content.count('#'),
                'docstrings': content.count('"""')
            }
            
            print(f'📁 {system_name}:')
            print(f'   Size: {analysis["size_kb"]:.1f} KB')
            print(f'   Lines: {analysis["lines"]:,}')
            print(f'   Classes: {analysis["classes"]}')
            print(f'   Functions: {analysis["functions"]}')
            print(f'   Imports: {analysis["imports"]}')
            
        else:
            analysis = {'exists': False}
            print(f'❌ {system_name}: File not found')
        
        architecture_analysis[system_name] = analysis
    
    return architecture_analysis

def test_master_integration_system():
    """Test Master Integration System"""
    print('\n🎯 TESTING MASTER INTEGRATION SYSTEM...')
    print('=' * 60)
    
    try:
        from src.core.integration.master_system import MasterIntegrationSystem, SystemConfig, SystemMode, IntegrationLevel
        
        # Create configuration
        config = SystemConfig(
            mode=SystemMode.SIMULATION,
            integration_level=IntegrationLevel.MODERATE,
            initial_balance=10000.0,
            use_neural_ensemble=True,
            use_reinforcement_learning=True
        )
        
        print('✅ Configuration created successfully')
        print(f'   Mode: {config.mode.value}')
        print(f'   Integration Level: {config.integration_level.value}')
        print(f'   Initial Balance: ${config.initial_balance:,.2f}')
        
        # Initialize system
        print('\n🔧 Initializing Master Integration System...')
        master_system = MasterIntegrationSystem(config)
        
        print('✅ Master Integration System initialized')
        print(f'   Components loaded: {len(master_system.components)}')
        
        # Get system status
        status = master_system.get_system_status()
        print(f'\n📊 System Status:')
        print(f'   Mode: {status["mode"]}')
        print(f'   Balance: ${status["total_balance"]:,.2f}')
        print(f'   Components: {len(status["components_status"])}')
        
        # Test with sample market data
        print(f'\n📈 Testing with sample market data...')
        from src.core.integration.master_system import MarketData
        
        sample_data = MarketData(
            timestamp=datetime.now(),
            symbol='XAUUSD',
            price=2650.50,
            high=2655.00,
            low=2645.00,
            volume=1500,
            technical_indicators={'rsi': 65.5, 'macd': 0.25}
        )
        
        master_system.add_market_data(sample_data)
        print('✅ Sample market data processed')
        
        return master_system
        
    except Exception as e:
        print(f'❌ Error testing Master Integration System: {e}')
        return None

def test_ultimate_xau_system():
    """Test Ultimate XAU System"""
    print('\n🚀 TESTING ULTIMATE XAU SYSTEM...')
    print('=' * 60)
    
    try:
        from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        # Create configuration
        config = SystemConfig(
            symbol='XAUUSD',
            live_trading=False,
            paper_trading=True,
            enable_kelly_criterion=True,
            enable_position_sizing=True,
            continuous_learning=True
        )
        
        print('✅ Ultimate XAU System configuration created')
        print(f'   Symbol: {config.symbol}')
        print(f'   Live Trading: {config.live_trading}')
        print(f'   Paper Trading: {config.paper_trading}')
        print(f'   Kelly Criterion: {config.enable_kelly_criterion}')
        
        # Initialize system (this might take some time)
        print('\n🔧 Initializing Ultimate XAU System (this may take a moment)...')
        start_time = time.time()
        
        ultimate_system = UltimateXAUSystem(config)
        
        init_time = time.time() - start_time
        print(f'✅ Ultimate XAU System initialized in {init_time:.2f}s')
        
        # Get system status
        status = ultimate_system.get_system_status()
        print(f'\n📊 Ultimate XAU System Status:')
        print(f'   Version: {status["version"]}')
        print(f'   Status: {status["status"]}')
        print(f'   Active Systems: {status["systems_active"]}/{status["systems_total"]}')
        print(f'   Production Mode: {status["production_mode"]}')
        print(f'   Trading Active: {status["trading_active"]}')
        print(f'   Learning Active: {status["learning_active"]}')
        
        return ultimate_system
        
    except Exception as e:
        print(f'❌ Error testing Ultimate XAU System: {e}')
        return None

def test_ai_master_integration():
    """Test AI Master Integration System"""
    print('\n🤖 TESTING AI MASTER INTEGRATION...')
    print('=' * 60)
    
    try:
        from src.core.integration.ai_master_integration import AIMasterIntegrationSystem, AISystemConfig
        
        # Create AI configuration
        ai_config = AISystemConfig(
            enable_neural_ensemble=True,
            enable_reinforcement_learning=True,
            enable_meta_learning=True,
            neural_confidence_threshold=0.7,
            min_confidence_threshold=0.6
        )
        
        print('✅ AI System configuration created')
        print(f'   Neural Ensemble: {ai_config.enable_neural_ensemble}')
        print(f'   Reinforcement Learning: {ai_config.enable_reinforcement_learning}')
        print(f'   Meta Learning: {ai_config.enable_meta_learning}')
        
        # Initialize AI system
        print('\n🔧 Initializing AI Master Integration...')
        ai_system = AIMasterIntegrationSystem(ai_config)
        
        print('✅ AI Master Integration initialized')
        
        # Test with sample data
        print(f'\n📊 Testing AI prediction capabilities...')
        from src.core.integration.ai_master_integration import AIMarketData
        
        sample_ai_data = AIMarketData(
            timestamp=datetime.now(),
            symbol='XAUUSD',
            price=2650.50,
            volume=1500,
            volatility=0.15,
            features=np.random.randn(95)  # 95 features as configured
        )
        
        # Get AI prediction
        prediction = ai_system.process_market_data(sample_ai_data)
        
        if prediction:
            print('✅ AI prediction generated successfully')
            print(f'   Action: {prediction.action}')
            print(f'   Confidence: {prediction.confidence:.3f}')
            print(f'   Position Size: {prediction.position_size:.3f}')
        else:
            print('⚠️ AI prediction not generated (insufficient data)')
        
        return ai_system
        
    except Exception as e:
        print(f'❌ Error testing AI Master Integration: {e}')
        return None

def demo_integrated_workflow():
    """Demo quy trình tích hợp của toàn hệ thống"""
    print('\n🔄 DEMO INTEGRATED WORKFLOW...')
    print('=' * 60)
    
    try:
        # 1. Initialize systems
        print('1️⃣ Initializing all systems...')
        
        from src.core.integration.master_system import MasterIntegrationSystem, SystemConfig, SystemMode
        
        config = SystemConfig(
            mode=SystemMode.SIMULATION,
            initial_balance=10000.0,
            use_neural_ensemble=True,
            use_reinforcement_learning=True
        )
        
        master_system = MasterIntegrationSystem(config)
        print('   ✅ Master system initialized')
        
        # 2. Simulate market data flow
        print('\n2️⃣ Simulating market data flow...')
        
        from src.core.integration.master_system import MarketData
        
        # Generate sample market data sequence
        base_price = 2650.0
        for i in range(5):
            # Simulate price movement
            price_change = np.random.randn() * 2.0
            current_price = base_price + price_change
            
            market_data = MarketData(
                timestamp=datetime.now() + timedelta(minutes=i),
                symbol='XAUUSD',
                price=current_price,
                high=current_price + abs(np.random.randn()),
                low=current_price - abs(np.random.randn()),
                volume=1000 + np.random.randint(0, 1000),
                technical_indicators={
                    'rsi': 50 + np.random.randn() * 15,
                    'macd': np.random.randn() * 0.5,
                    'bb_upper': current_price + 5,
                    'bb_lower': current_price - 5
                }
            )
            
            master_system.add_market_data(market_data)
            print(f'   📊 Data point {i+1}: ${current_price:.2f}')
            
            time.sleep(0.5)  # Simulate real-time
        
        # 3. Get recent signals
        print('\n3️⃣ Analyzing generated signals...')
        recent_signals = master_system.get_recent_signals(hours=1)
        
        if recent_signals:
            print(f'   📡 Generated {len(recent_signals)} signals')
            for i, signal in enumerate(recent_signals[-3:], 1):  # Show last 3
                print(f'   Signal {i}: {signal.signal_type} (conf: {signal.confidence:.3f})')
        else:
            print('   ⚠️ No signals generated yet')
        
        # 4. System performance
        print('\n4️⃣ System performance metrics...')
        status = master_system.get_system_status()
        
        print(f'   💰 Balance: ${status["total_balance"]:,.2f}')
        print(f'   📊 Open Positions: {status["total_positions"]}')
        print(f'   🎯 AI Systems Active: {status.get("neural_ensemble_active", False)} / {status.get("rl_agent_active", False)}')
        print(f'   ⚡ Components Status: {sum(status["components_status"].values())}/{len(status["components_status"])} active')
        
        return {
            'master_system': master_system,
            'signals_generated': len(recent_signals) if recent_signals else 0,
            'system_health': sum(status["components_status"].values()) / len(status["components_status"]) * 100
        }
        
    except Exception as e:
        print(f'❌ Error in integrated workflow demo: {e}')
        return None

def generate_analysis_report(architecture_analysis, demo_results):
    """Tạo báo cáo phân tích tổng hợp"""
    print('\n📋 GENERATING ANALYSIS REPORT...')
    print('=' * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'Core System Deep Analysis',
        'architecture_analysis': architecture_analysis,
        'demo_results': demo_results,
        'summary': {}
    }
    
    # Calculate summary metrics
    total_files = len(architecture_analysis)
    existing_files = sum(1 for analysis in architecture_analysis.values() if analysis.get('exists', False))
    
    total_lines = sum(analysis.get('lines', 0) for analysis in architecture_analysis.values() if analysis.get('exists', False))
    total_classes = sum(analysis.get('classes', 0) for analysis in architecture_analysis.values() if analysis.get('exists', False))
    total_functions = sum(analysis.get('functions', 0) for analysis in architecture_analysis.values() if analysis.get('exists', False))
    
    report['summary'] = {
        'total_core_files': total_files,
        'existing_files': existing_files,
        'file_coverage': existing_files / total_files * 100,
        'total_lines_of_code': total_lines,
        'total_classes': total_classes,
        'total_functions': total_functions,
        'system_complexity_score': (total_classes + total_functions) / 10,  # Rough complexity metric
        'demo_success': demo_results is not None,
        'system_health': demo_results.get('system_health', 0) if demo_results else 0
    }
    
    # Save report
    os.makedirs('analysis_reports', exist_ok=True)
    report_file = f'analysis_reports/core_system_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print(f'📊 ANALYSIS SUMMARY:')
    print(f'   Core Files: {existing_files}/{total_files} ({report["summary"]["file_coverage"]:.1f}%)')
    print(f'   Lines of Code: {total_lines:,}')
    print(f'   Classes: {total_classes}')
    print(f'   Functions: {total_functions}')
    print(f'   Complexity Score: {report["summary"]["complexity_score"]:.1f}/10')
    
    if demo_results:
        print(f'   Demo Success: ✅')
        print(f'   System Health: {report["summary"]["system_health"]:.1f}%')
        print(f'   Signals Generated: {demo_results.get("signals_generated", 0)}')
    else:
        print(f'   Demo Success: ❌')
    
    print(f'\n💾 Report saved to: {report_file}')
    return report_file

def main():
    """Main analysis function"""
    print('🔍 CORE SYSTEM DEEP ANALYSIS')
    print('=' * 80)
    print('🎯 Analyzing Ultimate XAU System V4.0 core architecture')
    print(f'🕐 Started at: {datetime.now()}')
    print()
    
    # 1. Analyze architecture
    architecture_analysis = analyze_system_architecture()
    
    # 2. Test individual systems
    master_system = test_master_integration_system()
    ultimate_system = test_ultimate_xau_system() 
    ai_system = test_ai_master_integration()
    
    # 3. Demo integrated workflow
    demo_results = demo_integrated_workflow()
    
    # 4. Generate comprehensive report
    report_file = generate_analysis_report(architecture_analysis, demo_results)
    
    print(f'\n🎉 CORE SYSTEM ANALYSIS COMPLETED!')
    print('✅ Architecture analyzed')
    print('✅ Systems tested individually') 
    print('✅ Integrated workflow demonstrated')
    print('✅ Comprehensive report generated')
    
    print(f'\n📊 KEY FINDINGS:')
    if master_system:
        print('✅ Master Integration System: OPERATIONAL')
    else:
        print('❌ Master Integration System: ISSUES DETECTED')
    
    if ultimate_system:
        print('✅ Ultimate XAU System: OPERATIONAL')
    else:
        print('❌ Ultimate XAU System: ISSUES DETECTED')
    
    if ai_system:
        print('✅ AI Master Integration: OPERATIONAL') 
    else:
        print('❌ AI Master Integration: ISSUES DETECTED')
    
    if demo_results:
        print('✅ Integrated Workflow: SUCCESSFUL')
    else:
        print('❌ Integrated Workflow: FAILED')
    
    print(f'\n🕐 Completed at: {datetime.now()}')

if __name__ == "__main__":
    main() 