import os
import json
import psutil
from datetime import datetime

print("📊 ULTIMATE SYSTEM V5.0 - UPDATE STATUS CHECK")
print("="*55)

# Check active processes
print("\n🔄 ACTIVE PROCESSES:")
python_procs = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
    try:
        if proc.info['name'] == 'python.exe':
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'ULTIMATE_SYSTEM_V5' in cmdline:
                memory_mb = proc.info['memory_info'].rss / (1024*1024)
                cpu_pct = proc.info['cpu_percent']
                python_procs.append({
                    'pid': proc.info['pid'],
                    'memory': memory_mb,
                    'cpu': cpu_pct,
                    'cmd': cmdline.split('\\')[-1] if '\\' in cmdline else cmdline
                })
    except:
        continue

if python_procs:
    for proc in python_procs:
        print(f"  ✅ PID {proc['pid']}: {proc['memory']:.1f}MB RAM, {proc['cpu']:.1f}% CPU")
        print(f"     Command: {proc['cmd']}")
    print(f"\n🎯 System is actively updating...")
else:
    print("  ❌ No active update processes found")

# Check for result files
print(f"\n📁 RESULT FILES:")
result_files = []
for file in os.listdir('.'):
    if file.startswith('ULTIMATE_SYSTEM_V5_TRAINING_RESULTS_'):
        size_kb = os.path.getsize(file) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(file))
        result_files.append({
            'file': file,
            'size': size_kb,
            'time': mtime
        })

if result_files:
    result_files.sort(key=lambda x: x['time'], reverse=True)
    latest = result_files[0]
    
    print(f"  ✅ Found {len(result_files)} result file(s)")
    print(f"  📊 Latest: {latest['file']}")
    print(f"  💾 Size: {latest['size']:.1f}KB")
    print(f"  ⏰ Created: {latest['time'].strftime('%H:%M:%S')}")
    
    # Try to read results
    try:
        with open(latest['file'], 'r') as f:
            results = json.load(f)
        
        print(f"\n🎉 SYSTEM UPDATE COMPLETED!")
        print(f"  🎯 Version: {results.get('system_version', 'N/A')}")
        print(f"  📈 Data: {results.get('data_info', {}).get('samples', 'N/A')} samples")
        print(f"  🧠 Features: {results.get('data_info', {}).get('features', 'N/A')}")
        
        if 'performance_metrics' in results and results['performance_metrics']:
            metrics = results['performance_metrics']
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"  💰 Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  📉 Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  🏆 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"  💵 Final Capital: ${metrics.get('final_capital', 0):,.2f}")
        
        # Check training components
        if 'training_history' in results:
            history = results['training_history']
            print(f"\n🔧 TRAINING COMPONENTS:")
            
            if 'dqn' in history:
                print(f"  ✅ DQN Agent: Trained (ε={history['dqn'].get('final_epsilon', 'N/A')})")
            
            if 'meta_learning' in history:
                meta = history['meta_learning']
                print(f"  ✅ Meta Learning: {meta.get('train_accuracy', 0):.3f} train, {meta.get('test_accuracy', 0):.3f} test")
            
            if 'cross_validation' in history:
                cv = history['cross_validation']
                if 'accuracy' in cv:
                    print(f"  ✅ Cross-Validation: {cv['accuracy']['mean']:.3f} ± {cv['accuracy']['std']:.3f}")
            
            if 'backtest' in history:
                print(f"  ✅ Backtesting: Completed")
        
    except Exception as e:
        print(f"  ⚠️ Could not read results: {e}")
        
else:
    print("  ⏳ No result files found - system still updating")

# Check verification files
verify_files = [f for f in os.listdir('.') if f.startswith('QUICK_FIX_VERIFICATION_')]
if verify_files:
    latest_verify = max(verify_files)
    print(f"\n✅ Previous verification completed: {latest_verify}")

print(f"\n{'='*55}")

if result_files and not python_procs:
    print("🎉 SYSTEM UPDATE COMPLETED SUCCESSFULLY!")
elif python_procs:
    print("🔄 SYSTEM UPDATE IN PROGRESS...")
else:
    print("⚠️ SYSTEM UPDATE STATUS UNCLEAR")

print(f"⏰ Status checked at: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*55}") 