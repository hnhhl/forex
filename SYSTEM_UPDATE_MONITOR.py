import os
import time
import json
from datetime import datetime
import psutil

print("📊 SYSTEM UPDATE MONITOR - ULTIMATE SYSTEM V5.0")
print("="*60)

def check_process_status():
    """Check if Python process is running"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'ULTIMATE_SYSTEM_V5_COMPLETE_UPDATE.py' in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'memory_mb': proc.info['memory_info'].rss / (1024*1024),
                        'cpu_percent': proc.info['cpu_percent'],
                        'cmdline': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return python_processes

def check_output_files():
    """Check for output files"""
    output_files = []
    
    # Check for training results
    for file in os.listdir('.'):
        if file.startswith('ULTIMATE_SYSTEM_V5_TRAINING_RESULTS_'):
            size = os.path.getsize(file) / 1024  # KB
            mtime = datetime.fromtimestamp(os.path.getmtime(file))
            output_files.append({
                'file': file,
                'size_kb': size,
                'modified': mtime.strftime('%H:%M:%S')
            })
    
    return sorted(output_files, key=lambda x: x['modified'], reverse=True)

def monitor_system_update():
    """Monitor the system update process"""
    start_time = datetime.now()
    last_check = start_time
    
    print(f"🚀 Monitoring started at: {start_time.strftime('%H:%M:%S')}")
    print("-"*60)
    
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        
        # Check processes
        processes = check_process_status()
        
        # Check output files
        output_files = check_output_files()
        
        print(f"\n⏰ {current_time.strftime('%H:%M:%S')} | Elapsed: {str(elapsed).split('.')[0]}")
        print("-"*40)
        
        if processes:
            print("🔄 ACTIVE PROCESSES:")
            for proc in processes:
                print(f"  PID {proc['pid']}: {proc['memory_mb']:.1f}MB RAM, {proc['cpu_percent']:.1f}% CPU")
        else:
            print("❌ No active update processes found")
        
        if output_files:
            print(f"\n📁 OUTPUT FILES ({len(output_files)}):")
            for file_info in output_files[:3]:  # Show latest 3
                print(f"  📊 {file_info['file']}")
                print(f"      Size: {file_info['size_kb']:.1f}KB | Modified: {file_info['modified']}")
        else:
            print("\n📁 No output files yet")
        
        # Check if process completed
        if not processes and output_files:
            print(f"\n🎉 SYSTEM UPDATE COMPLETED!")
            print(f"⏱️ Total time: {str(elapsed).split('.')[0]}")
            
            # Try to read latest results
            latest_file = output_files[0]['file']
            try:
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                print(f"\n📊 FINAL RESULTS SUMMARY:")
                print(f"  🎯 System Version: {results.get('system_version', 'N/A')}")
                print(f"  📈 Data Samples: {results.get('data_info', {}).get('samples', 'N/A')}")
                print(f"  🧠 Features: {results.get('data_info', {}).get('features', 'N/A')}")
                
                if 'performance_metrics' in results and results['performance_metrics']:
                    metrics = results['performance_metrics']
                    print(f"  💰 Total Return: {metrics.get('total_return', 0)*100:.2f}%")
                    print(f"  📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"  🏆 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                
                print(f"\n✅ Complete results saved in: {latest_file}")
                
            except Exception as e:
                print(f"⚠️ Could not read results file: {e}")
            
            break
        
        elif not processes and not output_files:
            # Check if enough time has passed
            if elapsed.total_seconds() > 300:  # 5 minutes
                print(f"\n⚠️ No activity detected for 5+ minutes")
                print(f"Process may have failed or completed without output")
                break
        
        # Wait before next check
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_system_update()
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
    
    print(f"\n📊 Monitor session ended at: {datetime.now().strftime('%H:%M:%S')}")
    print("="*60) 