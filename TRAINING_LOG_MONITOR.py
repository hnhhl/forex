"""
🔍 TRAINING LOG MONITOR
Theo dõi training process để xem nó đang làm gì
"""

import psutil
import time
import os
from datetime import datetime

def monitor_training_process():
    """Monitor training process details"""
    print("🔍 TRAINING PROCESS LOG MONITOR")
    print("="*60)
    
    try:
        # Find the training process
        training_process = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'ULTIMATE_SYSTEM_TRAINING.py' in cmdline:
                        training_process = psutil.Process(proc.info['pid'])
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not training_process:
            print("❌ Training process not found!")
            return
        
        print(f"✅ Found training process: PID {training_process.pid}")
        
        # Monitor for 5 minutes
        start_time = time.time()
        last_memory = 0
        last_cpu_time = 0
        
        print("\n📊 MONITORING (5 minutes):")
        print("Time     | Memory(MB) | CPU(%) | Memory Δ | Status")
        print("-" * 60)
        
        while time.time() - start_time < 300:  # 5 minutes
            try:
                # Get process info
                memory_mb = training_process.memory_info().rss / (1024 * 1024)
                cpu_percent = training_process.cpu_percent(interval=1)
                
                # Calculate deltas
                memory_delta = memory_mb - last_memory if last_memory > 0 else 0
                
                # Determine status
                if memory_delta > 10:
                    status = "🔥 ACTIVE TRAINING"
                elif memory_delta > 1:
                    status = "⚡ PROCESSING"
                elif cpu_percent > 50:
                    status = "🧠 COMPUTING"
                elif cpu_percent > 10:
                    status = "📊 WORKING"
                else:
                    status = "😴 IDLE/STUCK?"
                
                # Print status
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"{current_time} | {memory_mb:8.1f} | {cpu_percent:5.1f} | {memory_delta:+7.1f} | {status}")
                
                # Update last values
                last_memory = memory_mb
                
                # Check if process is still alive
                if not training_process.is_running():
                    print("❌ Process terminated!")
                    break
                
                time.sleep(10)  # Check every 10 seconds
                
            except psutil.NoSuchProcess:
                print("❌ Process no longer exists!")
                break
            except Exception as e:
                print(f"⚠️ Error monitoring: {e}")
                time.sleep(5)
        
        print("\n📋 MONITORING COMPLETED")
        
        # Final status
        try:
            if training_process.is_running():
                final_memory = training_process.memory_info().rss / (1024 * 1024)
                final_cpu = training_process.cpu_percent()
                
                print(f"\n🔍 FINAL STATUS:")
                print(f"   • Process: Still running")
                print(f"   • Memory: {final_memory:.1f} MB")
                print(f"   • CPU: {final_cpu:.1f}%")
                
                # Analysis
                if final_memory > 900:
                    print("   • Analysis: High memory usage - likely training neural networks")
                elif final_memory > 500:
                    print("   • Analysis: Moderate memory - processing data or traditional ML")
                else:
                    print("   • Analysis: Low memory - may be stuck or finished")
                    
                if final_cpu > 50:
                    print("   • CPU Status: High usage - actively computing")
                elif final_cpu > 10:
                    print("   • CPU Status: Moderate usage - background processing")
                else:
                    print("   • CPU Status: Low usage - may be waiting or stuck")
            else:
                print("   • Process: Terminated during monitoring")
        except:
            print("   • Process: Cannot get final status")
            
    except Exception as e:
        print(f"❌ Monitor error: {e}")

def check_training_files():
    """Check for any output files from training"""
    print(f"\n📁 CHECKING OUTPUT FILES:")
    
    # Check for any new files
    current_time = time.time()
    recent_files = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                file_time = os.path.getmtime(filepath)
                if current_time - file_time < 1800:  # Files modified in last 30 minutes
                    recent_files.append((filepath, file_time))
            except:
                continue
    
    if recent_files:
        print("   📝 Recent files (last 30 min):")
        recent_files.sort(key=lambda x: x[1], reverse=True)
        for filepath, file_time in recent_files[:10]:
            mod_time = datetime.fromtimestamp(file_time).strftime("%H:%M:%S")
            print(f"   • {mod_time}: {filepath}")
    else:
        print("   ⚠️ No recent files found")

def main():
    """Main monitoring function"""
    try:
        monitor_training_process()
        check_training_files()
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print("   1. If process shows 'IDLE/STUCK' for >5 min → Consider restarting")
        print("   2. If memory keeps growing → Training is progressing") 
        print("   3. If CPU is low but memory high → May be in I/O wait")
        print("   4. Check for any error messages in console")
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

if __name__ == "__main__":
    main() 