import psutil
import os
import time
from datetime import datetime, timedelta

print("🔍 SYSTEM HEALTH MONITOR - ULTIMATE SYSTEM V5.0")
print("="*55)

# Thông tin thời gian
current_time = datetime.now()
restart_time = datetime(2024, 6, 18, 8, 56, 0)  # Thời gian restart
elapsed_since_restart = current_time - restart_time

print(f"⏰ THỜI GIAN HOẠT ĐỘNG:")
print(f"  🔄 Restart lúc: 08:56:00")
print(f"  🕐 Hiện tại: {current_time.strftime('%H:%M:%S')}")
print(f"  ⏱️ Đã chạy: {str(elapsed_since_restart).split('.')[0]}")

# Kiểm tra process PID 3312
print(f"\n🔄 PROCESS MONITORING:")
try:
    proc = psutil.Process(3312)
    if proc.is_running():
        memory_info = proc.memory_info()
        memory_mb = memory_info.rss / (1024*1024)
        cpu_percent = proc.cpu_percent(interval=1)
        
        # Thông tin chi tiết process
        print(f"  ✅ PID 3312: ĐANG HOẠT ĐỘNG")
        print(f"     💾 Memory: {memory_mb:.1f}MB")
        print(f"     🖥️ CPU: {cpu_percent:.1f}%")
        print(f"     📊 Status: {proc.status()}")
        
        # Phân tích hoạt động
        if cpu_percent > 50:
            print(f"     🚀 HIGH CPU - đang training intensively")
        elif cpu_percent > 10:
            print(f"     🔄 MODERATE CPU - đang xử lý data")
        elif cpu_percent > 1:
            print(f"     💭 LOW CPU - đang tính toán background")
        else:
            print(f"     😴 IDLE - có thể đang chờ I/O")
        
        # Memory analysis
        if memory_mb > 500:
            print(f"     📈 HIGH MEMORY - neural network training")
        elif memory_mb > 400:
            print(f"     🧠 NORMAL MEMORY - data processing")
        else:
            print(f"     💾 LOW MEMORY - initialization phase")
            
    else:
        print(f"  ❌ Process đã dừng")
        
except psutil.NoSuchProcess:
    print(f"  ❌ Process PID 3312 không tồn tại")
except Exception as e:
    print(f"  ❌ Lỗi: {e}")

# Kiểm tra tất cả Python processes
print(f"\n🐍 TẤT CẢ PYTHON PROCESSES:")
python_procs = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
    try:
        if proc.info['name'] == 'python.exe':
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            memory_mb = proc.info['memory_info'].rss / (1024*1024)
            cpu_pct = proc.info['cpu_percent']
            
            python_procs.append({
                'pid': proc.info['pid'],
                'memory': memory_mb,
                'cpu': cpu_pct,
                'cmd': cmdline.split('\\')[-1] if '\\' in cmdline else cmdline[:50]
            })
    except:
        continue

for proc in python_procs:
    if 'ULTIMATE_SYSTEM_V5' in proc['cmd']:
        print(f"  🎯 PID {proc['pid']}: {proc['memory']:.1f}MB, {proc['cpu']:.1f}% CPU")
        print(f"     📋 {proc['cmd']}")
    else:
        print(f"  📄 PID {proc['pid']}: {proc['memory']:.1f}MB, {proc['cpu']:.1f}% CPU")

# Kiểm tra file system
print(f"\n📁 FILE SYSTEM STATUS:")
result_files = [f for f in os.listdir('.') if f.startswith('ULTIMATE_SYSTEM_V5_TRAINING_RESULTS_')]
print(f"  📊 Result files: {len(result_files)}")

# Kiểm tra các file log/temp
temp_files = []
for f in os.listdir('.'):
    if any(ext in f.lower() for ext in ['.tmp', '.log', '.cache', '.temp']):
        try:
            size_kb = os.path.getsize(f) / 1024
            temp_files.append((f, size_kb))
        except:
            continue

if temp_files:
    print(f"  📄 Temp/Log files: {len(temp_files)}")
    for f, size in temp_files[-3:]:  # Show last 3
        print(f"     📄 {f} ({size:.1f}KB)")

# Tài nguyên hệ thống
print(f"\n💻 SYSTEM RESOURCES:")
cpu_percent = psutil.cpu_percent(interval=1)
memory = psutil.virtual_memory()
disk = psutil.disk_usage('C:')

print(f"  🖥️ CPU Usage: {cpu_percent:.1f}%")
print(f"  💾 RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
print(f"  💽 Disk C: {disk.percent:.1f}% ({disk.free/1024**3:.1f}GB free)")

# Health status
print(f"\n🏥 HEALTH STATUS:")
elapsed_minutes = elapsed_since_restart.total_seconds() / 60

if elapsed_minutes > 35:
    print(f"  ⚠️ RUNNING TOO LONG ({elapsed_minutes:.1f} min)")
    print(f"     Có thể cần kiểm tra lại")
elif elapsed_minutes > 25:
    print(f"  ⏰ NEAR COMPLETION ({elapsed_minutes:.1f} min)")
    print(f"     Sắp hoàn thành")
elif elapsed_minutes > 15:
    print(f"  🔄 PROGRESSING WELL ({elapsed_minutes:.1f} min)")
    print(f"     Đang tiến triển tốt")
else:
    print(f"  🚀 EARLY STAGE ({elapsed_minutes:.1f} min)")
    print(f"     Vẫn trong giai đoạn đầu")

# Dự đoán
remaining_time = max(0, 25 - elapsed_minutes)
estimated_completion = current_time + timedelta(minutes=remaining_time)

print(f"\n🎯 PREDICTION:")
print(f"  ⏰ Estimated remaining: {remaining_time:.1f} minutes")
print(f"  🏁 Expected completion: {estimated_completion.strftime('%H:%M:%S')}")

# Network/IO activity (if available)
try:
    io_counters = psutil.disk_io_counters()
    if io_counters:
        print(f"\n💿 DISK I/O ACTIVITY:")
        print(f"  📖 Read: {io_counters.read_bytes/1024**2:.1f}MB")
        print(f"  📝 Write: {io_counters.write_bytes/1024**2:.1f}MB")
except:
    pass

print(f"\n{'='*55}")

# Overall assessment
if result_files:
    print(f"🎉 STATUS: COMPLETED!")
elif elapsed_minutes > 40:
    print(f"⚠️ STATUS: POSSIBLY STUCK - NEEDS ATTENTION")
elif cpu_percent < 1 and elapsed_minutes > 30:
    print(f"😴 STATUS: IDLE TOO LONG - MAY BE STUCK")
else:
    print(f"✅ STATUS: HEALTHY & RUNNING")

print(f"⏰ Checked at: {current_time.strftime('%H:%M:%S')}")
print(f"{'='*55}") 