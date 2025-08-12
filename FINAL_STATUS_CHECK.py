import os
import psutil
from datetime import datetime, timedelta

print("🔍 KIỂM TRA TRẠNG THÁI CUỐI CÙNG - ULTIMATE SYSTEM V5.0")
print("="*60)

# Thời gian hiện tại
current_time = datetime.now()
start_time = datetime(2024, 6, 18, 8, 18, 41)
total_elapsed = current_time - start_time

print(f"⏰ THỜI GIAN:")
print(f"  🚀 Bắt đầu: 08:18:41")
print(f"  🕐 Hiện tại: {current_time.strftime('%H:%M:%S')}")
print(f"  ⏱️ Tổng thời gian: {str(total_elapsed).split('.')[0]}")

# Kiểm tra process PID 19200
print(f"\n🔄 KIỂM TRA PROCESS PID 19200:")
try:
    proc = psutil.Process(19200)
    if proc.is_running():
        memory_mb = proc.memory_info().rss / (1024*1024)
        cpu_percent = proc.cpu_percent()
        create_time = datetime.fromtimestamp(proc.create_time())
        process_elapsed = current_time - create_time
        
        print(f"  ✅ Process vẫn đang chạy")
        print(f"  💾 Memory: {memory_mb:.1f}MB")
        print(f"  🖥️ CPU: {cpu_percent:.1f}%")
        print(f"  ⏱️ Đã chạy: {str(process_elapsed).split('.')[0]}")
        
        # Phân tích trạng thái
        elapsed_minutes = total_elapsed.total_seconds() / 60
        print(f"\n📊 PHÂN TÍCH TRẠNG THÁI:")
        print(f"  ⏱️ Đã chạy: {elapsed_minutes:.1f} phút")
        
        if elapsed_minutes > 35:
            print(f"  ⚠️ CHẠY QUÁ LÂU - có thể gặp vấn đề:")
            print(f"     - Process bị treo (hang)")
            print(f"     - Lỗi trong quá trình training")
            print(f"     - Dataset quá lớn")
            print(f"     - Memory leak")
            
            if cpu_percent < 1.0:
                print(f"  🚨 CPU = {cpu_percent:.1f}% - Process có thể đã treo!")
                print(f"  💡 Khuyến nghị: Cần kiểm tra hoặc restart")
            
        elif elapsed_minutes > 30:
            print(f"  ⏰ Gần hoàn thành - còn vài phút")
        else:
            print(f"  ✅ Đang chạy bình thường")
            
    else:
        print(f"  ❌ Process đã dừng")
        
except psutil.NoSuchProcess:
    print(f"  ❌ Process PID 19200 không tồn tại")
except Exception as e:
    print(f"  ❌ Lỗi kiểm tra process: {e}")

# Kiểm tra file kết quả
print(f"\n📁 KIỂM TRA FILE KẾT QUẢ:")
result_files = [f for f in os.listdir('.') if f.startswith('ULTIMATE_SYSTEM_V5_TRAINING_RESULTS_')]
if result_files:
    print(f"  ✅ Tìm thấy {len(result_files)} file kết quả:")
    for f in result_files:
        size_kb = os.path.getsize(f) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        print(f"    📊 {f} ({size_kb:.1f}KB, {mtime.strftime('%H:%M:%S')})")
    print(f"  🎉 QUÁ TRÌNH ĐÃ HOÀN THÀNH!")
else:
    print(f"  ⏳ Chưa có file kết quả")

# Kiểm tra log files hoặc error files
print(f"\n📋 KIỂM TRA LOG/ERROR:")
log_files = []
for f in os.listdir('.'):
    if any(keyword in f.lower() for keyword in ['log', 'error', 'debug', 'traceback']):
        log_files.append(f)

if log_files:
    print(f"  📄 Tìm thấy {len(log_files)} log files:")
    for f in log_files[-3:]:  # Show last 3
        print(f"    📄 {f}")
else:
    print(f"  ✅ Không có log/error files")

# Tài nguyên hệ thống
print(f"\n💻 TÀI NGUYÊN HỆ THỐNG:")
cpu_percent = psutil.cpu_percent(interval=1)
memory = psutil.virtual_memory()
print(f"  🖥️ CPU: {cpu_percent:.1f}%")
print(f"  💾 RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

# Kết luận
print(f"\n{'='*60}")
elapsed_minutes = total_elapsed.total_seconds() / 60

if result_files:
    print(f"🎉 HOÀN THÀNH! Quá trình cập nhật đã xong!")
    print(f"⏱️ Tổng thời gian: {elapsed_minutes:.1f} phút")
elif elapsed_minutes > 40:
    print(f"⚠️ QUÁ TRÌNH CÓ THỂ GẶP VẤN ĐỀ!")
    print(f"💡 Khuyến nghị: Kiểm tra hoặc restart process")
else:
    print(f"🔄 VẪN ĐANG CHẠY... Chờ thêm vài phút")

print(f"⏰ Kiểm tra lúc: {current_time.strftime('%H:%M:%S')}")
print(f"{'='*60}") 