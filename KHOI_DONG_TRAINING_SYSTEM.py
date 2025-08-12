#!/usr/bin/env python3
"""
🚀 KHỞI ĐỘNG TRAINING SYSTEM
Ultimate XAU Super System V4.0

Launcher tổng hợp cho tất cả training options
"""

import os
import sys
from datetime import datetime
import subprocess

def print_banner():
    """In banner hệ thống"""
    print("🚀" + "=" * 68 + "🚀")
    print("🏆" + " " * 20 + "ULTIMATE XAU SUPER SYSTEM V4.0" + " " * 17 + "🏆")
    print("🤖" + " " * 25 + "TRAINING SYSTEM LAUNCHER" + " " * 20 + "🤖")
    print("🚀" + "=" * 68 + "🚀")
    print()

def show_current_status():
    """Hiển thị trạng thái hiện tại"""
    print("📊 TRẠNG THÁI HỆ THỐNG HIỆN TẠI:")
    print("-" * 50)
    
    # Kiểm tra models đã trained
    models_dir = "training/xauusdc/models"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        print(f"✅ Models đã trained: {len(models)}")
        
        best_models = [
            ("M15_dir_2.h5", "84.0%"),
            ("M30_dir_2.h5", "77.6%"),
            ("M15_dir_4.h5", "72.2%"),
            ("H1_dir_2.h5", "67.1%")
        ]
        
        print("🏆 Top Models:")
        for model, acc in best_models:
            status = "✅" if model in models else "❌"
            print(f"  {status} {model}: {acc} accuracy")
    else:
        print("❌ Chưa có models nào được trained")
    
    # Kiểm tra data
    data_dir = "training/xauusdc/data"
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        print(f"✅ Training data: {len(data_files)} timeframes")
    else:
        print("❌ Chưa có training data")
        
    print()

def show_options():
    """Hiển thị các options"""
    print("🎯 CÁC LỰA CHỌN TRAINING:")
    print("-" * 50)
    print("1. 🔄 RE-TRAINING XAU/USDc (Cập nhật models hiện tại)")
    print("2. 🚀 TRAINING SYMBOLS KHÁC (EURUSD, GBPUSD, etc.)")
    print("3. 📊 DEMO LIVE PREDICTIONS (Test models hiện tại)")
    print("4. 📈 PHÂN TÍCH DỮ LIỆU TRAINING")
    print("5. 🎪 TRAINING NÂNG CAO (LSTM, Multi-timeframe)")
    print("6. 🔍 KIỂM TRA TRẠNG THÁI HỆ THỐNG")
    print("7. ❌ THOÁT")
    print()

def run_retraining():
    """Chạy re-training cho XAU/USDc"""
    print("🔄 KHỞI ĐỘNG RE-TRAINING XAU/USDc...")
    print("=" * 50)
    
    try:
        # Chạy training system
        result = subprocess.run([sys.executable, "XAUUSDC_TRAINING_SYSTEM_OPTIMIZED.py"], 
                              capture_output=False, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n✅ RE-TRAINING THÀNH CÔNG!")
        else:
            print("\n❌ RE-TRAINING THẤT BẠI!")
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy re-training: {e}")

def run_multi_symbol_training():
    """Chạy training cho nhiều symbols"""
    print("🚀 KHỞI ĐỘNG MULTI-SYMBOL TRAINING...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "TRAINING_MULTI_SYMBOLS.py"], 
                              capture_output=False, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n✅ MULTI-SYMBOL TRAINING THÀNH CÔNG!")
        else:
            print("\n❌ MULTI-SYMBOL TRAINING THẤT BẠI!")
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy multi-symbol training: {e}")

def run_live_demo():
    """Chạy demo live predictions"""
    print("📊 KHỞI ĐỘNG LIVE PREDICTIONS DEMO...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "DEMO_XAUUSDC_PREDICTION_SYSTEM.py"], 
                              capture_output=False, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n✅ LIVE DEMO HOÀN THÀNH!")
        else:
            print("\n❌ LIVE DEMO THẤT BẠI!")
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy live demo: {e}")

def run_data_analysis():
    """Chạy phân tích dữ liệu"""
    print("📈 KHỞI ĐỘNG PHÂN TÍCH DỮ LIỆU...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "PHAN_TICH_DU_LIEU_TRAINING.py"], 
                              capture_output=False, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n✅ PHÂN TÍCH DỮ LIỆU HOÀN THÀNH!")
        else:
            print("\n❌ PHÂN TÍCH DỮ LIỆU THẤT BẠI!")
            
    except Exception as e:
        print(f"❌ Lỗi khi phân tích dữ liệu: {e}")

def advanced_training_options():
    """Hiển thị options training nâng cao"""
    print("🎪 TRAINING NÂNG CAO OPTIONS:")
    print("-" * 40)
    print("1. LSTM/GRU Models Training")
    print("2. Multi-timeframe Feature Integration")
    print("3. Attention Mechanism Models")
    print("4. Ensemble Learning Optimization")
    print("5. Reinforcement Learning Integration")
    print("6. Quay lại menu chính")
    
    choice = input("\nChọn option (1-6): ").strip()
    
    if choice == "1":
        print("🔮 LSTM/GRU Training đang được phát triển...")
        print("💡 Sẽ có trong phiên bản tiếp theo!")
    elif choice == "2":
        print("🔗 Multi-timeframe Integration đang được phát triển...")
        print("💡 Sẽ có trong phiên bản tiếp theo!")
    elif choice == "3":
        print("🎯 Attention Mechanism đang được phát triển...")
        print("💡 Sẽ có trong phiên bản tiếp theo!")
    elif choice == "4":
        print("🎪 Ensemble Optimization đang được phát triển...")
        print("💡 Sẽ có trong phiên bản tiếp theo!")
    elif choice == "5":
        print("🤖 Reinforcement Learning đang được phát triển...")
        print("💡 Sẽ có trong phiên bản tiếp theo!")
    elif choice == "6":
        return
    else:
        print("❌ Lựa chọn không hợp lệ!")

def main():
    """Main launcher function"""
    
    while True:
        print_banner()
        show_current_status()
        show_options()
        
        try:
            choice = input("🎯 Nhập lựa chọn (1-7): ").strip()
            
            if choice == "1":
                run_retraining()
                
            elif choice == "2":
                run_multi_symbol_training()
                
            elif choice == "3":
                run_live_demo()
                
            elif choice == "4":
                run_data_analysis()
                
            elif choice == "5":
                advanced_training_options()
                
            elif choice == "6":
                show_current_status()
                
            elif choice == "7":
                print("👋 Cảm ơn bạn đã sử dụng Ultimate XAU Super System V4.0!")
                print("🚀 Hẹn gặp lại!")
                break
                
            else:
                print("❌ Lựa chọn không hợp lệ! Vui lòng chọn từ 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng chương trình!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        print("\n" * 2)

if __name__ == "__main__":
    main() 