#!/usr/bin/env python3
# Simple system check for AI3.0

import sys
import os

def main():
    print("=" * 50)
    print("🔍 AI3.0 SYSTEM CHECK")
    print("=" * 50)
    
    # Python info
    print(f"🐍 Python Version: {sys.version}")
    print(f"📍 Python Path: {sys.executable}")
    
    # Current directory
    print(f"📁 Current Directory: {os.getcwd()}")
    
    # Check trained_models
    if os.path.exists("trained_models"):
        files = os.listdir("trained_models")
        pkl_count = len([f for f in files if f.endswith('.pkl')])
        keras_count = len([f for f in files if f.endswith('.keras')])
        h5_count = len([f for f in files if f.endswith('.h5')])
        
        print(f"🤖 Models Found:")
        print(f"   • PKL Models: {pkl_count}")
        print(f"   • Keras Models: {keras_count}")
        print(f"   • H5 Models: {h5_count}")
        print(f"   • Total Models: {pkl_count + keras_count + h5_count}")
    else:
        print("❌ trained_models directory not found!")
    
    # Check training systems
    training_files = [
        "MASS_TRAINING_SYSTEM_AI30.py",
        "comprehensive_training_fixed.py"
    ]
    
    print(f"🚀 Training Systems:")
    for file in training_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            print(f"   ✅ {file} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file} (not found)")
    
    print("=" * 50)
    print("✅ System check completed!")
    
    # Try importing key libraries
    print("\n🔧 Testing Python Libraries:")
    libs = ['numpy', 'pandas', 'tensorflow', 'sklearn']
    for lib in libs:
        try:
            __import__(lib)
            print(f"   ✅ {lib}")
        except ImportError:
            print(f"   ❌ {lib} (not installed)")
    
    print("\n🎯 Ready for AI3.0 training!")

if __name__ == "__main__":
    main()