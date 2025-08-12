@echo off
title AI3.0 Ultimate Training
echo ========================================
echo     AI3.0 ULTIMATE TRAINING LAUNCHER
echo ========================================
cd /d "C:\ai3.0"
echo [INFO] Directory: %CD%
echo [INFO] Starting training...
echo ========================================
python.exe ULTIMATE_REAL_DATA_TRAINING_171_MODELS.py
echo ========================================
echo [COMPLETE] Training finished
pause 