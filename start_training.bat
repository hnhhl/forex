@echo off
echo ==========================================
echo    ULTIMATE XAU TRAINING LAUNCHER
echo ==========================================
echo [INFO] Switching to AI3.0 directory...
cd /d C:\ai3.0

echo [INFO] Current directory: %CD%
echo [INFO] Checking training file...
if exist "ULTIMATE_REAL_DATA_TRAINING_171_MODELS.py" (
    echo [SUCCESS] Training file found!
    echo [INFO] Starting training with ALL 1.1M+ records...
    echo ==========================================
    python ULTIMATE_REAL_DATA_TRAINING_171_MODELS.py
) else (
    echo [ERROR] Training file not found!
    echo [ERROR] Please ensure you are in the correct directory
    pause
)

echo ==========================================
echo [COMPLETE] Training finished or stopped
pause 