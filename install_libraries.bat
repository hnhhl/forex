@echo off
echo ========================================
echo   INSTALLING AI3.0 PYTHON LIBRARIES
echo ========================================

echo.
echo [1/8] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/8] Installing numpy...
python -m pip install numpy

echo.
echo [3/8] Installing pandas...
python -m pip install pandas

echo.
echo [4/8] Installing scikit-learn...
python -m pip install scikit-learn

echo.
echo [5/8] Installing tensorflow...
python -m pip install tensorflow

echo.
echo [6/8] Installing matplotlib...
python -m pip install matplotlib seaborn

echo.
echo [7/8] Installing ML libraries...
python -m pip install lightgbm xgboost

echo.
echo [8/8] Installing additional libraries...
python -m pip install joblib pickle-mixin

echo.
echo ========================================
echo   INSTALLATION COMPLETED!
echo ========================================

echo.
echo Verifying installation...
python simple_check.py

echo.
echo Next steps:
echo 1. Run: python check_models.py
echo 2. Run: python MASS_TRAINING_SYSTEM_AI30.py
echo.
pause 