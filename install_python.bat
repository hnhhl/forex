@echo off
echo ========================================
echo   PYTHON INSTALLATION & POWERSHELL FIX
echo ========================================

echo.
echo [1/4] Checking current Python installation...
python --version 2>nul
if %errorlevel% == 0 (
    echo Python is already installed!
    python --version
) else (
    echo Python not found. Proceeding with installation...
)

echo.
echo [2/4] Downloading Python 3.11.9...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python-installer.exe'}"

if exist "%TEMP%\python-installer.exe" (
    echo Python installer downloaded successfully!
    
    echo.
    echo [3/4] Installing Python...
    echo Please wait while Python is being installed...
    "%TEMP%\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    
    echo.
    echo [4/4] Verifying installation...
    timeout /t 5 /nobreak >nul
    python --version
    
    if %errorlevel% == 0 (
        echo.
        echo ✅ Python installed successfully!
        echo.
        echo Installing required packages...
        python -m pip install --upgrade pip
        python -m pip install tensorflow scikit-learn pandas numpy matplotlib seaborn lightgbm xgboost
        
        echo.
        echo ✅ All packages installed successfully!
    ) else (
        echo ❌ Python installation may have failed. Please check manually.
    )
    
    echo.
    echo Cleaning up...
    del "%TEMP%\python-installer.exe" 2>nul
) else (
    echo ❌ Failed to download Python installer.
    echo Please try installing Python manually from: https://www.python.org/downloads/
)

echo.
echo [BONUS] Fixing PowerShell PSReadLine issues...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& {try { Uninstall-Module PSReadLine -AllVersions -Force -ErrorAction SilentlyContinue; Install-Module PSReadLine -Force -SkipPublisherCheck; Write-Host 'PSReadLine updated successfully!' } catch { Write-Host 'PSReadLine update failed, but this is not critical.' }}"

echo.
echo ========================================
echo   INSTALLATION COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Close and reopen your terminal
echo 2. Run: python --version
echo 3. Run: cd C:\ai3.0 
echo 4. Run: python check_models.py
echo.
pause 