@echo off
echo ========================================
echo   INSTALLING PYTHON FROM PYTHON.ORG
echo ========================================

echo.
echo Downloading Python 3.11.9...
curl -o "%TEMP%\python-3.11.9-amd64.exe" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

if exist "%TEMP%\python-3.11.9-amd64.exe" (
    echo.
    echo Download successful! Installing Python...
    echo.
    echo IMPORTANT: When the installer opens:
    echo 1. Check "Add Python to PATH"
    echo 2. Click "Install Now"
    echo.
    pause
    
    "%TEMP%\python-3.11.9-amd64.exe"
    
    echo.
    echo Installation completed. Please:
    echo 1. Close this window
    echo 2. Open a new Command Prompt
    echo 3. Run: python --version
    echo 4. Run: cd C:\ai3.0
    echo 5. Run: python check_models.py
    
) else (
    echo.
    echo Download failed. Please manually download Python from:
    echo https://www.python.org/downloads/
)

echo.
pause 