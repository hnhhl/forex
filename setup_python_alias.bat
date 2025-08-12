@echo off
echo ========================================
echo   SETTING UP PYTHON ALIAS & PATH
echo ========================================

echo.
echo [1/3] Adding Python to System PATH...
setx PATH "%PATH%;C:\Program Files\Python311;C:\Program Files\Python311\Scripts" /M

echo.
echo [2/3] Creating Python aliases...
doskey python="C:\Program Files\Python311\python.exe" $*
doskey pip="C:\Program Files\Python311\python.exe" -m pip $*

echo.
echo [3/3] Creating batch files for easy access...
echo @echo off > C:\ai3.0\py.bat
echo "C:\Program Files\Python311\python.exe" %%* >> C:\ai3.0\py.bat

echo @echo off > C:\ai3.0\pip.bat
echo "C:\Program Files\Python311\python.exe" -m pip %%* >> C:\ai3.0\pip.bat

echo.
echo ========================================
echo   PYTHON SETUP COMPLETED!
echo ========================================

echo.
echo Now you can use:
echo   py --version
echo   py simple_check.py
echo   pip install package_name
echo.
echo Or with full path:
echo   "C:\Program Files\Python311\python.exe" script.py
echo.
pause 