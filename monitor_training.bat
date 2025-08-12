@echo off
title AI3.0 Training Monitor
:loop
cls
echo ========================================
echo    AI3.0 TRAINING MONITOR
echo ========================================
echo Time: %date% %time%
echo.

echo [PYTHON PROCESS]
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,@{Name='Memory(MB)';Expression={[math]::Round($_.WorkingSet/1MB,2)}} | Format-Table -AutoSize"

echo.
echo [MODELS COUNT]
powershell -Command "Get-ChildItem trained_models/ -ErrorAction SilentlyContinue | Measure-Object | Select-Object Count"

echo.
echo [LATEST RESULTS]
powershell -Command "Get-ChildItem training_results/ | Sort-Object LastWriteTime -Descending | Select-Object -First 2 Name,LastWriteTime | Format-Table -AutoSize"

echo.
echo [GPU USAGE]
powershell -Command "Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Select-Object -First 1 CookedValue"

echo ========================================
echo Press Ctrl+C to stop monitoring
timeout /t 10 /nobreak >nul
goto loop 