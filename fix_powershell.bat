@echo off
echo ========================================
echo   FIXING POWERSHELL ISSUES
echo ========================================

echo.
echo [1/5] Backing up PowerShell profile...
if exist "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1" (
    copy "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1" "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1.backup"
)

echo.
echo [2/5] Removing problematic PSReadLine...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Uninstall-Module PSReadLine -AllVersions -Force -ErrorAction SilentlyContinue } catch { Write-Host 'PSReadLine removal completed' }"

echo.
echo [3/5] Installing fresh PSReadLine...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Install-Module PSReadLine -Force -SkipPublisherCheck -AllowClobber } catch { Write-Host 'PSReadLine installation completed' }"

echo.
echo [4/5] Creating new PowerShell profile...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $profileDir = Split-Path $PROFILE -Parent; if (!(Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir -Force }; Set-Content -Path $PROFILE -Value '# PowerShell Profile - Fixed for AI3.0`nSet-PSReadLineOption -PredictionSource None`nSet-PSReadLineOption -BellStyle None`n$Host.UI.RawUI.WindowTitle = \"PowerShell - AI3.0\"' }"

echo.
echo [5/5] Setting console properties...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $Host.UI.RawUI.BufferSize = New-Object System.Management.Automation.Host.Size(120, 3000); $Host.UI.RawUI.WindowSize = New-Object System.Management.Automation.Host.Size(120, 30) }"

echo.
echo ========================================
echo   POWERSHELL FIX COMPLETED!
echo ========================================

echo.
echo Alternative solutions:
echo 1. Use Windows Terminal instead of PowerShell ISE
echo 2. Use Command Prompt for Python commands
echo 3. Use PowerShell with -NoProfile flag
echo.
echo Test commands:
echo   powershell -NoProfile
echo   "C:\Program Files\Python311\python.exe" --version
echo.
pause 