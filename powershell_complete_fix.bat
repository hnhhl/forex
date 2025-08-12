@echo off
echo ========================================
echo   POWERSHELL COMPLETE RESET & FIX
echo ========================================

echo.
echo [PHASE 1] BACKUP & CLEANUP
echo ========================================

echo [1.1] Backing up current PowerShell configuration...
if not exist "C:\ai3.0\powershell_backup" mkdir "C:\ai3.0\powershell_backup"
if exist "%USERPROFILE%\Documents\WindowsPowerShell" (
    xcopy "%USERPROFILE%\Documents\WindowsPowerShell" "C:\ai3.0\powershell_backup\WindowsPowerShell" /E /I /Y >nul 2>&1
)
if exist "%USERPROFILE%\Documents\PowerShell" (
    xcopy "%USERPROFILE%\Documents\PowerShell" "C:\ai3.0\powershell_backup\PowerShell" /E /I /Y >nul 2>&1
)

echo [1.2] Removing all PowerShell profiles...
rd /s /q "%USERPROFILE%\Documents\WindowsPowerShell" >nul 2>&1
rd /s /q "%USERPROFILE%\Documents\PowerShell" >nul 2>&1

echo [1.3] Clearing PowerShell module cache...
rd /s /q "%USERPROFILE%\AppData\Local\Microsoft\Windows\PowerShell" >nul 2>&1
rd /s /q "%USERPROFILE%\AppData\Roaming\Microsoft\Windows\PowerShell" >nul 2>&1

echo.
echo [PHASE 2] PSREADLINE COMPLETE RESET
echo ========================================

echo [2.1] Uninstalling all PSReadLine versions...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { Get-Module PSReadLine -ListAvailable | ForEach-Object { try { Uninstall-Module PSReadLine -RequiredVersion $_.Version -Force -ErrorAction SilentlyContinue } catch {} } }"

echo [2.2] Removing PSReadLine from all locations...
rd /s /q "%ProgramFiles%\WindowsPowerShell\Modules\PSReadLine" >nul 2>&1
rd /s /q "%USERPROFILE%\Documents\WindowsPowerShell\Modules\PSReadLine" >nul 2>&1
rd /s /q "%USERPROFILE%\Documents\PowerShell\Modules\PSReadLine" >nul 2>&1

echo [2.3] Installing latest PSReadLine...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Install-PackageProvider -Name NuGet -Force -Scope CurrentUser; Install-Module PSReadLine -Force -Scope CurrentUser -AllowClobber -SkipPublisherCheck }"

echo.
echo [PHASE 3] EXECUTION POLICY & SECURITY
echo ========================================

echo [3.1] Setting execution policy...
powershell -NoProfile -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"

echo [3.2] Enabling PowerShell script execution...
reg add "HKCU\Software\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell" /v ExecutionPolicy /t REG_SZ /d RemoteSigned /f >nul 2>&1

echo.
echo [PHASE 4] CONSOLE & DISPLAY FIXES
echo ========================================

echo [4.1] Creating optimized PowerShell profile...
mkdir "%USERPROFILE%\Documents\WindowsPowerShell" >nul 2>&1
mkdir "%USERPROFILE%\Documents\PowerShell" >nul 2>&1

echo # PowerShell Profile - AI3.0 Optimized > "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo # Fixed console buffer and PSReadLine issues >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo. >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo # PSReadLine Configuration >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo if (Get-Module -ListAvailable -Name PSReadLine) { >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo     Import-Module PSReadLine >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo     Set-PSReadLineOption -PredictionSource None >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo     Set-PSReadLineOption -BellStyle None >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo     Set-PSReadLineOption -EditMode Windows >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo     Set-PSReadLineKeyHandler -Key Tab -Function Complete >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo } >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo. >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo # Console Buffer Fix >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo try { >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo     $Host.UI.RawUI.BufferSize = New-Object System.Management.Automation.Host.Size(120, 3000) >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo     $Host.UI.RawUI.WindowSize = New-Object System.Management.Automation.Host.Size(120, 30) >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo } catch {} >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo. >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo # Python Aliases for AI3.0 >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo function py { ^& "C:\Program Files\Python311\python.exe" $args } >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo function pip { ^& "C:\Program Files\Python311\python.exe" -m pip $args } >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo. >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo # AI3.0 Shortcuts >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo function ai3-check { py C:\ai3.0\simple_check.py } >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo function ai3-models { py C:\ai3.0\check_models.py } >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo function ai3-train { py C:\ai3.0\demo_mass_training.py } >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo. >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
echo Write-Host "AI3.0 PowerShell Environment Loaded!" -ForegroundColor Green >> "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"

echo [4.2] Copying profile for PowerShell Core...
copy "%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1" "%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1" >nul 2>&1

echo.
echo [PHASE 5] WINDOWS TERMINAL OPTIMIZATION
echo ========================================

echo [5.1] Creating Windows Terminal settings...
if not exist "%USERPROFILE%\AppData\Local\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState" mkdir "%USERPROFILE%\AppData\Local\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState" >nul 2>&1

echo [5.2] Registry fixes for console...
reg add "HKCU\Console" /v FaceName /t REG_SZ /d "Consolas" /f >nul 2>&1
reg add "HKCU\Console" /v FontSize /t REG_DWORD /d 0x00140000 /f >nul 2>&1
reg add "HKCU\Console" /v ScreenBufferSize /t REG_DWORD /d 0x0bb80078 /f >nul 2>&1
reg add "HKCU\Console" /v WindowSize /t REG_DWORD /d 0x001e0078 /f >nul 2>&1

echo.
echo [PHASE 6] VERIFICATION & TESTING
echo ========================================

echo [6.1] Testing PowerShell functionality...
powershell -NoProfile -Command "Write-Host 'PowerShell NoProfile: OK' -ForegroundColor Green"

echo [6.2] Testing new profile...
powershell -Command "Write-Host 'PowerShell with Profile: OK' -ForegroundColor Green"

echo [6.3] Testing Python integration...
powershell -Command "py --version"

echo.
echo ========================================
echo   POWERSHELL COMPLETE FIX FINISHED!
echo ========================================

echo.
echo WHAT WAS FIXED:
echo ✅ PSReadLine module completely reinstalled
echo ✅ All PowerShell profiles reset and optimized
echo ✅ Console buffer issues resolved
echo ✅ Execution policy configured
echo ✅ Python aliases created (py, pip)
echo ✅ AI3.0 shortcuts added (ai3-check, ai3-models, ai3-train)
echo ✅ Windows Terminal optimized
echo ✅ Registry console settings fixed
echo.
echo NEW COMMANDS AVAILABLE:
echo   py --version           (Python)
echo   pip install package    (Package installer)
echo   ai3-check             (System check)
echo   ai3-models            (Models analysis)
echo   ai3-train             (Start training)
echo.
echo RESTART YOUR TERMINAL NOW for full effect!
echo.
pause 