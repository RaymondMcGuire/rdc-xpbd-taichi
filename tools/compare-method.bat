@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not exist "..\.venv\Scripts\python.exe" (
  echo ERROR: ..\.venv\Scripts\python.exe not found. Run scripts\init.bat first.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo Usage:
  echo   %~nx0 data1.json data2.json [data3.json ...] [compare_convergence.py args]
  echo   %~nx0 --pattern "..\output\cow\*\convergence_plots\*.json" --log --output convergence.png
  echo.
  echo Notes:
  echo   - Use JSON files exported by run_neohookean_usd.py with --plot-frames.
  echo   - This plots hydrostatic and deviatoric error convergence curves.
  pause
  exit /b 1
)

"..\.venv\Scripts\python.exe" compare_convergence.py %*

pause
