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
  echo   %~nx0 path\to\json_or_directory --baseline-file baseline.json --format markdown
  echo   %~nx0 "..\output\cow" --baseline-method "block_neohookean" --plot convergence_table.png
  echo.
  echo Notes:
  echo   - Recursively scans convergence JSON files.
  echo   - Reports mean/final hydrostatic, deviatoric, and aggregate errors.
  pause
  exit /b 1
)

"..\.venv\Scripts\python.exe" summarize_errors.py %*

pause
