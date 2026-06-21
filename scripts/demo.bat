@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\.."

if not exist .venv\Scripts\python.exe (
  echo ERROR: .venv not found. Run scripts\init.bat first.
  pause
  exit /b 1
)

uv run python run_neohookean_usd.py --model assets/tetmesh/cow.node --scale 1.5 --young 500000000 --arch gpu --dt 0.01 --fps 60 --frames 301 --substeps 1 --iterations 200 --block-neohookean --cheb --optimize-usd --streaming --use-binary
if errorlevel 1 goto :fail

echo.
echo Demo finished successfully.
pause
exit /b 0

:fail
echo.
echo Demo failed.
pause
exit /b 1
