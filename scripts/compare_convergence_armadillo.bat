@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\.."

if not exist .venv\Scripts\python.exe (
  echo ERROR: .venv not found. Run scripts\init.bat first.
  pause
  exit /b 1
)

set "MODEL=assets/tetmesh/armadillo.node"
set "SCALE=0.01"
set "YOUNG=500000000"
set "FRAME_ID=300"
set "COMMON_ARGS=--model %MODEL% --scale %SCALE% --young %YOUNG% --arch gpu --dt 0.01 --fps 60 --frames 301 --substeps 1 --iterations 200 --no-usd --monitor-convergence --plot-frames --plot-frame-ids %FRAME_ID% --print-convergence"

echo.
echo Running Armadillo convergence comparison...
echo.

echo [1/4] NH
uv run python run_neohookean_usd.py %COMMON_ARGS% --split-neohookean --no-cheb
if errorlevel 1 goto :fail

echo [2/4] NH-DC
uv run python run_neohookean_usd.py %COMMON_ARGS% --split-neohookean --cheb
if errorlevel 1 goto :fail

echo [3/4] BNH
uv run python run_neohookean_usd.py %COMMON_ARGS% --block-neohookean --no-cheb
if errorlevel 1 goto :fail

echo [4/4] BNH-DC
uv run python run_neohookean_usd.py %COMMON_ARGS% --block-neohookean --cheb
if errorlevel 1 goto :fail

set "YOUNG_TAG=5p000e08"
set "CONV_DIR=output\armadillo"
set "NH_JSON=%CONV_DIR%\xpbd_split_neohookean_no_cheby_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "NHDC_JSON=%CONV_DIR%\xpbd_split_neohookean_dynamic_cheby_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "BNH_JSON=%CONV_DIR%\xpbd_block_neohookean_no_cheby_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "BNHDC_JSON=%CONV_DIR%\xpbd_block_neohookean_dynamic_cheby_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "PLOT_DIR=output\compare_convergence_armadillo"

if not exist "%PLOT_DIR%" mkdir "%PLOT_DIR%"

echo.
echo Plotting convergence comparison...
.venv\Scripts\python.exe tools\compare_convergence.py "%NH_JSON%" "%NHDC_JSON%" "%BNH_JSON%" "%BNHDC_JSON%" --labels "NH" "NH-DC" "BNH" "BNH-DC" --log --output "%PLOT_DIR%\armadillo_convergence.png" --title "Armadillo frame %FRAME_ID%"
if errorlevel 1 goto :fail

echo.
echo Convergence data:
echo   %CONV_DIR%
echo Comparison plot:
echo   %PLOT_DIR%\armadillo_convergence.png
echo.
pause
exit /b 0

:fail
echo.
echo Armadillo convergence comparison failed.
pause
exit /b 1
