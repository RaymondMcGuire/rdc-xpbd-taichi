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
echo Running Figure 7 Armadillo convergence cases...
echo.

echo [1/4] Split Neo-Hookean (NH)
uv run python run_neohookean_usd.py %COMMON_ARGS%
if errorlevel 1 goto :fail

echo [2/4] Split Neo-Hookean + Chebyshev (NHC)
uv run python run_neohookean_usd.py %COMMON_ARGS% --cheb
if errorlevel 1 goto :fail

echo [3/4] Block Neo-Hookean (BNH)
uv run python run_neohookean_usd.py %COMMON_ARGS% --neohookean-block
if errorlevel 1 goto :fail

echo [4/4] Block Neo-Hookean + Chebyshev (BNHC)
uv run python run_neohookean_usd.py %COMMON_ARGS% --neohookean-block --cheb
if errorlevel 1 goto :fail

set "YOUNG_TAG=5p000e08"
set "CONV_DIR=output\armadillo"
set "NH_JSON=%CONV_DIR%\xpbd_neohookean_no_cheb_rho_dyn_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "NHC_JSON=%CONV_DIR%\xpbd_neohookean_cheb_rho_dyn_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "BNH_JSON=%CONV_DIR%\xpbd_block_neohookean_no_cheb_rho_dyn_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "BNHC_JSON=%CONV_DIR%\xpbd_block_neohookean_cheb_rho_dyn_young_%YOUNG_TAG%\convergence_plots\f0300_s00.json"
set "FIGURE_DIR=output\figure7_armadillo"

if not exist "%FIGURE_DIR%" mkdir "%FIGURE_DIR%"

echo.
echo Plotting Figure 7 convergence comparison...
.venv\Scripts\python.exe tools\compare_convergence.py "%NH_JSON%" "%NHC_JSON%" "%BNH_JSON%" "%BNHC_JSON%" --labels "NH" "NHC" "BNH" "BNHC" --log --output "%FIGURE_DIR%\figure7_armadillo_convergence.png" --title "Armadillo frame %FRAME_ID%"
if errorlevel 1 goto :fail

echo.
echo Figure 7 convergence data:
echo   %CONV_DIR%
echo Figure 7 comparison plot:
echo   %FIGURE_DIR%\figure7_armadillo_convergence.png
echo.
pause
exit /b 0

:fail
echo.
echo Figure 7 Armadillo convergence run failed.
pause
exit /b 1
