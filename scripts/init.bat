@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\.."

set "UV_CACHE_DIR=%CD%\.uv-cache"
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%" >nul 2>nul

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment with Python 3.10...
  uv venv .venv --python 3.10
  if errorlevel 1 goto :fail
)

echo Updating submodules...
git submodule update --init --recursive
if errorlevel 1 goto :fail

echo Syncing dependencies from pyproject.toml...
uv sync --python .venv\Scripts\python.exe
if errorlevel 1 goto :fail

echo.
echo Setup complete.
echo Activate with:
echo   call .venv\Scripts\activate.bat
echo.
echo Verify local patcher source with:
echo   .venv\Scripts\python.exe -c "import meshtaichi_patcher_core as m; print(m.__file__)"
pause
exit /b 0

:fail
echo.
echo Setup failed.
pause
exit /b 1
