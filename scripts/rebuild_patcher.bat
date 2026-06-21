@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\.."

if not exist .venv\Scripts\python.exe (
  echo ERROR: .venv not found. Run scripts\init.bat first.
  pause
  exit /b 1
)

if not exist third_party\meshtaichi_patcher\setup.py (
  echo ERROR: submodule not found: third_party\meshtaichi_patcher
  echo Run: git submodule update --init --recursive
  pause
  exit /b 1
)

echo Updating submodules...
git submodule update --init --recursive
if errorlevel 1 goto :fail

echo Cleaning old patcher build directory...
if exist third_party\meshtaichi_patcher\build rmdir /s /q third_party\meshtaichi_patcher\build

echo Rebuilding and reinstalling local meshtaichi_patcher...
uv run pip install -e third_party\\meshtaichi_patcher --no-build-isolation --force-reinstall --no-deps
if errorlevel 1 goto :fail

echo Verifying import...
uv run python -c "import meshtaichi_patcher_core as m; print(m.__file__); print(getattr(m, '__version__', 'dev'))"
if errorlevel 1 goto :fail

echo.
echo Rebuild finished successfully.
pause
exit /b 0

:fail
echo.
echo Rebuild failed.
pause
exit /b 1
