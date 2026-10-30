@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  uv venv --python 3.13 .venv
  if errorlevel 1 exit /b 1
)
uv pip install --python ".venv\Scripts\python.exe" -r requirements-btl3.lock
if errorlevel 1 exit /b 1
uv pip install --python ".venv\Scripts\python.exe" --no-deps -e .
exit /b %errorlevel%
