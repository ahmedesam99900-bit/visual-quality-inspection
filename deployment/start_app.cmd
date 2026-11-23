@echo off
setlocal
chcp 65001 >nul
for %%I in ("%~dp0..") do set "PCB1_PROJECT=%%~fI"
set "PCB1_PYTHON=%PCB1_PROJECT%\.venv\Scripts\python.exe"
if not exist "%PCB1_PYTHON%" (
    echo Python was not found in the project's .venv.
    pause
    exit /b 1
)
"%PCB1_PYTHON%" -B -X utf8 "%~dp0app.py" --project-root "%PCB1_PROJECT%" --open
if errorlevel 1 pause
