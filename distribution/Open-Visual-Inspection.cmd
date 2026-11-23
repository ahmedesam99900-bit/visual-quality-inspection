@echo off
setlocal
chcp 65001 >nul
set "VQI_ROOT=%~dp0"
if not exist "%VQI_ROOT%runtime\python.exe" (
    echo Extract every file from the Windows ZIP before opening the app.
    pause
    exit /b 1
)
if not defined VQI_WORKSPACE set "VQI_WORKSPACE=%LOCALAPPDATA%\VisualInspectionStudio\workspace"
"%VQI_ROOT%runtime\python.exe" -B -E -s -X utf8 "%VQI_ROOT%deployment\app.py" --workspace "%VQI_WORKSPACE%" --starter "%VQI_ROOT%starter" --port 8766 --open
if errorlevel 1 pause
