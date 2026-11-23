@echo off
setlocal
chcp 65001 >nul
set "VQI_PROJECT=D:\AI_Projects\visual-quality-inspection"
set "VQI_PYTHON=%VQI_PROJECT%\.venv\Scripts\python.exe"
set "VQI_UPDATE=%~dp0original-project-update.zip"
if not exist "%VQI_PYTHON%" (
    echo The original project Python was not found at "%VQI_PYTHON%".
    pause
    exit /b 1
)
if not exist "%VQI_UPDATE%" (
    echo Keep original-project-update.zip next to this launcher.
    pause
    exit /b 1
)
"%VQI_PYTHON%" -B -X utf8 -c "from pathlib import Path; from tempfile import TemporaryDirectory; from zipfile import ZipFile; import os, runpy, sys; temp=TemporaryDirectory(); p=Path(temp.name); ZipFile(os.environ['VQI_UPDATE']).extractall(p); sys.argv=[str(p/'install.py'), os.environ['VQI_PROJECT']]; runpy.run_path(sys.argv[0], run_name='__main__')"
if errorlevel 1 (
    pause
    exit /b 1
)
call "%VQI_PROJECT%\deployment\start_app.cmd"
