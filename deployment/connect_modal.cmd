@echo off
setlocal
for %%I in ("%~dp0..") do set "VQI_PROJECT=%%~fI"
set "VQI_PYTHON=%VQI_PROJECT%\.venv\Scripts\python.exe"
if not exist "%VQI_PYTHON%" (
    echo Create the project Python environment before connecting Modal.
    pause
    exit /b 1
)

pushd "%VQI_PROJECT%" || exit /b 1
"%VQI_PYTHON%" -B -c "import modal" >nul 2>&1
if errorlevel 1 (
    "%VQI_PYTHON%" -m pip --version >nul 2>&1
    if errorlevel 1 (
        uv pip install --python "%VQI_PYTHON%" -r requirements-cloud.txt
    ) else (
        "%VQI_PYTHON%" -m pip install -r requirements-cloud.txt
    )
    if errorlevel 1 goto failed
)
"%VQI_PYTHON%" -B -m modal setup
if errorlevel 1 goto failed
"%VQI_PYTHON%" -B -m modal deploy deployment/modal_worker.py
if errorlevel 1 goto failed
echo Modal GPU training is connected. Reopen the application.
popd
pause
exit /b 0

:failed
echo Modal setup stopped. Review the message above.
popd
pause
exit /b 1
