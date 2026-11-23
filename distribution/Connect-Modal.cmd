@echo off
setlocal
set "VQI_ROOT=%~dp0"
if not exist "%VQI_ROOT%runtime\python.exe" (
    echo Extract every file from the Windows ZIP before connecting Modal.
    pause
    exit /b 1
)
pushd "%VQI_ROOT%" || exit /b 1
"%VQI_ROOT%runtime\python.exe" -B -E -s -m modal setup
if errorlevel 1 goto failed
"%VQI_ROOT%runtime\python.exe" -B -E -s -m modal deploy deployment/modal_worker.py
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
