@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where uv >nul 2>nul
    if errorlevel 1 (
        echo.
        echo GPR Layer Audit could not start because uv is not installed.
        echo Install uv from https://docs.astral.sh/uv/ and run this file again.
        echo.
        pause
        exit /b 1
    )

    echo Preparing the GPR Layer Audit Python environment...
    uv sync --extra dev
    if errorlevel 1 (
        echo.
        echo Environment setup failed. Review the messages above.
        pause
        exit /b 1
    )
)

echo Starting GPR Layer Audit from source...
echo Keep this window open while the application is running.
echo.
".venv\Scripts\python.exe" -m gpr_layer_audit.ui.app

if errorlevel 1 (
    echo.
    echo GPR Layer Audit stopped with an error. Review the traceback above.
    pause
)

endlocal
