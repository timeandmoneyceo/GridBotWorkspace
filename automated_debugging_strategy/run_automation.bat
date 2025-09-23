@echo off
REM Automated Debugging and Optimization Launcher
REM This batch file launches the master automation pipeline

echo ====================================================================
echo        Automated Debugging and Optimization System
echo ====================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not available or not in PATH
    echo Please ensure Python is installed and accessible
    pause
    exit /b 1
)

REM Change to the automation directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM Check if required files exist
if not exist "master_automation_pipeline.py" (
    echo Error: master_automation_pipeline.py not found
    echo Please ensure you are in the correct directory
    pause
    exit /b 1
)

if not exist "GridbotBackup.py" (
    echo Warning: GridbotBackup.py not found in current directory
)

if not exist "gridbot_websocket_server.py" (
    echo Warning: gridbot_websocket_server.py not found in current directory
)

echo Checking LLM server connection...
REM Simple check if the LLM server is running (optional)
REM You can uncomment and modify this if you want to check connectivity
REM curl -s http://localhost:11434/api/tags >nul 2>&1
REM if errorlevel 1 (
REM     echo Warning: LLM server may not be running at localhost:11434
REM     echo Please ensure your smoll2 model is running
REM )

echo.
echo Starting automation pipeline...
echo.

REM Run the master automation pipeline
python master_automation_pipeline.py --verbose %*

REM Check the exit code
if errorlevel 1 (
    echo.
    echo ====================================================================
    echo Automation pipeline completed with errors
    echo Check the logs for details
    echo ====================================================================
) else (
    echo.
    echo ====================================================================
    echo Automation pipeline completed successfully!
    echo ====================================================================
)

echo.
echo Press any key to exit...
pause >nul