@echo off
REM =============================================================================
REM Air & Insights Agent - Quick Setup Script (Windows)
REM =============================================================================
REM This script automates the entire setup process for testing and verification.
REM 
REM Usage:
REM   setup.bat
REM
REM What it does:
REM   1. Creates Python virtual environment
REM   2. Installs dependencies
REM   3. Sets up environment variables
REM   4. Starts the server
REM   5. Runs verification tests
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo           Air and Insights Agent - Quick Setup
echo ================================================================
echo.

REM -----------------------------------------------------------------------------
REM Step 1: Check Python version
REM -----------------------------------------------------------------------------
echo [Step 1/5] Checking Python version...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    echo         Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM -----------------------------------------------------------------------------
REM Step 2: Create virtual environment
REM -----------------------------------------------------------------------------
echo.
echo [Step 2/5] Setting up virtual environment...

if exist "venv" (
    echo      Virtual environment already exists, using it...
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM -----------------------------------------------------------------------------
REM Step 3: Install dependencies
REM -----------------------------------------------------------------------------
echo.
echo [Step 3/5] Installing dependencies...

pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [OK] Dependencies installed

REM -----------------------------------------------------------------------------
REM Step 4: Setup environment variables
REM -----------------------------------------------------------------------------
echo.
echo [Step 4/5] Configuring environment...

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo      Created .env from .env.example
    ) else (
        echo GITHUB_TOKEN=> .env
        echo NASA_API_KEY=DEMO_KEY>> .env
        echo LOG_LEVEL=INFO>> .env
        echo      Created default .env file
    )
)

echo [OK] Environment configured
echo.
echo [NOTE] If you have a GitHub token, edit .env and add:
echo        GITHUB_TOKEN=your_token_here
echo        Get one at: https://github.com/settings/tokens

REM -----------------------------------------------------------------------------
REM Step 5: Start server
REM -----------------------------------------------------------------------------
echo.
echo [Step 5/5] Starting server...

REM Start server in a new window
start "Air Insights Server" cmd /c "venv\Scripts\python main.py"

echo      Waiting for server to start...
timeout /t 5 /nobreak >nul

REM -----------------------------------------------------------------------------
REM Verification
REM -----------------------------------------------------------------------------
echo.
echo ================================================================
echo                    Verification Tests
echo ================================================================
echo.

REM Test health endpoint
echo [Test 1] Health Check...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [WARN] Server may not be ready yet. Wait a few seconds and try again.
) else (
    echo [OK] Server is running
)

echo.
echo ================================================================
echo                    Setup Complete!
echo ================================================================
echo.
echo The Air and Insights Agent is now running!
echo.
echo Access Points:
echo    Web UI:      http://localhost:8000/
echo    Swagger:     http://localhost:8000/docs
echo    OpenAPI:     http://localhost:8000/openapi.json
echo.
echo Demo prompts to try:
echo    1. "What's the PM2.5 and temperature around 42.6977, 23.3219
echo        for the next 6 hours and should I run outdoors?"
echo    2. "Show today's NASA APOD and summarize in 2 lines."
echo.
echo To stop the server: Close the "Air Insights Server" window
echo.

REM Open browser
echo Opening web UI in browser...
start http://localhost:8000/

pause
