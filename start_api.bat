@echo off
echo Chinese Legal RAG Text Generation API
echo =====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.7+ and add to PATH.
    pause
    exit /b 1
)

echo Starting API server...
echo.
echo API will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the API
python run_api.py

pause
