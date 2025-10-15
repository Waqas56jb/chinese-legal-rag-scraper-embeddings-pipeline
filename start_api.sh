#!/bin/bash

echo "Chinese Legal RAG Text Generation API"
echo "====================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python not found. Please install Python 3.7+"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
echo

echo "Starting API server..."
echo
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo
echo "Press Ctrl+C to stop the server"
echo

# Run the API
$PYTHON_CMD run_api.py

echo
echo "API server stopped."
