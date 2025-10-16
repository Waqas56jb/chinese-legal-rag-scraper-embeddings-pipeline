#!/usr/bin/env python3
"""
Startup script for Chinese Legal RAG Text Generation API
Supports local development and ngrok deployment
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "main.py",
        "dataset/dataset_clean.csv",
        "outputs_seq_models"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files/directories:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure you have:")
        print("1. Trained models in 'outputs_seq_models' directory")
        print("2. Dataset file at 'dataset/dataset_clean.csv'")
        return False
    
    print("✅ All required files found")
    return True


# Load environment variables early so child processes (e.g., uvicorn) see them
load_dotenv()

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_local_server(host="0.0.0.0", port=8000, reload=True):
    """Run the FastAPI server locally"""
    print(f"🚀 Starting API server on {host}:{port}")
    print(f"📖 API Documentation: http://localhost:{port}/docs")
    print(f"🔍 Alternative docs: http://localhost:{port}/redoc")
    print(f"❤️  Health check: http://localhost:{port}/health")
    
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        print("❌ uvicorn not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn[standard]"])
        import uvicorn
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

def setup_ngrok(port=8000, auth_token=None):
    """Setup and run ngrok tunnel"""
    try:
        # Check if ngrok is installed
        result = subprocess.run(["ngrok", "version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ngrok not found. Please install ngrok:")
            print("   1. Download from: https://ngrok.com/download")
            print("   2. Extract and add to PATH")
            print("   3. Sign up at https://ngrok.com/ to get auth token")
            return False
        
        print("✅ ngrok found")
        
        # Set auth token if provided
        if auth_token:
            print("🔑 Setting ngrok auth token...")
            subprocess.run(["ngrok", "config", "add-authtoken", auth_token])
        
        print(f"🌐 Starting ngrok tunnel on port {port}...")
        print("⏳ This will create a public URL for your API...")
        
        # Start ngrok in background
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", str(port), "--log=stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give ngrok time to start
        time.sleep(3)
        
        # Get ngrok URL
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:4040/api/tunnels"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if data.get("tunnels"):
                    public_url = data["tunnels"][0]["public_url"]
                    print(f"🎉 ngrok tunnel active!")
                    print(f"🌍 Public URL: {public_url}")
                    print(f"📖 API Docs: {public_url}/docs")
                    print(f"❤️  Health: {public_url}/health")
                else:
                    print("⚠️  ngrok started but no tunnels found")
            else:
                print("⚠️  Could not retrieve ngrok URL, but tunnel should be active")
                print("   Check http://localhost:4040 for ngrok dashboard")
        except:
            print("⚠️  Could not retrieve ngrok URL automatically")
            print("   Check http://localhost:4040 for ngrok dashboard")
        
        return True
        
    except FileNotFoundError:
        print("❌ ngrok not found in PATH")
        return False

def main():
    parser = argparse.ArgumentParser(description="Chinese Legal RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--ngrok", action="store_true", help="Start ngrok tunnel")
    parser.add_argument("--ngrok-token", help="ngrok auth token")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-checks", action="store_true", help="Skip file checks")
    
    args = parser.parse_args()
    
    print("🚀 Chinese Legal RAG Text Generation API")
    print("=" * 50)
    
    # Check requirements
    if not args.skip_checks and not check_requirements():
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Setup ngrok if requested
    if args.ngrok:
        if not setup_ngrok(args.port, args.ngrok_token):
            print("❌ Failed to setup ngrok")
            sys.exit(1)
        
        # Give ngrok time to fully initialize
        time.sleep(2)
    
    # Start the server
    try:
        run_local_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
