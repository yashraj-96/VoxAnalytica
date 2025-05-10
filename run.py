#!/usr/bin/env python3
"""
Easy startup script for DataWhisperer
This script handles dependency checking and starts the application
with fallback options for missing dependencies.
"""

import os
import sys
import subprocess
import platform
import time

def check_dependency(module_name):
    """Check if a Python module is installed."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def install_module(module_name):
    """Install a Python module using pip."""
    print(f"Installing {module_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", module_name
        ])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {module_name}")
        return False

def check_ollama():
    """Check if Ollama is installed and running."""
    # Check if Ollama is installed
    try:
        subprocess.run(
            ["which", "ollama"] if platform.system() != "Windows" else ["where", "ollama"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Check if Ollama server is running
        try:
            import requests
            requests.get("http://localhost:11434/api/version")
            return True
        except:
            # Try to start Ollama server
            print("Starting Ollama server...")
            if platform.system() == "Windows":
                subprocess.Popen(["ollama", "serve"], 
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                subprocess.Popen(["ollama", "serve"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(3)
            return True
            
    except:
        print("Ollama not found or not running. Will use fallback mode.")
        return False

def main():
    """Check dependencies and start the application."""
    print("=== DataWhisperer Startup ===")
    
    # Check essential dependencies
    essential_deps = ["streamlit", "pandas", "numpy", "matplotlib"]
    for dep in essential_deps:
        if not check_dependency(dep):
            if not install_module(dep):
                print(f"ERROR: Essential dependency {dep} could not be installed.")
                print("Please install it manually with: pip install " + dep)
                return False
    
    # Check Ollama
    if not check_ollama():
        print("⚠️ Running in text-only simulation mode without Ollama")
        # Set environment variable to trigger simulation mode
        os.environ["SIMULATE_LLM"] = "true"
    
    # Check for PyAudio and Whisper (for voice features)
    voice_available = True
    
    if not check_dependency("sounddevice") or not check_dependency("soundfile"):
        print("⚠️ Sound recording libraries not available")
        voice_available = False
    
    # Start the application
    print("\nStarting DataWhisperer...")
    os.environ["PYTHONPATH"] = os.getcwd()
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    
    return True

if __name__ == "__main__":
    main()