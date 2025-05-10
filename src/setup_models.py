# setup_models.py - Setup required models for DataWhisperer
import os
import sys
import wget
import subprocess
import platform
import requests
import time

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        # Check if the ollama command exists
        subprocess.run(
            ["which", "ollama"] if platform.system() != "Windows" else ["where", "ollama"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False

def install_ollama():
    """Guide user through installing Ollama."""
    system = platform.system()
    print("\nOllama not found. Please install Ollama:")
    
    if system == "Darwin":  # macOS
        print("For macOS, download from: https://ollama.ai/download/mac")
        print("Or install via Homebrew with: brew install ollama")
    elif system == "Linux":
        print("For Linux, run: curl -fsSL https://ollama.ai/install.sh | sh")
    elif system == "Windows":
        print("For Windows, download from: https://ollama.ai/download/windows")
    
    print("\nAfter installing Ollama, run this script again.")
    return False

def start_ollama_server():
    """Start the Ollama server if it's not already running."""
    try:
        # Check if Ollama server is running
        requests.get("http://localhost:11434/api/version")
        print("Ollama server is already running.")
        return True
    except requests.RequestException:
        # Start Ollama server
        print("Starting Ollama server...")
        try:
            # Start in background
            if platform.system() == "Windows":
                subprocess.Popen(["ollama", "serve"], 
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                subprocess.Popen(["ollama", "serve"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            
            # Wait for server to start
            max_retries = 5
            for i in range(max_retries):
                time.sleep(2)  # Wait a bit
                try:
                    requests.get("http://localhost:11434/api/version")
                    print("Ollama server started successfully.")
                    return True
                except requests.RequestException:
                    if i == max_retries - 1:
                        print("Failed to start Ollama server.")
                        return False
                    print(f"Waiting for Ollama server to start (attempt {i+1}/{max_retries})...")
            
            return False
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            return False

def pull_qwen_model():
    """Pull the Qwen model for Ollama."""
    print("\nPulling Qwen 4B model. This may take a while (~3GB download)...")
    try:
        subprocess.run(["ollama", "pull", "qwen:4b"], check=True)
        print("✓ Qwen 4B model pulled successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error pulling Qwen model: {e}")
        return False

def download_whisper_model():
    """Download the Whisper model."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists("models/whisper-tiny.en.pt"):
        print("\nDownloading Whisper tiny.en model...")
        try:
            # Different platforms have different ways to download
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # For M-series Mac
                url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin"
                output_path = "models/whisper-tiny.en.pt"
                wget.download(url, out=output_path)
            else:
                # For other platforms, use pip to download
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "git+https://github.com/openai/whisper.git", 
                    "--upgrade"
                ])
                import whisper
                whisper.load_model("tiny.en", download_root="models")
                
            print("\n✓ Whisper model downloaded successfully!")
            return True
        except Exception as e:
            print(f"\n✗ Error downloading Whisper model: {e}")
            print("Please download manually and place in the 'models/' directory.")
            return False
    else:
        print("✓ Whisper model already exists.")
        return True

def main():
    """Set up models for DataWhisperer."""
    print("=== DataWhisperer Model Setup ===")
    print("This script will set up the required models for DataWhisperer.")
    print("1. Check/Install Ollama")
    print("2. Pull Qwen 4B model through Ollama")
    print("3. Download Whisper tiny.en model")
    print("===================================")
    
    # Check and install Ollama if needed
    if not check_ollama_installed():
        if not install_ollama():
            return False
    
    # Start Ollama server
    if not start_ollama_server():
        print("Failed to start Ollama server. Please start it manually with 'ollama serve'")
        return False
    
    # Pull Qwen model
    if not pull_qwen_model():
        print("Failed to pull Qwen model. Please try manually with 'ollama pull qwen:4b'")
        return False
    
    # Download Whisper model
    if not download_whisper_model():
        print("Failed to download Whisper model.")
        return False
    
    print("\nSetup completed successfully!")
    print("You can now run the application with: streamlit run app.py")
    return True

if __name__ == "__main__":
    main()