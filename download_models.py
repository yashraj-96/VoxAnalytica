
import os
import sys
import wget
import zipfile
import tarfile
import gdown
import subprocess
import platform

def main():
    """Download and prepare models for DataWhisperer."""
    print("=== DataWhisperer Model Downloader ===")
    print("This script will download the required models for DataWhisperer.")
    print("Models will be stored in the 'models/' directory.")
    print("Total download size: ~500MB")
    print("===================================")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check for whisper-cpp
    if not os.path.exists("models/whisper-tiny.en.pt"):
        print("\n[1/2] Downloading Whisper tiny.en model...")
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
        except Exception as e:
            print(f"\n✗ Error downloading Whisper model: {e}")
            print("Please download manually and place in the 'models/' directory.")
    else:
        print("✓ Whisper model already exists.")
    
    # Check for Phi-2 model
    if not os.path.exists("models/phi-2-q4.gguf"):
        print("\n[2/2] Downloading Phi-2 quantized model...")
        try:
            # URL for the quantized Phi-2 model
            # Using Hugging Face's 4-bit quantized version
            url = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
            output_path = "models/phi-2-q4.gguf"
            
            # Download the model
            wget.download(url, out=output_path)
            
            print("\n✓ Phi-2 model downloaded successfully!")
        except Exception as e:
            print(f"\n✗ Error downloading Phi-2 model: {e}")
            print("Please download manually from https://huggingface.co/TheBloke/phi-2-GGUF/")
            print("and place the Q4_K_M model in the 'models/' directory as 'phi-2-q4.gguf'.")
    else:
        print("✓ Phi-2 model already exists.")
    
    print("\nModel downloads completed!")
    print("You can now run the application with: streamlit run app.py")

if __name__ == "__main__":
    main()