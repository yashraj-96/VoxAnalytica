# src/speech.py - Alternative version using SpeechRecognition
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np

# Try to import whisper, but fall back to speech_recognition if needed
try:
    import whisper
    _use_whisper = True
except ImportError:
    try:
        import speech_recognition as sr
        _use_whisper = False
        print("Using SpeechRecognition instead of whisper")
    except ImportError:
        # Need to install at least one of them
        import subprocess
        import sys
        print("Installing SpeechRecognition as a fallback...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "SpeechRecognition", "pyaudio"
        ])
        import speech_recognition as sr
        _use_whisper = False

# Cache for the whisper model
_whisper_model = None

def get_whisper_model():
    """Load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None and _use_whisper:
        # Check if a Whisper model exists in the models directory
        if os.path.exists('models/whisper-tiny.en.pt'):
            _whisper_model = whisper.load_model('models/whisper-tiny.en.pt')
        else:
            # Fall back to downloading the small model if necessary
            _whisper_model = whisper.load_model('tiny.en')
    return _whisper_model

def record_audio(duration=5, sample_rate=16000):
    """Record audio from the microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        
    Returns:
        Path to the temporary audio file
    """
    # Create a temporary file
    temp_file = tempfile.mktemp(suffix='.wav')
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1
    )
    sd.wait()
    
    # Save the recording
    sf.write(temp_file, recording, sample_rate)
    
    return temp_file

def transcribe_audio(audio_file):
    """Transcribe audio file to text.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Transcribed text
    """
    if _use_whisper:
        # Use whisper for transcription
        model = get_whisper_model()
        result = model.transcribe(audio_file)
        transcribed_text = result["text"].strip()
    else:
        # Use SpeechRecognition as fallback
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                # Try Google's speech recognition service
                transcribed_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcribed_text = "Could not understand audio"
            except sr.RequestError:
                # Fall back to offline Sphinx if Google fails
                try:
                    import pocketsphinx
                    transcribed_text = recognizer.recognize_sphinx(audio_data)
                except:
                    transcribed_text = "Could not connect to speech recognition service"
    
    # Clean up the temporary file
    if os.path.exists(audio_file):
        os.remove(audio_file)
    
    return transcribed_text