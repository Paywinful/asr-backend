import os
import torch
from faster_whisper import WhisperModel

# Construct the model path relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output")  # assumes 'output' folder is next to this script

# Choose the device (CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the converted Whisper model
model = WhisperModel(MODEL_PATH, device=device, compute_type="int8")

def transcribe_with_whisper(audio_data):
    """Transcribe audio using Whisper model."""
    segments, _ = model.transcribe(audio_data, beam_size=5, language='en', word_timestamps=False, vad_filter=True)
    
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    
    return transcription
