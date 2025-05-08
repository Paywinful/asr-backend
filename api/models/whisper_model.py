# import subprocess
import torch
from faster_whisper import WhisperModel



# After conversion, set the model path to the converted model's directory
MODEL_NAME = r"C:\Users\fiifi\OneDrive\Documents\ASR WEB\akan-asr-api\api\models\output"  # Adjust path after conversion

# Choose the device for processing (CUDA or CPU) this is optional
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(MODEL_NAME, device=device, compute_type="int8")

def transcribe_with_whisper(audio_data):
    """Transcribe audio using Whisper model."""
    segments, _ = model.transcribe(audio_data, beam_size=5, language='en', word_timestamps=False, vad_filter=True)
    
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    
    return transcription

