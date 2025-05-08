from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
# from models.wav2vec2_model import transcribe_with_wav2vec2
from models.whisper_model import transcribe_with_whisper
import tempfile
import shutil

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...), model: str = Form(...)):
    # if model == "wav2vec2":
        # return {"transcription": transcribe_with_wav2vec2(file.file)}
    if model == "whisper":
        return {"transcription": transcribe_with_whisper(file.file)}
    else:
        return {"error": "Unsupported model"}
