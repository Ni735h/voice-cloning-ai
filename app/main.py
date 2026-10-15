from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

import os
import shutil

from app.model import generate_cloned_voice


app = FastAPI(
    title="Voice Cloning AI API",
    description=(
        "AI voice cloning API with automatic "
        "audio preprocessing."
    ),
    version="1.0.0"
)


VOICE_SAMPLE_PATH = (
    "data/voice_samples/uploaded_voice"
)

CURRENT_VOICE_FILE = None


class VoiceRequest(BaseModel):
    text: str


@app.get("/")
def home():

    return {
        "message": (
            "Voice Cloning AI API is running!"
        )
    }


@app.post("/upload-voice")
async def upload_voice(
    file: UploadFile = File(...)
):

    global CURRENT_VOICE_FILE

    allowed_extensions = [
        ".wav",
        ".mp3",
        ".m4a",
        ".mpeg",
        ".mpga"
    ]

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid file name."
        )

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Supported formats: "
                "WAV, MP3, M4A, MPEG and MPGA."
            )
        )

    os.makedirs(
        "data/voice_samples",
        exist_ok=True
    )

    file_path = (
        VOICE_SAMPLE_PATH +
        extension
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    CURRENT_VOICE_FILE = file_path

    return {
        "message": (
            "Voice sample uploaded successfully!"
        ),
        "file": file_path
    }


@app.post("/generate-voice")
def generate_voice(
    request: VoiceRequest
):

    global CURRENT_VOICE_FILE

    if CURRENT_VOICE_FILE is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a voice sample first."
            )
        )

    if not os.path.exists(
        CURRENT_VOICE_FILE
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Voice sample could not be found."
            )
        )

    if not request.text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Text cannot be empty."
            )
        )

    try:

        output_path, audio_info = (
            generate_cloned_voice(
                request.text,
                ref_audio=CURRENT_VOICE_FILE
            )
        )

        return {
            "message": (
                "Voice generated successfully!"
            ),
            "file": output_path,
            "audio_analysis": audio_info
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )