import os
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel


app = FastAPI(title="Simple Transcription Pipeline")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm"}

jobs: Dict[str, Dict[str, Any]] = {}

model = WhisperModel("base", device="cpu", compute_type="int8")


def validate_file(filename: str):
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {extension}"
        )


def transcribe_audio(file_path: str) -> Dict[str, Any]:
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        vad_filter=True
    )

    transcript_segments: List[Dict[str, Any]] = []

    for segment in segments:
        transcript_segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    full_text = " ".join(item["text"] for item in transcript_segments)

    return {
        "language": info.language,
        "text": full_text,
        "segments": transcript_segments
    }


@app.post("/transcriptions")
async def create_transcription(file: UploadFile = File(...)):
    validate_file(file.filename)

    job_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()
    saved_file = UPLOAD_DIR / f"{job_id}{file_extension}"

    with saved_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {
        "status": "processing",
        "file_path": str(saved_file),
        "result": None,
        "error": None
    }

    try:
        result = transcribe_audio(str(saved_file))

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result

    except Exception as error:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(error)

    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"]
    }


@app.get("/transcriptions/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "error": jobs[job_id]["error"]
    }


@app.get("/transcriptions/{job_id}/result")
def get_transcription_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != "completed":
        return {
            "job_id": job_id,
            "status": jobs[job_id]["status"],
            "message": "Transcription is not completed yet"
        }

    return {
        "job_id": job_id,
        "status": "completed",
        "result": jobs[job_id]["result"]
    }