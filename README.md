# Simple Transcription Pipeline

## Overview

This project is a simple transcription pipeline that accepts an audio file, transcribes speech into text, and returns timestamped transcript segments.

The focus is on engineering decisions, not training a model from scratch.

## Features

- Accepts audio files such as WAV, MP3, M4A, FLAC, AAC, OGG, and WEBM
- Uses an open-source speech-to-text model
- Returns full transcript text
- Returns timestamps per segment
- Exposes the pipeline as a REST API
- Tracks transcription status using job IDs

## Tech Stack

- Python
- FastAPI
- faster-whisper
- Uvicorn

## Setup

Install dependencies:

```bash
pip install -r requirements.txt