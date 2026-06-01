"""FastAPI app: upload storybook pages -> bilingual bedtime story -> audio."""
from __future__ import annotations

import uuid
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config, gemini_client, tts
from .models import AudioRequest, AudioResponse, StoryDraft, StoryResponse

app = FastAPI(
    title="Bilingual Bedtime Story Assistant",
    description=(
        "Upload pages from a Chinese picture book. The assistant reads them, "
        "retells the story in a warm motherly voice in English (keeping a few "
        "Chinese words), describes the pictures, and reads it aloud."
    ),
    version="1.0.0",
)

# Serve generated audio.
app.mount("/audio", StaticFiles(directory=str(config.OUTPUT_DIR)), name="audio")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = config.STATIC_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "gemini_key_set": bool(config.GEMINI_API_KEY)}


async def _read_images(images: List[UploadFile]) -> list[tuple[bytes, str]]:
    """Validate uploads and return (bytes, mime_type) tuples in upload order."""
    if not images:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    page_data: list[tuple[bytes, str]] = []
    for img in images:
        content_type = (img.content_type or "").lower()
        if content_type not in config.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{content_type}' for {img.filename}. "
                f"Allowed: {sorted(config.ALLOWED_IMAGE_TYPES)}",
            )
        data = await img.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{img.filename} is empty.")
        page_data.append((data, content_type))
    return page_data


async def _synthesize(text: str) -> str:
    """Render narration text to a fresh MP3 and return its public /audio URL."""
    audio_name = f"story_{uuid.uuid4().hex}.mp3"
    audio_path = config.OUTPUT_DIR / audio_name
    try:
        await tts.synthesize(text, str(audio_path))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Audio synthesis failed: {exc}")
    return f"/audio/{audio_name}"


@app.post("/api/story", response_model=StoryDraft)
async def make_story(images: List[UploadFile] = File(...)) -> StoryDraft:
    """Turn uploaded storybook pages into a bilingual bedtime story (text only).

    Returns the structured story and an editable narration script. No audio is
    generated yet — the client reviews/edits the text, then calls /api/audio.
    """
    page_data = await _read_images(images)
    try:
        story = gemini_client.generate_story(page_data)
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=502, detail=f"Story generation failed: {exc}")

    return StoryDraft(
        story=story,
        full_narration=gemini_client.build_full_narration(story),
    )


@app.post("/api/audio", response_model=AudioResponse)
async def make_audio(req: AudioRequest) -> AudioResponse:
    """Read a (possibly edited) narration script aloud and return the audio URL."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Narration text is empty.")
    audio_url = await _synthesize(text)
    return AudioResponse(audio_url=audio_url)


@app.post("/api/tell", response_model=StoryResponse)
async def tell_story(images: List[UploadFile] = File(...)) -> StoryResponse:
    """Legacy one-shot: uploaded pages -> bilingual bedtime story + audio."""
    page_data = await _read_images(images)

    # 1. Vision + storytelling via Gemini.
    try:
        story = gemini_client.generate_story(page_data)
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=502, detail=f"Story generation failed: {exc}")

    full_narration = gemini_client.build_full_narration(story)

    # 2. Text-to-speech via Edge TTS.
    audio_url = await _synthesize(full_narration)

    return StoryResponse(
        story=story,
        full_narration=full_narration,
        audio_url=audio_url,
    )
