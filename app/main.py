"""FastAPI app: upload storybook pages -> bilingual bedtime story -> audio."""
import uuid
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import config, gemini_client, tts
from .models import AudioRequest, StoryDraft, StoryResponse


def _client_ip(request: Request) -> str:
    """Real client IP, honoring the X-Forwarded-For header set by hosts like Render."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)

app = FastAPI(
    title="Bilingual Bedtime Story Assistant",
    description=(
        "Upload pages from a Chinese picture book. The assistant reads them, "
        "retells the story in a warm motherly voice in English (keeping a few "
        "Chinese words), describes the pictures, and reads it aloud."
    ),
    version="1.0.0",
)

# Wire up per-client rate limiting (returns HTTP 429 when a limit is exceeded).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit(config.RATE_LIMIT_STORY)
async def make_story(request: Request, images: List[UploadFile] = File(...)) -> StoryDraft:
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


@app.post("/api/audio")
@limiter.limit(config.RATE_LIMIT_AUDIO)
async def make_audio(request: Request, req: AudioRequest) -> Response:
    """Read a (possibly edited) narration script aloud and return the MP3 directly.

    The audio bytes are streamed straight back in the response body (no file is
    written), so the client never makes a second request to a server-stored file
    that could be missing after a restart — which was causing 404s in the cloud.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Narration text is empty.")
    try:
        audio = await tts.synthesize_bytes(text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Audio synthesis failed: {exc}")
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="bedtime-story.mp3"'},
    )


@app.post("/api/tell", response_model=StoryResponse)
@limiter.limit(config.RATE_LIMIT_STORY)
async def tell_story(request: Request, images: List[UploadFile] = File(...)) -> StoryResponse:
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
