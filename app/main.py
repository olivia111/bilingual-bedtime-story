"""FastAPI app: upload storybook pages -> bilingual bedtime story -> audio."""
import logging
import re
import uuid
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import config, lithos_client, tts
from .models import AudioRequest, RestyleRequest, StoryDraft, StoryResponse


# uvicorn configures its own loggers but leaves the root logger bare, so app
# messages (e.g. how much each page was shrunk) would otherwise go nowhere.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s %(message)s")


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
        "retells the story aloud in warm, plain English (keeping a few "
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
    return {"status": "ok", "lithos_key_set": bool(config.LITHOS_API_KEY)}


async def _read_images(images: List[UploadFile]) -> List[Tuple[bytes, str]]:
    """Validate uploads and return (bytes, mime_type) tuples in upload order."""
    if not images:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    page_data: List[Tuple[bytes, str]] = []
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
    """Render narration text to a fresh audio file and return its public /audio URL."""
    audio, media_type = await tts.synthesize_story(text)
    ext = "wav" if media_type == "audio/wav" else "mp3"
    audio_name = f"story_{uuid.uuid4().hex}.{ext}"
    audio_path = config.OUTPUT_DIR / audio_name
    try:
        audio_path.write_bytes(audio)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Audio synthesis failed: {exc}")
    return f"/audio/{audio_name}"


def _parse_words(raw: Optional[str]) -> List[str]:
    """Split a reader-supplied word list on commas, spaces, or Chinese commas."""
    if not raw:
        return []
    parts = re.split(r"[,，、;；\s]+", raw.strip())
    seen: List[str] = []
    for part in parts:
        word = part.strip()
        if word and word not in seen:
            seen.append(word)
    return seen[:12]  # a bedtime story does not need more than this


@app.post("/api/story", response_model=StoryDraft)
@limiter.limit(config.RATE_LIMIT_STORY)
async def make_story(
    request: Request,
    images: List[UploadFile] = File(...),
    keep_words: str = Form(default=""),
) -> StoryDraft:
    """Turn uploaded storybook pages into a bilingual bedtime story (text only).

    Returns the structured story and an editable narration script. No audio is
    generated yet — the client reviews/edits the text, then calls /api/audio.

    `keep_words` is optional: when the reader already knows which Chinese words
    they want kept, those are used instead of the model's own picks.
    """
    page_data = await _read_images(images)
    try:
        story = lithos_client.generate_story(page_data, keep_words=_parse_words(keep_words))
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=502, detail=f"Story generation failed: {exc}")

    use_ssml = config.TTS_PROVIDER == "azure"
    return StoryDraft(
        story=story,
        full_narration=lithos_client.build_full_narration(story, use_ssml=use_ssml),
    )


@app.post("/api/vocab", response_model=StoryDraft)
@limiter.limit(config.RATE_LIMIT_STORY)
async def change_vocab(request: Request, req: RestyleRequest) -> StoryDraft:
    """Rebuild the narration around a different set of kept Chinese words.

    Text only — the page images are not re-uploaded, so this is much cheaper
    than regenerating the story.
    """
    try:
        story = lithos_client.restyle_story(req.story, req.keep)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Rewriting the story failed: {exc}")

    use_ssml = config.TTS_PROVIDER == "azure"
    return StoryDraft(
        story=story,
        full_narration=lithos_client.build_full_narration(story, use_ssml=use_ssml),
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
        audio, media_type = await tts.synthesize_story(text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Audio synthesis failed: {exc}")
    ext = "wav" if media_type == "audio/wav" else "mp3"
    return Response(
        content=audio,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="bedtime-story.{ext}"'},
    )


@app.post("/api/pinyin")
async def make_pinyin(req: AudioRequest) -> dict:
    """Return the narration with Chinese converted to Pinyin (what TTS receives)."""
    return {"pinyin": tts.to_pinyin(req.text), "enabled": config.TTS_PINYIN}


@app.post("/api/tell", response_model=StoryResponse)
@limiter.limit(config.RATE_LIMIT_STORY)
async def tell_story(request: Request, images: List[UploadFile] = File(...)) -> StoryResponse:
    """Legacy one-shot: uploaded pages -> bilingual bedtime story + audio."""
    page_data = await _read_images(images)

    # 1. Vision + storytelling via Lithos AI.
    try:
        story = lithos_client.generate_story(page_data)
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=502, detail=f"Story generation failed: {exc}")

    use_ssml = config.TTS_PROVIDER == "azure"
    full_narration = lithos_client.build_full_narration(story, use_ssml=use_ssml)

    # 2. Text-to-speech via Edge TTS.
    audio_url = await _synthesize(full_narration)

    return StoryResponse(
        story=story,
        full_narration=full_narration,
        audio_url=audio_url,
    )
