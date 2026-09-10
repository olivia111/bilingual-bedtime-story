"""Application configuration loaded from environment / .env file."""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


# Load .env sitting next to the project root (one level up from this file's package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_PLACEHOLDER = re.compile(r"^your_.*_here$", re.IGNORECASE)


def _secret(name: str) -> str:
    """Read a secret, treating an untouched .env.example placeholder as unset.

    Without this, "your_azure_speech_key_here" is a non-empty string, so the
    "is the key configured?" checks pass and the provider answers 401 instead.
    """
    value = os.getenv(name, "").strip()
    return "" if _PLACEHOLDER.match(value) else value

# Story generation runs on Lithos AI's OpenAI-compatible endpoint.
LITHOS_API_KEY = _secret("LITHOSAI_API_KEY")
LITHOS_BASE_URL = os.getenv("LITHOSAI_BASE_URL", "https://api.lithosai.cloud/v1").strip()
LITHOS_MODEL = os.getenv("LITHOSAI_MODEL", "moonshotai/Kimi-K3").strip()

# K3 reasons by default, which for this task is wasted latency and output
# tokens: transcribing a picture book does not need extended deliberation.
# Values: low, high, max (max is the server default). "" to omit the field.
LITHOS_REASONING_EFFORT = os.getenv("LITHOSAI_REASONING_EFFORT", "low").strip()

# Uploaded page photos are downscaled before being sent to the model: phone
# photos are large and base64 adds ~33% on top. 1536px on the long edge keeps
# storybook text legible while cutting the payload dramatically.
IMAGE_RESIZE = os.getenv("IMAGE_RESIZE", "true").strip().lower() in ("1", "true", "yes", "on")
IMAGE_MAX_DIM = int(os.getenv("IMAGE_MAX_DIM", "1536"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))
# Per-page ceiling after encoding; quality steps down until it fits.
IMAGE_MAX_BYTES = int(os.getenv("IMAGE_MAX_KB", "900")) * 1024

# Gemini is only used by the optional "gemini" TTS provider below.
GEMINI_API_KEY = _secret("GEMINI_API_KEY")

# TTS provider:
#   "azure"  = Azure AI Speech (needs a key; free F0 tier; full voice catalog
#              incl. native multilingual zh-CN-XiaoxiaoMultilingualNeural)
#   "gemini" = Gemini's native TTS (paid per call)
#   "edge"   = free Microsoft Edge voices (limited catalog)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "azure").strip().lower()

# Convert Chinese characters to tone-marked Pinyin before TTS (improves tones on
# voices that mispronounce characters). Affects audio only, not the displayed story.
TTS_PINYIN = os.getenv("TTS_PINYIN", "false").strip().lower() in ("1", "true", "yes", "on")

# Wrap each Chinese run with ",," so the voice pauses around it (audio only).
TTS_PAD_CHINESE = os.getenv("TTS_PAD_CHINESE", "true").strip().lower() in ("1", "true", "yes", "on")

# Azure AI Speech settings (used when TTS_PROVIDER == "azure").
AZURE_SPEECH_KEY = _secret("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus").strip()
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "zh-CN-XiaoxiaoMultilingualNeural").strip()
# Azure speaking style (mstts:express-as). "story" = storytelling. Empty = none.
AZURE_TTS_STYLE = os.getenv("AZURE_TTS_STYLE", "story").strip()
# Style intensity for mstts:express-as (e.g. 1.0 to 2.0). Empty = provider default.
AZURE_TTS_STYLE_DEGREE = os.getenv("AZURE_TTS_STYLE_DEGREE", "1.25").strip()
# Optional speaking role for compatible voices (e.g. Narrator, YoungAdultFemale).
AZURE_TTS_ROLE = os.getenv("AZURE_TTS_ROLE", "Narrator").strip()
# Auto-insert sentence and clause pauses in SSML for storytelling cadence.
AZURE_TTS_SENTENCE_BREAK_MS = os.getenv("AZURE_TTS_SENTENCE_BREAK_MS", "650").strip()
AZURE_TTS_CLAUSE_BREAK_MS = os.getenv("AZURE_TTS_CLAUSE_BREAK_MS", "220").strip()

# Gemini TTS settings (used when TTS_PROVIDER == "gemini").
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore").strip()
GEMINI_TTS_STYLE = os.getenv(
    "GEMINI_TTS_STYLE",
    "Read in a warm, gentle, soothing bedtime voice, slowly and softly:",
).strip()

# Edge TTS settings (used when TTS_PROVIDER == "edge").
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaMultilingualNeural").strip()
TTS_RATE = os.getenv("TTS_RATE", "-10%").strip()
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz").strip()

# Per-client rate limits (slowapi syntax; combine windows with ";").
# Story generation hits the paid Lithos AI API, so it is capped tighter than audio.
RATE_LIMIT_STORY = os.getenv("RATE_LIMIT_STORY", "10/minute;100/day").strip()
RATE_LIMIT_AUDIO = os.getenv("RATE_LIMIT_AUDIO", "30/minute;300/day").strip()

# Where generated audio files are written and served from.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

STATIC_DIR = PROJECT_ROOT / "static"

# Sound effect inserted between pages (blank-line-separated sections) in narration audio.
# MP3 for the edge provider; WAV (raw PCM) for the gemini provider.
PAGE_FLIP_SOUND = Path(__file__).resolve().parent / "assets" / "page-flip-01a.mp3"
PAGE_FLIP_WAV = Path(__file__).resolve().parent / "assets" / "page-flip-01a.wav"

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}
