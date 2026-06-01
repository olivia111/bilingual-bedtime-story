"""Application configuration loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env sitting next to the project root (one level up from this file's package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaMultilingualNeural").strip()
TTS_RATE = os.getenv("TTS_RATE", "-10%").strip()
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz").strip()

# Where generated audio files are written and served from.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

STATIC_DIR = PROJECT_ROOT / "static"

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}
