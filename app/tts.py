"""Text-to-speech using Microsoft Edge TTS.

Edge TTS is free and needs no API key. Its multilingual neural voices speak
English and Chinese in a single stream, which is exactly what the bilingual
narration needs.
"""
from __future__ import annotations

import edge_tts

from . import config


async def synthesize(text: str, out_path: str) -> str:
    """Render `text` to an MP3 file at `out_path`. Returns the path."""
    communicate = edge_tts.Communicate(
        text,
        voice=config.TTS_VOICE,
        rate=config.TTS_RATE,
        pitch=config.TTS_PITCH,
    )
    await communicate.save(out_path)
    return out_path


async def list_voices() -> list[dict]:
    """Return available Edge TTS voices (handy for picking one)."""
    return await edge_tts.list_voices()
