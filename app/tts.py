"""Text-to-speech using Microsoft Edge TTS.

Edge TTS is free and needs no API key. Its multilingual neural voices speak
English and Chinese in a single stream, which is exactly what the bilingual
narration needs.
"""
from __future__ import annotations

import edge_tts

from . import config


def _communicate(text: str) -> edge_tts.Communicate:
    return edge_tts.Communicate(
        text,
        voice=config.TTS_VOICE,
        rate=config.TTS_RATE,
        pitch=config.TTS_PITCH,
    )


async def synthesize_bytes(text: str) -> bytes:
    """Render `text` to MP3 bytes in memory (no file on disk).

    Returning the audio directly lets the API stream it straight to the client,
    which avoids depending on ephemeral server storage for a follow-up request.
    """
    audio = bytearray()
    async for chunk in _communicate(text).stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            audio.extend(chunk["data"])
    if not audio:
        raise RuntimeError("Text-to-speech returned no audio.")
    return bytes(audio)


async def synthesize(text: str, out_path: str) -> str:
    """Render `text` to an MP3 file at `out_path`. Returns the path."""
    await _communicate(text).save(out_path)
    return out_path


async def list_voices() -> list[dict]:
    """Return available Edge TTS voices (handy for picking one)."""
    return await edge_tts.list_voices()
