"""Text-to-speech using Microsoft Edge TTS.

Edge TTS is free and needs no API key. Its multilingual neural voices speak
English and Chinese in a single stream, which is exactly what the bilingual
narration needs.
"""
from __future__ import annotations

import re

import edge_tts

from . import config

_page_flip_cache: bytes | None = None


def _page_flip_bytes() -> bytes:
    """The page-flip sound effect (cached). Empty bytes if the file is missing."""
    global _page_flip_cache
    if _page_flip_cache is None:
        try:
            _page_flip_cache = config.PAGE_FLIP_SOUND.read_bytes()
        except OSError:
            _page_flip_cache = b""
    return _page_flip_cache


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


async def synthesize_story_bytes(text: str) -> bytes:
    """Narration to MP3 bytes, with a page-flip sound between sections.

    Sections are separated by blank lines (as produced by build_full_narration,
    one section per page), so a page-flip plays between each page. The flip clip
    is pre-normalized to edge-tts's format (24 kHz mono MP3), so the segments
    concatenate cleanly into one stream.
    """
    segments = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    if len(segments) <= 1:
        return await synthesize_bytes(text)

    flip = _page_flip_bytes()
    out = bytearray()
    for i, segment in enumerate(segments):
        if i and flip:
            out.extend(flip)
        out.extend(await synthesize_bytes(segment))
    return bytes(out)


async def synthesize(text: str, out_path: str) -> str:
    """Render `text` to an MP3 file at `out_path`. Returns the path."""
    await _communicate(text).save(out_path)
    return out_path


async def list_voices() -> list[dict]:
    """Return available Edge TTS voices (handy for picking one)."""
    return await edge_tts.list_voices()
