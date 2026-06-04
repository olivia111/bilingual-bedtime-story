"""Text-to-speech with two providers.

- "gemini": Gemini's native TTS models. Returns raw PCM, which we concatenate
  (with a page-flip between sections) and wrap into a single WAV. Handles
  Chinese and English in one voice with native-quality code-switching.
- "edge": Microsoft Edge TTS (free). Returns MP3, concatenated directly.

The provider is chosen by config.TTS_PROVIDER.
"""
from __future__ import annotations

import asyncio
import io
import re
import time
import urllib.request
import wave
from xml.sax.saxutils import escape

import edge_tts
from google import genai
from google.genai import types
from pypinyin import Style, pinyin

from . import config

_TTS_SAMPLE_RATE = 24000  # both providers emit 24 kHz mono

_CJK = re.compile(r"[㐀-䶿一-鿿]+")


def _to_pinyin(text: str) -> str:
    """Replace runs of Chinese characters with numbered-tone Pinyin, leaving the
    rest of the text untouched (e.g. "the 月亮 moon" -> "the yue4 liang4 moon")."""

    def repl(match: re.Match) -> str:
        syllables = pinyin(
            match.group(), style=Style.TONE3, neutral_tone_with_five=True
        )
        return " ".join(s[0] for s in syllables)

    return _CJK.sub(repl, text)


def to_pinyin(text: str) -> str:
    """Public wrapper: convert Chinese in `text` to numbered-tone Pinyin."""
    return _to_pinyin(text)


def _pad_chinese(text: str) -> str:
    """Wrap each run of Chinese characters with ",," so the voice pauses around it
    (e.g. "the 月亮 moon" -> "the ,,月亮 moon")."""
    return _CJK.sub(lambda m: f",, {m.group()}", text)


def _split_sections(text: str) -> list[str]:
    """Split narration into sections on blank lines (one section per page)."""
    sections = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    return sections or [text.strip()]


# --------------------------------------------------------------------------- #
# Edge TTS (free, MP3)
# --------------------------------------------------------------------------- #
_edge_flip_cache: bytes | None = None


def _edge_flip_bytes() -> bytes:
    global _edge_flip_cache
    if _edge_flip_cache is None:
        try:
            _edge_flip_cache = config.PAGE_FLIP_SOUND.read_bytes()
        except OSError:
            _edge_flip_cache = b""
    return _edge_flip_cache


def _communicate(text: str) -> edge_tts.Communicate:
    return edge_tts.Communicate(
        text,
        voice=config.TTS_VOICE,
        rate=config.TTS_RATE,
        pitch=config.TTS_PITCH,
    )


async def synthesize_bytes(text: str) -> bytes:
    """Render `text` to MP3 bytes in memory via Edge TTS."""
    audio = bytearray()
    async for chunk in _communicate(text).stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            audio.extend(chunk["data"])
    if not audio:
        raise RuntimeError("Text-to-speech returned no audio.")
    return bytes(audio)


async def _edge_story(sections: list[str]) -> bytes:
    if len(sections) <= 1:
        return await synthesize_bytes(sections[0])
    flip = _edge_flip_bytes()
    out = bytearray()
    for i, seg in enumerate(sections):
        if i and flip:
            out.extend(flip)
        out.extend(await synthesize_bytes(seg))
    return bytes(out)


# --------------------------------------------------------------------------- #
# Gemini TTS (paid, PCM -> WAV)
# --------------------------------------------------------------------------- #
_gemini_flip_cache: bytes | None = None


def _gemini_flip_pcm() -> bytes:
    """Raw PCM frames of the page-flip (24 kHz/16-bit/mono)."""
    global _gemini_flip_cache
    if _gemini_flip_cache is None:
        try:
            with wave.open(str(config.PAGE_FLIP_WAV), "rb") as w:
                _gemini_flip_cache = w.readframes(w.getnframes())
        except (OSError, wave.Error):
            _gemini_flip_cache = b""
    return _gemini_flip_cache


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_TTS_SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


_gemini_client: genai.Client | None = None


def _client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _gemini_tts_pcm(text: str, attempts: int = 3) -> bytes:
    """One Gemini TTS call -> raw PCM bytes (blocking; run via to_thread).

    Retries with backoff: the preview TTS models can return transient
    rate-limit/5xx errors, which would otherwise fail a multi-page story.
    """
    prompt = f"{config.GEMINI_TTS_STYLE} {text}" if config.GEMINI_TTS_STYLE else text
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=config.GEMINI_TTS_VOICE
                )
            )
        ),
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = _client().models.generate_content(
                model=config.GEMINI_TTS_MODEL, contents=prompt, config=cfg
            )
            data = resp.candidates[0].content.parts[0].inline_data.data
            if not data:
                raise RuntimeError("Gemini TTS returned no audio.")
            return data
        except Exception as exc:  # transient rate-limit / 5xx
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


async def _gemini_story(sections: list[str]) -> bytes:
    flip = _gemini_flip_pcm()
    pcm = bytearray()
    for i, seg in enumerate(sections):
        if i and flip:
            pcm.extend(flip)
        pcm.extend(await asyncio.to_thread(_gemini_tts_pcm, seg))
    return _pcm_to_wav(bytes(pcm))


# --------------------------------------------------------------------------- #
# Azure AI Speech (subscription key; MP3, full voice catalog)
# --------------------------------------------------------------------------- #
def _azure_tts_mp3(text: str, attempts: int = 3) -> bytes:
    """One Azure Speech REST call -> MP3 bytes (blocking; run via to_thread).

    Uses audio-24khz-48kbitrate-mono-mp3 so the result concatenates cleanly with
    the page-flip MP3. Retries on transient errors.
    """
    url = f"https://{config.AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    body = f"<prosody rate='{config.TTS_RATE}'>{escape(text)}</prosody>"
    if config.AZURE_TTS_STYLE:
        body = f"<mstts:express-as style='{config.AZURE_TTS_STYLE}'>{body}</mstts:express-as>"
    ssml = (
        "<speak version='1.0' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='zh-CN'>"
        f"<voice name='{config.AZURE_TTS_VOICE}'>{body}</voice></speak>"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": config.AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        "User-Agent": "bilingual-bedtime-story",
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, data=ssml.encode("utf-8"), headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("Azure TTS returned no audio.")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
    raise last_error  # type: ignore[misc]


async def _azure_story(sections: list[str]) -> bytes:
    if not config.AZURE_SPEECH_KEY:
        raise RuntimeError("AZURE_SPEECH_KEY is not set.")
    flip = _edge_flip_bytes()  # MP3 flip, same format as Azure output
    out = bytearray()
    for i, seg in enumerate(sections):
        if i and flip:
            out.extend(flip)
        out.extend(await asyncio.to_thread(_azure_tts_mp3, seg))
    return bytes(out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
async def synthesize_story(text: str) -> tuple[bytes, str]:
    """Narration -> (audio_bytes, media_type), with a page-flip between sections.

    Dispatches on config.TTS_PROVIDER. Returns WAV for gemini, MP3 for edge.
    """
    sections = _split_sections(text)
    if config.TTS_PINYIN:
        sections = [_to_pinyin(s) for s in sections]
    if config.TTS_PAD_CHINESE:
        sections = [_pad_chinese(s) for s in sections]
    if config.TTS_PROVIDER == "azure":
        return await _azure_story(sections), "audio/mpeg"
    if config.TTS_PROVIDER == "gemini":
        return await _gemini_story(sections), "audio/wav"
    return await _edge_story(sections), "audio/mpeg"


async def synthesize(text: str, out_path: str) -> str:
    """Render `text` to an MP3 file at `out_path` (Edge TTS). Used by /api/tell."""
    await _communicate(text).save(out_path)
    return out_path


async def list_voices() -> list[dict]:
    """Return available Edge TTS voices (handy for picking one)."""
    return await edge_tts.list_voices()
