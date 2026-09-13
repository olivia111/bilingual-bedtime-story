"""Downscale and normalize uploaded page photos before they go to the model.

Phone photos are typically 3-12 MP and several megabytes each, and base64
encoding adds a further ~33% on the wire. A storybook page only needs enough
resolution for the text to be legible, so each page is rotated upright, shrunk
to fit a bounding box, and re-encoded as JPEG. HEIC/HEIF pages are converted to
JPEG at the same time, since not every model endpoint accepts them.
"""
from __future__ import annotations

import io
import logging
from typing import List, Tuple

from PIL import Image, ImageOps

from . import config

logger = logging.getLogger(__name__)

# Formats worth leaving untouched when re-encoding would not help.
_PASSTHROUGH_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Let Pillow open HEIC/HEIF (iPhone photos) when the plugin is installed.
try:  # pragma: no cover - depends on optional dependency
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception:  # pragma: no cover
    _HEIF_OK = False


def _encode(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def prepare_image(data: bytes, mime_type: str) -> Tuple[bytes, str]:
    """Return a downscaled JPEG version of one page, or the original on failure.

    Never raises: if the bytes cannot be decoded, the original is passed
    through unchanged so the request can still be attempted.
    """
    if not config.IMAGE_RESIZE:
        return data, mime_type

    try:
        with Image.open(io.BytesIO(data)) as img:
            # Honour the EXIF orientation flag, otherwise sideways phone photos
            # reach the model rotated and the text is unreadable.
            img = ImageOps.exif_transpose(img)

            # Flatten transparency onto white; JPEG has no alpha channel.
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                flat = Image.new("RGB", img.size, (255, 255, 255))
                flat.paste(img, mask=img.split()[-1])
                img = flat
            elif img.mode != "RGB":
                img = img.convert("RGB")

            max_dim = config.IMAGE_MAX_DIM
            resized = max(img.size) > max_dim
            if resized:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)

            quality = config.IMAGE_JPEG_QUALITY
            out = _encode(img, quality)

            # If it is still heavy, step the quality down before giving up.
            while len(out) > config.IMAGE_MAX_BYTES and quality > 45:
                quality -= 10
                out = _encode(img, quality)

            # Re-encoding a small, already-compact image can make it bigger
            # (flat-colour PNGs especially). If nothing was downscaled and the
            # format is one the endpoint already accepts, keep the original.
            if not resized and len(out) >= len(data) and mime_type in _PASSTHROUGH_TYPES:
                return data, mime_type

            return out, "image/jpeg"
    except Exception as exc:
        logger.warning("Could not resize a %s page (%s); sending it as-is.", mime_type, exc)
        return data, mime_type


def prepare_images(images: List[Tuple[bytes, str]]) -> List[Tuple[bytes, str]]:
    """Downscale every page, logging the total size saved."""
    before = sum(len(d) for d, _ in images)
    out = [prepare_image(d, m) for d, m in images]
    after = sum(len(d) for d, _ in out)
    if before:
        logger.info(
            "Resized %d page(s): %.1f KB -> %.1f KB (%.0f%% smaller)",
            len(images), before / 1024, after / 1024,
            100 * (1 - after / before) if before else 0,
        )
    return out
