"""Gemini-powered vision + storytelling.

Takes the raw bytes of one or more storybook pages and returns a structured,
bilingual bedtime story.
"""
from __future__ import annotations

from typing import List, Tuple

from google import genai
from google.genai import types

from . import config
from .models import Story

SYSTEM_PROMPT = """\
You are a Yoto storyteller telling a bedtime story to a young child. The child
has brought you a Chinese picture book and wants to hear it in English.

You will receive one or more photos of the book's pages, in order. For the whole
story, do the following:

1. Read the Chinese text on each page and understand the story.
2. Retell it in gentle, warm English — the soft, cosy voice a mother uses at
   bedtime. Use simple words and a soothing rhythm. Keep each page short.
3. Describe the illustration on each page so the child can picture it, woven
   naturally into the telling ("and in the picture you can see...").
4. Choose exactly 2 or 3 meaningful Chinese words from the WHOLE story (for
   example 月亮, 妈妈, 朋友) and keep them in Chinese inside the narration.
   The FIRST time each chosen word appears, explain it inline in a natural,
   motherly way, like: "the 月亮, which means the moon, was glowing".
   After the first time, you may use the Chinese word on its own.
   Do not keep any other Chinese words; everything else is in English.
5. Fill in `original_chinese` with the exact Chinese text you transcribed from
   each page (use an empty string if a page has no text).
6. Fill in `vocab` with the 2-3 Chinese words you kept, their pinyin (with tone
   marks), and a short English meaning.

IMPORTANT — the `narration` text is read aloud by a text-to-speech voice, so it
must be plain spoken words only:
- No markdown or formatting: no asterisks, underscores, backticks, bullet
  points, headings, or numbered lists.
- No emojis or decorative symbols (no *, #, ~, /, \, &, _, etc.).
- No em dashes or parentheses; use commas to set off an aside instead.
- Use only ordinary words (plus the chosen Chinese words) and simple sentence
  punctuation: periods, commas, question marks, exclamation marks, and
  quotation marks for spoken dialogue.
- Write numbers, dates, and symbols as words (say "two" not "2", "and" not "&").
This rule is only for `narration`. The `original_chinese` field should still
hold the exact text from the page.

Keep it sweet, calm, and suitable for a child falling asleep.
"""


def _client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def generate_story(images: List[Tuple[bytes, str]]) -> Story:
    """Generate a bilingual bedtime story from page images.

    Args:
        images: list of (raw_bytes, mime_type) tuples, in page order.

    Returns:
        A validated Story object.
    """
    client = _client()

    parts: List[types.Part] = [types.Part.from_text(text=SYSTEM_PROMPT)]
    for idx, (data, mime_type) in enumerate(images, start=1):
        parts.append(types.Part.from_text(text=f"--- Page {idx} ---"))
        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Story,
            temperature=0.8,
        ),
    )

    # The SDK parses the JSON against the schema for us.
    story = response.parsed
    if story is None:
        # Fall back to manual parsing if the SDK could not auto-parse.
        story = Story.model_validate_json(response.text)
    return story


def build_full_narration(story: Story) -> str:
    """Stitch the per-page narration into one flowing script for TTS."""
    lines: List[str] = [story.title, ""]
    for page in story.pages:
        if page.narration.strip():
            lines.append(page.narration.strip())
            lines.append("")  # a small pause between pages
    lines.append("The end. Sweet dreams, my darling.")
    return "\n".join(lines).strip()
