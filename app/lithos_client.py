"""Lithos AI powered vision + storytelling.

Takes the raw bytes of one or more storybook pages and returns a structured,
bilingual bedtime story. Talks to Lithos AI's OpenAI-compatible endpoint
(https://api.lithosai.cloud/v1) using the Kimi-K3 model.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from . import config
from .images import prepare_images
from .models import Story

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are telling a bedtime story aloud for a young child to listen to. The child
has a Chinese picture book and wants to hear it in English.

You will receive one or more photos of the book's pages, in order. For the whole
story, do the following:

1. Read the Chinese text on each page and understand the story.
2. Retell it aloud in warm, plain English. Read the room like a good children's
   audio storyteller: unhurried, clear, a little playful, never syrupy and never
   talking down. Short sentences. Concrete words. Let the pictures and the
   events carry the feeling instead of announcing it.
3. Describe the illustration on each page so the child can picture it, woven
   into the telling rather than tacked on as a caption.
4. Choose around 3 meaningful Chinese words from the WHOLE story (for
   example 月亮, 朋友) and keep them in Chinese inside the narration.
   {word_choice}
   The FIRST time each chosen word appears, put English description right beside it:
   "Up climbed the moon, 月亮." or
   "She had found a friend, 朋友."
   After the first time, use the Chinese word on its own and trust the child to
   remember. Do not keep any other Chinese words; everything else is in English.
5. Fill in `original_chinese` with the exact Chinese text you transcribed from
   each page (use an empty string if a page has no text).
6. Fill in `vocab` with the Chinese words you kept, their pinyin (with tone
   marks), and a short English meaning.
6b. Also fill in `vocab_candidates` with 8 to 10 words from the book that
   would each make a good word to keep, including the ones you actually kept.
   Favour concrete, child-friendly words a parent might want to teach. Give
   each one pinyin and a short English meaning. This is the menu the reader
   chooses from, so do not narrate with these — only the `vocab` words appear
   in the narration.
7. Fill in `narration_ssml` for each page as Azure SSML-style content that uses
    per-sentence emotion tags. Use one or more blocks like:
    <mstts:express-as style="cheerful">...</mstts:express-as>
    <mstts:express-as style="sad">...</mstts:express-as>
    You may also use styles such as calm, empathetic, and excited.
    Important: this field must be an SSML fragment only, so do not include
    <speak> or <voice> wrappers.

VOICE — things that break the spell, so never write them:
- Pet names or terms of endearment of any kind: "my darling", "sweetheart",
  "my dear", "little one", "my love", "dear child". Speak TO the child without
  naming what they are to you.
- Dictionary phrasing around the Chinese words: "which means", "that means",
  "meaning", "the Chinese word for", "in Chinese we say", "or as we call it".
  The sentence itself does the teaching. If a sentence only works with one of
  these phrases, rewrite the sentence.
- Narrating the child's feelings or bedtime at them: "isn't that lovely",
  "now close your eyes", "what a sweet story". Just tell the story well.
- Winking at the grown-ups, morals bolted onto the end, or explaining the point
  of the story. Let it land on its own.

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

Keep it calm and easy to follow, the kind of telling a child can fall asleep to.

Reply with a single JSON object and nothing else — no prose, no code fences.
It must match this JSON schema exactly:

{schema}
"""


_FREE_CHOICE = (
    "The reader has not asked for particular words, so the choice is yours."
)

_REQUESTED = (
    "IMPORTANT: the reader has asked for these exact words to be kept: {words}. "
    "Keep every one of them (each explained inline the first time it appears) "
    "and keep no others, even if you would have chosen differently. If one of "
    "them does not appear in the book, use it anyway wherever it fits the story "
    "naturally."
)

RESTYLE_PROMPT = """\
Here is a bilingual bedtime story you wrote, as JSON:

{story}

The reader has now chosen exactly which Chinese words they want kept:
{words}

Rewrite it so that ONLY those words remain in Chinese. Every other Chinese word
becomes plain English. The FIRST time each kept word appears, make its meaning
obvious from the sentence around it by putting a plain English description right
beside it, for example: "Up climbed the 月亮, big and round and silver." After
that, use the word on its own. If a chosen word did not appear before, weave it
in wherever it fits the story naturally.

Keep the same voice: warm, plain, unhurried, a little playful. Never use pet
names ("my darling", "sweetheart", "little one") and never use dictionary
phrasing around the Chinese words ("which means", "meaning", "the Chinese word
for"). If a sentence only works with one of those, rewrite the sentence.

Keep the title, the page order, each page's `original_chinese`, each page's
`illustration`, and `vocab_candidates` exactly as they are. Rewrite `narration`
and `narration_ssml` for each page, and set `vocab` to exactly the chosen words
with pinyin and a short English meaning.

The same narration rules apply: `narration` is read aloud by a text-to-speech
voice, so plain spoken words only, no markdown, no emoji, no symbols, no em
dashes or parentheses, and numbers written as words.

Reply with a single JSON object and nothing else — no prose, no code fences.
It must match this JSON schema exactly:

{schema}
"""


def _story_schema() -> Dict[str, Any]:
    """The Story model as a JSON schema the model can follow."""
    return Story.model_json_schema()


def _client() -> OpenAI:
    if not config.LITHOS_API_KEY:
        raise RuntimeError(
            "LITHOSAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(
        api_key=config.LITHOS_API_KEY,
        base_url=config.LITHOS_BASE_URL,
    )


def _data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may carry fences or chatter."""
    # Some servers inline the reasoning trace in the content rather than in a
    # separate field; drop it before looking for the JSON.
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _request_story(messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Story:
    """Send a chat request and return a validated Story.

    If the endpoint will not enforce the JSON schema server-side, the call is
    retried with a looser response_format and the reply is validated here.
    """
    client = _client()

    kwargs: Dict[str, Any] = {
        "model": config.LITHOS_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "story", "schema": schema},
        },
    }
    if config.LITHOS_REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.LITHOS_REASONING_EFFORT

    # Fall back if the endpoint will not enforce the schema server-side.
    response_format_ladder = [{"type": "json_object"}, None]

    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            if response_format_ladder:
                nxt = response_format_ladder.pop(0)
                logger.warning("Request failed; retrying with response_format=%s.", nxt)
                if nxt is None:
                    kwargs.pop("response_format", None)
                else:
                    kwargs["response_format"] = nxt
                continue
            break

        text = response.choices[0].message.content or ""
        if text.strip():
            try:
                return Story.model_validate_json(_extract_json(text))
            except Exception as exc:
                last_error = exc
        else:
            last_error = RuntimeError("The endpoint returned an empty message.")

        if response_format_ladder:
            nxt = response_format_ladder.pop(0)
            if nxt is None:
                kwargs.pop("response_format", None)
            else:
                kwargs["response_format"] = nxt
            continue
        break

    raise RuntimeError(
        f"Lithos AI ({config.LITHOS_MODEL}) did not return a usable story: {last_error}"
    )


def generate_story(
    images: List[Tuple[bytes, str]],
    keep_words: Optional[List[str]] = None,
) -> Story:
    """Generate a bilingual bedtime story from page images.

    Args:
        images: list of (raw_bytes, mime_type) tuples, in page order.
        keep_words: Chinese words the reader asked to keep. When empty the
            model picks its own.

    Returns:
        A validated Story object.
    """
    # Downscale/normalize the pages first — smaller payload, faster upload, and
    # HEIC pages become JPEG, which every endpoint understands.
    images = prepare_images(images)

    schema = _story_schema()
    keep_words = [w for w in (keep_words or []) if w.strip()]
    word_choice = (
        _REQUESTED.format(words=", ".join(keep_words)) if keep_words else _FREE_CHOICE
    )
    prompt = SYSTEM_PROMPT.replace("{word_choice}", word_choice).replace(
        "{schema}", json.dumps(schema, ensure_ascii=False, indent=2)
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, (data, mime_type) in enumerate(images, start=1):
        content.append({"type": "text", "text": f"--- Page {idx} ---"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(data, mime_type)},
            }
        )

    return _request_story([{"role": "user", "content": content}], schema)


def restyle_story(story: Story, keep: List[str]) -> Story:
    """Rebuild a story's narration around the reader's chosen words.

    Text only — the page images are not re-uploaded, so this is much cheaper
    and faster than regenerating the story from scratch.
    """
    keep = [w.strip() for w in keep if w.strip()]
    if not keep:
        raise ValueError("Choose at least one Chinese word to keep.")

    schema = _story_schema()
    prompt = RESTYLE_PROMPT.format(
        story=story.model_dump_json(indent=2),
        words=", ".join(keep),
        schema=json.dumps(schema, ensure_ascii=False, indent=2),
    )
    restyled = _request_story([{"role": "user", "content": prompt}], schema)

    # The candidate menu is the reader's, not the model's, to change.
    if story.vocab_candidates and not restyled.vocab_candidates:
        restyled.vocab_candidates = story.vocab_candidates
    return restyled


def build_full_narration(story: Story, use_ssml: bool = False) -> str:
    """Stitch the per-page narration into one flowing script for TTS."""
    lines: List[str] = [story.title, ""]
    for page in story.pages:
        chunk = page.narration_ssml.strip() if use_ssml and page.narration_ssml.strip() else page.narration.strip()
        if chunk:
            lines.append(chunk)
            lines.append("")  # a small pause between pages
    return "\n".join(lines).strip()
