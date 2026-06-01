"""Pydantic models shared between the Gemini client and the API layer."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class VocabItem(BaseModel):
    """A Chinese word kept in the story, with its reading and meaning."""

    chinese: str = Field(description="The Chinese word, e.g. 月亮")
    pinyin: str = Field(description="Pinyin with tone marks, e.g. yuè liang")
    english: str = Field(description="Short English meaning, e.g. the moon")


class StoryPage(BaseModel):
    """One page / picture from the storybook."""

    page: int = Field(description="1-based page number, in upload order")
    original_chinese: str = Field(
        description="The Chinese text transcribed from the page (empty string if none)"
    )
    illustration: str = Field(
        description="A gentle description of what the illustration shows"
    )
    narration: str = Field(
        description=(
            "The bedtime narration for this page in a warm motherly tone. "
            "Weaves the story and the picture together. Keeps the chosen Chinese "
            "words inline and explains each one in English the first time it appears."
        )
    )


class Story(BaseModel):
    """The full generated bilingual bedtime story."""

    title: str = Field(description="A short, sweet English title for the story")
    pages: List[StoryPage] = Field(description="Pages in upload order")
    vocab: List[VocabItem] = Field(
        description="Glossary of the 2-3 Chinese words kept in the story"
    )


class StoryDraft(BaseModel):
    """Story + editable narration text, returned before audio is synthesized."""

    story: Story
    full_narration: str = Field(
        description="The stitched narration script, editable before audio synthesis"
    )


class AudioRequest(BaseModel):
    """Request to turn a (possibly edited) narration script into audio."""

    text: str = Field(description="The narration script to read aloud")


class AudioResponse(BaseModel):
    """The synthesized-audio result."""

    audio_url: str


class StoryResponse(BaseModel):
    """What the legacy one-shot /api/tell endpoint returns to the client."""

    story: Story
    full_narration: str
    audio_url: str
