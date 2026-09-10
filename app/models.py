"""Pydantic models shared between the Lithos AI client and the API layer."""
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
            "The bedtime narration for this page: warm, plain, unhurried spoken "
            "English. Weaves the story and the picture together. Keeps the chosen "
            "Chinese words inline, and the first time each one appears the "
            "surrounding sentence makes its meaning plain. No pet names, and no "
            "phrases like 'which means'."
        )
    )
    narration_ssml: str = Field(
        default="",
        description=(
            "Optional Azure SSML fragment for this page (inner content only, no "
            "<speak> or <voice> wrapper), using per-sentence mstts:express-as tags."
        ),
    )


class Story(BaseModel):
    """The full generated bilingual bedtime story."""

    title: str = Field(description="A short, sweet English title for the story")
    pages: List[StoryPage] = Field(description="Pages in upload order")
    vocab: List[VocabItem] = Field(
        description="Glossary of the Chinese words currently kept in the story"
    )
    vocab_candidates: List[VocabItem] = Field(
        default_factory=list,
        description=(
            "Every Chinese word from the book that would work well as a keeper, "
            "including the ones in `vocab`. The reader picks from this list."
        ),
    )


class StoryDraft(BaseModel):
    """Story + editable narration text, returned before audio is synthesized."""

    story: Story
    full_narration: str = Field(
        description="The stitched narration script, editable before audio synthesis"
    )


class RestyleRequest(BaseModel):
    """Rebuild a story's narration around a different set of kept words."""

    story: Story = Field(description="The story as it was last returned")
    keep: List[str] = Field(
        description="The Chinese words the reader wants kept, e.g. ['月亮', '朋友']"
    )


class AudioRequest(BaseModel):
    """Request to turn a (possibly edited) narration script into audio."""

    text: str = Field(description="The narration script to read aloud")


class StoryResponse(BaseModel):
    """What the legacy one-shot /api/tell endpoint returns to the client."""

    story: Story
    full_narration: str
    audio_url: str
