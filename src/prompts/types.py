"""Shared prompt data structures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """Store prompt text and expected output metadata."""

    text: str
    output_format: str
    time_unit: str | None = None
    box_format: str | None = None
    needs_timestamp_hint: bool = False
    needs_duration_hint: bool = False
    needs_frame_ids: bool = False
