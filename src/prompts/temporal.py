"""Temporal grounding prompts."""

from src.prompts.types import PromptSpec


def build_temporal_prompt(
    model_family: str,
    query_text: str,
    text_only: bool = False,
    selected_frame_ids: list[int] | None = None,
) -> PromptSpec:
    """Build a temporal grounding prompt specification."""
    query = query_text.strip()

    if model_family == "gemini":
        return PromptSpec(
            # Referenced vidi2.5 (Vidi2.5: Large Multimodal Models for Video Understanding and Creation)
            text=(
                "Answer with time ranges and do not output explanation. "
                f"What is the single time range corresponding to the text query: \"{query}\"? "
                "Output format:"
                "[start, end]"
            ),
            output_format="time_ranges",
            time_unit="seconds",
            needs_timestamp_hint=False,
            needs_duration_hint=False,
        )

    if model_family == "gpt":
        return PromptSpec(
            # Referenced vidi2.5 (Vidi2.5: Large Multimodal Models for Video Understanding and Creation)
            text=(
                "The input images are frames from a video. Output the frame indexes that "
                f"correspond to the text query: \"{query}\". Only output the index range, "
                "for example, 2-4, 6-8."
            ),
            output_format="frame_ranges",
            time_unit="frame_index",
            needs_frame_ids=True,
        )

    if model_family == "internvl":
        return PromptSpec(
            text=(
                f"Give you a textual query: {query}\n"
                "When does the described content occur in the video?\n"
                "Please return the timestamp in seconds. "
                "Output format:"
                "[start, end]"
            ),
            output_format="timestamps_seconds",
            time_unit="seconds",
            needs_timestamp_hint=True,
            needs_duration_hint=True,
        )

    if model_family == "qwen":
        return PromptSpec(
            # Referenced Qwen3 (Qwen3 Technical Report)
            text=(
                f"Give you a textual query: {query}\n"
                "When does the described content occur in the video?\n"
                "Please return the timestamp in seconds. "
                "Output format:"
                "[start, end]"
            ),
            output_format="timestamps_seconds",
            time_unit="seconds",
            needs_timestamp_hint=True,
            needs_duration_hint=True,
        )

    if model_family == "eagle":
        return PromptSpec(
            text=(
                f"Give you a textual query: {query}\n"
                "When does the described content occur in the video?\n"
                "Please return the timestamp in seconds. "
                "Output format:"
                "[start, end]"
            ),
            output_format="timestamps_seconds",
            time_unit="seconds",
            needs_timestamp_hint=True,
            needs_duration_hint=True,
        )

    if model_family == "llava_st":
        # official example: "Give you a textual query: 'person takes a laptop from the shelf'. When does the described content occur in the video? Please return the start and end timestamps."
        return PromptSpec(
            text=(
                f"Give you a textual query: '{query}'. "
                "When does the described content occur in the video? "
                "Please return the start and end timestamps."
            ),
            output_format="time_ranges",
            time_unit="sampled_frame_index",
            needs_timestamp_hint=True,
            needs_duration_hint=True,
        )


    raise ValueError("Invalid model inputs")
