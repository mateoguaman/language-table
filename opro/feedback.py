"""Feedback functions for per-episode and aggregate signals.

feedback()
    Called once per env after its episode ends.
    Receives the full trajectory (frames or state texts depending on modality)
    and the top-level task instruction. Queries the LLM/VLM and returns the
    natural-language response.

agg_feedback()
    Called once per OPRO iteration by OPROLoop.
    Receives all per-env feedback strings. Queries an LLM summarizer and
    returns one aggregated string appended to the optimizer history entry.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import numpy as np


async def feedback(
    frames: list[np.ndarray],
    state_texts: list[str],
    top_instruction: str,
    ll_instructions: list[str] | None = None,
    modality: str = "text",
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> str:
    """Query LLM/VLM for episode feedback and return the response text.

    Args:
        frames: list of RGB uint8 frames across the episode.
                Used when modality == "image".
        state_texts: list of state-text strings across the episode.
                     Used when modality == "text".
        top_instruction: the high-level task goal string.
        ll_instructions: low-level instructions issued by the steerer LLM at
                         each step (aligned with state_texts / frames).
        modality: "text" to use state_texts, "image" to use frames.
        model_id: Gemini model ID to query.
        thinking_level: Gemini thinking level for this call.

    Returns:
        LLM/VLM response text.
    """
    from opro.prompt import render_feedback_text_prompt, render_feedback_image_prompt
    from opro.models import call_model

    if modality == "image":
        prompt = render_feedback_image_prompt(
            top_instruction=top_instruction,
            ll_instructions=ll_instructions,
        )
        img_input = frames if frames else None
    else:
        prompt = render_feedback_text_prompt(
            top_instruction=top_instruction,
            state_texts=state_texts,
            ll_instructions=ll_instructions,
        )
        img_input = None

    resp = await call_model(
        prompt=prompt,
        img_input=img_input,
        thinking_level=thinking_level,
        json_output=False,
        model_id=model_id,
    )
    return resp.text


async def agg_feedback(
    feedbacks: list[str],
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> str:
    """Query LLM to aggregate per-env feedback and return the response text.

    Args:
        feedbacks: list of per-env feedback strings (may be empty strings).
        model_id: Gemini model ID to query.
        thinking_level: Gemini thinking level (same knob as per-env feedback).

    Returns:
        LLM response text summarizing the per-episode feedbacks.
    """
    from opro.prompt import render_agg_feedback_prompt
    from opro.models import call_model

    non_empty = [f for f in feedbacks if f and f.strip()]
    if not non_empty:
        return ""

    prompt = render_agg_feedback_prompt(non_empty)
    resp = await call_model(
        prompt=prompt,
        img_input=None,
        thinking_level=thinking_level,
        json_output=False,
        model_id=model_id,
    )
    return resp.text


# ---------------------------------------------------------------------------
# Type aliases for injection
# ---------------------------------------------------------------------------

FeedbackFn = Callable[
    [list[np.ndarray], list[str], str, "list[str] | None", str],
    "Awaitable[str]",
]
AggFeedbackFn = Callable[[list[str]], "Awaitable[str]"]
