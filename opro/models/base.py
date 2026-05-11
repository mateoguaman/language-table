"""Unified model response type and call_model router."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Load secrets into env vars before any provider is initialised.
try:
    import opro.models.secret  # noqa: F401
except ImportError:
    pass


@dataclass
class ModelResponse:
    text: str
    raw: Any = None


def _provider(model_id: str) -> str:
    m = model_id.lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "gpt"
    if m.startswith("claude"):
        return "claude"
    if m.startswith("deepseek"):
        return "deepseek"
    raise ValueError(f"Cannot infer provider from model_id={model_id!r}. "
                     "Prefix with gemini/gpt/claude/deepseek.")


async def call_model(
    prompt: str,
    model_id: str,
    img_input=None,
    video_input=None,
    thinking_level: str | None = None,
    json_output: bool = False,
    response_schema=None,
    media_resolution: str = "MEDIA_RESOLUTION_HIGH",
) -> ModelResponse:
    """Unified async model call.

    Args:
        prompt: Text prompt.
        model_id: Model identifier string (e.g. 'gemini-3-flash-preview',
                  'gpt-5.5', 'claude-opus-4-5').
        img_input: Image(s): np.ndarray, file path, bytes, or list thereof.
        video_input: Video: np.ndarray (frames), file path, or bytes.
        thinking_level: 'LOW' | 'MEDIUM' | 'HIGH' | None.
                        Gemini: native thinking level.
                        Claude: None → adaptive thinking; LOW/MEDIUM/HIGH →
                        extended thinking with budget_tokens 1024/8192/32000.
                        GPT: ignored (no thinking support on gpt-5.x).
        json_output: Request JSON-formatted output.
        response_schema: Optional schema for structured JSON (Gemini only).
        media_resolution: Gemini-specific media resolution hint.

    Returns:
        ModelResponse with .text containing the response string.
    """
    provider = _provider(model_id)

    if provider == "gemini":
        from opro.models.gemini import call_gemini_model
        return await call_gemini_model(
            prompt=prompt,
            model_id=model_id,
            img_input=img_input,
            video_input=video_input,
            thinking_level=thinking_level or "MEDIUM",
            json_output=json_output,
            response_schema=response_schema,
            media_resolution=media_resolution,
        )

    if provider == "gpt":
        from opro.models.gpt import call_gpt_model
        return await call_gpt_model(
            prompt=prompt,
            model_id=model_id,
            img_input=img_input,
            json_output=json_output,
        )

    if provider == "claude":
        from opro.models.claude import call_claude_model
        return await call_claude_model(
            prompt=prompt,
            model_id=model_id,
            img_input=img_input,
            thinking_level=thinking_level,
            json_output=json_output,
        )

    if provider == "deepseek":
        from opro.models.deepseek import call_deepseek_model
        return await call_deepseek_model(
            prompt=prompt,
            model_id=model_id,
            img_input=img_input,
            json_output=json_output,
            thinking_level=thinking_level,
        )

    raise ValueError(f"Unknown provider for model_id={model_id!r}")
