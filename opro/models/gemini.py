"""Gemini backend using google-genai SDK.

API key read from GOOGLE_API_KEY env var (required).
Optional GOOGLE_API_KEY_PREVIEW for preview/robotics models.
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
from google import genai
from google.genai import types

from opro.models.base import ModelResponse
from opro.models.media import encode_image_bytes, encode_video_bytes

# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------

_PREVIEW_MODEL_SUBSTRINGS = (
    "gemini-robotics-er-early-access",
    "gemini-robotics-er-1.5-preview",
    "gemini-robotics-er-1.6-preview",
)


class _GeminiKeyPool:
    def __init__(self, api_keys: list[str], max_retries: int = 5,
                 base_delay: float = 1.0, max_delay: float = 60.0):
        self._clients = [genai.Client(api_key=k) for k in api_keys]
        self._idx = 0
        self._lock = asyncio.Lock()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    async def _claim_next(self) -> tuple:
        async with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self._clients)
        return idx, self._clients[idx]

    async def _rotate(self, from_idx: int) -> None:
        async with self._lock:
            if self._idx == (from_idx + 1) % len(self._clients):
                self._idx = (self._idx + 1) % len(self._clients)
                print(f"[GeminiKeyPool] quota hit on key {from_idx}, rotating to {self._idx}")

    async def generate_content(self, **kwargs):
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                await asyncio.sleep(np.random.uniform(1, 3))
                return await client.aio.models.generate_content(**kwargs)
            except Exception as e:
                err = str(e)
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[GeminiKeyPool] attempt {attempt+1}/{self._max_retries+1} "
                      f"failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    await self._rotate(idx)
                await asyncio.sleep(delay)


def _load_pool(env_var: str) -> _GeminiKeyPool | None:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return _GeminiKeyPool(keys) if keys else None


_pool: _GeminiKeyPool | None = None
_pool_preview: _GeminiKeyPool | None = None


def _get_pool(preview: bool) -> _GeminiKeyPool:
    global _pool, _pool_preview
    if preview:
        if _pool_preview is None:
            _pool_preview = _load_pool("GOOGLE_API_KEY_PREVIEW")
            if _pool_preview is None:
                raise RuntimeError("GOOGLE_API_KEY_PREVIEW env var not set for preview models.")
        return _pool_preview
    else:
        if _pool is None:
            _pool = _load_pool("GOOGLE_API_KEY")
            if _pool is None:
                raise RuntimeError("GOOGLE_API_KEY env var not set.")
        return _pool


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

async def call_gemini_model(
    prompt: str,
    model_id: str = "gemini-3-flash-preview",
    img_input=None,
    video_input=None,
    thinking_level: str = "MEDIUM",
    json_output: bool = False,
    response_schema=None,
    media_resolution: str = "MEDIA_RESOLUTION_HIGH",
    include_thoughts: bool = False,
) -> ModelResponse:
    parts = []

    if video_input is not None:
        parts.append(types.Part.from_bytes(
            mime_type="video/mp4", data=encode_video_bytes(video_input)))

    if img_input is not None:
        for img in (img_input if isinstance(img_input, list) else [img_input]):
            parts.append(types.Part.from_bytes(
                mime_type="image/jpeg", data=encode_image_bytes(img)))

    parts.append(types.Part.from_text(text=prompt))

    json_kwargs: dict = (
        {"response_mime_type": "application/json",
         **({"response_schema": response_schema} if response_schema else {})}
        if json_output else {}
    )

    is_preview = any(s in model_id for s in _PREVIEW_MODEL_SUBSTRINGS)

    if not (is_preview or "gemini-3" in model_id):
        raise ValueError(f"Unsupported Gemini model_id: {model_id!r}")

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        ),
        media_resolution=media_resolution,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **json_kwargs,
    )

    pool = _get_pool(is_preview)
    raw = await pool.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )
    return ModelResponse(text=getattr(raw, "text", "") or "", raw=raw)
