"""OpenAI GPT backend (gpt-5.4, gpt-5.4-mini, gpt-5.5, etc.).

Keys read from OPENAI_API_KEY env var (comma-separated for key pool).
thinking_level is silently ignored — gpt-5.x uses standard completions.
Image input encoded as base64 data-URL.
"""
from __future__ import annotations

import asyncio
import os

from opro.models.base import ModelResponse
from opro.models.media import build_openai_content


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------

class _GPTKeyPool:
    def __init__(self, api_keys: list[str], max_retries: int = 5,
                 base_delay: float = 1.0, max_delay: float = 60.0):
        import openai
        self._clients = [openai.AsyncOpenAI(api_key=k) for k in api_keys]
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
                print(f"[GPTKeyPool] quota hit on key {from_idx}, rotating to {self._idx}")

    async def chat_completions_create(self, **kwargs):
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                return await client.chat.completions.create(**kwargs)
            except Exception as e:
                err = str(e)
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[GPTKeyPool] attempt {attempt+1}/{self._max_retries+1} "
                      f"failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if "429" in err or "rate_limit" in err.lower():
                    await self._rotate(idx)
                await asyncio.sleep(delay)


_pool: _GPTKeyPool | None = None


def _get_pool() -> _GPTKeyPool:
    global _pool
    if _pool is None:
        raw = os.environ.get("OPENAI_API_KEY", "").strip()
        if not raw:
            raise RuntimeError("OPENAI_API_KEY env var not set.")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        _pool = _GPTKeyPool(keys)
    return _pool


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

async def call_gpt_model(
    prompt: str,
    model_id: str = "gpt-5.5",
    img_input=None,
    json_output: bool = False,
) -> ModelResponse:
    pool = _get_pool()
    content = build_openai_content(prompt, img_input)

    kwargs: dict = dict(
        model=model_id,
        messages=[{"role": "user", "content": content}],
    )
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await pool.chat_completions_create(**kwargs)
    text = resp.choices[0].message.content or ""
    return ModelResponse(text=text, raw=resp)
