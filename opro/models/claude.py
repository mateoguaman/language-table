"""Anthropic Claude backend (haiku, sonnet, opus).

Keys read from ANTHROPIC_API_KEY env var (comma-separated for key pool).
thinking / extended thinking (Messages API per anthropic SDK types):

- thinking_level is None: thinking param omitted entirely.
- thinking_level is a string: passed directly as ``thinking={"type": thinking_level}``.
  e.g. "adaptive" for sonnet+, "enabled" for haiku (no budget needed).
"""
from __future__ import annotations

import asyncio
import os

from opro.models.base import ModelResponse
from opro.models.media import build_anthropic_content, strip_json_fences


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------

class _ClaudeKeyPool:
    def __init__(self, api_keys: list[str], max_retries: int = 5,
                 base_delay: float = 1.0, max_delay: float = 60.0):
        import anthropic
        self._clients = [anthropic.AsyncAnthropic(api_key=k) for k in api_keys]
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
                print(f"[ClaudeKeyPool] quota hit on key {from_idx}, rotating to {self._idx}")

    async def messages_create(self, **kwargs):
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                return await client.messages.create(**kwargs)
            except Exception as e:
                err = str(e)
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[ClaudeKeyPool] attempt {attempt+1}/{self._max_retries+1} "
                      f"failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if "429" in err or "rate_limit" in err.lower() or "overloaded" in err.lower():
                    await self._rotate(idx)
                await asyncio.sleep(delay)


_pool: _ClaudeKeyPool | None = None


def _get_pool() -> _ClaudeKeyPool:
    global _pool
    if _pool is None:
        raw = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not raw:
            raise RuntimeError("ANTHROPIC_API_KEY env var not set.")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        _pool = _ClaudeKeyPool(keys)
    return _pool


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

async def call_claude_model(
    prompt: str,
    model_id: str = "claude-opus-4-5",
    img_input=None,
    thinking_level: str | None = "adaptive",
    json_output: bool = False,
) -> ModelResponse:
    pool = _get_pool()
    content = build_anthropic_content(prompt, img_input)

    kwargs: dict = dict(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
    )
    if thinking_level is not None:
        thinking_param: dict = {"type": thinking_level}
        if thinking_level == "enabled":
            thinking_param["budget_tokens"] = 1024
        kwargs["thinking"] = thinking_param

    resp = await pool.messages_create(**kwargs)

    text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    text = "\n".join(text_parts)

    if json_output and text:
        text = strip_json_fences(text)

    return ModelResponse(text=text, raw=resp)
