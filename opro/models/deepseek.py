"""DeepSeek backend (deepseek-v4-flash, deepseek-v4-pro).

Uses Anthropic SDK with DeepSeek's Anthropic-compatible base URL.
Keys read from DEEPSEEK_API_KEY env var (comma-separated for key pool).
thinking_level maps to output_config effort when not None (Anthropic format):
    LOW/MEDIUM → high  (DeepSeek maps both to high)
    HIGH       → high
    MAX        → max
    None       → no thinking block
"""
from __future__ import annotations

import asyncio
import os

from opro.models.base import ModelResponse
from opro.models.media import strip_json_fences

_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"

# low/medium → high, high → high, max → max (per DeepSeek docs)
_EFFORT_MAP = {
    "LOW": "high",
    "MEDIUM": "high",
    "HIGH": "high",
    "MAX": "max",
}


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------

class _DeepSeekKeyPool:
    def __init__(self, api_keys: list[str], max_retries: int = 5,
                 base_delay: float = 1.0, max_delay: float = 60.0):
        import anthropic
        self._clients = [
            anthropic.AsyncAnthropic(
                api_key=k,
                base_url=_DEEPSEEK_ANTHROPIC_BASE_URL,
            )
            for k in api_keys
        ]
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
                print(f"[DeepSeekKeyPool] quota hit on key {from_idx}, rotating to {self._idx}")

    async def messages_create(self, **kwargs):
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                return await client.messages.create(**kwargs)
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "rate_limit" in err.lower()
                if is_rate_limit:
                    delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                    await self._rotate(idx)
                else:
                    delay = self._base_delay
                print(f"[DeepSeekKeyPool] attempt {attempt+1}/{self._max_retries+1} "
                      f"failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(delay)


_pool: _DeepSeekKeyPool | None = None


def _get_pool() -> _DeepSeekKeyPool:
    global _pool
    if _pool is None:
        raw = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not raw:
            raise RuntimeError("DEEPSEEK_API_KEY env var not set.")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        _pool = _DeepSeekKeyPool(keys)
    return _pool


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

async def call_deepseek_model(
    prompt: str,
    model_id: str = "deepseek-v4-flash",
    img_input=None,
    json_output: bool = False,
    thinking_level: str | None = None,
) -> ModelResponse:
    pool = _get_pool()

    kwargs: dict = dict(
        model=model_id,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    if thinking_level is not None:
        effort = _EFFORT_MAP.get(thinking_level.upper())
        if effort is None:
            raise ValueError(f"Invalid thinking_level={thinking_level!r}. Use LOW/MEDIUM/HIGH/MAX.")
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
        # output_config not in Anthropic SDK schema; pass via extra_body
        kwargs["extra_body"] = {"output_config": {"effort": effort}}

    resp = await pool.messages_create(**kwargs)

    text = ""
    for block in resp.content:
        if block.type == "text":
            text = block.text
            break

    if json_output and text:
        text = strip_json_fences(text)

    return ModelResponse(text=text, raw=resp)
