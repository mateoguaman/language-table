"""Self-contained env utilities copied from run_langtable_gemini.py.

Includes: RPC helpers, LLM steerer batch call, state-to-text conversion,
video writing, and JSON response parsing.
"""

from __future__ import annotations

import asyncio
import json
import re
import textwrap
import time
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np

from opro.models import call_model
from language_table.lamer.protocol import EnvRequest, send_message, recv_message
from language_table.environments.workspace_xy import normalize_workspace_xy

# ---------------------------------------------------------------------------
# RPC
# ---------------------------------------------------------------------------

MAX_LLM_RETRIES = 5
MAX_INSTRUCTION_CHARS = 256


def rpc(sock, method: str, *args, **kwargs):
    req = EnvRequest(
        request_id=str(time.monotonic()),
        method=method,
        args=args,
        kwargs=kwargs,
    )
    send_message(sock, req)
    resp = recv_message(sock)
    if resp.status == "error":
        raise RuntimeError(f"{method} failed:\n{resp.error_message}")
    return resp.result


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict | None:
    """Extract and validate JSON with 'history' and 'instruction' keys."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    instruction = data.get("instruction", "")
    if not instruction or len(instruction) > MAX_INSTRUCTION_CHARS:
        return None
    return data


# ---------------------------------------------------------------------------
# State text
# ---------------------------------------------------------------------------

def state_to_text(state_obs: dict) -> str:
    """Build a textual description of block positions from a state dict."""
    lines: list[str] = []
    for key in sorted(state_obs.keys()):
        m = re.match(r"^block_(.+)_translation$", key)
        if not m:
            continue
        name = m.group(1)
        mask = state_obs.get(f"block_{name}_mask")
        if mask is None or float(np.asarray(mask).reshape(-1)[0]) < 0.5:
            continue
        xy = np.asarray(state_obs[key], dtype=np.float64).reshape(-1)
        if xy.size < 2:
            continue
        xn, yn = normalize_workspace_xy(float(xy[0]), float(xy[1]))
        lines.append(f"  {name}: x={xn:.3f}, y={yn:.3f}")
    if not lines:
        return "(no blocks on table)"
    return "Block positions (normalized [0,1]):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Single LLM call with retry
# ---------------------------------------------------------------------------

async def call_one_with_retry(
    prompt: str,
    img: Optional[np.ndarray],
    model_id: str,
    fallback: str,
    thinking_level: str = "MEDIUM",
) -> dict:
    """Call the steerer VLM; returns dict with at least 'instruction' key."""
    for attempt in range(MAX_LLM_RETRIES):
        try:
            resp = await call_model(
                prompt=prompt,
                img_input=img,
                thinking_level=thinking_level,
                media_resolution="MEDIA_RESOLUTION_LOW",
                json_output=True,
                model_id=model_id,
            )
        except Exception as e:
            print(f"[warn] gemini call error (try {attempt+1}/{MAX_LLM_RETRIES}): {e}")
            await asyncio.sleep(1)
            continue

        data = parse_json_response(resp.text)
        if data is not None:
            return data
        print(f"[warn] parse failed (try {attempt+1}/{MAX_LLM_RETRIES})")

    print(f"[warn] retries exhausted, using fallback: {fallback!r}")
    return {"plan": "(fallback)", "instruction": fallback}


# ---------------------------------------------------------------------------
# Batched LLM call across all active envs
# ---------------------------------------------------------------------------

async def llm_batch(
    meta_prompt_rendered_list: list[str],
    images: list[Optional[np.ndarray]],
    model_id: str,
    fallback: str,
    thinking_level: str = "MEDIUM",
) -> list[dict]:
    """Run one steerer VLM call per env concurrently.

    Args:
        meta_prompt_rendered_list: fully rendered prompt per env (from MetaPrompt.render()).
        images: RGB frame per env; None if modality=="text".
        model_id: Gemini model ID.
        fallback: instruction to use if all retries fail.
        thinking_level: Gemini thinking level for steerer calls.

    Returns:
        List of response dicts, one per env.
    """
    return await asyncio.gather(*[
        call_one_with_retry(prompt, img, model_id, fallback, thinking_level=thinking_level)
        for prompt, img in zip(meta_prompt_rendered_list, images)
    ])


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def to_uint8_rgb(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0:
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _wrap_lines(text: str, width_chars: int, max_lines: int) -> list[str]:
    if not (text or "").strip():
        return [""]
    lines = textwrap.wrap(text, width=width_chars, break_long_words=True, break_on_hyphens=False)
    return lines[:max_lines] if lines else [""]


def overlay_instructions_rgb(
    rgb: np.ndarray,
    high_level: str,
    low_level: str,
) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(np.clip(w / 720.0, 0.35, 0.85))
    thickness = max(1, int(round(scale * 2)))
    margin = max(12, int(18 * scale))
    line_skip = int(22 * scale + 6)
    inner_w = max(0, w - 2 * margin)
    px_per_char = max(8.0, 12.5 * scale)
    width_chars = max(16, int(inner_w / px_per_char))

    top_lines = _wrap_lines(high_level, width_chars, max_lines=5)
    bot_lines = _wrap_lines(low_level, width_chars, max_lines=5)

    def _stroke_line(xy, line, color):
        x, y = xy
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(bgr, line, (x + dx, y + dy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(bgr, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    y = margin + int(18 * scale)
    for line in top_lines:
        if line:
            _stroke_line((margin, y), line, (255, 255, 255))
        y += line_skip

    y = h - margin
    for line in reversed(bot_lines):
        if line:
            _stroke_line((margin, y), line, (220, 245, 255))
        y -= line_skip

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def write_env_videos(log_dir: str, frames_per_env: list[list], fps: float, suffix: str = "") -> None:
    import os

    # Pip: install `imageio[ffmpeg]` so imageio can write MP4 (pulls in imageio-ffmpeg for the plugin).
    for i, frames in enumerate(frames_per_env):
        if not frames:
            continue
        fname = f"env_{i:03d}{suffix}.mp4"
        path = os.path.join(log_dir, fname)
        h, w = frames[0].shape[:2]
        rgb_frames: list[np.ndarray] = []
        for f in frames:
            arr = np.asarray(f)
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[:2] != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
            rgb_frames.append(arr)
        # libx264 / yuv420p require even width and height (VS Code / Chromium playback).
        fh, fw = rgb_frames[0].shape[:2]
        ew, eh = fw - (fw % 2), fh - (fh % 2)
        if ew < 2 or eh < 2:
            ew, eh = max(2, ew), max(2, eh)
        if (ew, eh) != (fw, fh):
            rgb_frames = [
                cv2.resize(fr, (ew, eh), interpolation=cv2.INTER_AREA) for fr in rgb_frames
            ]
        imageio.mimwrite(
            path,
            rgb_frames,
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        print(f"[video] wrote {path} ({len(frames)} frames @ {fps} fps)")
