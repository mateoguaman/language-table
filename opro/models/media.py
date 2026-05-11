"""Shared media encoding utilities for all model backends."""
from __future__ import annotations

import base64
import io
import os

import numpy as np


def encode_image_bytes(img) -> bytes:
    """Encode an image (ndarray / path / bytes) to raw JPEG bytes."""
    if isinstance(img, np.ndarray):
        import imageio
        buf = io.BytesIO()
        imageio.imwrite(buf, img, format="JPEG", macro_block_size=1)
        return buf.getvalue()
    if isinstance(img, str) and os.path.exists(img):
        with open(img, "rb") as f:
            return f.read()
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    return img


def encode_image_b64(img) -> str:
    """Encode an image to a base64 string."""
    return base64.b64encode(encode_image_bytes(img)).decode()


def encode_video_bytes(video_input) -> bytes:
    """Encode a video (ndarray frames / path / bytes) to raw MP4 bytes."""
    if isinstance(video_input, np.ndarray):
        import imageio
        buf = io.BytesIO()
        imageio.mimwrite(buf, video_input, format="mp4", fps=1, macro_block_size=1)
        return buf.getvalue()
    if isinstance(video_input, str) and os.path.exists(video_input):
        with open(video_input, "rb") as f:
            return f.read()
    if isinstance(video_input, (bytes, bytearray)):
        return bytes(video_input)
    return video_input


def build_openai_content(prompt: str, img_input) -> str | list:
    """Build an OpenAI messages content block (text or vision list)."""
    if img_input is None:
        return prompt
    imgs = img_input if isinstance(img_input, list) else [img_input]
    content: list = []
    for img in imgs:
        b64 = encode_image_b64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})
    return content


def build_anthropic_content(prompt: str, img_input) -> list:
    """Build an Anthropic messages content block."""
    imgs = img_input if isinstance(img_input, list) else ([img_input] if img_input is not None else [])
    content: list = []
    for img in imgs:
        b64 = encode_image_b64(img)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": prompt})
    return content


def strip_json_fences(text: str) -> str:
    """Remove markdown code fences around JSON if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return stripped
