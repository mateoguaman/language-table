"""vLLM-backed vision-language policy.

The policy calls an OpenAI-compatible vLLM server with a prompt and the
current RGB observation, then returns the generated language command.
"""

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib import error, request

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 1000.0


class VLLMPolicy:
    """Generate language commands from images using a vLLM server.

    Parameters
    ----------
    prompt : str
        Text prompt sent alongside each image.
    url : str
        Address of the vLLM server. May be either the server root
        (``http://host:8000``) or the OpenAI-compatible root
        (``http://host:8000/v1``).
    """

    def __init__(
        self,
        prompt: str,
        url: str,
        max_history_messages: Optional[int] = None,
    ):
        if max_history_messages is not None and max_history_messages < 1:
            raise ValueError("max_history_messages must be at least 1 or None")

        self.prompt = prompt
        self.base_url = _normalize_base_url(url)
        self.model = _discover_model(self.base_url)
        self.max_history_messages = max_history_messages
        self.messages: List[Dict[str, Any]] = []
        logger.info("VLLMPolicy using model %s at %s", self.model, self.base_url)

    def reset(self) -> None:
        """Clear the chat history for a new episode or task."""
        self.messages.clear()

    def step(self, image: np.ndarray) -> str:
        """Return a language command generated from ``image``."""
        image_data_url = _encode_image_data_url(image)
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ],
        }
        messages = self._trimmed_history(self.messages + [user_message])

        payload = {
            "model": self.model,
            "messages": messages,
            # "max_tokens": 32,
            "temperature": 1.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {
                    "enable_thinking": True, 
                },
            }
        }

        response = _post_json(f"{self.base_url}/chat/completions", payload)
        content = _extract_message_content(response)
        self.messages = self._trimmed_history(
            messages + [{"role": "assistant", "content": content}]
        )
        print(f"Content: {content}")
        return _extract_atomic_instruction(content)

    def _trimmed_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.max_history_messages is None:
            return messages
        return messages[-self.max_history_messages:]


def _normalize_base_url(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("vLLM server url must be a non-empty string")

    base_url = url.strip().rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _discover_model(base_url: str) -> str:
    response = _get_json(f"{base_url}/models")
    try:
        models = response["data"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Malformed vLLM models response from {base_url}/models: {response!r}"
        ) from exc

    if not models:
        raise RuntimeError(f"No models are served by vLLM at {base_url}")

    try:
        return models[0]["id"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Malformed first model entry from {base_url}/models: {models[0]!r}"
        ) from exc


def _encode_image_data_url(image: np.ndarray) -> str:
    rgb = _to_uint8_rgb(image)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a NumPy array, got {type(image).__name__}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape (height, width, 3) with RGB channels; "
            f"got {image.shape}"
        )

    if image.dtype == np.uint8:
        return image

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if np.nanmin(image) >= 0.0 and np.nanmax(image) <= 1.0:
            image = image * 255.0
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def _get_json(url: str) -> Dict[str, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    return _send_json_request(req)


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return _send_json_request(req)


def _send_json_request(req: request.Request) -> Dict[str, Any]:
    try:
        with request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM request failed with HTTP {exc.code} for {req.full_url}: {details}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Could not reach vLLM server at {req.full_url}: {exc.reason}"
        ) from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"vLLM returned non-JSON response from {req.full_url}: {body[:500]!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"vLLM returned unexpected JSON from {req.full_url}: {parsed!r}"
        )
    return parsed


def _extract_message_content(response: Dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed vLLM completion response: {response!r}") from exc

    if not isinstance(content, str):
        raise RuntimeError(
            f"Expected vLLM message content to be a string, got {type(content).__name__}"
        )
    return content.strip()


def _extract_atomic_instruction(content: str) -> str:
    match = re.search(
        r"<atomic_instruction>\s*(.*?)\s*</atomic_instruction>",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return content.strip()
