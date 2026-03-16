"""
Gemini API policy for language-table environments.

Translates natural-language action strings (from the outer-loop LLM) into
sequences of (2,) low-level actions via the Gemini API.

Usage:
    policy = GeminiPolicy()
    result = policy.translate(state_text, action_text, disturbance=None)
    # result["disturbed_actions"]  -> List[np.ndarray]  (execute these)
    # result["true_actions"]       -> List[np.ndarray]  (for logging)
    # result["disturbance"]        -> str               (reuse on next attempt)
"""

import asyncio
import json
import logging
import os
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types

async def _call_gemini_async(
    prompt: str,
    model_id: str = "gemini-3.1-flash-lite-preview",
    max_output_tokens: int = 1024,
    timeout: float = 30.0,
) -> str:
    """Call the Gemini API and return the raw response text.

    Raises ``asyncio.TimeoutError`` if the call exceeds *timeout* seconds.
    """

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY env var is not set. "
            "Set it in .env.language_table.secrets or export it."
        )

    client = genai.Client(api_key=api_key)
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                response_mime_type="application/json",
                max_output_tokens=max_output_tokens,
            ),
        ),
        timeout=timeout,
    )
    return response.text

async def _language_to_action_seq_async(
    state: str,
    action: str,
    disturbance: Optional[str] = None,
    model_id: str = "gemini-3.1-flash-lite-preview",
    max_output_tokens: int = 1024,
    timeout: float = 30.0,
) -> dict:
    """Translate a natural-language action string into [y, x] action arrays.

    Returns a dict with keys: true_actions, disturbance, disturbed_actions.
    """
    
    prompt = textwrap.dedent("""\
        You are a robot in a language table environment with the following \
state described in natural language: "{STATE}".

        The robot can move in [x,y] directions. Each action moves you \
a maximum of 0.05 units in x and/or y direction.

        You're given the following action sequence in natural language: \
"{ACTION}". Turn this into a sequence of [x,y] actions.

        Here's the twist: you're an adversarial persona. Apply a deterministic disturbance to the action mapping that leads to task failure.

        Respond in the following JSON format:
        {
            "true_actions": [[x,y], ...],
            "disturbance": {DISTURBANCE},
            "disturbed_actions": [[x,y], ...]
        }
        """)
    prompt = prompt.replace("{STATE}", state)
    prompt = prompt.replace("{ACTION}", action)
    prompt = prompt.replace("{DISTURBANCE}", disturbance if disturbance else "str")

    result = await _call_gemini_async(
        prompt, model_id=model_id,
        max_output_tokens=max_output_tokens, timeout=timeout,
    )
    return json.loads(result)


class GeminiPolicy:
    """API-based policy that calls Gemini to translate action strings.

    Unlike the VLA (LAVAPolicy), this policy is called once per LLM turn
    to translate the full action string into a variable-length sequence of
    low-level actions.  No observation feedback is used within the sequence.

    Parameters
    ----------
    action_clip : float
        Clip each action component to [-action_clip, action_clip].
    model_id : str
        Gemini model identifier.
    max_output_tokens : int
        Maximum tokens Gemini may generate per request.
    timeout : float
        Per-request timeout in seconds.  Applies to each individual Gemini
        API call (retries each get their own timeout).
    """

    def __init__(
        self,
        action_clip: float = 0.1,
        model_id: str = "gemini-3.1-flash-lite-preview",
        max_output_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self.action_clip = action_clip
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    async def translate_async(
        self,
        state_text: str,
        action_text: str,
        disturbance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate a natural-language action string into low-level actions.

        Parameters
        ----------
        state_text : str
            Current env state from ``state_to_text()``.
        action_text : str
            The LLM's action string, e.g. "move left by 0.2, then up by 0.1".
        disturbance : str or None
            If None (first attempt), Gemini generates a creative disturbance.
            If provided (subsequent attempts), Gemini reuses the same one.

        Returns
        -------
        dict with keys:
            ``true_actions``      : list[np.ndarray] -- undisturbed (2,) actions
            ``disturbance``       : str -- disturbance description
            ``disturbed_actions`` : list[np.ndarray] -- actions to execute
        """
        if not action_text.strip():
            logger.warning("Empty action_text, returning empty action list")
            return {
                "true_actions": [],
                "disturbance": disturbance or "",
                "disturbed_actions": [],
            }

        def _parse_actions(raw_list: list) -> List[np.ndarray]:
            actions = []
            for pair in raw_list:
                arr = np.asarray(pair, dtype=np.float32)
                if arr.shape != (2,):
                    logger.warning("Skipping malformed action: %s", pair)
                    continue
                arr = np.clip(arr, -self.action_clip, self.action_clip)
                if not np.isfinite(arr).all():
                    logger.error("Non-finite action from Gemini: %s", pair)
                    continue
                actions.append(arr)
            return actions

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                raw = await _language_to_action_seq_async(
                    state_text, action_text, disturbance,
                    model_id=self.model_id,
                    max_output_tokens=self.max_output_tokens,
                    timeout=self.timeout,
                )
                return {
                    "true_actions": _parse_actions(raw["true_actions"]),
                    "disturbance": raw["disturbance"],
                    "disturbed_actions": _parse_actions(raw["disturbed_actions"]),
                }
            except asyncio.TimeoutError:
                logger.warning(
                    "Gemini request timed out after %.1fs (attempt %d/%d)",
                    self.timeout, attempt, max_retries,
                )
                if attempt == max_retries:
                    logger.error("Gemini translate timed out after %d attempts", max_retries)
                    return {
                        "true_actions": [],
                        "disturbance": disturbance or "",
                        "disturbed_actions": [],
                    }
            except Exception as e:
                if attempt == max_retries:
                    logger.error("Gemini translate failed after %d attempts: %s", max_retries, e)
                    raise
                logger.warning("Gemini translate failed (attempt %d/%d): %s. Retrying...", attempt, max_retries, e)


