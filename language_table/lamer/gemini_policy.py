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
                thinking_config=types.ThinkingConfig(thinking_level="LOW", include_thoughts=False),
                response_mime_type="application/json",
            ),
        ),
        timeout=timeout,
    )
    return response.text

# async def _language_to_action_seq_async(
#     state: str,
#     action: str,
#     disturbance: Optional[str] = None,
#     model_id: str = "gemini-3.1-flash-lite-preview",
#     max_output_tokens: int = 1024,
#     timeout: float = 30.0,
# ) -> dict:
#     """Translate a natural-language action string into [y, x] action arrays.

#     Returns a dict with keys: true_actions, disturbance, disturbed_actions.
#     """
    
#     prompt = textwrap.dedent("""\
#         You are a robot in a language table environment with the following \
# state described in natural language: "{STATE}".

#         The robot can move in [x,y] directions. Each action moves you \
# a maximum of 0.05 units in x and/or y direction.

#         You're given the following action sequence in natural language: \
# "{ACTION}". Turn this into a sequence of [x,y] actions.

#         Here's the twist: you're an adversarial persona. Apply a deterministic disturbance to the action mapping that leads to task failure.

#         Respond in the following JSON format:
#         {
#             "true_actions": [[x,y], ...],
#             "disturbance": {DISTURBANCE},
#             "disturbed_actions": [[x,y], ...]
#         }
#         """)
#     prompt = prompt.replace("{STATE}", state)
#     prompt = prompt.replace("{ACTION}", action)
#     prompt = prompt.replace("{DISTURBANCE}", disturbance if disturbance else "str")

#     result = await _call_gemini_async(
#         prompt, model_id=model_id,
#         max_output_tokens=max_output_tokens, timeout=timeout,
#     )
#     return json.loads(result)


# ── Axis inversions (deterministic) ────────────────────────────────────────
def apply_y_inversion(actions, seed):
    """Y-axis reversed. Signature: all vertical motion is backwards."""
    return actions * np.array([1.0, -1.0])

def apply_x_inversion(actions, seed):
    """X-axis reversed. Signature: all horizontal motion is backwards."""
    return actions * np.array([-1.0, 1.0])

def apply_full_negation(actions, seed):
    """Both axes reversed. Signature: robot moves directly away from target."""
    return -actions

# ── Axis swaps (deterministic) ─────────────────────────────────────────────
def apply_axis_swap(actions, seed):
    """X and Y swapped. Signature: horizontal commands produce vertical motion."""
    return actions[:, ::-1].copy()

def apply_axis_swap_y_negated(actions, seed):
    """Axes swapped and Y negated. Signature: compound swap + one inverted axis."""
    swapped = actions[:, ::-1].copy()
    return swapped * np.array([1.0, -1.0])

# ── Large stochastic rotations ─────────────────────────────────────────────
def apply_large_rotation(actions, seed):
    """Random rotation in [90°, 270°], seeded. Always a large angular offset."""
    rng = np.random.default_rng(seed)
    # Sample from [π/2, 3π/2] — avoids near-identity rotations
    angle = rng.uniform(np.pi / 2, 3 * np.pi / 2)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return actions @ rot.T

def apply_large_rotation_neg(actions, seed):
    """Random rotation in [-90°, -270°], seeded. Mirror of apply_large_rotation."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-3 * np.pi / 2, -np.pi / 2)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return actions @ rot.T

# ── Random bias ────────────────────────────────────────────────────────────
def apply_large_bias(actions, seed):
    """Adds a large constant drift in a random direction, seeded.
    Magnitude is comparable to the max action (0.04–0.07), so it dominates."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi)
    magnitude = rng.uniform(0.04, 0.07)
    bias = magnitude * np.array([np.cos(angle), np.sin(angle)])
    return actions + bias

PERTURBATION_REGISTRY = {
    "y_inversion":          apply_y_inversion,
    "x_inversion":          apply_x_inversion,
    "full_negation":        apply_full_negation,
    "axis_swap":            apply_axis_swap,
    "axis_swap_y_negated":  apply_axis_swap_y_negated,
    "large_rotation":       apply_large_rotation,
    "large_rotation_neg":   apply_large_rotation_neg,
    "large_bias":           apply_large_bias,
}

# Seeded 80/20 train/val split of perturbation keys (seed=0, fixed).
_rng_split = np.random.default_rng(0)
_all_keys = list(PERTURBATION_REGISTRY.keys())
_shuffled = _rng_split.permutation(_all_keys).tolist()
_n_train = int(round(0.7 * len(_shuffled)))
TRAIN_PERTURBATION_KEYS: List[str] = _shuffled[:_n_train]
VAL_PERTURBATION_KEYS:   List[str] = _shuffled[_n_train:]
del _rng_split, _all_keys, _shuffled, _n_train

async def _language_to_action_seq_async(
    state: str,
    action: str,
    model_id: str = "gemini-3.1-flash-lite-preview",
    timeout: float = 30.0,
) -> dict:
    """Translate a natural-language action string into [x,y] action arrays.

    Returns a dict with keys: true_actions, disturbance, disturbed_actions.
    """
    
    prompt = textwrap.dedent("""\
        You are a robot in a language table environment with the following \
state described in natural language: "{STATE}".

        The robot can move in [x,y] directions. Each action moves you \
a maximum of 0.05 units in x (up/down) and/or y (left/right) direction.

        You're given the following instruction in natural language: \
"{ACTION}".

        Reason about the steps needed and turn the instruction into a sequence of [x,y] actions.

        If the task is impossible, return a zero action: [[0,0]].

        Respond in the following JSON format:
        {
            "true_actions": [[x,y], ...]
        }
        """)
    prompt = prompt.replace("{STATE}", state)
    prompt = prompt.replace("{ACTION}", action)

    result = await _call_gemini_async(
        prompt, model_id=model_id,
        timeout=timeout,
    )
    logger.debug("Gemini raw response: %s", result)
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
    timeout : float
        Per-request timeout in seconds.  Applies to each individual Gemini
        API call (retries each get their own timeout).
    split : str
        ``"train"``, ``"val"``, or ``"none"``.  Restricts perturbation
        sampling to the corresponding 80/20 subset of ``PERTURBATION_REGISTRY``
        (seeded, fixed).  Use ``"none"`` to disable disturbances entirely so
        that ``disturbed_actions`` is identical to ``true_actions``.
        Defaults to ``"train"``.
    """

    def __init__(
        self,
        action_clip: float = 0.1,
        model_id: str = "gemini-3.1-flash-lite-preview",
        timeout: float = 30.0,
        split: str = "train",
    ):
        if split not in ("train", "val", "none"):
            raise ValueError(f"split must be 'train', 'val', or 'none', got {split!r}")
        self.action_clip = action_clip
        self.model_id = model_id
        self.timeout = timeout
        self.split = split
        self.perturbation_keys = (
            TRAIN_PERTURBATION_KEYS if split == "train" else
            VAL_PERTURBATION_KEYS if split == "val" else
            []
        )
        logger.info("GeminiPolicy split=%s, perturbations=%s", split, self.perturbation_keys)

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

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:

                # remove task description from state
                state_text_mod = state_text.split("\n")[1:]
                state_text_mod = "\n".join(state_text_mod)

                raw = await _language_to_action_seq_async(
                    state_text_mod, action_text,
                    model_id=self.model_id,
                    timeout=self.timeout,
                )

                true_actions = np.array(raw["true_actions"], dtype=np.float32)

                logger.info(
                    "Translated %d actions for: %.60s",
                    len(true_actions), action_text,
                )
                logger.debug("raw prediction: %s", raw)
                logger.debug("raw action: %s", action_text)
                logger.debug("raw state: %s", state_text_mod)

                if self.split == "none":
                    return {
                        "true_actions": true_actions.tolist(),
                        "disturbance": "",
                        "disturbed_actions": true_actions.tolist(),
                    }

                if disturbance is None:
                    disturbance_fn = np.random.choice(self.perturbation_keys)
                    seed = np.random.randint(0, 1000000)
                else:
                    disturbance_fn, seed = disturbance.split(":")
                    seed = int(seed)
                disturbance = f"{disturbance_fn}:{seed}"

                disturbed_actions = PERTURBATION_REGISTRY[disturbance_fn](true_actions, seed)
                return {
                    "true_actions": true_actions.tolist(),
                    "disturbance": disturbance,
                    "disturbed_actions": disturbed_actions.tolist(),
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
                logger.warning("Gemini translate failed (attempt %d/%d): %s", attempt, max_retries, e)
                if attempt == max_retries:
                    logger.error("Gemini translate failed after %d attempts, returning empty actions: %s", max_retries, e)
                    return {
                        "true_actions": [],
                        "disturbance": disturbance or "",
                        "disturbed_actions": [],
                    }


