"""Scoring functions for inner-loop evaluation.

All scoring functions are called at EVERY step, uniformly. The evaluator
does not special-case reward type; the ScoringFn implementation decides
when and how to compute the reward.

ScoringFn protocol:
    async def __call__(step, done, state, frame, state_text, top_instruction) -> float

Two implementations:
    GeometricScoring  – calls tetromino_t_reward_from_state() each step.
    VLMScoring        – returns 0.0 mid-episode; fires VLM verifier at done=True.
                        Respects obs_modality: image uses RGB frame,
                        text uses a text-only verifier prompt.
"""

from __future__ import annotations

import asyncio
import json
import re
import textwrap
from typing import Literal, Optional, Protocol

import numpy as np

from opro.geometric_reward import tetromino_t_reward_from_state

VALID_TETROMINOES = ["I", "O", "T", "L", "J", "S", "Z"]

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ScoringFn(Protocol):
    async def __call__(
        self,
        step: int,
        done: bool,
        state: dict,
        frame: Optional[np.ndarray],
        state_text: str,
        top_instruction: str,
    ) -> float: ...


# ---------------------------------------------------------------------------
# GeometricScoring
# ---------------------------------------------------------------------------

class GeometricScoring:
    """Returns tetromino geometric overlap score at every step."""

    async def __call__(
        self,
        step: int,
        done: bool,
        state: dict,
        frame: Optional[np.ndarray],
        state_text: str,
        top_instruction: str,
    ) -> float:
        if not state:
            return 0.0
        return float(tetromino_t_reward_from_state(state))


# ---------------------------------------------------------------------------
# VLM text verifier prompt
# ---------------------------------------------------------------------------

_VLM_TEXT_VERIFIER_PROMPT = textwrap.dedent("""\
    <role>
    You are a geometry expert specialized in polyominoes.
    </role>

    <task>
    Given the block positions below, assess how well the blocks are arranged
    to form the one-sided tetromino '{letter}' when connected with edges.
    </task>

    <rules>
    - Account for 0, 90, 180, or 270 degrees rotation of the shape.
    - If the shape does not match '{letter}', return the letter it resembles instead.
    </rules>

    <definitions>
    Definitions of One-Sided Tetrominoes:
    I: 4 blocks in a straight line.
    O: 2x2 square.
    T: A 3-block row with 1 block centered above/below it.
    L: A 3-block column with 1 block attached to the bottom-right.
    J: A 3-block column with 1 block attached to the bottom-left.
    S: Two horizontal blocks with two more shifted right on the row above.
    Z: Two horizontal blocks with two more shifted left on the row above.
    </definitions>

    Block positions (normalized [0, 1]):
    {state_text}

    Respond in the following JSON format:
    {{
        "looks_like": <one of "I", "O", "T", "L", "J", "S", "Z">
    }}
""")

_VLM_IMAGE_VERIFIER_PROMPT = textwrap.dedent("""\
    <role>
    You are a geometry expert specialized in polyominoes.
    </role>

    <task>
    Look at the top-down RGB image of a table with colored blocks. List all the blocks in the image.
    Assess how well the blocks are arranged to form the one-sided tetromino '{letter}' when connected with edges.
    </task>

    <rules>
    - Account for 0, 90, 180, or 270 degrees rotation of the shape.
    - If the shape does not match '{letter}', return the letter it resembles instead.
    </rules>

    <definitions>
    Definitions of One-Sided Tetrominoes:
    I: 4 blocks in a straight line.
    O: 2x2 square.
    T: A 3-block row with 1 block centered above/below it.
    L: A 3-block column with 1 block attached to the bottom-right.
    J: A 3-block column with 1 block attached to the bottom-left.
    S: Two horizontal blocks with two more shifted right on the row above.
    Z: Two horizontal blocks with two more shifted left on the row above.
    </definitions>

    Respond in the following JSON format:
    {{
        "looks_like": <one of "I", "O", "T", "L", "J", "S", "Z">
    }}
""")


def _extract_letter(instruction: str) -> Optional[str]:
    m = re.search(r'\b([' + ''.join(VALID_TETROMINOES) + r'])\b', instruction)
    return m.group(1) if m else None


async def _call_vlm_verifier(
    letter: str,
    frame: Optional[np.ndarray],
    state_text: str,
    model_id: str,
    obs_modality: Literal["image", "text"],
    timeout: float = 60.0,
    max_retries: int = 5,
) -> float:
    """Call the VLM verifier; return 100.0 on match, 0.0 otherwise."""
    from language_table.lamer.gemini_policy import _call_gemini_async

    if obs_modality == "image":
        prompt = _VLM_IMAGE_VERIFIER_PROMPT.replace("{letter}", letter)
        img_arg = frame
    else:
        prompt = _VLM_TEXT_VERIFIER_PROMPT.replace("{letter}", letter).replace("{state_text}", state_text)
        img_arg = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = await asyncio.wait_for(
                _call_gemini_async(
                    prompt, model_id=model_id, timeout=timeout,
                    thinking_level="MEDIUM", image=img_arg,
                ),
                timeout=timeout,
            )
            data = json.loads(raw)
            looks_like = data.get("looks_like", "")
            return 100.0 if looks_like == letter else 0.0
        except asyncio.TimeoutError:
            print(f"[VLMScoring] timeout (attempt {attempt}/{max_retries})")
        except Exception as e:
            print(f"[VLMScoring] error (attempt {attempt}/{max_retries}): {e}")
    return 0.0


# ---------------------------------------------------------------------------
# VLMScoring
# ---------------------------------------------------------------------------

class VLMScoring:
    """Returns 0.0 every step except terminal; fires VLM verifier at done=True."""

    def __init__(
        self,
        model_id: str,
        obs_modality: Literal["image", "text"] = "text",
        verifier_timeout: float = 60.0,
        verifier_max_retries: int = 5,
    ) -> None:
        self.model_id = model_id
        self.obs_modality = obs_modality
        self.verifier_timeout = verifier_timeout
        self.verifier_max_retries = verifier_max_retries

    async def __call__(
        self,
        step: int,
        done: bool,
        state: dict,
        frame: Optional[np.ndarray],
        state_text: str,
        top_instruction: str,
    ) -> float:
        if not done:
            return 0.0

        letter = _extract_letter(top_instruction)
        if not letter:
            print(f"[VLMScoring] could not extract tetromino letter from: {top_instruction!r}")
            return 0.0

        return await _call_vlm_verifier(
            letter=letter,
            frame=frame,
            state_text=state_text,
            model_id=self.model_id,
            obs_modality=self.obs_modality,
            timeout=self.verifier_timeout,
            max_retries=self.verifier_max_retries,
        )
