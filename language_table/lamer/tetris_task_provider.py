"""
Tetris (tetromino) task provider for ``reward_type=custom``.

On reset, each env samples a one-sided tetromino letter uniformly from the
``shapes`` kwarg (lets callers control train vs. validation shape splits) and
gets a natural-language instruction. At the end of each outer turn, the
provider passes the current text observation to Gemini and asks it to score
how well the block layout matches the target letter.

Rewards are computed concurrently via ``asyncio.gather`` in
``precompute_rewards`` (called once per turn by the LAVA env manager); the
per-env ``reward()`` method just reads from the cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import textwrap
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel

from .custom_task_provider import TaskProvider
from .gemini_policy import _call_gemini_async

logger = logging.getLogger(__name__)


VALID_TETROMINOES: List[str] = ["I", "O", "T", "L", "J", "S", "Z"]


class CharMatch(BaseModel):
    match_score: int
    looks_like_instead: Literal["I", "O", "T", "L", "J", "S", "Z"]


# _PROMPT_TEMPLATE = textwrap.dedent("""
#     <role>
#     You are a geometry expert specialized in polyominoes.
#     </role>

#     <task>
#     Given the following scene description:
#     {{text_obs}}
#     Assess how well the blocks are arranged to form the following one-sided tetromino: {{letter}}
#     </task>

#     <rules>
#     - Consider all 4 rotations (0, 90, 180, 270) of the shape.
#     - Return a match score between 0 and 100.
#     - If the shape does not match {{letter}} but resembles one of the other letters, return the letter it resembles.
#     </rules>

#     <definitions>
#     Definitions of One-Sided Tetrominoes:
#     I: 4 blocks in a straight line.
#     O: 2x2 square.
#     T: A 3-block row with 1 block centered above/below it.
#     L: A 3-block column with 1 block attached to the bottom-right.
#     J: A 3-block column with 1 block attached to the bottom-left.
#     S: Two horizontal blocks with two more shifted right on the row above.
#     Z: Two horizontal blocks with two more shifted left on the row above.
#     </definitions>

#     Respond in the following JSON format:
#     {
#         "match_score": <int between 0 and 100>,
#         "looks_like_instead": <one of "I", "O", "T", "L", "J", "S", "Z">
#     }
# """)

_PROMPT_TEMPLATE = textwrap.dedent("""
    <role>
    You are a geometry expert specialized in polyominoes.
    </role>

    <task>
    Given the following scene description:
    {{text_obs}}
    Assess how well the blocks are arranged to form the following one-sided tetromino: {{letter}}
    </task>

    <rules>
    - Return a match score between 0 (no match) and 100 (perfect match).
    - If the shape does not match {{letter}} but resembles one of the other letters, return the letter it resembles.
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
    {
        "match_score": <int between 0 (no match) and 100 (perfect match)>,
        "looks_like_instead": <one of "I", "O", "T", "L", "J", "S", "Z">
    }
""")


def _build_prompt(letter: str, text_obs: str) -> str:

    # remove end-effector information - might work w/o but this way is cleaner
    text_obs_clean = text_obs.split("Blocks:")[-1]
    prompt = _PROMPT_TEMPLATE
    prompt = prompt.replace("{{letter}}", letter)
    prompt = prompt.replace("{{text_obs}}", text_obs_clean)
    return prompt


async def _score_async(
    letter: str,
    text_obs: str,
    model: str,
    timeout: float,
    max_retries: int = 5,
) -> Optional[CharMatch]:
    """Call Gemini to score how well ``text_obs`` forms a ``letter`` tetromino.

    Mirrors the retry pattern in ``gemini_policy.GeminiPolicy.translate_async``:
    up to ``max_retries`` attempts, each with its own ``timeout``. Returns
    ``None`` if all attempts fail (caller decides how to handle missing rewards).
    """
    prompt = _build_prompt(letter, text_obs)

    for attempt in range(1, max_retries + 1):
        try:
            raw = await _call_gemini_async(
                prompt, model_id=model, timeout=timeout,
            )
            data = json.loads(raw)
            return CharMatch(**data)
        except asyncio.TimeoutError:
            logger.warning(
                "TetrisTaskProvider Gemini call timed out after %.1fs "
                "(letter=%s, attempt %d/%d)",
                timeout, letter, attempt, max_retries,
            )
            if attempt == max_retries:
                logger.error(
                    "TetrisTaskProvider Gemini call timed out after %d attempts "
                    "(letter=%s)", max_retries, letter,
                )
                return None
        except Exception as e:
            logger.warning(
                "TetrisTaskProvider Gemini call failed (letter=%s, attempt %d/%d): %s",
                letter, attempt, max_retries, e,
            )
            if attempt == max_retries:
                logger.error(
                    "TetrisTaskProvider Gemini call failed after %d attempts "
                    "(letter=%s): %s", max_retries, letter, e,
                )
                return None
    return None


class TetrisTaskProvider(TaskProvider):
    """Task provider that rewards arranging blocks into a tetromino shape.

    Parameters
    ----------
    shapes
        Iterable of one-sided tetromino letters to sample from on reset.
        Must be a subset of ``VALID_TETROMINOES``.
    model
        Gemini model id used for scoring. Passed straight through to
        ``gemini_policy._call_gemini_async``.
    instruction_template
        Format string with a ``{letter}`` placeholder.
    seed
        Seed for the per-group shape sampler. ``None`` = non-deterministic.
    group_n
        Number of worker environments per meta-RL group. Workers within a
        group share the same target shape (drawn once per group on reset)
        so that grouped rollouts see identical tasks and world seeds.
        Wired automatically from the env pool in ``build_task_provider``.
    timeout
        Per-call timeout for Gemini requests (seconds). Retries each get
        their own timeout.
    success_threshold
        Match score (in ``[0, 100]``) at/above which a turn is treated as a
        success. Read by the env manager and used to set ``info["won"]``
        (the env itself does not flag wins because the VLM is the validator).

    Notes
    -----
    The Gemini API key is read from ``GOOGLE_API_KEY`` inside
    ``gemini_policy._call_gemini_async``. ``thinking_level`` is hardcoded
    to ``"LOW"`` there as well; we deliberately follow that working policy
    to avoid drift.
    """

    def __init__(
        self,
        shapes: Optional[List[str]] = None,
        model: str = "gemini-3.1-flash-lite-preview",
        instruction_template: str = "Arrange the blocks into the shape of a {letter} tetromino.",
        seed: Optional[int] = None,
        group_n: int = 1,
        timeout: float = 60.0,
        success_threshold: float = 100.0,
    ) -> None:
        if shapes is None:
            shapes = list(VALID_TETROMINOES)
        shapes = list(shapes)
        invalid = [s for s in shapes if s not in VALID_TETROMINOES]
        if invalid:
            raise ValueError(
                f"TetrisTaskProvider got unknown shapes {invalid}. "
                f"Valid shapes: {VALID_TETROMINOES}"
            )
        if not shapes:
            raise ValueError("TetrisTaskProvider requires at least one shape.")

        if group_n < 1:
            raise ValueError(f"group_n must be >= 1, got {group_n}.")

        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY env var is not set. "
                "Set it in .env.language_table.secrets or export it."
            )

        self._shapes = shapes
        self._model = model
        self._instruction_template = instruction_template
        self._timeout = float(timeout)
        self._group_n = int(group_n)
        self._rng = np.random.default_rng(seed)
        self.success_threshold = float(success_threshold)

        self._env_shape: Dict[int, str] = {}
        self._group_shape: Dict[int, str] = {}
        self._cached_rewards: Dict[int, float] = {}
        self._last_details: Dict[int, Dict[str, Any]] = {}

    def instruction(
        self,
        text_obs: str,
        image: Optional[np.ndarray],
        env_idx: int,
    ) -> str:
        # Workers are laid out as contiguous groups of size ``group_n``
        # (see LanguageTableMultiProcessEnv which does np.repeat(seeds,
        # group_n)). Draw exactly once per group so every env in a group
        # shares the same target letter and world seed.
        group_idx = env_idx // self._group_n
        if env_idx % self._group_n == 0:
            letter = str(self._rng.choice(self._shapes))
            self._group_shape[group_idx] = letter
        else:
            letter = self._group_shape[group_idx]
        self._env_shape[env_idx] = letter
        return self._instruction_template.format(letter=letter)

    def precompute_rewards(
        self,
        text_obs_batch: List[str],
        image_batch: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Score the entire batch concurrently and cache per-env rewards."""
        batch = len(text_obs_batch)
        self._cached_rewards = {i: 0.0 for i in range(batch)}
        self._last_details = {i: {} for i in range(batch)}

        call_indices: List[int] = []
        for i in range(batch):
            if i not in self._env_shape:
                logger.warning(
                    "TetrisTaskProvider.precompute_rewards: env_idx=%d has no "
                    "assigned shape (instruction() not called yet). Skipping.",
                    i,
                )
                continue
            call_indices.append(i)

        if not call_indices:
            return

        async def _run() -> List[Any]:
            tasks = [
                _score_async(
                    letter=self._env_shape[i],
                    text_obs=text_obs_batch[i],
                    model=self._model,
                    timeout=self._timeout,
                )
                for i in call_indices
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(_run())

        for i, res in zip(call_indices, results):
            if isinstance(res, BaseException):
                logger.warning(
                    "TetrisTaskProvider unexpected exception for env_idx=%d "
                    "(letter=%s): %r",
                    i,
                    self._env_shape.get(i),
                    res,
                )
                continue
            if not isinstance(res, CharMatch):
                # _score_async returned None after exhausting retries; leave
                # reward at the default 0.0 and don't populate details.
                continue
            self._cached_rewards[i] = float(res.match_score)
            self._last_details[i] = {
                "letter": self._env_shape[i],
                "match_score": int(res.match_score),
                "looks_like_instead": res.looks_like_instead,
            }

            logger.info("PREDICTION" + str(self._last_details[i]) + text_obs_batch[i] + "\n\n")

    def reward(
        self,
        text_obs: str,
        image: Optional[np.ndarray],
        env_idx: int,
    ) -> float:
        return float(self._cached_rewards.get(env_idx, 0.0))
