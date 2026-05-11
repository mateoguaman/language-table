"""Core dataclasses for OPRO meta-prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class MetaPrompt:
    """Holds the single optimizable block of the steerer prompt.

    The fixed skeleton (role, table grid, state/instruction slots) and
    OUTPUT_SPEC are managed by prompt.py and never part of this dataclass.
    """

    editable: str

    def render(
        self,
        state_text: str,
        top_instruction: str,
        previous_response: str,
    ) -> str:
        """Assemble the full steerer prompt for one env step."""
        from opro.prompt import FIXED_SKELETON, OUTPUT_SPEC

        return (
            FIXED_SKELETON.format(
                state_text=state_text,
                top_instruction=top_instruction,
                previous_response=previous_response,
            )
            + "\n"
            + self.editable
            + "\n\n"
            + OUTPUT_SPEC
        )


@dataclass
class EvaluationResult:
    """Raw per-environment results from one f(x) evaluation.

    Aggregation (success_rate, agg_feedback) is performed by OPROLoop,
    not here. Both frame and state-text lists are always populated;
    one will be empty lists per env when that modality is not in use.
    """

    candidate: MetaPrompt
    iteration: int
    per_env_won: list[bool]
    per_env_reward: list[float]
    per_env_feedback: list[str]
    per_env_frames: list[list[np.ndarray]]      # empty inner lists if modality=="text"
    per_env_state_texts: list[list[str]]         # empty inner lists if modality=="image"
    per_env_initially_won: list[bool] = field(default_factory=list)

    @property
    def n_envs(self) -> int:
        return len(self.per_env_won)

    @property
    def success_rate(self) -> float:
        if not self.per_env_won:
            return 0.0
        return sum(self.per_env_won) / len(self.per_env_won)

    @property
    def adjusted_success_rate(self) -> float:
        """Success rate excluding envs already solved at reset.

        Denominator = n_envs - n_initially_won, keeping ceiling at 1.0.
        Falls back to success_rate if no initially-won data or all envs pre-solved.
        """
        if not self.per_env_initially_won:
            return self.success_rate
        n_initial = sum(self.per_env_initially_won)
        denom = self.n_envs - n_initial
        if denom <= 0:
            return self.success_rate
        n_won_by_agent = sum(
            w and not i
            for w, i in zip(self.per_env_won, self.per_env_initially_won)
        )
        return n_won_by_agent / denom

    @property
    def mean_reward(self) -> float:
        if not self.per_env_reward:
            return 0.0
        return float(np.mean(self.per_env_reward))
