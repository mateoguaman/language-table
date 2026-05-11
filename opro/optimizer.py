"""OptimizerLLM: outer-loop OPRO optimizer.

Receives the history of (EvaluationResult, score, agg_feedback) entries,
builds an OPRO-style meta-prompt showing the last K entries sorted by
score descending, and proposes a new MetaPrompt editable block.

The new editable block is extracted from the LLM response via
<editable>...</editable> tags with retry logic.
"""

from __future__ import annotations

import asyncio
import re
import textwrap
from dataclasses import dataclass
from typing import Optional

from opro.prompt import FIXED_SKELETON, OUTPUT_SPEC
from opro.opro_types import MetaPrompt

MAX_RETRIES = 5


@dataclass
class HistoryEntry:
    """Aggregated entry for the optimizer history."""
    iteration: int
    score: float          # adjusted_success_rate (excludes pre-solved envs)
    n_won: int
    n_envs: int
    n_initially_won: int  # envs already solved at reset (not attributable to prompt)
    agg_feedback: str
    editable: str         # the candidate's editable block


_OPTIMIZER_SYSTEM_PROMPT = (
    textwrap.dedent("""\
    You are an expert prompt engineer optimizing the middle section of a steerer prompt for a Language-Table robot agent.

    Full prompt layout at runtime (you only rewrite the middle):
      [FIXED PREFIX below] + [your <editable> block] + [FIXED SUFFIX below]

    The prefix fills {state_text}, {top_instruction}, and {previous_response} each step. You must not duplicate,
    paraphrase, or re-teach anything already in the fixed prefix or suffix—especially: role/goal preamble, 3x3 grid
    legend, the section titles Current State / Task Goal / Your Previous Response, or JSON output / "history" /
    "instruction" instructions. Put only extra steering content in <editable> (rules, tactics, examples, heuristics).

    Guidelines:
    - Use feedback from history entries; be specific and actionable.
    - You may restructure, add, remove, or rewrite freely within the editable scope.

    Return ONLY the new editable block wrapped in <editable>...</editable> tags.

    --- FIXED PREFIX (verbatim template; not editable) ---

    """)
    + FIXED_SKELETON
    + textwrap.dedent("""\


    --- FIXED SUFFIX (verbatim; not editable) ---

    """)
    + OUTPUT_SPEC
    + "\n"
)


def _build_optimizer_prompt(
    history: list[HistoryEntry],
    history_window: int,
) -> str:
    """Build the OPRO meta-prompt from history entries."""
    # Show last K entries sorted by score descending (best first)
    window = sorted(history[-history_window:], key=lambda e: e.score, reverse=True)

    entries_text = ""
    for entry in window:
        fb_section = f"\n[FEEDBACK]\n{entry.agg_feedback}" if entry.agg_feedback.strip() else ""
        n_active = entry.n_envs - entry.n_initially_won
        initially_note = (
            f" ({entry.n_initially_won} pre-solved at reset, not prompt-attributable; "
            f"score = {entry.n_won - entry.n_initially_won}/{n_active} active envs)"
            if entry.n_initially_won > 0 else ""
        )
        entries_text += textwrap.dedent(f"""\
--- Iteration {entry.iteration} | Score: {entry.score:.3f} | Won: {entry.n_won}/{entry.n_envs}{initially_note} ---
[EDITABLE BLOCK]
{entry.editable}
{fb_section}
""")

    return textwrap.dedent(f"""\
{_OPTIMIZER_SYSTEM_PROMPT}

# History of Evaluated Prompts (best first within window)

{entries_text}
# Your Task

Propose a new editable block that improves on the above. Return it wrapped in <editable>...</editable> tags.
""")


def _extract_editable(text: str) -> Optional[str]:
    """Extract content between <editable>...</editable> tags."""
    m = re.search(r"<editable>(.*?)</editable>", text, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


class OptimizerLLM:
    """Proposes new MetaPrompt candidates given OPRO history.

    Args:
        model_id: Gemini model ID for the optimizer LLM.
        history_window: max number of history entries shown per proposal.
        thinking_level: Gemini thinking level for optimizer calls.
    """

    def __init__(
        self,
        model_id: str,
        history_window: int = 5,
        thinking_level: str = "HIGH",
    ) -> None:
        self.model_id = model_id
        self.history_window = history_window
        self.thinking_level = thinking_level

    async def propose(
        self,
        history: list[HistoryEntry],
        base: MetaPrompt,
    ) -> MetaPrompt:
        """Propose a new MetaPrompt given the current history.

        Falls back to the base (current best) if all retries fail.
        """
        from opro.models import call_model

        optimizer_prompt = _build_optimizer_prompt(history, self.history_window)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await call_model(
                    prompt=optimizer_prompt,
                    img_input=None,
                    thinking_level=self.thinking_level,
                    json_output=False,
                    model_id=self.model_id,
                )
                raw_text = resp.text
                editable = _extract_editable(raw_text)
                if editable:
                    print(f"[optimizer] proposed new editable (attempt {attempt})")
                    return MetaPrompt(editable=editable)
                print(f"[optimizer] parse failed: no <editable> tags (attempt {attempt}/{MAX_RETRIES})")
            except Exception as e:
                print(f"[optimizer] call error (attempt {attempt}/{MAX_RETRIES}): {e}")
                await asyncio.sleep(2 ** attempt)

        print("[optimizer] all retries failed, returning base candidate unchanged")
        return MetaPrompt(editable=base.editable)
