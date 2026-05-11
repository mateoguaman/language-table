"""Prompt constants and the default editable block for the steerer VLM.

Structure of the full rendered prompt (assembled by MetaPrompt.render()):

    FIXED_SKELETON          ← never touched by optimizer
      {state_text}
      {top_instruction}
      {previous_response}

    <editable>              ← optimizer rewrites this freely
      ... rules, tips, example ...
    </editable text, no tags in actual prompt>

    OUTPUT_SPEC             ← fixed, always appended, never part of editable
"""

import textwrap

from language_table.environments.workspace_xy import TABLE_GRID_PROMPT

# ---------------------------------------------------------------------------
# Fixed skeleton – role, goal, table-grid description, dynamic slots.
# Never given to the optimizer as editable content.
# ---------------------------------------------------------------------------
FIXED_SKELETON = (
    textwrap.dedent("""\
You are an expert agent operating in a Language-Table environment. You see RGB when enabled; state text lists workspace-normalized coordinates **x**, **y** that match the camera layout below.

# Goal
You steer a language-conditioned robot policy that pushes colored blocks on a table by issuing short natural language commands.

# Table Grid
""")
    + TABLE_GRID_PROMPT
    + textwrap.dedent("""\

# Current State
{state_text}

# Task Goal
{top_instruction}

# Your Previous Response
{previous_response}\
""")
)

# ---------------------------------------------------------------------------
# Output spec – always appended by render(), never editable.
# Keeps JSON keys stable regardless of what the optimizer writes.
# ---------------------------------------------------------------------------
OUTPUT_SPEC = textwrap.dedent("""\
Respond with the following JSON format:
{{
    "plan": "<forward-looking plan: next steps to reach the goal>",
    "instruction": "<one short natural language command for the robot>"
}}\
""")

# ---------------------------------------------------------------------------
# Default editable block – the optimizer's starting point.
# Extracted from the original PLAN_PROMPT_TEXT (Rules + Tips + Example).
# Does NOT include the output format (that lives in OUTPUT_SPEC).
# ---------------------------------------------------------------------------
# DEFAULT_EDITABLE = textwrap.dedent("""\
# # Rules
# - Issue ONE short natural language command per turn.
# - Always refer to blocks as "color + shape" (e.g. "red moon", "blue cube"). Never say "the red block".

# # Tips for Planning
# - Plan the shortest path to accomplish the task minimizing the number of moves and distance the blocks must be moved.
# - Plan a collision free path to move the blocks to their target locations.
# - Use your previous response (stored in "plan") to infer the plan for the next step. If the previous response is not completed yet you can repeat the same plan until it is completed.

# # Example
# Here's an example of an L shape:
# Target configuration:
#     | red moon        | (empty)    | (empty) |
#     | blue cube       | (empty)    | (empty) |
#     | yellow pentagon | green star | (empty) |\
# """)

DEFAULT_EDITABLE = textwrap.dedent("""\
# Rules
Each action must use this exact format:
push the <color> <shape> to the <location>

- Allowed colors: red, blue, green, yellow
- Allowed shapes: moon, cube, star, pentagon
- Allowed locations: top left corner, top center, top right corner, center left, center, center right, bottom left corner, bottom center, bottom right corner

Plan shortest moves. Minimize block travel. Avoid collisions.

You cannot push blocks out of corners. Once they're in a corner, they're stuck. Be careful and replan accordingly if stuck!

Only replan if you're stuck.

# Example
Here's an example of an L shape:
Target configuration:
    | red moon        | (empty)    | (empty) |
    | blue cube       | (empty)    | (empty) |
    | yellow pentagon | green star | (empty) |

Report 'plan' like in the example above, i.e., target configuration, then plan.
""")


# ---------------------------------------------------------------------------
# Feedback prompts – used by feedback() to ask a VLM/LLM for episode critique.
# ---------------------------------------------------------------------------
FEEDBACK_TEXT_PROMPT = textwrap.dedent("""\
You are evaluating a robot episode in the Language-Table environment.

# Task Goal
{top_instruction}

# Episode Trajectory
The following state observations and low-level instructions were recorded step-by-step:

{trajectory}

# Your Task
Assess whether the robot made progress toward the goal.
Point out mistakes, inefficiencies, or successful strategies.
Be concise and specific. Return plain text, no JSON.\
""")

FEEDBACK_IMAGE_PROMPT = textwrap.dedent("""\
You are evaluating a robot episode in the Language-Table environment.

# Task Goal
{top_instruction}

# Episode Trajectory
The following low-level instructions were issued at each step:

{ll_instructions}

Images of the workspace are attached in order (one per step).

# Your Task
Assess whether the robot made progress toward the goal based on the visual evidence.
Point out mistakes, inefficiencies, or successful strategies.
Be concise and specific. Return plain text, no JSON.\
""")


def render_feedback_text_prompt(
    top_instruction: str,
    state_texts: list[str],
    ll_instructions: list[str] | None = None,
) -> str:
    """Render the text-modality feedback prompt."""
    lines: list[str] = []
    for i, st in enumerate(state_texts):
        lines.append(f"Step {i}:")
        lines.append(st)
        if ll_instructions and i < len(ll_instructions):
            lines.append(f"  → instruction: {ll_instructions[i]}")
    trajectory = "\n".join(lines)
    return FEEDBACK_TEXT_PROMPT.format(
        top_instruction=top_instruction,
        trajectory=trajectory,
    )


def render_feedback_image_prompt(
    top_instruction: str,
    ll_instructions: list[str] | None = None,
) -> str:
    """Render the image-modality feedback prompt (frames attached separately)."""
    if ll_instructions:
        ll_str = "\n".join(
            f"Step {i}: {instr}" for i, instr in enumerate(ll_instructions)
        )
    else:
        ll_str = "(no low-level instructions recorded)"
    return FEEDBACK_IMAGE_PROMPT.format(
        top_instruction=top_instruction,
        ll_instructions=ll_str,
    )


AGG_FEEDBACK_PROMPT = textwrap.dedent("""\
You are summarizing feedback from multiple robot episodes in the Language-Table environment.

# Per-Episode Feedback
{per_episode_feedback}

# Your Task
Synthesize the above into a single concise summary.
Identify recurring failure modes, consistent successes, and actionable patterns.
The summary will be used to improve the robot's instruction policy.
Return plain text, no JSON.\
""")


def render_agg_feedback_prompt(feedbacks: list[str]) -> str:
    """Render the aggregate feedback prompt from per-episode feedback strings."""
    numbered = "\n\n".join(
        f"Episode {i + 1}:\n{f}" for i, f in enumerate(feedbacks)
    )
    return AGG_FEEDBACK_PROMPT.format(per_episode_feedback=numbered)


def make_default_meta_prompt() -> "MetaPrompt":
    """Return a MetaPrompt initialised with DEFAULT_EDITABLE."""
    from opro.opro_types import MetaPrompt
    return MetaPrompt(editable=DEFAULT_EDITABLE)
