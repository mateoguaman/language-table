#!/usr/bin/env python3
"""OPRO meta-prompt optimizer for Language-Table.

Outer loop: OptimizerLLM proposes new meta-prompt editable blocks.
Inner loop: Evaluator rolls out the steerer VLM + low-level policy over
            n_envs parallel environments and scores with ScoringFn.

Usage:
    python run_opro.py \\
        --n_iterations 10 \\
        --n_steps 30 \\
        --steerer_model gemini-3-flash-preview \\
        --optimizer_model gemini-3-pro-preview \\
        --reward_fn geometric \\
        --obs_modality text \\
        --output_dir ./opro_runs
"""

import asyncio
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

# Ensure language-table directory is on the path so opro/ package and
# sibling modules (gemini_legacy, tetris_shape_reward, etc.) are importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opro.evaluator import Evaluator
from opro.feedback import agg_feedback, feedback
from opro.loop import OPROLoop
from opro.optimizer import OptimizerLLM
from opro.prompt import make_default_meta_prompt
from opro.scoring import GeometricScoring, VLMScoring

HOST = "127.0.0.1"
PORT = 50051


def parse_args():
    p = ArgumentParser(description="OPRO meta-prompt optimizer for Language-Table")
    p.add_argument(
        "--n_iterations", type=int, default=10, help="Number of outer OPRO iterations"
    )
    p.add_argument(
        "--n_steps",
        type=int,
        default=30,
        help="Max inner-loop steps per episode rollout",
    )
    p.add_argument(
        "--steerer_model",
        type=str,
        default="claude-haiku-4-5-20251001", # "gpt-5.4-mini",
        help="Model ID for the inner-loop steerer VLM (gemini-*, gpt-*, claude-*)",
    )
    p.add_argument(
        "--feedback_model",
        type=str,
        default="claude-haiku-4-5-20251001", # "deepseek-v4-flash", # "gemini-3-flash-preview",
        help="Model ID for per-env feedback and agg_feedback (gemini-*, gpt-*, claude-*)",
    )
    p.add_argument(
        "--optimizer_model",
        type=str,
        default="claude-haiku-4-5-20251001", # #"claude-3-5-haiku-latest", # "gemini-3-flash-preview",
        help="Model ID for the outer-loop optimizer LLM (gemini-*, gpt-*, claude-*)",
    )
    p.add_argument(
        "--reward_fn",
        choices=["geometric", "vlm"],
        default="geometric",
        help=(
            "geometric: mean per-step tetromino geometric score in [0,1]. "
            "vlm: mean per-step score (0 mid-episode, verifier at terminal)."
        ),
    )
    p.add_argument(
        "--obs_modality",
        choices=["image", "text"],
        default="text",
        help=(
            "image: RGB frames fed to steerer VLM and feedback(). "
            "text: block position text fed to steerer VLM and feedback(). "
            "Outer-loop optimizer is always text-only."
        ),
    )
    p.add_argument(
        "--history_window",
        type=int,
        default=5,
        help="Number of past history entries shown to the optimizer (sorted by score desc)",
    )
    p.add_argument(
        "--thinking",
        type=str,
        default=None,
        help="Override thinking level for all components. NONE=no thinking; Claude: adaptive/enabled; Gemini: LOW/MEDIUM/HIGH; GPT: ignored.",
    )
    p.add_argument(
        "--optimizer_thinking",
        type=str,
        default="MEDIUM",
        dest="update_thinking",
        help="Thinking level for outer-loop optimizer. NONE=no thinking; Claude: adaptive/enabled; Gemini: LOW/MEDIUM/HIGH; GPT: ignored.",
    )
    p.add_argument(
        "--steerer_thinking",
        type=str,
        default="MEDIUM",
        help="Thinking level for inner-loop steerer VLM. NONE=no thinking.",
    )
    p.add_argument(
        "--feedback_thinking",
        type=str,
        default="LOW",
        help="Thinking level for per-env feedback and agg_feedback. NONE=no thinking.",
    )
    p.add_argument(
        "--video_fps",
        type=float,
        default=10.0,
        help="Frames per second for saved episode videos",
    )
    p.add_argument("--host", type=str, default=HOST, help="Env server host")
    p.add_argument("--port", type=int, default=PORT, help="Env server port")
    p.add_argument(
        "--output_dir",
        type=str,
        default="./opro_runs",
        help="Root directory for logs, videos, and best-prompt files",
    )
    p.add_argument(
        "--skip_feedback",
        action="store_true",
        help="Skip per-env feedback and aggregation (pass empty strings to optimizer)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override all model IDs (optimizer, feedback, steerer) with this value",
    )
    return p.parse_args()


async def main():
    args = parse_args()

    if args.model is not None:
        args.steerer_model = args.model
        args.feedback_model = args.model
        args.optimizer_model = args.model

    if args.thinking is not None:
        args.steerer_thinking = args.thinking
        args.update_thinking = args.thinking
        args.feedback_thinking = args.thinking

    # "NONE" sentinel → None (adaptive thinking)
    for attr in ("steerer_thinking", "update_thinking", "feedback_thinking"):
        if getattr(args, attr, None) == "NONE":
            setattr(args, attr, None)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[opro] run_id={run_id}")
    print(f"[opro] log_dir={log_dir}")
    print(f"[opro] n_iterations={args.n_iterations} n_steps={args.n_steps}")
    print(
        f"[opro] steerer={args.steerer_model} feedback_model={args.feedback_model} "
        f"optimizer={args.optimizer_model}"
    )
    print(f"[opro] reward_fn={args.reward_fn} obs_modality={args.obs_modality}")
    print(f"[opro] skip_feedback={args.skip_feedback}")
    print(
        f"[opro] thinking: steerer={args.steerer_thinking} "
        f"update={args.update_thinking} feedback={args.feedback_thinking}"
    )

    # --- Build feedback functions ---
    if args.skip_feedback:

        def feedback_fn(*a, **kw) -> str:
            return ""

        def agg_feedback_fn(feedbacks: list) -> str:
            return ""

    else:
        _feedback_logged = False

        async def feedback_fn(*a, **kw) -> str:
            nonlocal _feedback_logged
            result = await feedback(
                *a,
                **kw,
                modality=args.obs_modality,
                model_id=args.feedback_model,
                thinking_level=args.feedback_thinking,
            )
            if not _feedback_logged:
                print(
                    f"[feedback] first response (env 0, truncated 500):\n{result[:500]}"
                )
                _feedback_logged = True
            return result

        async def agg_feedback_fn(feedbacks: list) -> str:
            result = await agg_feedback(
                feedbacks,
                model_id=args.feedback_model,
                thinking_level=args.feedback_thinking,
            )
            print(f"[feedback] agg_feedback response:\n{result[:500]}")
            return result

    # --- Build scoring function ---
    if args.reward_fn == "geometric":
        scoring_fn = GeometricScoring()
    else:
        scoring_fn = VLMScoring(
            model_id=args.steerer_model,
            obs_modality=args.obs_modality,
        )

    # --- Build components ---
    evaluator = Evaluator(
        host=args.host,
        port=args.port,
        n_steps=args.n_steps,
        steerer_model=args.steerer_model,
        obs_modality=args.obs_modality,
        scoring_fn=scoring_fn,
        video_fps=args.video_fps,
        feedback_fn=feedback_fn,
        steerer_thinking=args.steerer_thinking,
    )

    optimizer = OptimizerLLM(
        model_id=args.optimizer_model,
        history_window=args.history_window,
        thinking_level=args.update_thinking,
    )

    loop = OPROLoop(
        evaluator=evaluator,
        optimizer=optimizer,
        n_iterations=args.n_iterations,
        log_dir=log_dir,
        agg_feedback_fn=agg_feedback_fn,
    )

    # --- Run ---
    initial_candidate = make_default_meta_prompt()
    best = await loop.run(initial_candidate)

    print("\n[opro] Optimization complete.")
    print(f"[opro] Best editable block:\n{'-'*60}\n{best.editable}\n{'-'*60}")
    print(f"[opro] Results saved to: {log_dir}")


if __name__ == "__main__":
    asyncio.run(main())
