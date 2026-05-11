"""OPROLoop: outer loop orchestrating OPRO meta-prompt optimization.

Each iteration:
  1. Evaluate current candidate via Evaluator (f(x)).
  2. Aggregate per-env results: success_rate + agg_feedback.
  3. Append HistoryEntry to history.
  4. Log iteration results to JSON.
  5. Call OptimizerLLM to propose next candidate.
  6. Repeat until budget exhausted.

Returns the best MetaPrompt found across all iterations.
"""

from __future__ import annotations

import json
import os
import time
from typing import List

from opro.evaluator import Evaluator
from opro.feedback import AggFeedbackFn, agg_feedback as default_agg_feedback
from opro.optimizer import HistoryEntry, OptimizerLLM
from opro.opro_types import EvaluationResult, MetaPrompt

_LOG_SEP = "=" * 72


def _log_meta_prompt_after_update(iteration: int, editable: str) -> None:
    print(
        f"\n{_LOG_SEP}\n[OPRO] meta-prompt editable (after update, next iter index {iteration})\n"
        f"{_LOG_SEP}\n{editable.rstrip()}\n{_LOG_SEP}\n"
    )


class OPROLoop:
    """Outer OPRO optimization loop.

    Args:
        evaluator: Evaluator instance (f(x)).
        optimizer: OptimizerLLM instance.
        n_iterations: total number of OPRO iterations.
        log_dir: root directory for all logs and videos.
        agg_feedback_fn: function to aggregate per-env feedback strings.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        optimizer: OptimizerLLM,
        n_iterations: int,
        log_dir: str,
        agg_feedback_fn: AggFeedbackFn = default_agg_feedback,
    ) -> None:
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.n_iterations = n_iterations
        self.log_dir = log_dir
        self.agg_feedback_fn = agg_feedback_fn
        os.makedirs(log_dir, exist_ok=True)

    async def run(self, initial_candidate: MetaPrompt) -> MetaPrompt:
        """Run the OPRO loop; return best MetaPrompt found."""
        history: list[HistoryEntry] = []
        all_results: list[dict] = []

        candidate = initial_candidate
        best_candidate = candidate
        best_score = -1.0

        for iteration in range(self.n_iterations):
            print(f"\n{'='*60}")
            print(f"[OPRO] Iteration {iteration + 1}/{self.n_iterations}")
            print(f"{'='*60}")

            # --- Evaluate ---
            t_eval = time.monotonic()
            result: EvaluationResult = await self.evaluator(
                candidate=candidate,
                iteration=iteration,
                log_dir=self.log_dir,
            )
            t_eval_elapsed = time.monotonic() - t_eval
            print(f"[timing iter={iteration}] total_eval={t_eval_elapsed:.3f}s")

            # --- Aggregate ---
            score = result.adjusted_success_rate
            n_won = sum(result.per_env_won)
            n_envs = result.n_envs
            n_initially_won = sum(result.per_env_initially_won)
            t_agg = time.monotonic()
            _agg_result = self.agg_feedback_fn(result.per_env_feedback)
            import inspect
            aggregated_feedback = await _agg_result if inspect.isawaitable(_agg_result) else _agg_result
            print(f"[timing iter={iteration}] feedback_aggregation={time.monotonic()-t_agg:.3f}s")

            print(
                f"[OPRO iter={iteration}] score={score:.3f} (adjusted) "
                f"raw={result.success_rate:.3f} "
                f"won={n_won}/{n_envs} initially_won={n_initially_won}/{n_envs} "
                f"mean_reward={result.mean_reward:.3f}"
            )
            if aggregated_feedback:
                print(f"[OPRO iter={iteration}] feedback: {aggregated_feedback[:200]}")

            # Track best
            if score > best_score:
                best_score = score
                best_candidate = candidate
                print(f"[OPRO iter={iteration}] *** new best score: {best_score:.3f} ***")

            # --- Build history entry ---
            entry = HistoryEntry(
                iteration=iteration,
                score=score,
                n_won=n_won,
                n_envs=n_envs,
                n_initially_won=n_initially_won,
                agg_feedback=aggregated_feedback,
                editable=candidate.editable,
            )
            history.append(entry)

            # --- Log iteration ---
            iter_log = {
                "iteration": iteration,
                "score": score,
                "score_raw": result.success_rate,
                "n_won": n_won,
                "n_envs": n_envs,
                "n_initially_won": n_initially_won,
                "mean_reward": result.mean_reward,
                "per_env_won": result.per_env_won,
                "per_env_initially_won": result.per_env_initially_won,
                "per_env_reward": result.per_env_reward,
                "per_env_feedback": result.per_env_feedback,
                "agg_feedback": aggregated_feedback,
                "editable": candidate.editable,
                "best_score_so_far": best_score,
                "timing_s": {"total_eval": round(t_eval_elapsed, 3)},
            }
            all_results.append(iter_log)
            self._write_log(all_results)
            self._save_iter_log(iter_log)
            self._save_best(best_candidate, best_score)
            self._save_score_plot(all_results)

            # --- Propose next candidate (skip on last iteration) ---
            if iteration < self.n_iterations - 1:
                print(f"[OPRO iter={iteration}] calling optimizer...")
                t_propose = time.monotonic()
                candidate = await self.optimizer.propose(
                    history=history,
                    base=best_candidate,
                )
                t_propose_elapsed = time.monotonic() - t_propose
                print(f"[timing iter={iteration}] prompt_update={t_propose_elapsed:.3f}s")
                iter_log["timing_s"]["prompt_update"] = round(t_propose_elapsed, 3)
                _log_meta_prompt_after_update(iteration + 1, candidate.editable)

        print(f"\n[OPRO] Done. Best score: {best_score:.3f}")
        return best_candidate

    def _write_log(self, all_results: list[dict]) -> None:
        path = os.path.join(self.log_dir, "opro_history.json")
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)

    def _save_best(self, best: MetaPrompt, score: float) -> None:
        path = os.path.join(self.log_dir, "best_prompt.txt")
        with open(path, "w") as f:
            f.write(f"# Best score: {score:.4f}\n\n")
            f.write(best.editable)
        print(f"[OPRO] best prompt saved to {path}")

    def _save_iter_log(self, iter_log: dict) -> None:
        """Write human-readable per-iteration log: meta-prompt + per-env results + feedback."""
        i = iter_log["iteration"]
        path = os.path.join(self.log_dir, f"iter_{i:03d}.txt")
        sep = "=" * 72
        lines = [
            sep,
            f"ITERATION {i}",
            sep,
            "",
            "--- META-PROMPT (editable block) ---",
            iter_log["editable"].rstrip(),
            "",
            "--- RESULTS ---",
            f"  score (adjusted):         {iter_log['score']:.4f}",
            f"  score_raw (all envs):     {iter_log['score_raw']:.4f}",
            f"  mean_reward:              {iter_log['mean_reward']:.4f}",
            f"  won: {iter_log['n_won']}/{iter_log['n_envs']} (initially_won={iter_log['n_initially_won']})",
            f"  best_score_so_far:        {iter_log['best_score_so_far']:.4f}",
            "",
            "--- PER-ENV RESULTS ---",
        ]
        n = len(iter_log["per_env_won"])
        for env_idx in range(n):
            won = iter_log["per_env_won"][env_idx]
            rew = iter_log["per_env_reward"][env_idx]
            fb = iter_log["per_env_feedback"][env_idx] if iter_log["per_env_feedback"] else ""
            lines.append(f"  env {env_idx:02d}  won={won}  reward={rew:.4f}  feedback={fb!r}")
        lines += [
            "",
            "--- AGGREGATED FEEDBACK ---",
            iter_log["agg_feedback"] or "(none)",
            "",
            sep,
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _save_score_plot(self, all_results: List[dict]) -> None:
        """Save score-progression line plot (score + best_score_so_far vs iteration)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        iters = [r["iteration"] for r in all_results]
        scores = [r["score"] for r in all_results]
        best_scores = [r["best_score_so_far"] for r in all_results]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iters, scores, marker="o", label="score (iter)")
        ax.plot(iters, best_scores, marker="s", linestyle="--", label="best so far")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score (success rate)")
        ax.set_title("OPRO Score Progression")
        ax.set_xticks(iters)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(self.log_dir, "score_progression.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
