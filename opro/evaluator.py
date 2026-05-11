"""Evaluator: f(x) for the OPRO outer loop.

Runs one full rollout of the steerer VLM + low-level policy over n_envs
parallel environments for a given MetaPrompt, collects per-env trajectories,
and returns a raw EvaluationResult (no aggregation).

Aggregation (success_rate, agg_feedback) is performed by OPROLoop.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from opro.env_utils import (
    llm_batch,
    overlay_instructions_rgb,
    rpc,
    state_to_text,
    to_uint8_rgb,
    write_env_videos,
)
from opro.feedback import FeedbackFn, feedback
from opro.scoring import ScoringFn
from opro.opro_types import EvaluationResult, MetaPrompt


def _get_state_texts(obs: dict, n: int) -> list[str]:
    """Prefer obs['text'] batch (env manager); fall back to state_to_text."""
    text_obs_list = obs.get("text") or []
    state_list = obs.get("state") or [{}] * n
    if isinstance(text_obs_list, list) and len(text_obs_list) == n:
        return [str(t) for t in text_obs_list]
    return [state_to_text(state_list[i]) for i in range(n)]


class Evaluator:
    """Evaluates a MetaPrompt by rolling it out over n_envs environments.

    Args:
        host: env server host.
        port: env server port.
        n_steps: max inner-loop steps per episode.
        steerer_model: Gemini model ID for the steerer VLM.
        obs_modality: "image" (RGB frames) or "text" (block position text).
        scoring_fn: ScoringFn instance called at every step.
        feedback_fn: per-env feedback function (default stub).
        video_fps: frames per second for saved videos.
        steerer_thinking: Gemini thinking level for inner-loop steerer VLM.
    """

    def __init__(
        self,
        host: str,
        port: int,
        n_steps: int,
        steerer_model: str,
        obs_modality: Literal["image", "text"],
        scoring_fn: ScoringFn,
        feedback_fn: FeedbackFn = feedback,
        video_fps: float = 10.0,
        steerer_thinking: str = "MEDIUM",
    ) -> None:
        self.host = host
        self.port = port
        self.n_steps = n_steps
        self.steerer_model = steerer_model
        self.obs_modality = obs_modality
        self.scoring_fn = scoring_fn
        self.feedback_fn = feedback_fn
        self.video_fps = video_fps
        self.steerer_thinking = steerer_thinking

    async def __call__(
        self,
        candidate: MetaPrompt,
        iteration: int,
        log_dir: str,
    ) -> EvaluationResult:
        """Roll out candidate over n_envs envs; return raw EvaluationResult."""
        iter_log_dir = os.path.join(log_dir, f"iter_{iteration:04d}")
        os.makedirs(iter_log_dir, exist_ok=True)

        with socket.create_connection((self.host, self.port), timeout=30) as sock:
            props = rpc(sock, "get_properties")
            n = props["num_processes"]
            block_mode = props.get("block_mode") or "BLOCK_8"
            num_blocks = int(block_mode.split("_")[-1]) if "_" in block_mode else 8
            all_blocks = [
                "red moon", "red pentagon", "blue moon", "blue cube",
                "green cube", "green star", "yellow star", "yellow pentagon",
            ]
            block_names = f"The {num_blocks} active blocks are: {', '.join(all_blocks[:num_blocks])}."
            print(f"[eval iter={iteration}] n_envs={n} block_mode={block_mode}")

            # Per-env accumulators (reward = mean step score over scored steps)
            per_env_frames: list[list[np.ndarray]] = [[] for _ in range(n)]
            per_env_state_texts: list[list[str]] = [[] for _ in range(n)]
            per_env_instructions: list[list[str]] = [[] for _ in range(n)]
            per_env_score_sum: list[float] = [0.0] * n
            per_env_score_count: list[int] = [0] * n
            per_env_won: list[bool] = [False] * n
            prev_responses: list[str] = ["(none)"] * n
            all_done: list[bool] = [False] * n

            t_reset = time.monotonic()
            obs, infos = rpc(sock, "reset")
            print(f"[timing iter={iteration}] env_reset={time.monotonic()-t_reset:.3f}s")
            top_instruction = obs["text"][0].split("\n")[0]
            print(f"[eval iter={iteration}] task: {top_instruction!r}")

            # Score initial obs to detect pre-solved envs (before any agent action)
            init_state_list = obs.get("state") or [{}] * n
            init_state_texts = _get_state_texts(obs, n)
            init_score_tasks = [
                self.scoring_fn(
                    step=-1,
                    done=True,
                    state=init_state_list[i] if isinstance(init_state_list[i], dict) else {},
                    frame=to_uint8_rgb(obs["image"][i]) if obs.get("image") is not None else None,
                    state_text=init_state_texts[i],
                    top_instruction=top_instruction,
                )
                for i in range(n)
            ]
            WIN_THRESHOLD = 0.9
            init_scores = await asyncio.gather(*init_score_tasks)
            per_env_initially_won: list[bool] = [float(s) >= WIN_THRESHOLD for s in init_scores]
            n_initially_won = sum(per_env_initially_won)
            print(f"[eval iter={iteration}] initially_won={n_initially_won}/{n} scores={[f'{s:.3f}' for s in init_scores]}")

            if obs.get("image") is None and self.obs_modality == "image":
                raise RuntimeError(
                    "Server returned obs['image']=None. Launch with --include_rgb."
                )

            # Record initial frames
            if obs.get("image") is not None:
                for i, img in enumerate(obs["image"]):
                    rgb = to_uint8_rgb(img)
                    annotated = overlay_instructions_rgb(rgb, top_instruction, "(no low-level command yet)")
                    per_env_frames[i].append(annotated.copy())

            step_bar = tqdm(
                range(self.n_steps),
                desc=f"eval iter={iteration}",
                unit="step",
                leave=False,
            )
            for step in step_bar:
                if all(all_done):
                    break

                active = [i for i, d in enumerate(all_done) if not d]

                # Build per-env state texts (same source after reset and after each step).
                text_obs_list = obs.get("text") or []
                state_list = obs.get("state") or [{}] * n
                cur_state_texts = _get_state_texts(obs, n)

                # Debug: compare obs["text"] (manager) vs local state_to_text(obs["state"]) for env=0
                if step == 0:
                    manager_text = str(text_obs_list[0]) if (isinstance(text_obs_list, list) and text_obs_list) else "(unavailable)"
                    local_text = state_to_text(state_list[0]) if state_list else "(unavailable)"
                    print(f"[state-text-debug iter={iteration} step={step}] obs['text'][0]:\n{manager_text}")
                    print(f"[state-text-debug iter={iteration} step={step}] state_to_text(obs['state'][0]):\n{local_text}")
                    # Compare only the block-positions section (manager text includes task+EE prefix)
                    BLOCK_HEADER = "Block positions (normalized [0,1]):"
                    manager_block = manager_text[manager_text.find(BLOCK_HEADER):] if BLOCK_HEADER in manager_text else manager_text
                    if manager_block != local_text:
                        print(f"[state-text-debug iter={iteration} step={step}] MISMATCH between sources")

                # Render prompts for active envs
                rendered_prompts = [
                    candidate.render(
                        state_text=cur_state_texts[i],
                        top_instruction=top_instruction,
                        previous_response=prev_responses[i],
                    )
                    for i in active
                ]
                images_active = (
                    [to_uint8_rgb(obs["image"][i]) for i in active]
                    if self.obs_modality == "image" and obs.get("image") is not None
                    else [None] * len(active)
                )

                t_vlm = time.monotonic()
                responses_active = await llm_batch(
                    meta_prompt_rendered_list=rendered_prompts,
                    images=images_active,
                    model_id=self.steerer_model,
                    fallback=top_instruction,
                    thinking_level=self.steerer_thinking,
                )
                print(f"[timing iter={iteration} step={step}] vlm_inference={time.monotonic()-t_vlm:.3f}s n_active={len(active)}")

                instructions = [top_instruction] * n
                for idx, eid in enumerate(active):
                    resp = responses_active[idx]
                    instructions[eid] = resp.get("instruction", top_instruction)
                    prev_responses[eid] = json.dumps(resp)
                    if eid == 0:
                        print(
                            f"[eval iter={iteration} step={step}] env=0 "
                            f"history={resp.get('history','')!r} → {instructions[eid]!r}"
                        )

                t_step = time.monotonic()
                obs, rewards, dones, infos = rpc(sock, "step", instructions, phase="play")
                print(f"[timing iter={iteration} step={step}] env_step={time.monotonic()-t_step:.3f}s")

                # Collect frames / state texts for this step
                new_state_list = obs.get("state") or [{}] * n
                new_state_texts = _get_state_texts(obs, n)

                for i in range(n):
                    per_env_instructions[i].append(instructions[i])
                    if self.obs_modality == "text":
                        per_env_state_texts[i].append(new_state_texts[i])
                    if obs.get("image") is not None:
                        rgb = to_uint8_rgb(obs["image"][i])
                        annotated = overlay_instructions_rgb(rgb, top_instruction, instructions[i])
                        per_env_frames[i].append(annotated.copy())

                # Client-side done: env server doesn't signal termination when
                # rewards are computed externally, so derive it here.
                # An env is terminal if the server flagged it OR this is the
                # last step for an env that is still active.
                is_last_step = (step == self.n_steps - 1)
                effective_done = [
                    bool(dones[i]) or (is_last_step and not all_done[i])
                    for i in range(n)
                ]

                # Score every env (active or done) at this step
                score_tasks = [
                    self.scoring_fn(
                        step=step,
                        done=effective_done[i],
                        state=new_state_list[i] if isinstance(new_state_list[i], dict) else {},
                        frame=to_uint8_rgb(obs["image"][i]) if (obs.get("image") is not None and not all_done[i]) else None,
                        state_text=new_state_texts[i],
                        top_instruction=top_instruction,
                    )
                    for i in range(n)
                ]
                t_score = time.monotonic()
                step_scores = await asyncio.gather(*score_tasks)
                print(f"[timing iter={iteration} step={step}] scoring={time.monotonic()-t_score:.3f}s")

                for i in range(n):
                    if not all_done[i]:
                        per_env_score_sum[i] += float(step_scores[i])
                        per_env_score_count[i] += 1
                    # Mark won if score crossed threshold (server or client side)
                    if (infos and i < len(infos) and infos[i].get("won")) or float(step_scores[i]) >= WIN_THRESHOLD:
                        per_env_won[i] = True
                    if effective_done[i]:
                        all_done[i] = True

                print(
                    f"[eval iter={iteration} step={step}] "
                    f"scores={[f'{s:.3f}' for s in step_scores]} dones={dones}"
                )
                n_won = sum(per_env_won)
                step_bar.set_postfix(active=len(active), won=f"{n_won}/{n}")

            per_env_reward: list[float] = [
                (per_env_score_sum[i] / per_env_score_count[i])
                if per_env_score_count[i] > 0
                else 0.0
                for i in range(n)
            ]

            # Write videos
            write_env_videos(iter_log_dir, per_env_frames, self.video_fps)

            # Save terminal images
            if obs.get("image") is not None:
                import cv2
                for i, img in enumerate(obs["image"]):
                    rgb = to_uint8_rgb(img)
                    r_str = f"{per_env_reward[i]:.4f}".replace(".", "p")
                    img_path = os.path.join(iter_log_dir, f"env_{i:03d}_terminal_reward{r_str}.png")
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_path, bgr)
                    print(f"[eval iter={iteration}] env={i} reward={per_env_reward[i]:.4f} → {img_path}")

            # Compute per-env feedback (concurrent; supports both sync and async FeedbackFn)
            async def _call_feedback(i: int) -> str:
                result = self.feedback_fn(
                    frames=per_env_frames[i],
                    state_texts=per_env_state_texts[i],
                    top_instruction=top_instruction,
                    ll_instructions=per_env_instructions[i],
                )
                import inspect
                if inspect.isawaitable(result):
                    return await result
                return result  # type: ignore[return-value]

            t_feedback = time.monotonic()
            per_env_fb: list[str] = list(
                await asyncio.gather(*[_call_feedback(i) for i in range(n)])
            )
            print(f"[timing iter={iteration}] feedback_generation={time.monotonic()-t_feedback:.3f}s n_envs={n}")

            return EvaluationResult(
                candidate=candidate,
                iteration=iteration,
                per_env_won=per_env_won,
                per_env_reward=per_env_reward,
                per_env_feedback=per_env_fb,
                per_env_frames=per_env_frames,
                per_env_state_texts=per_env_state_texts,
                per_env_initially_won=per_env_initially_won,
            )
