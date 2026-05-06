"""
LanguageTableEnvironmentManager — implements the same interface as LaMer's
EnvironmentManagerBase (same method signatures) without importing from LaMer.

Follows the two-loop VLA architecture from pybullet_vla/env_manager.py:
- Outer loop: LLM proposes goal strings
- Inner loop: SmolVLA (or random) executes low-level actions in PyBullet
"""

import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .frame_annotator import annotate_frames
from .state_to_text import batch_state_to_text, _decode_instruction

logger = logging.getLogger(__name__)

def clean_string(s: str) -> str:

    if s == "!":
        # NOTE: environment has done=True so first qwen token is passed ("!")
        return "inactive environment"

    cleaned = re.sub(r"[^a-z ,]", "", s.lower()).strip()
    
    if not cleaned:
        # NOTE: empty instruction after cleaning
        return "empty instruction"
    
    return cleaned


class LanguageTableEnvironmentManager:
    """Environment manager wrapping Language Table envs with a VLA inner loop.

    Parameters
    ----------
    envs : LanguageTableMultiProcessEnv
        Vectorised Language Table environment.
    num_attempts : int
        Number of meta-RL attempts per episode.
    max_turns : int
        Max outer-loop turns per attempt.
    do_reflection : bool
        Whether to include a reflection phase.
    max_inner_steps : int
        Number of inner-loop VLA steps per outer step.
    """

    def __init__(
        self,
        envs,
        num_attempts=1,
        max_turns=1,
        do_reflection=False,
        max_inner_steps=100,
        reflection_type="reflection_only",
        vla_policy=None,
        include_rgb=False,
        split="train",
        frame_subsample=5,
        n_steps=1,
        benchmark_timing=False,
        benchmark_trace_inner_steps=False,
    ):
        self.envs = envs
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts
        self.max_turns = max_turns
        self.do_reflection = do_reflection
        self.reflection_type = reflection_type
        self.max_inner_steps = max_inner_steps
        self.vla = vla_policy
        self._include_rgb = include_rgb
        self.split = split
        self._frame_subsample = max(1, frame_subsample)
        self.n_steps = n_steps
        self._benchmark_timing = benchmark_timing
        self._benchmark_trace_inner_steps = benchmark_trace_inner_steps

        # Action space bounds (Language Table uses [-0.1, 0.1]^2)
        self._action_low = -0.1
        self._action_high = 0.1

        # Meta-RL state
        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections: List[Dict] = [{} for _ in range(self.num_processes)]

        # Cached text observations
        self._init_text_obs: List[str] = [""] * self.num_processes
        self._last_text_obs: List[str] = [""] * self.num_processes
        self._last_infos: List[Dict] = [{} for _ in range(self.num_processes)]
        # Raw obs dicts cached for SmolVLA (needs RGB at the start of each inner loop)
        self._last_obs_list: List[Dict] = [{} for _ in range(self.num_processes)]
        self._last_goal_strings: List[str] = [""] * self.num_processes
        self._task_strings: List[str] = [""] * self.num_processes

    def _log_step_failure(
        self,
        inner_step: int,
        goal_strings: List[str],
        actions: List[np.ndarray],
        active_mask: np.ndarray,
    ) -> None:
        """Emit the last step context before re-raising an env failure."""
        active_indices = np.flatnonzero(active_mask).tolist()
        sample_context = []
        for env_idx in active_indices[:5]:
            action = np.asarray(actions[env_idx], dtype=np.float32)
            sample_context.append({
                "env_idx": env_idx,
                "goal": goal_strings[env_idx][:160],
                "action": action.tolist(),
                "action_norm": float(np.linalg.norm(action)),
            })

        logger.exception(
            "Language Table play step failed: inner_step=%d active=%d/%d "
            "active_indices_sample=%s context_sample=%s",
            inner_step,
            int(active_mask.sum()),
            len(active_mask),
            active_indices[:10],
            sample_context,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _extract_task_strings(self, obs_list):
        """Decode the environment task instruction from each observation."""
        tasks = []
        for obs in obs_list:
            instr = obs.get("instruction")
            tasks.append(_decode_instruction(instr) if instr is not None else "")
        return tasks

    def _extract_images(self, obs_list):
        """Extract RGB images from obs dicts for the client response.

        Only includes images if include_rgb=True (i.e. the outer LLM client
        needs images). The VLA accesses RGB directly from the obs dicts
        without going through this method.
        """
        if self._include_rgb and obs_list and "rgb" in obs_list[0]:
            return [obs["rgb"] for obs in obs_list]
        return None

    def _attach_benchmark_payload(self, infos, payload):
        """Attach a benchmark payload to the first info dict only."""
        if not self._benchmark_timing or not infos:
            return infos
        first_info = dict(infos[0])
        first_info["benchmark_env"] = payload
        infos = list(infos)
        infos[0] = first_info
        return infos

    def _get_last_predict_stats(
        self,
        predict_elapsed_s: float,
        n_active: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Return policy timing stats, with a SmolVLA-compatible fallback."""
        get_stats = getattr(self.vla, "get_last_predict_stats", None)
        if callable(get_stats):
            stats = dict(get_stats())
        else:
            stats = {}

        stats.setdefault("kind", "vla_predict")
        stats.setdefault("build_batch_s", 0.0)
        stats.setdefault("gather_active_rgb_s", 0.0)
        stats.setdefault("frame_update_s", 0.0)
        stats.setdefault("token_batch_s", 0.0)
        stats.setdefault("instruction_to_device_s", 0.0)
        stats.setdefault("forward_s", predict_elapsed_s)
        stats.setdefault("total_s", predict_elapsed_s)
        stats.setdefault("n_active", n_active)
        stats.setdefault("batch_size", batch_size)
        stats.setdefault("preprocess_mode", self.vla.__class__.__name__)
        return stats

    def reset(self):
        """Reset all envs and return text observations."""
        reset_t0 = time.perf_counter()
        obs_list, infos = self.envs.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections = [{} for _ in range(self.num_processes)]

        if self.vla is not None:
            self.vla.reset(num_envs=self.num_processes)

        self._last_obs_list = obs_list
        text_obs = batch_state_to_text(obs_list)
        self._init_text_obs = text_obs
        self._last_text_obs = text_obs
        self._last_infos = infos
        self._task_strings = self._extract_task_strings(obs_list)

        observations = {
            "text": text_obs,
            "image": self._extract_images(obs_list),
            "anchor": text_obs,
        }
        infos = self._attach_benchmark_payload(
            infos,
            {
                "kind": "env_server_call",
                "method": "reset",
                "split": self.split,
                "server_elapsed_s": float(time.perf_counter() - reset_t0),
                "num_processes": int(self.num_processes),
            },
        )
        return observations, infos

    def step(self, text_actions: List[str], phase: str = "play"):
        """Execute a step in the meta-RL loop.

        phase="play": text_actions are goal strings → run VLA inner loop.
        phase="reflect": text_actions are reflections → store and return dummy obs.
        """
        assert phase in ("play", "reflect")

        if phase == "reflect":
            return self._handle_reflect_step(text_actions)

        return self._handle_play_step(text_actions)

    def restart(self):
        """Restart envs for the next meta-RL attempt."""
        restart_t0 = time.perf_counter()
        obs_list, infos = self.envs.restart()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        if self.vla is not None:
            self.vla.reset(num_envs=self.num_processes)

        self._last_obs_list = obs_list
        text_obs = batch_state_to_text(obs_list)
        self._last_text_obs = text_obs
        self._last_infos = infos
        self._task_strings = self._extract_task_strings(obs_list)

        observations = {
            "text": text_obs,
            "image": self._extract_images(obs_list),
            "anchor": text_obs,
        }
        infos = self._attach_benchmark_payload(
            infos,
            {
                "kind": "env_server_call",
                "method": "restart",
                "split": self.split,
                "server_elapsed_s": float(time.perf_counter() - restart_t0),
                "num_processes": int(self.num_processes),
            },
        )
        return observations, infos

    def reflect(self):
        """Return prompts for the reflection phase."""
        reflect_t0 = time.perf_counter()
        infos = [
            {"action_is_valid": True, "won": False}
            for _ in range(self.num_processes)
        ]

        reflect_obs = self._build_reflect_prompt()

        observations = {
            "text": reflect_obs,
            "image": None,
            "anchor": ["reflection"] * self.num_processes,
        }
        infos = self._attach_benchmark_payload(
            infos,
            {
                "kind": "env_server_call",
                "method": "reflect",
                "split": self.split,
                "server_elapsed_s": float(time.perf_counter() - reflect_t0),
                "num_processes": int(self.num_processes),
            },
        )
        return observations, infos

    def success_evaluator(self, **kwargs):
        """Evaluate episode success."""
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            wons = [False for _ in range(self.num_attempts)]
            last_reward_per_attempt: Dict[int, float] = {}
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item["active_masks"]:
                    info = total_infos[bs][i]
                    traj_idx = batch_item["traj_idx"]
                    if batch_item["phase"] == "play":
                        wons[traj_idx] = wons[traj_idx] or info.get("won", False)
                        if traj_idx not in last_reward_per_attempt:
                            last_reward_per_attempt[traj_idx] = info.get(
                                "total_reward", 0.0)

            best_reward = max(last_reward_per_attempt.values()) if last_reward_per_attempt else 0.0

            _won = False
            for traj_idx, won in enumerate(wons):
                _won = _won or won
                success[f"success_rate[{traj_idx}]"].append(_won)

            if self.n_steps > 1:
                completion = min(best_reward / 100.0, 1.0)
                success["partial_completion"].append(completion)

        return {key: np.array(value) for key, value in success.items()}

    def build_text_obs(self, phase: str = "play") -> List[str]:
        """Build text observations."""
        if phase == "reflect":
            return self._build_reflect_prompt()
        return self._last_text_obs

    def close(self):
        """Close all envs."""
        if self.vla is not None and hasattr(self.vla, "close"):
            self.vla.close()
        self.envs.close()

    # ------------------------------------------------------------------
    # SmolVLA inner loop
    # ------------------------------------------------------------------

    def _handle_play_step(self, goal_strings: List[str]):
        """Run the SmolVLA inner loop for a full episode.

        When a SmolVLA policy is provided, it receives the LLM's goal strings
        as language conditioning and produces actions from RGB observations.
        Otherwise, falls back to random actions.
        """
        
        play_t0 = time.perf_counter()
        raw_strings = list(goal_strings)
        for i, raw in enumerate(raw_strings):   
            cleaned = clean_string(raw)
            goal_strings[i] = cleaned

        batch = self.num_processes
        active_mask = np.ones(batch, dtype=bool)
        last_rewards = np.zeros(batch, dtype=np.float32)
        final_dones = np.zeros(batch, dtype=bool)
        last_obs = None
        last_infos = [{} for _ in range(batch)]
        self._last_goal_strings = list(goal_strings)

        all_frames = [[] for _ in range(batch)]
        all_frame_rewards = [[] for _ in range(batch)]
        for i in range(batch):
            if "rgb" in self._last_obs_list[i]:
                all_frames[i].append(self._last_obs_list[i]["rgb"].copy())
                all_frame_rewards[i].append(0.0)

        # Seed SmolVLA with the obs cached from reset()/restart().
        # After the first env.step(), we use its returned obs instead — just
        # like the original eval loop: obs = env.reset(); while: action =
        # policy(obs); obs = env.step(action).
        obs_list_for_vla = self._last_obs_list if self.vla is not None else None
        inner_step_records = []

        for inner_step in range(self.max_inner_steps):
            n_active_before = int(active_mask.sum())
            if self.vla is not None:
                # Batched SmolVLA inference: the LLM's goal strings condition
                # the policy, and RGB/state observations are sent to the
                # LeRobot policy server in a single request when supported.
                predict_t0 = time.perf_counter()
                actions = self.vla.predict(goal_strings, obs_list_for_vla, active_mask)
                vla_stats = self._get_last_predict_stats(
                    predict_elapsed_s=time.perf_counter() - predict_t0,
                    n_active=n_active_before,
                    batch_size=batch,
                )
            else:
                # Random actions fallback (no VLA loaded)
                if inner_step == 0:
                    logger.warning(
                        "No VLA policy loaded — using random actions for the "
                        "inner loop. Set --vla_checkpoint to use the SmolVLA policy."
                    )
                actions = [
                    np.random.uniform(self._action_low, self._action_high, size=(2,)).astype(np.float32)
                    for _ in range(batch)
                ]
                for i in range(batch):
                    if not active_mask[i]:
                        actions[i] = np.zeros(2, dtype=np.float32)
                vla_stats = {
                    "kind": "vla_predict",
                    "build_batch_s": 0.0,
                    "gather_active_rgb_s": 0.0,
                    "frame_update_s": 0.0,
                    "token_batch_s": 0.0,
                    "instruction_to_device_s": 0.0,
                    "forward_s": 0.0,
                    "total_s": 0.0,
                    "n_active": n_active_before,
                    "batch_size": batch,
                    "preprocess_mode": "random",
                }

            action_array = np.asarray(actions, dtype=np.float32)
            invalid_action_mask = ~np.isfinite(action_array).all(axis=1)
            if invalid_action_mask.any():
                bad_indices = np.flatnonzero(invalid_action_mask).tolist()
                logger.error(
                    "Invalid action batch at inner_step=%d bad_envs=%s goals=%s actions=%s",
                    inner_step,
                    bad_indices,
                    [goal_strings[i][:160] for i in bad_indices[:5]],
                    [action_array[i].tolist() for i in bad_indices[:5]],
                )

            cached_obs = (
                obs_list_for_vla
                if obs_list_for_vla is not None
                else self._last_obs_list
            )
            try:
                obs_list, rewards, dones, infos = self.envs.step(
                    actions,
                    active_mask=active_mask,
                    cached_obs=cached_obs,
                    cached_infos=last_infos,
                )
            except Exception:
                self._log_step_failure(inner_step, goal_strings, actions, active_mask)
                raise
            env_step_stats = self.envs.get_last_step_stats()

            # Feed fresh observations back to SmolVLA for the next iteration,
            # matching the original eval loop: action = policy(obs); obs = env.step(action)
            if self.vla is not None:
                obs_list_for_vla = obs_list

            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=bool)

            for i in range(batch):
                if active_mask[i]:
                    last_rewards[i] = rewards[i]

            if (inner_step + 1) % self._frame_subsample == 0:
                for i in range(batch):
                    if active_mask[i] and "rgb" in obs_list[i]:
                        all_frames[i].append(obs_list[i]["rgb"].copy())
                        all_frame_rewards[i].append(float(last_rewards[i]))

            newly_done = dones & active_mask
            final_dones |= newly_done
            active_mask &= ~dones
            inner_step_records.append(
                {
                    "inner_step": int(inner_step),
                    "n_active_before": int(n_active_before),
                    "n_active_after": int(active_mask.sum()),
                    "n_newly_done": int(newly_done.sum()),
                    "vla_total_s": float(vla_stats.get("total_s", 0.0)),
                    "vla_build_batch_s": float(vla_stats.get("build_batch_s", 0.0)),
                    "vla_gather_active_rgb_s": float(vla_stats.get("gather_active_rgb_s", 0.0)),
                    "vla_frame_update_s": float(vla_stats.get("frame_update_s", 0.0)),
                    "vla_token_batch_s": float(vla_stats.get("token_batch_s", 0.0)),
                    "vla_instruction_to_device_s": float(vla_stats.get("instruction_to_device_s", 0.0)),
                    "vla_forward_s": float(vla_stats.get("forward_s", 0.0)),
                    "env_step_total_s": float(env_step_stats.get("total_s", 0.0)),
                    "ray_dispatch_s": float(env_step_stats.get("dispatch_s", 0.0)),
                    "ray_collect_s": float(env_step_stats.get("collect_s", 0.0)),
                    "ray_unpack_s": float(env_step_stats.get("unpack_s", 0.0)),
                }
            )
            if newly_done.any():
                logger.info(
                    "Language Table inner step %d completed envs=%s remaining_active=%d",
                    inner_step,
                    np.flatnonzero(newly_done).tolist(),
                    int(active_mask.sum()),
                )

            last_obs = obs_list
            for i in range(batch):
                if newly_done[i] or inner_step == self.max_inner_steps - 1:
                    last_infos[i] = infos[i]

            if not active_mask.any():
                break

        timed_out = active_mask.copy()
        # if timed_out.any():
        #     logger.info(
        #         "Language Table inner loop timed out after %d steps: "
        #         "envs=%s (these did NOT win)",
        #         self.max_inner_steps,
        #         np.flatnonzero(timed_out).tolist(),
        #     )

        final_obs = last_obs if last_obs is not None else [{}] * batch
        text_obs = batch_state_to_text(final_obs)
        self._last_text_obs = text_obs
        self._last_obs_list = final_obs
        self._last_infos = last_infos

        observations = {
            "text": text_obs,
            "image": self._extract_images(final_obs),
            "anchor": text_obs,
        }

        for i, info in enumerate(last_infos):
            info["is_action_valid"] = np.array(1.0)
            info["won"] = bool(final_dones[i])
            info["total_reward"] = float(last_rewards[i])
            info["frames"] = annotate_frames(
                all_frames[i],
                traj_idx=self.curr_traj_idx,
                turn_idx=self.curr_turn_idx,
                instruction=goal_strings[i],
                task=self._task_strings[i],
                rewards=all_frame_rewards[i],
            )
            info["language_instruction"] = goal_strings[i]

        benchmark_payload = {
            "kind": "env_server_play_step",
            "method": "step",
            "split": self.split,
            "server_elapsed_s": float(time.perf_counter() - play_t0),
            "traj_idx": int(self.curr_traj_idx),
            "turn_idx": int(self.curr_turn_idx),
            "max_inner_steps": int(self.max_inner_steps),
            "inner_steps_completed": int(len(inner_step_records)),
            "timed_out_envs": int(timed_out.sum()),
            "n_processes": int(batch),
            "vla_total_s": float(sum(r["vla_total_s"] for r in inner_step_records)),
            "env_step_total_s": float(sum(r["env_step_total_s"] for r in inner_step_records)),
            "ray_dispatch_s": float(sum(r["ray_dispatch_s"] for r in inner_step_records)),
            "ray_collect_s": float(sum(r["ray_collect_s"] for r in inner_step_records)),
            "ray_unpack_s": float(sum(r["ray_unpack_s"] for r in inner_step_records)),
        }
        if self._benchmark_trace_inner_steps:
            benchmark_payload["inner_steps"] = inner_step_records
        last_infos = self._attach_benchmark_payload(last_infos, benchmark_payload)

        self.curr_turn_idx += 1

        return observations, last_rewards / 100.0, final_dones, last_infos

    def _handle_reflect_step(self, text_actions: List[str]):
        """Store reflections from the LLM."""
        reflect_t0 = time.perf_counter()
        for i, reflection in enumerate(text_actions):
            self.reflections[i][self.curr_traj_idx] = reflection

        infos = [
            {"action_is_valid": True, "won": False, "is_action_valid": np.array(1.0)}
            for _ in range(self.num_processes)
        ]
        observations = {
            "text": "",
            "image": None,
            "anchor": "",
        }
        rewards = np.zeros(self.num_processes, dtype=np.float32)
        dones = np.array([False] * self.num_processes)
        infos = self._attach_benchmark_payload(
            infos,
            {
                "kind": "env_server_call",
                "method": "step_reflect",
                "split": self.split,
                "server_elapsed_s": float(time.perf_counter() - reflect_t0),
                "num_processes": int(self.num_processes),
            },
        )
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reflect_prompt(self) -> List[str]:
        """Build text prompts for the reflection phase."""
        prompts = []
        for i in range(self.num_processes):
            parts = [f"Initial state:\n{self._init_text_obs[i]}"]
            parts.append(f"\nEpisode outcome:\n{self._last_text_obs[i]}")

            for traj_idx in sorted(self.reflections[i]):
                parts.append(
                    f"\nReflection (attempt {traj_idx}):\n"
                    f"{self.reflections[i][traj_idx]}"
                )

            parts.append(
                "\nBased on the episode outcome, reflect on what went wrong "
                "and propose a new goal for the next attempt."
            )
            prompts.append("\n".join(parts))
        return prompts
