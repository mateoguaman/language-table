"""
LanguageTableEnvironmentManager (Gemini variant) — open-loop action execution.

Unlike the LAVA variant (lava_env_manager.py) which runs a VLA inner loop
with per-step observation feedback, this variant:
1. Receives a natural-language action string from the outer-loop LLM
2. Calls GeminiPolicy.translate() to get a variable-length action sequence
3. Executes all actions open-loop (no observation feedback to the policy)

The number of low-level env steps per LLM turn is determined dynamically
by the length of the translated action sequence.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .frame_annotator import annotate_frames
from .state_to_text import batch_state_to_text, _decode_instruction

logger = logging.getLogger(__name__)


class LanguageTableEnvironmentManager:
    """Environment manager with Gemini-based open-loop action execution.

    Parameters
    ----------
    envs : LanguageTableMultiProcessEnv
        Vectorised Language Table environment.
    policy : GeminiPolicy
        Translates action strings into low-level action sequences.
    num_attempts : int
        Number of meta-RL attempts per episode.
    max_turns : int
        Max outer-loop LLM turns per attempt.
    do_reflection : bool
        Whether to include a reflection phase.
    max_inner_steps : int or None
        Safety cap on low-level steps per turn. None = no cap.
    """

    def __init__(
        self,
        envs,
        policy,
        num_attempts=1,
        max_turns=1,
        do_reflection=False,
        max_inner_steps=200,
        reflection_type="reflection_only",
        include_rgb=False,
        frame_subsample=5,
    ):
        self.envs = envs
        self.policy = policy
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts
        self.max_turns = max_turns
        self.do_reflection = do_reflection
        self.reflection_type = reflection_type
        self._max_inner_steps = max_inner_steps
        self._include_rgb = include_rgb
        self._frame_subsample = max(1, frame_subsample)

        # Meta-RL state
        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections: List[Dict] = [{} for _ in range(self.num_processes)]

        # Cached text observations
        self._init_text_obs: List[str] = [""] * self.num_processes
        self._last_text_obs: List[str] = [""] * self.num_processes
        self._last_infos: List[Dict] = [{} for _ in range(self.num_processes)]
        self._last_obs_list: List[Dict] = [{} for _ in range(self.num_processes)]

        # Per-env disturbance strings: None until first attempt generates them
        self._disturbances: List[Optional[str]] = [None] * self.num_processes
        self._task_strings: List[str] = [""] * self.num_processes

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
        if self._include_rgb and obs_list and "rgb" in obs_list[0]:
            return [obs["rgb"] for obs in obs_list]
        return None

    def reset(self):
        """Reset all envs and return text observations."""
        obs_list, infos = self.envs.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections = [{} for _ in range(self.num_processes)]
        self._disturbances = [None] * self.num_processes

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
        return observations, infos

    def step(self, text_actions: List[str], phase: str = "play"):
        assert phase in ("play", "reflect")
        if phase == "reflect":
            return self._handle_reflect_step(text_actions)
        return self._handle_play_step(text_actions)

    def restart(self):
        """Restart envs for the next meta-RL attempt.

        Disturbances are preserved so every attempt faces the same
        perturbation within the same episode.
        """
        obs_list, infos = self.envs.restart()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

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
        return observations, infos

    def reflect(self):
        """Return prompts for the reflection phase."""
        infos = [
            {"action_is_valid": True, "won": False}
            for _ in range(self.num_processes)
        ]
        observations = {
            "text": self._build_reflect_prompt(),
            "image": None,
            "anchor": ["reflection"] * self.num_processes,
        }
        return observations, infos

    def success_evaluator(self, **kwargs):
        """Evaluate episode success."""
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            wons = [False for _ in range(self.num_attempts)]
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item["active_masks"]:
                    info = total_infos[bs][i]
                    traj_idx = batch_item["traj_idx"]
                    if batch_item["phase"] == "play":
                        wons[traj_idx] = wons[traj_idx] or info.get("won", False)

            _won = False
            for traj_idx, won in enumerate(wons):
                _won = _won or won
                success[f"success_rate[{traj_idx}]"].append(_won)

        return {key: np.array(value) for key, value in success.items()}

    def build_text_obs(self, phase: str = "play") -> List[str]:
        if phase == "reflect":
            return self._build_reflect_prompt()
        return self._last_text_obs

    def close(self):
        self.envs.close()

    # ------------------------------------------------------------------
    # Open-loop action execution
    # ------------------------------------------------------------------

    def _handle_play_step(self, goal_strings: List[str]):
        """Translate goal strings via Gemini, then execute open-loop."""
        batch = self.num_processes

        logger.debug("Goal strings: %s", goal_strings)

        # 1. Translate each goal string into a variable-length action list
        # actions_per_env: List[List[np.ndarray]] = []
        # for i in range(batch):
        #     result = self.policy.translate(
        #         state_text=self._last_text_obs[i],
        #         action_text=goal_strings[i],
        #         disturbance=self._disturbances[i],
        #     )
        #     if self._disturbances[i] is None and result["disturbance"]:
        #         self._disturbances[i] = result["disturbance"]
        #     actions_per_env.append(result["disturbed_actions"])

        async def _translate_all():
            return await asyncio.gather(*[self.policy.translate_async(
                state_text=self._last_text_obs[i],
                action_text=goal_strings[i],
                disturbance=self._disturbances[i],
            ) for i in range(batch)])

        results = asyncio.run(_translate_all())
        for i, result in enumerate(results):
            if self._disturbances[i] is None and result["disturbance"]:
                logger.info("env %d: assigned perturbation %s", i, result["disturbance"])
                self._disturbances[i] = result["disturbance"]

        actions_per_env = [result["disturbed_actions"] for result in results]
        # use true_actions for debugging purposes
        # actions_per_env = [result["true_actions"] for result in results]

        # 2. Determine number of steps (variable, with optional safety cap)
        if not any(actions_per_env):
            num_steps = 0
        else:
            num_steps = max(len(a) for a in actions_per_env)
        if self._max_inner_steps is not None:
            num_steps = min(num_steps, self._max_inner_steps)

        logger.info(
            "Gemini play step: executing %d inner steps across %d envs",
            num_steps, batch,
        )

        # 3. Execute open-loop
        active_mask = np.ones(batch, dtype=bool)
        total_rewards = np.zeros(batch, dtype=np.float32)
        final_dones = np.zeros(batch, dtype=bool)
        current_obs = list(self._last_obs_list)
        last_infos: List[Dict] = [{} for _ in range(batch)]

        all_frames = [[] for _ in range(batch)]
        for i in range(batch):
            if "rgb" in self._last_obs_list[i]:
                all_frames[i].append(self._last_obs_list[i]["rgb"].copy())

        for step_idx in range(num_steps):
            step_actions = []
            for i in range(batch):
                if step_idx < len(actions_per_env[i]) and active_mask[i]:
                    step_actions.append(actions_per_env[i][step_idx])
                else:
                    step_actions.append(np.zeros(2, dtype=np.float32))

            # Deactivate envs that exhausted their action list
            for i in range(batch):
                if step_idx >= len(actions_per_env[i]):
                    active_mask[i] = False

            try:
                obs_list, rewards, dones, infos = self.envs.step(
                    step_actions,
                    active_mask=active_mask,
                    cached_obs=current_obs,
                    cached_infos=last_infos,
                )
            except Exception:
                logger.exception(
                    "Env step failed at inner step %d, active=%d/%d",
                    step_idx, int(active_mask.sum()), batch,
                )
                raise

            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=bool)

            total_rewards += rewards * active_mask

            if (step_idx + 1) % self._frame_subsample == 0:
                for i in range(batch):
                    if active_mask[i] and "rgb" in obs_list[i]:
                        all_frames[i].append(obs_list[i]["rgb"].copy())

            newly_done = dones & active_mask
            final_dones |= newly_done
            active_mask &= ~dones

            for i in range(batch):
                if active_mask[i] or newly_done[i]:
                    current_obs[i] = obs_list[i]
                if newly_done[i] or step_idx == num_steps - 1:
                    last_infos[i] = infos[i]

            if not active_mask.any():
                break

        final_obs = current_obs
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
            info["disturbance"] = self._disturbances[i]
            info["won"] = bool(final_dones[i])
            info["frames"] = annotate_frames(
                all_frames[i],
                traj_idx=self.curr_traj_idx,
                turn_idx=self.curr_turn_idx,
                instruction=goal_strings[i],
                task=self._task_strings[i],
            )

        self.curr_turn_idx += 1


        return observations, total_rewards / 100.0, final_dones, last_infos

    def _handle_reflect_step(self, text_actions: List[str]):
        """Store reflections from the LLM."""
        for i, reflection in enumerate(text_actions):
            self.reflections[i][self.curr_traj_idx] = reflection

        infos = [
            {"action_is_valid": True, "won": False, "is_action_valid": np.array(1.0)}
            for _ in range(self.num_processes)
        ]
        observations = {"text": "", "image": None, "anchor": ""}
        rewards = np.zeros(self.num_processes, dtype=np.float32)
        dones = np.array([False] * self.num_processes)
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reflect_prompt(self) -> List[str]:
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
