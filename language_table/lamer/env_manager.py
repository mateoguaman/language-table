"""
LanguageTableEnvironmentManager — implements the same interface as LaMer's
EnvironmentManagerBase (same method signatures) without importing from LaMer.

Follows the two-loop VLA architecture from pybullet_vla/env_manager.py:
- Outer loop: LLM proposes goal strings
- Inner loop: VLA (or random) executes low-level actions in PyBullet
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .state_to_text import batch_state_to_text

logger = logging.getLogger(__name__)


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
    ):
        self.envs = envs
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts
        self.max_turns = max_turns
        self.do_reflection = do_reflection
        self.reflection_type = reflection_type
        self.max_inner_steps = max_inner_steps

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

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _extract_images(self, obs_list):
        """Extract RGB images from obs dicts if present."""
        if obs_list and "rgb" in obs_list[0]:
            return [obs["rgb"] for obs in obs_list]
        return None

    def reset(self):
        """Reset all envs and return text observations."""
        obs_list, infos = self.envs.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections = [{} for _ in range(self.num_processes)]

        text_obs = batch_state_to_text(obs_list)
        self._init_text_obs = text_obs
        self._last_text_obs = text_obs
        self._last_infos = infos

        observations = {
            "text": text_obs,
            "image": self._extract_images(obs_list),
            "anchor": text_obs,
        }
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
        obs_list, infos = self.envs.restart()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        text_obs = batch_state_to_text(obs_list)
        self._last_text_obs = text_obs
        self._last_infos = infos

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

        reflect_obs = self._build_reflect_prompt()

        observations = {
            "text": reflect_obs,
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
        """Build text observations."""
        if phase == "reflect":
            return self._build_reflect_prompt()
        return self._last_text_obs

    def close(self):
        """Close all envs."""
        self.envs.close()

    # ------------------------------------------------------------------
    # VLA inner loop
    # ------------------------------------------------------------------

    def _handle_play_step(self, goal_strings: List[str]):
        """Run the VLA inner loop for a full episode.

        Currently uses random actions. Replace with real VLA inference later.
        """
        batch = self.num_processes
        active_mask = np.ones(batch, dtype=bool)
        total_rewards = np.zeros(batch, dtype=np.float32)
        final_dones = np.zeros(batch, dtype=bool)
        last_obs = None
        last_infos = [{} for _ in range(batch)]

        for inner_step in range(self.max_inner_steps):
            # Generate random actions for all envs (placeholder for VLA)
            actions = [
                np.random.uniform(self._action_low, self._action_high, size=(2,)).astype(np.float32)
                for _ in range(batch)
            ]

            # Zero out actions for inactive envs
            for i in range(batch):
                if not active_mask[i]:
                    actions[i] = np.zeros(2, dtype=np.float32)

            obs_list, rewards, dones, infos = self.envs.step(actions)

            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=bool)

            total_rewards += rewards * active_mask

            newly_done = dones & active_mask
            final_dones |= newly_done
            active_mask &= ~dones

            last_obs = obs_list
            for i in range(batch):
                if newly_done[i] or inner_step == self.max_inner_steps - 1:
                    last_infos[i] = infos[i]

            if not active_mask.any():
                break

        # Mark timed-out envs as done
        final_dones |= active_mask
        active_mask[:] = False

        final_obs = last_obs if last_obs is not None else [{}] * batch
        text_obs = batch_state_to_text(final_obs)
        self._last_text_obs = text_obs
        self._last_infos = last_infos
        self.curr_turn_idx += 1

        observations = {
            "text": text_obs,
            "image": self._extract_images(final_obs),
            "anchor": text_obs,
        }

        for info in last_infos:
            info["is_action_valid"] = np.array(1.0)

        return observations, total_rewards, final_dones, last_infos

    def _handle_reflect_step(self, text_actions: List[str]):
        """Store reflections from the LLM."""
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
