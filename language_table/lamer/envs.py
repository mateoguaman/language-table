"""
Ray-parallelized Language Table environment wrapper.

Follows the same pattern as LaMer's SokobanMultiProcessEnv:
each Ray actor holds an independent LanguageTable gym.Env instance.
"""

import copy
import logging

import ray
import numpy as np

from language_table.environments.language_table import LanguageTable
from language_table.environments.blocks import LanguageTableBlockVariants

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.01)
class LanguageTableWorker:
    """Ray remote actor wrapping a single LanguageTable env.

    Parameters
    ----------
    render_obs : bool
        If True (default), _compute_state() renders RGB via getCameraImage()
        on every step/reset, and the RGB is included in the returned obs dict.
        Required when the inner-loop policy is a VLA that consumes images.
        Set to False to skip rendering entirely for maximum throughput
        (e.g. random-action baselines or text-only policies).
    return_full_state : bool
        If True (default), returns the full state dict (with block positions).
        If False, returns the filtered observation dict (4 keys).
    """

    def __init__(self, block_mode, reward_factory_cls=None, seed=None,
                 return_full_state=True, render_obs=True,
                 **env_kwargs):
        if isinstance(block_mode, str):
            block_mode = LanguageTableBlockVariants(block_mode)
        self.seed_val = seed
        self._return_full_state = return_full_state
        self._render_obs = render_obs
        self.env = LanguageTable(
            block_mode=block_mode,
            reward_factory=reward_factory_cls,
            seed=seed,
            **env_kwargs,
        )
        self.env._render_text_in_image = False
        # Save original render function so we can toggle it on/off
        self._orig_render = self.env._render_camera
        self._noop_render = lambda *a, **kw: None
        # Snapshot saved after each reset() for exact restart()
        self._restart_bullet_state = None
        self._restart_python_state = None

    def _skip_render(self):
        """Disable RGB rendering temporarily."""
        self.env._render_camera = self._noop_render

    def _restore_render(self):
        """Restore original render function."""
        self.env._render_camera = self._orig_render

    def _extract_obs(self, obs_or_state, state):
        """Pick the right observation to return based on config."""
        if self._return_full_state:
            result = dict(state)
        else:
            result = dict(obs_or_state)
        if not self._render_obs:
            result.pop("rgb", None)
        return result

    def step(self, action, skip_rgb: bool = False):
        """Step the environment.

        Parameters
        ----------
        action : array-like
            2-D end-effector delta action.
        skip_rgb : bool
            When True, suppress the camera render for this step (obs will not
            contain an 'rgb' key).  Used by the chunked-action path in the
            manager so that steps where the policy is draining a buffered chunk
            do not pay the camera cost.  Ignored when render_obs=False (already
            no render happening).
        """
        suppress = not self._render_obs or skip_rgb
        if suppress:
            self._skip_render()
        try:
            obs, reward, done, info = self.env.step(action)
        except Exception:
            logger.exception(
                "LanguageTableWorker.step failed: seed=%s instruction=%r action=%s "
                "target_pose=%s last_effector=%s",
                self.seed_val,
                getattr(self.env, "_instruction_str", None),
                np.asarray(action).tolist(),
                getattr(getattr(self.env, "_target_effector_pose", None), "translation", None),
                None if getattr(self.env, "_last_state", None) is None else
                np.asarray(self.env._last_state.get("effector_translation", [])).tolist(),
            )
            raise
        finally:
            if suppress:
                self._restore_render()

        info["low_level_seed"] = self.seed_val
        result_obs = self._extract_obs(obs, self.env._last_state)
        if skip_rgb:
            result_obs.pop("rgb", None)
        return result_obs, reward, done, info

    def reset(self, seed=None):
        if seed is not None:
            self.seed_val = seed
            if self.env._reward_calculator is not None:
                self.env.seed(seed)
            else:
                self.env._rng = np.random.RandomState(seed=seed)

        if not self._render_obs:
            self._skip_render()
        try:
            self.env.reset()
        finally:
            if not self._render_obs:
                self._restore_render()

        obs = self._extract_obs(None, self.env._last_state)
        self._save_restart_snapshot(obs)
        return obs, {}

    def render(self):
        return self.env.render(mode="rgb_array")

    def restart(self):
        """Restore the exact simulation state from the last reset().

        Uses PyBullet saveState/restoreState for physics-level fidelity
        plus cached Python-level task state, so the restored state is
        bit-for-bit identical to what reset() produced.
        """
        if self._restart_bullet_state is None:
            # No snapshot yet — fall back to fresh reset
            return self.reset(seed=self.seed_val)

        env = self.env

        # Restore PyBullet physics state (joints, contacts, velocities, poses)
        env._pybullet_client.restoreState(self._restart_bullet_state)

        # Restore Python-level task state
        ps = self._restart_python_state
        env._target_effector_pose = ps["target_effector_pose"]
        env._robot.set_target_effector_pose(env._target_effector_pose)
        env._blocks_on_table = ps["blocks_on_table"]
        env._instruction = ps["instruction"]
        env._instruction_str = ps["instruction_str"]
        env._start_block = ps["start_block"]
        env._oracle_target_block = ps["oracle_target_block"]
        env._oracle_target_translation = ps["oracle_target_translation"]
        env._target_absolute_location = ps["target_absolute_location"]
        env._target_relative_location = ps["target_relative_location"]

        # Restore reward calculator internal state
        if env._reward_calculator is not None and ps["reward_rng_state"] is not None:
            env._reward_calculator._rng.set_state(ps["reward_rng_state"])
            if ps.get("reward_delay_steps") is not None:
                env._reward_calculator._in_reward_zone_steps = ps["reward_delay_steps"]

        # Compute state after restoring (no _last_state available here
        # since we didn't go through env.step/reset).
        if not self._render_obs:
            self._skip_render()
        try:
            state = self.env._compute_state(request_task_update=False)
        finally:
            if not self._render_obs:
                self._restore_render()

        if self._return_full_state:
            obs = dict(state)
        else:
            obs = self.env._compute_observation(state=state)
        if not self._render_obs:
            obs.pop("rgb", None)
        return obs, {}

    def get_pybullet_state(self):
        return self.env.get_pybullet_state()

    def get_object_positions(self):
        """Return gripper and block positions as plain Python values."""
        env = self.env
        gripper_pose = env._robot.forward_kinematics()
        positions = {
            "gripper": np.asarray(gripper_pose.translation, dtype=np.float64).tolist(),
            "gripper_target": np.asarray(
                env._target_effector_pose.translation, dtype=np.float64
            ).tolist(),
            "blocks": {},
            "blocks_on_table": list(getattr(env, "_blocks_on_table", [])),
            "seed": self.seed_val,
            "instruction": getattr(env, "_instruction_str", None),
        }

        for block_name, block_id in env._block_to_pybullet_id.items():
            pos, _ = env._pybullet_client.getBasePositionAndOrientation(block_id)
            positions["blocks"][block_name] = np.asarray(pos, dtype=np.float64).tolist()

        return positions

    def _save_restart_snapshot(self, obs):
        """Capture full simulation + task state for exact restart."""
        env = self.env

        # Remove any previous snapshot to free PyBullet memory
        if self._restart_bullet_state is not None:
            try:
                env._pybullet_client.removeState(self._restart_bullet_state)
            except Exception:
                pass

        self._restart_bullet_state = env._pybullet_client.saveState()

        reward_rng_state = None
        reward_delay_steps = None
        reward_state = None
        if env._reward_calculator is not None:
            reward_rng_state = env._reward_calculator._rng.get_state()
            reward_delay_steps = getattr(
                env._reward_calculator, "_in_reward_zone_steps", None
            )
            reward_state = copy.deepcopy(env._reward_calculator.__dict__)

        self._restart_python_state = {
            "target_effector_pose": copy.deepcopy(env._target_effector_pose),
            "blocks_on_table": list(env._blocks_on_table),
            "instruction": env._instruction.copy() if env._instruction is not None else None,
            "instruction_str": env._instruction_str,
            "start_block": env._start_block,
            "oracle_target_block": env._oracle_target_block,
            "oracle_target_translation": (
                env._oracle_target_translation.copy()
                if env._oracle_target_translation is not None
                else None
            ),
            "target_absolute_location": env._target_absolute_location,
            "target_relative_location": env._target_relative_location,
            "reward_rng_state": reward_rng_state,
            "reward_delay_steps": reward_delay_steps,
            "reward_state": reward_state,
        }


class LanguageTableMultiProcessEnv:
    """
    Ray-based vectorised wrapper for LanguageTable environments.
    Each Ray actor holds an independent LanguageTable instance.
    """

    def __init__(
        self,
        num_envs,
        block_mode="BLOCK_4",
        reward_factory_cls=None,
        seed=0,
        group_n=1,
        is_train=True,
        num_cpus=None,
        timeout=None,
        return_full_state=True,
        render_obs=True,
        max_seeds: int = 16,
        **env_kwargs,
    ):
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = num_envs
        self.num_processes = num_envs * group_n
        self.timeout = timeout
        self.block_mode = block_mode if isinstance(block_mode, str) else block_mode.value
        self.max_seeds = max_seeds

        np.random.seed(seed)

        WorkerCls = LanguageTableWorker
        if num_cpus is not None:
            WorkerCls = LanguageTableWorker.options(num_cpus=num_cpus)

        self.workers = [
            WorkerCls.remote(
                block_mode, reward_factory_cls, seed=seed + i,
                return_full_state=return_full_state,
                render_obs=render_obs,
                **env_kwargs,
            )
            for i in range(self.num_processes)
        ]

    def step(self, actions, active_mask=None, cached_obs=None, cached_infos=None,
             skip_rgb_mask=None):
        """Step active envs in parallel.

        Parameters
        ----------
        actions : list[np.ndarray]
            One action per environment.
        active_mask : np.ndarray | None
            Optional bool mask of envs to step. Inactive envs are returned
            from cache without sending another step to their workers.
        cached_obs : list[dict] | None
            Previous observations to reuse for inactive envs.
        cached_infos : list[dict] | None
            Previous infos to reuse for inactive envs.
        skip_rgb_mask : np.ndarray[bool] | None
            Per-env flag; when True for an active env the camera render is
            suppressed and obs will not contain 'rgb'.  Used by the chunked-
            action loop in the manager.
        """
        assert len(actions) == self.num_processes

        if skip_rgb_mask is None:
            skip_rgb_mask = np.zeros(self.num_processes, dtype=bool)
        else:
            skip_rgb_mask = np.asarray(skip_rgb_mask, dtype=bool)

        if active_mask is None:
            futures = [
                w.step.remote(a, skip_rgb=bool(s))
                for w, a, s in zip(self.workers, actions, skip_rgb_mask)
            ]
            results = ray.get(futures, timeout=self.timeout)
            obs_list, reward_list, done_list, info_list = [], [], [], []
            for obs, reward, done, info in results:
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
            return obs_list, reward_list, done_list, info_list

        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape != (self.num_processes,):
            raise ValueError(
                f"active_mask shape must be ({self.num_processes},), got {active_mask.shape}"
            )

        obs_list = list(cached_obs) if cached_obs is not None else [{} for _ in range(self.num_processes)]
        reward_list = [0.0 for _ in range(self.num_processes)]
        done_list = [bool(not is_active) for is_active in active_mask]
        info_list = (
            [dict(info) for info in cached_infos]
            if cached_infos is not None
            else [{} for _ in range(self.num_processes)]
        )

        active_indices = np.flatnonzero(active_mask).tolist()
        if not active_indices:
            for idx in range(self.num_processes):
                info_list[idx]["skipped_step_after_done"] = True
            return obs_list, reward_list, done_list, info_list

        futures = [
            self.workers[idx].step.remote(actions[idx], skip_rgb=bool(skip_rgb_mask[idx]))
            for idx in active_indices
        ]
        results = ray.get(futures, timeout=self.timeout)
        for idx, (obs, reward, done, info) in zip(active_indices, results):
            obs_list[idx] = obs
            reward_list[idx] = reward
            done_list[idx] = done
            info_list[idx] = info

        for idx in range(self.num_processes):
            if not active_mask[idx]:
                info_list[idx]["skipped_step_after_done"] = True
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """Reset all envs with unique seeds."""
        if self.is_train:
            seeds = np.random.randint(0, self.max_seeds, size=self.env_num)
        else:
            seeds = np.random.randint(0, self.max_seeds, size=self.env_num)

        seeds = np.repeat(seeds, self.group_n).tolist()
        # DEBUG: set all seeds to 0
        # seeds = np.zeros(self.num_processes, dtype=int)
        
        futures = [w.reset.remote(seed=s) for w, s in zip(self.workers, seeds)]
        results = ray.get(futures, timeout=self.timeout)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, env_idx=None):
        """Render envs. Returns list of RGB arrays or single array."""
        if env_idx is not None:
            return ray.get(self.workers[env_idx].render.remote(), timeout=self.timeout)
        futures = [w.render.remote() for w in self.workers]
        return ray.get(futures, timeout=self.timeout)

    def get_object_positions(self):
        """Return gripper and block positions for every worker env."""
        futures = [w.get_object_positions.remote() for w in self.workers]
        return ray.get(futures, timeout=self.timeout)

    def restart(self):
        """Restart all envs (meta-RL attempt loop)."""
        futures = [w.restart.remote() for w in self.workers]
        results = ray.get(futures, timeout=self.timeout)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        for w in self.workers:
            ray.kill(w)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
