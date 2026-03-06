"""
Ray-parallelized Language Table environment wrapper.

Follows the same pattern as LaMer's SokobanMultiProcessEnv:
each Ray actor holds an independent LanguageTable gym.Env instance.
"""

import copy

import ray
import numpy as np

from language_table.environments.language_table import LanguageTable
from language_table.environments.blocks import LanguageTableBlockVariants


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

    def step(self, action):
        if not self._render_obs:
            self._skip_render()
        try:
            obs, reward, done, info = self.env.step(action)
        finally:
            if not self._render_obs:
                self._restore_render()

        result_obs = self._extract_obs(obs, self.env._last_state)
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
        if env._reward_calculator is not None:
            reward_rng_state = env._reward_calculator._rng.get_state()
            reward_delay_steps = getattr(
                env._reward_calculator, "_in_reward_zone_steps", None
            )

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
        **env_kwargs,
    ):
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = num_envs
        self.num_processes = num_envs * group_n
        self.timeout = timeout

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

    def step(self, actions):
        """Step all envs in parallel. actions: list of (2,) arrays."""
        assert len(actions) == self.num_processes
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures, timeout=self.timeout)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """Reset all envs with unique seeds."""
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        seeds = np.repeat(seeds, self.group_n).tolist()

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
