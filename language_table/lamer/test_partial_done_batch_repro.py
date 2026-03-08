#!/usr/bin/env python3
"""Stress test the old partial-done batching pattern with real envs.

This intentionally reproduces the pre-fix behavior:
- active envs continue with oracle actions
- finished envs are still stepped with zero actions

If simple batched post-done stepping is enough to trigger the quaternion bug,
this should surface it without the outer LLM harness.
"""

import argparse
import sys
import types

import numpy as np
from tf_agents.environments import gym_wrapper

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block

try:
    import imageio as _imageio  # noqa: F401
except ModuleNotFoundError:
    sys.modules["imageio"] = types.SimpleNamespace(imwrite=lambda *args, **kwargs: None)

from language_table.environments.oracles import push_oracle_rrt_slowdown


def quat_info(base_env):
    effector_state = base_env._pybullet_client.getLinkState(
        base_env._robot.xarm,
        base_env._robot.effector_link,
        computeForwardKinematics=1,
    )
    quat = np.asarray(effector_state[1], dtype=np.float64)
    return quat, float(np.linalg.norm(quat))


def make_env(seed):
    base_env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        seed=seed,
    )
    if not hasattr(base_env, "get_control_frequency"):
        base_env.get_control_frequency = lambda: base_env._control_frequency
    wrapped_env = gym_wrapper.GymWrapper(base_env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
        wrapped_env,
        use_ee_planner=True,
    )
    return base_env, wrapped_env, oracle


def reset_with_plan(wrapped_env, base_env, oracle, max_resets):
    for _ in range(max_resets):
        ts = wrapped_env.reset()
        raw_state = base_env.compute_state()
        if oracle.get_plan(raw_state):
            return ts
    raise RuntimeError(f"failed to find valid oracle plan after {max_resets} resets")


def run_trial(num_envs, seed, max_steps, max_resets):
    env_triplets = [make_env(seed + i) for i in range(num_envs)]
    done_mask = np.zeros(num_envs, dtype=bool)
    timesteps = []
    first_done_step = [None] * num_envs
    zero_action_counts = np.zeros(num_envs, dtype=np.int32)
    zero_action = np.zeros(2, dtype=np.float32)

    try:
        for idx, (base_env, wrapped_env, oracle) in enumerate(env_triplets):
            timesteps.append(reset_with_plan(wrapped_env, base_env, oracle, max_resets))

        for step_idx in range(max_steps):
            active_this_step = int((~done_mask).sum())
            newly_done = []
            for env_idx, (base_env, wrapped_env, oracle) in enumerate(env_triplets):
                try:
                    if done_mask[env_idx]:
                        _, reward, done, _ = base_env.step(zero_action)
                        zero_action_counts[env_idx] += 1
                    else:
                        action = oracle.action(timesteps[env_idx]).action
                        timesteps[env_idx] = wrapped_env.step(action)
                        done = bool(timesteps[env_idx].is_last())
                        reward = 0.0
                        if done:
                            done_mask[env_idx] = True
                            first_done_step[env_idx] = step_idx
                            newly_done.append(env_idx)
                    quat, quat_norm = quat_info(base_env)
                except Exception as exc:
                    print(
                        f"FAIL env={env_idx} step={step_idx} active={not done_mask[env_idx]} "
                        f"zero_action_steps={int(zero_action_counts[env_idx])}: {type(exc).__name__}: {exc}"
                    )
                    return 1

                if not np.isfinite(quat_norm) or quat_norm <= 1e-12:
                    print(
                        f"FAIL env={env_idx} step={step_idx}: invalid quaternion "
                        f"norm={quat_norm} quat={quat.tolist()}"
                    )
                    return 1

            print(
                f"step={step_idx} active={active_this_step} newly_done={newly_done} "
                f"done_total={int(done_mask.sum())}"
            )

            if done_mask.all() and np.all(zero_action_counts >= 10):
                break

        print("PASS")
        print(f"first_done_step={first_done_step}")
        print(f"zero_action_counts={zero_action_counts.tolist()}")
        return 0
    finally:
        for base_env, _, _ in env_triplets:
            try:
                base_env.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_resets", type=int, default=20)
    args = parser.parse_args()
    raise SystemExit(
        run_trial(
            num_envs=args.num_envs,
            seed=args.seed,
            max_steps=args.max_steps,
            max_resets=args.max_resets,
        )
    )


if __name__ == "__main__":
    main()
