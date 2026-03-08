#!/usr/bin/env python3
"""Direct repro for stepping Language Table after done=True.

Uses the repo's oracle policy to solve a real LanguageTable episode, then
continues stepping the same env with zero actions after the first terminal
transition. Prints the raw end-effector quaternion and whether any exception
occurs during those extra steps.

Usage:
    ltvenv/bin/python -m language_table.lamer.test_step_after_done
"""

import argparse
import math
import sys
import traceback
import types

import numpy as np
from tf_agents.environments import gym_wrapper

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block

try:
    import imageio as _imageio  # noqa: F401
except ModuleNotFoundError:
    # The oracle planner imports imageio only for optional debug image writing.
    sys.modules["imageio"] = types.SimpleNamespace(imwrite=lambda *args, **kwargs: None)

from language_table.environments.oracles import push_oracle_rrt_slowdown


def _quat_info(base_env):
    effector_state = base_env._pybullet_client.getLinkState(
        base_env._robot.xarm,
        base_env._robot.effector_link,
        computeForwardKinematics=1,
    )
    quat = np.asarray(effector_state[1], dtype=np.float64)
    return {
        "quat": quat.tolist(),
        "norm": float(np.linalg.norm(quat)),
        "translation": np.asarray(effector_state[0], dtype=np.float64).tolist(),
    }


def _make_env(seed: int):
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


def _reset_with_valid_plan(wrapped_env, base_env, oracle, max_resets: int):
    for reset_idx in range(max_resets):
        ts = wrapped_env.reset()
        raw_state = base_env.compute_state()
        if oracle.get_plan(raw_state):
            return ts, reset_idx
    raise RuntimeError(
        f"Failed to find a valid oracle motion plan after {max_resets} resets"
    )


def run_once(seed: int, max_steps: int, post_done_steps: int, max_resets: int):
    base_env, wrapped_env, oracle = _make_env(seed)

    try:
        ts, reset_idx = _reset_with_valid_plan(
            wrapped_env,
            base_env,
            oracle,
            max_resets=max_resets,
        )
        print(f"Found valid plan after {reset_idx + 1} reset(s)")

        step_count = 0
        while not ts.is_last() and step_count < max_steps:
            action = oracle.action(ts).action
            ts = wrapped_env.step(action)
            step_count += 1

        print(f"Episode finished: done={bool(ts.is_last())} steps={step_count}")
        if not ts.is_last():
            print("Oracle did not finish the episode within max_steps")
            return 2

        done_quat = _quat_info(base_env)
        print(
            "Quaternion at terminal step:",
            f"norm={done_quat['norm']:.6f}",
            f"quat={done_quat['quat']}",
            f"translation={done_quat['translation']}",
        )

        zero_action = np.zeros(2, dtype=np.float32)
        for extra_idx in range(post_done_steps):
            before = _quat_info(base_env)
            print(
                f"Post-done step {extra_idx + 1} before:",
                f"norm={before['norm']:.6f}",
                f"quat={before['quat']}",
            )
            try:
                obs, reward, done, _ = base_env.step(zero_action)
                _ = obs, reward
            except Exception as exc:
                print(f"Post-done step {extra_idx + 1} raised: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return 1

            after = _quat_info(base_env)
            print(
                f"Post-done step {extra_idx + 1} after:",
                f"done={done}",
                f"norm={after['norm']:.6f}",
                f"quat={after['quat']}",
            )

            if not math.isfinite(after["norm"]) or after["norm"] <= 1e-12:
                print("Detected invalid quaternion after stepping post-done")
                return 1

        print("No invalid quaternion observed during post-done stepping")
        return 0
    finally:
        try:
            base_env.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--post_done_steps", type=int, default=25)
    parser.add_argument("--max_resets", type=int, default=20)
    args = parser.parse_args()
    raise SystemExit(
        run_once(
            seed=args.seed,
            max_steps=args.max_steps,
            post_done_steps=args.post_done_steps,
            max_resets=args.max_resets,
        )
    )


if __name__ == "__main__":
    main()
