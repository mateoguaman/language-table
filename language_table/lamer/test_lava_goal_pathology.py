#!/usr/bin/env python3
"""Probe whether odd goal strings cause pathological LAVA behavior."""

import argparse
import os

import numpy as np

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block
from language_table.lamer.lava_policy import LAVAPolicy


def decode_instruction(instruction_array):
    arr = np.asarray(instruction_array)
    non_zero = arr[arr != 0]
    if non_zero.shape[0] == 0:
        return ""
    return bytes(non_zero.tolist()).decode("utf-8")


def quat_info(base_env):
    effector_state = base_env._pybullet_client.getLinkState(
        base_env._robot.xarm,
        base_env._robot.effector_link,
        computeForwardKinematics=1,
    )
    quat = np.asarray(effector_state[1], dtype=np.float64)
    return quat, float(np.linalg.norm(quat))


def weird_goals(native_instruction):
    return [
        ("native", native_instruction),
        ("empty", ""),
        ("reflection", "I failed before. Next time I should push more carefully and be more strategic."),
        ("json", '{"plan":["approach","push"],"target":"red star","note":"be precise"}'),
        ("gibberish", "zxqv red star !!! <tool_call> north north reflect"),
        ("long_repeat", ("push the red block to the blue block " * 50).strip()),
        ("contradiction", "do not move anything. also push the red star directly into the blue cube now."),
    ]


def run_probe(checkpoint_path, rollout_steps, seed):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_prefix = os.path.basename(checkpoint_path).rsplit("_", 1)[0] + "_"

    policy = LAVAPolicy(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
    )

    envs = []
    obs_list = []
    native_instruction = None
    goals = []
    labels = []
    for idx in range(7):
        env = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
            reward_factory=block2block.BlockToBlockReward,
            seed=seed + idx,
        )
        obs = env.reset()
        envs.append(env)
        obs_list.append(dict(env._last_state))
        if native_instruction is None:
            native_instruction = decode_instruction(env._last_state.get("instruction", []))

    for label, goal in weird_goals(native_instruction):
        labels.append(label)
        goals.append(goal)

    if len(envs) != len(goals):
        raise ValueError("env count must match goal count")

    policy.reset(num_envs=len(envs))
    active_mask = np.ones(len(envs), dtype=bool)

    try:
        print("Static goal probe:")
        actions = policy.predict(goals, obs_list, active_mask)
        for idx, action in enumerate(actions):
            action = np.asarray(action, dtype=np.float32)
            print(
                f"  {labels[idx]:>13}: finite={np.isfinite(action).all()} "
                f"norm={float(np.linalg.norm(action)):.6f} action={action.tolist()} "
                f"goal_len={len(goals[idx])}"
            )

        print("\nShort rollout probe:")
        for step_idx in range(rollout_steps):
            actions = policy.predict(goals, obs_list, active_mask)
            for env_idx, env in enumerate(envs):
                action = np.asarray(actions[env_idx], dtype=np.float32)
                if not np.isfinite(action).all():
                    print(f"FAIL {labels[env_idx]} step={step_idx}: non-finite action={action.tolist()}")
                    return 1
                try:
                    obs, reward, done, _ = env.step(action)
                    obs_list[env_idx] = dict(env._last_state)
                    quat, quat_norm = quat_info(env)
                except Exception as exc:
                    print(
                        f"FAIL {labels[env_idx]} step={step_idx}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return 1

                if not np.isfinite(quat_norm) or quat_norm <= 1e-12:
                    print(
                        f"FAIL {labels[env_idx]} step={step_idx}: "
                        f"invalid quaternion norm={quat_norm} quat={quat.tolist()}"
                    )
                    return 1

                print(
                    f"  step={step_idx:02d} label={labels[env_idx]:>13} "
                    f"done={done} reward={reward:.3f} action_norm={float(np.linalg.norm(action)):.6f} "
                    f"quat_norm={quat_norm:.6f}"
                )
        print("PASS")
        return 0
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(
        run_probe(
            checkpoint_path=args.checkpoint_path,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
