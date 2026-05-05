"""Simple GUI smoke test for the Language Table ManiSkill environment.

Creates the environment in viewer/GUI mode and steps forever with random actions.
"""

from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np

import language_table.environments.maniskill_env  # registers LanguageTable-v1
from language_table.environments import blocks


BLOCK_MODE_CHOICES = {
    mode.value: mode
    for mode in (
        blocks.LanguageTableBlockVariants.BLOCK_4,
        blocks.LanguageTableBlockVariants.BLOCK_8,
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open LanguageTable-v1 in the ManiSkill GUI and step forever."
    )
    parser.add_argument(
        "--block_mode",
        choices=sorted(BLOCK_MODE_CHOICES),
        default=blocks.LanguageTableBlockVariants.BLOCK_4.value,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--obs_mode",
        default="state_dict",
        help="ManiSkill observation mode to use.",
    )
    parser.add_argument(
        "--control_mode",
        default="pd_ee_delta_pos",
        help="ManiSkill control mode to use.",
    )
    parser.add_argument(
        "--sleep_s",
        type=float,
        default=1.0 / 60.0,
        help="Delay after each step to keep the GUI responsive.",
    )
    parser.add_argument(
        "--render_mode",
        default="human",
        help="Gym render mode. Use 'human' to open the GUI viewer.",
    )
    parser.add_argument(
        "--sim_backend",
        default="cpu",
        help="ManiSkill sim backend. Common values: auto, cpu, gpu.",
    )
    parser.add_argument(
        "--render_backend",
        default="cpu",
        help="ManiSkill render backend. Use 'cpu' for the GUI viewer.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the GUI and use rgb_array rendering instead.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = None
    render_mode = "rgb_array" if args.headless else args.render_mode

    try:
        env = gym.make(
            "LanguageTable-v1",
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            render_mode=render_mode,
            num_envs=1,
            block_mode=BLOCK_MODE_CHOICES[args.block_mode],
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
        )
        env.reset(seed=args.seed)
        env.render()
        step_count = 0

        while True:
            action = env.action_space.sample()

            # Keep this deterministic if the sampled action space uses numpy.
            if isinstance(action, np.ndarray):
                action = rng.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=env.action_space.shape,
                ).astype(env.action_space.dtype)

            _, _, terminated, truncated, _ = env.step(action)
            env.render()
            step_count += 1

            if step_count % 100 == 0:
                print(f"Stepped {step_count} times")

            if bool(np.any(np.asarray(terminated))) or bool(np.any(np.asarray(truncated))):
                print(f"Episode ended at step {step_count}, resetting")
                env.reset()

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
    except RuntimeError as exc:
        if "Renderer does not support display" in str(exc):
            display = os.environ.get("DISPLAY")
            wayland_display = os.environ.get("WAYLAND_DISPLAY")
            raise RuntimeError(
                "Failed to create the GUI viewer. "
                f"DISPLAY={display!r}, WAYLAND_DISPLAY={wayland_display!r}. "
                "Try running from a desktop terminal and keep "
                "--render_backend=cpu. If you still want GPU physics, use "
                "--sim_backend=gpu --render_backend=cpu."
            ) from exc
        raise
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
