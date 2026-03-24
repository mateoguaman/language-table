#!/usr/bin/env python3
"""
Interactive debugging script for the Language Table environment (standalone,
no server required).

Usage:
    ltvenv/bin/python test_language_table.py [--num_envs 2] [--vla_policy gemini]

Commands at the interactive prompt:
    <goal string>   — step(phase="play") with that goal for all envs
    reset           — call reset()
    restart         — call restart()
    reflect         — call reflect() then step(phase="reflect")
    quit / exit     — shut down
"""

import argparse
import time

import numpy as np

from language_table.environments.rewards.block2block import BlockToBlockReward
from language_table.environments.rewards.block2absolutelocation import BlockToAbsoluteLocationReward
from language_table.environments.rewards.block2relativelocation import BlockToRelativeLocationReward
from language_table.environments.rewards.block2block_relative_location import BlockToBlockRelativeLocationReward
from language_table.environments.rewards.point2block import PointToBlockReward
from language_table.environments.rewards.separate_blocks import SeparateBlocksReward
from language_table.environments.rewards.composite import CompositeReward
from language_table.lamer.gemini_env_manager import LanguageTableEnvironmentManager

REWARD_TYPES = {
    "composite": CompositeReward,
    "block2block": BlockToBlockReward,
    "block2absolutelocation": BlockToAbsoluteLocationReward,
    "block2relativelocation": BlockToRelativeLocationReward,
    "block2block_relative_location": BlockToBlockRelativeLocationReward,
    "point2block": PointToBlockReward,
    "separate_blocks": SeparateBlocksReward,
    "none": None,
}
from language_table.lamer.envs import LanguageTableMultiProcessEnv


def print_obs(obs, label="obs"):
    texts = obs.get("text", [])
    if isinstance(texts, str):
        texts = [texts]
    print(f"\n--- {label} ---")
    for i, t in enumerate(texts):
        print(f"  [env {i}] {t}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Standalone Language Table env debugger")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--block_mode", type=str, default="BLOCK_4")
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=1)
    parser.add_argument("--max_inner_steps", type=int, default=100)
    parser.add_argument("--do_reflection", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward_type", type=str, default="block2block",
                        choices=list(REWARD_TYPES.keys()),
                        help="Task/reward type (default: block2block)")
    parser.add_argument("--vla_policy", type=str, default=None,
                        choices=["lava", "gemini"],
                        help="Inner-loop policy. Omit for random actions.")
    parser.add_argument("--vla_checkpoint", type=str, default=None,
                        help="Path to LAVA checkpoint. Required when --vla_policy=lava.")
    parser.add_argument("--gemini_action_clip", type=float, default=0.1)
    args = parser.parse_args()

    if args.vla_policy == "lava" and not args.vla_checkpoint:
        parser.error("--vla_policy=lava requires --vla_checkpoint")

    render_obs = args.vla_policy == "lava"

    print(f"Initialising {args.num_envs} envs (block_mode={args.block_mode})...")
    envs = LanguageTableMultiProcessEnv(
        num_envs=args.num_envs,
        block_mode=args.block_mode,
        reward_factory_cls=REWARD_TYPES[args.reward_type],
        seed=args.seed,
        render_obs=render_obs,
    )

    # Load VLA policy
    vla_policy = None
    if args.vla_policy == "lava":
        import os
        from language_table.lamer.lava_policy import LAVAPolicy

        checkpoint_dir = os.path.dirname(args.vla_checkpoint)
        checkpoint_prefix = os.path.basename(args.vla_checkpoint).rsplit("_", 1)[0] + "_"
        print(f"Loading LAVA policy from {checkpoint_dir} (prefix={checkpoint_prefix})...")
        vla_policy = LAVAPolicy(checkpoint_dir=checkpoint_dir, checkpoint_prefix=checkpoint_prefix)
        print("LAVA policy loaded.")

        manager = LanguageTableEnvironmentManager(
            envs=envs,
            num_attempts=args.num_attempts,
            max_turns=args.max_turns,
            do_reflection=args.do_reflection,
            max_inner_steps=args.max_inner_steps,
            vla_policy=vla_policy,
        )
    elif args.vla_policy == "gemini":
        from language_table.lamer.gemini_policy import GeminiPolicy

        print(f"Using Gemini API policy (action_clip={args.gemini_action_clip})")
        vla_policy = GeminiPolicy(action_clip=args.gemini_action_clip)

        policy = GeminiPolicy()
        manager = LanguageTableEnvironmentManager(
            envs=envs,
            policy=policy,
            num_attempts=args.num_attempts,
            max_turns=args.max_turns,
            do_reflection=args.do_reflection,
            max_inner_steps=args.max_inner_steps,
        )

    num_envs = envs.num_processes

    print(f"Ready. num_envs={num_envs}, num_attempts={args.num_attempts}, "
          f"max_turns={args.max_turns}, do_reflection={args.do_reflection}")

    print("\nResetting envs...")
    obs, infos = manager.reset()
    print_obs(obs, label="initial obs")

    print("Type a goal string to step, or: reset / restart / reflect / quit")
    print("  Example goals:")
    print("    push the red star to the blue cube")
    print("    move the green moon to the left of the yellow pentagon\n")

    try:
        while True:
            try:
                user_input = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "q"):
                break

            elif cmd == "reset":
                print("Resetting...")
                t0 = time.perf_counter()
                obs, infos = manager.reset()
                print(f"Done ({time.perf_counter() - t0:.2f}s)")
                print_obs(obs, label="reset obs")

            elif cmd == "restart":
                print("Restarting (meta-RL attempt)...")
                t0 = time.perf_counter()
                obs, infos = manager.restart()
                print(f"Done ({time.perf_counter() - t0:.2f}s)")
                print_obs(obs, label="restart obs")

            elif cmd == "reflect":
                print("Fetching reflect prompts...")
                obs, infos = manager.reflect()
                print_obs(obs, label="reflect prompts")

                reflection = input("  Enter reflection text (Enter = use default): ").strip()
                if not reflection:
                    reflection = "I should try a different approach next time."
                reflections = [reflection] * num_envs

                print("Submitting reflections...")
                t0 = time.perf_counter()
                obs, rewards, dones, infos = manager.step(reflections, phase="reflect")
                print(f"Done ({time.perf_counter() - t0:.2f}s)")

            else:
                goals = [user_input] * num_envs
                print(f"Stepping with goal: '{user_input}' (x{num_envs} envs)...")
                t0 = time.perf_counter()
                obs, rewards, dones, infos = manager.step(goals, phase="play")
                elapsed = time.perf_counter() - t0
                print(f"Done ({elapsed:.2f}s)")
                print_obs(obs, label="post-step obs")
                rewards = np.asarray(rewards)
                dones = np.asarray(dones)
                print(f"  rewards: {rewards.tolist()}")
                print(f"  dones:   {dones.tolist()}")
                wons = [info.get("won", False) for info in infos]
                print(f"  won:     {wons}")

    finally:
        manager.close()
        print("Envs closed.")


if __name__ == "__main__":
    main()
