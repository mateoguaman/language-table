"""
Evaluation driver for Language Table policies.

Sweeps (block_modes x seeds x reward_types) for ONE policy. Loads the model
once, emits per-episode CSV + per-cell JSON + summary JSON. Optional video
recording for selected episode indices.

Designed to be called from training/eval/run_benchmark.sh.

Usage (single policy, full matrix):
    ./ltvenv/bin/python training/eval/run_eval.py \
        --policy_type=lerobot \
        --checkpoint_path=/path/to/pretrained_model \
        --output_dir=eval_results/run \
        --block_modes BLOCK_8 BLOCK_4 \
        --seeds 0 1 2 \
        --reward_types blocktoblock separate point2block \
        --num_episodes=50 --max_steps=200

The LeRobot server is expected to be running already (start it via
training/eval/lerobot_policy_server.py). Pass --server_python to auto-spawn.
"""

import argparse
import csv
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

import jax
import imageio.v2 as imageio
from absl import logging

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block1_to_corner
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import composite as composite_reward
from language_table.environments.rewards import multistep_block_to_location
from language_table.environments.rewards import point2block
from language_table.environments.rewards import separate_blocks
from language_table.environments.rewards import sort_colors_to_corners
from language_table.eval import wrappers as env_wrappers

from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers


# (reward_class, oracle_compatible). oracle_compatible=False means we skip the
# RRT pushing-oracle solvability filter for that reward type (some tasks have
# no single "push this block here" semantics that the oracle understands).
REWARD_REGISTRY = {
    "blocktoblock":                 (block2block.BlockToBlockReward, True),
    "blocktoabsolutelocation":      (block2absolutelocation.BlockToAbsoluteLocationReward, True),
    "blocktoblockrelativelocation": (block2block_relative_location.BlockToBlockRelativeLocationReward, True),
    "blocktorelativelocation":      (block2relativelocation.BlockToRelativeLocationReward, True),
    "separate":                     (separate_blocks.SeparateBlocksReward, False),
    "block1tocorner":               (block1_to_corner.Block1ToCornerLocationReward, True),
    "point2block":                  (point2block.PointToBlockReward, False),
    "sortcolorstocorners":          (sort_colors_to_corners.SortColorsToCornersReward, False),
    "multistep":                    (multistep_block_to_location.make_multistep_reward(n_steps=2), False),
    "composite":                    (composite_reward.CompositeReward, False),
}

BLOCK_MODE_CHOICES = ["BLOCK_4", "BLOCK_8", "BLOCK_4_WPOLE", "BLOCK_8_WPOLE"]


def get_reward_factory(name):
    return REWARD_REGISTRY[name][0]


def is_oracle_compatible(name):
    return REWARD_REGISTRY[name][1]


def get_block_mode(name):
    return getattr(blocks.LanguageTableBlockVariants, name)


def load_lava_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()


def create_lava_policy(checkpoint_path, env, config):
    """Build a LAVA BCJaxPyPolicy with weights restored from checkpoint."""
    from language_table.train import policy as jax_policy
    from language_table.train.networks import lava
    model = lava.SequenceLAVMSE(action_size=2, **config.model)
    return jax_policy.BCJaxPyPolicy(
        env.time_step_spec(),
        env.action_spec(),
        model=model,
        checkpoint_path=checkpoint_path,
        rng=jax.random.PRNGKey(0))


def create_lerobot_policy(env, host, port, server_python, checkpoint_path):
    """Connect (and optionally spawn) the LeRobot policy server."""
    from training.eval.lerobot_policy_wrapper import LeRobotPolicyClient
    server_proc = None
    if server_python:
        server_script = os.path.join(
            os.path.dirname(__file__), "lerobot_policy_server.py")
        cmd = [server_python, server_script,
               "--checkpoint_path", checkpoint_path,
               "--host", host, "--port", str(port)]
        print(f"Spawning policy server: {' '.join(cmd)}")
        server_proc = subprocess.Popen(cmd)
        time.sleep(10)
    policy = LeRobotPolicyClient(
        env.time_step_spec(), env.action_spec(),
        host=host, port=port, connect_timeout=120.0)
    return policy, server_proc


def create_env(reward_factory, block_mode_name, seed, delay_reward_steps,
               use_lava_wrappers=False, lava_config=None):
    """Create Language Table env with policy-appropriate wrappers."""
    env = language_table.LanguageTable(
        block_mode=get_block_mode(block_mode_name),
        reward_factory=reward_factory,
        delay_reward_steps=delay_reward_steps,
        seed=seed)
    env = gym_wrapper.GymWrapper(env)

    if use_lava_wrappers:
        if lava_config is None:
            raise ValueError("LAVA wrappers require lava_config")
        env = env_wrappers.ClipTokenWrapper(env)
        env = env_wrappers.CentralCropImageWrapper(
            env,
            target_width=lava_config.data_target_width,
            target_height=lava_config.data_target_height,
            random_crop_factor=lava_config.random_crop_factor)
        env = tfa_wrappers.HistoryWrapper(
            env, history_length=lava_config.sequence_length,
            tile_first_step_obs=True)
    else:
        env = tfa_wrappers.HistoryWrapper(
            env, history_length=1, tile_first_step_obs=True)

    return env


def get_underlying_env(env):
    """Unwrap to the raw LanguageTable for render() / succeeded / compute_state."""
    cur = env
    while True:
        if isinstance(cur, language_table.LanguageTable):
            return cur
        nxt = getattr(cur, "_env", None) or getattr(cur, "env", None) or \
              getattr(cur, "gym", None) or getattr(cur, "_gym_env", None)
        if nxt is None or nxt is cur:
            return cur
        cur = nxt


def reset_with_oracle(env, max_attempts=20):
    """Reset until RRT oracle finds a feasible plan, else give up and return."""
    ts = None
    for _ in range(max_attempts):
        ts = env.reset()
        try:
            oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
                env, use_ee_planner=True)
            raw_state = env.compute_state()
            if oracle.get_plan(raw_state):
                return ts
        except Exception:
            return ts
    return ts


def get_instruction(time_step):
    obs = time_step.observation
    raw = obs.get("instruction") if hasattr(obs, "get") else None
    if raw is None:
        return ""
    if hasattr(raw, "ndim") and raw.ndim == 2:
        raw = raw[-1]
    nz = raw[raw != 0]
    if nz.shape[0] == 0:
        return ""
    try:
        return bytes(nz.tolist()).decode("utf-8", errors="replace")
    except Exception:
        return ""


def evaluate_cell(policy, env, num_episodes, max_steps, use_oracle,
                  video_episodes, video_dir, csv_writer, csv_meta,
                  reset_policy=None):
    raw_env = get_underlying_env(env)
    successes = 0
    episode_lengths = []
    video_episodes = set(video_episodes or [])

    for ep_num in range(num_episodes):
        ts = reset_with_oracle(env) if use_oracle else env.reset()
        if reset_policy:
            reset_policy()

        instruction = get_instruction(ts)
        record = ep_num in video_episodes
        frames = []
        if record:
            try:
                frames.append(raw_env.render(mode="rgb_array"))
            except Exception:
                record = False

        episode_steps = 0
        while not ts.is_last():
            policy_step = policy.action(ts, ())
            ts = env.step(policy_step.action)
            episode_steps += 1
            if record:
                try:
                    frames.append(raw_env.render(mode="rgb_array"))
                except Exception:
                    pass
            if episode_steps >= max_steps:
                break

        success = bool(raw_env.succeeded)
        if success:
            successes += 1
            episode_lengths.append(episode_steps)
        else:
            episode_lengths.append(max_steps)

        if csv_writer is not None:
            csv_writer.writerow({
                **csv_meta,
                "episode": ep_num,
                "success": int(success),
                "steps": episode_steps if success else max_steps,
                "instruction": instruction[:200],
            })

        if record and frames:
            tag = "success" if success else "fail"
            video_path = os.path.join(
                video_dir, f"ep{ep_num:03d}_{tag}.mp4")
            try:
                imageio.mimwrite(video_path, frames, fps=10, codec="libx264",
                                 macro_block_size=1)
            except Exception as e:
                logging.warning(f"Video write failed for {video_path}: {e}")

    return {
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": successes / num_episodes if num_episodes else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "median_episode_length": float(np.median(episode_lengths)) if episode_lengths else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one Language Table policy across a sweep.")
    parser.add_argument("--policy_type", required=True, choices=["lava", "lerobot"])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config", default=None,
                        help="LAVA config .py (required for --policy_type=lava)")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--block_modes", nargs="+", default=["BLOCK_8"],
                        choices=BLOCK_MODE_CHOICES)
    parser.add_argument("--reward_types", nargs="+",
                        default=list(REWARD_REGISTRY.keys()),
                        choices=list(REWARD_REGISTRY.keys()))
    parser.add_argument("--delay_reward_steps", type=int, default=0,
                        help="Frames the reward zone must be sustained for success")
    parser.add_argument("--video_episodes", type=int, nargs="*", default=[],
                        help="Episode indices to record per cell (e.g. 0 5 10)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip cells whose result.json already exists")

    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=50100)
    parser.add_argument("--server_python", default=None,
                        help="If set, this script spawns the LeRobot server")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-episode CSV (append-mode so re-runs accumulate).
    csv_path = os.path.join(args.output_dir, "episodes.csv")
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, "a", newline="")
    csv_fields = ["policy_type", "checkpoint", "block_mode", "seed",
                  "reward_type", "delay_reward_steps", "episode", "success",
                  "steps", "instruction"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        csv_writer.writeheader()
        csv_file.flush()

    # Load policy once. For LAVA, we need a reference env to provide the spec.
    server_proc = None
    lava_config = None
    use_lava_wrappers = (args.policy_type == "lava")

    if args.policy_type == "lava":
        if args.config is None:
            parser.error("--config is required for --policy_type=lava")
        lava_config = load_lava_config(args.config)

    ref_env = create_env(
        get_reward_factory(args.reward_types[0]),
        args.block_modes[0], args.seeds[0], args.delay_reward_steps,
        use_lava_wrappers=use_lava_wrappers, lava_config=lava_config)

    if args.policy_type == "lava":
        policy = create_lava_policy(args.checkpoint_path, ref_env, lava_config)
    else:
        policy, server_proc = create_lerobot_policy(
            ref_env, args.server_host, args.server_port,
            args.server_python, args.checkpoint_path)

    cells = list(itertools.product(
        args.block_modes, args.seeds, args.reward_types))
    summary = {}
    print(f"Total cells: {len(cells)} "
          f"({len(args.block_modes)} block_modes x "
          f"{len(args.seeds)} seeds x {len(args.reward_types)} rewards)")
    print(f"Episodes per cell: {args.num_episodes}, max_steps={args.max_steps}, "
          f"delay_reward_steps={args.delay_reward_steps}")
    bench_t0 = time.time()

    for cell_idx, (block_mode, seed, reward_name) in enumerate(cells):
        cell_key = f"{block_mode}/{seed}/{reward_name}"
        cell_dir = os.path.join(args.output_dir, block_mode, str(seed),
                                reward_name)
        cell_json = os.path.join(cell_dir, "result.json")

        if args.skip_existing and os.path.exists(cell_json):
            with open(cell_json) as f:
                summary[cell_key] = json.load(f)
            print(f"[{cell_idx+1}/{len(cells)}] {cell_key}: SKIP "
                  f"(existing {summary[cell_key].get('success_rate', '?')})")
            continue

        os.makedirs(cell_dir, exist_ok=True)
        video_dir = os.path.join(cell_dir, "videos")
        if args.video_episodes:
            os.makedirs(video_dir, exist_ok=True)

        try:
            env = create_env(
                get_reward_factory(reward_name), block_mode, seed,
                args.delay_reward_steps,
                use_lava_wrappers=use_lava_wrappers, lava_config=lava_config)
        except Exception as e:
            err = f"env_init: {e}"
            print(f"[{cell_idx+1}/{len(cells)}] {cell_key}: ERROR {err}")
            summary[cell_key] = {"error": err}
            with open(cell_json, "w") as f:
                json.dump(summary[cell_key], f, indent=2)
            continue

        csv_meta = {
            "policy_type": args.policy_type,
            "checkpoint": args.checkpoint_path,
            "block_mode": block_mode,
            "seed": seed,
            "reward_type": reward_name,
            "delay_reward_steps": args.delay_reward_steps,
        }
        reset_fn = policy.reset if hasattr(policy, "reset") else None

        t0 = time.time()
        try:
            cell_result = evaluate_cell(
                policy, env, args.num_episodes, args.max_steps,
                use_oracle=is_oracle_compatible(reward_name),
                video_episodes=args.video_episodes,
                video_dir=video_dir,
                csv_writer=csv_writer, csv_meta=csv_meta,
                reset_policy=reset_fn)
            csv_file.flush()
        except Exception as e:
            traceback.print_exc()
            err = f"eval: {e}"
            print(f"[{cell_idx+1}/{len(cells)}] {cell_key}: ERROR {err}")
            summary[cell_key] = {"error": err}
            with open(cell_json, "w") as f:
                json.dump(summary[cell_key], f, indent=2)
            continue
        finally:
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass

        cell_result.update({
            "block_mode": block_mode, "seed": seed,
            "reward_type": reward_name,
            "delay_reward_steps": args.delay_reward_steps,
            "elapsed_sec": time.time() - t0,
        })
        with open(cell_json, "w") as f:
            json.dump(cell_result, f, indent=2)
        summary[cell_key] = cell_result

        print(f"[{cell_idx+1}/{len(cells)}] {cell_key}: "
              f"{cell_result['success_rate']:.1%} "
              f"({cell_result['successes']}/{cell_result['num_episodes']}) "
              f"mean_steps={cell_result['mean_episode_length']:.1f} "
              f"in {cell_result['elapsed_sec']:.0f}s")

    elapsed = time.time() - bench_t0
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "policy_type": args.policy_type,
            "checkpoint": args.checkpoint_path,
            "args": {k: v for k, v in vars(args).items()},
            "elapsed_sec": elapsed,
            "cells": summary,
        }, f, indent=2)

    print(f"\nDone in {elapsed:.0f}s")
    print(f"Summary: {summary_path}")
    print(f"Per-episode CSV: {csv_path}")

    if server_proc is not None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except Exception:
            server_proc.kill()
    if hasattr(policy, "close"):
        try:
            policy.close()
        except Exception:
            pass
    csv_file.close()


if __name__ == "__main__":
    main()
