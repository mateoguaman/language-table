"""
Evaluation driver for Language Table policies.

Supports both LAVA (original BCJaxPyPolicy) and LeRobot policies (via TCP server).
Outputs JSON results for comparison.

Usage (LAVA baseline):
    ./ltvenv/bin/python training/eval/run_eval.py \
        --policy_type=lava \
        --checkpoint_path=<flax_checkpoint> \
        --config=language_table/train/configs/language_table_sim_local.py \
        --output_dir=eval_results/lava_baseline

Usage (LeRobot policy):
    # First start the policy server (in another terminal or let this script spawn it):
    ./lerobot_env/bin/python training/eval/lerobot_policy_server.py \
        --checkpoint_path=outputs/smolvla_expert_oracle/checkpoints/last/pretrained_model

    # Then run eval:
    ./ltvenv/bin/python training/eval/run_eval.py \
        --policy_type=lerobot \
        --checkpoint_path=outputs/smolvla_expert_oracle/checkpoints/last/pretrained_model \
        --output_dir=eval_results/smolvla_expert_oracle

    # Or let this script auto-spawn the server:
    ./ltvenv/bin/python training/eval/run_eval.py \
        --policy_type=lerobot \
        --checkpoint_path=outputs/smolvla_expert_oracle/checkpoints/last/pretrained_model \
        --server_python=./lerobot_env/bin/python \
        --output_dir=eval_results/smolvla_expert_oracle
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time

from absl import logging
import jax
import numpy as np

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import separate_blocks
from language_table.eval import wrappers as env_wrappers

from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

REWARD_FACTORIES = {
    "blocktoblock": block2block.BlockToBlockReward,
    "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
    "blocktoblockrelativelocation": block2block_relative_location.BlockToBlockRelativeLocationReward,
    "blocktorelativelocation": block2relativelocation.BlockToRelativeLocationReward,
    "separate": separate_blocks.SeparateBlocksReward,
}


def create_lava_policy(checkpoint_path, config_path, env):
    """Create LAVA BCJaxPyPolicy."""
    from ml_collections import config_flags
    from language_table.train import policy as jax_policy
    from language_table.train.networks import lava
    import importlib.util

    # Load config
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()

    model = lava.SequenceLAVMSE(action_size=2, **config.model)
    policy = jax_policy.BCJaxPyPolicy(
        env.time_step_spec(),
        env.action_spec(),
        model=model,
        checkpoint_path=checkpoint_path,
        rng=jax.random.PRNGKey(0))
    return policy, config


def create_lerobot_policy(checkpoint_path, env, host, port, server_python):
    """Create LeRobot policy client (optionally spawning the server)."""
    from training.eval.lerobot_policy_wrapper import LeRobotPolicyClient

    server_proc = None
    if server_python:
        # Spawn the policy server
        server_script = os.path.join(
            os.path.dirname(__file__), "lerobot_policy_server.py")
        cmd = [
            server_python, server_script,
            "--checkpoint_path", checkpoint_path,
            "--host", host,
            "--port", str(port),
        ]
        print(f"Spawning policy server: {' '.join(cmd)}")
        server_proc = subprocess.Popen(cmd)
        # Wait for server to start
        time.sleep(10)

    policy = LeRobotPolicyClient(
        env.time_step_spec(),
        env.action_spec(),
        host=host,
        port=port,
        connect_timeout=120.0,
    )
    return policy, server_proc


def create_env(reward_factory, use_lava_wrappers=False, config=None):
    """Create Language Table environment with appropriate wrappers."""
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=reward_factory,
        seed=0)
    env = gym_wrapper.GymWrapper(env)

    if use_lava_wrappers and config is not None:
        # LAVA needs CLIP tokens and center crop
        env = env_wrappers.ClipTokenWrapper(env)
        env = env_wrappers.CentralCropImageWrapper(
            env,
            target_width=config.data_target_width,
            target_height=config.data_target_height,
            random_crop_factor=config.random_crop_factor)
        env = tfa_wrappers.HistoryWrapper(
            env, history_length=config.sequence_length, tile_first_step_obs=True)
    else:
        # LeRobot policies get raw observations (they resize internally)
        # Use history_length=1 (no stacking needed)
        env = tfa_wrappers.HistoryWrapper(
            env, history_length=1, tile_first_step_obs=True)

    return env


def evaluate(policy, env, num_episodes=50, max_steps=200, reset_policy=None):
    """Run evaluation episodes, return success count and details."""
    successes = 0
    episode_lengths = []

    for ep_num in range(num_episodes):
        # Reset env with oracle-validated initialization
        oracle_policy = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
            env, use_ee_planner=True)
        plan_success = False
        while not plan_success:
            ts = env.reset()
            raw_state = env.compute_state()
            plan_success = oracle_policy.get_plan(raw_state)

        if reset_policy:
            reset_policy()

        episode_steps = 0
        while not ts.is_last():
            policy_step = policy.action(ts, ())
            ts = env.step(policy_step.action)
            episode_steps += 1
            if episode_steps >= max_steps:
                break

        if env.succeeded:
            successes += 1
            episode_lengths.append(episode_steps)
            logging.info(f"Episode {ep_num}: success ({episode_steps} steps)")
        else:
            episode_lengths.append(max_steps)
            logging.info(f"Episode {ep_num}: failure")

    return {
        "successes": successes,
        "total": num_episodes,
        "success_rate": successes / num_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "median_episode_length": float(np.median(episode_lengths)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Language Table policies")
    parser.add_argument("--policy_type", required=True,
                        choices=["lava", "lerobot"],
                        help="Policy type to evaluate")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to policy checkpoint")
    parser.add_argument("--config", default=None,
                        help="Config file path (required for LAVA)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write results JSON")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Episodes per reward type")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--reward_types", nargs="+",
                        default=list(REWARD_FACTORIES.keys()),
                        help="Reward types to evaluate")
    # LeRobot server options
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=50100)
    parser.add_argument("--server_python", default=None,
                        help="Python binary for spawning server (e.g., ./lerobot_env/bin/python). "
                             "If not set, assumes server is already running.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Create policy ---
    server_proc = None
    config = None
    policy = None
    reset_fn = None
    use_lava_wrappers = False

    if args.policy_type == "lava":
        if args.config is None:
            parser.error("--config is required for LAVA policy")
        # Create a dummy env to get specs (will create per-reward envs later)
        dummy_env = create_env(
            block2block.BlockToBlockReward, use_lava_wrappers=True, config=None)
        # Actually need config first
        use_lava_wrappers = True

    elif args.policy_type == "lerobot":
        use_lava_wrappers = False

    # --- Evaluate per reward type ---
    results = {}
    all_policy = None

    for reward_name in args.reward_types:
        print(f"\n{'='*60}")
        print(f"Evaluating: {reward_name}")
        print(f"{'='*60}")

        reward_factory = REWARD_FACTORIES[reward_name]

        if args.policy_type == "lava":
            env = create_env(reward_factory, use_lava_wrappers=True, config=config)
            if all_policy is None:
                all_policy, config = create_lava_policy(
                    args.checkpoint_path, args.config, env)
                # Re-create env now that we have config
                env = create_env(reward_factory, use_lava_wrappers=True, config=config)
                all_policy, _ = create_lava_policy(
                    args.checkpoint_path, args.config, env)
            policy = all_policy

        elif args.policy_type == "lerobot":
            env = create_env(reward_factory, use_lava_wrappers=False)
            if all_policy is None:
                all_policy, server_proc = create_lerobot_policy(
                    args.checkpoint_path, env,
                    host=args.server_host,
                    port=args.server_port,
                    server_python=args.server_python)
            policy = all_policy
            reset_fn = policy.reset

        reward_results = evaluate(
            policy, env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            reset_policy=reset_fn,
        )
        results[reward_name] = reward_results
        print(f"  -> Success rate: {reward_results['success_rate']:.1%} "
              f"({reward_results['successes']}/{reward_results['total']})")

    # --- Save results ---
    output = {
        "policy_type": args.policy_type,
        "checkpoint_path": args.checkpoint_path,
        "num_episodes_per_reward": args.num_episodes,
        "max_steps": args.max_steps,
        "results": results,
        "summary": {
            "mean_success_rate": float(np.mean(
                [r["success_rate"] for r in results.values()])),
        },
    }

    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Summary: {args.policy_type} @ {args.checkpoint_path}")
    print(f"{'='*60}")
    print(f"{'Reward Type':<35} {'Success Rate':>12}")
    print(f"{'-'*35} {'-'*12}")
    for reward_name, r in results.items():
        print(f"{reward_name:<35} {r['success_rate']:>11.1%}")
    print(f"{'-'*35} {'-'*12}")
    print(f"{'MEAN':<35} {output['summary']['mean_success_rate']:>11.1%}")

    # Cleanup
    if server_proc:
        server_proc.terminate()
        server_proc.wait()
    if hasattr(policy, 'close'):
        policy.close()


if __name__ == "__main__":
    main()
