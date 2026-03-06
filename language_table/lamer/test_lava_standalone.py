"""
Standalone evaluation of the LAVA policy on single Language Table environments.

Validates that:
  1. The checkpoint loads correctly
  2. Preprocessing matches what the model was trained on
  3. The policy produces meaningful (non-random) actions
  4. Success rates are reasonable per reward type

Usage:
    # From the language-table repo root:
    ltvenv/bin/python -m language_table.lamer.test_lava_standalone \
        --checkpoint_dir /path/to/checkpoints/ \
        --num_episodes 10

    # Quick smoke test (1 episode, block2block only):
    ltvenv/bin/python -m language_table.lamer.test_lava_standalone \
        --checkpoint_dir /path/to/checkpoints/ \
        --num_episodes 1 --reward_types block2block

    # Save videos:
    ltvenv/bin/python -m language_table.lamer.test_lava_standalone \
        --checkpoint_dir /path/to/checkpoints/ \
        --num_episodes 5 --save_videos --video_dir /tmp/lava_eval_videos
"""

import argparse
import collections
import logging
import os
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


REWARD_REGISTRY = {
    "block2block": "language_table.environments.rewards.block2block.BlockToBlockReward",
    "block2absolutelocation": "language_table.environments.rewards.block2absolutelocation.BlockToAbsoluteLocationReward",
    "block2relativelocation": "language_table.environments.rewards.block2relativelocation.BlockToRelativeLocationReward",
    "block2block_relative_location": "language_table.environments.rewards.block2block_relative_location.BlockToBlockRelativeLocationReward",
    "separate_blocks": "language_table.environments.rewards.separate_blocks.SeparateBlocksReward",
}


def _import_reward_class(dotted_path):
    """Import a reward class from a dotted path."""
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def decode_instruction(instruction_array):
    """Decode int32 instruction array to string."""
    arr = np.asarray(instruction_array)
    non_zero = arr[arr != 0]
    if non_zero.shape[0] == 0:
        return ""
    return bytes(non_zero.tolist()).decode("utf-8")


def evaluate_single_env(policy, reward_name, reward_factory, num_episodes,
                        max_episode_steps, block_mode, seed, save_videos,
                        video_dir):
    """Evaluate the policy on a single reward type."""
    from language_table.environments.language_table import LanguageTable
    from language_table.environments.blocks import LanguageTableBlockVariants

    if isinstance(block_mode, str):
        block_mode = LanguageTableBlockVariants(block_mode)

    env = LanguageTable(
        block_mode=block_mode,
        reward_factory=reward_factory,
        seed=seed,
    )

    successes = 0
    total_rewards_list = []
    episode_lengths = []

    for ep in range(num_episodes):
        env.reset()
        state = env._last_state
        instruction = decode_instruction(state.get("instruction", []))
        logger.info("  Episode %d/%d — Task: %s", ep + 1, num_episodes, instruction)

        # Reset policy frame buffers for this single env
        policy.reset(num_envs=1)

        # Use observation rgb (180x320), NOT env.render() which is 444x640.
        # The model was trained on 180x320 images from the observation space.
        obs = dict(state)
        frames = [env.render(mode="rgb_array")] if save_videos else []

        total_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < max_episode_steps:
            # Build single-env inputs
            active_mask = np.array([True])
            actions = policy.predict(
                goals=[instruction],
                obs_list=[obs],
                active_mask=active_mask,
            )
            action = actions[0]

            gym_obs, reward, done, info = env.step(action)
            state = env._last_state
            obs = dict(state)

            total_reward += reward
            step_count += 1

            if save_videos:
                frames.append(env.render(mode="rgb_array"))

        if env.succeeded:
            successes += 1
            logger.info("    -> SUCCESS (reward=%.3f, steps=%d)", total_reward, step_count)
        else:
            logger.info("    -> FAILURE (reward=%.3f, steps=%d)", total_reward, step_count)

        total_rewards_list.append(total_reward)
        episode_lengths.append(step_count)

        if save_videos and frames:
            os.makedirs(video_dir, exist_ok=True)
            status = "success" if env.succeeded else "failure"
            video_path = os.path.join(video_dir, f"{reward_name}_ep{ep}_{status}.mp4")
            try:
                import mediapy
                mediapy.write_video(video_path, frames, fps=10)
                logger.info("    Saved video to %s", video_path)
            except ImportError:
                logger.warning("    mediapy not installed, skipping video save")

    env.close()

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    avg_reward = np.mean(total_rewards_list) if total_rewards_list else 0.0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0.0

    return {
        "reward_type": reward_name,
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standalone LAVA policy evaluation on Language Table")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing the Flax checkpoint")
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="bc_resnet_sim_checkpoint_",
                        help="Checkpoint file prefix")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes per reward type")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--block_mode", type=str, default="BLOCK_8",
                        help="Block mode (BLOCK_4, BLOCK_8, etc.)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward_types", type=str, nargs="+",
                        default=None,
                        help="Reward types to evaluate (default: all)")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save episode videos")
    parser.add_argument("--video_dir", type=str, default="/tmp/lava_eval_videos")
    args = parser.parse_args()

    # Load LAVA policy
    from language_table.lamer.lava_policy import LAVAPolicy

    logger.info("Loading LAVA policy...")
    t0 = time.time()
    policy = LAVAPolicy(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
    )
    logger.info("Policy loaded in %.1fs", time.time() - t0)

    # Select reward types
    reward_types = args.reward_types or list(REWARD_REGISTRY.keys())

    # Evaluate
    results = {}
    for reward_name in reward_types:
        if reward_name not in REWARD_REGISTRY:
            logger.warning("Unknown reward type: %s (skipping)", reward_name)
            continue

        reward_cls = _import_reward_class(REWARD_REGISTRY[reward_name])
        logger.info("=== Evaluating: %s ===", reward_name)

        result = evaluate_single_env(
            policy=policy,
            reward_name=reward_name,
            reward_factory=reward_cls,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_steps,
            block_mode=args.block_mode,
            seed=args.seed,
            save_videos=args.save_videos,
            video_dir=args.video_dir,
        )
        results[reward_name] = result

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Reward Type':<35} {'Success Rate':>12} {'Avg Reward':>10} {'Avg Len':>8}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<35} {r['success_rate']:>11.1%} {r['avg_reward']:>10.3f} {r['avg_length']:>8.1f}")
    print("=" * 60)

    total_successes = sum(r["successes"] for r in results.values())
    total_episodes = sum(r["num_episodes"] for r in results.values())
    print(f"Overall: {total_successes}/{total_episodes} "
          f"({total_successes/total_episodes:.1%})")


if __name__ == "__main__":
    main()
