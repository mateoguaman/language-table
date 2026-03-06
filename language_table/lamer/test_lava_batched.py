"""
Batched evaluation of the LAVA policy using LanguageTableMultiProcessEnv.

Validates that:
  1. Batched inference produces correct actions across multiple parallel envs
  2. RGB images are properly returned from Ray workers (render_obs=True)
  3. Throughput is acceptable for use in the LaMer training loop

Usage:
    ltvenv/bin/python -m language_table.lamer.test_lava_batched \
        --checkpoint_dir /path/to/checkpoints/ \
        --num_envs 4 --max_steps 100
"""

import argparse
import logging
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def decode_instruction(instruction_array):
    arr = np.asarray(instruction_array)
    non_zero = arr[arr != 0]
    if non_zero.shape[0] == 0:
        return ""
    return bytes(non_zero.tolist()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Batched LAVA policy evaluation on Language Table")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="bc_resnet_sim_checkpoint_")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Inner-loop steps per episode")
    parser.add_argument("--block_mode", type=str, default="BLOCK_4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import ray
    if not ray.is_initialized():
        ray.init()

    from language_table.environments.rewards.block2block import BlockToBlockReward
    from language_table.lamer.envs import LanguageTableMultiProcessEnv
    from language_table.lamer.lava_policy import LAVAPolicy

    # Load LAVA policy
    logger.info("Loading LAVA policy...")
    t0 = time.time()
    policy = LAVAPolicy(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
    )
    logger.info("Policy loaded in %.1fs", time.time() - t0)

    # Create parallel envs with RGB included
    logger.info("Creating %d parallel envs (render_obs=True)...", args.num_envs)
    envs = LanguageTableMultiProcessEnv(
        num_envs=args.num_envs,
        block_mode=args.block_mode,
        reward_factory_cls=BlockToBlockReward,
        seed=args.seed,
        group_n=1,
        return_full_state=True,
        render_obs=True,
    )

    # Reset and get initial observations
    logger.info("Resetting environments...")
    obs_list, infos = envs.reset()

    # Verify RGB is present
    assert "rgb" in obs_list[0], "RGB not found in observations! Check render_obs flag."
    logger.info("RGB shape per env: %s", obs_list[0]["rgb"].shape)

    # Extract instructions
    instructions = []
    for obs in obs_list:
        instr = decode_instruction(obs.get("instruction", []))
        instructions.append(instr)
    logger.info("Tasks: %s", instructions[:3])

    # Run inner-loop evaluation
    policy.reset(num_envs=args.num_envs)
    active_mask = np.ones(args.num_envs, dtype=bool)
    total_rewards = np.zeros(args.num_envs, dtype=np.float32)
    step_times = []

    logger.info("Running %d inner-loop steps...", args.max_steps)
    for step in range(args.max_steps):
        t_step = time.time()

        actions = policy.predict(instructions, obs_list, active_mask)

        obs_list, rewards, dones, step_infos = envs.step(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)

        total_rewards += rewards * active_mask
        newly_done = dones & active_mask
        active_mask &= ~dones

        step_time = time.time() - t_step
        step_times.append(step_time)

        if step % 20 == 0 or not active_mask.any():
            logger.info(
                "  Step %3d: active=%d, step_time=%.3fs, cumulative_reward=%s",
                step, active_mask.sum(), step_time,
                total_rewards.tolist())

        if not active_mask.any():
            logger.info("All envs done at step %d", step)
            break

    # Summary
    step_times_np = np.array(step_times)
    successes = (total_rewards > 0).sum()

    print("\n" + "=" * 60)
    print("BATCHED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Num envs:           {args.num_envs}")
    print(f"Total steps:        {len(step_times)}")
    print(f"Success rate:       {successes}/{args.num_envs} ({successes/args.num_envs:.0%})")
    print(f"Rewards:            {total_rewards.tolist()}")
    print(f"Step time (mean):   {step_times_np.mean():.4f}s")
    print(f"Step time (p50):    {np.percentile(step_times_np, 50):.4f}s")
    print(f"Step time (p95):    {np.percentile(step_times_np, 95):.4f}s")
    # Skip first step (JIT compilation)
    if len(step_times) > 1:
        warm = step_times_np[1:]
        print(f"Step time (mean, warm): {warm.mean():.4f}s")
        print(f"Throughput (warm):  {args.num_envs / warm.mean():.1f} env-steps/s")
    print("=" * 60)

    envs.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
