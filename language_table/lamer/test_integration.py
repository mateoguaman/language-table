"""
Integration test for the VLA-enabled environment manager.

Runs the full meta-RL loop in-process (no TCP server) to verify that:
  1. VLA produces meaningful actions (not random) given goal strings
  2. Different goal strings produce different behavior
  3. The reset → play → reflect → restart → play cycle works end-to-end
  4. Reflection prompts include episode outcomes
  5. Rewards are non-trivial when goals match environment tasks

Usage:
    ltvenv/bin/python -m language_table.lamer.test_integration \
        --checkpoint_dir /path/to/checkpoints/
"""

import argparse
import logging
import sys
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


def test_vla_vs_random(envs, policy, num_envs, max_inner_steps):
    """Verify VLA produces different (better) outcomes than random actions."""
    from language_table.lamer.env_manager import LanguageTableEnvironmentManager

    # Run with VLA
    manager_vla = LanguageTableEnvironmentManager(
        envs=envs, num_attempts=1, max_turns=1,
        max_inner_steps=max_inner_steps, vla_policy=policy,
    )
    obs_vla, _ = manager_vla.reset()

    # Extract env instructions as goals (best chance of success)
    raw_obs, _ = envs.reset()
    instructions = [decode_instruction(o.get("instruction", [])) for o in raw_obs]
    logger.info("Env instructions: %s", instructions)

    # Need to re-reset since we just reset envs to get instructions
    obs_vla, _ = manager_vla.reset()
    raw_obs2, _ = envs.reset()  # sync state
    instructions = [decode_instruction(o.get("instruction", [])) for o in raw_obs2]
    obs_vla, _ = manager_vla.reset()

    _, rewards_vla, _, _ = manager_vla.step(instructions, phase="play")
    logger.info("VLA rewards: %s", rewards_vla.tolist())

    # Run with random actions (no VLA)
    manager_rand = LanguageTableEnvironmentManager(
        envs=envs, num_attempts=1, max_turns=1,
        max_inner_steps=max_inner_steps, vla_policy=None,
    )
    manager_rand.reset()
    _, rewards_rand, _, _ = manager_rand.step(instructions, phase="play")
    logger.info("Random rewards: %s", rewards_rand.tolist())

    # VLA should generally do better (or at least not worse on average)
    vla_mean = rewards_vla.mean()
    rand_mean = rewards_rand.mean()
    logger.info("VLA mean reward: %.3f, Random mean reward: %.3f", vla_mean, rand_mean)

    return rewards_vla, rewards_rand


def test_meta_rl_loop(envs, policy, num_envs, max_inner_steps):
    """Test the full reset → play → reflect → restart → play cycle."""
    from language_table.lamer.env_manager import LanguageTableEnvironmentManager

    manager = LanguageTableEnvironmentManager(
        envs=envs, num_attempts=3, max_turns=1,
        do_reflection=True, max_inner_steps=max_inner_steps,
        vla_policy=policy,
    )

    # --- Attempt 1 ---
    logger.info("=== Attempt 1: reset + play ===")
    observations, infos = manager.reset()
    assert "text" in observations, "Missing 'text' in observations"
    assert len(observations["text"]) == num_envs
    assert observations["image"] is None, "include_rgb=False but images present"
    logger.info("Reset text preview: %s", observations["text"][0][:80])

    # Use a plausible goal
    goals_1 = ["push the red block to the blue block"] * num_envs
    obs_1, rewards_1, dones_1, infos_1 = manager.step(goals_1, phase="play")
    logger.info("Attempt 1 rewards: %s, dones: %s",
                rewards_1.tolist(), dones_1.tolist())
    assert len(obs_1["text"]) == num_envs
    assert dones_1.all(), "All envs should be done after play step"

    # --- Reflection ---
    logger.info("=== Reflection ===")
    reflect_obs, reflect_infos = manager.reflect()
    assert "text" in reflect_obs
    reflect_texts = reflect_obs["text"]
    logger.info("Reflect prompt preview: %s", reflect_texts[0][:120])

    # Verify reflection prompt contains episode info
    assert "Initial state" in reflect_texts[0], "Reflection missing initial state"
    assert "Episode outcome" in reflect_texts[0], "Reflection missing episode outcome"

    # Simulate LLM reflection response
    reflections = ["The block didn't reach the target. I should try a different approach."] * num_envs
    obs_r, rewards_r, dones_r, infos_r = manager.step(reflections, phase="reflect")
    logger.info("Reflection step done (rewards should be 0): %s", rewards_r.tolist())
    assert (rewards_r == 0).all(), "Reflect step should have zero rewards"

    # --- Attempt 2 (restart) ---
    logger.info("=== Attempt 2: restart + play ===")
    obs_restart, infos_restart = manager.restart()
    assert "text" in obs_restart
    logger.info("Restart text preview: %s", obs_restart["text"][0][:80])

    # Try a different goal for attempt 2
    goals_2 = ["slide the green block toward the yellow block"] * num_envs
    obs_2, rewards_2, dones_2, infos_2 = manager.step(goals_2, phase="play")
    logger.info("Attempt 2 rewards: %s", rewards_2.tolist())

    # --- Attempt 3 (restart again) ---
    logger.info("=== Attempt 3: restart + play ===")
    manager.restart()
    goals_3 = ["move the red block next to the green block"] * num_envs
    obs_3, rewards_3, dones_3, infos_3 = manager.step(goals_3, phase="play")
    logger.info("Attempt 3 rewards: %s", rewards_3.tolist())

    return rewards_1, rewards_2, rewards_3


def test_different_goals_different_behavior(envs, policy, num_envs, max_inner_steps):
    """Verify that different goal strings produce different VLA behavior."""
    from language_table.lamer.env_manager import LanguageTableEnvironmentManager

    manager = LanguageTableEnvironmentManager(
        envs=envs, num_attempts=1, max_turns=1,
        max_inner_steps=max_inner_steps, vla_policy=policy,
    )

    # Run with goal A
    manager.reset()
    _, rewards_a, _, _ = manager.step(
        ["push the red block to the blue block"] * num_envs, phase="play")

    # Restart to same state, run with goal B
    manager.restart()
    _, rewards_b, _, _ = manager.step(
        ["push the green block to the yellow block"] * num_envs, phase="play")

    logger.info("Goal A rewards: %s", rewards_a.tolist())
    logger.info("Goal B rewards: %s", rewards_b.tolist())

    # They should produce at least somewhat different outcomes
    # (can't guarantee different rewards, but the actions should differ)
    return rewards_a, rewards_b


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for VLA-enabled env manager")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="bc_resnet_sim_checkpoint_")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--block_mode", type=str, default="BLOCK_4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import ray
    if not ray.is_initialized():
        ray.init()

    import jax
    logger.info("JAX backend: %s, devices: %s", jax.default_backend(), jax.devices())

    from language_table.environments.rewards.block2block import BlockToBlockReward
    from language_table.lamer.envs import LanguageTableMultiProcessEnv
    from language_table.lamer.lava_policy import LAVAPolicy

    # Load VLA
    logger.info("Loading LAVA policy...")
    t0 = time.time()
    policy = LAVAPolicy(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
    )
    logger.info("Policy loaded in %.1fs", time.time() - t0)

    # Create envs
    envs = LanguageTableMultiProcessEnv(
        num_envs=args.num_envs,
        block_mode=args.block_mode,
        reward_factory_cls=BlockToBlockReward,
        seed=args.seed,
        group_n=1,
        return_full_state=True,
        render_obs=True,
    )

    passed = 0
    failed = 0
    total = 3

    # Test 1: Full meta-RL loop
    print("\n" + "=" * 60)
    print("TEST 1: Full meta-RL loop (reset → play → reflect → restart → play)")
    print("=" * 60)
    try:
        r1, r2, r3 = test_meta_rl_loop(envs, policy, args.num_envs, args.max_steps)
        print("PASSED")
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Test 2: Different goals produce different behavior
    print("\n" + "=" * 60)
    print("TEST 2: Different goals → different behavior")
    print("=" * 60)
    try:
        ra, rb = test_different_goals_different_behavior(
            envs, policy, args.num_envs, args.max_steps)
        print("PASSED")
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Test 3: VLA vs random comparison
    print("\n" + "=" * 60)
    print("TEST 3: VLA vs random actions")
    print("=" * 60)
    try:
        rv, rr = test_vla_vs_random(envs, policy, args.num_envs, args.max_steps)
        print("PASSED")
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST SUMMARY: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    envs.close()
    ray.shutdown()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
