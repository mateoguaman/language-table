# coding=utf-8
"""Tests for the CompositeReward class.

Run from the language-table repo root:
    ltvenv/bin/python -m pytest language_table/environments/rewards/test_composite.py -v

Or directly:
    ltvenv/bin/python language_table/environments/rewards/test_composite.py
"""

import collections
import unittest

import numpy as np

from language_table.environments import blocks as blocks_module
from language_table.environments.language_table import LanguageTable
from language_table.environments.rewards import task_info
from language_table.environments.rewards.block2block import BlockToBlockReward
from language_table.environments.rewards.block2absolutelocation import BlockToAbsoluteLocationReward
from language_table.environments.rewards.block2relativelocation import BlockToRelativeLocationReward
from language_table.environments.rewards.block2block_relative_location import BlockToBlockRelativeLocationReward
from language_table.environments.rewards.point2block import PointToBlockReward
from language_table.environments.rewards.separate_blocks import SeparateBlocksReward
from language_table.environments.rewards.composite import (
    CompositeReward,
    ALL_REWARD_CLASSES,
)

BLOCK_MODE = blocks_module.LanguageTableBlockVariants.BLOCK_4


def _make_env(reward_cls, seed=42):
    """Create a LanguageTable env with the given reward factory."""
    return LanguageTable(
        block_mode=BLOCK_MODE,
        reward_factory=reward_cls,
        seed=seed,
    )


class TestCompositeReward(unittest.TestCase):
    """Tests for CompositeReward."""

    def test_constructor_creates_all_sub_rewards(self):
        """CompositeReward should instantiate one sub-reward per class."""
        rng = np.random.RandomState(0)
        reward = CompositeReward(
            goal_reward=100.0,
            rng=rng,
            delay_reward_steps=0,
            block_mode=BLOCK_MODE,
        )
        self.assertEqual(len(reward._sub_rewards), len(ALL_REWARD_CLASSES))
        for sub, cls in zip(reward._sub_rewards, ALL_REWARD_CLASSES):
            self.assertIsInstance(sub, cls)

    def test_reset_returns_valid_task_info(self):
        """Each reset should return a valid TaskInfo (not FAILURE)."""
        env = _make_env(CompositeReward, seed=42)
        # Run many resets to exercise different task types
        for i in range(50):
            env.reset()
            self.assertIsNotNone(
                env._instruction_str,
                f"instruction_str was None on reset {i}")
            self.assertTrue(
                len(env._instruction_str) > 0,
                f"instruction_str was empty on reset {i}")

    def test_all_task_types_sampled(self):
        """Over many resets, all 6 task families should appear."""
        env = _make_env(CompositeReward, seed=0)
        task_info_types = set()
        for _ in range(200):
            env.reset()
            # The _task_info attribute is set by _set_task_info
            info = env._task_info
            task_info_types.add(type(info).__name__)

        expected_types = {
            'Block2BlockTaskInfo',
            'Point2BlockTaskInfo',
            'Block2RelativeLocationTaskInfo',
            'Block2LocationTaskInfo',
            'Block2BlockRelativeLocationTaskInfo',
            'SeparateBlocksTaskInfo',
        }
        self.assertEqual(
            task_info_types, expected_types,
            f"Missing task types: {expected_types - task_info_types}")

    def test_reward_returns_float_and_bool(self):
        """reward() should return (float, bool) after reset."""
        env = _make_env(CompositeReward, seed=42)
        for _ in range(10):
            env.reset()
            action = np.array([0.0, 0.0], dtype=np.float32)
            _, reward, done, _ = env.step(action)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)

    def test_seed_propagates_to_sub_rewards(self):
        """seed() should propagate to all sub-rewards."""
        rng = np.random.RandomState(0)
        reward = CompositeReward(
            goal_reward=100.0,
            rng=rng,
            delay_reward_steps=0,
            block_mode=BLOCK_MODE,
        )
        new_rng = np.random.RandomState(99)
        reward.seed(new_rng)
        for sub in reward._sub_rewards:
            self.assertIs(sub._rng, new_rng)

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same sequence of task types."""
        instructions_a = []
        env_a = _make_env(CompositeReward, seed=123)
        for _ in range(20):
            env_a.reset()
            instructions_a.append(env_a._instruction_str)

        instructions_b = []
        env_b = _make_env(CompositeReward, seed=123)
        for _ in range(20):
            env_b.reset()
            instructions_b.append(env_b._instruction_str)

        self.assertEqual(instructions_a, instructions_b)

    def test_instruction_matches_task_type(self):
        """Instructions should contain keywords matching their task type."""
        env = _make_env(CompositeReward, seed=7)
        for _ in range(50):
            env.reset()
            info = env._task_info
            inst = env._instruction_str.lower()

            if isinstance(info, task_info.Point2BlockTaskInfo):
                self.assertTrue(
                    any(kw in inst for kw in ['point', 'move the arm', 'move your arm', 'move next', 'move close', 'move to']),
                    f"Point task instruction missing keywords: {inst}")
            elif isinstance(info, task_info.SeparateBlocksTaskInfo):
                self.assertTrue(
                    any(kw in inst for kw in ['separate', 'pull', 'push']),
                    f"Separate task instruction missing keywords: {inst}")

    def test_custom_reward_classes(self):
        """Passing a subset of reward_classes should work."""
        rng = np.random.RandomState(0)
        reward = CompositeReward(
            goal_reward=100.0,
            rng=rng,
            delay_reward_steps=0,
            block_mode=BLOCK_MODE,
            reward_classes=[BlockToBlockReward, PointToBlockReward],
        )
        self.assertEqual(len(reward._sub_rewards), 2)
        self.assertIsInstance(reward._sub_rewards[0], BlockToBlockReward)
        self.assertIsInstance(reward._sub_rewards[1], PointToBlockReward)


class TestIndividualRewardsInEnv(unittest.TestCase):
    """Sanity-check each reward class works standalone in an env."""

    def _run_reward_smoke_test(self, reward_cls, num_resets=20, num_steps=5):
        """Common smoke test: reset many times, step a few times each."""
        env = _make_env(reward_cls, seed=42)
        success_count = 0
        for i in range(num_resets):
            obs = env.reset()
            self.assertIsNotNone(env._instruction_str,
                                 f"{reward_cls.__name__}: no instruction on reset {i}")
            for _ in range(num_steps):
                action = np.random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)
                obs, reward, done, info = env.step(action)
                self.assertIsInstance(reward, float)
                self.assertIsInstance(done, bool)
                if done:
                    success_count += 1
                    break
        return success_count

    def test_block2block(self):
        self._run_reward_smoke_test(BlockToBlockReward)

    def test_point2block(self):
        self._run_reward_smoke_test(PointToBlockReward)

    def test_block2relativelocation(self):
        self._run_reward_smoke_test(BlockToRelativeLocationReward)

    def test_block2absolutelocation(self):
        self._run_reward_smoke_test(BlockToAbsoluteLocationReward)

    def test_block2block_relative_location(self):
        self._run_reward_smoke_test(BlockToBlockRelativeLocationReward)

    def test_separate_blocks(self):
        self._run_reward_smoke_test(SeparateBlocksReward)

    def test_composite(self):
        self._run_reward_smoke_test(CompositeReward)


class TestCompositeRewardDistribution(unittest.TestCase):
    """Test that task selection is roughly uniform."""

    def test_distribution_is_roughly_uniform(self):
        """Over 600 resets, each of 6 task types should appear ~100 times."""
        rng = np.random.RandomState(42)
        reward = CompositeReward(
            goal_reward=100.0,
            rng=rng,
            delay_reward_steps=0,
            block_mode=BLOCK_MODE,
        )
        # Build a minimal state for reset
        env = _make_env(CompositeReward, seed=42)

        type_counts = collections.Counter()
        n_resets = 600
        for _ in range(n_resets):
            env.reset()
            info = env._task_info
            type_counts[type(info).__name__] += 1

        # Each type should appear at least 30 times out of 600 (5% vs expected ~17%)
        for name in [
            'Block2BlockTaskInfo',
            'Point2BlockTaskInfo',
            'Block2RelativeLocationTaskInfo',
            'Block2LocationTaskInfo',
            'Block2BlockRelativeLocationTaskInfo',
            'SeparateBlocksTaskInfo',
        ]:
            self.assertGreater(
                type_counts[name], 30,
                f"{name} appeared only {type_counts[name]} times in {n_resets} resets")


if __name__ == '__main__':
    unittest.main()
