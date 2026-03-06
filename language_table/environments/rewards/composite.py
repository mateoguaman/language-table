# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Composite reward that randomly selects from all task families on each reset.

On each reset(), one of the sub-reward classes is chosen uniformly at random.
That sub-reward handles instruction generation and reward computation until
the next reset().

Methods like get_current_task_info that only some sub-rewards define are
delegated dynamically via __getattr__, so that hasattr() on the composite
returns True only when the active sub-reward actually has the method.
"""

from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards.block2block import BlockToBlockReward
from language_table.environments.rewards.block2absolutelocation import BlockToAbsoluteLocationReward
from language_table.environments.rewards.block2relativelocation import BlockToRelativeLocationReward
from language_table.environments.rewards.block2block_relative_location import BlockToBlockRelativeLocationReward
from language_table.environments.rewards.point2block import PointToBlockReward
from language_table.environments.rewards.separate_blocks import SeparateBlocksReward

ALL_REWARD_CLASSES = [
    BlockToBlockReward,
    PointToBlockReward,
    BlockToRelativeLocationReward,
    BlockToAbsoluteLocationReward,
    BlockToBlockRelativeLocationReward,
    SeparateBlocksReward,
]


class CompositeReward(base_reward.LanguageTableReward):
  """Composite reward that delegates to a randomly chosen sub-reward."""

  def __init__(self, goal_reward, rng, delay_reward_steps, block_mode,
               reward_classes=None):
    super(CompositeReward, self).__init__(
        goal_reward, rng, delay_reward_steps, block_mode)
    classes = reward_classes or ALL_REWARD_CLASSES
    self._sub_rewards = [
        cls(goal_reward=goal_reward,
            rng=rng,
            delay_reward_steps=delay_reward_steps,
            block_mode=block_mode)
        for cls in classes
    ]
    self._active = None

  def seed(self, rng):
    self._rng = rng
    for sub in self._sub_rewards:
      sub.seed(rng)

  def reset(self, state, blocks_on_table):
    """Pick a random sub-reward and delegate reset to it."""
    idx = self._rng.randint(len(self._sub_rewards))
    self._active = self._sub_rewards[idx]
    return self._active.reset(state, blocks_on_table)

  def reward(self, state):
    if self._active is None:
      raise ValueError('must call .reset before .reward')
    return self._active.reward(state)

  def get_goal_region(self):
    if self._active is None:
      return None, None
    return self._active.get_goal_region()

  def __getattr__(self, name):
    """Delegate unknown attributes to the active sub-reward.

    This makes hasattr(composite, 'get_current_task_info') return True
    only when the active sub-reward actually defines it. The env's
    _compute_state() checks hasattr before calling, so this ensures
    sub-rewards that don't track dynamic targets (e.g. BlockToBlockReward)
    are handled correctly.
    """
    # Avoid infinite recursion for attributes accessed during __init__
    if name.startswith('_'):
      raise AttributeError(name)
    active = self.__dict__.get('_active')
    if active is not None and hasattr(active, name):
      return getattr(active, name)
    raise AttributeError(
        f"'{type(self).__name__}' has no attribute '{name}' "
        f"(active sub-reward: {type(active).__name__ if active else None})")
