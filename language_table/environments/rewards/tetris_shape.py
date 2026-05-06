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

"""Defines block2absolutelocation reset and reward."""
import enum

from typing import Any, List
from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np


# There's a small offset in the Y direction to subtract.
# The red dots represent the bounds of the arm, which are not exactly in the
# center of the boards.
# This should only matter for this reward, which deals with absolute locations.
X_BUFFER = 0.025

X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
X_MIN = X_MIN_REAL - X_BUFFER
X_MAX = X_MAX_REAL - X_BUFFER
Y_MIN = Y_MIN_REAL
Y_MAX = Y_MAX_REAL
CENTER_X = (X_MAX - X_MIN) / 2. + X_MIN
CENTER_Y = (Y_MAX - Y_MIN)/2. + Y_MIN

BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE = 0.15
BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE = 0.15


class Locations(enum.Enum):
  TOP = 'top'
  TOP_LEFT = 'top_left'
  TOP_RIGHT = 'top_right'
  CENTER = 'center'
  CENTER_LEFT = 'center_left'
  CENTER_RIGHT = 'center_right'
  BOTTOM = 'bottom'
  BOTTOM_LEFT = 'bottom_left'
  BOTTOM_RIGHT = 'bottom_right'


ABSOLUTE_LOCATIONS = {
    'top': [X_MIN, CENTER_Y],
    'top_left': [X_MIN, Y_MIN],
    'top_right': [X_MIN, Y_MAX],
    'center': [CENTER_X, CENTER_Y],
    'center_left': [CENTER_X, Y_MIN],
    'center_right': [CENTER_X, Y_MAX],
    'bottom': [X_MAX, CENTER_Y],
    'bottom_left': [X_MAX, Y_MIN],
    'bottom_right': [X_MAX, Y_MAX],
}

tetris_shape = np.array([
    [1, 1, 1],
    [0, 1, 0]])


def generate_3x3_configurations(shape):
  """Returns all 3x3 padded rotations of a 2x3 shape.

  A 2x3 shape has two valid 3x3 placements for each of its four rotations:
  padding above/below for 2x3 orientations and padding left/right for 3x2
  orientations.
  """
  shape = np.asarray(shape)
  if shape.shape != (2, 3):
    raise ValueError('shape must be a 2x3 array.')

  configurations = []
  for num_rotations in range(4):
    rotated = np.rot90(shape, k=num_rotations)
    if rotated.shape == (2, 3):
      configurations.append(np.pad(rotated, ((0, 1), (0, 0))))
      configurations.append(np.pad(rotated, ((1, 0), (0, 0))))
    else:
      configurations.append(np.pad(rotated, ((0, 0), (0, 1))))
      configurations.append(np.pad(rotated, ((0, 0), (1, 0))))
  return configurations


def _translation_to_grid_index(translation):
  """Returns a 3x3 grid index if translation is in an absolute-location cell."""
  translation = np.array(translation)
  location_names = np.array([
      ['top_left', 'top', 'top_right'],
      ['center_left', 'center', 'center_right'],
      ['bottom_left', 'bottom', 'bottom_right'],
  ])
  target_locations = np.array([
      [[X_MIN, Y_MIN], [X_MIN, CENTER_Y], [X_MIN, Y_MAX]],
      [[CENTER_X, Y_MIN], [CENTER_X, CENTER_Y], [CENTER_X, Y_MAX]],
      [[X_MAX, Y_MIN], [X_MAX, CENTER_Y], [X_MAX, Y_MAX]],
  ])
  distances = np.linalg.norm(target_locations - translation, axis=2)
  row, col = np.unravel_index(np.argmin(distances), distances.shape)

  if location_names[row, col] == Locations.CENTER.value:
    target_dist = BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE
  else:
    target_dist = BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE

  if distances[row, col] < target_dist:
    return row, col
  return None


def blocks_to_3x3_array(state, blocks_on_table):
  """Returns a 3x3 occupancy grid for the blocks on the board.

  Rows are ordered top-to-bottom and columns are ordered left-to-right. For
  example, blocks in bottom-left, bottom, bottom-right, and center return:
  [[0, 0, 0],
   [0, 1, 0],
   [1, 1, 1]]
  """
  grid = np.zeros((3, 3), dtype=np.int32)
  for block in blocks_on_table:
    translation = state['block_%s_translation' % block]
    grid_index = _translation_to_grid_index(translation)
    if grid_index is not None:
      row, col = grid_index
      grid[row, col] = 1
  return grid


def generate_all_instructions(block_mode):
  """Generate all instructions for block2relativeposition."""
  all_instructions = ["make a tetris T shape out of the blocks"]

  return all_instructions


class TetrisShapeReward(base_reward.LanguageTableReward):
  """Calculates reward/instructions for 'push block to absolute location'."""

  def __init__(self, goal_reward, rng, delay_reward_steps,
               block_mode):
    super(TetrisShapeReward, self).__init__(
        goal_reward=goal_reward,
        rng=rng,
        delay_reward_steps=delay_reward_steps,
        block_mode=block_mode)
    self._block = None
    self._instruction = None
    self._location = None
    self._target_translation = None
    self._configurations = generate_3x3_configurations(tetris_shape)

  def _sample_instruction(
      self, block, blocks_on_table, location):
    """Randomly sample a task involving two objects."""
    
    return "make a tetris T shape out of the blocks"

  def reset(self, state, blocks_on_table):
    """Chooses new target block and location."""
    self._block = self._sample_object(blocks_on_table) # so we don't get error
    self._location = self._rng.choice(list(sorted(ABSOLUTE_LOCATIONS.keys()))) # so we don't get error
    self._target_translation = np.copy(ABSOLUTE_LOCATIONS[self._location])
    self._instruction = self._sample_instruction(self._block, blocks_on_table, self._location)
    info = self.get_current_task_info(state)
    self._blocks_on_table = blocks_on_table
    # If the state of the board already triggers the reward, try to reset
    # again with a new configuration.
    # if self._in_goal_region(state, self._block, self._target_translation):
    #   # Try again with a new board configuration.
    #   return task_info.FAILURE
    return info

  def reset_to(
      self, state, block, location, blocks_on_table):
    """Reset to a particular task definition."""
    self._block = self._sample_object(blocks_on_table) # so we don't get error
    self._location = self._rng.choice(list(sorted(ABSOLUTE_LOCATIONS.keys()))) # so we don't get error
    self._target_translation = np.copy(ABSOLUTE_LOCATIONS[self._location])
    self._instruction = self._sample_instruction(self._block, blocks_on_table, self._location)

    info = self.get_current_task_info(state)
    self._blocks_on_table = blocks_on_table
    return info

  @property
  def target_translation(self):
    return self._target_translation

  def get_goal_region(self):
    if self._location == Locations.CENTER.value:
      return self._target_translation, BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE
    return self._target_translation, BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE

  def reward(self, state):
    """Calculates reward given state."""
    grid = blocks_to_3x3_array(state, self._blocks_on_table)
    for configuration in self._configurations:
      if np.all(grid == configuration):
        return self._goal_reward, True
    return 0.0, False

  def reward_for(
      self, state, pushing_block, target_translation):
    """Returns 1. if pushing_block is in location."""
    self.reward(state)

  def reward_for_info(self, state, info):
    return self.reward(state)

  def debug_info(self, state):
    """Returns 1. if pushing_block is in location."""
    # Get current location of the target block.
    current_translation, _ = self._get_pose_for_block(
        self._block, state)
    # Compute distance between current translation and target.
    dist = np.linalg.norm(
        np.array(current_translation) - np.array(self._target_translation))
    return dist

  def get_current_task_info(self, state):
    return task_info.Block2LocationTaskInfo(
        instruction=self._instruction,
        block=self._block,
        location=self._location,
        target_translation=self._target_translation)
