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

"""Reward for sorting blocks by color into the four table corners.

V1 (SortColorsToCornersFixedReward): fixed color-to-corner mapping.
V2 (SortColorsToCornersReward): randomised mapping stated in the instruction.
"""

import collections
from typing import Dict, List, Optional, Tuple

from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np

# ---------------------------------------------------------------------------
# Corner coordinates (same buffered bounds as block2absolutelocation)
# ---------------------------------------------------------------------------
X_BUFFER = 0.025
X_MIN = 0.15 - X_BUFFER
X_MAX = 0.60 - X_BUFFER
Y_MIN = -0.3048
Y_MAX = 0.3048

CORNER_LOCATIONS = {
    'top_left':     np.array([X_MIN, Y_MIN]),
    'top_right':    np.array([X_MIN, Y_MAX]),
    'bottom_left':  np.array([X_MAX, Y_MIN]),
    'bottom_right': np.array([X_MAX, Y_MAX]),
}

CORNER_NAMES = list(CORNER_LOCATIONS.keys())

CORNER_SYNONYMS = {
    'top_left':     ['top left corner', 'upper left corner', 'top left'],
    'top_right':    ['top right corner', 'upper right corner', 'top right'],
    'bottom_left':  ['bottom left corner', 'lower left corner', 'bottom left'],
    'bottom_right': ['bottom right corner', 'lower right corner',
                     'bottom right'],
}

CORNER_DISTANCE_THRESHOLD = 0.115

FIXED_COLOR_TO_CORNER = {
    'red':    'top_left',
    'blue':   'top_right',
    'green':  'bottom_left',
    'yellow': 'bottom_right',
}

ALL_COLORS = list(FIXED_COLOR_TO_CORNER.keys())

# ---------------------------------------------------------------------------
# Instruction helpers
# ---------------------------------------------------------------------------
INSTRUCTION_VERBS = ['push', 'move', 'slide', 'put']

FIXED_INSTRUCTION_PREFIXES = [
    'sort all blocks by color into the corners:',
    'group each color into its own corner:',
    'push blocks of the same color to the same corner:',
    'organize the blocks so each corner has one color:',
]


def _build_color_corner_clause(color_to_corner, rng):
    """Build the 'red to top left, blue to ...' part of an instruction."""
    clauses = []
    for color in ALL_COLORS:
        if color not in color_to_corner:
            continue
        corner = color_to_corner[color]
        corner_text = rng.choice(CORNER_SYNONYMS[corner])
        clauses.append(f'{color} blocks to the {corner_text}')
    if len(clauses) <= 2:
        return ' and '.join(clauses)
    return ', '.join(clauses[:-1]) + ', and ' + clauses[-1]


def _build_fixed_instruction(color_to_corner, rng):
    """Build a V1 instruction with a generic prefix + explicit assignments."""
    prefix = rng.choice(FIXED_INSTRUCTION_PREFIXES)
    body = _build_color_corner_clause(color_to_corner, rng)
    return f'{prefix} {body}'


def _build_dynamic_instruction(color_to_corner, rng):
    """Build a V2 instruction that explicitly states each colour→corner."""
    verb = rng.choice(INSTRUCTION_VERBS)
    body = _build_color_corner_clause(color_to_corner, rng)
    return f'{verb} {body}'


# ---------------------------------------------------------------------------
# Helpers shared by both variants
# ---------------------------------------------------------------------------

def _group_blocks_by_color(blocks_on_table):
    """Return {color: [block_id, ...]} for blocks present on the table."""
    groups = collections.defaultdict(list)
    for block in blocks_on_table:
        color = block.split('_')[0]
        groups[color].append(block)
    return dict(groups)


def _build_block_to_target(color_to_corner, color_groups):
    """Map every block id to its target corner coordinate."""
    block_to_target = {}
    for color, corner_name in color_to_corner.items():
        if color not in color_groups:
            continue
        target = CORNER_LOCATIONS[corner_name]
        for block in color_groups[color]:
            block_to_target[block] = np.copy(target)
    return block_to_target


def _count_blocks_in_place(state, block_to_target, get_translation_fn):
    """Return the number of blocks within threshold of their target."""
    placed = 0
    for block, target in block_to_target.items():
        pos = np.array(get_translation_fn(block, state))
        if np.linalg.norm(pos - target) < CORNER_DISTANCE_THRESHOLD:
            placed += 1
    return placed


def _all_blocks_in_place(state, block_to_target, get_translation_fn):
    """Return True iff every block is within threshold of its target."""
    return _count_blocks_in_place(
        state, block_to_target, get_translation_fn) == len(block_to_target)


def _first_misplaced_block(state, block_to_target, get_translation_fn):
    """Return (block, target) for the first block not yet in its corner."""
    for block, target in block_to_target.items():
        pos = np.array(get_translation_fn(block, state))
        if np.linalg.norm(pos - target) >= CORNER_DISTANCE_THRESHOLD:
            return block, target
    return None, None


def _any_block_already_in_place(state, block_to_target, get_translation_fn):
    """True if at least one block starts inside its target corner."""
    for block, target in block_to_target.items():
        pos = np.array(get_translation_fn(block, state))
        if np.linalg.norm(pos - target) < CORNER_DISTANCE_THRESHOLD:
            return True
    return False


# ===================================================================
# V1 – Fixed colour→corner mapping
# ===================================================================

class SortColorsToCornersFixedReward(base_reward.LanguageTableReward):
    """Reward for sorting blocks by colour into fixed corners."""

    def __init__(self, goal_reward, rng, delay_reward_steps, block_mode):
        super().__init__(goal_reward=goal_reward, rng=rng,
                         delay_reward_steps=delay_reward_steps,
                         block_mode=block_mode)
        self._instruction = None
        self._color_to_corner = None
        self._block_to_target = None

    # ----- reset / reset_to -------------------------------------------

    def reset(self, state, blocks_on_table):
        color_groups = _group_blocks_by_color(blocks_on_table)
        color_to_corner = {
            c: FIXED_COLOR_TO_CORNER[c]
            for c in color_groups if c in FIXED_COLOR_TO_CORNER
        }
        if len(color_to_corner) < 2:
            return task_info.FAILURE

        block_to_target = _build_block_to_target(color_to_corner, color_groups)
        if _any_block_already_in_place(
                state, block_to_target, self._get_translation_for_block):
            return task_info.FAILURE

        return self.reset_to(state, color_to_corner, blocks_on_table)

    def reset_to(self, state, color_to_corner, blocks_on_table):
        self._color_to_corner = dict(color_to_corner)
        color_groups = _group_blocks_by_color(blocks_on_table)
        self._block_to_target = _build_block_to_target(
            self._color_to_corner, color_groups)
        self._instruction = self._sample_instruction()
        self._in_reward_zone_steps = 0
        return self.get_current_task_info(state)

    def _sample_instruction(self):
        return _build_fixed_instruction(self._color_to_corner, self._rng)

    # ----- task info --------------------------------------------------

    def get_current_task_info(self, state):
        return task_info.SortColorsToCornersTaskInfo(
            instruction=self._instruction,
            color_to_corner=dict(self._color_to_corner),
            block_to_target=dict(self._block_to_target),
        )

    # ----- reward -----------------------------------------------------

    @property
    def target_translation(self):
        if self._block_to_target:
            return next(iter(self._block_to_target.values()))
        return None

    def get_goal_region(self):
        if self._block_to_target is None:
            return None, None
        return self.target_translation, CORNER_DISTANCE_THRESHOLD

    def reward(self, state):
        return self.reward_for(state, self._block_to_target)

    def reward_for(self, state, block_to_target):
        reward = 0.0
        done = False
        if _all_blocks_in_place(
                state, block_to_target, self._get_translation_for_block):
            if self._in_reward_zone_steps >= self._delay_reward_steps:
                reward = self._goal_reward
                done = True
            else:
                logging.info('In reward zone for %d steps',
                             self._in_reward_zone_steps)
                self._in_reward_zone_steps += 1
        else:
            self._in_reward_zone_steps = 0
        return reward, done

    def reward_for_info(self, state, info):
        return self.reward_for(state, info.block_to_target)

    def debug_info(self, state):
        block, target = _first_misplaced_block(
            state, self._block_to_target, self._get_translation_for_block)
        if block is None:
            return 0.0
        pos = np.array(self._get_translation_for_block(block, state))
        return float(np.linalg.norm(pos - target))


# ===================================================================
# V2 – Dynamic (randomised) colour→corner mapping
# ===================================================================

class SortColorsToCornersReward(SortColorsToCornersFixedReward):
    """Reward with a randomly assigned colour→corner mapping per episode.

    The mapping is included in the natural-language instruction so the
    agent knows which colour goes where.
    """

    def reset(self, state, blocks_on_table):
        color_groups = _group_blocks_by_color(blocks_on_table)
        colors_present = sorted(c for c in color_groups if c in ALL_COLORS)

        if len(colors_present) < 2:
            return task_info.FAILURE

        corners = list(CORNER_NAMES)
        self._rng.shuffle(corners)
        color_to_corner = {
            color: corners[i] for i, color in enumerate(colors_present)
        }

        block_to_target = _build_block_to_target(color_to_corner, color_groups)
        if _any_block_already_in_place(
                state, block_to_target, self._get_translation_for_block):
            return task_info.FAILURE

        return self.reset_to(state, color_to_corner, blocks_on_table)

    def _sample_instruction(self):
        return _build_dynamic_instruction(self._color_to_corner, self._rng)


# ===================================================================
# V3 – Dynamic mapping with one-shot per-block reward
# ===================================================================

class SortColorsToCornersPartialReward(SortColorsToCornersReward):
    """Like V2 but awards goal_reward/n_blocks once per newly placed block.

    Each block triggers a one-time reward of goal_reward/n_blocks when it
    first enters its target corner.  Blocks that leave and re-enter are
    NOT rewarded again (prevents accumulation across inner steps).
    done=True only fires when ALL blocks are currently in their corners
    (after delay_reward_steps consecutive qualifying steps).
    """

    def reset_to(self, state, color_to_corner, blocks_on_table):
        self._rewarded_blocks = set()
        return super().reset_to(state, color_to_corner, blocks_on_table)

    def reward_for(self, state, block_to_target):
        n_blocks = len(block_to_target)
        per_block_reward = self._goal_reward / n_blocks

        currently_placed = set()
        newly_rewarded = 0
        for block, target in block_to_target.items():
            pos = np.array(self._get_translation_for_block(block, state))
            if np.linalg.norm(pos - target) < CORNER_DISTANCE_THRESHOLD:
                currently_placed.add(block)
                if block not in self._rewarded_blocks:
                    self._rewarded_blocks.add(block)
                    newly_rewarded += 1

        reward = newly_rewarded * per_block_reward
        done = False

        if len(currently_placed) == n_blocks:
            if self._in_reward_zone_steps >= self._delay_reward_steps:
                done = True
            else:
                logging.info('In reward zone for %d steps',
                             self._in_reward_zone_steps)
                self._in_reward_zone_steps += 1
        else:
            self._in_reward_zone_steps = 0

        return reward, done
