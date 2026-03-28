"""Multi-step block-to-location reward with configurable locations, shapes, colors.

On each reset(), the reward samples n_steps distinct (block, location) pairs.
Blocks are referenced in the instruction by shape only, color only, or both,
depending on which of shapes/colors is set (None = omit from instruction).

Partial reward: goal_reward / n_steps per block when it first reaches its
target.  done=True only when ALL blocks are simultaneously in their target
locations (after delay_reward_steps).
"""

from typing import List, Optional, Set

from language_table.environments.rewards import task_info
from language_table.environments.rewards.block2absolutelocation import (
    ABSOLUTE_LOCATIONS,
    LOCATION_SYNONYMS,
)
from language_table.environments.rewards.sort_colors_to_corners import (
    CORNER_DISTANCE_THRESHOLD,
    SortColorsToCornersPartialReward,
    _any_block_already_in_place,
)
from language_table.environments.rewards.synonyms import (
    PREPOSITIONS,
    PUSH_VERBS,
    get_block_synonyms,
)
import numpy as np


MULTISTEP_VERBS = PUSH_VERBS + ['place the', 'nudge the', 'drag the']
JOINERS = ['and', 'then', 'and then', ', then']


def _block_descriptor(block, describe_by, blocks_on_table, rng):
    """Return a randomized text reference for *block*."""
    color, shape = block.split('_')
    if describe_by == 'shape':
        return shape
    if describe_by == 'color':
        return f'{color} block'
    return rng.choice(get_block_synonyms(block, blocks_on_table))


def _build_multistep_instruction(block_loc_pairs, describe_by,
                                 blocks_on_table, rng):
    """Build a randomized natural-language instruction."""
    verb = rng.choice(MULTISTEP_VERBS)
    clauses = []
    for i, (block, loc_name) in enumerate(block_loc_pairs):
        block_text = _block_descriptor(block, describe_by,
                                       blocks_on_table, rng)
        loc_text = rng.choice(LOCATION_SYNONYMS[loc_name])
        prep = rng.choice(PREPOSITIONS)
        if i == 0:
            clauses.append(f'{verb} {block_text} {prep} {loc_text}')
        else:
            clauses.append(f'the {block_text} {prep} {loc_text}')

    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        joiner = rng.choice(JOINERS)
        return f'{clauses[0]} {joiner} {clauses[1]}'
    joiner = rng.choice(JOINERS)
    return ', '.join(clauses[:-1]) + f', {joiner} ' + clauses[-1]


def make_multistep_reward(
    locations: Optional[List[str]] = None,
    shapes: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    n_steps: int = 2,
):
    """Factory returning a configured multi-step reward class.

    Parameters
    ----------
    locations : list[str] | None
        Pool of location names (keys of ABSOLUTE_LOCATIONS).  Each reset
        samples n_steps locations without replacement.  Defaults to all 9.
    shapes : list[str] | None
        If set, only blocks whose shape is in this list are eligible targets
        and the instruction references blocks by shape (color omitted).
    colors : list[str] | None
        If set, only blocks whose color is in this list are eligible targets
        and the instruction references blocks by color (shape omitted).
        When both shapes and colors are set the instruction uses both.
    n_steps : int
        Number of (block, location) sub-goals per episode.

    Returns
    -------
    type
        A reward class compatible with the LanguageTable reward_factory
        interface.
    """
    _locations = list(locations) if locations else list(ABSOLUTE_LOCATIONS.keys())
    _shapes: Optional[Set[str]] = set(shapes) if shapes else None
    _colors: Optional[Set[str]] = set(colors) if colors else None
    _n_steps = n_steps

    if _shapes is not None and _colors is None:
        _describe_by = 'shape'
    elif _colors is not None and _shapes is None:
        _describe_by = 'color'
    else:
        _describe_by = 'full'

    class MultiStepBlockToLocationReward(SortColorsToCornersPartialReward):

        def reset(self, state, blocks_on_table):
            if _shapes is not None:
                eligible = [b for b in blocks_on_table
                            if b.split('_')[1] in _shapes]
            elif _colors is not None:
                eligible = [b for b in blocks_on_table
                            if b.split('_')[0] in _colors]
            else:
                eligible = list(blocks_on_table)

            if len(eligible) < _n_steps:
                return task_info.FAILURE
            if len(_locations) < _n_steps:
                return task_info.FAILURE

            selected_blocks = list(self._rng.choice(
                eligible, _n_steps, replace=False))
            locs = list(_locations)
            self._rng.shuffle(locs)
            selected_locs = locs[:_n_steps]

            block_to_target = {}
            block_to_loc_name = {}
            for block, loc in zip(selected_blocks, selected_locs):
                block_to_target[block] = np.array(
                    ABSOLUTE_LOCATIONS[loc], dtype=np.float64)
                block_to_loc_name[block] = loc

            if _any_block_already_in_place(
                    state, block_to_target, self._get_translation_for_block):
                return task_info.FAILURE

            return self._reset_to_multistep(
                state, block_to_target, block_to_loc_name, blocks_on_table)

        def _reset_to_multistep(self, state, block_to_target,
                                block_to_loc_name, blocks_on_table):
            self._block_to_target = dict(block_to_target)
            self._color_to_corner = dict(block_to_loc_name)
            self._instruction = _build_multistep_instruction(
                list(block_to_loc_name.items()), _describe_by,
                blocks_on_table, self._rng)
            self._in_reward_zone_steps = 0
            return self.get_current_task_info(state)

        def get_current_task_info(self, state):
            return task_info.SortColorsToCornersTaskInfo(
                instruction=self._instruction,
                color_to_corner=dict(self._color_to_corner),
                block_to_target=dict(self._block_to_target),
            )

    return MultiStepBlockToLocationReward
