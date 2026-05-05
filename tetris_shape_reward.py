"""Geometric T-tetromino reward from Language Table state dict (no vision/LLM).

Expects keys produced by ``LanguageTable._compute_state`` / worker with
``return_full_state=True``:
    block_<name>_translation  : (2,) float array [x, y] in metres
    block_<name>_mask         : (1,) float in [0, 1]

The reward is 1.0 when the four on-table blocks snap onto a grid that matches
any 90° rotation of the T tetromino, 0.0 otherwise.

Usage
-----
>>> from tetris_shape_reward import tetromino_t_reward_from_state
>>> reward = tetromino_t_reward_from_state(state_obs)
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import numpy as np

Pair = Tuple[int, int]


# ---------------------------------------------------------------------------
# Grid / rotation helpers
# ---------------------------------------------------------------------------

def _normalize(cells: Iterable[Pair]) -> FrozenSet[Pair]:
    """Translate so that min row == 0 and min col == 0."""
    cells = list(cells)
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _rotate_ccw(cells: Iterable[Pair]) -> Set[Pair]:
    """90° counter-clockwise rotation: (r, c) → (-c, r)."""
    return {(-c, r) for r, c in cells}


def _all_rotations(base: Set[Pair]) -> Set[FrozenSet[Pair]]:
    seen: Set[FrozenSet[Pair]] = set()
    cur = set(base)
    for _ in range(4):
        cur = _rotate_ccw(cur)
        seen.add(_normalize(cur))
    return seen


# T tetromino:   □
#              □ □ □
# canonical offsets (row, col), 0-indexed with top of stem at row 0:
_T_BASE: Set[Pair] = {(0, 1), (1, 0), (1, 1), (1, 2)}

T_ROTATIONS: Set[FrozenSet[Pair]] = _all_rotations(_T_BASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tetromino_t_reward_from_state(
    state_obs: Dict[str, np.ndarray],
    cell_size: float = 0.045,
    mask_threshold: float = 0.5,
) -> float:
    """Return 1.0 if on-table blocks form a T tetromino (any rotation), else 0.0.

    Parameters
    ----------
    state_obs
        Full state dict from ``LanguageTable._compute_state`` (``return_full_state=True``).
        Keys: ``block_<name>_translation`` (shape ``(2,)``), ``block_<name>_mask`` (shape ``(1,)``).
    cell_size
        Grid quantisation spacing in metres. Must match the typical centre-to-centre
        distance between adjacent blocks in the sim (~0.04–0.05 m).
    mask_threshold
        Blocks with ``mask[0] >= threshold`` are considered on-table.

    Returns
    -------
    float
        1.0 on success, 0.0 otherwise. Scale to 100.0 if matching TetrisTaskProvider.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    xs: List[float] = []
    ys: List[float] = []
    for key in state_obs:
        m = re.match(r"^block_(.+)_translation$", key)
        if not m:
            continue
        name = m.group(1)
        mask = state_obs.get(f"block_{name}_mask")
        if mask is None or float(np.asarray(mask).reshape(-1)[0]) < mask_threshold:
            continue
        xy = np.asarray(state_obs[key], dtype=np.float64).reshape(-1)
        if xy.size < 2:
            continue
        xs.append(float(xy[0]))
        ys.append(float(xy[1]))

    if len(xs) != 4:
        return 0.0

    xy = np.stack([xs, ys], axis=1)
    g = np.rint(xy / float(cell_size)).astype(np.int64)

    # Duplicate grid cells → blocks on top of each other → no valid shape.
    if np.unique(g, axis=0).shape[0] != 4:
        return 0.0

    cells = _normalize((int(r), int(c)) for r, c in g)
    return 1.0 if cells in T_ROTATIONS else 0.0
