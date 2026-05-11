"""Geometric T-tetromino reward from Language Table state dict (no vision/LLM).

Expects keys produced by ``LanguageTable._compute_state`` / worker with
``return_full_state=True``:
    block_<name>_translation  : (2,) float array [x, y] in metres
    block_<name>_mask         : (1,) float in [0, 1]

The table uses a 3x3 grid from ``normalize_workspace_xy`` (see
``language_table.environments.workspace_xy``).
Two T-tetromino orientations are accepted (top-heavy and bottom-heavy):

    Top-heavy T:        Bottom-heavy T:
      X X X               O O O
      O X O               X X X
      O O O               O X O

    cells: {(0,0),(0,1),(0,2),(1,1)}    cells: {(1,0),(1,1),(1,2),(2,1)}

Usage
-----
>>> from tetris_shape_reward import tetromino_t_reward_from_state
>>> reward = tetromino_t_reward_from_state(state_obs)
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Set, Tuple

import numpy as np

from language_table.environments.workspace_xy import normalize_workspace_xy

Pair = Tuple[int, int]

# Only two T orientations accepted (row-aligned, top-heavy or bottom-heavy).
# Grid: col from xn (board top→bottom); row from yn (low yn = board right).
_T_TOP: FrozenSet[Pair] = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
_T_BOTTOM: FrozenSet[Pair] = frozenset({(1, 0), (1, 1), (1, 2), (2, 1)})

T_VALID: Set[FrozenSet[Pair]] = {_T_TOP, _T_BOTTOM}


def _to_grid_cell(xn: float, yn: float) -> Pair:
    """Map normalized coords to (row, col) in 3x3 grid. Clamps to [0, 2]."""
    col = int(np.clip(int(xn * 3), 0, 2))
    row = int(np.clip(int(yn * 3), 0, 2))
    return (row, col)


def tetromino_t_reward_from_state(
    state_obs: Dict[str, np.ndarray],
    mask_threshold: float = 0.5,
) -> float:
    """Return 1.0 if on-table blocks form an accepted T-tetromino, else 0.0.

    Checks only the two row-aligned T orientations (top-heavy / bottom-heavy)
    on a 3x3 grid derived from normalised workspace coordinates.

    Parameters
    ----------
    state_obs
        Full state dict from ``LanguageTable._compute_state`` (``return_full_state=True``).
        Keys: ``block_<name>_translation`` (shape ``(2,)``), ``block_<name>_mask`` (shape ``(1,)``).
    mask_threshold
        Blocks with ``mask[0] >= threshold`` are considered on-table.

    Returns
    -------
    float
        1.0 on success, 0.0 otherwise.
    """
    cells: list[Pair] = []
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
        xn, yn = normalize_workspace_xy(float(xy[0]), float(xy[1]))
        cells.append(_to_grid_cell(xn, yn))

    if len(cells) != 4:
        return 0.0

    cell_set: FrozenSet[Pair] = frozenset(cells)

    # Duplicate grid cells → blocks on top of each other → no valid shape.
    if len(cell_set) != 4:
        return 0.0

    return 1.0 if cell_set in T_VALID else 0.0
