"""Geometric T-tetromino reward from Language Table state dict (no vision/LLM).

Expects keys produced by ``LanguageTable._compute_state`` / worker with
``return_full_state=True``:
    block_<name>_translation  : (2,) float array [x, y] in metres
    block_<name>_mask         : (1,) float in [0, 1]

Normalized coordinates match ``language_table.environments.workspace_xy`` /
``prompt.py`` (same as ``tetris_shape_reward.py``):

    ``col = clip(int(xn * 3))`` from ``xn`` (workspace **x**: board **top→bottom**,
    camera ~top→bottom on RGB).

    ``row = clip(int(yn * 3))`` from ``yn`` with ``yn = (Y_MAX - y) / (Y_MAX-Y_MIN)``
    (workspace **y** flipped; camera ~right→left; ``block2absolutelocation`` **left**
    ↔ large ``yn``).

    Grid cell ``(row, col)`` uses those thirds; row 0 at low ``yn`` is board **right**.

Two T-tetromino orientations are accepted (bar at image-top col or image-middle col).
Grid: row=yn-strip (0=right, 2=left), col=xn-strip (0=img-top, 2=img-bottom).
Bar runs horizontally in the camera image (across yn / row axis).

    Top T (bar at col=0):    Middle T (bar at col=1):
      col: 0 1 2               col: 0 1 2
      row0: X . .              row0: . X .
      row1: X X .   stem→      row1: . X X   stem→
      row2: X . .              row2: . X .

    cells: {(0,0),(1,0),(2,0),(1,1)}    cells: {(0,1),(1,1),(2,1),(1,2)}

Dense reward (dense=True, default)
-----------------------------------
Uses optimal one-to-one assignment (Hungarian algorithm) between the 4 blocks
and the 4 target cell centers of each valid T orientation.  For each T:

    cost = min over bijections π of  Σ_i ||pos_i - center_{π(i)}||²
    mean_cost = cost / 4   (mean squared distance per matched block)
    score = exp(-mean_cost / (2 * sigma²))

    reward = max over T_VALID of score

This enforces that each block is matched to a **distinct** T cell, preventing
high rewards from blocks clustered on a single cell.

    cell center for (row r, col c): xn = c/3+1/6, yn = r/3+1/6
    world_y = Y_MAX - yn*(Y_MAX-Y_MIN)  (row 0 → world y near +Y_MAX)
    sigma controls sharpness; default 0.15 (≈ half a cell width of 0.333).

Binary reward (dense=False)
----------------------------
Returns 1.0 iff blocks exactly occupy a valid T cell set, else 0.0.

Usage
-----
>>> from opro.geometric_reward import tetromino_t_reward_from_state
>>> reward = tetromino_t_reward_from_state(state_obs)            # dense
>>> reward = tetromino_t_reward_from_state(state_obs, dense=False)  # binary
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from language_table.environments.workspace_xy import normalize_workspace_xy

Pair = Tuple[int, int]

# Grid is 3x3; ``col`` from ``xn`` (image top→bottom); ``row`` from ``yn`` (image right→left).
# T bar runs along the ROW axis (yn spread = horizontal in camera image).
# "Top" T: bar at col=0 (image top), stem at col=1 center.
# "Middle" T: bar at col=1 (image middle), stem at col=2 center.
_T_TOP: FrozenSet[Pair] = frozenset({(0, 0), (1, 0), (2, 0), (1, 1)})
_T_BOTTOM: FrozenSet[Pair] = frozenset({(0, 1), (1, 1), (2, 1), (1, 2)})

T_VALID: Set[FrozenSet[Pair]] = {_T_TOP, _T_BOTTOM}

# Cell centers in normalized [0,1] as (xn, yn): xn from col, yn from row.
_T_TOP_CENTERS: np.ndarray = np.array(
    [(c / 3.0 + 1.0 / 6.0, r / 3.0 + 1.0 / 6.0) for r, c in sorted(_T_TOP)],
    dtype=np.float64,
)  # shape (4, 2)
_T_BOTTOM_CENTERS: np.ndarray = np.array(
    [(c / 3.0 + 1.0 / 6.0, r / 3.0 + 1.0 / 6.0) for r, c in sorted(_T_BOTTOM)],
    dtype=np.float64,
)  # shape (4, 2)

_T_VALID_CENTERS: List[np.ndarray] = [_T_TOP_CENTERS, _T_BOTTOM_CENTERS]

_DENSE_SIGMA: float = 0.15  # ~half a cell width


def _to_grid_cell(xn: float, yn: float) -> Pair:
    """Map normalized coords to (row, col); see module docstring."""
    col = int(np.clip(int(xn * 3), 0, 2))
    row = int(np.clip(int(yn * 3), 0, 2))
    return (row, col)


def _candidate_cells(xn: float, yn: float, margin: float) -> List[Pair]:
    """Return all grid cells a block could belong to given boundary margin.

    A block within ``margin`` (in normalized units) of a cell boundary is
    considered a candidate for both neighbouring cells.  At most 2×2 = 4
    candidates per block; typically 1.
    """
    def axis_candidates(v: float) -> List[int]:
        primary = int(np.clip(int(v * 3), 0, 2))
        cands = [primary]
        for boundary in (1.0 / 3.0, 2.0 / 3.0):
            if abs(v - boundary) < margin:
                neighbour = primary + (1 if v < boundary else -1)
                neighbour = int(np.clip(neighbour, 0, 2))
                if neighbour not in cands:
                    cands.append(neighbour)
        return cands

    col_cands = axis_candidates(xn)
    row_cands = axis_candidates(yn)
    return [(r, c) for r in row_cands for c in col_cands]


def _assignment_dense_reward(
    positions: np.ndarray,
    sigma: float = _DENSE_SIGMA,
) -> float:
    """Assignment-based reward in [0, 1] for a set of block positions.

    Uses the Hungarian algorithm to find the optimal one-to-one matching
    between blocks and target cell centers,     then applies a Gaussian to the **mean** matched squared distance
    (total cost / 4), same per-block scale as the old average-of-Gaussians
    formulation. Unlike nearest-neighbor scoring this penalises
    multiple blocks assigned to the same cell.

    Parameters
    ----------
    positions : (4, 2) array of normalized [0,1] block positions (xn, yn).
    sigma     : Gaussian kernel width.

    Returns
    -------
    float in [0, 1]: max over both T targets of assignment Gaussian score.
    """
    if len(positions) == 0:
        return 0.0

    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    best = 0.0
    for target_centers in _T_VALID_CENTERS:
        # Cost matrix: (N_blocks, 4_centers) squared distances
        diffs = positions[:, None, :] - target_centers[None, :, :]  # (N, 4, 2)
        cost = np.sum(diffs ** 2, axis=-1)                           # (N, 4)
        row_ind, col_ind = linear_sum_assignment(cost)
        total_sq_dist = float(cost[row_ind, col_ind].sum())
        n = float(len(positions))
        mean_sq_dist = total_sq_dist / n
        score = float(np.exp(-mean_sq_dist * inv2s2))
        if score > best:
            best = score
    return min(best, 1.0)


def tetromino_t_reward_from_state(
    state_obs: Dict[str, np.ndarray],
    mask_threshold: float = 0.5,
    dense: bool = True,
    sigma: float = _DENSE_SIGMA,
    boundary_margin: float = 0.05,
) -> float:
    """Reward signal for T-tetromino formation from raw state dict.

    Parameters
    ----------
    state_obs
        Full state dict from ``LanguageTable._compute_state`` (``return_full_state=True``).
        Keys: ``block_<name>_translation`` (shape ``(2,)``), ``block_<name>_mask`` (shape ``(1,)``).
    mask_threshold
        Blocks with ``mask[0] >= threshold`` are considered on-table.
    dense
        If True (default), return Gaussian-based continuous reward in [0, 1].
        If False, return 1.0 only when blocks exactly form a valid T, else 0.0.
    sigma
        Gaussian kernel width for dense mode (default 0.15 ≈ half a cell).
    boundary_margin
        Normalized-coordinate margin around cell boundaries.  A block within
        this distance of a boundary is treated as a candidate for both
        neighbouring cells; all combinations are checked for T validity.
        Default 0.05 ≈ 15 % of a cell width.

    Returns
    -------
    float in [0, 1].
    """
    positions: List[Tuple[float, float]] = []
    candidate_cells: List[List[Pair]] = []

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
        positions.append((xn, yn))
        candidate_cells.append(_candidate_cells(xn, yn, boundary_margin))

    # Check if any combination of candidate cells forms a valid T
    config_achieved = False
    if len(candidate_cells) == 4:
        for combo in itertools.product(*candidate_cells):
            if len(frozenset(combo)) == 4 and frozenset(combo) in T_VALID:
                config_achieved = True
                break
    if config_achieved:
        print("SUCCESS")
        return 1.0

    if dense:
        if not positions:
            return 0.0
        return _assignment_dense_reward(np.array(positions, dtype=np.float64), sigma=sigma)

    # Binary mode
    return 0.0
