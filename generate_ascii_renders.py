"""Render each ASCII letter/digit by arranging BLOCK_8 blocks on the table.

Outputs:
  <output_dir>/positions.json    — per-character block (x, y) positions
  <output_dir>/renders/<CH>.png  — rendered scene with the arm hidden

Iteration flow:
  # After editing ascii_shapes.PATTERNS: regenerate JSON & renders
  ./ltvenv/bin/python generate_ascii_renders.py --regenerate

  # Later sessions (JSON is source of truth): edit JSON directly and re-render
  ./ltvenv/bin/python generate_ascii_renders.py

  # Re-render one character
  ./ltvenv/bin/python generate_ascii_renders.py --only A
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from language_table.environments import blocks as blocks_module
from language_table.environments import language_table
from language_table.environments.rewards import block2block

import ascii_shapes


PositionsPerChar = Dict[str, Dict[str, Tuple[float, float]]]


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def _metadata(block_set: Tuple[str, ...]) -> dict:
    return {
        "block_mode": "BLOCK_8",
        "block_set": list(block_set),
        "grid": {
            "cols": ascii_shapes.GRID_COLS,
            "rows": ascii_shapes.GRID_ROWS,
            "cell_size": ascii_shapes.CELL_SIZE,
            "center_x": ascii_shapes.CENTER_X,
            "center_y": ascii_shapes.CENTER_Y,
            "row_axis": "row 0 = low x (renders at top of image)",
            "col_axis": "col 0 = low y (renders on left of image)",
        },
        "park_positions": [list(p) for p in ascii_shapes.PARK_POSITIONS],
    }


def build_positions(block_set: Tuple[str, ...]) -> PositionsPerChar:
    return {
        ch: ascii_shapes.character_positions(ch, block_set)
        for ch in ascii_shapes.PATTERNS
    }


def save_json(path: str, positions: PositionsPerChar, block_set: Tuple[str, ...]) -> None:
    data = {
        "metadata": _metadata(block_set),
        "characters": {
            ch: {
                "pattern": ascii_shapes.PATTERNS.get(ch),
                "positions": {b: list(xy) for b, xy in pos.items()},
            }
            for ch, pos in positions.items()
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Tuple[PositionsPerChar, dict]:
    with open(path) as f:
        data = json.load(f)
    positions: PositionsPerChar = {
        ch: {b: tuple(xy) for b, xy in entry["positions"].items()}
        for ch, entry in data["characters"].items()
    }
    return positions, data["metadata"]


# ---------------------------------------------------------------------------
# Block placement
# ---------------------------------------------------------------------------

def place_blocks(env, positions: Dict[str, Tuple[float, float]]) -> None:
    """Teleport each block to (x, y, 0) on the table. No sim step."""
    pb = env._pybullet_client
    # Match the flat-on-side orientation the env uses (pi/2 about x, 0 yaw).
    flat_quat = pb.getQuaternionFromEuler([math.pi / 2, 0, 0])
    for block_name, xy in positions.items():
        block_id = env._block_to_pybullet_id[block_name]
        x, y = float(xy[0]), float(xy[1])
        pb.resetBasePositionAndOrientation(block_id, [x, y, 0.0], flat_quat)


def _safe_filename(ch: str) -> str:
    if ch.isalnum():
        return f"{ch}.png"
    return f"u{ord(ch):04x}.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_REPO_ROOT, "ascii_output"),
        help="directory for positions.json and renders/",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rebuild positions.json from ascii_shapes.PATTERNS (overwrites existing)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="render only these characters (default: all)",
    )
    args = parser.parse_args()

    ascii_shapes.validate_all_patterns()

    json_path = os.path.join(args.output_dir, "positions.json")
    renders_dir = os.path.join(args.output_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)

    block_set = blocks_module.FIXED_8_COMBINATION

    if args.regenerate or not os.path.exists(json_path):
        positions = build_positions(block_set)
        save_json(json_path, positions, block_set)
        print(f"[json] wrote {json_path}")
    else:
        positions, _ = load_json(json_path)
        print(f"[json] loaded {json_path}")

    if args.only:
        missing = [c for c in args.only if c not in positions]
        if missing:
            raise SystemExit(f"unknown chars requested: {missing}")
        positions = {c: positions[c] for c in args.only}

    env = language_table.LanguageTable(
        block_mode=blocks_module.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0,
    )
    env.reset()

    for ch, pos in positions.items():
        place_blocks(env, pos)
        img = env.render_no_arm()
        fname = _safe_filename(ch)
        plt.imsave(os.path.join(renders_dir, fname), img)
        print(f"[render] {ch} -> {fname}")


if __name__ == "__main__":
    main()
