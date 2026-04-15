"""Hand-designed 8-cell patterns for rendering letters/digits with BLOCK_8.

Each character is drawn on a 3-col x 5-row grid where each "X" cell maps to one
of the 8 blocks. Patterns use <=8 cells; any leftover blocks go to fixed
parking positions far off the table so they don't show up in the render.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Grid config
# ---------------------------------------------------------------------------

GRID_COLS = 3
GRID_ROWS = 5
CELL_SIZE = 0.05  # meters between adjacent cell centers

# Workspace centers (see language_table/environments/constants.py).
CENTER_X = 0.375
CENTER_Y = 0.0

# Parking spots for blocks not used by the current letter. Far off the table
# (matches the env's own "hide a block" convention at line ~840 of
# language_table.py), so parked blocks don't show up in the render.
PARK_POSITIONS: List[Tuple[float, float]] = [
    (5.0, 5.0),
    (5.2, 5.0),
    (5.4, 5.0),
    (5.6, 5.0),
    (5.8, 5.0),
    (6.0, 5.0),
    (6.2, 5.0),
    (6.4, 5.0),
]


# ---------------------------------------------------------------------------
# Glyph patterns (<= 8 filled cells each)
# ---------------------------------------------------------------------------
#
# "X" = filled cell (one block), "." = empty. Many glyphs are compromised:
# in a 3x5 grid with only 8 pixels, several characters (H, M, B, E, R, S, 8,
# etc.) cannot be rendered unambiguously. Tune these freely — the generator
# and loader stay in sync as long as each pattern has <=8 "X" cells.

PATTERNS: Dict[str, List[str]] = {
    # Letters A-Z
    "A": [".X.", "X.X", "XXX", "X.X", "..."],
    "B": ["XX.", "X.X", "XX.", "X.X", "..."],
    "C": ["XXX", "X..", "X..", "X..", "XX."],
    "D": ["XX.", "X.X", "X.X", "XX.", "..."],
    "E": ["XX.", "X..", "XX.", "X..", "XX."],
    "F": ["XXX", "X..", "XX.", "X..", "X.."],
    "G": [".XX", "X..", "X.X", "X.X", ".X."],
    "H": ["X.X", "XXX", "X.X", "...", "..."],
    "I": ["XXX", ".X.", ".X.", "XXX", "..."],
    "J": [".XX", "..X", "..X", "X.X", "XX."],
    "K": ["X.X", "X..", "XX.", "X..", "X.X"],
    "L": ["X..", "X..", "X..", "X..", "XXX"],
    "M": ["X.X", ".X.", "X.X", "X.X", "..."],
    "N": ["X.X", "XX.", ".XX", "X.X", "..."],
    "O": [".X.", "X.X", "X.X", "X.X", ".X."],
    "P": ["XX.", "X.X", "XX.", "X..", "X.."],
    "Q": [".X.", "X.X", "X.X", "X.X", "..X"],
    "R": ["XX.", "X.X", "XX.", "X..", ".X."],
    "S": [".XX", "X..", ".X.", "..X", "XX."],
    "T": ["XXX", ".X.", ".X.", ".X.", ".X."],
    "U": ["X.X", "X.X", "X.X", "X.X", "..."],
    "V": ["X.X", "X.X", "X.X", ".X.", "..."],
    "W": ["X.X", "X.X", "XXX", ".X.", "..."],
    "X": ["X.X", ".X.", ".X.", ".X.", "X.X"],
    "Y": ["X.X", "X.X", ".X.", ".X.", ".X."],
    "Z": ["XX.", "..X", ".X.", "X..", "XXX"],
    # Digits 0-9
    "0": [".X.", "X.X", "X.X", "X.X", ".X."],
    "1": [".X.", "XX.", ".X.", ".X.", "XXX"],
    "2": ["XX.", "..X", ".X.", "X..", "XXX"],
    "3": ["XX.", "..X", "XX.", "..X", "XX."],
    "4": ["X.X", "X.X", "XXX", "..X", "..."],
    "5": ["XX.", "X..", "XX.", "..X", "XX."],
    "6": [".X.", "X..", "XX.", "X.X", "XX."],
    "7": ["XXX", "..X", ".X.", ".X.", ".X."],
    "8": [".X.", "X.X", "XX.", "X.X", ".X."],
    "9": ["XX.", "X.X", "XX.", "..X", ".X."],
}


# ---------------------------------------------------------------------------
# Grid <-> world mapping
# ---------------------------------------------------------------------------

def pattern_cells(pattern: List[str]) -> List[Tuple[int, int]]:
    """Return list of (col, row) tuples of filled cells, in reading order."""
    if len(pattern) != GRID_ROWS:
        raise ValueError(f"pattern must have {GRID_ROWS} rows, got {len(pattern)}")
    cells = []
    for row, rowstr in enumerate(pattern):
        if len(rowstr) != GRID_COLS:
            raise ValueError(
                f"row {row} must have {GRID_COLS} cols, got {len(rowstr)}"
            )
        for col, ch in enumerate(rowstr):
            if ch == "X":
                cells.append((col, row))
            elif ch != ".":
                raise ValueError(f"unknown cell char {ch!r}")
    return cells


def cell_to_world(col: int, row: int) -> Tuple[float, float]:
    """Grid (col, row) -> (x, y) workspace coords.

    The camera looks back along -x, so we flip only the row axis (so row 0
    appears at the top of the image). The col axis maps to world y in its
    natural direction (col 0 = left of glyph = left of image); flipping it
    too would mirror asymmetric letters like L, J, R, S, Z, 2, 3, 5, 6, 7.
    """
    col_center = (GRID_COLS - 1) / 2.0
    row_center = (GRID_ROWS - 1) / 2.0
    x = CENTER_X - (row_center - row) * CELL_SIZE
    y = CENTER_Y + (col - col_center) * CELL_SIZE
    return (x, y)


def character_positions(
    character: str, block_set: Tuple[str, ...]
) -> Dict[str, Tuple[float, float]]:
    """Assign the 8 blocks to grid cells (arbitrary order) + parking spots."""
    if character not in PATTERNS:
        raise KeyError(f"no pattern defined for {character!r}")
    cells = pattern_cells(PATTERNS[character])
    if len(cells) > len(block_set):
        raise ValueError(
            f"pattern for {character!r} uses {len(cells)} cells, "
            f"only {len(block_set)} blocks available"
        )
    positions: Dict[str, Tuple[float, float]] = {}
    for i, block in enumerate(block_set):
        if i < len(cells):
            positions[block] = cell_to_world(*cells[i])
        else:
            positions[block] = PARK_POSITIONS[i - len(cells)]
    return positions


def validate_all_patterns(max_cells: int = 8) -> None:
    """Sanity-check every pattern: shape, char set, and cell count."""
    for ch, pat in PATTERNS.items():
        cells = pattern_cells(pat)
        if len(cells) > max_cells:
            raise ValueError(
                f"pattern for {ch!r} has {len(cells)} cells (> {max_cells})"
            )


if __name__ == "__main__":
    validate_all_patterns()
    print(f"OK: {len(PATTERNS)} patterns, all <= 8 cells")
    for ch, pat in PATTERNS.items():
        n = sum(row.count("X") for row in pat)
        print(f"  {ch}: {n} cells")
