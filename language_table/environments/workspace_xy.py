# coding=utf-8
"""Normalize table workspace (x, y) in metres to ``(xn, yn)`` in ``[0, 1]``.

Used by text observations, geometric rewards, and prompts so numbers stay
consistent. Bounds come from ``constants.py`` (same as PyBullet workspace).

World axes (simulator / ``block_*_translation``):

- **x**: along the board from ``X_MIN`` toward ``X_MAX``. Dataset absolute
  locations call the ``X_MIN`` side ``top`` and ``X_MAX`` ``bottom``
  (see ``rewards/block2absolutelocation.ABSOLUTE_LOCATIONS``).

- **y**: across the board from ``Y_MIN`` (``left``) to ``Y_MAX`` (``right``).

Normalized pair (what prompts and logs label ``x`` and ``y``):

- ``xn = (x - X_MIN) / (X_MAX - X_MIN)`` — 0 at board ``top`` (low world-x),
  1 at ``bottom`` (high world-x). In the **default RGB camera**, this tracks
  roughly **top → bottom** on the image.

- ``yn = (Y_MAX - y) / (Y_MAX - Y_MIN)`` — 0 at board ``right`` (world ``y``
  near ``Y_MAX``), 1 at ``left`` (near ``Y_MIN``). In the default camera, this
  tracks roughly **right → left** on the image.

So the earlier wording “x = left→right columns, y = top→bottom rows” was wrong:
those axes are **workspace** axes, not image horizontal/vertical as labels
suggested. Use the camera mapping above for interpreting RGB + numbers together.

3×3 grid for rewards (``opro/geometric_reward``, ``tetris_shape_reward``):
``col`` from ``xn`` thirds, ``row`` from ``yn`` thirds (see ``_to_grid_cell``).
"""

from __future__ import annotations

import textwrap

from language_table.environments.constants import X_MAX, X_MIN, Y_MAX, Y_MIN


TABLE_GRID_PROMPT = textwrap.dedent("""\
The values labeled **x** and **y** in state text are workspace-normalized **xn**, **yn** in [0, 1] (see ``normalize_workspace_xy``), not raw pixels.

For the **default RGB camera**, **x** varies roughly **top→bottom** on the image (small **x** toward dataset/board **top**, low world-X). **y** varies roughly **right→left** (small **y** toward **right**, high world-y; large **y** toward **left**, low world-y). Matches ``block2absolutelocation`` vocabulary (**top** ↔ low X, **left** ↔ low Y).

3×3 grid (thirds of **x** and **y**):
    x ∈ [0.00, 0.33) → top third (camera)    | x ∈ [0.33, 0.67) → middle third | x ∈ [0.67, 1.00] → bottom third
    y ∈ [0.00, 0.33) → right third (camera)  | y ∈ [0.33, 0.67) → middle third | y ∈ [0.67, 1.00] → left third
""")


def normalize_workspace_xy(x: float, y: float) -> tuple[float, float]:
    """Map world metres to normalized ``(xn, yn)``."""
    xn = (x - X_MIN) / (X_MAX - X_MIN)
    yn = (Y_MAX - y) / (Y_MAX - Y_MIN)
    return float(xn), float(yn)
