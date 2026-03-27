"""v3 — Vectorized batch assembly with contiguous ring buffers.

Changes only the frame-buffer storage and batch-assembly phase.
Preprocessing is unchanged (sequential TF, same as baseline) so any speedup
is purely from eliminating per-env ``deque → list → np.stack`` and the
inner token-tiling loop.

Key changes:
  * Frame buffers are a single pre-allocated (num_envs, seq_len, H, W, 3) array.
  * A ring-buffer write index per env avoids deque overhead.
  * Token tiling uses broadcasting instead of a Python loop.
"""

from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from .base import (
    BatchBuilder,
    DATA_TARGET_HEIGHT,
    DATA_TARGET_WIDTH,
)
from .v0_baseline import preprocess_rgb_tf


class VectorizedAssemblyBatchBuilder(BatchBuilder):
    """Optimization: contiguous ring buffer + vectorized token tiling."""

    name = "v3_vectorized_assembly"

    def reset(self, num_envs: int) -> None:
        sl = self.sequence_length
        self._num_envs = num_envs
        self._frames = np.zeros(
            (num_envs, sl, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3),
            dtype=np.float32,
        )
        self._write_idx = np.zeros(num_envs, dtype=np.int64)
        self._filled = np.zeros(num_envs, dtype=bool)

    def _update_frame_buffer(self, env_idx: int, rgb_uint8: np.ndarray) -> None:
        frame = preprocess_rgb_tf(rgb_uint8)
        if not self._filled[env_idx]:
            # Tile first frame across all seq positions
            self._frames[env_idx, :] = frame[np.newaxis, :]
            self._filled[env_idx] = True
            self._write_idx[env_idx] = 0
        else:
            wi = self._write_idx[env_idx]
            self._frames[env_idx, wi] = frame
            self._write_idx[env_idx] = (wi + 1) % self.sequence_length

    def build_batch(self, goals, obs_list, active_mask):
        batch_size = len(goals)

        # --- Phase 1: update frame buffers (sequential TF — unchanged) ---
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                continue
            self._update_frame_buffer(i, rgb)

        # --- Phase 2: assemble RGB batch via vectorized ring-buffer read ---
        sl = self.sequence_length
        offsets = np.arange(sl)  # (seq_len,)
        read_order = (offsets[np.newaxis, :] + self._write_idx[:batch_size, np.newaxis]) % sl
        env_idx = np.arange(batch_size)[:, np.newaxis]
        rgb_batch = self._frames[env_idx, read_order]  # (B, seq_len, H, W, 3)

        # Zero out inactive envs
        inactive = ~active_mask[:batch_size]
        if inactive.any():
            rgb_batch[inactive] = 0.0

        # --- Phase 2b: vectorized token tiling ---
        clip_batch = np.zeros(
            (batch_size, sl, 77), dtype=np.int32,
        )
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            tokens = self._tokenize(goals[i])
            clip_batch[i, :] = tokens[np.newaxis, :]  # broadcast across seq_len

        return {
            "rgb": jnp.array(rgb_batch),
            "instruction_tokenized_clip": jnp.array(clip_batch),
        }

    def get_frame_state(self) -> List[Optional[np.ndarray]]:
        """Reconstruct chronological frame order for comparison."""
        sl = self.sequence_length
        out: List[Optional[np.ndarray]] = []
        for i in range(self._num_envs):
            if not self._filled[i]:
                out.append(None)
                continue
            wi = self._write_idx[i]
            order = (np.arange(sl) + wi) % sl
            out.append(self._frames[i, order].copy())
        return out
