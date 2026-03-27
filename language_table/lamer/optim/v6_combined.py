"""v6 — Combined optimizations.

Three combined variants that pair the best preprocessing strategy with the
best assembly strategy:

  v6a : batched NumPy+CV2 + ring buffer + vectorized tokens
  v6b : threaded CV2     + ring buffer + vectorized tokens
  v6c : JAX GPU          + ring buffer + vectorized tokens
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .base import (
    BatchBuilder,
    DATA_TARGET_HEIGHT,
    DATA_TARGET_WIDTH,
    compute_crop_params,
)
from .v2_numpy_cv2 import batch_preprocess_rgb_np
from .v4_threaded_cv2 import threaded_preprocess_rgb
from .v5_jax_gpu import jax_preprocess_rgb_batch


# ---------------------------------------------------------------------------
# Shared ring-buffer mixin
# ---------------------------------------------------------------------------

class _RingBufferMixin:
    """Mixin providing ring-buffer frame storage + vectorized assembly."""

    def _ring_reset(self, num_envs: int, sequence_length: int) -> None:
        self._num_envs = num_envs
        self._sl = sequence_length
        self._frames = np.zeros(
            (num_envs, sequence_length, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3),
            dtype=np.float32,
        )
        self._write_idx = np.zeros(num_envs, dtype=np.int64)
        self._filled = np.zeros(num_envs, dtype=bool)

    def _ring_update(self, env_idx: int, frame: np.ndarray) -> None:
        if not self._filled[env_idx]:
            self._frames[env_idx, :] = frame[np.newaxis, :]
            self._filled[env_idx] = True
            self._write_idx[env_idx] = 0
        else:
            wi = self._write_idx[env_idx]
            self._frames[env_idx, wi] = frame
            self._write_idx[env_idx] = (wi + 1) % self._sl

    def _ring_assemble(self, batch_size, goals, active_mask, tokenize_fn):
        sl = self._sl
        offsets = np.arange(sl)
        read_order = (offsets[np.newaxis, :] + self._write_idx[:batch_size, np.newaxis]) % sl
        env_idx = np.arange(batch_size)[:, np.newaxis]
        rgb_batch = self._frames[env_idx, read_order]

        inactive = ~active_mask[:batch_size]
        if inactive.any():
            rgb_batch[inactive] = 0.0

        clip_batch = np.zeros((batch_size, sl, 77), dtype=np.int32)
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            tokens = tokenize_fn(goals[i])
            clip_batch[i, :] = tokens[np.newaxis, :]

        return {
            "rgb": jnp.array(rgb_batch),
            "instruction_tokenized_clip": jnp.array(clip_batch),
        }

    def _ring_frame_state(self) -> List[Optional[np.ndarray]]:
        sl = self._sl
        out: List[Optional[np.ndarray]] = []
        for i in range(self._num_envs):
            if not self._filled[i]:
                out.append(None)
                continue
            wi = self._write_idx[i]
            order = (np.arange(sl) + wi) % sl
            out.append(self._frames[i, order].copy())
        return out


# ---------------------------------------------------------------------------
# v6a: NumPy+CV2 batch preprocessing + ring buffer
# ---------------------------------------------------------------------------

class CombinedNpCv2RingBuilder(_RingBufferMixin, BatchBuilder):
    """NumPy+CV2 batch preprocessing + ring buffer + vectorized tokens."""

    name = "v6a_np_cv2_ring"

    def reset(self, num_envs: int) -> None:
        self._ring_reset(num_envs, self.sequence_length)

    def build_batch(self, goals, obs_list, active_mask):
        batch_size = len(goals)

        active_indices = []
        raw_images = []
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                continue
            active_indices.append(i)
            raw_images.append(rgb)

        if raw_images:
            processed = batch_preprocess_rgb_np(np.stack(raw_images, axis=0))
            for j, env_i in enumerate(active_indices):
                self._ring_update(env_i, processed[j])

        return self._ring_assemble(batch_size, goals, active_mask, self._tokenize)

    def get_frame_state(self):
        return self._ring_frame_state()


# ---------------------------------------------------------------------------
# v6b: Threaded CV2 preprocessing + ring buffer
# ---------------------------------------------------------------------------

class CombinedThreadedRingBuilder(_RingBufferMixin, BatchBuilder):
    """Threaded CV2 preprocessing + ring buffer + vectorized tokens."""

    name = "v6b_threaded_ring"

    def __init__(self, max_workers: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_workers = max_workers

    def reset(self, num_envs: int) -> None:
        self._ring_reset(num_envs, self.sequence_length)

    def build_batch(self, goals, obs_list, active_mask):
        batch_size = len(goals)

        active_indices = []
        raw_images = []
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                continue
            active_indices.append(i)
            raw_images.append(rgb)

        if raw_images:
            processed = threaded_preprocess_rgb(
                np.stack(raw_images, axis=0), self.max_workers
            )
            for j, env_i in enumerate(active_indices):
                self._ring_update(env_i, processed[j])

        return self._ring_assemble(batch_size, goals, active_mask, self._tokenize)

    def get_frame_state(self):
        return self._ring_frame_state()


# ---------------------------------------------------------------------------
# v6c: JAX GPU preprocessing + ring buffer
# ---------------------------------------------------------------------------

class CombinedJaxRingBuilder(_RingBufferMixin, BatchBuilder):
    """JAX GPU preprocessing + ring buffer + vectorized tokens."""

    name = "v6c_jax_ring"

    def reset(self, num_envs: int) -> None:
        self._ring_reset(num_envs, self.sequence_length)

    def build_batch(self, goals, obs_list, active_mask):
        batch_size = len(goals)

        active_indices = []
        raw_images = []
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                continue
            active_indices.append(i)
            raw_images.append(rgb)

        if raw_images:
            processed_jax = jax_preprocess_rgb_batch(np.stack(raw_images, axis=0))
            processed = np.asarray(processed_jax)
            for j, env_i in enumerate(active_indices):
                self._ring_update(env_i, processed[j])

        return self._ring_assemble(batch_size, goals, active_mask, self._tokenize)

    def get_frame_state(self):
        return self._ring_frame_state()
