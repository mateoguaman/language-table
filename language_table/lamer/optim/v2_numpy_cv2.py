"""v2 — Replace TF with pure NumPy + OpenCV.

Avoids TF eager-mode dispatch overhead entirely.  Float conversion and crop
are vectorized NumPy ops; resize uses OpenCV (fast native C++).

NOTE: ``cv2.resize`` bilinear interpolation may produce slightly different
results from ``tf.image.resize`` at sub-pixel boundaries.  The test suite
measures both exact and approximate equivalence.
"""

from collections import deque
from typing import Any, Dict, List, Optional

import cv2
import jax.numpy as jnp
import numpy as np

from .base import (
    BatchBuilder,
    DATA_TARGET_HEIGHT,
    DATA_TARGET_WIDTH,
    compute_crop_params,
)


def batch_preprocess_rgb_np(images_uint8: np.ndarray) -> np.ndarray:
    """Preprocess a batch of images using NumPy + OpenCV.

    Parameters
    ----------
    images_uint8 : (N, H, W, 3) uint8 array

    Returns
    -------
    (N, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3) float32 array
    """
    n, h, w, _ = images_uint8.shape
    off_h, off_w, crop_h, crop_w = compute_crop_params(h, w)

    # Vectorized float conversion + crop (single contiguous slice)
    images = images_uint8[:, off_h:off_h + crop_h, off_w:off_w + crop_w, :]
    images = images.astype(np.float32) * (1.0 / 255.0)

    # Resize — cv2 doesn't support batch, but it's extremely fast per image
    if crop_h != DATA_TARGET_HEIGHT or crop_w != DATA_TARGET_WIDTH:
        out = np.empty(
            (n, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3), dtype=np.float32
        )
        for i in range(n):
            out[i] = cv2.resize(
                images[i],
                (DATA_TARGET_WIDTH, DATA_TARGET_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        return out

    return np.ascontiguousarray(images)


class NumpyCv2BatchBuilder(BatchBuilder):
    """Optimization: replace TF image ops with NumPy + OpenCV."""

    name = "v2_numpy_cv2"

    def reset(self, num_envs: int) -> None:
        self._frame_buffers: List[deque] = [
            deque(maxlen=self.sequence_length) for _ in range(num_envs)
        ]

    def build_batch(self, goals, obs_list, active_mask):
        batch_size = len(goals)

        # --- Phase 1a: collect raw images ---
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

        # --- Phase 1b: batch preprocess with NumPy + cv2 ---
        if raw_images:
            stacked = np.stack(raw_images, axis=0)
            processed = batch_preprocess_rgb_np(stacked)

            for j, env_i in enumerate(active_indices):
                frame = processed[j]
                buf = self._frame_buffers[env_i]
                if len(buf) == 0:
                    for _ in range(self.sequence_length):
                        buf.append(frame)
                else:
                    buf.append(frame)

        # --- Phase 2: assemble batch (same as baseline) ---
        rgb_batch = np.zeros(
            (batch_size, self.sequence_length, DATA_TARGET_HEIGHT,
             DATA_TARGET_WIDTH, 3),
            dtype=np.float32,
        )
        clip_batch = np.zeros(
            (batch_size, self.sequence_length, 77), dtype=np.int32,
        )

        for i in range(batch_size):
            if not active_mask[i]:
                continue
            buf = self._frame_buffers[i]
            if len(buf) > 0:
                rgb_batch[i] = np.stack(list(buf), axis=0)
            tokens = self._tokenize(goals[i])
            for t in range(self.sequence_length):
                clip_batch[i, t] = tokens

        return {
            "rgb": jnp.array(rgb_batch),
            "instruction_tokenized_clip": jnp.array(clip_batch),
        }

    def get_frame_state(self) -> List[Optional[np.ndarray]]:
        out = []
        for buf in self._frame_buffers:
            out.append(np.stack(list(buf), axis=0) if len(buf) > 0 else None)
        return out
