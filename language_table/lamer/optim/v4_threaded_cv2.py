"""v4 — Thread-pool parallelism with OpenCV resize.

Uses ``concurrent.futures.ThreadPoolExecutor`` to resize images in parallel
across CPU cores.  OpenCV releases the GIL for its C++ kernels, so threads
achieve true parallelism on the resize step.

Float conversion and crop are done with vectorized NumPy (zero-copy slice),
then each cropped image is resized via cv2 in a separate thread.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
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


def _resize_one(image_f32: np.ndarray) -> np.ndarray:
    """Resize a single (H, W, 3) float32 image with OpenCV."""
    return cv2.resize(
        image_f32,
        (DATA_TARGET_WIDTH, DATA_TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


def threaded_preprocess_rgb(
    images_uint8: np.ndarray,
    max_workers: int = 8,
) -> np.ndarray:
    """Preprocess a batch of images using NumPy + threaded OpenCV resize.

    Parameters
    ----------
    images_uint8 : (N, H, W, 3) uint8 array
    max_workers : Number of threads for parallel resize.

    Returns
    -------
    (N, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3) float32 array
    """
    n, h, w, _ = images_uint8.shape
    off_h, off_w, crop_h, crop_w = compute_crop_params(h, w)

    cropped = images_uint8[:, off_h:off_h + crop_h, off_w:off_w + crop_w, :]
    cropped = cropped.astype(np.float32) * (1.0 / 255.0)

    if crop_h == DATA_TARGET_HEIGHT and crop_w == DATA_TARGET_WIDTH:
        return np.ascontiguousarray(cropped)

    workers = min(max_workers, n)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resized = list(pool.map(_resize_one, [cropped[i] for i in range(n)]))

    return np.stack(resized, axis=0)


class ThreadedCv2BatchBuilder(BatchBuilder):
    """Optimization: parallel OpenCV resize via thread pool."""

    name = "v4_threaded_cv2"

    def __init__(self, max_workers: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_workers = max_workers

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

        # --- Phase 1b: threaded preprocess ---
        if raw_images:
            stacked = np.stack(raw_images, axis=0)
            processed = threaded_preprocess_rgb(stacked, self.max_workers)

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
