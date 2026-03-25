"""v5 — JAX GPU-side preprocessing.

Upload raw uint8 images to GPU in a single transfer (4× smaller than float32),
then run float conversion, crop, and resize entirely in JAX on the GPU.

This avoids CPU-side preprocessing entirely and reduces host→device transfer.

NOTE: ``jax.image.resize`` may produce slightly different values from
``tf.image.resize`` at sub-pixel boundaries.
"""

from collections import deque
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


def jax_preprocess_rgb_batch(images_uint8: np.ndarray) -> jnp.ndarray:
    """Preprocess a batch on GPU using JAX ops.

    Parameters
    ----------
    images_uint8 : (N, H, W, 3) uint8 numpy array

    Returns
    -------
    (N, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3) float32 jax array (on device)
    """
    n, h, w, c = images_uint8.shape
    off_h, off_w, crop_h, crop_w = compute_crop_params(h, w)

    # Upload uint8 to device — 4× smaller transfer than float32
    images = jnp.array(images_uint8)

    # Float conversion on device
    images = images.astype(jnp.float32) / 255.0

    # Crop — just a device-side slice
    images = images[:, off_h:off_h + crop_h, off_w:off_w + crop_w, :]

    # Resize on GPU
    images = jax.image.resize(
        images,
        (n, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, c),
        method="bilinear",
    )
    return images


class JaxGpuBatchBuilder(BatchBuilder):
    """Optimization: preprocess images on GPU using JAX."""

    name = "v5_jax_gpu"

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

        # --- Phase 1b: JAX GPU preprocess → back to CPU for frame buffers ---
        if raw_images:
            stacked = np.stack(raw_images, axis=0)
            processed_jax = jax_preprocess_rgb_batch(stacked)
            processed = np.asarray(processed_jax)

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
