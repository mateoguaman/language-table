"""v1 — Batched TF operations.

Instead of calling ``tf.image.*`` once per image in a Python for-loop, stack
all active images into a single (N, H, W, 3) tensor and run the three TF ops
just once.  This eliminates ~3×N TF dispatch overheads and ~N ``.numpy()``
sync barriers.
"""

from collections import deque
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from .base import (
    BatchBuilder,
    DATA_TARGET_HEIGHT,
    DATA_TARGET_WIDTH,
    RANDOM_CROP_FACTOR,
)


def batch_preprocess_rgb_tf(images_uint8: np.ndarray) -> np.ndarray:
    """Preprocess a batch of images using a single set of TF ops.

    Replicates the *exact* scalar arithmetic of the per-image baseline:
    ``tf.image.convert_image_dtype`` (not manual division), and the crop
    offsets are computed from the *float* scaled dimensions before int-cast,
    matching the baseline's ``(raw_h - scaled_h) // 2`` path.

    Parameters
    ----------
    images_uint8 : (N, H, W, 3) uint8 array

    Returns
    -------
    (N, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3) float32 array
    """
    # convert_image_dtype(uint8→float32) = cast * (1/255) — matches baseline
    images = tf.image.convert_image_dtype(images_uint8, dtype=tf.float32)

    raw_h = tf.cast(tf.shape(images)[1], tf.float32)
    raw_w = tf.cast(tf.shape(images)[2], tf.float32)
    scaled_h = raw_h * RANDOM_CROP_FACTOR
    scaled_w = raw_w * RANDOM_CROP_FACTOR
    off_h = tf.cast((raw_h - scaled_h) // 2, tf.int32)
    off_w = tf.cast((raw_w - scaled_w) // 2, tf.int32)
    crop_h = tf.cast(scaled_h, tf.int32)
    crop_w = tf.cast(scaled_w, tf.int32)

    # tf.image.crop_to_bounding_box supports 4-D tensors natively
    images = tf.image.crop_to_bounding_box(images, off_h, off_w, crop_h, crop_w)
    images = tf.image.resize(images, [DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH])

    return images.numpy()


class BatchTFBatchBuilder(BatchBuilder):
    """Optimization: batch all TF image ops into a single call."""

    name = "v1_batch_tf"

    def reset(self, num_envs: int) -> None:
        self._frame_buffers: List[deque] = [
            deque(maxlen=self.sequence_length) for _ in range(num_envs)
        ]

    def build_batch(self, goals, obs_list, active_mask):
        batch_size = len(goals)

        # --- Phase 1a: collect raw images for active envs ---
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

        # --- Phase 1b: batch preprocess ---
        if raw_images:
            stacked = np.stack(raw_images, axis=0)
            processed = batch_preprocess_rgb_tf(stacked)

            # --- Phase 1c: update frame buffers ---
            for j, env_i in enumerate(active_indices):
                frame = processed[j]
                buf = self._frame_buffers[env_i]
                if len(buf) == 0:
                    for _ in range(self.sequence_length):
                        buf.append(frame)
                else:
                    buf.append(frame)

        # --- Phase 2: assemble batch arrays (unchanged from baseline) ---
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
