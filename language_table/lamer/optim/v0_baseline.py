"""v0 — Exact copy of the original _build_batch logic.

This is the reference implementation.  Every other variant is tested against
the outputs of this class.
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


def preprocess_rgb_tf(rgb_uint8: np.ndarray) -> np.ndarray:
    """Per-image preprocessing — exact copy of ``_preprocess_rgb``."""
    image = tf.image.convert_image_dtype(rgb_uint8, dtype=tf.float32)

    raw_h = tf.cast(tf.shape(image)[0], tf.float32)
    raw_w = tf.cast(tf.shape(image)[1], tf.float32)
    scaled_h = raw_h * RANDOM_CROP_FACTOR
    scaled_w = raw_w * RANDOM_CROP_FACTOR
    offset_h = tf.cast((raw_h - scaled_h) // 2, tf.int32)
    offset_w = tf.cast((raw_w - scaled_w) // 2, tf.int32)
    target_h = tf.cast(scaled_h, tf.int32)
    target_w = tf.cast(scaled_w, tf.int32)

    image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, target_h, target_w
    )
    image = tf.image.resize(image, [DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH])
    return image.numpy()


class BaselineBatchBuilder(BatchBuilder):
    """Exact replica of ``LAVAPolicy._build_batch``."""

    name = "v0_baseline"

    def reset(self, num_envs: int) -> None:
        self._frame_buffers: List[deque] = [
            deque(maxlen=self.sequence_length) for _ in range(num_envs)
        ]

    def _update_frame_buffer(self, env_idx: int, rgb_uint8: np.ndarray) -> None:
        frame = preprocess_rgb_tf(rgb_uint8)
        buf = self._frame_buffers[env_idx]
        if len(buf) == 0:
            for _ in range(self.sequence_length):
                buf.append(frame)
        else:
            buf.append(frame)

    def build_batch(self, goals, obs_list, active_mask):
        batch_size = len(goals)

        # --- Phase 1: update frame buffers (sequential TF preprocessing) ---
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                continue
            self._update_frame_buffer(i, rgb)

        # --- Phase 2: assemble batch arrays ---
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

        # --- Phase 3: host → device ---
        return {
            "rgb": jnp.array(rgb_batch),
            "instruction_tokenized_clip": jnp.array(clip_batch),
        }

    def get_frame_state(self) -> List[Optional[np.ndarray]]:
        out = []
        for buf in self._frame_buffers:
            if len(buf) > 0:
                out.append(np.stack(list(buf), axis=0))
            else:
                out.append(None)
        return out
