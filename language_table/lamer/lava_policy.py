"""
LAVA (Language-Augmented Visual Attention) policy wrapper for batched inference.

Replicates the exact inference pipeline from the original language-table eval:
  1. ClipTokenWrapper — CLIP-tokenizes the instruction string
  2. CentralCropImageWrapper — float32 conversion, central crop (0.95), resize
  3. HistoryWrapper(4, tile_first_step_obs=True) — stacks last 4 observations
  4. BCJaxPyPolicy — expand_dims(0), model.apply, denormalize, clip

See: language_table/eval/main.py, language_table/train/policy.py

Usage:
    policy = LAVAPolicy(checkpoint_dir="/path/to/checkpoints/")
    policy.reset(num_envs=128)
    actions = policy.predict(goals, obs_list, active_mask)
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax.training import checkpoints

from language_table.common import clip_tokenizer
from language_table.train.networks import lava

logger = logging.getLogger(__name__)

# Default model config — matches language_table_resnet_sim_local.py
DEFAULT_MODEL_CONFIG = dict(
    dense_resnet_width=1024,
    dense_resnet_num_blocks=2,
    lava_sequence_length=4,
    lava_num_layers=4,
    lava_temporal_transformer_num_layers=2,
    lava_d_model=128,
    lava_num_heads=2,
    lava_pyramid_fuse_layers=(2, 3, 4),
    lava_image_encoder="resnet",
    lava_lang_encoder="clip",
)

# Preprocessing constants — from language_table_resnet_sim_local.py
DATA_TARGET_WIDTH = 320
DATA_TARGET_HEIGHT = 180
RANDOM_CROP_FACTOR = 0.95

EPS = np.finfo(np.float32).eps


def _preprocess_rgb(rgb_uint8):
    """Preprocess a single RGB image to match the eval pipeline exactly.

    Replicates CentralCropImageWrapper from language_table/eval/wrappers.py:
      1. tf.image.convert_image_dtype(uint8 -> float32)  =>  divide by 255
      2. Central crop with factor 0.95 (matching random crop at eval time)
      3. tf.image.resize to (180, 320) with bilinear interpolation

    Using TF ops to ensure bit-identical results with the original eval code.
    """
    image = tf.image.convert_image_dtype(rgb_uint8, dtype=tf.float32)

    # Central crop — matches crop_test_image() in eval/wrappers.py
    raw_h = tf.cast(tf.shape(image)[0], tf.float32)
    raw_w = tf.cast(tf.shape(image)[1], tf.float32)
    scaled_h = raw_h * RANDOM_CROP_FACTOR
    scaled_w = raw_w * RANDOM_CROP_FACTOR
    offset_h = tf.cast((raw_h - scaled_h) // 2, tf.int32)
    offset_w = tf.cast((raw_w - scaled_w) // 2, tf.int32)
    target_h = tf.cast(scaled_h, tf.int32)
    target_w = tf.cast(scaled_w, tf.int32)
    image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, target_h, target_w)

    # Resize — matches resize_images() in eval/wrappers.py
    image = tf.image.resize(image, [DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH])

    return image.numpy()


def _compute_crop_params(h: int, w: int):
    """Return (offset_h, offset_w, crop_h, crop_w) for a given image size.

    Replicates the TF baseline's float-first arithmetic: offsets are computed
    from the *float* scaled dimensions (before int truncation) so they match
    ``tf.cast((raw_h - scaled_h) // 2, tf.int32)`` exactly.
    """
    scaled_h = float(h) * RANDOM_CROP_FACTOR
    scaled_w = float(w) * RANDOM_CROP_FACTOR
    off_h = int((h - scaled_h) // 2)
    off_w = int((w - scaled_w) // 2)
    crop_h = int(scaled_h)
    crop_w = int(scaled_w)
    return off_h, off_w, crop_h, crop_w


def _batch_preprocess_rgb_tf(images_uint8: np.ndarray) -> np.ndarray:
    """Preprocess a batch of images using a single set of TF ops.

    Bit-exact with the per-image ``_preprocess_rgb``: uses the same
    ``tf.image.convert_image_dtype`` and float-first crop offset arithmetic.

    Parameters
    ----------
    images_uint8 : (N, H, W, 3) uint8 array

    Returns
    -------
    (N, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3) float32 array
    """
    images = tf.image.convert_image_dtype(images_uint8, dtype=tf.float32)

    raw_h = tf.cast(tf.shape(images)[1], tf.float32)
    raw_w = tf.cast(tf.shape(images)[2], tf.float32)
    scaled_h = raw_h * RANDOM_CROP_FACTOR
    scaled_w = raw_w * RANDOM_CROP_FACTOR
    off_h = tf.cast((raw_h - scaled_h) // 2, tf.int32)
    off_w = tf.cast((raw_w - scaled_w) // 2, tf.int32)
    crop_h = tf.cast(scaled_h, tf.int32)
    crop_w = tf.cast(scaled_w, tf.int32)

    images = tf.image.crop_to_bounding_box(images, off_h, off_w, crop_h, crop_w)
    images = tf.image.resize(images, [DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH])

    return images.numpy()


def _jax_preprocess_rgb_batch(images_uint8: np.ndarray) -> np.ndarray:
    """Preprocess a batch on GPU using JAX ops.

    Uploads uint8 images to device (4x smaller than float32), then runs
    float conversion, crop, and bilinear resize entirely on the accelerator.

    Max pixel error vs TF baseline: ~2.4e-7 (float32 epsilon — effectively
    identical; would round to the same uint8 values).

    Parameters
    ----------
    images_uint8 : (N, H, W, 3) uint8 array

    Returns
    -------
    (N, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3) float32 numpy array
    """
    n, h, w, c = images_uint8.shape
    off_h, off_w, crop_h, crop_w = _compute_crop_params(h, w)

    images = jnp.array(images_uint8)
    images = images.astype(jnp.float32) / 255.0
    images = images[:, off_h:off_h + crop_h, off_w:off_w + crop_w, :]
    images = jax.image.resize(
        images,
        (n, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, c),
        method="bilinear",
    )
    return np.asarray(images)


PREPROCESS_MODES = ("original", "batched_tf", "jax_gpu")


def _tokenize_instruction_string(instruction_str, tokenizer):
    """CLIP-tokenize a string instruction.

    Matches ClipTokenWrapper._tokenize() from language_table/eval/wrappers.py,
    which decodes the int32 instruction array to a string then tokenizes.

    Returns (77,) int64 array matching the tokenizer output dtype.
    """
    tokens = clip_tokenizer.tokenize_text(instruction_str, tokenizer)
    return tokens.numpy()[0]  # (77,) int64


class LAVAPolicy:
    """Batched LAVA policy for use as inner-loop VLA.

    Replicates the exact inference pipeline from the original language-table
    evaluation code (eval/main.py + train/policy.py).

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the Flax checkpoint file.
    checkpoint_prefix : str
        Prefix of the checkpoint file (default: 'bc_resnet_sim_checkpoint_').
    model_config : dict, optional
        Model architecture config. Uses resnet sim defaults if not provided.
    sequence_length : int
        Number of frames in the observation history (default: 4).
        Must match the config used during training.
    action_clip : float
        Clip actions to [-action_clip, action_clip] (default: 0.1).
    preprocess_mode : str
        Image preprocessing strategy for ``_build_batch``:
        - ``"original"`` — per-image TF ops (baseline).
        - ``"batched_tf"`` — batch all images into one TF call (default;
          ~2x end-to-end speedup, bit-exact with original).
        - ``"jax_gpu"`` — preprocess on GPU via JAX (~2.1x end-to-end
          speedup, ~6e-4 max action error vs original).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_prefix: str = "bc_resnet_sim_checkpoint_",
        model_config: Optional[dict] = None,
        sequence_length: int = 4,
        action_clip: float = 0.1,
        preprocess_mode: str = "batched_tf",
    ):
        if preprocess_mode not in PREPROCESS_MODES:
            raise ValueError(
                f"preprocess_mode must be one of {PREPROCESS_MODES}, "
                f"got {preprocess_mode!r}")
        self.preprocess_mode = preprocess_mode
        self.sequence_length = sequence_length
        self.action_clip = action_clip
        self._frame_buffers: List[deque] = []

        # Build model — same as eval/main.py line 76
        config = model_config or DEFAULT_MODEL_CONFIG
        self.model = lava.SequenceLAVMSE(action_size=2, **config)

        # Load checkpoint — same as train/policy.py lines 43-47
        logger.info("Loading LAVA checkpoint from %s (prefix=%s)",
                     checkpoint_dir, checkpoint_prefix)
        state_dict = checkpoints.restore_checkpoint(
            checkpoint_dir, None, prefix=checkpoint_prefix)
        if state_dict is None:
            raise ValueError(
                f"Failed to load checkpoint from {checkpoint_dir} "
                f"with prefix={checkpoint_prefix}")

        self.variables = {
            "params": state_dict["params"],
            "batch_stats": state_dict["batch_stats"],
        }

        # Action denormalization stats — same as train/policy.py lines 53-57
        self.action_mean = np.array(
            state_dict["norm_info"]["action_statistics"]["mean"])
        self.action_std = np.array(
            state_dict["norm_info"]["action_statistics"]["std"])

        # Build CLIP tokenizer — same as eval/wrappers.py lines 33-34
        vocab_lookup = clip_tokenizer.create_vocab()
        self._tokenizer = clip_tokenizer.ClipTokenizer(vocab_lookup)

        # Cache for tokenized instructions (instruction_str -> tokens)
        self._token_cache: Dict[str, np.ndarray] = {}

        # JIT-compile the forward pass
        self._forward_jit = jax.jit(self._forward_fn)

        logger.info("LAVA policy loaded (preprocess_mode=%s, action_mean=%s, "
                     "action_std=%s)",
                     self.preprocess_mode, self.action_mean, self.action_std)

    def _tokenize(self, instruction_str: str) -> np.ndarray:
        """CLIP-tokenize with caching. Returns (77,) int64 array."""
        if instruction_str not in self._token_cache:
            self._token_cache[instruction_str] = _tokenize_instruction_string(
                instruction_str, self._tokenizer)
        return self._token_cache[instruction_str]

    def _forward_fn(self, variables, observation):
        """Pure function for JIT: model forward pass.

        Same as BCJaxPyPolicy._run_action_inference (train/policy.py lines 68-82)
        but without the single-sample expand_dims (we already pass batched input).
        """
        normalized_action = self.model.apply(
            variables, observation, train=False)
        action = (
            normalized_action * jnp.maximum(self.action_std, EPS)
            + self.action_mean)
        action = jnp.clip(action, -self.action_clip, self.action_clip)
        return action

    def reset(self, num_envs: int):
        """Clear frame history buffers for all environments.

        Mirrors HistoryWrapper behavior: on reset, the buffer is empty
        and will be filled by tiling the first frame.
        """
        self._frame_buffers = [
            deque(maxlen=self.sequence_length) for _ in range(num_envs)
        ]

    def _update_frame_buffer(self, env_idx: int, rgb_uint8: np.ndarray):
        """Update a single env's frame history.

        Matches HistoryWrapper(tile_first_step_obs=True):
        - First call after reset: tile the frame to fill the buffer
        - Subsequent calls: append (deque auto-evicts oldest)
        """
        frame = _preprocess_rgb(rgb_uint8)
        buf = self._frame_buffers[env_idx]
        if len(buf) == 0:
            for _ in range(self.sequence_length):
                buf.append(frame)
        else:
            buf.append(frame)

    def _build_batch(
        self,
        goals: List[str],
        obs_list: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Build batched model input from frame buffers and goals.

        Produces the same observation dict shape that the HistoryWrapper +
        BCJaxPyPolicy pipeline creates:
          rgb: (B, seq_len, H, W, 3) float32
          instruction_tokenized_clip: (B, seq_len, 77) int32

        The preprocessing strategy is determined by ``self.preprocess_mode``.
        """
        batch_size = len(goals)

        # ---- Phase 1: preprocess images & update frame buffers ----
        if self.preprocess_mode == "original":
            self._phase1_original(batch_size, obs_list, active_mask)
        else:
            self._phase1_batched(batch_size, obs_list, active_mask)

        # ---- Phase 2: assemble batch arrays (shared across all modes) ----
        rgb_batch = np.zeros(
            (batch_size, self.sequence_length, DATA_TARGET_HEIGHT,
             DATA_TARGET_WIDTH, 3),
            dtype=np.float32)

        clip_batch = np.zeros(
            (batch_size, self.sequence_length, 77), dtype=np.int32)

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

    # ------------------------------------------------------------------
    # Phase 1 variants — image preprocessing & frame buffer update
    # ------------------------------------------------------------------

    def _phase1_original(self, batch_size, obs_list, active_mask):
        """Per-image TF preprocessing (original baseline)."""
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                logger.warning("Env %d: no 'rgb' in obs, skipping", i)
                continue
            self._update_frame_buffer(i, rgb)

    def _phase1_batched(self, batch_size, obs_list, active_mask):
        """Collect all active images, batch-preprocess, update buffers."""
        active_indices = []
        raw_images = []
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                logger.warning("Env %d: no 'rgb' in obs, skipping", i)
                continue
            active_indices.append(i)
            raw_images.append(rgb)

        if not raw_images:
            return

        stacked = np.stack(raw_images, axis=0)
        if self.preprocess_mode == "batched_tf":
            processed = _batch_preprocess_rgb_tf(stacked)
        else:  # jax_gpu
            processed = _jax_preprocess_rgb_batch(stacked)

        for j, env_i in enumerate(active_indices):
            frame = processed[j]
            buf = self._frame_buffers[env_i]
            if len(buf) == 0:
                for _ in range(self.sequence_length):
                    buf.append(frame)
            else:
                buf.append(frame)

    def predict(
        self,
        goals: List[str],
        obs_list: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Run batched LAVA inference.

        Parameters
        ----------
        goals : list[str]
            One natural-language goal per environment (from the outer LLM).
        obs_list : list[dict]
            Per-env observation dicts. Must include 'rgb' key with uint8 images.
        active_mask : np.ndarray
            Boolean mask of shape (num_envs,). True = needs an action.

        Returns
        -------
        list[np.ndarray]
            One (2,) action array per environment. Inactive env actions are zeros.
        """
        batch_size = len(goals)
        n_active = int(active_mask.sum())
        if n_active == 0:
            return [np.zeros(2, dtype=np.float32) for _ in range(batch_size)]

        empty_goal_indices = [
            idx for idx, goal in enumerate(goals)
            if active_mask[idx] and not goal.strip()
        ]
        if empty_goal_indices:
            logger.warning(
                "Empty goal strings passed to LAVA for envs=%s",
                empty_goal_indices[:10],
            )

        # Build batched input (also updates frame buffers)
        observation = self._build_batch(goals, obs_list, active_mask)

        rgb_obs = np.asarray(observation["rgb"])
        clip_obs = np.asarray(observation["instruction_tokenized_clip"])
        rgb_finite = bool(np.isfinite(rgb_obs).all())
        if not rgb_finite:
            nan_envs = np.flatnonzero(
                ~np.isfinite(rgb_obs.reshape(len(goals), -1)).all(axis=1)
            ).tolist()
            logger.error("Non-finite RGB inputs for envs=%s", nan_envs[:10])

        # Run JIT-compiled forward pass (denormalization + clip included)
        actions = np.array(self._forward_jit(self.variables, observation))
        invalid_action_mask = ~np.isfinite(actions).all(axis=1)
        if invalid_action_mask.any():
            bad_indices = np.flatnonzero(invalid_action_mask).tolist()

            diag_lines = [
                f"LAVA non-finite actions for envs={bad_indices[:10]}",
                f"  rgb_all_finite={rgb_finite}  rgb_range=[{float(rgb_obs.min()):.3f}, {float(rgb_obs.max()):.3f}]",
            ]
            for idx in bad_indices[:5]:
                tokens = clip_obs[idx, 0].tolist()
                nonzero_tokens = [t for t in tokens if t != 0]
                diag_lines.append(
                    f"  env={idx}  goal={goals[idx][:160]!r}"
                    f"  action={actions[idx].tolist()}"
                    f"  clip_tokens({len(nonzero_tokens)} nonzero)={nonzero_tokens[:15]}"
                )
            diag_msg = "\n".join(diag_lines)
            logger.error(diag_msg)
            raise ValueError(diag_msg)

        # Zero out inactive envs
        actions[~active_mask] = 0.0

        return [actions[i].astype(np.float32) for i in range(batch_size)]
