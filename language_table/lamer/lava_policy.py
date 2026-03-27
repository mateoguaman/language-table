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
    """Preprocess a single RGB image (OpenCV, for debugging / single-env use).

    Replicates CentralCropImageWrapper from language_table/eval/wrappers.py:
      1. uint8 -> float32 (divide by 255)
      2. Central crop with factor 0.95
      3. Bilinear resize to (180, 320)
    """
    image = rgb_uint8.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    sh, sw = int(h * RANDOM_CROP_FACTOR), int(w * RANDOM_CROP_FACTOR)
    oh, ow = (h - sh) // 2, (w - sw) // 2
    image = image[oh:oh + sh, ow:ow + sw]
    return cv2.resize(
        image, (DATA_TARGET_WIDTH, DATA_TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR)


def _preprocess_rgb_batch(rgb_uint8_batch):
    """Preprocess a batch of RGB images via JAX on GPU.

    CPU: central crop (numpy slice — zero-copy view).
    GPU: uint8 → float32 conversion + bilinear resize (parallel across batch).

    Returns a JAX array that stays on GPU.
    """
    h, w = rgb_uint8_batch.shape[1], rgb_uint8_batch.shape[2]
    sh = int(h * RANDOM_CROP_FACTOR)
    sw = int(w * RANDOM_CROP_FACTOR)
    oh = (h - sh) // 2
    ow = (w - sw) // 2
    cropped = rgb_uint8_batch[:, oh:oh + sh, ow:ow + sw, :]
    images = jnp.array(cropped, dtype=jnp.float32) / 255.0
    n = images.shape[0]
    return jax.image.resize(
        images, (n, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3),
        method='bilinear')


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
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_prefix: str = "bc_resnet_sim_checkpoint_",
        model_config: Optional[dict] = None,
        sequence_length: int = 4,
        action_clip: float = 0.1,
    ):
        self.sequence_length = sequence_length
        self.action_clip = action_clip
        self._num_envs = 0
        self._frame_array: Optional[jnp.ndarray] = None
        self._frame_step: Optional[np.ndarray] = None

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

        logger.info("LAVA policy loaded (action_mean=%s, action_std=%s)",
                     self.action_mean, self.action_std)

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
        """Clear frame history for all environments.

        Allocates a single contiguous frame array on GPU instead of
        per-env deques, and a CPU counter to track first-step tiling.
        """
        self._num_envs = num_envs
        self._frame_array = jnp.zeros(
            (num_envs, self.sequence_length, DATA_TARGET_HEIGHT,
             DATA_TARGET_WIDTH, 3), dtype=jnp.float32)
        self._frame_step = np.zeros(num_envs, dtype=np.int64)

    def _update_frame_buffers_batch(
        self,
        env_indices: List[int],
        rgb_list: List[np.ndarray],
    ):
        """Batch-update frame histories for multiple envs.

        Preprocesses all RGB images on GPU via JAX, then updates the
        pre-allocated frame array with vectorized scatter operations
        (no per-env Python loop).
        """
        if not rgb_list:
            return

        preprocessed = _preprocess_rgb_batch(np.stack(rgb_list, axis=0))

        idx = np.array(env_indices)
        is_first = self._frame_step[idx] == 0

        if is_first.any():
            first_env = idx[is_first]
            tiled = jnp.repeat(
                preprocessed[is_first][:, None],
                self.sequence_length, axis=1)
            self._frame_array = self._frame_array.at[first_env].set(tiled)

        cont_mask = ~is_first
        if cont_mask.any():
            cont_env = idx[cont_mask]
            shifted = jnp.concatenate(
                [self._frame_array[cont_env, 1:],
                 preprocessed[cont_mask][:, None]],
                axis=1)
            self._frame_array = self._frame_array.at[cont_env].set(shifted)

        self._frame_step[idx] += 1

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

        The frame array is already maintained on GPU as a contiguous
        (N, seq_len, H, W, 3) buffer, so no per-env stacking or
        CPU→GPU transfer is needed for RGB data.
        """
        batch_size = len(goals)

        active_indices = []
        active_rgbs = []
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                logger.warning("Env %d: no 'rgb' in obs, skipping", i)
                continue
            active_indices.append(i)
            active_rgbs.append(rgb)

        self._update_frame_buffers_batch(active_indices, active_rgbs)

        clip_batch = np.zeros(
            (batch_size, self.sequence_length, 77), dtype=np.int32)
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            tokens = self._tokenize(goals[i])
            clip_batch[i] = tokens[None]

        return {
            "rgb": self._frame_array,
            "instruction_tokenized_clip": jnp.array(clip_batch),
        }

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

        rgb_finite = bool(jnp.isfinite(observation["rgb"]).all())
        if not rgb_finite:
            rgb_obs_np = np.asarray(observation["rgb"])
            nan_envs = np.flatnonzero(
                ~np.isfinite(rgb_obs_np.reshape(len(goals), -1)).all(axis=1)
            ).tolist()
            logger.error("Non-finite RGB inputs for envs=%s", nan_envs[:10])

        # Run JIT-compiled forward pass (denormalization + clip included)
        actions = np.array(self._forward_jit(self.variables, observation))
        invalid_action_mask = ~np.isfinite(actions).all(axis=1)
        if invalid_action_mask.any():
            bad_indices = np.flatnonzero(invalid_action_mask).tolist()
            rgb_obs_np = np.asarray(observation["rgb"])
            clip_obs_np = np.asarray(observation["instruction_tokenized_clip"])

            diag_lines = [
                f"LAVA non-finite actions for envs={bad_indices[:10]}",
                f"  rgb_all_finite={rgb_finite}  rgb_range=[{float(rgb_obs_np.min()):.3f}, {float(rgb_obs_np.max()):.3f}]",
            ]
            for idx in bad_indices[:5]:
                tokens = clip_obs_np[idx, 0].tolist()
                nonzero_tokens = [t for t in tokens if t != 0]
                diag_lines.append(
                    f"  env={idx}  goal={goals[idx][:160]!r}"
                    f"  action={actions[idx].tolist()}"
                    f"  clip_tokens({len(nonzero_tokens)} nonzero)={nonzero_tokens[:15]}"
                )
            diag_msg = "\n".join(diag_lines)
            logger.warning(diag_msg)
            actions[invalid_action_mask] = 0.0

        # Zero out inactive envs
        actions[~active_mask] = 0.0

        return [actions[i].astype(np.float32) for i in range(batch_size)]
