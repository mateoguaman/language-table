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
import os
import time
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

        # JIT-compile forward passes
        self._forward_jit = jax.jit(self._forward_fn)
        self._forward_raw_jit = jax.jit(self._forward_raw_fn)

        # Directory for NaN crash dumps — sits inside the session log dir
        # so dumps are grouped with the env server logs for this run.
        session_id = os.environ.get("LAVA_SESSION_ID", "no_session")
        default_dump_dir = os.path.join(
            os.path.expanduser("~"), "LaMer", "logs", session_id, "nan_dumps")
        self._nan_dump_dir = os.environ.get(
            "LAVA_NAN_DUMP_DIR", default_dump_dir)
        self._session_id = session_id

        # Rolling history of per-step action statistics for trend diagnosis.
        # Each entry is a dict recorded in predict().  Kept in a fixed-size
        # deque so memory stays bounded even on very long runs.
        self._action_history_maxlen = 200
        self._action_history: deque = deque(
            maxlen=self._action_history_maxlen)
        self._predict_call_count = 0

        # Accumulator for per-outer-step stats.  Reset by
        # get_and_reset_step_stats() after each outer step completes.
        self._step_stats_accum: List[dict] = []

        logger.info("LAVA policy loaded (action_mean=%s, action_std=%s)",
                     self.action_mean, self.action_std)

    def _tokenize(self, instruction_str: str) -> np.ndarray:
        """CLIP-tokenize with caching. Returns (77,) int64 array."""
        if instruction_str not in self._token_cache:
            self._token_cache[instruction_str] = _tokenize_instruction_string(
                instruction_str, self._tokenizer)
        return self._token_cache[instruction_str]

    def _forward_raw_fn(self, variables, observation):
        """Pure function for JIT: model forward pass WITHOUT denormalization.

        Returns the raw normalized action from the model, before any
        denormalization or clipping.  Used for diagnostics only.
        """
        return self.model.apply(variables, observation, train=False)

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
        """
        batch_size = len(goals)

        # Update frame buffers with new observations
        for i in range(batch_size):
            if not active_mask[i]:
                continue
            rgb = obs_list[i].get("rgb")
            if rgb is None:
                logger.warning("Env %d: no 'rgb' in obs, skipping", i)
                continue
            self._update_frame_buffer(i, rgb)

        # Allocate batched arrays
        rgb_batch = np.zeros(
            (batch_size, self.sequence_length, DATA_TARGET_HEIGHT,
             DATA_TARGET_WIDTH, 3),
            dtype=np.float32)

        # HistoryWrapper stacks all obs keys along seq axis, so the model
        # sees instruction_tokenized_clip as (B, seq_len, 77).
        # The encoder takes [:, 0] to get (B, 77) for the CLIP TextEncoder.
        clip_batch = np.zeros(
            (batch_size, self.sequence_length, 77), dtype=np.int32)

        for i in range(batch_size):
            if not active_mask[i]:
                continue

            # Stack frame history into (seq_len, H, W, 3)
            buf = self._frame_buffers[i]
            if len(buf) > 0:
                rgb_batch[i] = np.stack(list(buf), axis=0)

            # Tokenize goal string — same for all timesteps in the sequence,
            # matching ClipTokenWrapper which tokenizes once on reset and
            # reuses for all steps
            tokens = self._tokenize(goals[i])
            for t in range(self.sequence_length):
                clip_batch[i, t] = tokens

        return {
            "rgb": jnp.array(rgb_batch),
            "instruction_tokenized_clip": jnp.array(clip_batch),
        }

    def _dump_nan_diagnostics(
        self,
        bad_indices: List[int],
        actions: np.ndarray,
        observation: Dict[str, jnp.ndarray],
        goals: List[str],
        obs_list: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> str:
        """Dump full diagnostics when NaN/Inf actions are detected.

        Saves a .npz file with everything needed to reproduce the issue and
        logs a detailed human-readable summary.  Returns the path to the dump.
        """
        ts = int(time.time())
        os.makedirs(self._nan_dump_dir, exist_ok=True)
        dump_path = os.path.join(self._nan_dump_dir, f"nan_dump_{ts}.npz")

        # ── 1. Re-run the raw (pre-denorm) forward pass for diagnosis ──
        try:
            raw_actions = np.array(
                self._forward_raw_jit(self.variables, observation))
        except Exception as e:
            logger.error("Failed to re-run raw forward pass: %s", e)
            raw_actions = None

        # ── 2. Collect per-bad-env diagnostics ──
        diag_lines = [
            "=" * 72,
            "LAVA NaN/Inf ACTION DIAGNOSTIC DUMP",
            "=" * 72,
            f"Session         : {self._session_id}",
            f"Timestamp       : {ts}",
            f"Dump file       : {dump_path}",
            f"Bad env indices : {bad_indices}",
            f"Total envs      : {len(goals)}",
            f"Active envs     : {int(active_mask.sum())}",
            "",
            "── Denormalization stats ──",
            f"  action_mean = {self.action_mean}",
            f"  action_std  = {self.action_std}",
            f"  EPS         = {EPS}",
            f"  max(std,EPS)= {np.maximum(self.action_std, EPS)}",
            f"  action_clip = {self.action_clip}",
            f"  mean finite?  {np.isfinite(self.action_mean).all()}",
            f"  std finite?   {np.isfinite(self.action_std).all()}",
            "",
        ]

        # ── 3. Per-env breakdown ──
        for idx in bad_indices[:10]:
            diag_lines.append(f"── Env {idx} ──")
            diag_lines.append(f"  goal          : {goals[idx]!r}")
            diag_lines.append(f"  active        : {active_mask[idx]}")
            diag_lines.append(
                f"  final action  : {actions[idx].tolist()}")

            if raw_actions is not None:
                diag_lines.append(
                    f"  raw (pre-denorm) action: {raw_actions[idx].tolist()}")
                diag_lines.append(
                    f"  raw finite?   : {np.isfinite(raw_actions[idx]).all()}")

            # Observation-level checks
            rgb_obs = observation["rgb"]
            clip_obs = observation["instruction_tokenized_clip"]
            rgb_i = np.array(rgb_obs[idx])
            clip_i = np.array(clip_obs[idx])
            diag_lines.append(
                f"  rgb stats     : shape={rgb_i.shape}  "
                f"min={rgb_i.min():.6f}  max={rgb_i.max():.6f}  "
                f"mean={rgb_i.mean():.6f}  finite={np.isfinite(rgb_i).all()}")
            diag_lines.append(
                f"  clip tokens   : shape={clip_i.shape}  "
                f"min={clip_i.min()}  max={clip_i.max()}")

            # Frame buffer state
            buf = self._frame_buffers[idx] if idx < len(self._frame_buffers) else None
            if buf is not None:
                diag_lines.append(f"  frame_buf len : {len(buf)}")
                for fi, frame in enumerate(buf):
                    diag_lines.append(
                        f"    frame[{fi}]   : shape={frame.shape}  "
                        f"min={frame.min():.6f}  max={frame.max():.6f}  "
                        f"finite={np.isfinite(frame).all()}")

            # Raw source observation
            raw_rgb = obs_list[idx].get("rgb") if idx < len(obs_list) else None
            if raw_rgb is not None:
                diag_lines.append(
                    f"  raw obs rgb   : shape={raw_rgb.shape}  "
                    f"dtype={raw_rgb.dtype}  "
                    f"min={raw_rgb.min()}  max={raw_rgb.max()}")
            else:
                diag_lines.append("  raw obs rgb   : None")

            diag_lines.append("")

        # ── 4. Batch-wide statistics ──
        diag_lines.append("── Batch-wide action stats ──")
        diag_lines.append(
            f"  final actions : shape={actions.shape}  "
            f"min={np.nanmin(actions):.6f}  max={np.nanmax(actions):.6f}  "
            f"nan_count={np.isnan(actions).sum()}  "
            f"inf_count={np.isinf(actions).sum()}")
        if raw_actions is not None:
            diag_lines.append(
                f"  raw actions   : shape={raw_actions.shape}  "
                f"min={np.nanmin(raw_actions):.6f}  "
                f"max={np.nanmax(raw_actions):.6f}  "
                f"nan_count={np.isnan(raw_actions).sum()}  "
                f"inf_count={np.isinf(raw_actions).sum()}")

        rgb_np = np.array(observation["rgb"])
        diag_lines.append(
            f"  rgb batch     : shape={rgb_np.shape}  "
            f"min={rgb_np.min():.6f}  max={rgb_np.max():.6f}  "
            f"nan_count={np.isnan(rgb_np).sum()}")
        diag_lines.append("=" * 72)

        # ── 5. Rolling history — show the trend leading up to the crash ──
        history = list(self._action_history)
        diag_lines.append(f"── Action history (last {len(history)} predict() calls) ──")
        diag_lines.append(
            f"  {'step':>6s}  {'robot_l2_mean':>13s}  {'robot_l2_max':>12s}  "
            f"{'robot_abs_max':>13s}  {'model_l2_max':>12s}  {'model_abs_max':>13s}  "
            f"{'model_nan':>9s}  {'model_inf':>9s}  {'img_nan':>7s}")
        for h in history:
            diag_lines.append(
                f"  {h['step']:6d}  "
                f"{h['vla/robot_action_l2_mean']:13.6f}  "
                f"{h['vla/robot_action_l2_max']:12.6f}  "
                f"{h['vla/robot_action_abs_max']:13.6f}  "
                f"{h['vla/model_output_l2_max']:12.6f}  "
                f"{h['vla/model_output_abs_max']:13.6f}  "
                f"{h['vla/model_output_nan_count']:9d}  "
                f"{h['vla/model_output_inf_count']:9d}  "
                f"{h['vla/image_nan_count']:7d}")
        diag_lines.append("=" * 72)

        # ── 6. Log everything ──
        full_msg = "\n".join(diag_lines)
        logger.error("\n%s", full_msg)

        # ── 7. Save .npz with arrays for offline reproduction ──
        save_dict = {
            "bad_indices": np.array(bad_indices),
            "actions": actions,
            "active_mask": active_mask,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "rgb_obs": np.array(observation["rgb"]),
            "clip_obs": np.array(observation["instruction_tokenized_clip"]),
        }
        if raw_actions is not None:
            save_dict["raw_actions"] = raw_actions

        # Save rolling history as structured arrays for plotting
        if history:
            for key in history[0]:
                save_dict[f"history_{key}"] = np.array(
                    [h[key] for h in history])

        # Save raw source RGB for bad envs (uint8, before preprocessing)
        for idx in bad_indices[:10]:
            raw_rgb = obs_list[idx].get("rgb") if idx < len(obs_list) else None
            if raw_rgb is not None:
                save_dict[f"raw_rgb_env{idx}"] = raw_rgb

        # Save goals as a separate text file alongside the npz
        goals_path = dump_path.replace(".npz", "_goals.txt")
        try:
            with open(goals_path, "w") as f:
                for i, g in enumerate(goals):
                    marker = " *** BAD ***" if i in bad_indices else ""
                    f.write(f"[{i:3d}] active={active_mask[i]} {g!r}{marker}\n")
        except Exception as e:
            logger.error("Failed to write goals file: %s", e)

        try:
            np.savez_compressed(dump_path, **save_dict)
            logger.error("Diagnostic dump saved to: %s", dump_path)
            logger.error("Goals saved to: %s", goals_path)
        except Exception as e:
            logger.error("Failed to save diagnostic dump: %s", e)

        return dump_path

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

        # Run both the full forward pass and the raw (pre-denorm) pass.
        # The raw pass is cheap (same JIT'd model, no extra GPU work beyond
        # skipping denorm) and gives us the trend data we need.
        actions = np.array(self._forward_jit(self.variables, observation))
        raw_actions = np.array(
            self._forward_raw_jit(self.variables, observation))

        # Record per-step statistics into the rolling history.
        #
        # Naming convention:
        #   "robot_action"  = final 2D action sent to PyBullet (after denorm + clip)
        #   "model_output"  = raw network output before denormalization
        #   "l2"            = L2 (Euclidean) vector magnitude
        #   "abs_max"       = largest absolute scalar component
        self._predict_call_count += 1
        active_actions = actions[active_mask]
        active_raw = raw_actions[active_mask]
        rgb_np = np.array(observation["rgb"])
        active_rgb = rgb_np[active_mask]
        step_record = {
            "step": self._predict_call_count,
            # Robot actions (what PyBullet receives)
            "vla/robot_action_l2_mean": float(np.linalg.norm(active_actions, axis=1).mean()),
            "vla/robot_action_l2_max": float(np.linalg.norm(active_actions, axis=1).max()),
            "vla/robot_action_abs_max": float(np.abs(active_actions).max()),
            "vla/robot_action_nan_count": int(np.isnan(active_actions).sum()),
            "vla/robot_action_inf_count": int(np.isinf(active_actions).sum()),
            # Model output (before denormalization — if this blows up, the
            # network itself is numerically unstable)
            "vla/model_output_l2_mean": float(np.linalg.norm(active_raw, axis=1).mean()),
            "vla/model_output_l2_max": float(np.linalg.norm(active_raw, axis=1).max()),
            "vla/model_output_abs_max": float(np.abs(active_raw).max()),
            "vla/model_output_nan_count": int(np.isnan(active_raw).sum()),
            "vla/model_output_inf_count": int(np.isinf(active_raw).sum()),
            # Image observations fed to VLA
            "vla/image_pixel_mean": float(active_rgb.mean()),
            "vla/image_pixel_max": float(active_rgb.max()),
            "vla/image_nan_count": int(np.isnan(active_rgb).sum()),
            # Counts
            "vla/n_active_envs": int(active_mask.sum()),
        }
        self._action_history.append(step_record)
        self._step_stats_accum.append(step_record)

        invalid_action_mask = ~np.isfinite(actions).all(axis=1)
        if invalid_action_mask.any():
            bad_indices = np.flatnonzero(invalid_action_mask).tolist()
            dump_path = self._dump_nan_diagnostics(
                bad_indices, actions, observation, goals, obs_list, active_mask,
            )
            raise ValueError(
                f"LAVA produced non-finite actions for envs={bad_indices[:10]}"
                f" (diagnostics saved to {dump_path})"
            )

        # Zero out inactive envs
        actions[~active_mask] = 0.0

        return [actions[i].astype(np.float32) for i in range(batch_size)]

    def get_and_reset_step_stats(self) -> Dict[str, float]:
        """Aggregate VLA stats across all inner-loop predict() calls since the
        last call to this method, then reset the accumulator.

        Called once per outer step (i.e. per LLM action) by the env manager.
        Returns a flat dict suitable for merging into infos and ultimately
        into wandb metrics.

        Aggregation rules:
        - l2_mean, image_pixel_mean  → averaged across inner steps
        - l2_max, abs_max, pixel_max → max across inner steps
        - nan/inf counts             → summed across inner steps
        - n_active_envs              → from last inner step
        - n_inner_steps              → count of predict() calls
        """
        records = self._step_stats_accum
        self._step_stats_accum = []

        if not records:
            return {}

        result: Dict[str, float] = {}
        result["vla/n_inner_steps"] = float(len(records))

        # Keys and how to aggregate them
        _MEAN_KEYS = [
            "vla/robot_action_l2_mean",
            "vla/model_output_l2_mean",
            "vla/image_pixel_mean",
        ]
        _MAX_KEYS = [
            "vla/robot_action_l2_max",
            "vla/robot_action_abs_max",
            "vla/model_output_l2_max",
            "vla/model_output_abs_max",
            "vla/image_pixel_max",
        ]
        _SUM_KEYS = [
            "vla/robot_action_nan_count",
            "vla/robot_action_inf_count",
            "vla/model_output_nan_count",
            "vla/model_output_inf_count",
            "vla/image_nan_count",
        ]

        for key in _MEAN_KEYS:
            result[key] = float(np.mean([r[key] for r in records]))
        for key in _MAX_KEYS:
            result[key] = float(np.max([r[key] for r in records]))
        for key in _SUM_KEYS:
            result[key] = float(np.sum([r[key] for r in records]))

        result["vla/n_active_envs"] = records[-1]["vla/n_active_envs"]

        return result
