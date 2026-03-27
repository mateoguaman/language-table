"""Base class, constants, and shared utilities for _build_batch variants."""

from typing import Any, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Preprocessing constants — from language_table_resnet_sim_local.py
# ---------------------------------------------------------------------------
DATA_TARGET_WIDTH = 320
DATA_TARGET_HEIGHT = 180
RANDOM_CROP_FACTOR = 0.95
CLIP_TOKEN_LENGTH = 77

# Pre-compute crop offsets for the standard 180×320 input size.  These are
# re-derived dynamically for other sizes, but caching the common case avoids
# redundant TF/numpy scalar math on every call.
_STD_H, _STD_W = 180, 320
_STD_CROP_H = int(_STD_H * RANDOM_CROP_FACTOR)
_STD_CROP_W = int(_STD_W * RANDOM_CROP_FACTOR)
_STD_OFF_H = (_STD_H - _STD_CROP_H) // 2
_STD_OFF_W = (_STD_W - _STD_CROP_W) // 2


def compute_crop_params(h: int, w: int):
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


# ---------------------------------------------------------------------------
# BatchBuilder — abstract interface shared by every variant
# ---------------------------------------------------------------------------

class BatchBuilder:
    """Interface that every _build_batch variant must implement.

    Subclasses must provide ``reset`` and ``build_batch``.
    ``get_frame_state`` is optional and used only by the test suite.
    """

    name: str = "base"

    def __init__(
        self,
        sequence_length: int = 4,
        tokenize_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.sequence_length = sequence_length
        if tokenize_fn is not None:
            self._tokenize = tokenize_fn
        else:
            # Default to the real CLIP tokenizer so benchmarks measure
            # actual tokenization cost.
            self._tokenize = get_real_tokenizer()

    def reset(self, num_envs: int) -> None:
        raise NotImplementedError

    def build_batch(
        self,
        goals: List[str],
        obs_list: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    def get_frame_state(self) -> List[Optional[np.ndarray]]:
        """Return a list of (seq_len, H, W, 3) arrays per env (or None)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def mock_tokenize(text: str) -> np.ndarray:
    """Deterministic mock tokenizer for tests that don't need real CLIP."""
    h = hash(text) & 0x7FFFFFFF
    rng = np.random.RandomState(h)
    return rng.randint(0, 49408, size=(CLIP_TOKEN_LENGTH,)).astype(np.int64)


def get_real_tokenizer() -> Callable[[str], np.ndarray]:
    """Build the real CLIP tokenizer from language_table and return a
    caching tokenize function ``str -> (77,) int64``."""
    from language_table.common import clip_tokenizer

    vocab = clip_tokenizer.create_vocab()
    tok = clip_tokenizer.ClipTokenizer(vocab)
    cache: Dict[str, np.ndarray] = {}

    def _tokenize(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = clip_tokenizer.tokenize_text(text, tok).numpy()[0]
        return cache[text]

    return _tokenize
