"""Comprehensive equivalence tests for _build_batch optimization variants.

Every variant is tested against v0_baseline to ensure outputs are either
bit-exact or within a tight tolerance (for variants that replace the resize
implementation).

Uses the **real** CLIP tokenizer from language_table so that tokenization
cost and correctness are both exercised.

Run:
    cd /path/to/language-table
    python -m pytest language_table/lamer/optim/test_equivalence.py -v
"""

from typing import Any, Callable, Dict, List

import jax.numpy as jnp
import numpy as np
import pytest

from language_table.lamer.optim.base import (
    BatchBuilder,
    DATA_TARGET_HEIGHT,
    DATA_TARGET_WIDTH,
    get_real_tokenizer,
)
from language_table.lamer.optim.v0_baseline import BaselineBatchBuilder
from language_table.lamer.optim.v1_batch_tf import BatchTFBatchBuilder
from language_table.lamer.optim.v2_numpy_cv2 import NumpyCv2BatchBuilder
from language_table.lamer.optim.v3_vectorized_assembly import (
    VectorizedAssemblyBatchBuilder,
)
from language_table.lamer.optim.v4_threaded_cv2 import ThreadedCv2BatchBuilder
from language_table.lamer.optim.v5_jax_gpu import JaxGpuBatchBuilder
from language_table.lamer.optim.v6_combined import (
    CombinedJaxRingBuilder,
    CombinedNpCv2RingBuilder,
    CombinedThreadedRingBuilder,
)


# ===== Helpers =====

def make_obs_list(batch_size: int, h: int = 480, w: int = 640, seed: int = 0):
    """Create reproducible dummy observations."""
    rng = np.random.RandomState(seed)
    return [
        {"rgb": rng.randint(0, 256, (h, w, 3), dtype=np.uint8)}
        for _ in range(batch_size)
    ]


def make_goals(batch_size: int, unique: bool = False):
    if unique:
        return [f"push block {i} to target {i}" for i in range(batch_size)]
    return ["push the red block to the blue block"] * batch_size


# Variants that use the *same TF ops* as baseline → must be bit-exact.
EXACT_VARIANTS = [
    ("v1_batch_tf", BatchTFBatchBuilder),
    ("v3_vec_assembly", VectorizedAssemblyBatchBuilder),
]

# Variants that change the resize kernel → approximate match only.
APPROX_VARIANTS = [
    ("v2_numpy_cv2", NumpyCv2BatchBuilder),
    ("v4_threaded_cv2", lambda **kw: ThreadedCv2BatchBuilder(max_workers=4, **kw)),
    ("v5_jax_gpu", JaxGpuBatchBuilder),
    ("v6a_np_cv2_ring", CombinedNpCv2RingBuilder),
    ("v6b_threaded_ring", lambda **kw: CombinedThreadedRingBuilder(max_workers=4, **kw)),
    ("v6c_jax_ring", CombinedJaxRingBuilder),
]

ALL_VARIANTS = EXACT_VARIANTS + APPROX_VARIANTS


def _make_builder(cls_or_factory, **kwargs) -> BatchBuilder:
    if isinstance(cls_or_factory, type):
        return cls_or_factory(**kwargs)
    return cls_or_factory(**kwargs)


# ===== Fixtures =====

@pytest.fixture(scope="session")
def real_tokenize_fn():
    """Build the real CLIP tokenizer once per test session."""
    return get_real_tokenizer()


@pytest.fixture(params=[v[0] for v in ALL_VARIANTS], ids=[v[0] for v in ALL_VARIANTS])
def variant_info(request, real_tokenize_fn):
    """Yield (name, builder_factory, is_exact, tokenize_fn) for each variant."""
    name = request.param
    tok = real_tokenize_fn
    for n, factory in EXACT_VARIANTS:
        if n == name:
            return (n, factory, True, tok)
    for n, factory in APPROX_VARIANTS:
        if n == name:
            return (n, factory, False, tok)
    raise ValueError(f"Unknown variant: {name}")


# ===== Test: output shapes and dtypes =====

class TestShapesAndDtypes:
    """Verify all variants produce the right shapes and dtypes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_output_shapes(self, variant_info, batch_size):
        name, factory, _, tok = variant_info
        builder = _make_builder(factory, tokenize_fn=tok)
        builder.reset(batch_size)

        goals = make_goals(batch_size)
        obs = make_obs_list(batch_size, seed=42)
        mask = np.ones(batch_size, dtype=bool)

        result = builder.build_batch(goals, obs, mask)

        rgb = result["rgb"]
        clip = result["instruction_tokenized_clip"]

        assert rgb.shape == (batch_size, 4, DATA_TARGET_HEIGHT, DATA_TARGET_WIDTH, 3), \
            f"{name}: rgb shape {rgb.shape}"
        assert clip.shape == (batch_size, 4, 77), \
            f"{name}: clip shape {clip.shape}"

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_dtypes(self, variant_info, batch_size):
        name, factory, _, tok = variant_info
        builder = _make_builder(factory, tokenize_fn=tok)
        builder.reset(batch_size)

        result = builder.build_batch(
            make_goals(batch_size),
            make_obs_list(batch_size, seed=7),
            np.ones(batch_size, dtype=bool),
        )

        assert result["rgb"].dtype == jnp.float32, f"{name}: rgb dtype"
        assert result["instruction_tokenized_clip"].dtype == jnp.int32, f"{name}: clip dtype"


# ===== Test: bit-exact equivalence for TF-based variants =====

class TestExactEquivalence:
    """Variants that use identical TF ops must match the baseline bit-for-bit."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
    @pytest.mark.parametrize("variant_name,variant_cls", EXACT_VARIANTS)
    def test_single_step_exact(self, batch_size, variant_name, variant_cls,
                               real_tokenize_fn):
        tok = real_tokenize_fn
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(variant_cls, tokenize_fn=tok)

        baseline.reset(batch_size)
        variant.reset(batch_size)

        goals = make_goals(batch_size)
        obs = make_obs_list(batch_size, seed=123)
        mask = np.ones(batch_size, dtype=bool)

        ref = baseline.build_batch(goals, obs, mask)
        out = variant.build_batch(goals, obs, mask)

        np.testing.assert_array_equal(
            np.asarray(out["rgb"]),
            np.asarray(ref["rgb"]),
            err_msg=f"{variant_name} rgb mismatch at bs={batch_size}",
        )
        np.testing.assert_array_equal(
            np.asarray(out["instruction_tokenized_clip"]),
            np.asarray(ref["instruction_tokenized_clip"]),
            err_msg=f"{variant_name} clip mismatch at bs={batch_size}",
        )

    @pytest.mark.parametrize("variant_name,variant_cls", EXACT_VARIANTS)
    def test_multi_step_exact(self, variant_name, variant_cls, real_tokenize_fn):
        """Run 6 steps with different images; verify every step matches."""
        tok = real_tokenize_fn
        bs = 4
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(variant_cls, tokenize_fn=tok)

        baseline.reset(bs)
        variant.reset(bs)

        goals = make_goals(bs)
        for step in range(6):
            obs = make_obs_list(bs, seed=step * 100)
            mask = np.ones(bs, dtype=bool)

            ref = baseline.build_batch(goals, obs, mask)
            out = variant.build_batch(goals, obs, mask)

            np.testing.assert_array_equal(
                np.asarray(out["rgb"]),
                np.asarray(ref["rgb"]),
                err_msg=f"{variant_name} step={step} rgb",
            )
            np.testing.assert_array_equal(
                np.asarray(out["instruction_tokenized_clip"]),
                np.asarray(ref["instruction_tokenized_clip"]),
                err_msg=f"{variant_name} step={step} clip",
            )


# ===== Test: approximate equivalence for non-TF variants =====

class TestApproxEquivalence:
    """Variants with different resize kernels must be close to baseline."""

    RGB_ATOL = 2e-3  # ≈ 0.5/255, half a uint8 level
    RGB_RTOL = 1e-3

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    @pytest.mark.parametrize("variant_name,variant_factory", APPROX_VARIANTS)
    def test_single_step_approx(self, batch_size, variant_name, variant_factory,
                                real_tokenize_fn):
        tok = real_tokenize_fn
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(variant_factory, tokenize_fn=tok)

        baseline.reset(batch_size)
        variant.reset(batch_size)

        goals = make_goals(batch_size)
        obs = make_obs_list(batch_size, seed=456)
        mask = np.ones(batch_size, dtype=bool)

        ref = baseline.build_batch(goals, obs, mask)
        out = variant.build_batch(goals, obs, mask)

        np.testing.assert_allclose(
            np.asarray(out["rgb"]),
            np.asarray(ref["rgb"]),
            atol=self.RGB_ATOL,
            rtol=self.RGB_RTOL,
            err_msg=f"{variant_name} rgb approx mismatch at bs={batch_size}",
        )
        np.testing.assert_array_equal(
            np.asarray(out["instruction_tokenized_clip"]),
            np.asarray(ref["instruction_tokenized_clip"]),
            err_msg=f"{variant_name} clip tokens must be exact",
        )

    @pytest.mark.parametrize("variant_name,variant_factory", APPROX_VARIANTS)
    def test_multi_step_approx(self, variant_name, variant_factory,
                               real_tokenize_fn):
        tok = real_tokenize_fn
        bs = 4
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(variant_factory, tokenize_fn=tok)

        baseline.reset(bs)
        variant.reset(bs)

        goals = make_goals(bs)
        for step in range(6):
            obs = make_obs_list(bs, seed=step * 77)
            mask = np.ones(bs, dtype=bool)

            ref = baseline.build_batch(goals, obs, mask)
            out = variant.build_batch(goals, obs, mask)

            np.testing.assert_allclose(
                np.asarray(out["rgb"]),
                np.asarray(ref["rgb"]),
                atol=self.RGB_ATOL,
                rtol=self.RGB_RTOL,
                err_msg=f"{variant_name} step={step} rgb approx",
            )


# ===== Test: active_mask patterns =====

class TestActiveMask:
    """Verify correct handling of various active-mask patterns."""

    @pytest.mark.parametrize("pattern", [
        "all_active",
        "all_inactive",
        "alternating",
        "first_half",
        "single_active",
    ])
    def test_mask_patterns(self, variant_info, pattern):
        name, factory, is_exact, tok = variant_info
        bs = 8

        if pattern == "all_active":
            mask = np.ones(bs, dtype=bool)
        elif pattern == "all_inactive":
            mask = np.zeros(bs, dtype=bool)
        elif pattern == "alternating":
            mask = np.array([i % 2 == 0 for i in range(bs)])
        elif pattern == "first_half":
            mask = np.array([i < bs // 2 for i in range(bs)])
        elif pattern == "single_active":
            mask = np.zeros(bs, dtype=bool)
            mask[3] = True

        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(factory, tokenize_fn=tok)

        baseline.reset(bs)
        variant.reset(bs)

        goals = make_goals(bs)
        obs = make_obs_list(bs, seed=999)

        ref = baseline.build_batch(goals, obs, mask)
        out = variant.build_batch(goals, obs, mask)

        ref_rgb = np.asarray(ref["rgb"])
        out_rgb = np.asarray(out["rgb"])

        if is_exact:
            np.testing.assert_array_equal(out_rgb, ref_rgb,
                                          err_msg=f"{name} mask={pattern}")
        else:
            np.testing.assert_allclose(out_rgb, ref_rgb,
                                       atol=2e-3, rtol=1e-3,
                                       err_msg=f"{name} mask={pattern}")

        np.testing.assert_array_equal(
            np.asarray(out["instruction_tokenized_clip"]),
            np.asarray(ref["instruction_tokenized_clip"]),
            err_msg=f"{name} clip mask={pattern}",
        )


# ===== Test: inactive envs produce zeros =====

class TestInactiveEnvsZero:
    """Inactive environments must have all-zero rgb and clip entries."""

    def test_inactive_zeros(self, variant_info):
        name, factory, _, tok = variant_info
        bs = 8
        mask = np.array([True, False, True, False, True, False, True, False])

        builder = _make_builder(factory, tokenize_fn=tok)
        builder.reset(bs)

        result = builder.build_batch(
            make_goals(bs), make_obs_list(bs, seed=11), mask,
        )

        rgb = np.asarray(result["rgb"])
        clip = np.asarray(result["instruction_tokenized_clip"])

        for i in range(bs):
            if not mask[i]:
                assert np.all(rgb[i] == 0), f"{name} env {i} rgb not zero"
                assert np.all(clip[i] == 0), f"{name} env {i} clip not zero"


# ===== Test: first-frame tiling =====

class TestFirstFrameTiling:
    """On the first call after reset, the frame buffer should be filled
    with ``sequence_length`` copies of the first frame."""

    @pytest.mark.parametrize("variant_name,variant_cls", EXACT_VARIANTS)
    def test_tiling_exact(self, variant_name, variant_cls, real_tokenize_fn):
        builder = _make_builder(variant_cls, tokenize_fn=real_tokenize_fn)
        builder.reset(1)

        obs = make_obs_list(1, seed=0)
        result = builder.build_batch(["go"], obs, np.array([True]))
        rgb = np.asarray(result["rgb"])[0]  # (seq_len, H, W, 3)

        for t in range(1, 4):
            np.testing.assert_array_equal(
                rgb[t], rgb[0],
                err_msg=f"{variant_name}: frame {t} != frame 0 after tiling",
            )


# ===== Test: frame buffer rollover =====

class TestFrameBufferRollover:
    """After enough steps, the oldest frames should have been evicted."""

    @pytest.mark.parametrize("variant_name,variant_cls", EXACT_VARIANTS)
    def test_rollover(self, variant_name, variant_cls, real_tokenize_fn):
        tok = real_tokenize_fn
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(variant_cls, tokenize_fn=tok)

        baseline.reset(1)
        variant.reset(1)

        for step in range(8):
            obs = make_obs_list(1, seed=step * 50)
            mask = np.array([True])
            ref = baseline.build_batch(["go"], obs, mask)
            out = variant.build_batch(["go"], obs, mask)

            np.testing.assert_array_equal(
                np.asarray(out["rgb"]),
                np.asarray(ref["rgb"]),
                err_msg=f"{variant_name} rollover step={step}",
            )


# ===== Test: unique vs repeated goals =====

class TestGoalVariety:
    """Unique goals should produce unique token rows."""

    def test_unique_goals(self, variant_info):
        name, factory, _, tok = variant_info
        bs = 4
        builder = _make_builder(factory, tokenize_fn=tok)
        builder.reset(bs)

        goals = make_goals(bs, unique=True)
        obs = make_obs_list(bs, seed=1)
        mask = np.ones(bs, dtype=bool)

        result = builder.build_batch(goals, obs, mask)
        clip = np.asarray(result["instruction_tokenized_clip"])

        for i in range(bs):
            for j in range(i + 1, bs):
                assert not np.array_equal(clip[i, 0], clip[j, 0]), \
                    f"{name}: envs {i} and {j} have identical tokens for different goals"


# ===== Test: various input image sizes =====

class TestImageSizes:
    """Verify that different input resolutions are handled correctly."""

    @pytest.mark.parametrize("h,w", [(180, 320), (240, 320), (480, 640), (360, 480)])
    def test_various_sizes(self, variant_info, h, w):
        name, factory, is_exact, tok = variant_info
        bs = 2

        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(factory, tokenize_fn=tok)

        baseline.reset(bs)
        variant.reset(bs)

        obs = make_obs_list(bs, h=h, w=w, seed=42)
        goals = make_goals(bs)
        mask = np.ones(bs, dtype=bool)

        ref = baseline.build_batch(goals, obs, mask)
        out = variant.build_batch(goals, obs, mask)

        if is_exact:
            np.testing.assert_array_equal(
                np.asarray(out["rgb"]), np.asarray(ref["rgb"]),
                err_msg=f"{name} size=({h},{w})",
            )
        else:
            np.testing.assert_allclose(
                np.asarray(out["rgb"]), np.asarray(ref["rgb"]),
                atol=2e-3, rtol=1e-3,
                err_msg=f"{name} size=({h},{w})",
            )


# ===== Test: frame state consistency =====

class TestFrameStateConsistency:
    """After N steps, get_frame_state() for variant should match baseline."""

    @pytest.mark.parametrize("variant_name,variant_cls", EXACT_VARIANTS)
    def test_frame_state_matches(self, variant_name, variant_cls,
                                 real_tokenize_fn):
        tok = real_tokenize_fn
        bs = 4
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(variant_cls, tokenize_fn=tok)

        baseline.reset(bs)
        variant.reset(bs)

        for step in range(6):
            obs = make_obs_list(bs, seed=step)
            mask = np.ones(bs, dtype=bool)
            baseline.build_batch(make_goals(bs), obs, mask)
            variant.build_batch(make_goals(bs), obs, mask)

        ref_state = baseline.get_frame_state()
        out_state = variant.get_frame_state()

        for i in range(bs):
            if ref_state[i] is None:
                assert out_state[i] is None
            else:
                np.testing.assert_array_equal(
                    out_state[i], ref_state[i],
                    err_msg=f"{variant_name} frame_state env={i}",
                )


# ===== Test: determinism =====

class TestDeterminism:
    """Same inputs must produce identical outputs across two separate runs."""

    def test_deterministic(self, variant_info):
        name, factory, _, tok = variant_info
        bs = 4

        results = []
        for _ in range(2):
            builder = _make_builder(factory, tokenize_fn=tok)
            builder.reset(bs)
            out = builder.build_batch(
                make_goals(bs),
                make_obs_list(bs, seed=42),
                np.ones(bs, dtype=bool),
            )
            results.append({k: np.asarray(v) for k, v in out.items()})

        np.testing.assert_array_equal(results[0]["rgb"], results[1]["rgb"],
                                      err_msg=f"{name} not deterministic (rgb)")
        np.testing.assert_array_equal(
            results[0]["instruction_tokenized_clip"],
            results[1]["instruction_tokenized_clip"],
            err_msg=f"{name} not deterministic (clip)",
        )


# ===== Test: quantitative error report =====

class TestErrorReport:
    """Not a pass/fail test — prints the max/mean error for each approx variant."""

    @pytest.mark.parametrize("variant_name,variant_factory", APPROX_VARIANTS)
    def test_error_magnitude(self, variant_name, variant_factory, capsys,
                             real_tokenize_fn):
        tok = real_tokenize_fn
        bs = 8
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        variant = _make_builder(variant_factory, tokenize_fn=tok)

        baseline.reset(bs)
        variant.reset(bs)

        obs = make_obs_list(bs, seed=42)
        goals = make_goals(bs)
        mask = np.ones(bs, dtype=bool)

        ref = np.asarray(baseline.build_batch(goals, obs, mask)["rgb"])
        out = np.asarray(variant.build_batch(goals, obs, mask)["rgb"])

        diff = np.abs(out - ref)
        max_err = diff.max()
        mean_err = diff.mean()
        pct_exact = (diff == 0).mean() * 100

        print(f"\n  {variant_name}: max_err={max_err:.6f}  mean_err={mean_err:.6f}"
              f"  exact_pixels={pct_exact:.1f}%")

        # Sanity: error should be small (< 1% of value range)
        assert max_err < 0.01, f"{variant_name}: max error {max_err} too large"
