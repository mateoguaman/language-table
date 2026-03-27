"""Measure exact numerical discrepancy between baseline TF and each variant.

Produces detailed error statistics: max/mean/median error, uint8-equivalent
magnitude, per-pixel exact-match rate, and whether differences are biased.

Usage:
    cd /path/to/language-table
    python -m language_table.lamer.optim.measure_discrepancy [--batch_size 64]
"""

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

from language_table.lamer.optim.base import get_real_tokenizer
from language_table.lamer.optim.v0_baseline import BaselineBatchBuilder
from language_table.lamer.optim.v1_batch_tf import BatchTFBatchBuilder
from language_table.lamer.optim.v2_numpy_cv2 import NumpyCv2BatchBuilder
from language_table.lamer.optim.v4_threaded_cv2 import ThreadedCv2BatchBuilder
from language_table.lamer.optim.v5_jax_gpu import JaxGpuBatchBuilder


def make_obs_list(batch_size, h, w, seed=42):
    rng = np.random.RandomState(seed)
    return [
        {"rgb": rng.randint(0, 256, (h, w, 3), dtype=np.uint8)}
        for _ in range(batch_size)
    ]


def error_report(name, ref, out):
    """Print detailed error analysis between two (B, seq, H, W, 3) arrays."""
    diff = out.astype(np.float64) - ref.astype(np.float64)
    abs_diff = np.abs(diff)

    n_pixels = diff.size
    n_exact = int((abs_diff == 0.0).sum())
    pct_exact = n_exact / n_pixels * 100

    max_err = abs_diff.max()
    mean_err = abs_diff.mean()
    median_err = np.median(abs_diff)
    p99_err = np.percentile(abs_diff, 99)
    p999_err = np.percentile(abs_diff, 99.9)

    # Signed stats (detect systematic bias)
    mean_signed = diff.mean()
    std_signed = diff.std()

    # uint8-equivalent: how many uint8 levels does the max error span?
    max_err_uint8 = max_err * 255
    mean_err_uint8 = mean_err * 255

    # Histogram of error magnitudes
    thresholds = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Total pixels compared:   {n_pixels:>12,}")
    print(f"  Exactly matching:        {n_exact:>12,}  ({pct_exact:.2f}%)")
    print(f"")
    print(f"  Max  absolute error:     {max_err:>15.10f}  ({max_err_uint8:.4f} uint8 levels)")
    print(f"  Mean absolute error:     {mean_err:>15.10f}  ({mean_err_uint8:.4f} uint8 levels)")
    print(f"  Median absolute error:   {median_err:>15.10f}")
    print(f"  P99 absolute error:      {p99_err:>15.10f}")
    print(f"  P99.9 absolute error:    {p999_err:>15.10f}")
    print(f"")
    print(f"  Mean signed error:       {mean_signed:>+15.10f}  (bias)")
    print(f"  Std of signed error:     {std_signed:>15.10f}")
    print(f"")
    print(f"  Error distribution:")
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        count = int(((abs_diff > lo) & (abs_diff <= hi)).sum())
        pct = count / n_pixels * 100
        print(f"    ({lo:.0e}, {hi:.0e}]:  {count:>12,}  ({pct:>6.2f}%)")
    count_above = int((abs_diff > thresholds[-1]).sum())
    pct_above = count_above / n_pixels * 100
    print(f"    > {thresholds[-1]:.0e}:          {count_above:>12,}  ({pct_above:>6.2f}%)")

    return max_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_h", type=int, default=180,
                        help="Image height (default 180 = real env)")
    parser.add_argument("--img_w", type=int, default=320,
                        help="Image width (default 320 = real env)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3,
                        help="Number of sequential build_batch calls to test")
    args = parser.parse_args()

    tok = get_real_tokenizer()
    goals = ["push the red block to the blue block"] * args.batch_size
    mask = np.ones(args.batch_size, dtype=bool)

    variants = [
        ("v1_batch_tf (should be bit-exact)", BatchTFBatchBuilder(tokenize_fn=tok)),
        ("v2_numpy_cv2", NumpyCv2BatchBuilder(tokenize_fn=tok)),
        ("v4_threaded_cv2", ThreadedCv2BatchBuilder(max_workers=8, tokenize_fn=tok)),
        ("v5_jax_gpu", JaxGpuBatchBuilder(tokenize_fn=tok)),
    ]

    print(f"Discrepancy analysis: bs={args.batch_size}, "
          f"img=({args.img_h}×{args.img_w}), steps={args.steps}, seed={args.seed}")

    for step in range(args.steps):
        print(f"\n{'#'*70}")
        print(f"  STEP {step + 1}/{args.steps}")
        print(f"{'#'*70}")

        obs = make_obs_list(args.batch_size, args.img_h, args.img_w,
                            seed=args.seed + step * 100)

        # Run baseline
        baseline = BaselineBatchBuilder(tokenize_fn=tok)
        baseline.reset(args.batch_size)
        if step > 0:
            # Pre-fill buffer so step>0 tests the append (non-tiling) path
            for s in range(step):
                prev_obs = make_obs_list(args.batch_size, args.img_h, args.img_w,
                                         seed=args.seed + s * 100)
                baseline.build_batch(goals, prev_obs, mask)

        ref_result = baseline.build_batch(goals, obs, mask)
        ref_rgb = np.asarray(ref_result["rgb"])
        ref_clip = np.asarray(ref_result["instruction_tokenized_clip"])

        for vname, vbuilder in variants:
            vbuilder.reset(args.batch_size)
            if step > 0:
                for s in range(step):
                    prev_obs = make_obs_list(args.batch_size, args.img_h, args.img_w,
                                             seed=args.seed + s * 100)
                    vbuilder.build_batch(goals, prev_obs, mask)

            out_result = vbuilder.build_batch(goals, obs, mask)
            out_rgb = np.asarray(out_result["rgb"])
            out_clip = np.asarray(out_result["instruction_tokenized_clip"])

            error_report(vname, ref_rgb, out_rgb)

            # Token check
            clip_match = np.array_equal(out_clip, ref_clip)
            print(f"  CLIP tokens exact match: {'YES' if clip_match else 'NO'}")
            if not clip_match:
                n_diff = int((out_clip != ref_clip).sum())
                print(f"    Token mismatches: {n_diff}")


if __name__ == "__main__":
    main()
