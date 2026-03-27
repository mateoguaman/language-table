"""Benchmark runner for _build_batch optimization variants.

Measures wall-clock time of ``build_batch`` for each variant at increasing
batch sizes.  Does NOT require a GPU model — only the preprocessing + batch
assembly is timed.

Uses the **real** CLIP tokenizer so tokenization cost is included in the
measurements.

Usage:
    cd /path/to/language-table
    python -m language_table.lamer.optim.bench [--batch_sizes 1,2,4,8,...] \
                                                [--num_warmup 2] \
                                                [--num_iters 10]
"""

import argparse
import gc
import os
import sys
import time
from typing import Any, Callable, Dict, List, Type

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info noise

from language_table.lamer.optim.base import BatchBuilder, get_real_tokenizer
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


def make_obs_list(batch_size: int, h: int = 480, w: int = 640, seed: int = 0):
    rng = np.random.RandomState(seed)
    return [
        {"rgb": rng.randint(0, 256, (h, w, 3), dtype=np.uint8)}
        for _ in range(batch_size)
    ]


def make_goals(batch_size: int):
    return ["push the red block to the blue block"] * batch_size


def benchmark_one(
    builder: BatchBuilder,
    batch_size: int,
    num_warmup: int,
    num_iters: int,
    img_h: int = 480,
    img_w: int = 640,
) -> Dict[str, float]:
    """Time ``build_batch`` and return stats in milliseconds."""
    goals = make_goals(batch_size)
    mask = np.ones(batch_size, dtype=bool)

    # Warmup (pre-fill frame buffers + JIT / TF graph caching)
    for i in range(num_warmup):
        builder.reset(batch_size)
        obs = make_obs_list(batch_size, h=img_h, w=img_w, seed=i)
        _ = builder.build_batch(goals, obs, mask)

    # Timed runs
    times = []
    for i in range(num_iters):
        builder.reset(batch_size)
        obs = make_obs_list(batch_size, h=img_h, w=img_w, seed=1000 + i)

        t0 = time.perf_counter()
        _ = builder.build_batch(goals, obs, mask)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def create_builders() -> List[BatchBuilder]:
    """Instantiate all variant builders with the real CLIP tokenizer."""
    tok = get_real_tokenizer()
    kw = dict(tokenize_fn=tok)
    return [
        BaselineBatchBuilder(**kw),
        BatchTFBatchBuilder(**kw),
        NumpyCv2BatchBuilder(**kw),
        VectorizedAssemblyBatchBuilder(**kw),
        ThreadedCv2BatchBuilder(max_workers=4, **kw),
        ThreadedCv2BatchBuilder(max_workers=8, **kw),
        ThreadedCv2BatchBuilder(max_workers=16, **kw),
        JaxGpuBatchBuilder(**kw),
        CombinedNpCv2RingBuilder(**kw),
        CombinedThreadedRingBuilder(max_workers=8, **kw),
        CombinedJaxRingBuilder(**kw),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark _build_batch optimization variants"
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8,16,32,64,128,256,512,1024",
        help="Comma-separated batch sizes",
    )
    parser.add_argument("--num_warmup", type=int, default=2)
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--img_h", type=int, default=480,
                        help="Input image height (default 480)")
    parser.add_argument("--img_w", type=int, default=640,
                        help="Input image width (default 640)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    builders = create_builders()

    # Assign unique names (handle duplicate class names from ThreadedCv2)
    names = []
    for b in builders:
        n = b.name
        if hasattr(b, "max_workers"):
            n += f"_w{b.max_workers}"
        names.append(n)

    # Header
    col_w = max(len(n) for n in names) + 2
    print()
    print("=" * (col_w + 20 * len(batch_sizes)))
    print("BUILD_BATCH BENCHMARK  (times in ms, lower is better)")
    print(f"  warmup={args.num_warmup}  iters={args.num_iters}"
          f"  img=({args.img_h}×{args.img_w})")
    print("=" * (col_w + 20 * len(batch_sizes)))

    header = f"{'Variant':<{col_w}}"
    for bs in batch_sizes:
        header += f" | {'bs=' + str(bs):>16}"
    print(header)
    print("-" * len(header))

    # Baseline times for speedup calculation
    baseline_times: Dict[int, float] = {}

    for builder, name in zip(builders, names):
        row = f"{name:<{col_w}}"

        for bs in batch_sizes:
            try:
                stats = benchmark_one(
                    builder, bs, args.num_warmup, args.num_iters,
                    img_h=args.img_h, img_w=args.img_w,
                )
                mean = stats["mean_ms"]
                std = stats["std_ms"]

                if name == "v0_baseline":
                    baseline_times[bs] = mean

                # Show time and speedup vs baseline
                if name != "v0_baseline" and bs in baseline_times:
                    speedup = baseline_times[bs] / mean if mean > 0 else float("inf")
                    row += f" | {mean:>8.1f}±{std:<4.0f}{speedup:>3.1f}×"
                else:
                    row += f" | {mean:>8.1f}±{std:<4.0f}    "

            except Exception as e:
                err = str(e)[:12]
                row += f" | {'ERR:' + err:>16}"

            gc.collect()

        print(row)

    print("=" * len(header))

    # Summary: best variant per batch size
    print("\nBEST VARIANT PER BATCH SIZE:")
    for bs in batch_sizes:
        if bs not in baseline_times:
            continue
        best_name = "v0_baseline"
        best_time = baseline_times[bs]
        for builder, name in zip(builders, names):
            if name == "v0_baseline":
                continue
            try:
                stats = benchmark_one(
                    builder, bs, 1, 3,
                    img_h=args.img_h, img_w=args.img_w,
                )
                if stats["mean_ms"] < best_time:
                    best_time = stats["mean_ms"]
                    best_name = name
            except Exception:
                pass
        speedup = baseline_times[bs] / best_time if best_time > 0 else 0
        print(f"  bs={bs:>5}: {best_name:<30}  {best_time:.1f}ms  ({speedup:.1f}× vs baseline)")


if __name__ == "__main__":
    main()
