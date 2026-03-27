"""
Benchmark the low-level VLA (LAVA) policy on a single GPU.

Measures:
  1. Max batch size before GPU OOM
  2. Inference latency vs batch size (with JIT warmup)
  3. GPU memory usage vs batch size
  4. Preprocessing (TF image ops + CLIP tokenization) vs JAX forward time

When ``--modes`` includes multiple preprocess modes (original, batched_tf,
jax_gpu), the benchmark runs each mode in turn and prints comparison tables
plus an accuracy check (action-level equivalence on identical inputs).

Runs in the language-table conda env (ltvenv).

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    ltvenv/bin/python -m language_table.lamer.benchmark_vla_standalone \
        --checkpoint_dir /path/to/checkpoints/ \
        --batch_sizes 1,2,4,8,16,32,64,128,256,512 \
        --num_warmup 3 --num_iters 20 \
        --modes original,batched_tf,jax_gpu
"""

import argparse
import gc
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_gpu_memory_mb():
    """Get current GPU memory usage via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits",
             f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(",")
            return int(used.strip()), int(total.strip())
    except Exception:
        pass
    return -1, -1


def get_jax_memory_mb():
    """Get JAX device memory stats."""
    try:
        devices = jax.devices()
        if devices:
            stats = devices[0].memory_stats()
            if stats:
                used = stats.get("bytes_in_use", 0) / 1024**2
                limit = stats.get("bytes_limit", 0) / 1024**2
                peak = stats.get("peak_bytes_in_use", 0) / 1024**2
                return used, limit, peak
    except Exception:
        pass
    return -1, -1, -1


def make_dummy_obs(batch_size, height=180, width=320):
    """Create dummy observations matching Language Table output."""
    return [
        {"rgb": np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)}
        for _ in range(batch_size)
    ]


def benchmark_vla_inference(policy, batch_size, num_warmup, num_iters):
    """Benchmark VLA inference at a given batch size.

    Returns dict with timing and memory stats, or None on OOM.
    """
    goals = ["push the red block to the blue block"] * batch_size
    active_mask = np.ones(batch_size, dtype=bool)

    policy.reset(num_envs=batch_size)

    # Warmup (includes JIT compilation on first call)
    for _ in range(num_warmup):
        try:
            obs_list_iter = make_dummy_obs(batch_size)
            _ = policy.predict(goals, obs_list_iter, active_mask)
        except Exception as e:
            if any(k in str(e).lower() for k in ("out of memory", "resource", "alloc")):
                return None
            raise

    # Timed iterations
    preprocess_times = []
    forward_times = []
    total_times = []

    for _ in range(num_iters):
        obs_list_iter = make_dummy_obs(batch_size)

        t_total_start = time.perf_counter()

        # Preprocess: build batch (TF image ops + CLIP tokenization)
        t_pre_start = time.perf_counter()
        observation = policy._build_batch(goals, obs_list_iter, active_mask)
        t_pre_end = time.perf_counter()

        # Forward: JAX JIT inference
        t_fwd_start = time.perf_counter()
        actions = policy._forward_jit(policy.variables, observation)
        actions.block_until_ready()
        t_fwd_end = time.perf_counter()

        t_total_end = time.perf_counter()

        preprocess_times.append(t_pre_end - t_pre_start)
        forward_times.append(t_fwd_end - t_fwd_start)
        total_times.append(t_total_end - t_total_start)

    jax_used, jax_limit, jax_peak = get_jax_memory_mb()
    gpu_used, gpu_total = get_gpu_memory_mb()

    return {
        "batch_size": batch_size,
        "preprocess_ms_mean": np.mean(preprocess_times) * 1000,
        "preprocess_ms_std": np.std(preprocess_times) * 1000,
        "forward_ms_mean": np.mean(forward_times) * 1000,
        "forward_ms_std": np.std(forward_times) * 1000,
        "total_ms_mean": np.mean(total_times) * 1000,
        "total_ms_std": np.std(total_times) * 1000,
        "throughput_samples_per_sec": batch_size / np.mean(total_times),
        "gpu_used_mb": gpu_used,
        "gpu_total_mb": gpu_total,
        "jax_used_mb": jax_used,
        "jax_peak_mb": jax_peak,
    }


def print_mode_comparison(all_results, modes, batch_sizes):
    """Print side-by-side comparison tables across preprocess modes."""
    baseline = modes[0]
    common_bs = sorted(
        set.intersection(*(set(r["batch_size"] for r in all_results[m])
                           for m in modes))
    )
    if not common_bs:
        print("No common batch sizes across modes — skipping comparison.")
        return

    def _lookup(mode, bs, key):
        for r in all_results[mode]:
            if r["batch_size"] == bs:
                return r[key]
        return None

    for metric, key, unit in [
        ("_build_batch (preprocess)", "preprocess_ms_mean", "ms"),
        ("predict (total)", "total_ms_mean", "ms"),
        ("throughput", "throughput_samples_per_sec", "samp/s"),
    ]:
        print(f"\n{'='*80}")
        print(f"COMPARISON: {metric} ({unit})")
        print(f"{'='*80}")
        cols = [f"{'BS':>6}"]
        for m in modes:
            cols.append(f"{m:>14}")
        for m in modes[1:]:
            cols.append(f"{m + ' spdup':>14}")
        header = " | ".join(cols)
        print(header)
        print("-" * len(header))
        for bs in common_bs:
            parts = [f"{bs:>6}"]
            base_val = _lookup(baseline, bs, key)
            for m in modes:
                v = _lookup(m, bs, key)
                parts.append(f"{v:>14.1f}" if v else f"{'---':>14}")
            for m in modes[1:]:
                v = _lookup(m, bs, key)
                if base_val and v and v > 0:
                    if "throughput" in key:
                        sp = v / base_val
                    else:
                        sp = base_val / v
                    parts.append(f"{sp:>13.2f}x")
                else:
                    parts.append(f"{'---':>14}")
            print(" | ".join(parts))

    # GPU memory
    print(f"\n{'='*80}")
    print("COMPARISON: GPU memory (MB)")
    print(f"{'='*80}")
    cols = [f"{'BS':>6}"]
    for m in modes:
        cols.append(f"{m:>14}")
    header = " | ".join(cols)
    print(header)
    print("-" * len(header))
    for bs in common_bs:
        parts = [f"{bs:>6}"]
        for m in modes:
            v = _lookup(m, bs, "gpu_used_mb")
            parts.append(f"{v:>14}" if v and v > 0 else f"{'---':>14}")
        print(" | ".join(parts))


def benchmark_accuracy(policy, modes, batch_sizes, accuracy_steps):
    """Compare actions across modes on identical inputs."""
    print(f"\n{'='*80}")
    print(f"ACCURACY: action equivalence ({accuracy_steps} predict steps)")
    print(f"{'='*80}")

    header = (f"{'BS':>6} | {'comparison':>30} | {'max|Δa|':>10} | "
              f"{'mean|Δa|':>10} | {'exact%':>7} | {'allclose':>9}")
    print(header)
    print("-" * len(header))

    baseline_mode = modes[0]
    for bs in batch_sizes:
        rng = np.random.RandomState(42)
        goals = [f"push the {rng.choice(['red','green','blue'])} block "
                 f"to the {rng.choice(['yellow','purple','orange'])} block"
                 for _ in range(bs)]
        all_obs = [
            [{"rgb": rng.randint(0, 256, (180, 320, 3), dtype=np.uint8)}
             for _ in range(bs)]
            for _ in range(accuracy_steps)
        ]
        active_mask = np.ones(bs, dtype=bool)

        actions_by_mode = {}
        for mode in modes:
            policy.preprocess_mode = mode
            policy.reset(num_envs=bs)
            step_actions = []
            for step in range(accuracy_steps):
                result = policy.predict(goals, all_obs[step], active_mask)
                step_actions.append(
                    np.stack([np.asarray(a) for a in result]))
            actions_by_mode[mode] = np.concatenate(step_actions, axis=0)

        baseline_actions = actions_by_mode[baseline_mode]
        for mode in modes[1:]:
            diff = np.abs(baseline_actions - actions_by_mode[mode])
            max_err = float(diff.max())
            mean_err = float(diff.mean())
            exact_pct = float((diff == 0).mean() * 100)
            close = bool(np.allclose(
                baseline_actions, actions_by_mode[mode],
                atol=1e-6, rtol=1e-5))
            label = f"{baseline_mode} vs {mode}"
            print(f"{bs:>6} | {label:>30} | {max_err:>10.2e} | "
                  f"{mean_err:>10.2e} | {exact_pct:>6.1f}% | "
                  f"{'yes' if close else 'NO':>9}")


def run_single_mode(policy, mode, batch_sizes, num_warmup, num_iters):
    """Benchmark all batch sizes for one preprocess mode. Returns results list."""
    policy.preprocess_mode = mode
    results = []
    max_batch_size = 0

    print(f"\n{'='*105}")
    print(f"VLA (LAVA) INFERENCE BENCHMARK — preprocess_mode={mode}")
    print(f"{'='*105}")
    print(f"{'Batch':>6} | {'Preprocess':>14} | {'Forward':>14} | {'Total':>14} | "
          f"{'Throughput':>12} | {'GPU Used':>10} | {'JAX Peak':>10} | Status")
    print(f"{'Size':>6} | {'(ms)':>14} | {'(ms)':>14} | {'(ms)':>14} | "
          f"{'(samp/s)':>12} | {'(MB)':>10} | {'(MB)':>10} |")
    print("-" * 105)

    for bs in batch_sizes:
        try:
            result = benchmark_vla_inference(
                policy, bs, num_warmup, num_iters)
        except Exception as e:
            error_str = str(e).lower()
            if any(k in error_str
                   for k in ("out of memory", "resource", "alloc")):
                result = None
            else:
                logger.error("Unexpected error at batch_size=%d: %s", bs, e)
                result = None

        if result is None:
            print(f"{bs:>6} | {'---':>14} | {'---':>14} | {'---':>14} | "
                  f"{'---':>12} | {'---':>10} | {'---':>10} | OOM")
            break
        else:
            max_batch_size = bs
            results.append(result)
            print(
                f"{bs:>6} "
                f"| {result['preprocess_ms_mean']:>9.1f} ± "
                f"{result['preprocess_ms_std']:<4.1f}"
                f"| {result['forward_ms_mean']:>9.1f} ± "
                f"{result['forward_ms_std']:<4.1f}"
                f"| {result['total_ms_mean']:>9.1f} ± "
                f"{result['total_ms_std']:<4.1f}"
                f"| {result['throughput_samples_per_sec']:>12.1f}"
                f" | {result['gpu_used_mb']:>10}"
                f" | {result['jax_peak_mb']:>10.0f}"
                f" | OK"
            )

        gc.collect()

    print("=" * 105)
    print(f"\nMax successful batch size: {max_batch_size}")

    if results:
        best = max(results, key=lambda r: r["throughput_samples_per_sec"])
        print(f"Best throughput: {best['throughput_samples_per_sec']:.1f} "
              f"samples/s at batch_size={best['batch_size']}")
        print(f"  preprocess={best['preprocess_ms_mean']:.1f}ms  "
              f"forward={best['forward_ms_mean']:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LAVA VLA inference")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing the LAVA Flax checkpoint")
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="bc_resnet_sim_checkpoint_")
    parser.add_argument("--batch_sizes", type=str,
                        default="1,2,4,8,16,32,64,128,256,512",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--modes", type=str, default="original",
                        help="Comma-separated preprocess modes to benchmark "
                             "(original, batched_tf, jax_gpu)")
    parser.add_argument("--accuracy_steps", type=int, default=3,
                        help="Number of predict steps for accuracy comparison "
                             "(only used when --modes has multiple entries)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    modes = [x.strip() for x in args.modes.split(",")]

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from language_table.lamer.lava_policy import LAVAPolicy

    gpu_used_init, gpu_total = get_gpu_memory_mb()
    logger.info("GPU: %d MB used / %d MB total (before model)", gpu_used_init, gpu_total)
    logger.info("JAX devices: %s  backend: %s", jax.devices(), jax.default_backend())

    logger.info("Loading LAVA policy from %s ...", args.checkpoint_dir)
    t0 = time.time()
    policy = LAVAPolicy(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        preprocess_mode=modes[0],
    )
    load_time = time.time() - t0
    gpu_used_model, _ = get_gpu_memory_mb()
    logger.info("Policy loaded in %.1fs (GPU: %d MB)", load_time, gpu_used_model)
    if len(modes) > 1:
        logger.info("Benchmarking preprocess modes: %s", modes)

    # --- Speed + memory benchmark for each mode ---
    all_mode_results = {}
    for mode in modes:
        all_mode_results[mode] = run_single_mode(
            policy, mode, batch_sizes, args.num_warmup, args.num_iters)

    # --- Cross-mode comparison (only with multiple modes) ---
    if len(modes) > 1:
        print_mode_comparison(all_mode_results, modes, batch_sizes)

        # Accuracy on a subset of batch sizes to keep runtime reasonable
        accuracy_bs = [bs for bs in batch_sizes if bs <= 2048]
        benchmark_accuracy(
            policy, modes, accuracy_bs, args.accuracy_steps)

    # --- Training implications (use baseline mode) ---
    results = all_mode_results.get(modes[0], [])
    print("\n" + "=" * 80)
    print("IMPLICATIONS FOR TRAINING")
    print("=" * 80)
    if results:
        print(f"\n{'Envs':>6} | {'Inner Steps':>12} | {'VLA Time (s)':>13} | Notes")
        print("-" * 60)
        for target_envs in [16, 32, 64, 128, 256, 512, 1024]:
            matching = [r for r in results if r["batch_size"] >= target_envs]
            if not matching:
                print(f"{target_envs:>6} | {'---':>12} | {'---':>13} | exceeds max batch")
                continue
            r = min(matching, key=lambda x: x["batch_size"])
            for inner_steps in [5, 10, 25, 50, 100]:
                total_s = r["total_ms_mean"] * inner_steps / 1000
                note = ""
                if total_s > 60:
                    note = "SLOW (>60s)"
                elif total_s > 30:
                    note = "borderline"
                print(f"{target_envs:>6} | {inner_steps:>12} | {total_s:>13.1f} | {note}")
            print("-" * 60)


if __name__ == "__main__":
    main()
