"""
Benchmark Language Table environment scaling with VLA inner loop.

Measures:
  1. End-to-end step time (VLA inference + PyBullet sim) vs number of envs
  2. CPU utilization vs number of envs
  3. Breakdown: VLA inference time vs env.step() time per inner step
  4. Optimal number of envs for a given CPU budget

Runs in the language-table conda env (ltvenv).

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    ltvenv/bin/python -m language_table.lamer.benchmark_envs_standalone \
        --checkpoint_dir /path/to/checkpoints/ \
        --env_counts 4,8,16,32,64,128 \
        --inner_steps 20 --cpus_per_gpu 8
"""

import argparse
import gc
import logging
import os
import time

import numpy as np
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_gpu_memory_mb():
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


def decode_instruction(instruction_array):
    arr = np.asarray(instruction_array)
    non_zero = arr[arr != 0]
    if non_zero.shape[0] == 0:
        return ""
    return bytes(non_zero.tolist()).decode("utf-8")


def benchmark_env_scaling(num_envs, checkpoint_dir, checkpoint_prefix,
                          inner_steps, block_mode, seed):
    """Run a full benchmark for a given number of environments.

    Creates a fresh ray cluster, envs, and policy for each run to get clean
    measurements.  Tears everything down before returning.
    """
    import ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)

    from language_table.environments.rewards.block2block import BlockToBlockReward
    from language_table.lamer.envs import LanguageTableMultiProcessEnv
    from language_table.lamer.lava_policy import LAVAPolicy

    # Create envs
    logger.info("Creating %d environments...", num_envs)
    t0 = time.time()
    envs = LanguageTableMultiProcessEnv(
        num_envs=num_envs,
        block_mode=block_mode,
        reward_factory_cls=BlockToBlockReward,
        seed=seed,
        group_n=1,
        return_full_state=True,
        render_obs=True,
    )
    env_create_time = time.time() - t0
    logger.info("Envs created in %.1fs", env_create_time)

    # Load VLA policy
    logger.info("Loading LAVA policy...")
    t0 = time.time()
    policy = LAVAPolicy(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
    )
    policy_load_time = time.time() - t0

    # Reset
    logger.info("Resetting %d environments...", num_envs)
    t0 = time.time()
    obs_list, infos = envs.reset()
    reset_time = time.time() - t0

    instructions = [decode_instruction(obs.get("instruction", [])) for obs in obs_list]

    policy.reset(num_envs=num_envs)
    active_mask = np.ones(num_envs, dtype=bool)

    # Run inner loop with detailed timing
    vla_times = []
    env_step_times = []
    total_step_times = []
    cpu_percents = []

    for step in range(inner_steps):
        _ = psutil.cpu_percent(interval=None)  # prime the counter

        t_total = time.perf_counter()

        # VLA inference
        t_vla = time.perf_counter()
        actions = policy.predict(instructions, obs_list, active_mask)
        vla_time = time.perf_counter() - t_vla

        # Environment step (PyBullet physics + render on Ray workers)
        t_env = time.perf_counter()
        obs_list, rewards, dones, step_infos = envs.step(actions, active_mask=active_mask)
        env_time = time.perf_counter() - t_env

        total_time = time.perf_counter() - t_total
        cpu_after = psutil.cpu_percent(interval=None)

        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)
        active_mask &= ~dones

        # Skip first step (JIT warmup)
        if step > 0:
            vla_times.append(vla_time)
            env_step_times.append(env_time)
            total_step_times.append(total_time)
            cpu_percents.append(cpu_after)

        if not active_mask.any():
            break

    gpu_used, gpu_total = get_gpu_memory_mb()

    envs.close()
    ray.shutdown()

    return {
        "num_envs": num_envs,
        "env_create_time_s": env_create_time,
        "reset_time_s": reset_time,
        "steps_completed": len(total_step_times),
        "vla_ms_mean": np.mean(vla_times) * 1000 if vla_times else 0,
        "vla_ms_p50": np.percentile(vla_times, 50) * 1000 if vla_times else 0,
        "vla_ms_p95": np.percentile(vla_times, 95) * 1000 if vla_times else 0,
        "env_step_ms_mean": np.mean(env_step_times) * 1000 if env_step_times else 0,
        "env_step_ms_p50": np.percentile(env_step_times, 50) * 1000 if env_step_times else 0,
        "env_step_ms_p95": np.percentile(env_step_times, 95) * 1000 if env_step_times else 0,
        "total_ms_mean": np.mean(total_step_times) * 1000 if total_step_times else 0,
        "total_ms_p50": np.percentile(total_step_times, 50) * 1000 if total_step_times else 0,
        "total_ms_p95": np.percentile(total_step_times, 95) * 1000 if total_step_times else 0,
        "throughput_envsteps_per_sec": num_envs / np.mean(total_step_times) if total_step_times else 0,
        "cpu_percent_mean": np.mean(cpu_percents) if cpu_percents else 0,
        "gpu_used_mb": gpu_used,
        "gpu_total_mb": gpu_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Language Table env scaling")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="bc_resnet_sim_checkpoint_")
    parser.add_argument("--env_counts", type=str, default="4,8,16,32,64,128",
                        help="Comma-separated env counts to test")
    parser.add_argument("--inner_steps", type=int, default=20,
                        help="Inner-loop steps per benchmark run")
    parser.add_argument("--block_mode", type=str, default="BLOCK_4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus_per_gpu", type=int, default=8,
                        help="CPU budget per GPU (for analysis)")
    parser.add_argument("--time_budget_s", type=float, default=30.0,
                        help="Max acceptable outer-step time in seconds")
    args = parser.parse_args()

    env_counts = [int(x) for x in args.env_counts.split(",")]

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    total_cpus = psutil.cpu_count(logical=True)
    logger.info("System CPUs: %d (budget per GPU: %d)", total_cpus, args.cpus_per_gpu)

    results = []

    print("\n" + "=" * 140)
    print("ENVIRONMENT SCALING BENCHMARK (VLA + PyBullet)")
    print("=" * 140)
    print(
        f"{'Envs':>6} | {'Create':>8} | {'Reset':>8} | "
        f"{'VLA (ms)':>14} | {'Env Step (ms)':>14} | {'Total (ms)':>14} | "
        f"{'Throughput':>12} | {'CPU %':>7} | {'GPU MB':>8} | "
        f"{'5-step(s)':>10} | {'100-step(s)':>11}"
    )
    print("-" * 140)

    for n in env_counts:
        try:
            r = benchmark_env_scaling(
                n, args.checkpoint_dir, args.checkpoint_prefix,
                args.inner_steps, args.block_mode, args.seed,
            )
            results.append(r)

            time_5 = r["total_ms_mean"] * 5 / 1000
            time_100 = r["total_ms_mean"] * 100 / 1000
            flag = "SLOW" if time_5 > args.time_budget_s else ""

            print(
                f"{n:>6} "
                f"| {r['env_create_time_s']:>8.1f} "
                f"| {r['reset_time_s']:>8.1f} "
                f"| {r['vla_ms_mean']:>8.1f} p95={r['vla_ms_p95']:>5.0f}"
                f"| {r['env_step_ms_mean']:>8.1f} p95={r['env_step_ms_p95']:>5.0f}"
                f"| {r['total_ms_mean']:>8.1f} p95={r['total_ms_p95']:>5.0f}"
                f"| {r['throughput_envsteps_per_sec']:>12.1f}"
                f" | {r['cpu_percent_mean']:>6.1f}%"
                f" | {r['gpu_used_mb']:>8}"
                f" | {time_5:>10.1f}"
                f" | {time_100:>11.1f}  {flag}"
            )

        except Exception as e:
            logger.error("Failed at num_envs=%d: %s", n, e, exc_info=True)
            print(f"{n:>6} | FAILED: {e}")

        gc.collect()

    print("=" * 140)

    # Analysis
    if not results:
        return

    print("\n" + "=" * 90)
    print("BOTTLENECK ANALYSIS")
    print("=" * 90)
    for r in results:
        total = r["total_ms_mean"]
        if total == 0:
            continue
        vla_pct = r["vla_ms_mean"] / total * 100
        env_pct = r["env_step_ms_mean"] / total * 100
        overhead = 100 - vla_pct - env_pct
        bottleneck = "VLA (GPU)" if vla_pct > env_pct else "ENV (CPU)"
        print(
            f"  {r['num_envs']:>4} envs: "
            f"VLA={vla_pct:>5.1f}%  Env={env_pct:>5.1f}%  other={overhead:>4.1f}%  "
            f"→ bottleneck: {bottleneck}"
        )

    print("\n" + "=" * 90)
    print(f"RECOMMENDED ENV COUNTS (time_budget={args.time_budget_s}s, cpus_per_gpu={args.cpus_per_gpu})")
    print("=" * 90)
    for inner_steps in [5, 10, 25, 50, 100]:
        best_envs = 0
        for r in results:
            outer_step_s = r["total_ms_mean"] * inner_steps / 1000
            if outer_step_s <= args.time_budget_s:
                best_envs = r["num_envs"]
        if best_envs > 0:
            matching = next(r for r in results if r["num_envs"] == best_envs)
            outer_s = matching["total_ms_mean"] * inner_steps / 1000
            print(f"  inner_steps={inner_steps:>3}: max {best_envs:>4} envs ({outer_s:.1f}s per outer step)")
        else:
            print(f"  inner_steps={inner_steps:>3}: even {results[0]['num_envs']} envs exceeds budget")


if __name__ == "__main__":
    main()
