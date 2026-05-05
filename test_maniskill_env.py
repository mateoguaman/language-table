"""Benchmarks for ManiSkill and original Language Table environments."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np

import language_table.environments.maniskill_env  # registers LanguageTable-v1
from language_table.environments import blocks
from language_table.environments import language_table


DEFAULT_MANISKILL_ENV_COUNTS = (1, 10, 100, 1000, 10000)

BLOCK_MODE_CHOICES = {
    mode.value: mode
    for mode in (
        blocks.LanguageTableBlockVariants.BLOCK_4,
        blocks.LanguageTableBlockVariants.BLOCK_8,
    )
}


@dataclass
class TrialMetrics:
    reset_s: float
    rollout_s: float
    resets_during_rollout: int


@dataclass
class BenchmarkResult:
    name: str
    device_type: str
    num_envs: int
    construction_s: float
    trials: List[TrialMetrics]


@dataclass
class BenchmarkSpec:
    name: str
    make_env: Callable[[blocks.LanguageTableBlockVariants, int], object]
    reset_env: Callable[[object, int], None]
    step_env: Callable[[object, np.ndarray], bool]
    close_env: Callable[[object], None]


@dataclass
class ScalingResult:
    num_envs: int
    device_type: str
    construction_s: float
    reset_s: float
    rollout_s: float
    control_steps_per_s: float
    env_steps_per_s: float
    per_step_s: float


def _make_maniskill_env(
    *,
    block_mode: blocks.LanguageTableBlockVariants,
    num_envs: int,
    obs_mode: str,
) -> object:
    return gym.make(
        "LanguageTable-v1",
        obs_mode=obs_mode,
        control_mode="pd_ee_delta_pos",
        num_envs=num_envs,
        block_mode=block_mode,
    )


def _make_single_maniskill_env(
    block_mode: blocks.LanguageTableBlockVariants, seed: int
) -> object:
    del seed
    return _make_maniskill_env(
        block_mode=block_mode,
        num_envs=1,
        obs_mode="sensor_data",
    )


def _reset_maniskill_env(env: object, seed: int) -> None:
    env.reset(seed=seed)


def _step_maniskill_env(env: object, action: np.ndarray) -> bool:
    _, _, terminated, truncated, _ = env.step(action)
    return bool(np.any(np.asarray(terminated))) or bool(np.any(np.asarray(truncated)))


def _close_maniskill_env(env: object) -> None:
    env.close()


def _make_pybullet_env(
    block_mode: blocks.LanguageTableBlockVariants, seed: int
) -> object:
    return language_table.LanguageTable(
        block_mode=block_mode,
        reward_factory=None,
        control_frequency=10.0,
        seed=seed,
        render_text_in_image=False,
    )


def _reset_pybullet_env(env: object, seed: int) -> None:
    del seed
    env.reset()


def _step_pybullet_env(env: object, action: np.ndarray) -> bool:
    _, _, done, _ = env.step(action)
    return bool(done)


def _close_pybullet_env(env: object) -> None:
    client = getattr(env, "pybullet_client", None)
    if client is not None:
        client.disconnect()


MANISKILL_SPEC = BenchmarkSpec(
    name="ManiSkill",
    make_env=_make_single_maniskill_env,
    reset_env=_reset_maniskill_env,
    step_env=_step_maniskill_env,
    close_env=_close_maniskill_env,
)

PYBULLET_SPEC = BenchmarkSpec(
    name="Original Language Table",
    make_env=_make_pybullet_env,
    reset_env=_reset_pybullet_env,
    step_env=_step_pybullet_env,
    close_env=_close_pybullet_env,
)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1_000.0:.2f} ms"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf")
    return numerator / denominator


def _generate_action_batches(length: int, count: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(-0.1, 0.1, size=(length, 2)).astype(np.float32)
        for _ in range(count)
    ]


def _generate_vector_actions(length: int, num_envs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, size=(length, num_envs, 2)).astype(np.float32)


def _parse_env_counts(value: str) -> List[int]:
    env_counts = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        env_count = int(part)
        if env_count <= 0:
            raise ValueError("Environment counts must be positive integers.")
        env_counts.append(env_count)
    if not env_counts:
        raise ValueError("At least one environment count is required.")
    return env_counts


def _resolve_env_runtime_info(env: object) -> tuple[str, int]:
    unwrapped = getattr(env, "unwrapped", env)
    raw_device = getattr(unwrapped, "device", None)
    if raw_device is None:
        raw_device = getattr(env, "device", None)

    if raw_device is None:
        device_type = "cpu"
    else:
        device_type = getattr(raw_device, "type", str(raw_device))

    raw_num_envs = getattr(unwrapped, "num_envs", None)
    if raw_num_envs is None:
        raw_num_envs = getattr(env, "num_envs", None)
    if raw_num_envs is None:
        raw_num_envs = 1

    return device_type, int(raw_num_envs)


def _print_compare_result(result: BenchmarkResult, num_steps: int) -> Dict[str, float]:
    reset_times = [trial.reset_s for trial in result.trials]
    rollout_times = [trial.rollout_s for trial in result.trials]
    step_times = [trial.rollout_s / num_steps for trial in result.trials]
    throughputs = [num_steps / trial.rollout_s for trial in result.trials]
    rollout_resets = [trial.resets_during_rollout for trial in result.trials]

    summary = {
        "construction_s": result.construction_s,
        "reset_mean_s": statistics.mean(reset_times),
        "rollout_mean_s": statistics.mean(rollout_times),
        "step_mean_s": statistics.mean(step_times),
        "throughput_mean": statistics.mean(throughputs),
        "rollout_std_s": statistics.stdev(rollout_times) if len(rollout_times) > 1 else 0.0,
        "reset_std_s": statistics.stdev(reset_times) if len(reset_times) > 1 else 0.0,
        "step_std_s": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "throughput_std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "max_rollout_resets": max(rollout_resets),
    }

    print(f"\n{result.name}")
    print(f"  device: {result.device_type}")
    print(f"  num_envs: {result.num_envs}")
    print(f"  construction: {_format_ms(summary['construction_s'])}")
    print(
        "  reset: "
        f"{_format_ms(summary['reset_mean_s'])} avg"
        f" (std {_format_ms(summary['reset_std_s'])})"
    )
    print(
        f"  {num_steps}-step rollout: "
        f"{_format_ms(summary['rollout_mean_s'])} avg"
        f" (std {_format_ms(summary['rollout_std_s'])})"
    )
    print(
        "  per-step: "
        f"{_format_ms(summary['step_mean_s'])} avg"
        f" (std {_format_ms(summary['step_std_s'])})"
    )
    print(
        "  throughput: "
        f"{summary['throughput_mean']:.2f} steps/s"
        f" (std {summary['throughput_std']:.2f})"
    )
    print(f"  resets during rollout (max trial): {summary['max_rollout_resets']}")
    return summary


def run_backend_benchmark(
    spec: BenchmarkSpec,
    block_mode: blocks.LanguageTableBlockVariants,
    seed: int,
    warmup_actions: np.ndarray,
    trial_actions: List[np.ndarray],
) -> BenchmarkResult:
    construction_start = time.perf_counter()
    env = spec.make_env(block_mode, seed)
    construction_s = time.perf_counter() - construction_start
    device_type, num_envs = _resolve_env_runtime_info(env)

    try:
        if len(warmup_actions) > 0:
            spec.reset_env(env, seed)
            for action in warmup_actions:
                if spec.step_env(env, action):
                    spec.reset_env(env, seed)

        trials: List[TrialMetrics] = []
        for trial_idx, actions in enumerate(trial_actions):
            trial_seed = seed + trial_idx

            reset_start = time.perf_counter()
            spec.reset_env(env, trial_seed)
            reset_s = time.perf_counter() - reset_start

            rollout_start = time.perf_counter()
            resets_during_rollout = 0
            for step_idx, action in enumerate(actions):
                episode_done = spec.step_env(env, action)
                if episode_done and step_idx + 1 < len(actions):
                    resets_during_rollout += 1
                    spec.reset_env(env, trial_seed + resets_during_rollout)
            rollout_s = time.perf_counter() - rollout_start

            trials.append(
                TrialMetrics(
                    reset_s=reset_s,
                    rollout_s=rollout_s,
                    resets_during_rollout=resets_during_rollout,
                )
            )

        return BenchmarkResult(
            name=spec.name,
            device_type=device_type,
            num_envs=num_envs,
            construction_s=construction_s,
            trials=trials,
        )
    finally:
        spec.close_env(env)


def run_compare_backends(
    *,
    num_steps: int,
    num_trials: int,
    warmup_steps: int,
    block_mode: blocks.LanguageTableBlockVariants,
    seed: int,
) -> None:
    warmup_actions = _generate_action_batches(length=warmup_steps, count=1, seed=seed)[0]
    trial_actions = _generate_action_batches(
        length=num_steps,
        count=num_trials,
        seed=seed + 1,
    )

    print("Language Table backend comparison")
    print(f"  block mode: {block_mode.value}")
    print(f"  trials: {num_trials}")
    print(f"  rollout length: {num_steps} steps")
    print(f"  warmup steps: {warmup_steps}")
    print("  action distribution: uniform in [-0.1, 0.1] for (dx, dy)")
    print("  note: includes observation generation, but not video recording")

    maniskill_result = run_backend_benchmark(
        spec=MANISKILL_SPEC,
        block_mode=block_mode,
        seed=seed,
        warmup_actions=warmup_actions,
        trial_actions=trial_actions,
    )
    pybullet_result = run_backend_benchmark(
        spec=PYBULLET_SPEC,
        block_mode=block_mode,
        seed=seed,
        warmup_actions=warmup_actions,
        trial_actions=trial_actions,
    )

    maniskill_summary = _print_compare_result(maniskill_result, num_steps)
    pybullet_summary = _print_compare_result(pybullet_result, num_steps)

    rollout_ratio = _safe_ratio(
        pybullet_summary["rollout_mean_s"],
        maniskill_summary["rollout_mean_s"],
    )
    step_ratio = _safe_ratio(
        pybullet_summary["step_mean_s"],
        maniskill_summary["step_mean_s"],
    )
    throughput_ratio = _safe_ratio(
        maniskill_summary["throughput_mean"],
        pybullet_summary["throughput_mean"],
    )

    print("\nRelative comparison")
    print(f"  rollout time ratio (original / ManiSkill): {rollout_ratio:.2f}x")
    print(f"  per-step time ratio (original / ManiSkill): {step_ratio:.2f}x")
    print(f"  throughput ratio (ManiSkill / original): {throughput_ratio:.2f}x")


def run_maniskill_scaling_case(
    *,
    num_envs: int,
    num_steps: int,
    warmup_steps: int,
    block_mode: blocks.LanguageTableBlockVariants,
    obs_mode: str,
    seed: int,
) -> ScalingResult:
    warmup_actions = _generate_vector_actions(
        length=warmup_steps,
        num_envs=num_envs,
        seed=seed,
    )
    rollout_actions = _generate_vector_actions(
        length=num_steps,
        num_envs=num_envs,
        seed=seed + 1,
    )

    construction_start = time.perf_counter()
    env = _make_maniskill_env(
        block_mode=block_mode,
        num_envs=num_envs,
        obs_mode=obs_mode,
    )
    construction_s = time.perf_counter() - construction_start

    try:
        device_type, resolved_num_envs = _resolve_env_runtime_info(env)
        if warmup_steps > 0:
            env.reset(seed=seed)
            for action in warmup_actions:
                env.step(action)

        reset_start = time.perf_counter()
        env.reset(seed=seed + 1)
        reset_s = time.perf_counter() - reset_start

        rollout_start = time.perf_counter()
        for action in rollout_actions:
            env.step(action)
        rollout_s = time.perf_counter() - rollout_start
    finally:
        env.close()

    control_steps_per_s = _safe_ratio(num_steps, rollout_s)
    env_steps_per_s = _safe_ratio(num_steps * num_envs, rollout_s)
    per_step_s = _safe_ratio(rollout_s, num_steps)

    return ScalingResult(
        num_envs=resolved_num_envs,
        device_type=device_type,
        construction_s=construction_s,
        reset_s=reset_s,
        rollout_s=rollout_s,
        control_steps_per_s=control_steps_per_s,
        env_steps_per_s=env_steps_per_s,
        per_step_s=per_step_s,
    )


def _run_maniskill_scaling_case_subprocess(
    *,
    num_envs: int,
    num_steps: int,
    warmup_steps: int,
    block_mode: blocks.LanguageTableBlockVariants,
    obs_mode: str,
    seed: int,
) -> ScalingResult:
    command = [
        sys.executable,
        __file__,
        "--benchmark",
        "maniskill_scaling_case",
        "--num_steps",
        str(num_steps),
        "--warmup_steps",
        str(warmup_steps),
        "--seed",
        str(seed),
        "--block_mode",
        block_mode.value,
        "--maniskill_scaling_obs_mode",
        obs_mode,
        "--maniskill_env_counts",
        str(num_envs),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or (
            f"Subprocess exited with code {completed.returncode}"
        )
        raise RuntimeError(message)

    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("Scaling subprocess produced no output.")

    try:
        payload = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Scaling subprocess did not return valid JSON. "
            f"Last stdout line was: {stdout_lines[-1]!r}"
        ) from exc

    return ScalingResult(**payload)


def _print_scaling_results(results: List[ScalingResult]) -> None:
    if not results:
        print("\nNo ManiSkill scaling results were collected.")
        return

    baseline = results[0]

    print("\nManiSkill scaling results")
    print(
        f"{'Envs':>8} | {'Create':>10} | {'Reset':>10} | {'Rollout':>10} | "
        f"{'Ctrl step/s':>12} | {'Env step/s':>12} | {'x vs 1env':>9} | "
        f"{'Eff.':>8} | {'Delta env step/s':>16} | {'Delta/addl env':>16}"
    )
    print("-" * 132)

    previous: Optional[ScalingResult] = None
    for result in results:
        speedup = _safe_ratio(result.env_steps_per_s, baseline.env_steps_per_s)
        efficiency = _safe_ratio(speedup, result.num_envs / baseline.num_envs)

        if previous is None:
            delta_env_steps_s = None
            delta_per_added_env = None
        else:
            delta_env_steps_s = result.env_steps_per_s - previous.env_steps_per_s
            delta_envs = result.num_envs - previous.num_envs
            delta_per_added_env = _safe_ratio(delta_env_steps_s, delta_envs)

        delta_env_steps_s_str = (
            f"{delta_env_steps_s:,.1f}" if delta_env_steps_s is not None else "-"
        )
        delta_per_added_env_str = (
            f"{delta_per_added_env:,.3f}" if delta_per_added_env is not None else "-"
        )

        print(
            f"{result.num_envs:>8,} | "
            f"{_format_ms(result.construction_s):>10} | "
            f"{_format_ms(result.reset_s):>10} | "
            f"{_format_ms(result.rollout_s):>10} | "
            f"{result.control_steps_per_s:>12,.2f} | "
            f"{result.env_steps_per_s:>12,.2f} | "
            f"{speedup:>9.2f} | "
            f"{efficiency * 100:>7.2f}% | "
            f"{delta_env_steps_s_str:>16} | "
            f"{delta_per_added_env_str:>16}"
        )
        previous = result

    print("\nResolved devices by env count")
    for result in results:
        print(f"  {result.num_envs:,} envs -> {result.device_type}")


def run_maniskill_scaling(
    *,
    env_counts: List[int],
    num_steps: int,
    warmup_steps: int,
    block_mode: blocks.LanguageTableBlockVariants,
    obs_mode: str,
    seed: int,
) -> None:
    print("ManiSkill vectorized scaling benchmark")
    print(f"  block mode: {block_mode.value}")
    print(f"  obs mode: {obs_mode}")
    print(f"  env counts: {', '.join(str(n) for n in env_counts)}")
    print(f"  measured rollout length: {num_steps} vector steps")
    print(f"  warmup steps: {warmup_steps}")
    print("  action distribution: uniform in [-0.1, 0.1] for (dx, dy)")
    print("  throughput metric: total env-steps / measured rollout time")

    results: List[ScalingResult] = []
    failures: List[tuple[int, str]] = []
    for idx, num_envs in enumerate(env_counts):
        print(f"Evaluating ManiSkill scaling case: num_envs={num_envs}")
        try:
            result = _run_maniskill_scaling_case_subprocess(
                num_envs=num_envs,
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                block_mode=block_mode,
                obs_mode=obs_mode,
                seed=seed + idx * 1000,
            )
            results.append(result)
        except Exception as exc:  # pragma: no cover - benchmark failure path
            failures.append((num_envs, str(exc)))

    _print_scaling_results(results)

    if failures:
        print("\nScaling failures")
        for num_envs, error in failures:
            print(f"  {num_envs:,} envs: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ManiSkill and original Language Table environments."
    )
    parser.add_argument(
        "--benchmark",
        choices=(
            "compare_backends",
            "maniskill_scaling",
            "maniskill_scaling_case",
            "all",
        ),
        default="compare_backends",
    )
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--block_mode",
        choices=sorted(BLOCK_MODE_CHOICES),
        default=blocks.LanguageTableBlockVariants.BLOCK_4.value,
    )
    parser.add_argument(
        "--maniskill_env_counts",
        default=",".join(str(n) for n in DEFAULT_MANISKILL_ENV_COUNTS),
        help="Comma-separated env counts for ManiSkill scaling runs.",
    )
    parser.add_argument(
        "--maniskill_scaling_obs_mode",
        default="state_dict",
        help="Observation mode for ManiSkill scaling. "
        "Use a lightweight mode for large env counts.",
    )
    args = parser.parse_args()

    if args.num_steps <= 0:
        raise ValueError("--num_steps must be positive.")
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative.")

    block_mode = BLOCK_MODE_CHOICES[args.block_mode]
    if args.benchmark in ("compare_backends", "all"):
        run_compare_backends(
            num_steps=args.num_steps,
            num_trials=args.num_trials,
            warmup_steps=args.warmup_steps,
            block_mode=block_mode,
            seed=args.seed,
        )

    if args.benchmark == "all":
        print("\n" + "=" * 80 + "\n")

    if args.benchmark in ("maniskill_scaling", "all"):
        env_counts = _parse_env_counts(args.maniskill_env_counts)
        run_maniskill_scaling(
            env_counts=env_counts,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            block_mode=block_mode,
            obs_mode=args.maniskill_scaling_obs_mode,
            seed=args.seed,
        )

    if args.benchmark == "maniskill_scaling_case":
        env_counts = _parse_env_counts(args.maniskill_env_counts)
        if len(env_counts) != 1:
            raise ValueError(
                "--benchmark maniskill_scaling_case requires exactly one env count."
            )
        result = run_maniskill_scaling_case(
            num_envs=env_counts[0],
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            block_mode=block_mode,
            obs_mode=args.maniskill_scaling_obs_mode,
            seed=args.seed,
        )
        print(json.dumps(result.__dict__))


if __name__ == "__main__":
    main()
