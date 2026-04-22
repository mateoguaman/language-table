#!/usr/bin/env python3
"""Probe the max per-GPU batch size for a LeRobot policy on a given dataset.

Runs `lerobot_train` as a subprocess for each candidate batch size (one GPU,
a handful of training steps), polls `nvidia-smi` to record peak memory, and
detects OOM via non-zero exit code. Prints a table of results and a
recommendation.

Why subprocess: we want the full training path (make_dataset + make_policy +
make_pre_post_processors + accelerate.prepare + autocast(bf16) + forward +
backward + optimizer.step). Building that inline via `TrainPipelineConfig`
fights draccus/CLI parsing. A subprocess call is the cleanest match to what
a real run actually does.

Usage (run on a single H200 with CUDA_VISIBLE_DEVICES=0 from an interactive
shell, after the dataset has been downloaded locally):

    CUDA_VISIBLE_DEVICES=0 ./lerobot_env_v51/bin/python training/probe_batch_size.py \
        --dataset_repo mateoguaman/language_table_sim_combined \
        --dataset_root /media/mateo/Storage/lerobot_datasets_v3/language_table_sim_combined \
        --batch_sizes 16,32,64,96,128,192,256

On Tillicum inside an allocation:
    srun --gres=gpu:1 --time=1:00:00 --pty bash
    module load ...; conda activate lerobot
    python training/probe_batch_size.py \
        --dataset_repo mateoguaman/language_table_sim_combined \
        --dataset_root "${DATASET_ROOT}/language_table_sim_combined" \
        --batch_sizes 32,64,96,128,192,256
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Preset configs encode (a) which pretrained checkpoint to load and (b) the
# freeze/train-expert knobs needed to flip between "expert-only" (action head
# + minimal projections) and "full" (everything unfrozen). Upstream defaults:
#   smolvla_base: train_expert_only=True,  freeze_vision_encoder=True  -> expert
#   pi0:          train_expert_only=False, freeze_vision_encoder=False -> full
#   pi05:         train_expert_only=False, freeze_vision_encoder=False -> full
# We pin each variant explicitly so probes don't silently inherit a default
# you didn't mean to test.
#
# rename_map + empty_cameras pad our 1-camera Language Table dataset up to the
# multi-camera layouts each VLA family expects. SmolVLA/pi0 use generic
# camera{1,2,3} naming; pi0.5 uses descriptive names
# (base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb). _preprocess_images only
# requires at least one declared image key to be present in the batch — the
# rest get zero-filled via the img_mask path.
_SMOLVLA_IMAGE_ARGS = [
    "--policy.empty_cameras=2",
    '--rename_map={"observation.images.rgb": "observation.images.camera1"}',
]
_PI0_IMAGE_ARGS = _SMOLVLA_IMAGE_ARGS
_PI05_IMAGE_ARGS = [
    "--policy.empty_cameras=2",
    '--rename_map={"observation.images.rgb": "observation.images.base_0_rgb"}',
]

PRESETS = {
    "smolvla_expert": {
        "policy_path": "lerobot/smolvla_base",
        "extra": _SMOLVLA_IMAGE_ARGS + [
            "--policy.train_expert_only=true",
            "--policy.freeze_vision_encoder=true",
        ],
    },
    "smolvla_full": {
        "policy_path": "lerobot/smolvla_base",
        "extra": _SMOLVLA_IMAGE_ARGS + [
            "--policy.train_expert_only=false",
            "--policy.freeze_vision_encoder=false",
        ],
    },
    "pi0_expert": {
        "policy_path": "lerobot/pi0",
        "extra": _PI0_IMAGE_ARGS + [
            "--policy.train_expert_only=true",
            "--policy.freeze_vision_encoder=true",
        ],
    },
    "pi0_full": {
        "policy_path": "lerobot/pi0",
        "extra": _PI0_IMAGE_ARGS + [
            "--policy.train_expert_only=false",
            "--policy.freeze_vision_encoder=false",
        ],
    },
    "pi05_expert": {
        "policy_path": "lerobot/pi05_base",
        "extra": _PI05_IMAGE_ARGS + [
            "--policy.train_expert_only=true",
            "--policy.freeze_vision_encoder=true",
        ],
    },
    "pi05_full": {
        "policy_path": "lerobot/pi05_base",
        "extra": _PI05_IMAGE_ARGS + [
            "--policy.train_expert_only=false",
            "--policy.freeze_vision_encoder=false",
        ],
    },
}


def run_smi_once(gpu_index: int) -> int:
    """Return used MiB for the given GPU index, or 0 on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def gpu_total_mib(gpu_index: int) -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def parse_batch_sizes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_cmd(args, batch_size: int, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={args.policy_path}",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--batch_size={batch_size}",
        f"--steps={args.steps}",
        "--save_checkpoint=false",
        "--log_freq=1",
        "--eval_freq=0",
        f"--num_workers={args.num_workers}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.chunk_size}",
        f"--dataset.video_backend={args.video_backend}",
        f"--output_dir={output_dir}",
        "--wandb.enable=false",
        "--seed=1000",
        "--policy.push_to_hub=false",
    ]
    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    cmd.extend(args.preset_extra)
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def probe_one(args, batch_size: int, gpu_index: int) -> dict:
    """Run one training attempt and record peak memory + status."""
    with tempfile.TemporaryDirectory(prefix=f"probe_bs{batch_size}_") as tmpd:
        out_dir = Path(tmpd) / "run"
        cmd = build_cmd(args, batch_size, out_dir)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # Silence tokenizer fork warning; also keep wandb off if somehow enabled.
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("WANDB_MODE", "disabled")

        baseline = run_smi_once(0)  # after CUDA_VISIBLE_DEVICES, our GPU is id 0

        log_path = Path(tmpd) / "train.log"
        log_f = log_path.open("w")
        print(f"  [bs={batch_size}] launching: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # own process group so we can kill children
        )

        peak = 0
        t_start = time.time()
        try:
            while proc.poll() is None:
                used = run_smi_once(0)
                if used > peak:
                    peak = used
                if time.time() - t_start > args.timeout_s:
                    print(f"  [bs={batch_size}] TIMEOUT after {args.timeout_s}s — killing")
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(3)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(args.poll_interval_s)
        finally:
            log_f.close()

        rc = proc.returncode if proc.returncode is not None else -1
        log_text = log_path.read_text(errors="replace")

        oom = False
        lower = log_text.lower()
        if "out of memory" in lower or "cuda out of memory" in lower or "outofmemoryerror" in lower:
            oom = True

        # Pull the first "avg_loss" / loss line as a sanity signal that we really trained
        loss_line = None
        for line in log_text.splitlines():
            if "loss:" in line and "step:" in line:
                loss_line = line.strip()
                break

        return {
            "batch_size": batch_size,
            "rc": rc,
            "fit": rc == 0,
            "oom": oom,
            "peak_used_mib": peak,
            "baseline_mib": baseline,
            "peak_delta_mib": max(0, peak - baseline),
            "sample_log_line": loss_line,
            "log_excerpt": log_text[-2000:] if (rc != 0) else "",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()),
                    default="smolvla_expert",
                    help="Policy + training-mode preset. `*_expert` freezes the VLM "
                         "backbone; `*_full` trains everything. Overrides --policy_path.")
    ap.add_argument("--policy_path", default=None,
                    help="Override the preset's pretrained path (optional)")
    ap.add_argument("--extra_args", default=None,
                    help="Extra CLI args appended verbatim to lerobot_train, shlex-split "
                         "(e.g. '--policy.train_expert_only=false --policy.freeze_vision_encoder=false')")
    ap.add_argument("--dataset_repo", required=True)
    ap.add_argument("--dataset_root", default=None,
                    help="Local path to dataset; skip to load via Hub cache")
    ap.add_argument("--batch_sizes", default="16,32,64,96,128,192,256",
                    help="Comma-separated list of batch sizes to try")
    ap.add_argument("--steps", type=int, default=5,
                    help="Training steps per probe — small but >1 so allocator settles")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--chunk_size", type=int, default=10)
    ap.add_argument("--video_backend", default="pyav")
    ap.add_argument("--gpu_index", type=int, default=0,
                    help="Physical GPU to bind CUDA_VISIBLE_DEVICES to")
    ap.add_argument("--poll_interval_s", type=float, default=0.5)
    ap.add_argument("--timeout_s", type=int, default=600,
                    help="Kill the probe subprocess after this many seconds")
    ap.add_argument("--output_json", default=None,
                    help="Optional path to dump full results as JSON")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    if args.policy_path is None:
        args.policy_path = preset["policy_path"]
    args.preset_extra = preset["extra"]

    sizes = parse_batch_sizes(args.batch_sizes)
    if not sizes:
        print("No batch sizes given", file=sys.stderr)
        sys.exit(2)

    total_mib = gpu_total_mib(args.gpu_index)
    print(f"GPU {args.gpu_index} total memory: {total_mib} MiB "
          f"(~{total_mib/1024:.1f} GiB)")
    print(f"Preset: {args.preset}")
    print(f"Policy: {args.policy_path}")
    print(f"Preset flags: {' '.join(args.preset_extra)}")
    if args.extra_args:
        print(f"Extra flags: {args.extra_args}")
    print(f"Dataset: {args.dataset_repo}  root={args.dataset_root or '<hub cache>'}")
    print(f"Probing batch sizes: {sizes}")
    print()

    results = []
    last_fit = None
    for bs in sizes:
        t0 = time.time()
        res = probe_one(args, bs, args.gpu_index)
        res["wall_s"] = round(time.time() - t0, 1)
        results.append(res)
        status = "FIT " if res["fit"] else ("OOM " if res["oom"] else "FAIL")
        pct = (res["peak_used_mib"] / total_mib * 100.0) if total_mib else 0.0
        print(f"  bs={bs:>4}  {status}  peak={res['peak_used_mib']:>6} MiB "
              f"({pct:5.1f}% of total)  Δ={res['peak_delta_mib']:>6} MiB  "
              f"t={res['wall_s']:>5.1f}s  rc={res['rc']}")
        if res["sample_log_line"]:
            print(f"        training line: {res['sample_log_line']}")
        if not res["fit"] and not res["oom"]:
            # Non-OOM failure (config error, missing dep, etc.) — show tail to help debug
            print(f"        (non-OOM failure — last log lines:)")
            for ln in res["log_excerpt"].splitlines()[-20:]:
                print(f"          | {ln}")
        if res["fit"]:
            last_fit = bs
        else:
            # Monotonic assumption: if bs=N fails with OOM, bs>N will too.
            if res["oom"]:
                print(f"  [stop] OOM at bs={bs} — skipping larger sizes")
                break

    print()
    print("=" * 72)
    if last_fit is None:
        print("No batch size fit. Try smaller --batch_sizes.")
    else:
        recommend = int(last_fit * 0.9)
        # Snap to multiple of 4 for DDP-friendliness
        recommend = max(1, (recommend // 4) * 4)
        print(f"Largest batch size that fit: {last_fit}")
        print(f"Recommended (90% of max, rounded to multiple of 4): {recommend}")
        print(f"Usage: BATCH_SIZE={recommend} sbatch training/slurm/train.slurm")
    print("=" * 72)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(
            {"gpu_total_mib": total_mib, "results": results, "largest_fit": last_fit},
            indent=2,
        ))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
