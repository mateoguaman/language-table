"""Validate a converted LeRobot Language Table dataset against its RLDS source.

Unlike validate_hf_dataset.py (which compares the IPEC-COMMUNITY HF dataset at
the distribution level because its episode ordering differs), our converter
preserves RLDS iteration order. This validator does a *positional* comparison
between RLDS episode[i] and LeRobot episode[i], which lets us check exact
equality for actions, states, rewards, and done flags.

Usage:
    source ltvenv/bin/activate
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python validate_lerobot_dataset.py \\
        --dataset_name language_table_blocktoblock_sim \\
        --output_dir ./lerobot_datasets \\
        [--num_episodes 20] [--skip_video]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# Must match convert_to_lerobot.py.
CHUNK_SIZE = 1000
GCS_BASE = "gs://gresearch/robotics"
DATASETS = {
    "language_table":                                 {"path": f"{GCS_BASE}/language_table/0.0.1/",                                 "expected_episodes": 442_226},
    "language_table_sim":                             {"path": f"{GCS_BASE}/language_table_sim/0.0.1/",                             "expected_episodes": 181_020},
    "language_table_blocktoblock_sim":                {"path": f"{GCS_BASE}/language_table_blocktoblock_sim/0.0.1/",                "expected_episodes":   8_000},
    "language_table_blocktoblock_4block_sim":         {"path": f"{GCS_BASE}/language_table_blocktoblock_4block_sim/0.0.1/",         "expected_episodes":   8_298},
    "language_table_blocktoblock_oracle_sim":         {"path": f"{GCS_BASE}/language_table_blocktoblock_oracle_sim/0.0.1/",         "expected_episodes": 200_000},
    "language_table_blocktoblockrelative_oracle_sim": {"path": f"{GCS_BASE}/language_table_blocktoblockrelative_oracle_sim/0.0.1/", "expected_episodes": 200_000},
    "language_table_blocktoabsolute_oracle_sim":      {"path": f"{GCS_BASE}/language_table_blocktoabsolute_oracle_sim/0.0.1/",      "expected_episodes": 200_000},
    "language_table_blocktorelative_oracle_sim":      {"path": f"{GCS_BASE}/language_table_blocktorelative_oracle_sim/0.0.1/",      "expected_episodes": 200_000},
    "language_table_separate_oracle_sim":             {"path": f"{GCS_BASE}/language_table_separate_oracle_sim/0.0.1/",             "expected_episodes": 200_000},
}


def decode_instruction(codes):
    return "".join(chr(int(c)) for c in codes if c != 0).strip()


# ── Output helpers ──────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def ok(msg: str):   print(f"  [PASS] {msg}")
def fail(msg: str): print(f"  [FAIL] {msg}")
def warn(msg: str): print(f"  [WARN] {msg}")
def info(msg: str): print(f"  [INFO] {msg}")


# ── Metadata checks ─────────────────────────────────────────────────────────

def load_lerobot_metadata(root: Path):
    with (root / "meta/info.json").open() as f:
        info_json = json.load(f)
    tasks = {}
    with (root / "meta/tasks.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            tasks[e["task_index"]] = e["task"]
    episodes = []
    with (root / "meta/episodes.jsonl").open() as f:
        for line in f:
            episodes.append(json.loads(line))
    with (root / "meta/stats.json").open() as f:
        stats = json.load(f)
    return info_json, tasks, episodes, stats


def check_metadata_internal_consistency(info_json, tasks, episodes, stats):
    section("Metadata internal consistency")

    total_ep = info_json["total_episodes"]
    total_fr = info_json["total_frames"]

    if len(episodes) == total_ep:
        ok(f"episodes.jsonl has {total_ep} entries (matches info.total_episodes)")
    else:
        fail(f"episodes.jsonl has {len(episodes)} entries, "
             f"info.total_episodes = {total_ep}")

    sum_lengths = sum(e["length"] for e in episodes)
    if sum_lengths == total_fr:
        ok(f"sum of episode lengths = {total_fr} (matches info.total_frames)")
    else:
        fail(f"sum of episode lengths = {sum_lengths}, "
             f"info.total_frames = {total_fr}")

    if len(tasks) == info_json["total_tasks"]:
        ok(f"tasks.jsonl has {len(tasks)} entries (matches info.total_tasks)")
    else:
        fail(f"tasks.jsonl has {len(tasks)}, "
             f"info.total_tasks = {info_json['total_tasks']}")

    for col in ["observation.state", "observation.effector_target_translation",
                "action", "next.reward"]:
        if col in stats:
            n = stats[col]["count"]
            if n == total_fr:
                ok(f"stats['{col}'].count = {n} (matches total_frames)")
            else:
                fail(f"stats['{col}'].count = {n}, expected {total_fr}")
        else:
            fail(f"stats missing column '{col}'")


def check_episode_count_vs_rlds(info_json, dataset_name, args_limit):
    section("Episode count vs RLDS source")
    expected = DATASETS[dataset_name]["expected_episodes"]
    actual = info_json["total_episodes"]
    # If the user ran with --num_episodes limit, don't expect the full count.
    if args_limit is not None and args_limit < expected:
        if actual == args_limit:
            ok(f"converted {actual} episodes (matches --num_episodes limit)")
        else:
            fail(f"converted {actual} episodes, --num_episodes was {args_limit}")
    else:
        if actual == expected:
            ok(f"converted {actual:,} / {expected:,} episodes")
        else:
            fail(f"converted {actual:,}, expected {expected:,}")


# ── Per-episode exact comparison ────────────────────────────────────────────

def parquet_path(root: Path, ep: int) -> Path:
    return root / f"data/chunk-{ep // CHUNK_SIZE:03d}/episode_{ep:06d}.parquet"


def video_path(root: Path, ep: int) -> Path:
    return (root / f"videos/chunk-{ep // CHUNK_SIZE:03d}"
                 / "observation.images.rgb" / f"episode_{ep:06d}.mp4")


def decode_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok_, frame = cap.read()
        if not ok_:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def compare_episodes(rlds_builder, root: Path, tasks: dict, num_episodes: int,
                     skip_video: bool):
    """Exact per-episode comparison: RLDS[i] vs LeRobot[i]."""
    import tensorflow_datasets as tfds

    section(f"Per-episode exact comparison (first {num_episodes} episodes)")

    ds = rlds_builder.as_dataset(split="train").take(num_episodes)

    total_actions = 0
    fail_count = 0
    psnr_all = []

    for ep_idx, ep in enumerate(ds):
        # RLDS
        rlds_actions, rlds_states, rlds_targets = [], [], []
        rlds_rewards, rlds_dones = [], []
        rlds_frames, rlds_instr = [], None
        for step in ep["steps"]:
            if bool(step["is_terminal"].numpy()):
                continue
            obs = step["observation"]
            rlds_actions.append(step["action"].numpy())
            rlds_states.append(obs["effector_translation"].numpy())
            rlds_targets.append(obs["effector_target_translation"].numpy())
            rlds_rewards.append(float(step["reward"].numpy()))
            rlds_dones.append(bool(step["is_last"].numpy()))
            if rlds_instr is None:
                rlds_instr = decode_instruction(obs["instruction"].numpy())
            if not skip_video:
                rlds_frames.append(obs["rgb"].numpy())

        rlds_actions = np.stack(rlds_actions).astype(np.float32)
        rlds_states  = np.stack(rlds_states).astype(np.float32)
        rlds_targets = np.stack(rlds_targets).astype(np.float32)
        rlds_rewards = np.asarray(rlds_rewards, dtype=np.float32)
        rlds_dones   = np.asarray(rlds_dones, dtype=bool)

        # LeRobot parquet
        pq_path = parquet_path(root, ep_idx)
        if not pq_path.exists():
            fail(f"episode {ep_idx}: parquet missing at {pq_path}")
            fail_count += 1
            continue
        tbl = pq.read_table(pq_path)
        lr_actions = np.asarray(tbl["action"].to_pylist(), dtype=np.float32)
        lr_states  = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float32)
        lr_targets = np.asarray(tbl["observation.effector_target_translation"].to_pylist(), dtype=np.float32)
        lr_rewards = np.asarray(tbl["next.reward"].to_pylist(), dtype=np.float32)
        lr_dones   = np.asarray(tbl["next.done"].to_pylist(), dtype=bool)
        lr_frame_ix = np.asarray(tbl["frame_index"].to_pylist())
        lr_task_ix = int(tbl["task_index"].to_pylist()[0])
        lr_instr = tasks.get(lr_task_ix, "<missing>")

        n = len(rlds_actions)
        status_by_check = []

        if tbl.num_rows != n:
            fail(f"episode {ep_idx}: parquet rows {tbl.num_rows} vs "
                 f"RLDS steps {n}")
            fail_count += 1
            continue

        # Exact numeric equality.
        eq = [
            ("action",  np.array_equal(rlds_actions, lr_actions)),
            ("state",   np.array_equal(rlds_states,  lr_states)),
            ("target",  np.array_equal(rlds_targets, lr_targets)),
            ("reward",  np.array_equal(rlds_rewards, lr_rewards)),
            ("done",    np.array_equal(rlds_dones,   lr_dones)),
            ("frame_ix", np.array_equal(lr_frame_ix, np.arange(n))),
            ("task",    rlds_instr == lr_instr),
        ]
        bad = [name for name, good in eq if not good]
        total_actions += n
        if bad:
            fail(f"episode {ep_idx} (n={n}): mismatches in {bad}")
            fail_count += 1
            continue

        # Video PSNR.
        if not skip_video:
            vid_path = video_path(root, ep_idx)
            if not vid_path.exists():
                fail(f"episode {ep_idx}: mp4 missing at {vid_path}")
                fail_count += 1
                continue
            mp4_frames = decode_video(vid_path)
            if len(mp4_frames) != n:
                fail(f"episode {ep_idx}: mp4 {len(mp4_frames)} frames, "
                     f"parquet {n} rows")
                fail_count += 1
                continue
            ep_psnrs = []
            for a, b in zip(rlds_frames, mp4_frames):
                mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
                if mse > 0:
                    ep_psnrs.append(10 * np.log10(255.0 ** 2 / mse))
            if ep_psnrs:
                ep_mean = np.mean(ep_psnrs)
                psnr_all.extend(ep_psnrs)
                info(f"episode {ep_idx:4d}: n={n:3d} OK (video PSNR mean={ep_mean:.1f} dB)")
            else:
                info(f"episode {ep_idx:4d}: n={n:3d} OK (lossless video?)")
        else:
            info(f"episode {ep_idx:4d}: n={n:3d} OK")

    section("Summary")
    if fail_count == 0:
        ok(f"{num_episodes} episodes validated, {total_actions} steps, 0 mismatches")
    else:
        fail(f"{fail_count} / {num_episodes} episodes had mismatches")

    if psnr_all:
        psnr = np.array(psnr_all)
        info(f"Video PSNR over all compared frames: "
             f"mean={psnr.mean():.2f} dB, min={psnr.min():.2f} dB, "
             f"max={psnr.max():.2f} dB")
        if psnr.mean() > 30.0:
            ok("Video PSNR > 30 dB (visually lossless for training)")
        else:
            warn("Video PSNR below 30 dB — check codec settings")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", required=True,
                        choices=sorted(DATASETS.keys()))
    parser.add_argument("--output_dir", required=True,
                        help="Parent directory that contains {dataset_name}/")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes for exact comparison (default 20).")
    parser.add_argument("--skip_video", action="store_true",
                        help="Skip MP4 decode + PSNR check.")
    parser.add_argument("--original_limit", type=int, default=None,
                        help="If the dataset was converted with --num_episodes N, "
                             "pass the same N here so expected counts check out.")
    args = parser.parse_args()

    root = Path(args.output_dir) / args.dataset_name
    if not root.exists():
        print(f"No such dataset at {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating {root}")

    info_json, tasks, episodes, stats = load_lerobot_metadata(root)

    check_metadata_internal_consistency(info_json, tasks, episodes, stats)
    check_episode_count_vs_rlds(info_json, args.dataset_name, args.original_limit)

    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(
        builder_dir=DATASETS[args.dataset_name]["path"])

    n_to_compare = min(args.num_episodes, info_json["total_episodes"])
    compare_episodes(builder, root, tasks, n_to_compare, args.skip_video)


if __name__ == "__main__":
    main()
