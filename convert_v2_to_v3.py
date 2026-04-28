"""Convert a Language Table LeRobot v2.0 dataset to v3.0 format.

Reads the existing per-episode parquets and MP4s from a v2.0 dataset and
produces the v3.0 layout:

  v2.0:  data/chunk-NNN/episode_NNNNNN.parquet   (one per episode)
         videos/chunk-NNN/observation.images.rgb/episode_NNNNNN.mp4

  v3.0:  data/chunk-NNN/file-NNN.parquet          (many episodes per file)
         videos/observation.images.rgb/chunk-NNN/file-NNN.mp4
         meta/tasks.parquet
         meta/episodes/chunk-NNN/file-NNN.parquet  (rich episode metadata)

Video files are concatenated with `ffmpeg -c copy` (lossless stream copy,
no re-encoding).  Parquets are read and re-written with pyarrow.

Usage:
    source ltvenv/bin/activate
    python convert_v2_to_v3.py \
        --input_dir /media/mateo/Storage/lerobot_datasets \
        --output_dir /media/mateo/Storage/lerobot_datasets_v3 \
        --dataset_name language_table_blocktoblock_sim
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ── Constants ──────────────────────────────────────────────────────────────

CHUNKS_SIZE = 1000           # max files per chunk directory
DATA_FILE_SIZE_MB = 100      # target max data parquet size
VIDEO_FILE_SIZE_MB = 200     # target max concatenated video size
FPS = 10
VIDEO_KEY = "observation.images.rgb"


# ── v2.0 path helpers ─────────────────────────────────────────────────────

def v2_parquet(root: Path, ep: int) -> Path:
    return root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"


def v2_video(root: Path, ep: int) -> Path:
    return (root / f"videos/chunk-{ep // 1000:03d}"
                 / VIDEO_KEY / f"episode_{ep:06d}.mp4")


# ── v3.0 path helpers ─────────────────────────────────────────────────────

def v3_data_path(root: Path, chunk: int, file: int) -> Path:
    return root / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"


def v3_video_path(root: Path, chunk: int, file: int) -> Path:
    return root / f"videos/{VIDEO_KEY}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def v3_episodes_path(root: Path, chunk: int, file: int) -> Path:
    return root / f"meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet"


def next_chunk_file(chunk: int, file: int) -> tuple[int, int]:
    """Increment file index, rolling to next chunk if needed."""
    file += 1
    if file >= CHUNKS_SIZE:
        chunk += 1
        file = 0
    return chunk, file


# ── Per-episode stats ──────────────────────────────────────────────────────

STAT_FEATURES_2D = ["observation.state", "observation.effector_target_translation",
                    "action"]
STAT_FEATURES_1D = ["next.reward"]
ALL_STAT_FEATURES = STAT_FEATURES_2D + STAT_FEATURES_1D


def compute_episode_stats(tbl: pa.Table) -> dict:
    """Compute per-episode min/max/mean/std/count for stat features."""
    stats = {}
    for col in STAT_FEATURES_2D:
        arr = np.asarray(tbl.column(col).to_pylist(), dtype=np.float64)
        n = arr.shape[0]
        stats[col] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [n, n],
        }
    for col in STAT_FEATURES_1D:
        arr = np.asarray(tbl.column(col).to_pylist(), dtype=np.float64).reshape(-1)
        n = arr.shape[0]
        stats[col] = {
            "min": [float(arr.min())],
            "max": [float(arr.max())],
            "mean": [float(arr.mean())],
            "std": [float(arr.std())],
            "count": [n],
        }
    return stats


# ── Video duration helper ─────────────────────────────────────────────────

def get_video_duration(path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_video_frame_count(path: Path) -> int:
    """Get frame count via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-count_frames",
           "-select_streams", "v:0",
           "-show_entries", "stream=nb_read_frames",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


# ── Main conversion ───────────────────────────────────────────────────────

def convert_dataset(input_root: Path, output_root: Path):
    """Convert one v2.0 dataset to v3.0."""

    # Load v2.0 metadata.
    with (input_root / "meta/info.json").open() as f:
        v2_info = json.load(f)
    total_episodes = v2_info["total_episodes"]
    total_frames = v2_info["total_frames"]

    # Load tasks.
    tasks = {}
    with (input_root / "meta/tasks.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            tasks[e["task_index"]] = e["task"]

    # Load episodes metadata (for task lists and lengths).
    episodes_meta = []
    with (input_root / "meta/episodes.jsonl").open() as f:
        for line in f:
            episodes_meta.append(json.loads(line))

    print(f"  Input: {total_episodes} episodes, {total_frames} frames, "
          f"{len(tasks)} tasks")

    output_root.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Convert data parquets ─────────────────────────────────
    print("\n  [Phase 1] Converting data parquets...")
    t0 = time.time()

    data_chunk, data_file = 0, 0
    data_accum = []          # list of pa.Table
    data_accum_bytes = 0
    global_idx = 0           # running global frame index

    # Per-episode tracking for episodes metadata.
    ep_records = []          # one dict per episode
    ep_stats = []            # one dict per episode

    for ep_idx in range(total_episodes):
        pq_path = v2_parquet(input_root, ep_idx)
        tbl = pq.read_table(pq_path)
        n = tbl.num_rows

        # Rewrite global `index` column to be contiguous.
        new_index = pa.array(range(global_idx, global_idx + n), type=pa.int64())
        col_idx = tbl.schema.get_field_index("index")
        tbl = tbl.set_column(col_idx, "index", new_index)

        # Track episode → data shard mapping.
        ep_records.append({
            "episode_index": ep_idx,
            "data_chunk": data_chunk,
            "data_file": data_file,
            "dataset_from": global_idx,
            "dataset_to": global_idx + n,
            "length": n,
            "tasks": episodes_meta[ep_idx].get("tasks", []),
        })

        # Compute per-episode stats.
        ep_stats.append(compute_episode_stats(tbl))

        data_accum.append(tbl)
        data_accum_bytes += pq_path.stat().st_size
        global_idx += n

        # Flush if over size threshold.
        if data_accum_bytes >= DATA_FILE_SIZE_MB * 1024 * 1024:
            out = v3_data_path(output_root, data_chunk, data_file)
            out.parent.mkdir(parents=True, exist_ok=True)
            merged = pa.concat_tables(data_accum)
            pq.write_table(merged, out, compression="snappy")
            data_accum = []
            data_accum_bytes = 0
            data_chunk, data_file = next_chunk_file(data_chunk, data_file)

        if (ep_idx + 1) % 10000 == 0:
            print(f"    {ep_idx + 1} / {total_episodes} episodes read "
                  f"({time.time() - t0:.1f}s)")

    # Flush remainder.
    if data_accum:
        out = v3_data_path(output_root, data_chunk, data_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged = pa.concat_tables(data_accum)
        pq.write_table(merged, out, compression="snappy")
        data_accum = []

    print(f"  [Phase 1] Done in {time.time() - t0:.1f}s")

    # ── Phase 2: Concatenate videos ────────────────────────────────────
    print("\n  [Phase 2] Concatenating videos (ffmpeg -c copy)...")
    t0 = time.time()

    vid_chunk, vid_file = 0, 0
    vid_accum_paths = []     # paths to accumulate
    vid_accum_bytes = 0
    vid_running_time = 0.0   # running timestamp within current output file

    for ep_idx in range(total_episodes):
        src = v2_video(input_root, ep_idx)
        src_size = src.stat().st_size
        n_frames = ep_records[ep_idx]["length"]
        ep_duration = n_frames / FPS

        ep_records[ep_idx]["vid_chunk"] = vid_chunk
        ep_records[ep_idx]["vid_file"] = vid_file
        ep_records[ep_idx]["vid_from_ts"] = vid_running_time
        ep_records[ep_idx]["vid_to_ts"] = vid_running_time + ep_duration

        vid_accum_paths.append(src)
        vid_accum_bytes += src_size
        vid_running_time += ep_duration

        # Flush if over size threshold.
        if vid_accum_bytes >= VIDEO_FILE_SIZE_MB * 1024 * 1024:
            _flush_video(output_root, vid_chunk, vid_file, vid_accum_paths)
            vid_accum_paths = []
            vid_accum_bytes = 0
            vid_running_time = 0.0
            vid_chunk, vid_file = next_chunk_file(vid_chunk, vid_file)

        if (ep_idx + 1) % 10000 == 0:
            print(f"    {ep_idx + 1} / {total_episodes} episodes queued "
                  f"({time.time() - t0:.1f}s)")

    # Flush remainder.
    if vid_accum_paths:
        _flush_video(output_root, vid_chunk, vid_file, vid_accum_paths)

    print(f"  [Phase 2] Done in {time.time() - t0:.1f}s")

    # ── Phase 3: Write metadata ────────────────────────────────────────
    print("\n  [Phase 3] Writing v3.0 metadata...")
    t0 = time.time()

    # tasks.parquet
    _write_tasks_parquet(output_root, tasks)

    # episodes metadata parquet
    _write_episodes_parquet(output_root, ep_records, ep_stats)

    # stats.json (copy from v2.0 — already correct)
    shutil.copy2(input_root / "meta/stats.json", output_root / "meta/stats.json")

    # info.json
    _write_info_json(output_root, v2_info, total_episodes, total_frames, len(tasks))

    print(f"  [Phase 3] Done in {time.time() - t0:.1f}s")


def _flush_video(root: Path, chunk: int, file: int, paths: list[Path]):
    """Concatenate a batch of mp4s into one output file via ffmpeg."""
    out = v3_video_path(root, chunk, file)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write concat list to a temp file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in paths:
            f.write(f"file '{p}'\n")
        list_path = f.name

    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", list_path,
               "-c", "copy", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
    finally:
        os.unlink(list_path)


def _write_tasks_parquet(root: Path, tasks: dict[int, str]):
    """Write meta/tasks.parquet with task string as index."""
    import pandas as pd

    df = pd.DataFrame([
        {"task_index": idx, "task": text}
        for idx, text in sorted(tasks.items())
    ])
    df = df.set_index("task")

    path = root / "meta/tasks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _write_episodes_parquet(root: Path, ep_records: list[dict],
                            ep_stats: list[dict]):
    """Write meta/episodes/chunk-000/file-000.parquet."""

    # Build columns.
    episode_index = []
    data_chunk_index = []
    data_file_index = []
    dataset_from_index = []
    dataset_to_index = []
    tasks_col = []
    length_col = []
    ep_meta_chunk = []
    ep_meta_file = []
    vid_chunk_col = []
    vid_file_col = []
    vid_from_ts_col = []
    vid_to_ts_col = []

    # Per-feature stat columns.
    stat_cols = {}
    for feat in ALL_STAT_FEATURES:
        for s in ["min", "max", "mean", "std", "count"]:
            stat_cols[f"stats/{feat}/{s}"] = []

    for rec in ep_records:
        episode_index.append(rec["episode_index"])
        data_chunk_index.append(rec["data_chunk"])
        data_file_index.append(rec["data_file"])
        dataset_from_index.append(rec["dataset_from"])
        dataset_to_index.append(rec["dataset_to"])
        tasks_col.append(rec["tasks"])
        length_col.append(rec["length"])
        ep_meta_chunk.append(0)
        ep_meta_file.append(0)
        vid_chunk_col.append(rec["vid_chunk"])
        vid_file_col.append(rec["vid_file"])
        vid_from_ts_col.append(rec["vid_from_ts"])
        vid_to_ts_col.append(rec["vid_to_ts"])

    for i, stats in enumerate(ep_stats):
        for feat in ALL_STAT_FEATURES:
            for s in ["min", "max", "mean", "std", "count"]:
                stat_cols[f"stats/{feat}/{s}"].append(stats[feat][s])

    # Build arrow table.
    columns = {
        "episode_index": pa.array(episode_index, type=pa.int64()),
        "data/chunk_index": pa.array(data_chunk_index, type=pa.int64()),
        "data/file_index": pa.array(data_file_index, type=pa.int64()),
        "dataset_from_index": pa.array(dataset_from_index, type=pa.int64()),
        "dataset_to_index": pa.array(dataset_to_index, type=pa.int64()),
        "tasks": pa.array(tasks_col, type=pa.list_(pa.string())),
        "length": pa.array(length_col, type=pa.int64()),
        f"meta/episodes/chunk_index": pa.array(ep_meta_chunk, type=pa.int64()),
        f"meta/episodes/file_index": pa.array(ep_meta_file, type=pa.int64()),
        f"videos/{VIDEO_KEY}/chunk_index": pa.array(vid_chunk_col, type=pa.int64()),
        f"videos/{VIDEO_KEY}/file_index": pa.array(vid_file_col, type=pa.int64()),
        f"videos/{VIDEO_KEY}/from_timestamp": pa.array(vid_from_ts_col, type=pa.float64()),
        f"videos/{VIDEO_KEY}/to_timestamp": pa.array(vid_to_ts_col, type=pa.float64()),
    }

    # Add stat columns.
    for key, vals in stat_cols.items():
        if "count" in key:
            columns[key] = pa.array(vals, type=pa.list_(pa.int64()))
        else:
            columns[key] = pa.array(vals, type=pa.list_(pa.float64()))

    tbl = pa.table(columns)

    path = v3_episodes_path(root, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path)


def _write_info_json(root: Path, v2_info: dict,
                     total_episodes: int, total_frames: int,
                     total_tasks: int):
    """Write v3.0 info.json."""

    # Start from v2.0 features, add fps to non-video features.
    features = {}
    for name, feat in v2_info["features"].items():
        feat = dict(feat)
        if feat["dtype"] == "video":
            # Restructure: "info" → "video_info"
            if "info" in feat:
                feat["video_info"] = feat.pop("info")
        else:
            # Add fps field for non-video features.
            feat["fps"] = FPS
        features[name] = feat

    info = {
        "codebase_version": "v3.0",
        "robot_type": v2_info.get("robot_type", "unknown"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": DATA_FILE_SIZE_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
    }

    path = root / "meta/info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(info, f, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────────

DATASETS = [
    "language_table",
    "language_table_sim",
    "language_table_blocktoblock_sim",
    "language_table_blocktoblock_4block_sim",
    "language_table_blocktoblock_oracle_sim",
    "language_table_blocktoblockrelative_oracle_sim",
    "language_table_blocktoabsolute_oracle_sim",
    "language_table_blocktorelative_oracle_sim",
    "language_table_separate_oracle_sim",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True,
                        help="Parent dir containing v2.0 datasets.")
    parser.add_argument("--output_dir", required=True,
                        help="Parent dir for v3.0 output.")
    parser.add_argument("--dataset_name", required=True,
                        choices=sorted(DATASETS),
                        help="Which dataset to convert.")
    args = parser.parse_args()

    input_root = Path(args.input_dir) / args.dataset_name
    output_root = Path(args.output_dir) / args.dataset_name

    if not input_root.exists():
        print(f"ERROR: {input_root} does not exist")
        return 1

    print(f"Converting {args.dataset_name}")
    print(f"  v2.0 input:  {input_root}")
    print(f"  v3.0 output: {output_root}")

    t_total = time.time()
    convert_dataset(input_root, output_root)

    print(f"\n[DONE] {args.dataset_name} converted to v3.0 "
          f"in {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
