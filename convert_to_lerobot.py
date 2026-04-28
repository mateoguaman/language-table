"""Convert a Language Table RLDS dataset to LeRobot v2.0 format.

Streams a Language Table RLDS dataset from GCS and writes the LeRobot v2.0
layout:

    {output_dir}/{dataset_name}/
    +-- meta/
    |   +-- info.json
    |   +-- tasks.jsonl
    |   +-- episodes.jsonl
    |   +-- stats.json
    +-- data/
    |   +-- chunk-{NNN}/
    |       +-- episode_{NNNNNN}.parquet
    +-- videos/
        +-- chunk-{NNN}/
            +-- observation.images.rgb/
                +-- episode_{NNNNNN}.mp4

Preserves every useful field from the RLDS source: [x, y] action, [x, y]
effector state, [x, y] effector target, per-step reward, per-step done, and
the natural-language instruction (via task_index / tasks.jsonl).

Usage:
    source ltvenv/bin/activate
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python convert_to_lerobot.py \
        --dataset_name language_table_blocktoblock_sim \
        --output_dir ./lerobot_datasets \
        [--num_episodes N]
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Quiet TF noise before the tf import inside tfds.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ── Dataset registry ────────────────────────────────────────────────────────

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

FPS = 10.0
IMG_H, IMG_W = 360, 640
CHUNK_SIZE = 1000  # episodes per data/videos chunk directory

FEATURES = {
    "observation.images.rgb": {
        "dtype": "video",
        "shape": [IMG_H, IMG_W, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": FPS,
            "video.height": IMG_H,
            "video.width": IMG_W,
            "video.channels": 3,
            "video.codec": "libx264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32", "shape": [2], "names": {"motors": ["x", "y"]},
    },
    "observation.effector_target_translation": {
        "dtype": "float32", "shape": [2], "names": {"motors": ["x", "y"]},
    },
    "action": {
        "dtype": "float32", "shape": [2], "names": {"motors": ["x", "y"]},
    },
    "next.reward": {"dtype": "float32", "shape": [1], "names": None},
    "next.done":   {"dtype": "bool",    "shape": [1], "names": None},
    "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
    "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
    "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
    "index":         {"dtype": "int64",   "shape": [1], "names": None},
    "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
}


# ── Path helpers ────────────────────────────────────────────────────────────

def chunk_for_ep(ep: int) -> int:
    return ep // CHUNK_SIZE


def parquet_path(root: Path, ep: int) -> Path:
    return root / f"data/chunk-{chunk_for_ep(ep):03d}/episode_{ep:06d}.parquet"


def video_path(root: Path, ep: int) -> Path:
    return (root / f"videos/chunk-{chunk_for_ep(ep):03d}"
                 / "observation.images.rgb" / f"episode_{ep:06d}.mp4")


# ── Decoding helpers ────────────────────────────────────────────────────────

def decode_instruction(codes: np.ndarray) -> str:
    """RLDS stores the instruction as int32 Unicode code points, zero-padded.

    Matches the canonical decode used in
    language_table/train/input_pipeline_rlds.py::_tokenize_instruction.
    """
    chars = [chr(int(c)) for c in codes if c != 0]
    return "".join(chars).strip()


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG to (H, W, 3) uint8 RGB."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None for a JPEG frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Video encoding ─────────────────────────────────────────────────────────
#
# We pipe raw RGB frames into the system ffmpeg binary and let libx264 do the
# H.264 encoding. cv2.VideoWriter on our Ubuntu box links against an ffmpeg
# build that lacks libx264, so it silently falls back to MPEG-4 Part 2 which
# has much worse compression. Using ffmpeg(1) directly matches what LeRobot
# itself does for video encoding and gives the libx264 encoder we actually want.

FFMPEG_BIN = shutil.which("ffmpeg")


def encode_video_ffmpeg(frames_rgb_uint8: list[np.ndarray], out_path: Path,
                         codec: str) -> str:
    """Encode a list of (H, W, 3) uint8 RGB frames to MP4 via ffmpeg.

    Returns the codec name that was actually used (for stats/info metadata).
    """
    if FFMPEG_BIN is None:
        raise RuntimeError("System ffmpeg not found; install it or use "
                           "--video_codec mp4v with the OpenCV fallback.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map our 'codec' arg to an ffmpeg -c:v value.
    codec_map = {
        "avc1": "libx264", "h264": "libx264", "libx264": "libx264",
        "hevc": "libx265", "libx265": "libx265",
        "mp4v": "mpeg4",
    }
    v_codec = codec_map.get(codec, codec)

    cmd = [
        FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{IMG_W}x{IMG_H}",
        "-r", str(FPS),
        "-i", "-",  # stdin
        "-c:v", v_codec,
        "-pix_fmt", "yuv420p",
    ]
    # Reasonable defaults for libx264/265 at 640x360@10fps. CRF 23 is the
    # ffmpeg default (visually lossless-ish at moderate bitrate).
    if v_codec in ("libx264", "libx265"):
        cmd += ["-preset", "fast", "-crf", "23"]
    cmd += [str(out_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for frame in frames_rgb_uint8:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if frame.shape != (IMG_H, IMG_W, 3):
                raise ValueError(f"frame shape {frame.shape} != "
                                 f"({IMG_H}, {IMG_W}, 3)")
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        # ffmpeg died early; fall through to read stderr below.
        pass
    err = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        err_text = err.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}) for {out_path}:\n{err_text}")

    return v_codec


# ── Phase 1: task vocabulary ────────────────────────────────────────────────

def phase1_build_task_vocab(builder, out_root: Path,
                             num_episodes: Optional[int]) -> dict[str, int]:
    """Stream once, collect unique first-step instructions, write tasks.jsonl."""
    import tensorflow_datasets as tfds

    print("\n[Phase 1] Building task vocabulary...")
    t0 = time.time()

    ds = builder.as_dataset(
        split="train",
        decoders={"steps": {"observation": {"rgb": tfds.decode.SkipDecoding()}}},
    )
    if num_episodes is not None:
        ds = ds.take(num_episodes)

    task_to_idx: dict[str, int] = {}
    count = 0
    for episode in ds:
        first = next(iter(episode["steps"].take(1)))
        instr = decode_instruction(first["observation"]["instruction"].numpy())
        if instr not in task_to_idx:
            task_to_idx[instr] = len(task_to_idx)
        count += 1
        if count % 10000 == 0:
            print(f"  {count} episodes scanned, {len(task_to_idx)} unique tasks "
                  f"({time.time() - t0:.1f}s)")

    tasks_path = out_root / "meta/tasks.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w") as f:
        for instr, idx in sorted(task_to_idx.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": instr}) + "\n")

    print(f"[Phase 1] Done: {count} episodes, {len(task_to_idx)} unique tasks "
          f"in {time.time() - t0:.1f}s. Wrote {tasks_path}")
    return task_to_idx


# ── Phase 2: convert episodes ────────────────────────────────────────────────

def phase2_convert_episodes(builder, out_root: Path,
                             task_to_idx: dict[str, int],
                             num_episodes: Optional[int],
                             codec: str):
    """Write one parquet + one MP4 per episode; write episodes.jsonl."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tensorflow_datasets as tfds

    print(f"\n[Phase 2] Converting episodes (codec={codec})...")
    t0 = time.time()

    ds = builder.as_dataset(
        split="train",
        decoders={"steps": {"observation": {"rgb": tfds.decode.SkipDecoding()}}},
    )
    if num_episodes is not None:
        ds = ds.take(num_episodes)

    # We rebuild episodes.jsonl from scratch so it stays consistent with the
    # parquets that actually exist on disk.
    episodes_path = out_root / "meta/episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    ep_file = episodes_path.open("w")

    global_index = 0
    total_frames = 0
    ep_count = 0
    written = 0
    resumed = 0
    actual_codec = codec

    for ep_idx, episode in enumerate(ds):
        ep_count += 1
        pq_path = parquet_path(out_root, ep_idx)
        vid_path = video_path(out_root, ep_idx)

        # Fast-path resume: both artefacts on disk and non-empty.
        if (pq_path.exists() and vid_path.exists()
                and pq_path.stat().st_size > 0 and vid_path.stat().st_size > 0):
            # Read just frame_index + task_index to update counters and episodes.jsonl.
            existing = pq.read_table(pq_path, columns=["frame_index", "task_index"])
            n = existing.num_rows
            task_ids = set(existing.column("task_index").to_pylist())
            # Reverse-lookup instructions for the episodes.jsonl entry.
            idx_to_task = {v: k for k, v in task_to_idx.items()}
            tasks = sorted({idx_to_task.get(i, "<unknown>") for i in task_ids})
            ep_file.write(json.dumps({
                "episode_index": ep_idx,
                "tasks": tasks,
                "length": n,
            }) + "\n")
            global_index += n
            total_frames += n
            resumed += 1
            continue

        # Stream steps for this episode.
        states, targets, actions, rewards, dones = [], [], [], [], []
        task_ids_for_ep: list[int] = []
        unique_instructions: set[str] = set()
        frames: list[np.ndarray] = []

        for step in episode["steps"]:
            if bool(step["is_terminal"].numpy()):
                continue  # Matches input_pipeline_rlds._is_not_terminal.

            obs = step["observation"]
            states.append(obs["effector_translation"].numpy().astype(np.float32))
            targets.append(obs["effector_target_translation"].numpy().astype(np.float32))
            actions.append(step["action"].numpy().astype(np.float32))
            rewards.append(np.float32(step["reward"].numpy()))
            dones.append(bool(step["is_last"].numpy()))

            instr = decode_instruction(obs["instruction"].numpy())
            unique_instructions.add(instr)
            tid = task_to_idx.get(instr)
            if tid is None:
                # Phase 1 missed it (e.g., mid-episode instruction change).
                tid = len(task_to_idx)
                task_to_idx[instr] = tid
            task_ids_for_ep.append(tid)

            jpeg = obs["rgb"]
            if hasattr(jpeg, "numpy"):
                jpeg = jpeg.numpy()
            frames.append(decode_jpeg(jpeg))

        if not frames:
            print(f"  [WARN] episode {ep_idx} had 0 non-terminal steps, skipping")
            continue

        n = len(frames)
        actions_np = np.stack(actions)
        states_np = np.stack(states)
        targets_np = np.stack(targets)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=bool)

        frame_indices = np.arange(n, dtype=np.int64)
        timestamps = (frame_indices.astype(np.float32) / np.float32(FPS))
        global_indices = np.arange(global_index, global_index + n, dtype=np.int64)
        episode_indices = np.full(n, ep_idx, dtype=np.int64)
        task_ids_np = np.asarray(task_ids_for_ep, dtype=np.int64)

        # Write parquet.
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            "observation.state": pa.array(
                list(states_np), type=pa.list_(pa.float32(), 2)),
            "observation.effector_target_translation": pa.array(
                list(targets_np), type=pa.list_(pa.float32(), 2)),
            "action": pa.array(
                list(actions_np), type=pa.list_(pa.float32(), 2)),
            "next.reward":   pa.array(rewards_np, type=pa.float32()),
            "next.done":     pa.array(dones_np,   type=pa.bool_()),
            "timestamp":     pa.array(timestamps, type=pa.float32()),
            "frame_index":   pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index":         pa.array(global_indices, type=pa.int64()),
            "task_index":    pa.array(task_ids_np, type=pa.int64()),
        })
        pq.write_table(table, pq_path, compression="snappy")

        # Write video.
        actual_codec = encode_video_ffmpeg(frames, vid_path, codec)

        # Update episodes.jsonl.
        ep_file.write(json.dumps({
            "episode_index": ep_idx,
            "tasks": sorted(unique_instructions),
            "length": n,
        }) + "\n")

        global_index += n
        total_frames += n
        written += 1

        if ep_count % 100 == 0:
            elapsed = time.time() - t0
            rate = written / max(elapsed, 1e-6)
            print(f"  {ep_count} episodes seen, {written} written, "
                  f"{resumed} resumed, {total_frames} frames, "
                  f"{rate:.1f} ep/s, elapsed {elapsed:.1f}s")

    ep_file.close()
    print(f"[Phase 2] Done: {ep_count} episodes seen, {written} written, "
          f"{resumed} resumed, {total_frames} frames "
          f"in {time.time() - t0:.1f}s")

    return {
        "total_episodes": ep_count,
        "total_frames": total_frames,
        "task_to_idx": task_to_idx,
        "codec": actual_codec,
    }


# ── Phase 3: stats + info.json ──────────────────────────────────────────────

def phase3_write_metadata(out_root: Path, result: dict,
                           dataset_name: str, codec: str):
    """Compute stats by reading back all parquets; write meta/info + stats."""
    import pyarrow.parquet as pq

    print("\n[Phase 3] Computing stats and writing metadata...")
    t0 = time.time()

    total_episodes = result["total_episodes"]
    total_frames = result["total_frames"]
    total_chunks = (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_tasks = len(result["task_to_idx"])

    # Rewrite tasks.jsonl in case Phase 2 added any late entries.
    tasks_path = out_root / "meta/tasks.jsonl"
    with tasks_path.open("w") as f:
        for instr, idx in sorted(result["task_to_idx"].items(),
                                  key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": instr}) + "\n")

    # Compute stats from parquets. 2D columns are list<float32>[2]; reward is
    # float32 scalar. We stream in batches to keep memory bounded.
    stat_cols_2d = ["observation.state",
                    "observation.effector_target_translation",
                    "action"]
    stat_cols_scalar = ["next.reward"]

    # Welford accumulators.
    acc = {c: {"n": 0, "mean": np.zeros(2, np.float64), "m2": np.zeros(2, np.float64),
               "min": np.full(2, np.inf), "max": np.full(2, -np.inf)}
           for c in stat_cols_2d}
    acc.update({c: {"n": 0, "mean": np.zeros(1, np.float64),
                     "m2": np.zeros(1, np.float64),
                     "min": np.full(1, np.inf), "max": np.full(1, -np.inf)}
                for c in stat_cols_scalar})

    def update(a, x):
        """x: (batch_n, dim) float64."""
        bn = x.shape[0]
        if bn == 0:
            return
        bmean = x.mean(axis=0)
        bm2 = ((x - bmean) ** 2).sum(axis=0)
        delta = bmean - a["mean"]
        new_n = a["n"] + bn
        a["mean"] = a["mean"] + delta * (bn / new_n)
        a["m2"] = a["m2"] + bm2 + (delta ** 2) * (a["n"] * bn / new_n)
        a["n"] = new_n
        a["min"] = np.minimum(a["min"], x.min(axis=0))
        a["max"] = np.maximum(a["max"], x.max(axis=0))

    ep_scanned = 0
    for ep_idx in range(total_episodes):
        pq_path = parquet_path(out_root, ep_idx)
        if not pq_path.exists():
            continue
        ep_scanned += 1
        cols = stat_cols_2d + stat_cols_scalar
        tbl = pq.read_table(pq_path, columns=cols)
        for c in stat_cols_2d:
            arr = np.asarray(tbl.column(c).to_pylist(), dtype=np.float64)
            update(acc[c], arr)
        for c in stat_cols_scalar:
            arr = np.asarray(tbl.column(c).to_pylist(),
                             dtype=np.float64).reshape(-1, 1)
            update(acc[c], arr)

        if ep_scanned % 1000 == 0:
            print(f"  stats: scanned {ep_scanned} parquets "
                  f"({time.time() - t0:.1f}s)")

    stats_out = {}
    for c, a in acc.items():
        n = max(a["n"], 1)
        std = np.sqrt(a["m2"] / n)
        stats_out[c] = {
            "mean": a["mean"].tolist(),
            "std": std.tolist(),
            "min": a["min"].tolist(),
            "max": a["max"].tolist(),
            "count": int(a["n"]),
        }

    with (out_root / "meta/stats.json").open("w") as f:
        json.dump(stats_out, f, indent=2)

    # info.json
    features = json.loads(json.dumps(FEATURES))  # deep copy
    features["observation.images.rgb"]["info"]["video.codec"] = codec

    info_dict = {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "robot_type": "xarm",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "features": features,
        "source": {
            "name": dataset_name,
            "format": "rlds",
            "gcs_path": DATASETS[dataset_name]["path"],
        },
    }
    with (out_root / "meta/info.json").open("w") as f:
        json.dump(info_dict, f, indent=2)

    print(f"[Phase 3] Done: {total_episodes} episodes, {total_frames} frames, "
          f"{total_tasks} tasks, {total_chunks} chunks, "
          f"stats from {ep_scanned} parquets in {time.time() - t0:.1f}s")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_name", required=True,
                        choices=sorted(DATASETS.keys()))
    parser.add_argument("--output_dir", required=True,
                        help="Parent dir; subdir {dataset_name}/ is created inside.")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Limit to first N episodes (for testing).")
    parser.add_argument("--video_codec", default="libx264",
                        help="ffmpeg video codec. Default 'libx264' (H.264). "
                             "Also valid: 'libx265', 'mpeg4'.")
    parser.add_argument("--skip_phase1", action="store_true",
                        help="Reuse existing meta/tasks.jsonl instead of rebuilding.")
    args = parser.parse_args()

    import tensorflow_datasets as tfds

    info = DATASETS[args.dataset_name]
    out_root = Path(args.output_dir) / args.dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Source  : {info['path']}")
    print(f"Output  : {out_root}")
    print(f"Expected: {info['expected_episodes']:,} episodes")
    if args.num_episodes:
        print(f"Limit   : first {args.num_episodes} episodes")

    builder = tfds.builder_from_directory(builder_dir=info["path"])

    if args.skip_phase1 and (out_root / "meta/tasks.jsonl").exists():
        task_to_idx: dict[str, int] = {}
        with (out_root / "meta/tasks.jsonl").open() as f:
            for line in f:
                entry = json.loads(line)
                task_to_idx[entry["task"]] = entry["task_index"]
        print(f"[Phase 1] Loaded {len(task_to_idx)} tasks from existing tasks.jsonl")
    else:
        task_to_idx = phase1_build_task_vocab(builder, out_root, args.num_episodes)

    result = phase2_convert_episodes(builder, out_root, task_to_idx,
                                     args.num_episodes, args.video_codec)

    phase3_write_metadata(out_root, result, args.dataset_name,
                          result["codec"])

    print(f"\n[DONE] Conversion complete for {args.dataset_name}")


if __name__ == "__main__":
    main()
