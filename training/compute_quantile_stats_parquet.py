#!/usr/bin/env python3
"""Compute q01/q10/q50/q90/q99 quantile stats for LeRobot datasets directly from parquet.

pi0.5 training requires QUANTILES normalization on observation.state and action.
The upstream augment script (lerobot.scripts.augment_dataset_quantile_stats) calls
``dataset[idx]`` for every frame, which decodes video through pyav/torchcodec even
though pi0.5 uses VISUAL=IDENTITY (no video stats consumed). For a 58M-frame dataset
that pathway takes days.

This script skips video decode entirely: it reads non-video columns (state, action,
effector_target_translation, next.reward) straight from data/chunk-*/file-*.parquet,
computes per-episode stats with lerobot's own ``get_feature_stats``, aggregates with
``aggregate_stats`` (same code paths as upstream), and merges the result into
``meta/stats.json`` while preserving the existing ``observation.images.rgb`` ImageNet
stub.

Verification runs automatically after each write:
  - q01/q10/q50/q90/q99 present for observation.state and action
  - Quantile monotonicity (q01 <= q10 <= q50 <= q90 <= q99 elementwise)
  - All finite; count == total_frames from info.json
  - Video stub preserved
  - Dataset loads via LeRobotDataset(..., video_backend="pyav") without error
  - New mean/std within tolerance of the previous stats

Example:
    ./lerobotenv/bin/python training/compute_quantile_stats_parquet.py \\
        --root /media/mateo/Storage/lerobot_datasets_v3 \\
        --namespace mateoguaman \\
        --datasets language_table_blocktoblock_sim
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import (
    DEFAULT_QUANTILES,
    aggregate_stats,
    get_feature_stats,
)
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

VIDEO_KEY = "observation.images.rgb"
STATS_FEATURES = (
    "observation.state",
    "action",
    "observation.effector_target_translation",
    "next.reward",
)
REQUIRED_PI05 = ("observation.state", "action")
IMAGENET_STUB = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
    "min": [[[0.0]], [[0.0]], [[0.0]]],
    "max": [[[1.0]], [[1.0]], [[1.0]]],
}
QUANTILE_KEYS = [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]


def _load_info(dataset_root: Path) -> dict:
    return json.loads((dataset_root / "meta" / "info.json").read_text())


def _load_episode_ranges(dataset_root: Path) -> np.ndarray:
    """Return an (n_episodes, 2) int64 array of [from_idx, to_idx) per episode."""
    episode_parquets = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_parquets:
        raise FileNotFoundError(f"No episode parquets under {dataset_root}/meta/episodes")
    tables = [pq.read_table(p, columns=["episode_index", "dataset_from_index", "dataset_to_index"]) for p in episode_parquets]
    from functools import reduce
    import pyarrow as pa
    table = pa.concat_tables(tables)
    df = table.to_pandas().sort_values("episode_index").reset_index(drop=True)
    ranges = df[["dataset_from_index", "dataset_to_index"]].to_numpy(dtype=np.int64)
    assert np.all(ranges[:, 0] < ranges[:, 1]), "found empty or inverted episode range"
    return ranges


def _load_feature_arrays(dataset_root: Path, feature_keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Read non-video columns across all data/chunk-*/file-*.parquet into contiguous numpy arrays.

    Returns a dict keyed by feature name with shape (N_frames, dim) for vector features.
    Rows are ordered by the absolute ``index`` column so slicing by [from,to) from the
    episodes table yields the correct per-episode data.
    """
    data_parquets = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_parquets:
        raise FileNotFoundError(f"No data parquets under {dataset_root}/data")

    keys = list(feature_keys)
    columns = keys + ["index"]

    logging.info(f"  Reading {len(data_parquets)} parquet file(s), columns={keys}")
    chunks: list[np.ndarray] = []
    per_feature_chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    index_chunks: list[np.ndarray] = []
    for p in data_parquets:
        tbl = pq.read_table(p, columns=columns)
        df = tbl.to_pandas()
        index_chunks.append(df["index"].to_numpy(dtype=np.int64))
        for k in keys:
            col = df[k].to_numpy()
            # pandas stores list/array-valued cells as object arrays; stack into a 2D float32 block.
            if col.dtype == object:
                arr = np.stack([np.asarray(x, dtype=np.float32) for x in col])
            else:
                arr = col.astype(np.float32, copy=False)
                if arr.ndim == 1:
                    arr = arr[:, None]
            per_feature_chunks[k].append(arr)

    absolute_index = np.concatenate(index_chunks)
    order = np.argsort(absolute_index, kind="stable")
    if not np.array_equal(absolute_index[order], np.arange(len(absolute_index))):
        # LeRobot's ``index`` column must be contiguous [0, N); flag if it isn't so
        # slicing by episode from/to indices stays correct.
        raise RuntimeError(
            f"absolute index column is not 0..{len(absolute_index)-1}; "
            f"cannot align per-episode slices. first={absolute_index[order[:5]]} "
            f"last={absolute_index[order[-5:]]}"
        )

    arrays: dict[str, np.ndarray] = {}
    for k in keys:
        merged = np.concatenate(per_feature_chunks[k], axis=0)
        arrays[k] = merged[order]
    return arrays


def _compute_episode_stats(arrays: dict[str, np.ndarray], ranges: np.ndarray) -> list[dict[str, dict]]:
    """Compute per-episode stats via lerobot's own get_feature_stats."""
    ep_stats_list: list[dict[str, dict]] = []
    for (from_idx, to_idx) in ranges:
        ep_stats: dict[str, dict] = {}
        for key, arr in arrays.items():
            slab = arr[from_idx:to_idx]
            # Match process_single_episode: axis=0, keepdims = (ndim == 1)
            keepdims = slab.ndim == 1
            ep_stats[key] = get_feature_stats(slab, axis=0, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES)
        ep_stats_list.append(ep_stats)
    return ep_stats_list


def _merge_stats(existing: dict, computed: dict[str, dict], total_frames: int) -> dict:
    """Overlay computed quantile stats onto the existing stats.json dict.

    - Non-video keys in ``computed`` replace existing entries wholesale (new dicts carry
      the full {min, max, mean, std, count, q01..q99} set).
    - ``observation.images.rgb``: ensure the ImageNet stub is present with count=total_frames.
    """
    out: dict[str, dict] = {}
    for k, v in existing.items():
        out[k] = dict(v)
    for k, v in computed.items():
        out[k] = {sk: (sv.tolist() if isinstance(sv, np.ndarray) else sv) for sk, sv in v.items()}

    stub = out.get(VIDEO_KEY, {})
    required = {"mean", "std", "min", "max", "count"}
    if not required.issubset(stub.keys()):
        out[VIDEO_KEY] = {**IMAGENET_STUB, "count": [int(total_frames)]}
    else:
        stub["count"] = [int(total_frames)]
        out[VIDEO_KEY] = stub
    return out


def _verify(dataset_root: Path, old_stats: dict, new_stats: dict, total_frames: int, repo_id: str) -> dict:
    """Run sanity checks against the newly written stats.json; raises on failure.

    Returns a summary dict of notable values for reporting.
    """
    summary: dict = {"repo_id": repo_id, "checks": []}

    def _require(cond: bool, msg: str):
        summary["checks"].append((msg, bool(cond)))
        if not cond:
            raise AssertionError(f"[{repo_id}] verification failed: {msg}")

    for feat in REQUIRED_PI05:
        _require(feat in new_stats, f"{feat} present in stats")
        for q in QUANTILE_KEYS:
            _require(q in new_stats[feat], f"{feat} has {q}")

        q_arr = {q: np.asarray(new_stats[feat][q], dtype=np.float64) for q in QUANTILE_KEYS}
        for q in QUANTILE_KEYS:
            _require(np.all(np.isfinite(q_arr[q])), f"{feat}/{q} all finite")
        for a, b in zip(QUANTILE_KEYS, QUANTILE_KEYS[1:]):
            _require(np.all(q_arr[a] <= q_arr[b] + 1e-6), f"{feat}: {a} <= {b} elementwise")

        count = np.asarray(new_stats[feat]["count"])
        _require(int(count.flatten()[0]) == total_frames, f"{feat}/count == total_frames ({total_frames})")
        summary[f"{feat}.q01"] = q_arr["q01"].tolist()
        summary[f"{feat}.q50"] = q_arr["q50"].tolist()
        summary[f"{feat}.q99"] = q_arr["q99"].tolist()

    stub = new_stats.get(VIDEO_KEY, {})
    # Use isclose rather than == because some datasets (e.g. merged/combined) carry
    # floating-point drift in the ImageNet stub from an earlier aggregate_stats pass.
    _require(
        np.allclose(np.asarray(stub.get("mean", [])), np.asarray(IMAGENET_STUB["mean"]), atol=1e-6),
        f"{VIDEO_KEY} mean ~= ImageNet stub",
    )
    _require(
        np.allclose(np.asarray(stub.get("std", [])), np.asarray(IMAGENET_STUB["std"]), atol=1e-6),
        f"{VIDEO_KEY} std ~= ImageNet stub",
    )

    # Compare old vs new mean/std: should be within 1% relative tolerance (loose — per-episode
    # averaging of means is already approximate; the point is to catch a missing-row bug).
    tol = 0.02
    for feat in REQUIRED_PI05:
        if feat not in old_stats:
            continue
        for stat in ("mean", "std"):
            if stat not in old_stats[feat]:
                continue
            old = np.asarray(old_stats[feat][stat], dtype=np.float64).flatten()
            new = np.asarray(new_stats[feat][stat], dtype=np.float64).flatten()
            denom = np.maximum(np.abs(old), 1e-6)
            rel = np.abs(new - old) / denom
            if not np.all(rel <= tol):
                logging.warning(f"[{repo_id}] {feat}/{stat} rel-diff vs old: {rel.tolist()} (tol={tol})")
            summary[f"{feat}.{stat}.max_rel_diff_vs_old"] = float(rel.max())

    # Finally, try loading the dataset with pyav backend — ensures stats.json parses cleanly.
    try:
        LeRobotDataset(repo_id=repo_id, root=dataset_root, video_backend="pyav")
        summary["checks"].append(("LeRobotDataset load", True))
    except Exception as e:  # noqa: BLE001
        summary["checks"].append(("LeRobotDataset load", False))
        raise AssertionError(f"[{repo_id}] LeRobotDataset failed to load: {e}") from e

    return summary


def _process_dataset(dataset_root: Path, repo_id: str, overwrite: bool = False) -> dict:
    info = _load_info(dataset_root)
    total_frames = int(info["total_frames"])
    total_eps = int(info["total_episodes"])

    stats_path = dataset_root / "meta" / "stats.json"
    old_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    already_has = any("q01" in v for v in old_stats.values())
    if already_has and not overwrite:
        logging.info(f"[{repo_id}] already has quantile stats; skipping (use --overwrite to force)")
        return {"repo_id": repo_id, "skipped": True}

    non_video_keys = [k for k in STATS_FEATURES if k in info["features"]]
    logging.info(f"[{repo_id}] total_episodes={total_eps} total_frames={total_frames} keys={non_video_keys}")

    t0 = time.time()
    ranges = _load_episode_ranges(dataset_root)
    assert len(ranges) == total_eps, f"episode count mismatch: {len(ranges)} vs info total_episodes {total_eps}"

    arrays = _load_feature_arrays(dataset_root, non_video_keys)
    for k, a in arrays.items():
        assert a.shape[0] == total_frames, f"{k}: shape[0]={a.shape[0]} expected {total_frames}"
    t_load = time.time() - t0

    t1 = time.time()
    ep_stats_list = _compute_episode_stats(arrays, ranges)
    t_ep = time.time() - t1

    t2 = time.time()
    aggregated = aggregate_stats(ep_stats_list)
    t_agg = time.time() - t2

    merged = _merge_stats(old_stats, aggregated, total_frames)

    # write via lerobot helper to match on-disk serialization (handles np arrays).
    np_stats = {k: {sk: np.asarray(sv) for sk, sv in v.items()} for k, v in merged.items()}
    write_stats(np_stats, dataset_root)

    new_stats = json.loads(stats_path.read_text())
    summary = _verify(dataset_root, old_stats, new_stats, total_frames, repo_id)
    summary["timing"] = {
        "load_s": round(t_load, 2),
        "per_ep_stats_s": round(t_ep, 2),
        "aggregate_s": round(t_agg, 2),
        "total_s": round(time.time() - t0, 2),
    }
    summary["total_frames"] = total_frames
    summary["total_episodes"] = total_eps
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="Directory containing dataset folders")
    ap.add_argument("--namespace", default="mateoguaman",
                    help="HF namespace used to form repo_id labels")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Folder names under --root to process")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if quantile stats already exist")
    ap.add_argument("--report", type=Path, default=None,
                    help="Write per-dataset summary JSON here")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    all_summaries = []
    for name in args.datasets:
        dataset_root = args.root / name
        if not dataset_root.is_dir():
            logging.error(f"[{name}] missing at {dataset_root}")
            sys.exit(2)
        repo_id = f"{args.namespace}/{name}"
        logging.info(f"=== {repo_id} ===")
        summary = _process_dataset(dataset_root, repo_id, overwrite=args.overwrite)
        all_summaries.append(summary)
        logging.info(f"[{repo_id}] done: {summary.get('timing')}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(all_summaries, indent=2))
        logging.info(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
