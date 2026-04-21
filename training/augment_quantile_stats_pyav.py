#!/usr/bin/env python3
"""Wrapper around `augment_dataset_quantile_stats` that forces the pyav backend.

Tillicum's conda env lacks system ffmpeg libs (libavutil.so.*), so torchcodec —
LeRobot's default video backend — can't load. The upstream augment script
doesn't expose `--video-backend`, so we call its internals directly with a
LeRobotDataset constructed with `video_backend="pyav"`.

Usage:
    python training/augment_quantile_stats_pyav.py \
        --repo-id mateoguaml/language_table_sim_combined \
        --root "$DATASET_ROOT/language_table_sim_combined"
"""
import argparse
import logging
from pathlib import Path

from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.augment_dataset_quantile_stats import (
    compute_quantile_stats_for_dataset,
    has_quantile_stats,
)
from lerobot.utils.utils import init_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True, type=Path,
                    help="Local dataset root (quantile computation reads every frame)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if quantile stats already exist")
    ap.add_argument("--no-push", action="store_true",
                    help="Skip push_to_hub after computing (local-only update)")
    args = ap.parse_args()

    init_logging()

    logging.info(f"Loading {args.repo_id} from {args.root} (video_backend=pyav)")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        video_backend="pyav",
    )

    if not args.overwrite and has_quantile_stats(dataset.meta.stats):
        logging.info("Dataset already contains quantile statistics. No action needed.")
        return

    logging.info(
        f"Computing quantile stats over {dataset.meta.total_episodes} episodes / "
        f"{dataset.meta.total_frames} frames. This scans every frame once."
    )
    new_stats = compute_quantile_stats_for_dataset(dataset)

    logging.info(f"Writing updated stats.json to {dataset.meta.root}")
    dataset.meta.stats = new_stats
    write_stats(new_stats, dataset.meta.root)
    logging.info("Local update complete.")

    if not args.no_push:
        logging.info(f"Pushing updated stats to hf.co/datasets/{args.repo_id}")
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
