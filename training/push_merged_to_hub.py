#!/usr/bin/env python3
"""Push a locally-merged LeRobot v3.0 dataset to the HF Hub.

Uses `upload_large_folder` so partial uploads can resume on reconnect —
important for the ~48 GB combined sim dataset. Also tags the pushed
revision as `v3.0` so LeRobot's version resolver finds it.

Usage:
    ./lerobot_env_v51/bin/python training/push_merged_to_hub.py \
        --local_root /media/mateo/Storage/lerobot_datasets_v3/language_table_sim_combined \
        --repo_id mateoguaman/language_table_sim_combined
"""
import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_root", required=True, type=Path,
                    help="Path to the merged dataset on disk")
    ap.add_argument("--repo_id", required=True,
                    help="Target HF Hub repo_id (namespace/name)")
    ap.add_argument("--private", action="store_true",
                    help="Create as a private dataset")
    args = ap.parse_args()

    print(f"Loading {args.local_root}")
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.local_root)
    print(f"  episodes={ds.meta.total_episodes}  frames={ds.meta.total_frames}")
    print(f"Pushing to hf.co/datasets/{args.repo_id} (private={args.private})")
    ds.push_to_hub(
        private=args.private,
        upload_large_folder=True,
        tag_version=True,
    )
    print("Push complete.")


if __name__ == "__main__":
    main()
