#!/usr/bin/env python3
"""Merge the 8 sim-only Language Table datasets into a single LeRobot v3.0 dataset.

Excludes `language_table` (the real-robot dataset). Operates on local copies
under /media/mateo/Storage/lerobot_datasets_v3/ so the merge is fast and doesn't
round-trip through HF Hub.

After merging, patches the output `meta/stats.json` with ImageNet stats under
`observation.images.rgb` — the source stats.json files lack this key (empty
dicts get stripped by LeRobot's `flatten_dict`), so the merged output inherits
the same gap and training would crash with KeyError without the patch.

Usage:
    ./lerobot_env/bin/python training/merge_sim_datasets.py \
        --input_root /media/mateo/Storage/lerobot_datasets_v3 \
        --output_root /media/mateo/Storage/lerobot_datasets_v3 \
        --output_name language_table_sim_combined

    # Optionally push to HF Hub after:
    #   huggingface-cli upload mateoguaman/language_table_sim_combined <output_path> .
    # or add --push_to_hub and pass --hub_repo_id.
"""
import argparse
import json
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


SIM_DATASETS = [
    "language_table_sim",
    "language_table_blocktoblock_sim",
    "language_table_blocktoblock_4block_sim",
    "language_table_blocktoblock_oracle_sim",
    "language_table_blocktoblockrelative_oracle_sim",
    "language_table_blocktoabsolute_oracle_sim",
    "language_table_blocktorelative_oracle_sim",
    "language_table_separate_oracle_sim",
]

VIDEO_KEY = "observation.images.rgb"

# Shape (3,1,1) to satisfy `_validate_stat_value` for image features.
IMAGENET_MEAN = [[[0.485]], [[0.456]], [[0.406]]]
IMAGENET_STD = [[[0.229]], [[0.224]], [[0.225]]]
ZEROS_3_1_1 = [[[0.0]], [[0.0]], [[0.0]]]
ONES_3_1_1 = [[[1.0]], [[1.0]], [[1.0]]]


def _load_total_frames(dataset_root: Path) -> int:
    with (dataset_root / "meta" / "info.json").open() as f:
        return int(json.load(f)["total_frames"])


def patch_stats(dataset_root: Path, add_video_stub: bool = True) -> bool:
    """Fix two issues in legacy v3.0 stats.json files so lerobot 0.5.x accepts them.

    1. `count` is stored as a scalar int (e.g. 343688), but `_validate_stat_value`
       in compute_stats.py requires shape (1,). Wrap as a one-element list.
    2. No entry for the video key (`observation.images.rgb`). `aggregate_feature_stats`
       expects every feature to expose {min, max, mean, std, count}, so we insert
       a full ImageNet-based stub (bounds = [0,1], count = this dataset's frame
       total). The training loader overrides the mean/std anyway when
       `use_imagenet_stats=True`.

    Returns True if the stats.json was modified.
    """
    stats_path = dataset_root / "meta" / "stats.json"
    with stats_path.open() as f:
        stats = json.load(f)

    changed = False

    for feature_key, per_stat in stats.items():
        for stat_key, val in per_stat.items():
            if stat_key == "count" and isinstance(val, (int, float)):
                per_stat[stat_key] = [int(val)]
                changed = True

    if add_video_stub:
        required = {"mean", "std", "min", "max", "count"}
        existing = set(stats.get(VIDEO_KEY, {}).keys())
        if not required.issubset(existing):
            total_frames = _load_total_frames(dataset_root)
            stats[VIDEO_KEY] = {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "min": ZEROS_3_1_1,
                "max": ONES_3_1_1,
                "count": [total_frames],
            }
            changed = True

    if changed:
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, type=Path,
                    help="Directory containing the source dataset folders")
    ap.add_argument("--output_root", required=True, type=Path,
                    help="Directory where the merged dataset will be written")
    ap.add_argument("--output_name", default="language_table_sim_combined",
                    help="Folder name for the merged dataset (also used as repo_id suffix)")
    ap.add_argument("--hub_namespace", default="mateoguaman",
                    help="HF namespace used to form the output repo_id label")
    ap.add_argument("--push_to_hub", action="store_true",
                    help="Push merged dataset to HF Hub after merging")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Override the default sim-dataset list (useful for small test merges)")
    args = ap.parse_args()

    source_names = args.datasets if args.datasets else SIM_DATASETS

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    output_repo_id = f"{args.hub_namespace}/{args.output_name}"
    output_dir: Path = output_root / args.output_name

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Refusing to overwrite. Remove it first or pick a different --output_name."
        )

    # First pass: patch each source stats.json (idempotent) so merge's
    # aggregate_stats step passes 0.5.x's shape validation.
    print("Patching source stats.json files (in-place; idempotent)")
    for name in source_names:
        src_root = input_root / name
        src_stats = src_root / "meta" / "stats.json"
        if not src_stats.is_file():
            raise FileNotFoundError(f"Missing stats file: {src_stats}")
        modified = patch_stats(src_root)
        print(f"  {name}: {'patched' if modified else 'already OK'}")
    print()

    # Load each source from its local v3.0 directory.
    datasets = []
    total_eps = 0
    total_frames = 0
    for name in source_names:
        src_path = input_root / name
        if not src_path.is_dir():
            raise FileNotFoundError(f"Source dataset missing: {src_path}")
        print(f"Loading {name} from {src_path}")
        # repo_id is just a label here — `root=` makes it load from disk.
        ds = LeRobotDataset(
            repo_id=f"{args.hub_namespace}/{name}",
            root=src_path,
        )
        print(f"  episodes={ds.meta.total_episodes}  frames={ds.meta.total_frames}")
        total_eps += ds.meta.total_episodes
        total_frames += ds.meta.total_frames
        datasets.append(ds)

    print()
    print(f"Merging {len(datasets)} datasets -> {output_dir}")
    print(f"  expected totals: episodes={total_eps}  frames={total_frames}")
    print()

    merged = merge_datasets(
        datasets,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
    )

    print()
    print(f"Merge complete.")
    print(f"  output_dir={output_dir}")
    print(f"  episodes={merged.meta.total_episodes}  frames={merged.meta.total_frames}")

    # Source patches flow through `aggregate_stats`, so the merged file should
    # already be correct. Re-run patch_stats on the output as a safety net
    # (idempotent — no-op when sources are already patched).
    print()
    print("Verifying merged stats.json is complete")
    modified = patch_stats(output_dir)
    print(f"  merged stats.json: {'patched' if modified else 'already OK'}")

    if args.push_to_hub:
        print()
        print(f"Pushing to HF Hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()
        print("Done.")


if __name__ == "__main__":
    main()
