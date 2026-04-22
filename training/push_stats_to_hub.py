#!/usr/bin/env python3
"""Upload updated meta/stats.json files to HF Hub and re-tag v3.0.

Runs per dataset: HfApi.upload_file of meta/stats.json, then delete+create tag v3.0
so LeRobot's ``revision="v3.0"`` resolution picks up the new commit.
"""
import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--namespace", default="mateoguaman")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--tag", default="v3.0")
    ap.add_argument("--message", default="Add q01/q10/q50/q90/q99 quantile stats for pi0.5 training")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api = HfApi()

    for name in args.datasets:
        repo_id = f"{args.namespace}/{name}"
        stats_path = args.root / name / "meta" / "stats.json"
        if not stats_path.is_file():
            logging.error(f"[{repo_id}] stats.json missing at {stats_path}")
            continue

        if args.dry_run:
            logging.info(f"[{repo_id}] DRY RUN — would upload {stats_path} and re-tag {args.tag}")
            continue

        logging.info(f"[{repo_id}] uploading {stats_path} ({stats_path.stat().st_size} B)")
        api.upload_file(
            path_or_fileobj=str(stats_path),
            path_in_repo="meta/stats.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=args.message,
        )

        try:
            api.delete_tag(repo_id, tag=args.tag, repo_type="dataset")
            logging.info(f"[{repo_id}] deleted existing tag {args.tag}")
        except HfHubHTTPError as e:
            logging.info(f"[{repo_id}] could not delete tag {args.tag} (probably didn't exist): {e}")

        api.create_tag(repo_id, tag=args.tag, revision=None, repo_type="dataset")
        logging.info(f"[{repo_id}] created tag {args.tag} on latest commit")


if __name__ == "__main__":
    main()
