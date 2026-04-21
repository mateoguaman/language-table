#!/usr/bin/env python3
"""Add an empty stub for the video feature to each dataset's meta/stats.json.

LeRobot 0.5.x's factory.py does ``stats[key][stats_type] = ...`` for every
camera key (to populate ImageNet stats), which requires ``stats[key]`` to
already exist. Our v2->v3 conversion copied v2's stats.json unchanged, and
v2 never wrote stats for the video feature 'observation.images.rgb', so
training crashes with ``KeyError: 'observation.images.rgb'``.

This script downloads each repo's stats.json, inserts ``"<video_key>": {}``
if missing, and uploads the patched file back, then re-tags v3.0 to the new
commit so LeRobot resolves it via the version tag.

Run once: ``./ltvenv/bin/python training/patch_hf_stats.py``
"""
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

VIDEO_KEY = "observation.images.rgb"
TAG = "v3.0"

# Pre-populate with ImageNet stats (c,1,1). LeRobot's factory.py overwrites
# these with the same values when use_imagenet_stats=True, but the key must
# hold non-empty numeric data because load_stats runs the dict through
# flatten_dict which drops empty-dict values entirely.
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}
REPOS = [
    "mateoguaman/language_table",
    "mateoguaman/language_table_sim",
    "mateoguaman/language_table_blocktoblock_sim",
    "mateoguaman/language_table_blocktoblock_4block_sim",
    "mateoguaman/language_table_blocktoblock_oracle_sim",
    "mateoguaman/language_table_blocktoblockrelative_oracle_sim",
    "mateoguaman/language_table_blocktoabsolute_oracle_sim",
    "mateoguaman/language_table_blocktorelative_oracle_sim",
    "mateoguaman/language_table_separate_oracle_sim",
]

api = HfApi()
for repo_id in REPOS:
    stats_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/stats.json",
        repo_type="dataset",
        revision="main",
    )
    with open(stats_path) as f:
        stats = json.load(f)

    if stats.get(VIDEO_KEY) == IMAGENET_STATS:
        print(f"skip   {repo_id} (already has {VIDEO_KEY} stats)")
        continue

    stats[VIDEO_KEY] = IMAGENET_STATS
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(stats, tmp, indent=2)
        tmp_path = tmp.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="meta/stats.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add empty stats stub for {VIDEO_KEY} (LeRobot 0.5.x compat)",
    )
    Path(tmp_path).unlink()

    try:
        api.delete_tag(repo_id, tag=TAG, repo_type="dataset")
    except HfHubHTTPError:
        pass
    api.create_tag(repo_id, tag=TAG, repo_type="dataset")
    print(f"patched {repo_id} and re-tagged {TAG}")
