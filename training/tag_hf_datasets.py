#!/usr/bin/env python3
"""Tag each Language Table dataset repo on HF Hub with the LeRobot codebase version.

LeRobot 0.5.x resolves the dataset revision via `get_safe_version()`, which requires
the HF repo to have a git tag matching the codebase version (e.g. "v3.0"). Our
datasets are already stored in v3.0 format (per meta/info.json) but the repos
lack the tag, so LeRobot's revision lookup fails with RevisionNotFoundError
(surfaced as a confusing TypeError due to a bug in 0.5.1's error formatting).

Run once: `./ltvenv/bin/python training/tag_hf_datasets.py`
"""
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

TAG = "v3.0"
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
    try:
        api.create_tag(repo_id, tag=TAG, repo_type="dataset")
        print(f"tagged {repo_id} -> {TAG}")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print(f"skip   {repo_id} ({TAG} already exists)")
        else:
            print(f"FAIL   {repo_id}: {e}")
