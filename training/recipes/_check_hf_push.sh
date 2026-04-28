#!/usr/bin/env bash
# Fail-fast verification that this run can actually push to HF Hub.
#
# Sourced by recipes that pass --policy.push_to_hub=true. Skipped when
# PUSH_TO_HUB is set to anything other than "true". Expects HF_REPO_ID
# (e.g. "mateoguaman/smolvla_full_combined_sim_93185") in the environment.
#
# Errors out before launching training so multi-hour jobs don't blow up at
# the first checkpoint save with a 401 from the Hub.

if [ "${PUSH_TO_HUB:-true}" != "true" ]; then
    return 0 2>/dev/null || exit 0
fi

if [ -z "${HF_REPO_ID:-}" ]; then
    echo "ERROR: PUSH_TO_HUB is on but HF_REPO_ID is unset." >&2
    return 1 2>/dev/null || exit 1
fi

# Need HF_TOKEN in env, or a saved token at $HF_HOME/token (huggingface-cli login).
if [ -z "${HF_TOKEN:-}" ] && [ ! -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
    echo "ERROR: No HF auth found. Set HF_TOKEN in training/.env.user, or run" >&2
    echo "       'huggingface-cli login' on this node ($(hostname))." >&2
    return 1 2>/dev/null || exit 1
fi

python - "$HF_REPO_ID" <<'PY' || return 1 2>/dev/null || exit 1
import sys
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

repo_id = sys.argv[1]
ns = repo_id.split("/", 1)[0] if "/" in repo_id else None
if ns is None:
    sys.exit(f"ERROR: HF_REPO_ID must be 'namespace/name', got: {repo_id}")

api = HfApi()
try:
    me = api.whoami()
except HfHubHTTPError as e:
    sys.exit(f"ERROR: HF token is invalid or expired: {e}")

writable = {me["name"]} | {o["name"] for o in me.get("orgs", [])}
if ns not in writable:
    sys.exit(
        f"ERROR: '{me['name']}' has no write access to namespace '{ns}'. "
        f"Writable: {sorted(writable)}"
    )
print(f"HF push OK: {me['name']} -> {repo_id}")
PY
