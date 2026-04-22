# Quantile-stats augmentation for Language Table LeRobot datasets

## Why

pi0.5 training (`lerobot/policies/pi05/configuration_pi05.py:73-79`) defaults to
`normalization_mapping = {VISUAL: IDENTITY, STATE: QUANTILES, ACTION: QUANTILES}`.
Existing Language Table v3.0 datasets on the Hub only carried `{min, max, mean, std,
count}` — no `q01/q10/q50/q90/q99` — so pi0.5 would fail at normalization init with
`ValueError: QUANTILES normalization mode requires q01 and q99 stats`.

The upstream augment script (`lerobot.scripts.augment_dataset_quantile_stats`) solves
this generically but calls `dataset[idx]` per frame, which decodes video through
pyav/torchcodec even though pi0.5 never consumes video stats (VISUAL=IDENTITY
short-circuits in `lerobot/processor/normalize_processor.py:305-307`). On a 58M-frame
dataset that's days of pointless video decode.

## What this change does

- `training/compute_quantile_stats_parquet.py` reads non-video columns straight from
  `data/chunk-*/file-*.parquet` (zero video decode), computes per-episode stats with
  `lerobot.datasets.compute_stats.get_feature_stats`, aggregates with
  `aggregate_stats`, and merges the output into `meta/stats.json` while preserving
  the existing `observation.images.rgb` ImageNet stub.
- `training/push_stats_to_hub.py` uploads just `meta/stats.json` and re-tags `v3.0`.
- Runs against **all 10 Language Table LeRobot datasets** under
  `/media/mateo/Storage/lerobot_datasets_v3/`.

End-to-end wall time: ~25 min total across all datasets on local NVMe (vs. days for
the upstream script).

## Processed datasets

| Dataset | Episodes | Frames | Elapsed |
|---|--:|--:|--:|
| language_table_blocktoblock_4block_sim | 8,298 | 318,470 | 7.09s |
| language_table_blocktoblock_sim | 8,000 | 343,688 | 6.86s |
| language_table_separate_oracle_sim | 200,000 | 2,996,661 | 157.42s |
| language_table_sim | 181,020 | 4,484,403 | 149.05s |
| language_table | 442,226 | 7,045,476 | 353.88s |
| language_table_blocktorelative_oracle_sim | 200,000 | 8,455,815 | 173.05s |
| language_table_blocktoblock_oracle_sim | 200,000 | 12,770,620 | 181.36s |
| language_table_blocktoblockrelative_oracle_sim | 200,000 | 12,816,749 | 181.40s |
| language_table_blocktoabsolute_oracle_sim | 200,000 | 15,666,385 | 187.87s |
| language_table_sim_combined | 1,197,318 | 57,852,791 | ~15 min |

Sanity spot-check (state q01/q99 and action q01/q99):

| Dataset | state q01 | state q99 | action q01 | action q99 |
|---|---|---|---|---|
| language_table_blocktoblock_4block_sim | [0.2809, -0.1353] | [0.4678, 0.1363] | [-0.0274, -0.0350] | [0.0291, 0.0346] |
| language_table_blocktoblock_sim | [0.2782, -0.1363] | [0.4701, 0.1364] | [-0.0260, -0.0308] | [0.0276, 0.0314] |
| language_table_separate_oracle_sim | [0.3122, -0.0992] | [0.4387, 0.1008] | [-0.0139, -0.0132] | [0.0143, 0.0130] |
| language_table_sim | [0.3057, -0.0859] | [0.4369, 0.0807] | [-0.0190, -0.0215] | [0.0197, 0.0214] |
| language_table (real) | [0.3230, -0.1041] | [0.4776, 0.1153] | [-0.0483, -0.0690] | [0.0516, 0.0683] |
| language_table_blocktorelative_oracle_sim | [0.2946, -0.1110] | [0.4592, 0.1121] | [-0.0175, -0.0180] | [0.0175, 0.0181] |
| language_table_blocktoblock_oracle_sim | [0.2878, -0.1314] | [0.4675, 0.1324] | [-0.0176, -0.0186] | [0.0175, 0.0186] |
| language_table_blocktoblockrelative_oracle_sim | [0.2866, -0.1321] | [0.4681, 0.1330] | [-0.0177, -0.0186] | [0.0175, 0.0186] |
| language_table_blocktoabsolute_oracle_sim | [0.2718, -0.1517] | [0.4756, 0.1522] | [-0.0173, -0.0185] | [0.0173, 0.0186] |
| language_table_sim_combined | [0.2868, -0.1289] | [0.4648, 0.1293] | [-0.0176, -0.0186] | [0.0176, 0.0186] |

State ranges are consistent across datasets (table workspace bounds, xy only).
Action distributions are tight for oracle sims and wider for `language_table`
(real-human teleop variance). The combined dataset's action quantiles match the
oracle sims because those contribute ~1M of its 1.2M episodes.

All automated checks passed for every dataset: quantile monotonicity, finiteness,
count equals `info.json`'s `total_frames`, video stub preserved, `LeRobotDataset`
loads with `video_backend="pyav"`, new mean/std within ~3×10⁻⁶ relative of prior
values.

## Verification reports (JSON)

- `training/outputs/quantile_report_smoke.json` — blocktoblock_4block_sim
- `training/outputs/quantile_report_individuals.json` — 8 individual datasets
- `training/outputs/quantile_report_combined.json` — language_table_sim_combined

Each `meta/stats.json` now has a `meta/stats.json.bak` sibling holding the
pre-augmentation file (for rollback).

## HF Hub push

All 10 repos re-uploaded `meta/stats.json` and re-tagged `v3.0` on the new commit
via `training/push_stats_to_hub.py`.

## Tillicum follow-up

On Tillicum, pull only the updated `meta/stats.json` for each dataset (no need to
re-download the full video corpus):

```bash
cd /gpfs/projects/weirdlab/mateo/projects/language-table
set -a; source training/.env.tillicum; set +a

for name in \
    language_table \
    language_table_sim \
    language_table_blocktoblock_sim \
    language_table_blocktoblock_4block_sim \
    language_table_blocktoblock_oracle_sim \
    language_table_blocktoblockrelative_oracle_sim \
    language_table_blocktoabsolute_oracle_sim \
    language_table_blocktorelative_oracle_sim \
    language_table_separate_oracle_sim \
    language_table_sim_combined
do
    hf download \
        --repo-type dataset \
        --revision v3.0 \
        --include "meta/stats.json" \
        --local-dir "${DATASET_ROOT}/${name}" \
        "mateoguaman/${name}"
done
```

Then revert the pi0.5 normalization workaround in the probe harness (was set when
the stats weren't yet in place):

```
training/probe_batch_size.py — remove the
    '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
flag from the pi05_expert and pi05_full presets so probes exercise the real
QUANTILES normalization path.
```

Finally, re-submit the pi05 probes:

```bash
PRESETS="pi05_expert pi05_full" bash training/slurm/submit_all_probes.sh
```
