# Batch-Size Probe Results

Empirical VRAM usage for training Language Table VLA policies on a Tillicum
H200 (143,771 MiB / 140.4 GiB HBM3). Reproduce with
`training/probe_batch_size.py` — it runs `lerobot_train` as a subprocess for a
few steps per candidate batch size and polls `nvidia-smi` for peak memory.

## 2026-04-20: SmolVLA expert-only, 1×H200

- **Preset:** `smolvla_expert` (default — `train_expert_only=True`,
  `freeze_vision_encoder=True`; vision + SmolLM frozen, only the action
  expert + state projection train)
- **Policy:** `lerobot/smolvla_base`
- **Dataset:** `mateoguaman/language_table_sim_combined` (local copy at
  `$DATASET_ROOT/language_table_sim_combined`)
- **Probe steps:** 5 per batch size · `num_workers=2` · `chunk_size=10`
- **GPU:** 1×H200 (job 90448 on `g013`)
- **Job:** `salloc --gres=gpu:1 --cpus-per-task=8 --mem=200G --time=1h`

| batch size | peak VRAM | % of 140.4 GiB | update_s | dataload_s | fit |
|-----------:|----------:|---------------:|---------:|-----------:|:---:|
|         32 |  10.7 GiB |           7.4% |    0.79s |      6.54s | yes |
|         64 |  19.1 GiB |          13.3% |    1.27s |     10.18s | yes |
|         96 |  27.0 GiB |          18.7% |    1.02s |     14.19s | yes |
|        128 |  35.3 GiB |          24.5% |    1.15s |     17.71s | yes |
|        192 |  51.4 GiB |          35.8% |    1.44s |     24.67s | yes |
|        256 |  68.5 GiB |          47.6% |    1.77s |     32.98s | yes |

Raw output: `training/outputs/probe_smolvla_h200.json`.

### Takeaways

1. **bs=256 still fits at 47.6% of VRAM.** Expert-only finetuning has a tiny
   activation footprint (~130 MiB/sample above a ~7 GiB fixed cost), so we
   didn't find the ceiling. Extrapolating linearly: bs ≈ 512 would sit around
   ~125 GiB (~90% of VRAM). Re-probe with `--batch_sizes 320,384,448,512` if
   you want to pin down the max; otherwise **bs=256 per GPU is a safe choice
   for multi-GPU runs**, where NCCL buffers + grad bucketing add a few GiB.
2. **Data loading is the bottleneck, not compute.** `dataload_s` grows
   linearly with batch size (6.5s at bs=32 → 33s at bs=256) while `update_s`
   stays ≤1.8s. At `num_workers=2` the GPU spends >90% of step time idle
   waiting for pyav video decoding. Bump `NUM_WORKERS` (8 per GPU is the
   recipe default; try 16 on dedicated compute nodes) before increasing
   batch size — otherwise larger batches just spend more time blocked.
3. **Full effective batch.** On a 4×H200 node with the recipe default
   `NUM_WORKERS=8` and `BATCH_SIZE=256`, effective batch is 1024.

### What this does NOT measure

- **Full finetune.** Vision encoder + SmolLM backbone are frozen here. Full
  finetuning (all params trainable) will use much more memory per sample
  because activations flow through the vision+LM stacks. To probe it, run:
  ```
  python training/probe_batch_size.py --preset smolvla_full \
      --dataset_repo mateoguaman/language_table_sim_combined \
      --dataset_root "$DATASET_ROOT/language_table_sim_combined" \
      --batch_sizes 8,16,32,64,96
  ```
- **Pi0 / Pi0.5.** Presets `pi0_full`, `pi0_expert`, `pi05_full`,
  `pi05_expert` cover these. Upstream defaults for both pi0 families are
  *full* finetune (`train_expert_only=False`), so start with smaller
  `--batch_sizes 4,8,16,32`.
- **DDP overhead.** The probe runs 1 GPU only. Per-GPU VRAM rises by a few
  hundred MiB under NCCL + gradient bucketing; the 90%-of-max recommendation
  already leaves headroom.

## 2026-04-21: SmolVLA full finetune, 1×H200

- **Preset:** `smolvla_full` (`train_expert_only=False`,
  `freeze_vision_encoder=False`; every param trainable — vision encoder,
  SmolLM backbone, action expert, projections)
- **Policy:** `lerobot/smolvla_base`
- **Dataset:** `mateoguaman/language_table_sim_combined`
- **Probe steps:** 5 per batch size · `num_workers=2` · `chunk_size=10`
- **GPU:** 1×H200 (job 90788)

| batch size | peak VRAM | % of 140.4 GiB | update_s | dataload_s | fit |
|-----------:|----------:|---------------:|---------:|-----------:|:---:|
|          8 |  13.0 GiB |           9.3% |    1.24s |      3.13s | yes |
|         16 |  22.8 GiB |          16.3% |    1.15s |      4.01s | yes |
|         32 |  41.5 GiB |          29.6% |    1.25s |      5.96s | yes |
|         64 |  78.3 GiB |          55.8% |    1.55s |      9.74s | yes |
|         96 | 115.5 GiB |          82.3% |    1.84s |     13.78s | yes |

Raw output: `training/outputs/probe_smolvla_full_90788.json`. Largest fit at
bs=96 — next step (bs=128) would extrapolate to ~154 GiB and OOM.

### Takeaways

1. **bs=96 per GPU is the ceiling for full finetune** on H200 — leaving
   ~17% headroom for NCCL buffers / grad bucketing under DDP.
2. **Full finetune costs ~1.2 GiB/sample above a ~3 GiB fixed cost** (vs.
   ~130 MiB/sample for expert-only) — roughly 9× the per-sample
   activation footprint, driven by backprop through the vision encoder +
   SmolLM backbone.
3. **Data loading is still the bottleneck.** `dataload_s` (13.8s @ bs=96)
   dominates `update_s` (1.8s) at `num_workers=2`. Bump `NUM_WORKERS` for
   training runs.

## 2026-04-21: pi0.5 expert-only, 1×H200

- **Preset:** `pi05_expert` (`train_expert_only=True`,
  `freeze_vision_encoder=True`; PaliGemma backbone + SigLIP vision frozen,
  only the gemma_expert action head + projections train)
- **Policy:** `lerobot/pi05_base`
- **Dataset:** `mateoguaman/language_table_sim_combined`
- **Probe steps:** 5 per batch size · `num_workers=2` · `chunk_size=10`
- **GPU:** 1×H200 (job 90946)
- **Image args:** `--policy.empty_cameras=2 --rename_map={"observation.images.rgb": "observation.images.base_0_rgb"}`

| batch size | peak VRAM | % of 140.4 GiB | update_s | dataload_s | fit |
|-----------:|----------:|---------------:|---------:|-----------:|:---:|
|          8 |  79.7 GiB |          56.8% |    1.12s |      4.06s | yes |
|         16 | 139.2 GiB |          99.2% |    1.51s |      4.88s | yes |
|         32 | 139.8 GiB |          99.5% |        — |          — | OOM |

Raw output: `training/outputs/probe_pi05_expert_90946.json`. Largest fit at
bs=16 — recommended safe choice for DDP: **bs=12 per GPU** (90% rule).

### Takeaways

1. **pi0.5 is dramatically heavier than SmolVLA, even expert-only.**
   bs=8 pi0.5 expert uses 79.7 GiB; bs=32 SmolVLA expert uses 10.7 GiB. That's
   a ~25× per-sample jump driven by PaliGemma 3B activations (SigLIP + Gemma
   LM) vs SmolLM's ~135M.
2. **bs=16 is already at 99.2% of VRAM.** One more doubling OOMs. No head
   room for NCCL/grad bucketing at bs=16 under DDP — drop to bs=12.
3. **Compute is no longer ignorable.** `update_s` grew from 1.12s (bs=8) to
   1.51s (bs=16), while `dataload_s` stayed ~4–5s. Heavier backbone =
   higher update cost per step; data loading is still the bottleneck at
   `num_workers=2` but the gap is narrower than SmolVLA.

## 2026-04-21: pi0.5 full finetune, 1×H200

- **Preset:** `pi05_full` (`train_expert_only=False`,
  `freeze_vision_encoder=False`; every param trainable — SigLIP vision,
  PaliGemma backbone, gemma_expert, projections)
- **Policy:** `lerobot/pi05_base`
- **Dataset:** `mateoguaman/language_table_sim_combined`
- **Probe steps:** 5 per batch size · `num_workers=2` · `chunk_size=10`
- **GPU:** 1×H200 (job 90947)

| batch size | peak VRAM | % of 140.4 GiB | update_s | dataload_s | fit |
|-----------:|----------:|---------------:|---------:|-----------:|:---:|
|          4 |  95.9 GiB |          68.3% |    1.26s |      3.48s | yes |
|          8 | 139.3 GiB |          99.2% |        — |          — | OOM |

Raw output: `training/outputs/probe_pi05_full_90947.json`. Largest fit at
bs=4 — recommended: **bs=4 per GPU** (bs=8 OOMs; no safety margin to trim).

### Takeaways

1. **bs=4 is the ceiling for pi0.5 full finetune on H200.** Effective batch
   on a 4×H200 node is 16 (vs 1024 for SmolVLA expert). Plan gradient
   accumulation for any pi0.5 run that needs a larger effective batch.
2. **Full pi0.5 costs ~11 GiB/sample** above the ~50 GiB fixed model cost
   (vs ~1.2 GiB/sample for SmolVLA full, ~130 MiB/sample for SmolVLA
   expert). The vision encoder + PaliGemma backprop dominates.
3. **The recipe default `BATCH_SIZE=256` must be overridden per policy.**
   pi0.5 needs bs=4 (full) or bs=12 (expert), ~20–60× smaller than SmolVLA.

### Pi0.5-specific preprocessing note

pi0.5 declares its image inputs as `base_0_rgb`, `left_wrist_0_rgb`,
`right_wrist_0_rgb` (plus any `empty_camera_{i}` from `empty_cameras=N`) —
**not** the generic `camera{1,2,3}` naming used by SmolVLA/pi0. Language
Table has one camera, so the probe renames
`observation.images.rgb → observation.images.base_0_rgb`. The wrist cameras
stay missing; `_preprocess_images` only requires ≥1 declared image key in
the batch (the rest are zero-filled via `img_masks`). Use the SmolVLA/pi0
`camera1` mapping for those policies — it will not work for pi0.5.

## Notes on setup pitfalls

If you re-run this and see `RuntimeError: Disk quota exceeded` during the
first HF download:

1. `.env.tillicum` assigns `HF_HOME=/gpfs/scrubbed/$USER/.cache/huggingface`
   **but does not `export`** — use `set -a; source training/.env.tillicum;
   set +a` so the var propagates to the Python subprocess.
2. Pre-download the 51 GB dataset once with `hf download --repo-type dataset
   --revision v3.0 --local-dir "$DATASET_ROOT/language_table_sim_combined"
   mateoguaman/language_table_sim_combined` so subsequent probes / training
   runs skip the Hub round-trip.
