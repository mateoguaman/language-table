# Language Table policy benchmark

What `training/eval/run_benchmark.sh` actually does, in detail.

## Pipeline

```
run_benchmark.sh                       (orchestrator; bash)
 ├─ for each CHECKPOINT in CHECKPOINTS:
 │    ├─ if lerobot: spawn lerobot_policy_server.py (PyTorch, lerobotenv)
 │    │             — TCP bridge so the LeRobot policy can be queried from
 │    │             ltvenv (which runs JAX/PyBullet/tf_agents).
 │    └─ run_eval.py (ltvenv) sweeps the matrix for that one policy.
 └─ aggregate_results.py reads each policy's episodes.csv and emits
    comparison.md with Wilson 95% CIs.
```

## Policies under test

Defined as `"id|type|checkpoint_path|extra"` rows in `CHECKPOINTS=()` near the
top of `run_benchmark.sh`. Default:

| id | type | path |
|----|------|------|
| `smolvla_full_93185` | lerobot | `/media/mateo/Storage/lerobot_checkpoints/smolvla_full_combined_sim_93185/pretrained_model` |
| `lava_resnet`        | lava    | `/home/mateo/projects/Isaac/language-table/checkpoints/bc_resnet_sim_checkpoint_955000` |

LAVA runs in-process (JAX). LeRobot runs in a separate Python env (PyTorch
2.x, numpy 2.x, transformers) and is reached over a TCP socket. The policy
server is started before each LeRobot policy is evaluated and torn down
after, so only one server is alive at a time.

## Episode matrix

For each policy we sweep `block_modes × seeds × reward_types`, and run
`NUM_EPISODES` per cell. Every episode is a fresh env reset.

Default knobs (env-var overridable):

| Knob | Default | Meaning |
|------|---------|---------|
| `BLOCK_MODES` | `BLOCK_8` | Set of blocks on the table. `BLOCK_4`, `BLOCK_8`, `BLOCK_4_WPOLE`, `BLOCK_8_WPOLE`. |
| `SEEDS` | `0 1 2` | One env seed per (policy, block_mode, reward_type, seed) cell. |
| `REWARD_TYPES` | 8 of 10 (see below) | Picks reward class + decides goal/instruction sampling. |
| `NUM_EPISODES` | `50` | Episodes per cell. |
| `MAX_STEPS` | `200` | Hard episode cap; counted as failure if reached. |
| `DELAY_REWARD_STEPS` | `0` | Extra frames the goal must be sustained for success. `0` = first-touch success. |
| `VIDEO_EPISODES` | unset | Episode indices to record per cell (e.g. `"0 5 10"`). Empty = no videos. |
| `SKIP_EXISTING` | `true` | Cells whose `result.json` exists are skipped — safe to re-run. |

Total episodes = `policies × block_modes × seeds × reward_types × NUM_EPISODES`.
The default benchmark is therefore 2 × 1 × 3 × 8 × 50 = **2,400 episodes**.

## Reward types

The reward class drives both the success check and the natural-language
instruction the policy sees. From `REWARD_REGISTRY` in `run_eval.py`. The
last column is whether the cell uses the RRT pushing oracle to filter out
infeasible starts (only applies to single-block-pushing tasks).

| Name | What success means | Oracle filter |
|------|--------------------|---------------|
| `blocktoblock` | Push named block A within threshold of named block B. | yes |
| `blocktoabsolutelocation` | Push block to a named region (e.g. *top left*). | yes |
| `blocktoblockrelativelocation` | Push A to a named relation around B (e.g. *to the left of*). | yes |
| `blocktorelativelocation` | Push block to a named relative direction. | yes |
| `block1tocorner` | Push the only block into a named corner (BLOCK_1 mode only). | yes |
| `separate` | Move two named blocks far enough apart. | no |
| `point2block` | Bring the EE close to a named block (no contact required). | no |
| `composite` | Reset picks one of 9 sub-rewards uniformly at random. | no |

Two long-horizon rewards exist in the registry but are **off by default**
(set `REWARD_TYPES="..."` to opt in):

| Name | Why off by default |
|------|--------------------|
| `multistep` | Single compound instruction (*"push X to A and then Y to B"*) — OOD for policies trained on per-step LAVA data. |
| `sortcolorstocorners` | Same problem: one mega-instruction enumerating 4 color→corner mappings. |

## What an "episode" actually does

For each cell `(policy, block_mode, seed, reward_type)`:

1. **Env construction** — `language_table.LanguageTable(block_mode, reward_factory, delay_reward_steps, seed)` wrapped with `GymWrapper` then `HistoryWrapper(history_length=1 or lava.sequence_length)`. LAVA additionally gets `ClipTokenWrapper` and `CentralCropImageWrapper`.
2. **Reset loop** — for oracle-compatible rewards we re-reset (≤20 attempts) until the RRT oracle finds a feasible plan; otherwise a single `env.reset()`. This avoids scoring the policy on physically infeasible scenes.
3. **Rollout** — call `policy.action(time_step, ())` until `time_step.is_last()` or `episode_steps >= MAX_STEPS`.
4. **Success** — read `raw_env.succeeded` (set by the reward class when the goal condition holds for `delay_reward_steps` consecutive frames).
5. **Logging** — append one row to `episodes.csv` with `success`, `steps`, `instruction`. If recorded, also write `videos/ep{NNN}_{success|fail}.mp4` (10 fps, libx264).

## Outputs

```
eval_results/<BENCHMARK_NAME>/
├── server.log                       # latest LeRobot server stdout/stderr
├── comparison.md                    # final Wilson-95% table (this is the headline)
├── <policy_id>/
│   ├── summary.json                 # all cells + per-cell aggregate
│   ├── episodes.csv                 # one row per episode (the source of truth)
│   └── <block_mode>/<seed>/<reward>/
│       ├── result.json              # this cell's success_rate, mean/median steps
│       └── videos/ep000_success.mp4 # only if VIDEO_EPISODES was set
```

`comparison.md` contains, per `block_mode`:
- success-rate table with Wilson 95% CIs (binomial, z=1.96)
- mean-steps table

…plus an "overall pooled" table (across block_modes × reward_types × seeds)
and a "per reward type" pooled table.

CIs pool successes across all seeds × episodes per `(policy, block_mode,
reward_type)`. With defaults that's `n = 3 × 50 = 150` Bernoulli trials per
cell — wide enough to actually distinguish policies.

## How to run

Defaults:

```bash
bash training/eval/run_benchmark.sh
```

Common overrides:

```bash
# Quick: 1 seed, 10 eps, 3 rewards
SEEDS="0" NUM_EPISODES=10 \
    REWARD_TYPES="blocktoblock separate point2block" \
    bash training/eval/run_benchmark.sh

# Add BLOCK_4 and tighten success criterion
BLOCK_MODES="BLOCK_4 BLOCK_8" DELAY_REWARD_STEPS=10 \
    bash training/eval/run_benchmark.sh

# Record episode 0 of every cell
VIDEO_EPISODES="0" bash training/eval/run_benchmark.sh

# Include the long-horizon OOD rewards
REWARD_TYPES="blocktoblock separate point2block multistep sortcolorstocorners composite" \
    bash training/eval/run_benchmark.sh
```

To compare more checkpoints (e.g. multiple training snapshots), append rows
to the `CHECKPOINTS=()` array in `run_benchmark.sh`.

## Re-runs and partial failures

`SKIP_EXISTING=true` (default) means a re-run of the same `BENCHMARK_NAME`
only fills in cells whose `result.json` is missing. To force a clean run,
either change `BENCHMARK_NAME` or `rm -rf eval_results/<BENCHMARK_NAME>`.

`episodes.csv` is opened in append mode, so re-runs accumulate rows. If you
re-run only a subset of cells, the aggregator pools whatever is in the CSV.
