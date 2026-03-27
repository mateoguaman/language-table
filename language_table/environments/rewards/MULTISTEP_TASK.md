# Multistep Block-to-Location Task

Configurable multi-step push task where the agent must push N blocks to N target locations in a single episode.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `locations` | `list[str]` or `None` | Pool of target locations (sampled without replacement). Defaults to all 9 absolute locations. |
| `shapes` | `list[str]` or `None` | If set, restricts eligible blocks to these shapes; instruction references blocks **by shape only**. |
| `colors` | `list[str]` or `None` | If set, restricts eligible blocks to these colors; instruction references blocks **by color only**. |
| `n_steps` | `int` | Number of (block, location) sub-goals per episode. |

When both `shapes` and `colors` are set, instructions use the full descriptor (e.g. "red cube"). When neither is set, all blocks are eligible and fully described.

Setting a parameter to `None` (empty in `.env`) means it is unconstrained and omitted from the language instruction.

## Reward

**Partial reward**: each block that reaches its target earns `goal_reward / n_steps`. Episode ends (`done=True`) only when **all** blocks are simultaneously in place. Inherits from `SortColorsToCornersPartialReward`.

## Current Configurations

### Train (`n_steps=2`)
- Locations: `top_left, top_right, bottom_left, bottom_right` (corners)
- Shapes: `moon, cube, star, pentagon` (all shapes)
- Colors: unconstrained (not in instruction)
- Example instruction: *"push the star to the top left and the moon to the bottom right"*

### Validation (`n_steps=3`)
- Locations: `top, bottom, center_left, center_right` (edge midpoints)
- Shapes: unconstrained (not in instruction)
- Colors: `red, blue, green, yellow` (all colors)
- Example instruction: *"move the red block to the top, the blue block to the bottom, and the green block to the center left"*

## Available Locations

Defined in `block2absolutelocation.py` (`ABSOLUTE_LOCATIONS`):

```
top_left    top         top_right
center_left center      center_right
bottom_left bottom      bottom_right
```

## File Layout

| File | Role |
|------|------|
| `language_table/environments/rewards/multistep_block_to_location.py` | Reward class and `make_multistep_reward()` factory |
| `language_table/lamer/server_main.py` | CLI arg parsing, reward wiring (`--task_locations`, `--task_shapes`, `--task_colors`, `--task_n_steps`) |
| `language_table/lamer/lava_env_manager.py` | `n_steps` param, `partial_completion` metric in `success_evaluator()` |
| `LaMer/.env.language_table` | Task configuration via `TRAIN_TASK_*` / `VAL_TASK_*` env vars |
| `LaMer/scripts/submit_language_table.sh` | Exports task env vars to slurm |
| `LaMer/scripts/slurm/lamer_language_table.slurm` | Builds `--task_*` CLI flags from env vars |

## Configuration Flow

```
.env.language_table          (TRAIN_TASK_LOCATIONS=top_left,top_right,...)
       │
       ▼
submit_language_table.sh     (source .env, export TRAIN_TASK_*)
       │
       ▼
lamer_language_table.slurm   (build TRAIN_TASK_FLAGS="--task_locations ...")
       │
       ▼
server_main.py               (parse args, call make_multistep_reward())
       │
       ▼
multistep_block_to_location  (MultiStepBlockToLocationReward class)
```

## Adding New Configurations

1. Set the `TRAIN_TASK_*` / `VAL_TASK_*` variables in `.env.language_table`.
2. Leave a variable **empty** (e.g. `TRAIN_TASK_COLORS=`) to make it unconstrained and omit it from the instruction.
3. Ensure `n_steps` does not exceed the number of available locations or eligible blocks (4 blocks on the table by default with `BLOCK_4`).
