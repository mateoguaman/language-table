# Language Table

Language-Table is a suite of human-collected datasets and a multi-task continuous control benchmark for open vocabulary visuolinguomotor learning.

![](./docs/real.jpeg)      |  ![](./docs/sim.jpeg)
:-------------------------:|:-------------------------:|

## Installation

Installation with [uv](https://docs.astral.sh/uv/). `requirements.txt`
contains dependencies for running the environment and simple dataset examples.

```
uv venv --python 3.10 ./ltvenv
source ./ltvenv/bin/activate
uv pip install -r ./requirements.txt
uv pip install --no-deps git+https://github.com/google-research/scenic.git@ae21d9e884015aa7bc7cf1d489af53d16c249726
export PYTHONPATH=${PWD}:$PYTHONPATH  # or run echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
```

For running the full train script, install using `requirements_static.txt`, as
this contains pinned versions for running the full train script.

```
uv venv --python 3.10 ./ltvenvtrain
source ./ltvenvtrain/bin/activate
uv pip install --no-deps -r ./requirements_static.txt
export PYTHONPATH=${PWD}:$PYTHONPATH
```
## Quickstart

### Examples
#### Scripts
Run and edit the following examples:

Load the environment and run 5 random steps:

```
python3 language_table/examples/environment_example.py
```

Load dataset and print first 5 elements:

```
python3 language_table/examples/dataset_example.py
```

#### Train

```
source ./ltvenvtrain/bin/activate
mkdir -p /tmp/language_table_train/
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  python language_table/train/main.py --config=./language_table/train/configs/language_table_sim_local.py --workdir=/tmp/language_table_train/
```

#### Colab
See the [colab](https://colab.research.google.com/github/google-research/language-table/blob/main/language_table/examples/language_table_tutorial.ipynb) for a more complete tutorial.

### Data
```
import tensorflow_datasets as tfds
data_directory = 'gs://gresearch/robotics/language_table/0.0.1/'
dataset = tfds.builder_from_directory(data_directory).as_dataset()
```

### Environment
```
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block

env = language_table.LanguageTable(
  block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
  reward_factory=block2block.BlockToBlockReward,
  control_frequency=10.0,
)
obs = env.reset()
```

## Datasets

### Descriptions

* **Real Robot**
  * **language_table**: 442,226 episodes of real robot relabeled data.
* **Simulation (human)**
  * **language_table_sim**: 181,020 episodes of simulation relabeled data.
  * **language_table_blocktoblock_sim**: 8,000 episodes of single task "block to block" data.
  * **language_table_blocktoblock_4block_sim**: 8,298 episodes of single task "block to block" data in the 4 block configuration.
* **Simulation (oracle)**
  * **language_table_blocktoblock_oracle_sim**: 200,000 episodes of single task "block to block" data from an oracle scripted agent.
  * **language_table_blocktoblockrelative_oracle_sim**: 200,000 episodes of single task "block-to-block-relative" data from an oracle scripted agent.
  * **language_table_blocktoabsolute_oracle_sim**: 200,000 episodes of single task "block to absolute location" data from an oracle scripted agent.
  * **language_table_blocktorelative_oracle_sim**: 200,000 episodes of single task "block to relative location" data from an oracle scripted agent.
  * **language_table_separate_oracle_sim**: 200,000 episodes of single task "separate blocks" data from an oracle scripted agent.

### Summary Table

Dataset | Real/sim | Controlled by | Language-labeled by | # episodes
--------| --------- | ------------- | ----------------- | --------: 
language_table | real | human | human | 442,226
language_table_sim | sim | human | human | 181,020
language_table_blocktoblock_sim | sim | human | scripted | 8,000
language_table_blocktoblock_4block_sim |  sim | human | scripted | 8,298
language_table_blocktoblock_oracle_sim | sim | oracle | scripted | 200,000
language_table_blocktoblockrelative_oracle_sim | sim | oracle | scripted | 200,000
language_table_blocktoabsolute_oracle_sim | sim | oracle | scripted | 200,000
language_table_blocktorelative_oracle_sim | sim | oracle | scripted | 200,000
language_table_separate_oracle_sim | sim | oracle | scripted | 200,000

### Paths

Dataset | Data Location
--------| --------------
language_table | [gs://gresearch/robotics/language_table](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table/0.0.1/)
language_table_sim | [gs://gresearch/robotics/language_table_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_sim/0.0.1/)
language_table_blocktoblock_sim | [gs://gresearch/robotics/language_table_blocktoblock_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_sim/0.0.1/)
language_table_blocktoblock_4block_sim | [gs://gresearch/robotics/language_table_blocktoblock_4block_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1/)
language_table_blocktoblock_oracle_sim | [gs://gresearch/robotics/language_table_blocktoblock_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1/)
language_table_blocktoblockrelative_oracle_sim | [gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblockrelative_oracle_sim/0.0.1/)
language_table_blocktoabsolute_oracle_sim | [gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1/)
language_table_blocktorelative_oracle_sim | [gs://gresearch/robotics/language_table_blocktorelative_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1/)
language_table_separate_oracle_sim | [gs://gresearch/robotics/language_table_separate_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_separate_oracle_sim/0.0.1/)

## Checkpoints

Name | Config | Checkpoint Location
-----| -------| -------------------
BC+ResNet Sim| language_table/train/configs/language_table_resnet_sim_local.py | [gs://gresearch/robotics/language_table_checkpoints/bc_resnet_sim_checkpoint_955000](https://storage.googleapis.com/gresearch/robotics/language_table_checkpoints/bc_resnet_sim_checkpoint_955000)

## LaMer Integration

The `language_table/lamer/` module integrates this environment into the
[LaMer](https://github.com/mlbio-epfl/LaMer) meta-RL training framework.
Because the two codebases have incompatible dependencies (`gym==0.23` here vs
`gym==0.26.2` in LaMer), the environments run in a **separate process** and
communicate with LaMer over TCP using a pickle-based protocol.

### Architecture

```
LaMer process (lamer venv)              language-table process (ltvenv)
┌──────────────────────────┐            ┌──────────────────────────────┐
│  PPO Trainer             │            │  server_main.py              │
│    └─ RemoteEnvManager ──┼── TCP ───► │    └─ EnvServer              │
│       (client.py)        │            │       └─ LTEnvironmentMgr    │
│                          │            │           └─ MultiProcessEnv │
│                          │            │               └─ Ray workers │
│                          │            │                  └─ PyBullet │
└──────────────────────────┘            └──────────────────────────────┘
```


### Running the test suite

```bash
# Run default tests (single env, state-to-text, env manager, parallel)
python -m language_table.lamer.test_standalone --num_envs 4 --num_steps 50

# Run scaling sweep (1/2/4/8 envs throughput)
python -m language_table.lamer.test_standalone --scaling_sweep

# Run server round-trip test
python -m language_table.lamer.test_standalone --test_server
```

Test renders (PNG frames + MP4 video) are saved to `/tmp/lt_renders/` by default
(override with `--output_dir`).

### Starting the remote env server (for LaMer)

```bash
# Training server (e.g. 8 envs on port 50051)
ltvenv/bin/python -m language_table.lamer.server_main \
    --host 0.0.0.0 --port 50051 \
    --num_envs 8 --block_mode BLOCK_4 \
    --max_inner_steps 100 --num_attempts 3

# Validation server (e.g. 16 envs on port 50052)
ltvenv/bin/python -m language_table.lamer.server_main \
    --host 0.0.0.0 --port 50052 \
    --num_envs 16 --block_mode BLOCK_4 \
    --max_inner_steps 100 --num_attempts 3
```

Then point LaMer at these servers (see the LaMer README for the training command).

### Module overview

| File | Purpose |
|------|---------|
| `lamer/envs.py` | `LanguageTableWorker` (Ray actor) and `LanguageTableMultiProcessEnv` (vectorised wrapper) |
| `lamer/state_to_text.py` | Converts observation dicts to natural-language text for the LLM |
| `lamer/env_manager.py` | `LanguageTableEnvironmentManager` with VLA inner loop (random actions by default) |
| `lamer/protocol.py` | TCP wire protocol (length-prefixed pickle, pure stdlib) |
| `lamer/server.py` | `EnvServer` that wraps the environment manager |
| `lamer/server_main.py` | CLI entrypoint for the server |
| `lamer/test_standalone.py` | Test/demo script with rendering and video output |
| `lamer/test_connection.py` | Connection test for verifying server reachability (TCP + protocol) |

### Connection testing

Use `test_connection.py` to verify that the env servers are reachable and
responding correctly. This is especially useful on SLURM where networking
between nodes can be tricky.

```bash
# Basic connectivity (TCP + get_properties only)
ltvenv/bin/python -m language_table.lamer.test_connection \
    --host localhost --port 50051

# Full protocol test (reset, step, restart, reflect cycle)
ltvenv/bin/python -m language_table.lamer.test_connection \
    --host localhost --port 50051 --full

# Test both train + val servers
ltvenv/bin/python -m language_table.lamer.test_connection \
    --host localhost --port 50051 --val_port 50052 --full

# With latency benchmark
ltvenv/bin/python -m language_table.lamer.test_connection \
    --host localhost --port 50051 --full --latency
```

On SLURM, if the server runs on a different node, replace `localhost` with the
compute node hostname. Common troubleshooting:
- Check the port is listening: `ssh <node> 'ss -tlnp | grep 50051'`
- Test raw TCP: `nc -zv <node> 50051`
- Ensure no firewall blocks inter-node traffic on the chosen ports

### Meta-RL restart

`restart()` restores the **exact** simulation state from the last `reset()` using
PyBullet's `saveState`/`restoreState` plus cached Python-level task state
(instruction, block assignments, reward calculator RNG). This guarantees
bit-for-bit identical starting conditions across meta-RL attempts — all block
positions, robot joints, task instructions, and rendered images match exactly.

## Interactive Language: Talking to Robots in Real Time
[Project Website](https://interactive-language.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/pdf/2210.06407.pdf)

*Corey Lynch, Ayzaan Wahid, Jonathan Tompson, Tianli Ding, James Betker, Robert Baruch, Travis Armstrong, Pete Florence*

**Abstract.** We present a framework for building interactive, real-time, natural language-instructable robots in the real world, and we open source related assets (dataset, environment, benchmark, and policies). Trained with behavioral cloning on a dataset of hundreds of thousands of language-annotated trajectories, a produced policy can proficiently execute an order of magnitude more commands than previous works: specifically we estimate a 93.5% success rate on a set of 87,000 unique natural language strings specifying raw end-to-end visuolinguo-motor skills in the real world. We find that the same policy is capable of being guided by a human via real-time language to address a wide range of precise long-horizon rearrangement goals, e.g. "make a smiley face out of blocks". The dataset we release comprises nearly 600,000 language-labeled trajectories, an order of magnitude larger than prior available datasets. We hope the demonstrated results and associated assets enable further advancement of helpful, capable, natural-language-interactable robots.

## Note

This is not an officially supported Google product.
