# Installation

This repo uses two Python virtualenvs because the simulator and the LeRobot
training stack have incompatible dependencies:

| venv | Python | Used for |
|------|--------|----------|
| `ltvenv`     | 3.10 | PyBullet sim, JAX, `tf_agents`, RLDS dataset loaders, eval client, LAVA inference |
| `lerobotenv` | 3.12 | LeRobot policies (SmolVLA, π0, ACT, Diffusion), HF datasets, training, policy server |

Both are managed with [uv](https://docs.astral.sh/uv/). Run all commands from
the repo root.

## `ltvenv` — simulator + dataset tooling

```bash
uv venv --python 3.10 ./ltvenv
source ./ltvenv/bin/activate
uv pip install -r ./requirements.txt
uv pip install --no-deps git+https://github.com/google-research/scenic.git@ae21d9e884015aa7bc7cf1d489af53d16c249726
export PYTHONPATH=${PWD}:$PYTHONPATH
```

Persist the `PYTHONPATH` export with:

```bash
echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
```

## `lerobotenv` — LeRobot training and inference

```bash
uv venv --python 3.12 ./lerobotenv
source ./lerobotenv/bin/activate
uv pip install 'lerobot[pi,smolvla]>=0.5.1' av accelerate
```

`pi` is the π0 / π0.5 extra and `smolvla` is the SmolVLA extra; ACT and
Diffusion ship with the base install. `av` provides the PyAV video backend
used by the recipes (`--dataset.video_backend=pyav`).

## HuggingFace authentication

Required for pulling the SmolVLA base model on first training run, the
SmolVLA finetune at `mateoguaman/smolvla_lt_combined_sim_93185` used by the
interactive notebook, and pushing checkpoints during training:

```bash
huggingface-cli login
```

## Cluster setup (Tillicum)

Use conda instead of uv on the cluster:

```bash
cp training/.env.user.template training/.env.user
# edit training/.env.user and fill in WANDB_API_KEY and HF_TOKEN
training/setup_cluster_env.sh
```

This script creates the `lerobot` conda env with the same packages as
`lerobotenv`, sources `training/.env.tillicum` for GPFS paths, and
provisions the output / checkpoint / cache directories.

## Optional: `ltvenvtrain` for legacy JAX training

The original Google `language_table/train/main.py` script uses pinned
versions in `requirements_static.txt`. Skip unless you specifically want
the legacy BC-ResNet trainer:

```bash
uv venv --python 3.10 ./ltvenvtrain
source ./ltvenvtrain/bin/activate
uv pip install --no-deps -r ./requirements_static.txt
export PYTHONPATH=${PWD}:$PYTHONPATH
```

## Verification

```bash
./ltvenv/bin/python -m language_table.lamer.test_standalone --num_envs 2 --num_steps 10
./lerobotenv/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.cuda.is_available())"
```
