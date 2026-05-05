import argparse
import json
import os
from pathlib import Path

import numpy as np
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block1_to_corner
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import point2block
from language_table.environments.rewards import separate_blocks
from language_table.lamer.lava_policy import LAVAPolicy
from language_table.lamer.smolvla_policy import SmolVLAPolicy
from tf_agents.environments import gym_wrapper


REWARD_FACTORIES = {
    "blocktoblock": block2block.BlockToBlockReward,
    "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
    "blocktoblockrelativelocation": block2block_relative_location.BlockToBlockRelativeLocationReward,
    "blocktorelativelocation": block2relativelocation.BlockToRelativeLocationReward,
    "separate": separate_blocks.SeparateBlocksReward,
    "block1tocorner": block1_to_corner.Block1ToCornerLocationReward,
    "point2block": point2block.PointToBlockReward,
}


def _decode_instruction(obs):
    raw = obs.get("instruction")
    if raw is None:
        return ""
    if hasattr(raw, "ndim") and raw.ndim == 2:
        raw = raw[-1]
    return language_table.LanguageTable.decode_instruction(raw)


def _get_raw_env(env):
    cur = env
    while True:
        if isinstance(cur, language_table.LanguageTable):
            return cur
        nxt = (
            getattr(cur, "_env", None)
            or getattr(cur, "env", None)
            or getattr(cur, "gym", None)
            or getattr(cur, "_gym_env", None)
        )
        if nxt is None or nxt is cur:
            return cur
        cur = nxt


def _make_env(seed, block_mode, reward_type):
    env = language_table.LanguageTable(
        block_mode=getattr(blocks.LanguageTableBlockVariants, block_mode),
        reward_factory=REWARD_FACTORIES[reward_type],
        render_text_in_image=False,
        seed=seed,
    )
    env = gym_wrapper.GymWrapper(env)
    if not hasattr(env, "get_control_frequency"):
        env.get_control_frequency = lambda: env._control_frequency
    return env


def _make_policy(args):
    if args.policy == "smolvla":
        return SmolVLAPolicy(
            checkpoint_path=args.checkpoint,
            host=args.host,
            port=args.port,
            server_log=args.server_log,
        )

    checkpoint_dir, checkpoint_prefix = _resolve_lava_checkpoint(
        args.checkpoint, args.lava_checkpoint_prefix
    )
    return LAVAPolicy(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        preprocess_mode=args.lava_preprocess_mode,
    )


def _resolve_lava_checkpoint(checkpoint, checkpoint_prefix):
    path = Path(os.path.expanduser(checkpoint))
    if path.is_file():
        return str(path.parent), path.name
    return str(path), checkpoint_prefix


def evaluate(args):
    policy = _make_policy(args)
    results = []
    successes = 0

    try:
        for trial in range(args.num_trials):
            seed = args.seed + trial
            env = _make_env(seed, args.block_mode, args.reward_type)
            raw_env = _get_raw_env(env)
            policy.reset(num_envs=1)
            env.seed(seed)
            time_step = env.reset()
            obs = time_step.observation
            instruction = _decode_instruction(obs)
            success = False
            steps_taken = 0

            for step in range(args.max_steps):
                action = policy.predict(
                    goals=[instruction],
                    obs_list=[obs],
                    active_mask=np.array([True], dtype=bool),
                )[0]
                time_step = env.step(action)
                obs = time_step.observation
                steps_taken = step + 1

                if bool(raw_env.succeeded) or time_step.is_last():
                    success = bool(raw_env.succeeded) or time_step.is_last()
                    break

            if success:
                successes += 1

            result = {
                "trial": trial,
                "seed": seed,
                "success": bool(success),
                "steps": steps_taken,
                "instruction": instruction,
            }
            results.append(result)
            print(
                f"trial={trial:02d} seed={seed} success={int(success)} "
                f"steps={steps_taken} instruction={instruction!r}",
                flush=True,
            )

        summary = {
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "reward_type": args.reward_type,
            "block_mode": args.block_mode,
            "seed": args.seed,
            "num_trials": args.num_trials,
            "max_steps": args.max_steps,
            "successes": successes,
            "success_rate": successes / args.num_trials if args.num_trials else 0.0,
            "trials": results,
        }
        print(json.dumps(summary, indent=2))

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2) + "\n")

        return summary
    finally:
        close = getattr(policy, "close", None)
        if close is not None:
            close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a low-level Language Table policy over seeded trials."
    )
    parser.add_argument("--policy", choices=["lava", "smolvla"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument(
        "--block_mode",
        choices=["BLOCK_4", "BLOCK_8", "BLOCK_4_WPOLE", "BLOCK_8_WPOLE"],
        default="BLOCK_4",
    )
    parser.add_argument(
        "--reward_type",
        choices=sorted(REWARD_FACTORIES),
        default="blocktoabsolutelocation",
    )
    parser.add_argument("--output_json", default="")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50100)
    parser.add_argument("--server_log", default="/tmp/smolvla_low_level_eval.log")

    parser.add_argument(
        "--lava_checkpoint_prefix",
        default="bc_resnet_sim_checkpoint_",
        help="Prefix to use when --checkpoint is a LAVA checkpoint directory.",
    )
    parser.add_argument(
        "--lava_preprocess_mode",
        choices=["original", "batched_tf", "jax_gpu", "jax_fused"],
        default="jax_gpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
