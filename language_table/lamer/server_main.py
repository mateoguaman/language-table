"""
CLI launcher for the Language Table remote environment server.

Supports two modes:

1. Single-pool (original):
    ltvenv/bin/python -m language_table.lamer.server_main \
        --host 0.0.0.0 --port 50051 --num_envs 8 --split train \
        --reward_type multistep \
        --reward_kwargs '{"locations":["top_left"],"shapes":["moon"],"n_steps":2}'

2. Unified two-pool (shared VLA model, two ports):
    ltvenv/bin/python -m language_table.lamer.server_main \
        --unified \
        --host 0.0.0.0 --train_port 50051 --val_port 50052 \
        --train_num_envs 16 --train_group_n 8 \
        --val_num_envs 128 --val_group_n 1 \
        --reward_type multistep \
        --train_reward_kwargs '{...}' --val_reward_kwargs '{...}'

   In unified mode, a single process serves both train and val on
   separate TCP ports, sharing one LAVA model on one GPU with
   MEM_FRACTION=0.9. Since the PPO loop is sequential (rollout then
   validate), there is no concurrent GPU memory contention.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from language_table.environments.rewards.block2block import BlockToBlockReward
from language_table.environments.rewards.block2absolutelocation import BlockToAbsoluteLocationReward
from language_table.environments.rewards.block2relativelocation import BlockToRelativeLocationReward
from language_table.environments.rewards.block2block_relative_location import BlockToBlockRelativeLocationReward
from language_table.environments.rewards.point2block import PointToBlockReward
from language_table.environments.rewards.separate_blocks import SeparateBlocksReward
from language_table.environments.rewards.multistep_block_to_location import (
    make_multistep_reward,
)
from language_table.environments.rewards.composite import CompositeReward

REWARD_TYPES = {
    "composite": CompositeReward,
    "block2block": BlockToBlockReward,
    "block2absolutelocation": BlockToAbsoluteLocationReward,
    "block2relativelocation": BlockToRelativeLocationReward,
    "block2block_relative_location": BlockToBlockRelativeLocationReward,
    "point2block": PointToBlockReward,
    "separate_blocks": SeparateBlocksReward,
    "multistep": None,
    "custom": None,
    "none": None,
}

logger = logging.getLogger(__name__)


def _parse_reward_kwargs(reward_kwargs_json: Optional[str]) -> Dict[str, Any]:
    """Parse the `--reward_kwargs` JSON string into a dict."""
    if not reward_kwargs_json:
        return {}
    try:
        parsed = json.loads(reward_kwargs_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid --reward_kwargs JSON: {reward_kwargs_json!r} ({e})"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"--reward_kwargs must decode to a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def _build_reward(
    reward_type: str,
    reward_kwargs_json: Optional[str],
) -> Tuple[Optional[type], Dict[str, Any]]:
    """Build (reward_factory_cls, parsed_kwargs) from a reward_type and JSON.

    For ``multistep`` this calls ``make_multistep_reward(**kwargs)``.
    For ``custom`` and ``none`` the factory is ``None`` (env runs without
    a reward calculator; custom reward is emitted at the manager level).
    For langtable built-ins the factory is the class from ``REWARD_TYPES``;
    kwargs are parsed but currently unused by the built-ins.
    """
    kwargs = _parse_reward_kwargs(reward_kwargs_json)

    if reward_type == "multistep":
        factory = make_multistep_reward(**kwargs)
        logger.info("Built multistep reward with kwargs=%s", kwargs)
        return factory, kwargs
    if reward_type == "custom":
        logger.info("reward_type=custom: no env reward; manager-level custom reward active")
        return None, kwargs
    if reward_type == "none":
        return None, kwargs
    if reward_type not in REWARD_TYPES:
        raise ValueError(f"Unknown reward_type: {reward_type!r}")
    factory = REWARD_TYPES[reward_type]
    if kwargs:
        logger.warning(
            "reward_kwargs provided for reward_type=%s but built-in rewards "
            "do not accept kwargs today; ignoring: %s",
            reward_type, kwargs,
        )
    return factory, kwargs


def _load_vla_policy(checkpoint_path, preprocess_mode="original"):
    """Load a shared LAVA VLA policy from a checkpoint path."""
    from .lava_policy import LAVAPolicy

    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_prefix = (
        os.path.basename(checkpoint_path).rsplit("_", 1)[0] + "_"
    )
    logger.info(
        "Loading LAVA policy from %s (prefix=%s, preprocess_mode=%s)",
        checkpoint_dir, checkpoint_prefix, preprocess_mode,
    )
    policy = LAVAPolicy(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        preprocess_mode=preprocess_mode,
    )
    logger.info("LAVA policy loaded successfully.")
    return policy


def _create_env_pool(num_envs, group_n, block_mode, reward_factory_cls,
                     seed, render_obs, return_full_state):
    """Create a Ray-parallelized environment pool."""
    from .envs import LanguageTableMultiProcessEnv

    return LanguageTableMultiProcessEnv(
        num_envs=num_envs,
        block_mode=block_mode,
        reward_factory_cls=reward_factory_cls,
        seed=seed,
        group_n=group_n,
        return_full_state=return_full_state,
        render_obs=render_obs,
    )


def _create_manager(
    envs,
    args,
    vla_policy,
    split,
    group_n,
    reward_type: str,
    reward_kwargs: Dict[str, Any],
):
    """Create an environment manager (LAVA or Gemini)."""
    if args.policy == "gemini":
        from .gemini_policy import GeminiPolicy
        from .gemini_env_manager import LanguageTableEnvironmentManager

        policy = GeminiPolicy(
            timeout=args.gemini_timeout,
            split=split,
        )
        return LanguageTableEnvironmentManager(
            envs=envs,
            policy=policy,
            num_attempts=args.num_attempts,
            max_turns=args.max_turns,
            do_reflection=args.do_reflection,
            max_inner_steps=args.max_inner_steps,
            include_rgb=args.include_rgb,
        )

    from .lava_env_manager import LanguageTableEnvironmentManager

    n_steps = 1
    if reward_type == "multistep":
        try:
            n_steps = int(reward_kwargs.get("n_steps", 1))
        except (TypeError, ValueError):
            n_steps = 1

    custom_task_provider = None
    if reward_type == "custom":
        from .custom_task_provider import build_task_provider
        custom_task_provider = build_task_provider(reward_kwargs, group_n=group_n)

    return LanguageTableEnvironmentManager(
        envs=envs,
        num_attempts=args.num_attempts,
        max_turns=args.max_turns,
        do_reflection=args.do_reflection,
        max_inner_steps=args.max_inner_steps,
        vla_policy=vla_policy,
        include_rgb=args.include_rgb,
        n_steps=n_steps,
        custom_task_provider=custom_task_provider,
    )


def _run_single(args):
    """Original single-pool mode."""
    from .envs import LanguageTableMultiProcessEnv
    from .server import EnvServer

    block_mode = args.block_mode

    if args.no_reward:
        reward_factory_cls = None
        reward_kwargs: Dict[str, Any] = {}
    else:
        reward_factory_cls, reward_kwargs = _build_reward(
            args.reward_type, args.reward_kwargs,
        )

    render_obs = not args.no_render or args.vla_checkpoint is not None

    envs = LanguageTableMultiProcessEnv(
        num_envs=args.num_envs,
        block_mode=block_mode,
        reward_factory_cls=reward_factory_cls,
        seed=args.seed,
        group_n=args.group_n,
        return_full_state=not args.no_full_state,
        render_obs=render_obs,
    )

    logger.info("Warming up %d workers (test reset)...", args.num_envs * args.group_n)
    envs.reset()
    logger.info("All workers ready.")

    vla_policy = None
    if args.vla_checkpoint:
        vla_policy = _load_vla_policy(
            args.vla_checkpoint, args.preprocess_mode)

    manager = _create_manager(
        envs, args, vla_policy, args.split, args.group_n,
        args.reward_type, reward_kwargs,
    )

    server = EnvServer(manager, host=args.host, port=args.port)
    server.serve()


def _run_unified(args):
    """Unified two-pool mode: one VLA, two env pools, two ports."""
    from .server import MultiPoolEnvServer

    if args.no_reward:
        train_reward_cls = None
        val_reward_cls = None
        train_reward_kwargs: Dict[str, Any] = {}
        val_reward_kwargs: Dict[str, Any] = {}
    else:
        train_reward_cls, train_reward_kwargs = _build_reward(
            args.reward_type, args.train_reward_kwargs,
        )
        val_reward_cls, val_reward_kwargs = _build_reward(
            args.reward_type, args.val_reward_kwargs,
        )

    render_obs = not args.no_render or args.vla_checkpoint is not None

    # Load one shared VLA model
    vla_policy = None
    if args.vla_checkpoint:
        vla_policy = _load_vla_policy(
            args.vla_checkpoint, args.preprocess_mode)

    # Create train env pool
    logger.info(
        "Creating train pool: %d envs x group_n=%d = %d workers (block_mode=%s)",
        args.train_num_envs, args.train_group_n,
        args.train_num_envs * args.train_group_n,
        args.train_block_mode,
    )
    train_envs = _create_env_pool(
        num_envs=args.train_num_envs,
        group_n=args.train_group_n,
        block_mode=args.train_block_mode,
        reward_factory_cls=train_reward_cls,
        seed=args.seed,
        render_obs=render_obs,
        return_full_state=not args.no_full_state,
    )

    # Create val env pool (offset seed to avoid overlap)
    val_seed = args.seed + args.train_num_envs * args.train_group_n + 10000
    logger.info(
        "Creating val pool: %d envs x group_n=%d = %d workers (block_mode=%s)",
        args.val_num_envs, args.val_group_n,
        args.val_num_envs * args.val_group_n,
        args.val_block_mode,
    )
    val_envs = _create_env_pool(
        num_envs=args.val_num_envs,
        group_n=args.val_group_n,
        block_mode=args.val_block_mode,
        reward_factory_cls=val_reward_cls,
        seed=val_seed,
        render_obs=render_obs,
        return_full_state=not args.no_full_state,
    )

    # Warm up both pools
    total_workers = (args.train_num_envs * args.train_group_n +
                     args.val_num_envs * args.val_group_n)
    logger.info("Warming up %d workers (test reset)...", total_workers)
    train_envs.reset()
    val_envs.reset()
    logger.info("All workers ready.")

    # Create managers sharing the same VLA policy
    train_manager = _create_manager(
        train_envs, args, vla_policy, "train", args.train_group_n,
        args.reward_type, train_reward_kwargs,
    )
    val_manager = _create_manager(
        val_envs, args, vla_policy, "val", args.val_group_n,
        args.reward_type, val_reward_kwargs,
    )

    server = MultiPoolEnvServer(
        train_manager, val_manager,
        host=args.host,
        train_port=args.train_port,
        val_port=args.val_port,
    )
    server.serve()


def main():
    parser = argparse.ArgumentParser(description="Language Table remote env server")

    # Mode selection
    parser.add_argument("--unified", action="store_true",
                        help="Run in unified mode: one process, two ports, "
                             "shared VLA model. Use --train_port/--val_port "
                             "and --train_num_envs/--val_num_envs.")

    # Common args
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--block_mode", type=str, default="BLOCK_4")
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=1)
    parser.add_argument("--max_inner_steps", type=int, default=100)
    parser.add_argument("--do_reflection", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward_type", type=str, default="block2block",
                        choices=list(REWARD_TYPES.keys()))
    parser.add_argument("--reward_kwargs", type=str, default="{}",
                        help="JSON object of kwargs forwarded to the reward "
                             "factory. Schema depends on --reward_type. "
                             "Used in single-pool mode.")
    parser.add_argument("--no_reward", action="store_true")
    parser.add_argument("--no_full_state", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--include_rgb", action="store_true")
    parser.add_argument("--vla_checkpoint", type=str, default=None)
    parser.add_argument("--preprocess_mode", type=str, default="batched_tf",
                        choices=["original", "batched_tf", "jax_gpu"],
                        help="Image preprocessing strategy for LAVA _build_batch")
    parser.add_argument("--policy", type=str, default="lava",
                        choices=["lava", "gemini"])
    parser.add_argument("--gemini_timeout", type=float, default=30.0)

    # Single-pool mode args
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--group_n", type=int, default=1)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"])

    # Unified mode args
    parser.add_argument("--train_port", type=int, default=50051)
    parser.add_argument("--val_port", type=int, default=50052)
    parser.add_argument("--train_num_envs", type=int, default=16)
    parser.add_argument("--train_group_n", type=int, default=8)
    parser.add_argument("--val_num_envs", type=int, default=128)
    parser.add_argument("--val_group_n", type=int, default=1)
    parser.add_argument("--train_block_mode", type=str, default="BLOCK_4",
                        help="Block mode for the train pool (unified mode).")
    parser.add_argument("--val_block_mode", type=str, default="BLOCK_4",
                        help="Block mode for the val pool (unified mode).")
    parser.add_argument("--train_reward_kwargs", type=str, default="{}",
                        help="JSON kwargs for the train pool reward factory "
                             "(unified mode).")
    parser.add_argument("--val_reward_kwargs", type=str, default="{}",
                        help="JSON kwargs for the val pool reward factory "
                             "(unified mode).")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.unified:
        logger.info(
            "Unified server config: host=%s train_port=%d val_port=%d "
            "train_envs=%dx%d val_envs=%dx%d policy=%s max_steps=%d "
            "reward_type=%s train_block_mode=%s val_block_mode=%s",
            args.host, args.train_port, args.val_port,
            args.train_num_envs, args.train_group_n,
            args.val_num_envs, args.val_group_n,
            args.policy, args.max_inner_steps,
            args.reward_type, args.train_block_mode, args.val_block_mode,
        )
        _run_unified(args)
    else:
        logger.info(
            "Server config: host=%s port=%d envs=%d group_n=%d blocks=%s "
            "policy=%s split=%s max_steps=%d attempts=%d turns=%d "
            "reward_type=%s",
            args.host, args.port, args.num_envs, args.group_n,
            args.block_mode, args.policy, args.split,
            args.max_inner_steps, args.num_attempts, args.max_turns,
            args.reward_type,
        )
        _run_single(args)


if __name__ == "__main__":
    main()
