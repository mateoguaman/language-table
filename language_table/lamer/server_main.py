"""
CLI launcher for the Language Table remote environment server.

Supports two modes:

1. Single-pool (original):
    ltvenv/bin/python -m language_table.lamer.server_main \
        --host 0.0.0.0 --port 50051 --num_envs 8 --split train

2. Unified two-pool (shared VLA model, two ports):
    ltvenv/bin/python -m language_table.lamer.server_main \
        --unified \
        --host 0.0.0.0 --train_port 50051 --val_port 50052 \
        --train_num_envs 16 --train_group_n 8 \
        --val_num_envs 128 --val_group_n 1

   In unified mode, a single process serves both train and val on
   separate TCP ports, sharing one LAVA model on one GPU with
   MEM_FRACTION=0.9. Since the PPO loop is sequential (rollout then
   validate), there is no concurrent GPU memory contention.
"""

import argparse
import logging
import os

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
    "none": None,
}

logger = logging.getLogger(__name__)


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


def _create_manager(envs, args, vla_policy, split, group_n):
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
    else:
        from .lava_env_manager import LanguageTableEnvironmentManager

        n_steps = getattr(args, 'task_n_steps', 1)
        return LanguageTableEnvironmentManager(
            envs=envs,
            num_attempts=args.num_attempts,
            max_turns=args.max_turns,
            do_reflection=args.do_reflection,
            max_inner_steps=args.max_inner_steps,
            vla_policy=vla_policy,
            include_rgb=args.include_rgb,
            split=split,
            n_steps=n_steps,
            benchmark_timing=args.benchmark_timing,
            benchmark_trace_inner_steps=args.benchmark_trace_inner_steps,
        )


def _run_single(args):
    """Original single-pool mode."""
    from .envs import LanguageTableMultiProcessEnv
    from .server import EnvServer

    block_mode = args.block_mode

    if args.no_reward:
        reward_factory_cls = None
    elif args.reward_type == "multistep":
        locations = (args.task_locations.split(",")
                     if args.task_locations else None)
        colors = (args.task_colors.split(",")
                  if args.task_colors else None)
        shapes = (args.task_shapes.split(",")
                  if args.task_shapes else None)
        n_steps = args.task_n_steps

        reward_factory_cls = make_multistep_reward(
            locations=locations, shapes=shapes, colors=colors,
            n_steps=n_steps)
        logger.info(
            "Multistep reward: locations=%s shapes=%s colors=%s "
            "n_steps=%d block_mode=%s",
            locations, shapes, colors, n_steps, block_mode)
    else:
        reward_factory_cls = REWARD_TYPES[args.reward_type]

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

    manager = _create_manager(envs, args, vla_policy, args.split, args.group_n)

    server = EnvServer(manager, host=args.host, port=args.port)
    server.serve()


def _run_unified(args):
    """Unified two-pool mode: one VLA, two env pools, two ports."""
    from .server import MultiPoolEnvServer

    if args.no_reward:
        train_reward_cls = None
        val_reward_cls = None
    elif args.reward_type == "multistep":
        locations = (args.task_locations.split(",")
                     if args.task_locations else None)
        colors = (args.task_colors.split(",")
                  if args.task_colors else None)
        shapes = (args.task_shapes.split(",")
                  if args.task_shapes else None)
        n_steps = args.task_n_steps

        train_reward_cls = make_multistep_reward(
            locations=locations, shapes=shapes, colors=colors,
            n_steps=n_steps)
        val_reward_cls = train_reward_cls
        logger.info(
            "Multistep reward (unified): locations=%s shapes=%s colors=%s "
            "n_steps=%d", locations, shapes, colors, n_steps)
    else:
        train_reward_cls = REWARD_TYPES[args.reward_type]
        val_reward_cls = REWARD_TYPES[args.reward_type]

    render_obs = not args.no_render or args.vla_checkpoint is not None

    # Load one shared VLA model
    vla_policy = None
    if args.vla_checkpoint:
        vla_policy = _load_vla_policy(
            args.vla_checkpoint, args.preprocess_mode)

    # Create train env pool
    logger.info(
        "Creating train pool: %d envs × group_n=%d = %d workers",
        args.train_num_envs, args.train_group_n,
        args.train_num_envs * args.train_group_n,
    )
    train_envs = _create_env_pool(
        num_envs=args.train_num_envs,
        group_n=args.train_group_n,
        block_mode=args.block_mode,
        reward_factory_cls=train_reward_cls,
        seed=args.seed,
        render_obs=render_obs,
        return_full_state=not args.no_full_state,
    )

    # Create val env pool (offset seed to avoid overlap)
    val_seed = args.seed + args.train_num_envs * args.train_group_n + 10000
    logger.info(
        "Creating val pool: %d envs × group_n=%d = %d workers",
        args.val_num_envs, args.val_group_n,
        args.val_num_envs * args.val_group_n,
    )
    val_envs = _create_env_pool(
        num_envs=args.val_num_envs,
        group_n=args.val_group_n,
        block_mode=args.block_mode,
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
        train_envs, args, vla_policy, "train", args.train_group_n)
    val_manager = _create_manager(
        val_envs, args, vla_policy, "val", args.val_group_n)

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
    parser.add_argument("--no_reward", action="store_true")
    parser.add_argument("--no_full_state", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--include_rgb", action="store_true")
    parser.add_argument("--vla_checkpoint", type=str, default=None)
    parser.add_argument("--preprocess_mode", type=str, default="batched_tf",
                        choices=["original", "batched_tf", "jax_gpu", "jax_fused"],
                        help="Image preprocessing strategy for LAVA _build_batch")
    parser.add_argument("--policy", type=str, default="lava",
                        choices=["lava", "gemini"])
    parser.add_argument("--gemini_timeout", type=float, default=30.0)
    parser.add_argument("--benchmark_timing", action="store_true",
                        help="Attach structured timing metadata to env responses.")
    parser.add_argument("--benchmark_trace_inner_steps", action="store_true",
                        help="Include per-inner-step timing records in benchmark metadata.")

    # Multistep reward task configuration
    parser.add_argument("--task_locations", type=str, default=None,
                        help="Comma-separated location names from ABSOLUTE_LOCATIONS")
    parser.add_argument("--task_colors", type=str, default=None,
                        help="Comma-separated color names (blocks described by color)")
    parser.add_argument("--task_shapes", type=str, default=None,
                        help="Comma-separated shape names (blocks described by shape)")
    parser.add_argument("--task_n_steps", type=int, default=2,
                        help="Number of block-to-location sub-goals per episode")

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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.unified:
        logger.info(
            "Unified server config: host=%s train_port=%d val_port=%d "
            "train_envs=%d×%d val_envs=%d×%d policy=%s max_steps=%d",
            args.host, args.train_port, args.val_port,
            args.train_num_envs, args.train_group_n,
            args.val_num_envs, args.val_group_n,
            args.policy, args.max_inner_steps,
        )
        _run_unified(args)
    else:
        logger.info(
            "Server config: host=%s port=%d envs=%d group_n=%d blocks=%s "
            "policy=%s split=%s max_steps=%d attempts=%d turns=%d",
            args.host, args.port, args.num_envs, args.group_n,
            args.block_mode, args.policy, args.split,
            args.max_inner_steps, args.num_attempts, args.max_turns,
        )
        _run_single(args)


if __name__ == "__main__":
    main()
