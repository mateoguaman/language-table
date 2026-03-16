"""
CLI launcher for the Language Table remote environment server.

Usage:
    ltvenv/bin/python -m language_table.lamer.server_main \
        --host 0.0.0.0 --port 50051 --num_envs 8 --block_mode BLOCK_4
"""

import argparse
import logging

from language_table.environments.rewards.block2block import BlockToBlockReward
from language_table.environments.rewards.block2absolutelocation import BlockToAbsoluteLocationReward
from language_table.environments.rewards.block2relativelocation import BlockToRelativeLocationReward
from language_table.environments.rewards.block2block_relative_location import BlockToBlockRelativeLocationReward
from language_table.environments.rewards.point2block import PointToBlockReward
from language_table.environments.rewards.separate_blocks import SeparateBlocksReward
from language_table.environments.rewards.composite import CompositeReward

REWARD_TYPES = {
    "composite": CompositeReward,
    "block2block": BlockToBlockReward,
    "block2absolutelocation": BlockToAbsoluteLocationReward,
    "block2relativelocation": BlockToRelativeLocationReward,
    "block2block_relative_location": BlockToBlockRelativeLocationReward,
    "point2block": PointToBlockReward,
    "separate_blocks": SeparateBlocksReward,
    "none": None,
}

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Language Table remote env server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--group_n", type=int, default=1)
    parser.add_argument("--block_mode", type=str, default="BLOCK_4")
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=1)
    parser.add_argument("--max_inner_steps", type=int, default=100)
    parser.add_argument("--do_reflection", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward_type", type=str, default="block2block",
                        choices=list(REWARD_TYPES.keys()),
                        help="Reward type: 'composite' (all tasks), "
                             "or a specific task family, or 'none'")
    parser.add_argument("--no_reward", action="store_true",
                        help="(Deprecated, use --reward_type=none) "
                             "Run without reward")
    parser.add_argument("--no_full_state", action="store_true",
                        help="Return filtered obs instead of full state (omits block positions)")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip RGB rendering in _compute_state() for max throughput. "
                             "Only use when the inner-loop policy doesn't need images.")
    parser.add_argument("--include_rgb", action="store_true",
                        help="Include RGB images in responses sent over TCP to "
                             "the LLM client. Only needed if the outer-loop LLM "
                             "requires images. The inner-loop VLA gets images "
                             "directly from the Ray workers via render_obs.")
    parser.add_argument("--vla_checkpoint", type=str, default=None,
                        help="Full path to LAVA Flax checkpoint file (e.g. "
                             "/path/to/checkpoints/bc_resnet_sim_checkpoint_955000). "
                             "When set, the pre-trained LAVA policy handles inner-loop "
                             "actions instead of random actions. Implies rendering "
                             "is enabled (the VLA needs images).")
    parser.add_argument("--policy", type=str, default="lava",
                        choices=["lava", "gemini"],
                        help="Inner-loop policy type. 'lava' uses the LAVA VLA "
                             "(requires --vla_checkpoint). 'gemini' uses the "
                             "Gemini API to translate action strings.")
    parser.add_argument("--gemini_max_output_tokens", type=int, default=1024,
                        help="Max output tokens per Gemini API request.")
    parser.add_argument("--gemini_timeout", type=float, default=30.0,
                        help="Per-request timeout in seconds for Gemini API calls.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from .envs import LanguageTableMultiProcessEnv
    from .server import EnvServer

    if args.no_reward:
        reward_factory_cls = None
    else:
        reward_factory_cls = REWARD_TYPES[args.reward_type]

    # VLA needs RGB from Ray workers, so force rendering on
    render_obs = not args.no_render or args.vla_checkpoint is not None

    envs = LanguageTableMultiProcessEnv(
        num_envs=args.num_envs,
        block_mode=args.block_mode,
        reward_factory_cls=reward_factory_cls,
        seed=args.seed,
        group_n=args.group_n,
        return_full_state=not args.no_full_state,
        render_obs=render_obs,
    )

    # Warm up: do a test reset to ensure all Ray workers are alive and
    # PyBullet is initialized before we start accepting client connections.
    logger.info("Warming up %d workers (test reset)...", args.num_envs * args.group_n)
    envs.reset()
    logger.info("All workers ready.")

    if args.policy == "gemini":
        from .gemini_policy import GeminiPolicy
        from .gemini_env_manager import LanguageTableEnvironmentManager

        logger.info(
            "Using Gemini policy (action string translation, "
            "max_output_tokens=%d, timeout=%.1fs)",
            args.gemini_max_output_tokens, args.gemini_timeout,
        )
        policy = GeminiPolicy(
            max_output_tokens=args.gemini_max_output_tokens,
            timeout=args.gemini_timeout,
        )
        manager = LanguageTableEnvironmentManager(
            envs=envs,
            policy=policy,
            num_attempts=args.num_attempts,
            max_turns=args.max_turns,
            do_reflection=args.do_reflection,
            max_inner_steps=args.max_inner_steps,
            include_rgb=args.include_rgb,
        )
    else:
        # LAVA VLA policy (or random-action fallback if no checkpoint)
        from .lava_env_manager import LanguageTableEnvironmentManager

        vla_policy = None
        if args.vla_checkpoint:
            import os
            from .lava_policy import LAVAPolicy

            checkpoint_dir = os.path.dirname(args.vla_checkpoint)
            checkpoint_prefix = (
                os.path.basename(args.vla_checkpoint).rsplit("_", 1)[0] + "_"
            )
            logger.info(
                "Loading LAVA policy from %s (prefix=%s)",
                checkpoint_dir, checkpoint_prefix,
            )
            vla_policy = LAVAPolicy(
                checkpoint_dir=checkpoint_dir,
                checkpoint_prefix=checkpoint_prefix,
            )
            logger.info("LAVA policy loaded successfully.")

        manager = LanguageTableEnvironmentManager(
            envs=envs,
            num_attempts=args.num_attempts,
            max_turns=args.max_turns,
            do_reflection=args.do_reflection,
            max_inner_steps=args.max_inner_steps,
            vla_policy=vla_policy,
            include_rgb=args.include_rgb,
        )

    server = EnvServer(manager, host=args.host, port=args.port)
    server.serve()


if __name__ == "__main__":
    main()
