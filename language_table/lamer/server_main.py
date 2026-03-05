"""
CLI launcher for the Language Table remote environment server.

Usage:
    ltvenv/bin/python -m language_table.lamer.server_main \
        --host 0.0.0.0 --port 50051 --num_envs 8 --block_mode BLOCK_4
"""

import argparse
import logging

from language_table.environments.rewards.block2block import BlockToBlockReward

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
    parser.add_argument("--no_reward", action="store_true",
                        help="Run without reward (episodes never terminate via reward)")
    parser.add_argument("--no_full_state", action="store_true",
                        help="Return filtered obs instead of full state (omits block positions)")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip RGB rendering in _compute_state() for max throughput. "
                             "Only use when the inner-loop policy doesn't need images.")
    parser.add_argument("--include_rgb", action="store_true",
                        help="Include RGB images in observations sent over the wire. "
                             "Implies rendering is enabled (overrides --no_render).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from .envs import LanguageTableMultiProcessEnv
    from .env_manager import LanguageTableEnvironmentManager
    from .server import EnvServer

    reward_factory_cls = None if args.no_reward else BlockToBlockReward

    # include_rgb implies rendering must be on
    render_obs = not args.no_render or args.include_rgb

    envs = LanguageTableMultiProcessEnv(
        num_envs=args.num_envs,
        block_mode=args.block_mode,
        reward_factory_cls=reward_factory_cls,
        seed=args.seed,
        group_n=args.group_n,
        return_full_state=not args.no_full_state,
        render_obs=render_obs,
        include_rgb=args.include_rgb,
    )

    # Warm up: do a test reset to ensure all Ray workers are alive and
    # PyBullet is initialized before we start accepting client connections.
    logger.info("Warming up %d workers (test reset)...", args.num_envs * args.group_n)
    envs.reset()
    logger.info("All workers ready.")

    manager = LanguageTableEnvironmentManager(
        envs=envs,
        num_attempts=args.num_attempts,
        max_turns=args.max_turns,
        do_reflection=args.do_reflection,
        max_inner_steps=args.max_inner_steps,
    )

    server = EnvServer(manager, host=args.host, port=args.port)
    server.serve()


if __name__ == "__main__":
    main()
