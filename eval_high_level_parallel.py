import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import ray

from language_table.environments.rewards import block2absolutelocation
from language_table.lamer.envs import LanguageTableMultiProcessEnv
from language_table.lamer.frame_annotator import annotate_frame
from language_table.lamer.human_policy import HumanPolicy
from language_table.lamer.lava_policy import LAVAPolicy
from language_table.lamer.smolvla_policy import SmolVLAPolicy
from language_table.lamer.vllm_policy import VLLMPolicy


DEFAULT_TASK = "arrange the blocks into the tetris/tetromino shape: T"


class ScriptedHighLevelPolicy:
    def __init__(self, instruction):
        self.instruction = instruction

    def reset(self):
        pass

    def step(self, image):
        del image
        return self.instruction


def _make_high_level_prompt(task):
    return f"""
You are controlling a robot that arranges objects into a target shape.

Your job is to output exactly one next action.

Rules:
- Output only one XML tag.
- Do not output multiple actions.
- If the target shape is already complete, output exactly:
<atomic_instruction>done</atomic_instruction>

Each action must use this exact format:
<atomic_instruction>push the color shape to the location</atomic_instruction>

Allowed colors:
red, blue, green, yellow

Allowed shapes:
moon, cube, star, pentagon

Allowed locations:
top left corner, top center, top right corner,
center left, center, center right,
bottom left corner, bottom center, bottom right corner

Choose the single most useful next action that moves the current arrangement closer to completing the task.

Task:
{task}"""


def _parse_seeds(args):
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no seeds were parsed")
        return seeds
    return [args.seed + i for i in range(args.num_trials)]


def _make_low_level_policy(args):
    if args.low_level_policy == "smolvla":
        return SmolVLAPolicy(
            checkpoint_path=args.low_level_checkpoint,
            host=args.host,
            port=args.port,
            server_log=args.server_log,
            use_batch=not args.disable_low_level_batch,
        )

    return LAVAPolicy(
        checkpoint_dir=args.lava_checkpoint_dir,
        checkpoint_prefix=args.lava_checkpoint_prefix,
        preprocess_mode=args.lava_preprocess_mode,
    )


def _make_high_level_policies(args, prompt, num_envs):
    if args.policy == "vllm":
        return [
            VLLMPolicy(
                prompt=prompt,
                url=args.vllm_url,
                max_history_messages=args.max_history_messages,
            )
            for _ in range(num_envs)
        ]

    if args.policy == "scripted":
        return [ScriptedHighLevelPolicy(args.scripted_instruction) for _ in range(num_envs)]

    return [
        HumanPolicy(
            prompt=prompt,
            image_dir=str(Path(args.output_dir) / "human_policy" / f"seed{seed}"),
            image_prefix="observation",
        )
        for seed in _parse_seeds(args)
    ]


def _reset_workers_with_seeds(envs, seeds, timeout):
    futures = [worker.reset.remote(seed=seed) for worker, seed in zip(envs.workers, seeds)]
    results = ray.get(futures, timeout=timeout)
    obs_list, info_list = [], []
    for obs, info in results:
        obs_list.append(obs)
        info_list.append(info)
    return obs_list, info_list


def _write_video(path, frames, fps):
    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", macro_block_size=1)
    return True


def evaluate(args):
    seeds = _parse_seeds(args)
    num_envs = len(seeds)
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    prompt = _make_high_level_prompt(args.task)

    envs = LanguageTableMultiProcessEnv(
        num_envs=num_envs,
        block_mode=args.block_mode,
        reward_factory_cls=block2absolutelocation.BlockToAbsoluteLocationReward,
        seed=min(seeds),
        group_n=1,
        is_train=False,
        num_cpus=args.num_cpus,
        timeout=args.ray_timeout,
        return_full_state=True,
        render_obs=True,
        render_text_in_image=False,
    )
    low_level_policy = _make_low_level_policy(args)
    high_level_policies = _make_high_level_policies(args, prompt, num_envs)

    records = [
        {
            "trial": idx,
            "seed": seed,
            "high_level_done": False,
            "env_done": False,
            "steps": 0,
            "high_level_instructions": [],
            "video_path": str(video_dir / f"seed{seed}.mp4"),
        }
        for idx, seed in enumerate(seeds)
    ]
    frames = [[] for _ in range(num_envs)]

    try:
        low_level_policy.reset(num_envs=num_envs)
        for high_level_policy in high_level_policies:
            reset = getattr(high_level_policy, "reset", None)
            if reset is not None:
                reset()

        obs_list, last_infos = _reset_workers_with_seeds(
            envs, seeds, timeout=args.ray_timeout
        )
        active_mask = np.ones(num_envs, dtype=bool)

        for hl_step in range(args.num_hl_steps):
            if not active_mask.any():
                break

            rendered_frames = envs.render()
            high_level_instructions = ["done" for _ in range(num_envs)]
            active_indices = np.flatnonzero(active_mask).tolist()
            for env_idx in active_indices:
                instruction = high_level_policies[env_idx].step(rendered_frames[env_idx])
                instruction = instruction.strip()
                high_level_instructions[env_idx] = instruction
                records[env_idx]["high_level_instructions"].append(instruction)
                print(
                    f"seed={seeds[env_idx]} hl_step={hl_step} instruction={instruction!r}",
                    flush=True,
                )
                if instruction.lower() == "done":
                    records[env_idx]["high_level_done"] = True
                    active_mask[env_idx] = False

            if not active_mask.any():
                break

            for low_step in range(args.num_low_level_steps):
                if not active_mask.any():
                    break

                actions = low_level_policy.predict(
                    goals=high_level_instructions,
                    obs_list=obs_list,
                    active_mask=active_mask,
                )
                obs_list, rewards, dones, last_infos = envs.step(
                    actions,
                    active_mask=active_mask,
                    cached_obs=obs_list,
                    cached_infos=last_infos,
                )
                dones = np.asarray(dones, dtype=bool)

                for env_idx in np.flatnonzero(active_mask):
                    records[env_idx]["steps"] += 1
                    if "rgb" in obs_list[env_idx]:
                        frames[env_idx].append(
                            annotate_frame(
                                obs_list[env_idx]["rgb"],
                                traj_idx=env_idx,
                                turn_idx=records[env_idx]["steps"],
                                instruction=high_level_instructions[env_idx],
                                task=args.task,
                                reward=float(rewards[env_idx]),
                                font_size=args.font_size,
                            )
                        )

                newly_done = dones & active_mask
                for env_idx in np.flatnonzero(newly_done):
                    records[env_idx]["env_done"] = True
                active_mask &= ~dones

                if args.reset_freq > 0 and (low_step + 1) % args.reset_freq == 0:
                    low_level_policy.reset(num_envs=num_envs)

        for env_idx, seed in enumerate(seeds):
            video_path = video_dir / f"seed{seed}.mp4"
            try:
                records[env_idx]["video_written"] = _write_video(
                    video_path, frames[env_idx], args.fps
                )
            except Exception as exc:
                records[env_idx]["video_written"] = False
                records[env_idx]["video_error"] = str(exc)

        summary = {
            "task": args.task,
            "policy": args.policy,
            "low_level_policy": args.low_level_policy,
            "low_level_checkpoint": args.low_level_checkpoint,
            "seeds": seeds,
            "num_hl_steps": args.num_hl_steps,
            "num_low_level_steps": args.num_low_level_steps,
            "videos_dir": str(video_dir),
            "trials": records,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        close = getattr(low_level_policy, "close", None)
        if close is not None:
            close()
        envs.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a high-level policy over multiple Language Table seeds."
    )
    parser.add_argument("--seeds", default="", help="Comma-separated seed list.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--policy", choices=["vllm", "human", "scripted"], default="vllm")
    parser.add_argument("--vllm_url", default="http://localhost:8001")
    parser.add_argument("--max_history_messages", type=int, default=None)
    parser.add_argument(
        "--scripted_instruction",
        default="push the red moon to the center",
        help="High-level command used when --policy=scripted.",
    )
    parser.add_argument(
        "--low_level_policy", choices=["lava", "smolvla"], default="smolvla"
    )
    parser.add_argument(
        "--low_level_checkpoint",
        default="Sidharth-R/langtable-smolvla-finetuned",
    )
    parser.add_argument("--num_hl_steps", type=int, default=30)
    parser.add_argument("--num_low_level_steps", type=int, default=50)
    parser.add_argument("--reset_freq", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--font_size", type=int, default=12)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50101)
    parser.add_argument(
        "--server_log", default="/tmp/smolvla_parallel_high_level_eval.log"
    )
    parser.add_argument("--disable_low_level_batch", action="store_true")

    parser.add_argument(
        "--block_mode",
        choices=["BLOCK_4", "BLOCK_8", "BLOCK_4_WPOLE", "BLOCK_8_WPOLE"],
        default="BLOCK_4",
    )
    parser.add_argument("--num_cpus", type=float, default=None)
    parser.add_argument("--ray_timeout", type=float, default=None)

    parser.add_argument(
        "--lava_checkpoint_dir",
        default="/home/sidhraja/projects/LaMer/checkpoints",
    )
    parser.add_argument(
        "--lava_checkpoint_prefix",
        default="bc_resnet_sim_checkpoint_955000",
    )
    parser.add_argument(
        "--lava_preprocess_mode",
        choices=["original", "batched_tf", "jax_gpu", "jax_fused"],
        default="jax_gpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
