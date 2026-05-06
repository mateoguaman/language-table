import argparse
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tf_agents.environments import gym_wrapper

from language_table.environments import blocks, language_table
from language_table.environments.rewards import block2absolutelocation
from language_table.lamer.lava_policy import LAVAPolicy
from language_table.lamer.smolvla_policy import SmolVLAPolicy


def _get_font(size=14):
    for name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return "\n".join(lines)


def _annotate_frame(frame, task, instruction, run_idx, seed, font_size=12, padding=4):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_font(font_size)
    max_width = img.width - 2 * padding

    top_text = _wrap_text(draw, f"Run: {run_idx} | Seed: {seed} | Task: {task}", font, max_width)
    top_bbox = draw.multiline_textbbox((0, 0), top_text, font=font)
    top_h = top_bbox[3] - top_bbox[1]
    draw.rectangle([0, 0, img.width, top_h + 2 * padding], fill=(0, 0, 0, 160))
    draw.multiline_text((padding, padding), top_text, fill=(255, 255, 255, 255), font=font)

    bottom_text = _wrap_text(draw, f"Instruction: {instruction}", font, max_width)
    bottom_bbox = draw.multiline_textbbox((0, 0), bottom_text, font=font)
    bottom_h = bottom_bbox[3] - bottom_bbox[1]
    y0 = img.height - bottom_h - 2 * padding
    draw.rectangle([0, y0, img.width, img.height], fill=(0, 0, 0, 160))
    draw.multiline_text(
        (padding, y0 + padding),
        bottom_text,
        fill=(255, 255, 255, 255),
        font=font,
    )

    return np.array(img)


def _make_env(seed):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2absolutelocation.BlockToAbsoluteLocationReward,
        render_text_in_image=False,
        seed=seed,
    )
    env = gym_wrapper.GymWrapper(env)

    if not hasattr(env, "get_control_frequency"):
        env.get_control_frequency = lambda: env._control_frequency
    return env


def _make_policy(args):
    if args.low_level_policy == "smolvla":
        return SmolVLAPolicy(
            checkpoint_path=args.low_level_checkpoint,
            port=args.port,
            seed=args.seed,
        )

    policy_checkpoint_dir = "/home/sidhraja/projects/LaMer/checkpoints"
    policy_checkpoint_prefix = "bc_resnet_sim_checkpoint_955000"
    return LAVAPolicy(
        checkpoint_dir=policy_checkpoint_dir,
        checkpoint_prefix=policy_checkpoint_prefix,
    )


def _rollout_once(args, policy, run_idx, video_path):
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = _make_env(args.seed)
    frames = []
    total_reward = 0.0
    steps_taken = 0

    try:
        _reset_policy(policy, seed=args.seed)
        env.seed(args.seed)
        time_step = env.reset()
        obs = time_step.observation

        for step_idx in range(args.num_steps):
            action = policy.predict(
                goals=[args.instruction],
                obs_list=[obs],
                active_mask=np.array([True], dtype=bool),
            )[0]
            time_step = env.step(action)
            obs = time_step.observation

            if time_step.reward is not None:
                total_reward += float(np.asarray(time_step.reward))

            frame = env.render(mode="rgb_array")
            frames.append(
                _annotate_frame(
                    frame,
                    args.task,
                    args.instruction,
                    run_idx=run_idx,
                    seed=args.seed,
                )
            )
            steps_taken += 1

            if args.reset_freq > 0 and step_idx % args.reset_freq == 0:
                _reset_policy(policy, seed=args.seed)

            if args.stop_on_done and time_step.is_last():
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    if not frames:
        raise RuntimeError("No frames recorded; increase --num_steps or disable --stop_on_done")

    imageio.mimwrite(video_path, frames, fps=args.fps, macro_block_size=1)
    return {
        "run": run_idx,
        "seed": args.seed,
        "instruction": args.instruction,
        "video_path": str(video_path),
        "steps": steps_taken,
        "total_reward": total_reward,
    }


def _reset_policy(policy, seed):
    try:
        policy.reset(num_envs=1, seed=seed)
    except TypeError:
        policy.reset(num_envs=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the same Language Table seed and instruction repeatedly."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=50101)
    parser.add_argument(
        "--task",
        type=str,
        default="arrange the blocks into the tetris/tetromino shape: S",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="push the green star to the center",
        help="Fixed low-level instruction used for every rollout.",
    )
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--reset_freq", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stop_on_done", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("stochasticity_videos"),
    )
    parser.add_argument("--output_prefix", type=str, default="language_table_stochasticity")
    parser.add_argument(
        "--low_level_policy",
        type=str,
        default="smolvla",
        choices=["lava", "smolvla"],
    )
    parser.add_argument(
        "--low_level_checkpoint",
        type=str,
        default="Sidharth-R/langtable-smolvla-finetuned",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num_runs must be at least 1")
    if args.num_steps < 1:
        raise ValueError("--num_steps must be at least 1")
    if not args.instruction.strip():
        raise ValueError("--instruction must be non-empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = _make_policy(args)
    results = []

    try:
        for run_idx in range(args.num_runs):
            video_path = args.output_dir / (
                f"{args.output_prefix}_seed{args.seed}_run{run_idx:03d}.mp4"
            )
            print(
                f"Run {run_idx + 1}/{args.num_runs}: seed={args.seed}, "
                f"instruction={args.instruction!r}, video={video_path}"
            )
            results.append(_rollout_once(args, policy, run_idx, video_path))
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    metadata_path = args.output_dir / f"{args.output_prefix}_seed{args.seed}_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(
            {
                "seed": args.seed,
                "task": args.task,
                "instruction": args.instruction,
                "num_runs": args.num_runs,
                "num_steps": args.num_steps,
                "reset_freq": args.reset_freq,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
