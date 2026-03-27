#!/usr/bin/env python3
"""Test script to preview frame annotation on an existing GIF.

Usage:
    conda run -n lamer python test_write.py [--input <gif>] [--output <gif>]

Tweak TRAJ_IDX, TURN_IDX, INSTRUCTION, FONT_SIZE, BG_ALPHA below to
experiment with the overlay before deploying it in the env managers.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Add the local package so we can import frame_annotator directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from language_table.lamer.frame_annotator import annotate_frame, _get_font

# ---------------------------------------------------------------------------
# Parameters you can play with
# ---------------------------------------------------------------------------
TRAJ_IDX = 0
TURN_IDX = 1
INSTRUCTION = "push the blue cube to the left of the yellow star"
FONT_SIZE = 14
BG_ALPHA = 0.55
TASK = "sort the red cube in the top left corner and the green star in the bottom right corner"
REWARD = 50.0

DEFAULT_INPUT = (
    "/gpfs/home/memmelma/projects/LaMer/wandb/run-20260325_111550-akwob79t"
    "/files/media/videos/training/env_video_0_1_c3e52287eae1e193a7e2.gif"
)


def load_gif_frames(path: str):
    """Load all frames from a GIF as a list of (H, W, 3) uint8 numpy arrays."""
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frame = gif.convert("RGB")
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def save_gif(frames, path: str, duration: int = 100):
    """Save a list of (H, W, 3) uint8 numpy arrays as a GIF."""
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser(description="Preview frame text overlay on a GIF")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input GIF path")
    parser.add_argument("--output", "-o", default=None, help="Output GIF path (default: <input>_annotated.gif)")
    parser.add_argument("--traj", type=int, default=TRAJ_IDX, help="Trial / trajectory index")
    parser.add_argument("--turn", type=int, default=TURN_IDX, help="Turn / step index")
    parser.add_argument("--instruction", default=INSTRUCTION, help="Language instruction text")
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help="Font size in pixels")
    parser.add_argument("--bg-alpha", type=float, default=BG_ALPHA, help="Background opacity (0-1)")
    parser.add_argument("--task", default=TASK, help="Task text")
    parser.add_argument("--reward", type=float, default=REWARD, help="Current reward value")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.with_name(p.stem + "_annotated" + p.suffix))

    print(f"Loading GIF: {args.input}")
    frames = load_gif_frames(args.input)
    print(f"  {len(frames)} frames, shape {frames[0].shape}")

    print(f"Annotating with: Trial {args.traj} | Step {args.turn}")
    print(f"  Instruction: {args.instruction}")
    annotated = [
        annotate_frame(
            f,
            traj_idx=args.traj,
            turn_idx=args.turn,
            instruction=args.instruction,
            font_size=args.font_size,
            bg_alpha=args.bg_alpha,
            task=args.task,
            reward=args.reward,
        )
        for f in frames
    ]

    print(f"Saving annotated GIF: {args.output}")
    save_gif(annotated, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
