import imageio.v2 as imageio
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from language_table.environments import blocks, language_table
from language_table.environments.rewards import tetris_shape
from tf_agents.environments import gym_wrapper
from tf_agents.trajectories import time_step as ts
from language_table.lamer.lava_policy import LAVAPolicy
from language_table.lamer.smolvla_policy import SmolVLAPolicy
from language_table.lamer.vllm_policy import VLLMPolicy
from language_table.lamer.human_policy import HumanPolicy


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


def _annotate_frame(frame, task, instruction="", font_size=12, padding=4):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_font(font_size)
    max_width = img.width - 2 * padding

    top_text = _wrap_text(draw, f"Task: {task}", font, max_width)
    top_bbox = draw.multiline_textbbox((0, 0), top_text, font=font)
    top_h = top_bbox[3] - top_bbox[1]
    draw.rectangle([0, 0, img.width, top_h + 2 * padding], fill=(0, 0, 0, 160))
    draw.multiline_text((padding, padding), top_text, fill=(255, 255, 255, 255), font=font)

    if instruction:
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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", type=int, default=50101)
parser.add_argument("--task", type=str, default="arrange the blocks into the tetris/tetromino shape: S")
parser.add_argument("--policy", type=str, default="vllm", choices=["vllm", "human"])
parser.add_argument("--low_level_policy", type=str, default="smolvla", choices=["lava", "smolvla"])
parser.add_argument("--low_level_checkpoint", type=str, default="Sidharth-R/langtable-smolvla-finetuned")

args = parser.parse_args()
num_steps = 50
num_hl_steps = 30
reset_freq = 10
task = args.task
# action_scale = 2.0
instruction = f"""
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

block_mode = blocks.LanguageTableBlockVariants.BLOCK_4
reward_factory = tetris_shape.TetrisShapeReward

env = language_table.LanguageTable(
    block_mode=block_mode,
    reward_factory=reward_factory,
    render_text_in_image=False,
    seed=args.seed,
)

env = gym_wrapper.GymWrapper(env)

if not hasattr(env, "get_control_frequency"):
    env.get_control_frequency = lambda: env._control_frequency

# langtable policies
# policy_checkpoint_dir = "/home/sidhraja/projects/LaMer/checkpoints"
# policy_checkpoint_prefix = "bc_resnet_sim_checkpoint_955000"
# policy = LAVAPolicy(checkpoint_dir=policy_checkpoint_dir, checkpoint_prefix=policy_checkpoint_prefix)

# smolvla policy
if args.low_level_policy == "smolvla":
    policy_checkpoint_path = args.low_level_checkpoint
    policy = SmolVLAPolicy(checkpoint_path=policy_checkpoint_path, port=args.port)
elif args.low_level_policy == "lava":
    policy_checkpoint_dir = "/home/sidhraja/projects/LaMer/checkpoints"
    policy_checkpoint_prefix = "bc_resnet_sim_checkpoint_955000"
    policy = LAVAPolicy(checkpoint_dir=policy_checkpoint_dir, checkpoint_prefix=policy_checkpoint_prefix)

if args.policy == "vllm":
    hl_policy = VLLMPolicy(prompt=instruction, url="http://localhost:8000")
elif args.policy == "human":
    hl_policy = HumanPolicy(prompt=instruction)

policy.reset(num_envs=1)
env.seed(args.seed)
time_step = env.reset(); obs = time_step.observation
frames = []

# hl_instructions = [
#     "push the red moon to the left center",
#     "push the green star to the center",
#     "push the blue cube to the right center",
#     "push the yellow pentagon to the bottom center",
# ]

for j in range(num_hl_steps):
    frame = env.render(mode="rgb_array")
    # hl_instruction = hl_instructions[j]
    hl_instruction = hl_policy.step(frame)
    if hl_instruction == "done":
        break
    print(f"HL Instruction: {hl_instruction}")
    for i in range(num_steps):
        action = policy.predict(
            goals=[hl_instruction],
            obs_list=[obs],
            active_mask=np.array([True], dtype=bool),
        )[0]
        time_step = env.step(action)
        obs = time_step.observation

        frame = env.render(mode="rgb_array")
        frames.append(_annotate_frame(frame, task, hl_instruction))
        if i % reset_freq == 0:
            policy.reset(num_envs=1)
    # if time_step.is_last():
    #     break

imageio.mimwrite(f"language_table_{args.policy}.mp4", frames, fps=10, macro_block_size=1)