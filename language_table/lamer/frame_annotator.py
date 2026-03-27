"""Overlay trial / step / instruction text onto RGB frames."""

import textwrap
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """Try to load a monospace TTF; fall back to the built-in bitmap font."""
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


def annotate_frame(
    frame: np.ndarray,
    traj_idx: int,
    turn_idx: int,
    instruction: str,
    task: Optional[str] = None,
    font_size: int = 14,
    padding: int = 4,
    bg_alpha: float = 0.55,
) -> np.ndarray:
    """Burn a text overlay into a single HWC uint8 RGB frame.

    The trial/step/instruction banner is drawn at the top.
    If *task* is provided, it is drawn at the bottom of the image.

    Returns a new array (original is not modified).
    """
    img = Image.fromarray(frame)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(font_size)

    char_width = font.getlength("A") if hasattr(font, "getlength") else font_size * 0.6
    max_chars = max(10, int((img.width - 2 * padding) / char_width))

    # --- top banner: trial / step / instruction ---
    line1 = f"Trial {traj_idx} | Step {turn_idx}"
    wrapped = textwrap.fill(instruction, width=max_chars)
    text = f"{line1}\n{wrapped}"

    bbox = draw.multiline_textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    rect_x2 = text_w + 2 * padding
    rect_y2 = text_h + 2 * padding

    draw.rectangle([0, 0, rect_x2, rect_y2], fill=(0, 0, 0))
    draw.multiline_text((padding, padding), text, fill=(255, 255, 255), font=font)

    # --- bottom banner: task ---
    if task:
        wrapped_task = textwrap.fill(task, width=max_chars)
        task_bbox = draw.multiline_textbbox((0, 0), wrapped_task, font=font)
        task_w = task_bbox[2] - task_bbox[0]
        task_h = task_bbox[3] - task_bbox[1]

        rect_y1 = img.height - task_h - 2 * padding
        draw.rectangle(
            [0, rect_y1, task_w + 2 * padding, img.height],
            fill=(0, 0, 0),
        )
        draw.multiline_text(
            (padding, rect_y1 + padding),
            wrapped_task,
            fill=(255, 255, 255),
            font=font,
        )

    blended = Image.blend(img, overlay, bg_alpha)
    return np.array(blended)


def annotate_frames(
    frames: List[np.ndarray],
    traj_idx: int,
    turn_idx: int,
    instruction: str,
    task: Optional[str] = None,
    font_size: int = 14,
) -> List[np.ndarray]:
    """Annotate every frame in a list (returns new list, originals untouched)."""
    return [
        annotate_frame(
            f, traj_idx, turn_idx, instruction,
            task=task, font_size=font_size,
        )
        for f in frames
    ]
