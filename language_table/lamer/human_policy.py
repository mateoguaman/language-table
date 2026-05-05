"""Human-in-the-loop vision-language policy.

The policy saves each image observation to disk, then reads a natural-language
instruction from stdin. This is useful for manually driving high-level policy
steps from the same evaluation loop used by automated VLM policies.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class HumanPolicy:
    """Prompt a human for language commands from saved image observations."""

    def __init__(
        self,
        prompt: Optional[str] = None,
        image_dir: str = "./tmp/language_table_human_policy",
        image_prefix: str = "observation",
    ):
        self.prompt = prompt
        self.image_dir = Path(image_dir)
        self.image_prefix = image_prefix
        self.step_idx = 0
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Reset the saved-image counter for a new episode or task."""
        self.step_idx = 0

    def step(self, image: np.ndarray) -> str:
        """Save ``image`` and return a human-entered instruction."""
        image_path = self.image_dir / f"{self.image_prefix}.png"
        Image.fromarray(_to_uint8_rgb(image), mode="RGB").save(image_path)
        self.step_idx += 1

        print(f"\nSaved observation image to: {image_path}")
        if self.prompt:
            print(f"Prompt:\n{self.prompt}")

        instruction = input("Instruction: ").strip()
        while not instruction:
            instruction = input("Instruction cannot be empty. Enter instruction: ").strip()
        return instruction


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a NumPy array, got {type(image).__name__}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape (height, width, 3) with RGB channels; "
            f"got {image.shape}"
        )

    if image.dtype == np.uint8:
        return image

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if np.nanmin(image) >= 0.0 and np.nanmax(image) <= 1.0:
            image = image * 255.0
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)
