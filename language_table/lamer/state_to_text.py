"""
Convert Language Table observation dicts to natural-language descriptions.

Input: obs dict from LanguageTable.step() containing:
    - instruction (int32 array, UTF-8 encoded)
    - effector_translation (2,)
    - block_*_translation (2,) for each block
    - block_*_mask (1,) for each block (1.0 = present on table)

Output example:
    Task: push the red star to the blue cube
    End-effector: (0.350, -0.120)
    Blocks: red_star (0.42, -0.15), blue_cube (0.30, 0.20), green_moon (0.55, 0.10)
"""

import re
from typing import Any, Dict, List

import numpy as np


def _decode_instruction(instruction_array) -> str:
    """Decode int32 instruction array back to string."""
    if instruction_array is None:
        return ""
    arr = np.asarray(instruction_array)
    non_zero = arr[arr != 0]
    if non_zero.shape[0] == 0:
        return ""
    return bytes(non_zero.tolist()).decode("utf-8")


def state_to_text(obs: Dict[str, Any]) -> str:
    """Convert a single Language Table observation dict to text."""
    parts = []

    # Task instruction
    instruction = obs.get("instruction")
    if instruction is not None:
        task_str = _decode_instruction(instruction)
        if task_str:
            parts.append(f"Task: {task_str}")

    # End-effector position
    ee = obs.get("effector_translation")
    if ee is not None:
        parts.append(f"End-effector: ({ee[0]:.3f}, {ee[1]:.3f})")

    # Blocks present on the table
    block_entries = []
    for key in sorted(obs.keys()):
        m = re.match(r"block_(.+)_translation$", key)
        if m:
            block_name = m.group(1)
            mask_key = f"block_{block_name}_mask"
            mask = obs.get(mask_key)
            if mask is not None and float(mask[0]) < 0.5:
                continue  # block not on table
            pos = obs[key]
            block_entries.append(f"{block_name} ({pos[0]:.3f}, {pos[1]:.3f})")

    if block_entries:
        parts.append("Blocks: " + ", ".join(block_entries))

    return "\n".join(parts) if parts else "No observation available."


def batch_state_to_text(obs_list: List[Dict[str, Any]]) -> List[str]:
    """Convert a batch of observation dicts to text descriptions."""
    return [state_to_text(obs) for obs in obs_list]
