"""Debug script: call _language_to_action_seq_async N times and inspect disturbances."""

import argparse
import asyncio
import numpy as np

from language_table.lamer.gemini_policy import _language_to_action_seq_async

SAMPLE_STATE = (
    "Task: push the red moon to the blue cube\n"
    "End-effector: (0.350, -0.120)\n"
    "Blocks: red_moon (0.42, -0.15), blue_cube (0.30, 0.20), green_star (0.55, 0.10)"
)

SAMPLE_ACTION = "move left by 0.1, then move up by 0.3"


async def run(n: int, state: str, action: str, model_id: str):
    for i in range(n):
        try:
            result = await _language_to_action_seq_async(
                state=state, action=action, disturbance=None, model_id=model_id,
            )
        except Exception as e:
            print(f"\n[{i+1}/{n}] ERROR: {e}")
            continue

        true = np.array(result.get("true_actions", []))
        dist = np.array(result.get("disturbed_actions", []))
        desc = result.get("disturbance", "")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{n}] disturbance: {desc}")
        print(f"  true_actions:     {true.tolist()}")
        print(f"  disturbed_actions:{dist.tolist()}")

        if true.shape == dist.shape and true.size > 0:
            diff = dist - true
            print(f"  diff (dist-true): {diff.tolist()}")
            print(f"  max |diff|:       {np.abs(diff).max():.4f}")
            identical = np.allclose(true, dist, atol=1e-4)
            print(f"  effectively identical: {identical}")
        else:
            print(f"  (shape mismatch or empty — true: {true.shape}, dist: {dist.shape})")


def main():
    parser = argparse.ArgumentParser(description="Debug Gemini disturbance generation")
    parser.add_argument("-n", type=int, default=10, help="Number of calls")
    parser.add_argument("--state", type=str, default=SAMPLE_STATE)
    parser.add_argument("--action", type=str, default=SAMPLE_ACTION)
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview")
    args = parser.parse_args()

    asyncio.run(run(args.n, args.state, args.action, args.model))


if __name__ == "__main__":
    main()