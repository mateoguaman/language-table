"""
Compare evaluation results across multiple policies.

Usage:
    python training/eval/compare_policies.py \
        eval_results/lava_baseline/eval_results.json \
        eval_results/smolvla_expert_oracle/eval_results.json \
        eval_results/pi0_expert_oracle/eval_results.json
"""

import argparse
import json
import sys


def load_results(path):
    """Load eval results JSON."""
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Language Table policy evaluation results")
    parser.add_argument("results", nargs="+",
                        help="Paths to eval_results.json files")
    parser.add_argument("--format", choices=["table", "markdown", "csv"],
                        default="markdown",
                        help="Output format")
    args = parser.parse_args()

    # Load all results
    policies = []
    for path in args.results:
        data = load_results(path)
        # Create a short name from the policy type + checkpoint
        checkpoint = data.get("checkpoint_path", "unknown")
        name = f"{data['policy_type']}"
        # Try to extract a meaningful name from the checkpoint path
        parts = checkpoint.rstrip("/").split("/")
        for part in reversed(parts):
            if part not in ("pretrained_model", "checkpoints", "last"):
                name = part
                break
        policies.append({"name": name, "data": data})

    # Collect all reward types across all results
    all_rewards = []
    for p in policies:
        for r in p["data"]["results"]:
            if r not in all_rewards:
                all_rewards.append(r)

    # Print comparison
    if args.format == "markdown":
        print_markdown_table(policies, all_rewards)
    elif args.format == "csv":
        print_csv(policies, all_rewards)
    else:
        print_markdown_table(policies, all_rewards)


def print_markdown_table(policies, reward_types):
    """Print markdown comparison table."""
    # Header
    names = [p["name"] for p in policies]
    header = "| Reward Type | " + " | ".join(names) + " |"
    separator = "|" + "---|" * (len(names) + 1)

    print("\n## Policy Comparison\n")
    print(header)
    print(separator)

    # Rows
    for reward in reward_types:
        row = f"| {reward} |"
        for p in policies:
            results = p["data"]["results"].get(reward, {})
            rate = results.get("success_rate", None)
            if rate is not None:
                row += f" {rate:.1%} |"
            else:
                row += " — |"
        print(row)

    # Mean row
    row = "| **MEAN** |"
    for p in policies:
        mean = p["data"].get("summary", {}).get("mean_success_rate", None)
        if mean is not None:
            row += f" **{mean:.1%}** |"
        else:
            row += " — |"
    print(row)
    print()


def print_csv(policies, reward_types):
    """Print CSV comparison."""
    names = [p["name"] for p in policies]
    print("reward_type," + ",".join(names))

    for reward in reward_types:
        row = [reward]
        for p in policies:
            results = p["data"]["results"].get(reward, {})
            rate = results.get("success_rate", None)
            row.append(f"{rate:.4f}" if rate is not None else "")
        print(",".join(row))

    # Mean
    row = ["MEAN"]
    for p in policies:
        mean = p["data"].get("summary", {}).get("mean_success_rate", None)
        row.append(f"{mean:.4f}" if mean is not None else "")
    print(",".join(row))


if __name__ == "__main__":
    main()
