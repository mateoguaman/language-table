"""
Aggregate per-episode CSVs from multiple policies into comparison tables.

Reads <benchmark_dir>/<policy_id>/episodes.csv (one CSV per policy). Computes
per-(policy, block_mode, reward_type) success rate aggregated across seeds and
episodes, with Wilson score 95% CIs. Emits markdown.

Usage:
    # Auto-discover all policies in a benchmark dir
    python training/eval/aggregate_results.py eval_results/comprehensive_run

    # Or specify policies explicitly
    python training/eval/aggregate_results.py \
        --policies smolvla=eval_results/run/smolvla_full_93185 \
                   lava=eval_results/run/lava_resnet \
        --output comparison.md
"""

import argparse
import csv
import math
import os
from collections import defaultdict


def wilson(successes, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def load_episodes(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["success"] = int(r["success"])
            r["steps"] = int(r["steps"])
            r["seed"] = int(r["seed"])
            r["episode"] = int(r["episode"])
            rows.append(r)
    return rows


def fmt_pct(x):
    return f"{100.0 * x:5.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark_dir", nargs="?",
                        help="Auto-discover policies in this directory")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="name=path entries (path is dir or episodes.csv)")
    parser.add_argument("--output", default=None,
                        help="Write markdown here in addition to stdout")
    args = parser.parse_args()

    policies = {}
    if args.policies:
        for p in args.policies:
            name, path = p.split("=", 1)
            if os.path.isdir(path):
                path = os.path.join(path, "episodes.csv")
            policies[name] = path
    elif args.benchmark_dir:
        for entry in sorted(os.listdir(args.benchmark_dir)):
            csv_path = os.path.join(args.benchmark_dir, entry, "episodes.csv")
            if os.path.exists(csv_path):
                policies[entry] = csv_path
    else:
        parser.error("Provide benchmark_dir or --policies")

    if not policies:
        print("No policies found.")
        return

    # Load and aggregate.
    by_policy = {}
    for name, path in policies.items():
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping {name}")
            continue
        rows = load_episodes(path)
        if not rows:
            print(f"WARNING: empty {path}, skipping {name}")
            continue
        by_policy[name] = rows

    if not by_policy:
        print("No CSV data loaded.")
        return

    # data[policy][(block_mode, reward_type)] = {n, succ, sum_steps}
    data = defaultdict(lambda: defaultdict(lambda: {"n": 0, "succ": 0, "steps": 0}))
    for name, rows in by_policy.items():
        for r in rows:
            d = data[name][(r["block_mode"], r["reward_type"])]
            d["n"] += 1
            d["succ"] += r["success"]
            d["steps"] += r["steps"]

    block_modes = sorted({k[0] for d in data.values() for k in d.keys()})
    reward_types = sorted({k[1] for d in data.values() for k in d.keys()})
    pol_names = sorted(data.keys())

    out = []
    out.append("# Benchmark comparison\n")
    out.append(f"Policies: {', '.join(pol_names)}\n")

    for bm in block_modes:
        out.append(f"\n## block_mode = {bm}\n")
        # Success-rate table with Wilson CIs
        header = ["Reward type"] + [f"{p} (success [95% CI])" for p in pol_names]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for rt in reward_types:
            cells = [rt]
            for p in pol_names:
                d = data[p].get((bm, rt))
                if not d or d["n"] == 0:
                    cells.append("—")
                else:
                    rate, lo, hi = wilson(d["succ"], d["n"])
                    cells.append(
                        f"{fmt_pct(rate)} [{fmt_pct(lo)}, {fmt_pct(hi)}] "
                        f"(n={d['n']})")
            out.append("| " + " | ".join(cells) + " |")

        # Mean-steps table
        out.append("")
        header = ["Reward type"] + [f"{p} mean steps" for p in pol_names]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for rt in reward_types:
            cells = [rt]
            for p in pol_names:
                d = data[p].get((bm, rt))
                if not d or d["n"] == 0:
                    cells.append("—")
                else:
                    cells.append(f"{d['steps'] / d['n']:6.1f}")
            out.append("| " + " | ".join(cells) + " |")

    # Per-policy overall (across all block_modes x reward_types).
    out.append("\n## Overall (pooled across block_modes × reward_types)\n")
    out.append("| Policy | Success rate [95% CI] | Mean steps | Total episodes |")
    out.append("|---|---|---|---|")
    for p in pol_names:
        total_n = sum(d["n"] for d in data[p].values())
        total_s = sum(d["succ"] for d in data[p].values())
        total_steps = sum(d["steps"] for d in data[p].values())
        rate, lo, hi = wilson(total_s, total_n)
        mean_steps = total_steps / total_n if total_n else 0.0
        out.append(
            f"| {p} | {fmt_pct(rate)} [{fmt_pct(lo)}, {fmt_pct(hi)}] | "
            f"{mean_steps:.1f} | {total_n} |")

    # Per-policy per-reward (averaged across block_modes & seeds).
    out.append("\n## Per reward type (pooled across block_modes & seeds)\n")
    header = ["Reward type"] + [f"{p}" for p in pol_names]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for rt in reward_types:
        cells = [rt]
        for p in pol_names:
            n = sum(d["n"] for k, d in data[p].items() if k[1] == rt)
            s = sum(d["succ"] for k, d in data[p].items() if k[1] == rt)
            if n == 0:
                cells.append("—")
            else:
                rate, lo, hi = wilson(s, n)
                cells.append(
                    f"{fmt_pct(rate)} [{fmt_pct(lo)}, {fmt_pct(hi)}] (n={n})")
        out.append("| " + " | ".join(cells) + " |")

    text = "\n".join(out) + "\n"
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"\nWrote markdown: {args.output}")


if __name__ == "__main__":
    main()
