"""Analyze and visualize the extracted instruction data.

Run this after extract_instructions.py has completed.
Produces charts and summary statistics.
"""

import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruction_data")
OUT_DIR = os.path.join(DATA_DIR, "figures")


def load_data():
    with open(os.path.join(DATA_DIR, "all_instructions.json")) as f:
        all_instructions = json.load(f)
    with open(os.path.join(DATA_DIR, "unique_instructions.json")) as f:
        unique_instructions = json.load(f)
    with open(os.path.join(DATA_DIR, "per_dataset_instructions.json")) as f:
        per_dataset = json.load(f)
    with open(os.path.join(DATA_DIR, "template_instructions.json")) as f:
        template = json.load(f)
    return all_instructions, unique_instructions, per_dataset, template


def classify_instruction(instr):
    """Classify an instruction into a task category."""
    instr_lower = instr.lower()
    if any(w in instr_lower for w in ["separate", "apart", "pull"]):
        return "separate_blocks"
    if any(w in instr_lower for w in ["point", "touch", "reach"]):
        return "point_to_block"
    if re.search(r"(to the (top|bottom|left|right|center|middle))", instr_lower):
        return "block_to_absolute"
    if re.search(r"(above|below|left of|right of|to the left|to the right)", instr_lower):
        if re.search(r"(above|below|left of|right of)\s+(the\s+)?\w+\s+(block|cube|moon|star|pentagon|hexagon|diamond|heart|triangle)", instr_lower):
            return "block_to_block_relative"
    if re.search(r"(slightly|a bit|a little)?\s*(move|push|slide|nudge)\s+.*(up|down|left|right)", instr_lower):
        return "block_to_relative"
    if re.search(r"(push|slide|move)\s+.*(to|toward|next to|near|close to|into|against|onto)\s+", instr_lower):
        return "block_to_block"
    return "other"


def plot_dataset_sizes(per_dataset):
    """Bar chart of episodes per dataset."""
    fig, ax = plt.subplots(figsize=(14, 6))
    names = list(per_dataset.keys())
    short_names = [n.replace("language_table_", "").replace("language_table", "real") or "real"
                   for n in names]
    episodes = [per_dataset[n]["total_episodes"] for n in names]
    unique = [per_dataset[n]["unique_instructions"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, episodes, w, label="Total episodes", color="#4A90D9")
    ax.bar(x + w/2, unique, w, label="Unique instructions", color="#E8833A")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Episodes and Unique Instructions per Dataset")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dataset_sizes.png"), dpi=150)
    plt.close(fig)
    print("  Saved dataset_sizes.png")


def plot_instruction_categories(all_instructions):
    """Pie chart of instruction categories."""
    categories = Counter()
    for row in all_instructions:
        cat = classify_instruction(row["instruction"])
        categories[cat] += row["count"]

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = list(categories.keys())
    sizes = [categories[l] for l in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        pctdistance=0.85, textprops={"fontsize": 10}
    )
    ax.set_title("Instruction Distribution by Task Category")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "instruction_categories.png"), dpi=150)
    plt.close(fig)
    print("  Saved instruction_categories.png")


def plot_instruction_frequency(per_dataset):
    """Histogram of instruction frequency (how many times each instruction appears)."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, (ds_name, data) in enumerate(per_dataset.items()):
        if i >= 9:
            break
        ax = axes[i]
        counts = list(data["instruction_counts"].values())
        if counts:
            ax.hist(counts, bins=min(50, max(10, len(set(counts)))),
                    color="#4A90D9", edgecolor="white", alpha=0.8)
        short = ds_name.replace("language_table_", "").replace("language_table", "real") or "real"
        ax.set_title(short, fontsize=10)
        ax.set_xlabel("Occurrences")
        ax.set_ylabel("# Instructions")
        if max(counts, default=0) > 100:
            ax.set_xscale("log")

    fig.suptitle("Instruction Frequency Distribution per Dataset", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "instruction_frequency.png"), dpi=150)
    plt.close(fig)
    print("  Saved instruction_frequency.png")


def plot_top_instructions(all_instructions, top_n=30):
    """Horizontal bar chart of most common instructions overall."""
    counter = Counter()
    for row in all_instructions:
        counter[row["instruction"]] += row["count"]

    top = counter.most_common(top_n)
    fig, ax = plt.subplots(figsize=(12, 10))
    labels = [t[0][:60] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]

    ax.barh(range(len(labels)), values, color="#4A90D9", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Total occurrences (across all datasets)")
    ax.set_title(f"Top {top_n} Most Frequent Instructions")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "top_instructions.png"), dpi=150)
    plt.close(fig)
    print("  Saved top_instructions.png")


def plot_word_frequency(unique_instructions, top_n=40):
    """Bar chart of most common words across instructions."""
    stop_words = {"the", "a", "an", "to", "of", "and", "in", "on", "at", "is"}
    word_counter = Counter()
    for instr in unique_instructions:
        words = instr.lower().split()
        for w in words:
            if w not in stop_words and len(w) > 1:
                word_counter[w] += 1

    top = word_counter.most_common(top_n)
    fig, ax = plt.subplots(figsize=(14, 6))
    words = [t[0] for t in top]
    counts = [t[1] for t in top]

    ax.bar(range(len(words)), counts, color="#E8833A", alpha=0.8)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Occurrences across unique instructions")
    ax.set_title(f"Top {top_n} Words in Instructions (excluding stop words)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "word_frequency.png"), dpi=150)
    plt.close(fig)
    print("  Saved word_frequency.png")


def plot_top_verbs(all_instructions, top_n=30):
    """Bar chart of the most common first words (verbs) weighted by episode count."""
    verb_counter = Counter()
    for row in all_instructions:
        first_word = row["instruction"].strip().split()[0].lower()
        verb_counter[first_word] += row["count"]

    top = verb_counter.most_common(top_n)
    fig, ax = plt.subplots(figsize=(14, 6))
    verbs = [t[0] for t in top]
    counts = [t[1] for t in top]
    total = sum(verb_counter.values())

    bars = ax.bar(range(len(verbs)), counts, color="#6A5ACD", alpha=0.85)
    ax.set_xticks(range(len(verbs)))
    ax.set_xticklabels(verbs, rotation=50, ha="right", fontsize=10)
    ax.set_ylabel("Total episode count")
    ax.set_title(f"Top {top_n} Verbs (First Word of Instruction)")

    # Add percentage labels on top of bars
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "top_verbs.png"), dpi=150)
    plt.close(fig)
    print("  Saved top_verbs.png")


def plot_instruction_length(unique_instructions):
    """Histogram of instruction lengths (in words)."""
    lengths = [len(instr.split()) for instr in unique_instructions]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=range(1, max(lengths) + 2),
            color="#5CB85C", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Number of words")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Instruction Lengths (words)")
    ax.axvline(np.mean(lengths), color="red", linestyle="--",
               label=f"Mean: {np.mean(lengths):.1f} words")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "instruction_length.png"), dpi=150)
    plt.close(fig)
    print("  Saved instruction_length.png")


def print_summary(all_instructions, unique_instructions, per_dataset, template):
    """Print text summary."""
    print("\n" + "=" * 70)
    print("INSTRUCTION DATA SUMMARY")
    print("=" * 70)

    total_episodes = sum(d["total_episodes"] for d in per_dataset.values())
    print(f"\nTotal episodes across all datasets: {total_episodes:,}")
    print(f"Total unique instructions (dataset): {len(unique_instructions):,}")

    for mode, instrs in template.items():
        print(f"Template {mode}: {len(set(instrs)):,} unique")

    print(f"\nPer-dataset breakdown:")
    for ds_name, data in per_dataset.items():
        short = ds_name.replace("language_table_", "") or "real"
        print(f"  {short:50s} {data['total_episodes']:>8,} episodes, "
              f"{data['unique_instructions']:>6,} unique instructions")

    # Categories
    categories = Counter()
    for row in all_instructions:
        cat = classify_instruction(row["instruction"])
        categories[cat] += row["count"]

    print(f"\nInstruction categories (by episode count):")
    for cat, count in categories.most_common():
        pct = count / total_episodes * 100
        print(f"  {cat:30s} {count:>10,} ({pct:.1f}%)")

    # Show some examples
    print(f"\nSample instructions:")
    seen_cats = set()
    for row in all_instructions:
        cat = classify_instruction(row["instruction"])
        if cat not in seen_cats:
            seen_cats.add(cat)
            print(f"  [{cat}] \"{row['instruction']}\"")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    all_instructions, unique_instructions, per_dataset, template = load_data()

    print("\nGenerating visualizations...")
    plot_dataset_sizes(per_dataset)
    plot_instruction_categories(all_instructions)
    plot_instruction_frequency(per_dataset)
    plot_top_instructions(all_instructions)
    plot_word_frequency(unique_instructions)
    plot_top_verbs(all_instructions)
    plot_instruction_length(unique_instructions)

    print_summary(all_instructions, unique_instructions, per_dataset, template)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
