"""Extract all instructions from Language Table datasets.

This script:
1. Enumerates all template-generated instructions from reward modules
2. Streams each GCS dataset and extracts the instruction from each episode
3. Saves everything as a combined JSON dataset
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict

# ── Part 1: Template-generated instructions ──────────────────────────────────

def get_template_instructions():
    """Generate all possible template instructions for each block mode."""
    from language_table.environments import blocks
    from language_table.environments.rewards import instructions

    results = {}
    for mode_name, mode in [
        ("BLOCK_4", blocks.LanguageTableBlockVariants.BLOCK_4),
        ("BLOCK_8", blocks.LanguageTableBlockVariants.BLOCK_8),
    ]:
        all_instr = instructions.generate_all_instructions(mode)
        results[mode_name] = all_instr
        print(f"  {mode_name}: {len(all_instr)} template instructions "
              f"({len(set(all_instr))} unique)")
    return results


# ── Part 2: Extract instructions from GCS RLDS datasets ─────────────────────

DATASET_PATHS = {
    "language_table": "gs://gresearch/robotics/language_table/0.0.1/",
    "language_table_sim": "gs://gresearch/robotics/language_table_sim/0.0.1/",
    "language_table_blocktoblock_sim": "gs://gresearch/robotics/language_table_blocktoblock_sim/0.0.1/",
    "language_table_blocktoblock_4block_sim": "gs://gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1/",
    "language_table_blocktoblock_oracle_sim": "gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1/",
    "language_table_blocktoblockrelative_oracle_sim": "gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim/0.0.1/",
    "language_table_blocktoabsolute_oracle_sim": "gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1/",
    "language_table_blocktorelative_oracle_sim": "gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1/",
    "language_table_separate_oracle_sim": "gs://gresearch/robotics/language_table_separate_oracle_sim/0.0.1/",
}


def extract_instruction_from_step(step):
    """Extract instruction string from a single RLDS step."""
    import tensorflow as tf

    instruction = step["observation"]["instruction"]
    # instruction is a Tensor of shape (512,) with dtype int32
    # containing Unicode code points, padded with 0s
    if hasattr(instruction, "numpy"):
        codes = instruction.numpy()
    else:
        codes = instruction

    # Filter out zero-padding and convert code points to string
    chars = [chr(c) for c in codes if c != 0]
    return "".join(chars).strip()


def extract_instructions_from_dataset(dataset_name, dataset_path):
    """Stream a dataset and extract the instruction from each episode."""
    import tensorflow_datasets as tfds
    import tensorflow as tf

    print(f"\n  Loading {dataset_name}...")
    try:
        builder = tfds.builder_from_directory(builder_dir=dataset_path)
        ds = builder.as_dataset(
            split="train",
            # Skip decoding images for speed
            decoders={
                "steps": {"observation": {"rgb": tfds.decode.SkipDecoding()}}
            },
        )
    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        return []

    instructions = []
    episode_count = 0
    t0 = time.time()

    for episode in ds:
        # Get the first step's instruction (same for all steps in episode)
        first_step = next(iter(episode["steps"].take(1)))
        instr = extract_instruction_from_step(first_step)
        if instr:
            instructions.append(instr)
        episode_count += 1
        if episode_count % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    {episode_count} episodes processed "
                  f"({elapsed:.1f}s, {len(set(instructions))} unique so far)")

    elapsed = time.time() - t0
    unique = len(set(instructions))
    print(f"  Done: {episode_count} episodes, {unique} unique instructions "
          f"in {elapsed:.1f}s")
    return instructions


# ── Part 3: Save results ────────────────────────────────────────────────────

def save_results(template_instructions, dataset_instructions, output_dir):
    """Save all results as JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save template instructions
    template_file = os.path.join(output_dir, "template_instructions.json")
    with open(template_file, "w") as f:
        json.dump(template_instructions, f, indent=2)
    print(f"\nSaved template instructions to {template_file}")

    # 2. Save per-dataset instructions with counts
    all_instructions = []
    per_dataset_summary = {}

    for ds_name, instr_list in dataset_instructions.items():
        counter = Counter(instr_list)
        per_dataset_summary[ds_name] = {
            "total_episodes": len(instr_list),
            "unique_instructions": len(counter),
            "instruction_counts": dict(counter.most_common()),
        }

        for instr, count in counter.items():
            all_instructions.append({
                "instruction": instr,
                "dataset": ds_name,
                "count": count,
            })

    per_dataset_file = os.path.join(output_dir, "per_dataset_instructions.json")
    with open(per_dataset_file, "w") as f:
        json.dump(per_dataset_summary, f, indent=2)
    print(f"Saved per-dataset summary to {per_dataset_file}")

    # 3. Save the combined flat dataset (one row per unique instruction+dataset)
    combined_file = os.path.join(output_dir, "all_instructions.json")
    with open(combined_file, "w") as f:
        json.dump(all_instructions, f, indent=2)
    print(f"Saved combined instructions ({len(all_instructions)} rows) "
          f"to {combined_file}")

    # 4. Save a deduplicated list of all unique instructions across all datasets
    all_unique = sorted(set(row["instruction"] for row in all_instructions))
    unique_file = os.path.join(output_dir, "unique_instructions.json")
    with open(unique_file, "w") as f:
        json.dump(all_unique, f, indent=2)
    print(f"Saved {len(all_unique)} globally unique instructions "
          f"to {unique_file}")

    # 5. Save as CSV too for easy loading
    csv_file = os.path.join(output_dir, "all_instructions.csv")
    with open(csv_file, "w") as f:
        f.write("instruction,dataset,count\n")
        for row in all_instructions:
            # Escape commas and quotes in instruction text
            instr_escaped = row["instruction"].replace('"', '""')
            f.write(f'"{instr_escaped}",{row["dataset"]},{row["count"]}\n')
    print(f"Saved CSV to {csv_file}")

    return all_unique


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "instruction_data"
    )

    # ── Step 1: Template instructions ──
    print("=" * 60)
    print("Step 1: Generating template instructions")
    print("=" * 60)
    template_instructions = get_template_instructions()

    # ── Step 2: Extract from GCS datasets ──
    print("\n" + "=" * 60)
    print("Step 2: Extracting instructions from GCS datasets")
    print("=" * 60)

    dataset_instructions = {}

    # Allow selecting specific datasets via CLI args
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        paths = {k: v for k, v in DATASET_PATHS.items() if k in selected}
    else:
        paths = DATASET_PATHS

    for ds_name, ds_path in paths.items():
        instructions = extract_instructions_from_dataset(ds_name, ds_path)
        dataset_instructions[ds_name] = instructions

    # ── Step 3: Save ──
    print("\n" + "=" * 60)
    print("Step 3: Saving results")
    print("=" * 60)
    all_unique = save_results(
        template_instructions, dataset_instructions, output_dir
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mode, instrs in template_instructions.items():
        print(f"  Template {mode}: {len(set(instrs))} unique")
    for ds_name, instrs in dataset_instructions.items():
        print(f"  Dataset {ds_name}: {len(instrs)} episodes, "
              f"{len(set(instrs))} unique")
    print(f"\n  Total unique instructions across all datasets: {len(all_unique)}")
    print(f"\n  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
