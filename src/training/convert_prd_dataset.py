"""Convert Kaggle PRD-Synthetic dataset into SFT instruction/response format.
Push to HuggingFace as the SFT training dataset."""

import os
import json
from datasets import load_dataset, Dataset

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SFT_DATASET = os.getenv("HF_SFT_DATASET", "shizukadara2/pm-copilot-sft")
KAGGLE_DATASET = "karmukilandk/prd-synthetic-data"


def format_list(items: list) -> str:
    return "\n".join(f"  - {item}" for item in items)


def row_to_sft(row: dict) -> dict:
    instruction = (
        f"Write a Product Requirements Document (PRD) for {row['product_name']}, "
        f"a {row['product_type']} in the {row['industry']} industry. "
        f"The core problem: {row['problem_statement']}"
    )

    response = f"""## Product Requirements Document: {row['product_name']}

### Problem Definition
{row['problem_statement']}

**Industry:** {row['industry']}
**Product Type:** {row['product_type']}

### User Persona
**Target User:** {row['user_persona']}

### User Journey
{format_list(row['journey_steps'])}

### Goals
{format_list(row['goals'])}

### Key Features
{format_list(row['features'])}

### Success Metrics
{format_list(row['success_metrics'])}

### Risks & Mitigations
{format_list(row['risks'])}"""

    return {"instruction": instruction, "response": response}


def load_prd_data(data_dir: str = "data/external") -> list[dict]:
    """Load all PRD JSON files from the extracted Kaggle dataset."""
    from pathlib import Path
    all_rows = []
    data_path = Path(data_dir)

    json_files = list(data_path.rglob("*.json")) + list(data_path.rglob("*.jsonl"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        print("Download from Kaggle: kaggle datasets download karmukilandk/prd-synthetic-data")
        return []

    for jf in sorted(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            if jf.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        all_rows.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    all_rows.extend(data)
                else:
                    all_rows.append(data)
        print(f"  Loaded {jf.name} ({len(all_rows)} total rows)")

    return all_rows


def convert_and_upload(push_to_hub: bool = True, dry_run: bool = False, data_dir: str = "data/external"):
    rows = load_prd_data(data_dir)
    if not rows:
        return

    print(f"\nLoaded {len(rows)} PRD rows total")

    sft_data = []
    for row in rows:
        try:
            pair = row_to_sft(row)
            sft_data.append(pair)
        except (KeyError, TypeError) as e:
            print(f"  Skipping row: {e}")
            continue

    print(f"Converted {len(sft_data)} rows to SFT format")
    print(f"\nSample instruction:\n{sft_data[0]['instruction']}\n")
    print(f"Sample response (first 300 chars):\n{sft_data[0]['response'][:300]}...")

    if dry_run or not sft_data:
        return sft_data

    sft_ds = Dataset.from_list(sft_data)

    if push_to_hub:
        sft_ds.push_to_hub(HF_SFT_DATASET, token=HF_TOKEN)
        print(f"\nPushed {len(sft_data)} SFT pairs to {HF_SFT_DATASET}")
    else:
        sft_ds.save_to_disk("data/processed/sft")
        print(f"Saved locally to data/processed/sft")

    return sft_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", default="data/external", help="Path to extracted Kaggle dataset")
    args = parser.parse_args()
    convert_and_upload(push_to_hub=not args.local, dry_run=args.dry_run, data_dir=args.data_dir)
