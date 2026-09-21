"""Generate SFT data with Chain-of-Thought reasoning using a frontier model.

Self-Distilled Reasoning (SDR): the strong model generates reasoning traces
before producing structured PM output, so the fine-tuned model learns to think.
"""
import json
import os
import time
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SFT_DATASET = os.getenv("HF_SFT_DATASET", "shizukadara2/pm-copilot-sft")
HF_RAW_DATASET = os.getenv("HF_DATASET_NAME", "shizukadara2/pm-copilot-raw")

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "deepseek-ai/DeepSeek-V3-0324")

SYSTEM_PROMPT = """You are an elite Product Manager with 15+ years at top tech companies.

When given a product question, you MUST:

1. FIRST, think step-by-step inside <think> tags:
   - Identify the core problem and who it affects
   - Consider which PM frameworks apply (RICE, CIRCLES, Jobs-to-be-Done, etc.)
   - Analyze the target audience and their pain points
   - Evaluate potential solutions and their trade-offs
   - Select the right metrics and counter-metrics

2. THEN, produce your structured output inside <output> tags with these sections:
   ## Problem Definition
   ## User Persona
   ## Prioritized Solutions (with RICE or similar framework scoring)
   ## Trade-offs
   ## Primary Metrics & Counter-Metrics

Your reasoning must be genuine analysis, not filler. Show real PM thinking."""

PROMPT_GENERATION_SYSTEM = """You are a product management interview coach. Generate diverse, realistic PM prompts.

Each prompt should be a specific product question that requires structured thinking.
Cover different domains (B2B SaaS, Marketplace, FinTech, Consumer, E-commerce, HealthTech).
Cover different question types (improve a metric, design a feature, write a PRD, prioritize, diagnose).

Return a JSON array of prompt strings. No duplicates, no explanation."""


def generate_prompts_from_articles(n_prompts: int = 50) -> list[str]:
    """Use scraped articles to generate contextual PM prompts."""
    client = InferenceClient(token=HF_TOKEN)

    try:
        raw_ds = load_dataset(HF_RAW_DATASET, split="train")
    except Exception:
        print("Could not load raw dataset. Using built-in prompts only.")
        return []

    # Sample articles for context
    sample_indices = list(range(min(20, len(raw_ds))))
    titles = [raw_ds[i]["title"] for i in sample_indices if raw_ds[i].get("title")]
    context = "\n".join(f"- {t}" for t in titles[:20])

    user_msg = (
        f"Here are real PM article titles for context:\n{context}\n\n"
        f"Generate {n_prompts} diverse PM interview/case study prompts inspired by "
        f"these topics. Make them specific and actionable.\n\n"
        f"Return a JSON array of {n_prompts} strings."
    )

    result = client.chat_completion(
        messages=[
            {"role": "system", "content": PROMPT_GENERATION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        model=TEACHER_MODEL,
        max_tokens=2048,
        temperature=0.8,
    )
    response = result.choices[0].message.content

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        return json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        print("Failed to parse generated prompts. Using defaults.")
        return []


def load_existing_prompts() -> list[str]:
    """Load prompts from dpo_prompts.txt and seed examples."""
    prompts = []

    prompts_file = Path("data/prompts/dpo_prompts.txt")
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts.extend(line.strip() for line in f if line.strip())

    sft_file = Path("data/prompts/sft_examples.json")
    if sft_file.exists():
        with open(sft_file) as f:
            examples = json.load(f)
            prompts.extend(ex["instruction"] for ex in examples)

    return prompts


def generate_cot_response(prompt: str, client: InferenceClient) -> str | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model=TEACHER_MODEL,
            max_tokens=2048,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Failed generation: {e}")
        return None


def generate_sft_dataset(
    n_extra_prompts: int = 50,
    push_to_hub: bool = True,
    dry_run: bool = False,
    append: bool = True,
):
    # Collect prompts
    prompts = load_existing_prompts()
    print(f"Loaded {len(prompts)} existing prompts")

    if n_extra_prompts > 0:
        print(f"Generating {n_extra_prompts} additional prompts from articles...")
        extra = generate_prompts_from_articles(n_extra_prompts)
        prompts.extend(extra)
        print(f"Total prompts: {len(prompts)}")

    # Deduplicate
    prompts = list(dict.fromkeys(prompts))
    print(f"After dedup: {len(prompts)} unique prompts\n")

    # Generate CoT responses
    client = InferenceClient(token=HF_TOKEN)
    sft_data = []

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:70]}...")
        response = generate_cot_response(prompt, client)
        if response:
            sft_data.append({
                "instruction": prompt,
                "response": response,
            })
            print(f"  Generated ({len(response)} chars)")
        else:
            print(f"  Skipped")
        time.sleep(1)

    print(f"\nGenerated {len(sft_data)} CoT SFT pairs")

    if dry_run or not sft_data:
        if sft_data:
            print(f"\n--- Sample ---")
            print(f"Instruction: {sft_data[0]['instruction']}")
            print(f"Response:\n{sft_data[0]['response'][:500]}...")
        return sft_data

    new_ds = Dataset.from_list(sft_data)

    if push_to_hub:
        if append:
            try:
                old_ds = load_dataset(HF_SFT_DATASET, split="train")
                combined = concatenate_datasets([old_ds, new_ds])
                combined.push_to_hub(HF_SFT_DATASET, token=HF_TOKEN)
                print(f"Appended {len(sft_data)} CoT pairs (total: {len(combined)}) to {HF_SFT_DATASET}")
            except Exception:
                new_ds.push_to_hub(HF_SFT_DATASET, token=HF_TOKEN)
                print(f"Pushed {len(sft_data)} CoT pairs to {HF_SFT_DATASET}")
        else:
            new_ds.push_to_hub(HF_SFT_DATASET, token=HF_TOKEN)
            print(f"Pushed {len(sft_data)} CoT pairs to {HF_SFT_DATASET}")

    return sft_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-prompts", type=int, default=50, help="Number of extra prompts to generate")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-append", action="store_true", help="Overwrite instead of append")
    args = parser.parse_args()
    generate_sft_dataset(
        n_extra_prompts=args.extra_prompts,
        push_to_hub=not args.local,
        dry_run=args.dry_run,
        append=not args.no_append,
    )