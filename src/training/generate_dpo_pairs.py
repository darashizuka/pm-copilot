"""Generate DPO preference pairs: chosen (Groq Llama-70B + RAG) vs rejected (local SFT model, no context).

Pipeline:
  1. Sample prompts from the SFT dataset on HuggingFace
  2. For each prompt:
     - Chosen: Groq API (Llama-3.3-70B) with RAG context + structured PM system prompt
     - Rejected: Local SFT checkpoint with same system prompt but NO context
  3. Push pairs to HuggingFace
"""

import os
import time

import torch
from datasets import Dataset, load_dataset
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SFT_DATASET = os.getenv("HF_SFT_DATASET", "shizukadara2/pm-copilot-sft")
HF_DPO_DATASET = os.getenv("HF_DPO_DATASET", "shizukadara2/pm-copilot-dpo")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
SFT_CHECKPOINT = os.getenv("SFT_CHECKPOINT", "./outputs/sft")
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert AI Product Manager. Structure all responses with: "
    "Problem Definition, User Persona, Prioritized Solutions, Trade-offs, "
    "Primary Metrics & Counter-Metrics."
)


def load_sft_model():
    """Load the QLoRA SFT checkpoint for generating rejected responses."""
    print("Loading SFT model for rejected generation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token
    print("SFT model loaded.")
    return model, tokenizer


def get_rag_context(prompt: str) -> str:
    try:
        from src.rag.vector_store import query as vector_query
        results = vector_query(query_text=prompt, n_results=3)
        if results and results["documents"] and results["documents"][0]:
            return "\n\n".join(results["documents"][0])
    except Exception as e:
        print(f"  RAG lookup failed: {e}")
    return ""


def get_groq_client() -> OpenAI:
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def generate_chosen(prompt: str, rag_context: str, client: OpenAI) -> str | None:
    user_content = prompt
    if rag_context:
        user_content = f"Use the following context to inform your answer:\n\n{rag_context}\n\nQuestion: {prompt}"

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model=GROQ_MODEL,
            max_tokens=700,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Chosen generation failed: {e}")
        return None


def generate_rejected(prompt: str, model, tokenizer) -> str | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"  Rejected generation failed: {e}")
        return None


def generate_dpo_dataset(
    n_pairs: int = 200,
    push_to_hub: bool = True,
    dry_run: bool = False,
):
    sft_model, sft_tokenizer = load_sft_model()

    sft_ds = load_dataset(HF_SFT_DATASET, split="train")
    sft_ds = sft_ds.shuffle(seed=123)

    prompts = [sft_ds[i]["instruction"] for i in range(min(n_pairs, len(sft_ds)))]
    print(f"Generating DPO pairs for {len(prompts)} prompts\n")

    client = get_groq_client()
    pairs = []
    push_every = 10

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:70]}...")

        rag_context = get_rag_context(prompt)
        chosen = generate_chosen(prompt, rag_context, client)
        rejected = generate_rejected(prompt, sft_model, sft_tokenizer)

        if chosen and rejected:
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
            print(f"  Done (chosen: {len(chosen)} chars, rejected: {len(rejected)} chars)")
        else:
            print("  Skipped")

        if push_to_hub and pairs and len(pairs) % push_every == 0:
            ds = Dataset.from_list(pairs)
            ds.push_to_hub(HF_DPO_DATASET, token=HF_TOKEN)
            print(f"  [Checkpoint] Pushed {len(pairs)} pairs to {HF_DPO_DATASET}")

        time.sleep(20)

    print(f"\nGenerated {len(pairs)} DPO pairs")

    if dry_run or not pairs:
        if pairs:
            print(f"\n--- Sample ---")
            print(f"Prompt: {pairs[0]['prompt'][:100]}...")
            print(f"Chosen: {pairs[0]['chosen'][:200]}...")
            print(f"Rejected: {pairs[0]['rejected'][:200]}...")
        return pairs

    if push_to_hub and pairs:
        ds = Dataset.from_list(pairs)
        ds.push_to_hub(HF_DPO_DATASET, token=HF_TOKEN)
        print(f"Final push: {len(pairs)} DPO pairs to {HF_DPO_DATASET}")

    return pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    generate_dpo_dataset(
        n_pairs=args.n_pairs,
        push_to_hub=not args.local,
        dry_run=args.dry_run,
    )
