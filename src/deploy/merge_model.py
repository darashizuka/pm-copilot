"""Merge LoRA adapter into base model for deployment."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./outputs/dpo"
OUTPUT_PATH = "./outputs/merged"


def merge():
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading DPO adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {OUTPUT_PATH}...")
    merged.save_pretrained(OUTPUT_PATH)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"Done. Merged model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    merge()
