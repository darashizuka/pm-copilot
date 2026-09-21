"""QLoRA Supervised Fine-Tuning — run on Lightning AI with L4 GPU."""

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_config(path: str = "configs/sft_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/sft_config.yaml"):
    cfg = load_config(config_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias="none",
        task_type=cfg["lora"]["task_type"],
    )

    full_dataset = load_dataset(cfg["dataset"]["path"], split=cfg["dataset"]["split"])

    train_samples = cfg["dataset"].get("max_samples", 2000)
    val_ratio = cfg["dataset"].get("val_ratio", 0.05)
    total_needed = int(train_samples / (1 - val_ratio)) + 1
    if len(full_dataset) > total_needed:
        full_dataset = full_dataset.shuffle(seed=42).select(range(total_needed))
        print(f"Subsampled to {total_needed} examples (from full dataset)")

    def format_chat(example):
        messages = [
            {"role": "system", "content": "You are an expert AI Product Manager. Structure all responses with: Problem Definition, User Persona, Prioritized Solutions, Trade-offs, Primary Metrics & Counter-Metrics."},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    full_dataset = full_dataset.map(format_chat)

    split = full_dataset.train_test_split(test_size=val_ratio, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")

    sft_config = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        eval_strategy=cfg["training"]["save_strategy"],
        fp16=cfg["training"].get("fp16", False),
        bf16=cfg["training"].get("bf16", False),
        max_length=cfg["training"]["max_length"],
        dataset_text_field="text",
        packing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])
    print(f"SFT model saved to {cfg['training']['output_dir']}")


if __name__ == "__main__":
    train()
