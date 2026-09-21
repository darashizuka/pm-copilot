"""QLoRA Direct Preference Optimization — run on Lightning AI with L4 GPU."""

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def load_config(path: str = "configs/dpo_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/dpo_config.yaml"):
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

    dataset = load_dataset(cfg["dataset"]["path"], split=cfg["dataset"]["split"])

    def format_dpo(example):
        system = ("You are an expert AI Product Manager. Structure all responses with: "
                   "Problem Definition, User Persona, Prioritized Solutions, Trade-offs, "
                   "Primary Metrics & Counter-Metrics.")
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": example["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True),
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    dataset = dataset.map(format_dpo)

    dpo_config = DPOConfig(
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
        bf16=cfg["training"]["bf16"],
        beta=cfg["dpo"]["beta"],
        loss_type=cfg["dpo"]["loss_type"],
        max_length=cfg["training"]["max_length"],
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        args=dpo_config,
    )

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])
    print(f"DPO model saved to {cfg['training']['output_dir']}")


if __name__ == "__main__":
    train()
