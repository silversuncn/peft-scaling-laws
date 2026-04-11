#!/usr/bin/env python3
"""training_causal.py — CausalLM training pipeline for decoder-only models.

Used for Qwen2.5-0.5B validation experiments.
Runs as a standalone script, same interface as run_experiment.py.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Task to prompt template mapping
TASK_TEMPLATES = {
    "sst2": {
        "prompt": "Classify the sentiment of this sentence as 'positive' or 'negative'.\nSentence: {sentence}\nSentiment:",
        "labels": {0: "negative", 1: "positive"},
        "keys": ("sentence",),
    },
    "mrpc": {
        "prompt": "Are these two sentences semantically equivalent? Answer 'yes' or 'no'.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer:",
        "labels": {0: "no", 1: "yes"},
        "keys": ("sentence1", "sentence2"),
    },
    "cola": {
        "prompt": "Is this sentence grammatically correct? Answer 'yes' or 'no'.\nSentence: {sentence}\nAnswer:",
        "labels": {0: "no", 1: "yes"},
        "keys": ("sentence",),
    },
}


@dataclass
class CausalExperimentConfig:
    method: str
    model_name: str
    task_name: str
    seed: int
    train_subset_size: int
    num_train_epochs: float = 3.0
    learning_rate: float = 0.0  # 0 means auto-select based on method
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    max_length: int = 256
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    output_root: str = "artifacts/final_runs"


def get_default_lr(method: str) -> float:
    """Return method-appropriate learning rate for CausalLM."""
    if method == "full_ft":
        return 2e-5   # conservative for 0.5B full FT
    else:
        return 2e-4   # standard for LoRA/BitFit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_example(example, task_name):
    """Format a dataset example into a prompt string."""
    template = TASK_TEMPLATES[task_name]
    kwargs = {k: example[k] for k in template["keys"]}
    prompt = template["prompt"].format(**kwargs)
    label_text = template["labels"][example["label"]]
    return {"text": f"{prompt} {label_text}", "prompt": prompt, "label_text": label_text}


def run_causal_experiment(config: CausalExperimentConfig):
    """Run a single CausalLM fine-tuning experiment."""
    set_seed(config.seed)

    # Auto-select lr if not explicitly set
    if config.learning_rate <= 0:
        config.learning_rate = get_default_lr(config.method)
        print(f"  Auto lr={config.learning_rate} for method={config.method}")

    # Create run directory
    run_name = (f"{config.method}__{config.model_name.replace('/', '_')}"
                f"__{config.task_name}__n{config.train_subset_size}__s{config.seed}"
                f"__{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = PROJECT_ROOT / config.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Run: {run_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,  # bf16 to match training args
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())

    # Apply PEFT method
    if config.method == "lora":
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    elif config.method == "bitfit":
        for name, param in model.named_parameters():
            param.requires_grad = "bias" in name
    elif config.method == "full_ft":
        pass  # All parameters trainable
    else:
        raise ValueError(f"Unsupported method for CausalLM: {config.method}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")

    # Load and prepare dataset
    dataset = load_dataset("glue", config.task_name)
    train_ds = dataset["train"]
    if config.train_subset_size < len(train_ds):
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        train_ds = train_ds.select(indices[:config.train_subset_size])

    val_ds = dataset["validation"]

    # Format examples
    def tokenize_fn(examples):
        formatted = [format_example(
            {k: examples[k][i] for k in examples if k != "idx"},
            config.task_name
        ) for i in range(len(examples["label"]))]
        texts = [f["text"] for f in formatted]
        tokenized = tokenizer(texts, truncation=True, max_length=config.max_length,
                             padding="max_length", return_tensors=None)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    # Training
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,   # prevent gradient explosion on full_ft
        warmup_ratio=0.1,    # gradual warmup
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # Evaluate: generate predictions
    model.eval()
    correct = 0
    total = 0
    template = TASK_TEMPLATES[config.task_name]

    with torch.no_grad():
        for example in val_ds:
            kwargs = {k: example[k] for k in template["keys"]}
            prompt = template["prompt"].format(**kwargs)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=config.max_length).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip().lower()
            expected = template["labels"][example["label"]]
            if expected.lower() in generated:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save metrics
    metrics = {
        "run_dir": run_name,
        "config": asdict(config),
        "primary_metric_name": "accuracy",
        "eval_metrics": {
            "eval_primary_metric": accuracy,
            "eval_accuracy": accuracy,
            "eval_correct": correct,
            "eval_total": total,
        },
        "parameter_stats": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_pct,
        },
        "architecture": "causal_lm",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run a CausalLM fine-tuning experiment.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_subset_size", type=int, default=500)
    parser.add_argument("--output_root", type=str, default="artifacts/final_runs")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=0.0,
                        help="Learning rate. 0 = auto-select based on method.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    config = CausalExperimentConfig(
        method=args.method,
        model_name=args.model_name,
        task_name=args.task_name,
        seed=args.seed,
        train_subset_size=args.train_subset_size,
        output_root=args.output_root,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    run_causal_experiment(config)


if __name__ == "__main__":
    main()
