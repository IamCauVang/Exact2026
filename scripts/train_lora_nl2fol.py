#!/usr/bin/env python3
"""QLoRA SFT training for EXACT NL -> FOL/IR compiler.

Default model: Qwen/Qwen2.5-7B-Instruct.
The training target is JSON-only compiler output, not final answers.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def filter_supported_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by the installed library version."""
    sig = inspect.signature(callable_obj)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--sft-dir", default=os.environ.get("SFT_DIR", "data/sft_nl2fol"))
    ap.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "artifacts/lora_qwen25_7b_nl2fol"))
    ap.add_argument("--max-seq-length", type=int, default=int(os.environ.get("MAX_SEQ_LENGTH", "2048")))
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("EPOCHS", "2")))
    ap.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "2e-4")))
    ap.add_argument("--per-device-train-batch-size", type=int, default=int(os.environ.get("TRAIN_BS", "1")))
    ap.add_argument("--per-device-eval-batch-size", type=int, default=int(os.environ.get("EVAL_BS", "1")))
    ap.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("GRAD_ACCUM", "8")))
    ap.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "16")))
    ap.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    ap.add_argument("--lora-dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.05")))
    ap.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "100")))
    ap.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "100")))
    ap.add_argument("--max-train-samples", type=int, default=int(os.environ.get("MAX_TRAIN_SAMPLES", "0")))
    ap.add_argument("--max-valid-samples", type=int, default=int(os.environ.get("MAX_VALID_SAMPLES", "0")))
    ap.add_argument("--no-4bit", action="store_true")
    return ap.parse_args()


def format_chat(example: dict[str, Any], tokenizer) -> dict[str, str]:
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("EXACT NL2FOL QLoRA SFT", flush=True)
    for k, v in vars(args).items():
        print(f"{k}={v}", flush=True)
    print("=" * 80, flush=True)

    sft_dir = Path(args.sft_dir)
    train_file = sft_dir / "train.jsonl"
    valid_file = sft_dir / "valid.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}. Run scripts/build_sft_data.py first.")
    if not valid_file.exists():
        raise FileNotFoundError(f"Missing {valid_file}. Run scripts/build_sft_data.py first.")

    ds = load_dataset("json", data_files={"train": str(train_file), "validation": str(valid_file)})
    if args.max_train_samples > 0:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    if args.max_valid_samples > 0:
        ds["validation"] = ds["validation"].select(range(min(args.max_valid_samples, len(ds["validation"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = ds.map(lambda ex: format_chat(ex, tokenizer), remove_columns=ds["train"].column_names)

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_config_kwargs: dict[str, Any] = {
        "output_dir": str(out),

        # TRL SFT-specific arguments.
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
        "max_seq_length": args.max_seq_length,
        "packing": False,

        # Training arguments.
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "save_total_limit": 2,
        "fp16": True,
        "bf16": False,
        "optim": "paged_adamw_8bit" if quant_config is not None else "adamw_torch",
        "report_to": "none",
        "gradient_checkpointing": True,
        "warmup_steps": 10,
        "lr_scheduler_type": "cosine",
        "remove_unused_columns": False,
    }

    # Some TRL/Transformers versions use "evaluation_strategy" instead of "eval_strategy".
    sig = inspect.signature(SFTConfig)
    if "eval_strategy" not in sig.parameters and "evaluation_strategy" in sig.parameters:
        sft_config_kwargs["evaluation_strategy"] = sft_config_kwargs.pop("eval_strategy")

    sft_args = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_config_kwargs))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": ds["train"],
        "eval_dataset": ds["validation"],
        "peft_config": peft_config,

        # Newer TRL uses processing_class; older TRL uses tokenizer.
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }

    trainer = SFTTrainer(**filter_supported_kwargs(SFTTrainer.__init__, trainer_kwargs))

    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    train_report = {
        "model_name": args.model_name,
        "data_source": "sft_jsonl",
        "sft_dir": args.sft_dir,
        "output_dir": args.output_dir,
        "num_train_samples": len(ds["train"]),
        "num_valid_samples": len(ds["validation"]),
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "used_4bit": quant_config is not None,
    }
    (out / "train_report.json").write_text(json.dumps(train_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "training_meta.json").write_text(json.dumps(train_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAIN_DONE", json.dumps(train_report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
