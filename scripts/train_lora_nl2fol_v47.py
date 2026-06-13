#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import get_peft_model

try:
    from trl import SFTTrainer, SFTConfig
    HAS_TRL = True
except Exception:
    SFTTrainer = None  # type: ignore
    SFTConfig = None  # type: ignore
    HAS_TRL = False


def str2bool(v: Any) -> bool:
    return str(v).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_bnb_available() -> bool:
    return importlib.util.find_spec('bitsandbytes') is not None


def choose_precision(force_fp16: bool = False, force_bf16: bool = False):
    if not torch.cuda.is_available():
        return False, False, torch.float32
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if force_bf16 and bf16_supported:
        return True, False, torch.bfloat16
    if force_bf16 and not bf16_supported:
        print('[precision] FORCE_BF16=1 but BF16 not supported; using FP16.', flush=True)
        return False, True, torch.float16
    if force_fp16:
        return False, True, torch.float16
    if bf16_supported:
        return True, False, torch.bfloat16
    return False, True, torch.float16


def messages_to_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return '\n\n'.join(f"### {m.get('role','user').upper()}:\n{m.get('content','')}" for m in messages)


def rows_to_text_dataset(rows: list[dict[str, Any]], tokenizer: Any, min_confidence: float = 0.0) -> Dataset:
    texts = []
    kept = 0
    for row in rows:
        conf = float(row.get('confidence', row.get('weight', 1.0)) or 0.0)
        if conf < min_confidence:
            continue
        if row.get('messages'):
            text = messages_to_text(tokenizer, row['messages'])
        else:
            text = str(row.get('text', '')).strip()
        if text:
            texts.append({'text': text})
            kept += 1
    if not texts:
        raise ValueError('No training texts after filtering. Check input JSONL and min_confidence.')
    print(f'[dataset] kept {kept}/{len(rows)} samples', flush=True)
    return Dataset.from_list(texts)


def filter_kwargs(cls_or_func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(cls_or_func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    except Exception:
        return {k: v for k, v in kwargs.items() if v is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description='Train v4.7 LoRA parser on silver NL->FOL JSONL data.')
    ap.add_argument('--train', default=os.environ.get('TRAIN_FILE', 'data/silver_v47/clean.train.jsonl'))
    ap.add_argument('--valid', default=os.environ.get('VALID_FILE', 'data/silver_v47/clean.valid.jsonl'))
    ap.add_argument('--model', '--model-name', dest='model_name', default=os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct'))
    ap.add_argument('--out', '--output-dir', dest='output_dir', default=os.environ.get('OUTPUT_ADAPTER', 'outputs/adapter_v47'))
    ap.add_argument('--epochs', type=float, default=float(os.environ.get('TRAIN_EPOCHS', '2')))
    ap.add_argument('--max-steps', type=int, default=int(os.environ.get('TRAIN_MAX_STEPS', '-1')))
    ap.add_argument('--lr', type=float, default=float(os.environ.get('LEARNING_RATE', '2e-4')))
    ap.add_argument('--batch-size', type=int, default=int(os.environ.get('TRAIN_BATCH_SIZE', '1')))
    ap.add_argument('--grad-accum', type=int, default=int(os.environ.get('GRAD_ACCUM', '8')))
    ap.add_argument('--max-seq-length', type=int, default=int(os.environ.get('MAX_SEQ_LENGTH', '2048')))
    ap.add_argument('--lora-r', type=int, default=int(os.environ.get('LORA_R', '16')))
    ap.add_argument('--lora-alpha', type=int, default=int(os.environ.get('LORA_ALPHA', '32')))
    ap.add_argument('--min-confidence', type=float, default=float(os.environ.get('MIN_CONFIDENCE', '0.0')))
    ap.add_argument('--use-4bit', type=str2bool, default=str2bool(os.environ.get('USE_4BIT', '1')))
    ap.add_argument('--force-fp16', type=str2bool, default=str2bool(os.environ.get('FORCE_FP16', '0')))
    ap.add_argument('--force-bf16', type=str2bool, default=str2bool(os.environ.get('FORCE_BF16', '0')))
    ap.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '42')))
    args = ap.parse_args()

    set_seed(args.seed)
    train_path = Path(args.train)
    valid_path = Path(args.valid) if args.valid else None
    if not train_path.exists():
        raise FileNotFoundError(f'Train file not found: {train_path}')

    use_bf16, use_fp16, compute_dtype = choose_precision(args.force_fp16, args.force_bf16)
    print(f'[precision] bf16={use_bf16} fp16={use_fp16} compute_dtype={compute_dtype}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit and torch.cuda.is_available() and is_bnb_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif args.use_4bit:
        print('[quant] 4bit requested but bitsandbytes/CUDA unavailable; loading normal model.', flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=compute_dtype if torch.cuda.is_available() and quant_config is None else None,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    if getattr(model.config, 'pad_token_id', None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_rows = read_jsonl(train_path)
    valid_rows = read_jsonl(valid_path) if valid_path and valid_path.exists() else []
    train_ds = rows_to_text_dataset(train_rows, tokenizer, args.min_confidence)
    eval_ds = rows_to_text_dataset(valid_rows, tokenizer, args.min_confidence) if valid_rows else None

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    common_args = dict(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=5,
        save_strategy='epoch',
        eval_strategy='epoch' if eval_ds is not None else 'no',
        evaluation_strategy='epoch' if eval_ds is not None else 'no',
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim='paged_adamw_8bit' if quant_config is not None else 'adamw_torch',
    )

    if HAS_TRL:
        print('[trainer] using TRL SFTTrainer', flush=True)
        trl_args = dict(
            **common_args,
            max_seq_length=args.max_seq_length,
            packing=False,
            dataset_text_field='text',
        )
        cfg = SFTConfig(**filter_kwargs(SFTConfig, trl_args))
        trainer_kwargs = dict(
            model=model,
            args=cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
            tokenizer=tokenizer,
            processing_class=tokenizer,
            dataset_text_field='text',
            max_seq_length=args.max_seq_length,
            packing=False,
        )
        trainer = SFTTrainer(**filter_kwargs(SFTTrainer, trainer_kwargs))
    else:
        print('[trainer] TRL is not installed; using transformers.Trainer fallback', flush=True)
        # Standard causal-LM fine-tuning fallback: tokenize the same text field
        # and let DataCollatorForLanguageModeling create labels=input_ids.
        # This is slightly less ergonomic than TRL SFTTrainer but removes the
        # hard dependency on `trl`, which is absent in some Kaggle images.
        if getattr(model.config, 'use_cache', None) is not None:
            model.config.use_cache = False
        model = get_peft_model(model, peft_config)

        def tokenize_batch(batch):
            return tokenizer(
                batch['text'],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            )

        tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=['text'])
        tokenized_eval = eval_ds.map(tokenize_batch, batched=True, remove_columns=['text']) if eval_ds is not None else None
        hf_args = TrainingArguments(**filter_kwargs(TrainingArguments, common_args))
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=hf_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

    trainer.train()
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    meta = {
        'model_name': args.model_name,
        'train_file': str(train_path),
        'valid_file': str(valid_path) if valid_path else None,
        'num_train_samples': len(train_ds),
        'num_valid_samples': len(eval_ds) if eval_ds is not None else 0,
        'output_dir': str(out),
        'bf16': use_bf16,
        'fp16': use_fp16,
        'use_4bit': quant_config is not None,
    }
    (out / 'v47_train_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
