"""Kaggle training wrapper for EXACT NL2FOL LoRA.

This file is copied as runner.py by run_train_kaggle.sh.
It clones the repo, builds SFT data from full_data.json, trains a LoRA adapter, and exports a zip.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/IamCauVang/Exact2026.git")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DATASET = os.environ.get("DATASET", "data/full_data.json")
SFT_DIR = os.environ.get("SFT_DIR", "data/sft_nl2fol")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "artifacts/lora_qwen25_7b_nl2fol")
RUN_ID = os.environ.get("RUN_ID", "")
MAX_SEQ_LENGTH = os.environ.get("MAX_SEQ_LENGTH", "2048")
EPOCHS = os.environ.get("EPOCHS", "2")
LEARNING_RATE = os.environ.get("LEARNING_RATE", "2e-4")
TRAIN_BS = os.environ.get("TRAIN_BS", "1")
EVAL_BS = os.environ.get("EVAL_BS", "1")
GRAD_ACCUM = os.environ.get("GRAD_ACCUM", "8")
LORA_R = os.environ.get("LORA_R", "16")
LORA_ALPHA = os.environ.get("LORA_ALPHA", "32")
LORA_DROPOUT = os.environ.get("LORA_DROPOUT", "0.05")
MAX_TRAIN_SAMPLES = os.environ.get("MAX_TRAIN_SAMPLES", "0")
MAX_VALID_SAMPLES = os.environ.get("MAX_VALID_SAMPLES", "0")
NO_SILVER_OPTIONS = os.environ.get("NO_SILVER_OPTIONS", "0")

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "Exact2026"
TRAIN_OUT = WORK_DIR / "train_outputs"
ZIP_PATH = WORK_DIR / "exact_train_artifacts.zip"


def run(cmd, cwd=None, env=None):
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_shell(cmd, cwd=None):
    print("RUN:", cmd, flush=True)
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)


def main():
    print("=" * 80, flush=True)
    print("EXACT Kaggle NL2FOL Training Runtime", flush=True)
    for k in [
        "REPO_URL", "MODEL_NAME", "DATASET", "SFT_DIR", "OUTPUT_DIR", "RUN_ID",
        "MAX_SEQ_LENGTH", "EPOCHS", "LEARNING_RATE", "TRAIN_BS", "EVAL_BS",
        "GRAD_ACCUM", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "MAX_TRAIN_SAMPLES", "MAX_VALID_SAMPLES",
    ]:
        print(f"{k}={globals()[k]}", flush=True)
    print("=" * 80, flush=True)

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    if TRAIN_OUT.exists():
        shutil.rmtree(TRAIN_OUT)
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)

    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    dataset_path = REPO_DIR / DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Commit/push {DATASET} first.")

    req_train = REPO_DIR / "requirements_train.txt"
    req_kaggle = REPO_DIR / "requirements_kaggle.txt"
    if req_train.exists():
        run(["python", "-m", "pip", "install", "-q", "-r", str(req_train)])
    elif req_kaggle.exists():
        run(["python", "-m", "pip", "install", "-q", "-r", str(req_kaggle)])

    build_cmd = [
        "python", str(REPO_DIR / "scripts" / "build_sft_data.py"),
        "--dataset", str(dataset_path),
        "--out-dir", str(REPO_DIR / SFT_DIR),
        "--seed", "42",
    ]
    if NO_SILVER_OPTIONS == "1":
        build_cmd.append("--no-silver-options")
    run(build_cmd, cwd=REPO_DIR)

    env = os.environ.copy()
    env.update({
        "MODEL_NAME": MODEL_NAME,
        "SFT_DIR": str(REPO_DIR / SFT_DIR),
        "OUTPUT_DIR": str(REPO_DIR / OUTPUT_DIR),
        "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "TRAIN_BS": TRAIN_BS,
        "EVAL_BS": EVAL_BS,
        "GRAD_ACCUM": GRAD_ACCUM,
        "LORA_R": LORA_R,
        "LORA_ALPHA": LORA_ALPHA,
        "LORA_DROPOUT": LORA_DROPOUT,
        "MAX_TRAIN_SAMPLES": MAX_TRAIN_SAMPLES,
        "MAX_VALID_SAMPLES": MAX_VALID_SAMPLES,
    })
    run(["python", str(REPO_DIR / "scripts" / "train_lora_nl2fol.py")], cwd=REPO_DIR, env=env)

    # Export adapter and dataset report.
    adapter_src = REPO_DIR / OUTPUT_DIR
    sft_src = REPO_DIR / SFT_DIR
    if adapter_src.exists():
        shutil.copytree(adapter_src, TRAIN_OUT / "adapter", dirs_exist_ok=True)
    if sft_src.exists():
        shutil.copytree(sft_src, TRAIN_OUT / "sft_data", dirs_exist_ok=True)
    (TRAIN_OUT / "run_id.txt").write_text(RUN_ID, encoding="utf-8")

    print("[artifact] training outputs:", flush=True)
    run_shell("find /kaggle/working/train_outputs -maxdepth 5 -type f -print")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    run_shell("cd /kaggle/working/train_outputs && zip -r /kaggle/working/exact_train_artifacts.zip .")
    print("[artifact] final /kaggle/working files:", flush=True)
    run_shell("find /kaggle/working -maxdepth 2 -type f -print")
    print(f"TRAIN_DONE: {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
