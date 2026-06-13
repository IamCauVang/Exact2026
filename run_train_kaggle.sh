#!/usr/bin/env bash
set -euo pipefail

# Train NL->FOL LoRA adapter on Kaggle.
# Default model: Qwen/Qwen2.5-7B-Instruct
#
# Usage:
#   DATASET=data/full_data.json ./run_train_kaggle.sh
#   MAX_TRAIN_SAMPLES=200 MAX_VALID_SAMPLES=50 EPOCHS=1 ./run_train_kaggle.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-data/full_data.json}"
SFT_DIR="${SFT_DIR:-data/sft_nl2fol}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lora_qwen25_7b_nl2fol}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPO_URL="${REPO_URL:-https://github.com/IamCauVang/Exact2026.git}"
KERNEL="${KERNEL:-iamcauvang/exactv2-cauvang}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VALID_SAMPLES="${MAX_VALID_SAMPLES:-0}"
NO_SILVER_OPTIONS="${NO_SILVER_OPTIONS:-0}"

BUILD_DIR=".kaggle_train_build"
OUT_DIR="kaggle_train_outputs"
ART_DIR="train_artifacts"
ARTIFACT_ZIP="exact_train_artifacts.zip"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
START_TS="$(date +%s)"

format_duration() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  if [ "$h" -gt 0 ]; then
    printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"
  else
    printf "%02dm:%02ds" "$m" "$s"
  fi
}

elapsed() {
  local now
  now="$(date +%s)"
  format_duration $((now - START_TS))
}

notify_mac() {
  local title="$1"
  local message="$2"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$message\" with title \"$title\"" >/dev/null 2>&1 || true
  fi
}

finish_fail() {
  local e
  e="$(elapsed)"
  echo
  echo "❌ Training failed after $e."
  notify_mac "EXACT training failed" "Training failed after ${e}."
  printf '\a' || true
}

finish_success() {
  local e
  e="$(elapsed)"
  echo
  echo "✅ Training done in $e. Check ${ART_DIR}/"
  notify_mac "EXACT training finished" "Training completed in ${e}."
  printf '\a' || true
}

trap finish_fail ERR

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

KAGGLE_URL="https://www.kaggle.com/code/${KERNEL}"

echo "======================================"
echo "EXACT NL2FOL Train Runner"
echo "MODEL_NAME       = $MODEL_NAME"
echo "DATASET          = $DATASET"
echo "SFT_DIR          = $SFT_DIR"
echo "OUTPUT_DIR       = $OUTPUT_DIR"
echo "MAX_SEQ_LENGTH   = $MAX_SEQ_LENGTH"
echo "EPOCHS           = $EPOCHS"
echo "LEARNING_RATE    = $LEARNING_RATE"
echo "TRAIN_BS         = $TRAIN_BS"
echo "GRAD_ACCUM       = $GRAD_ACCUM"
echo "MAX_TRAIN_SAMPLES= $MAX_TRAIN_SAMPLES"
echo "MAX_VALID_SAMPLES= $MAX_VALID_SAMPLES"
echo "RUN_ID           = $RUN_ID"
echo "KAGGLE URL       = $KAGGLE_URL"
echo "START            = $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

echo "[$(elapsed)] [0/5] Ensure generated files are ignored"
touch .gitignore
grep -qxF "artifacts/" .gitignore || echo "artifacts/" >> .gitignore
grep -qxF "train_artifacts/" .gitignore || echo "train_artifacts/" >> .gitignore
grep -qxF "kaggle_train_outputs/" .gitignore || echo "kaggle_train_outputs/" >> .gitignore
grep -qxF ".kaggle_train_build/" .gitignore || echo ".kaggle_train_build/" >> .gitignore
grep -qxF "*.zip" .gitignore || echo "*.zip" >> .gitignore

echo "[$(elapsed)] [1/5] Commit and push current repo"
git add .
git commit -m "train nl2fol workflow" || true
git push

echo "[$(elapsed)] [2/5] Build temporary Kaggle training job folder"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp kaggle_job/train_runner.py "$BUILD_DIR/runner.py"
cp kaggle_job/kernel-metadata.json "$BUILD_DIR/kernel-metadata.json"

MODEL_NAME="$MODEL_NAME" DATASET="$DATASET" SFT_DIR="$SFT_DIR" OUTPUT_DIR="$OUTPUT_DIR" RUN_ID="$RUN_ID" \
REPO_URL="$REPO_URL" KERNEL="$KERNEL" MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH" EPOCHS="$EPOCHS" \
LEARNING_RATE="$LEARNING_RATE" TRAIN_BS="$TRAIN_BS" EVAL_BS="$EVAL_BS" GRAD_ACCUM="$GRAD_ACCUM" \
LORA_R="$LORA_R" LORA_ALPHA="$LORA_ALPHA" LORA_DROPOUT="$LORA_DROPOUT" \
MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" MAX_VALID_SAMPLES="$MAX_VALID_SAMPLES" NO_SILVER_OPTIONS="$NO_SILVER_OPTIONS" \
python - <<'PY_PATCH'
from pathlib import Path
import json, os, re

runner = Path('.kaggle_train_build/runner.py')
s = runner.read_text()
keys = [
    'REPO_URL','MODEL_NAME','DATASET','SFT_DIR','OUTPUT_DIR','RUN_ID','MAX_SEQ_LENGTH','EPOCHS',
    'LEARNING_RATE','TRAIN_BS','EVAL_BS','GRAD_ACCUM','LORA_R','LORA_ALPHA','LORA_DROPOUT',
    'MAX_TRAIN_SAMPLES','MAX_VALID_SAMPLES','NO_SILVER_OPTIONS'
]
for key in keys:
    val = os.environ.get(key, '')
    pat = rf'{key}\s*=\s*os\.environ\.get\("{key}",\s*"[^"]*"\)'
    repl = f'{key} = os.environ.get("{key}", "{val}")'
    s = re.sub(pat, repl, s)
runner.write_text(s)

meta = json.loads(Path('.kaggle_train_build/kernel-metadata.json').read_text())
meta['id'] = os.environ.get('KERNEL', meta.get('id', ''))
meta['code_file'] = 'runner.py'
meta['enable_gpu'] = 'true'
meta['enable_internet'] = 'true'
meta.setdefault('is_private', 'false')
Path('.kaggle_train_build/kernel-metadata.json').write_text(json.dumps(meta, indent=2))
PY_PATCH

echo "[$(elapsed)] [3/5] Push Kaggle training kernel"
PUSH_LOG="$(mktemp)"
if ! kaggle kernels push -p "$BUILD_DIR" --accelerator NvidiaTeslaT4 2>&1 | tee "$PUSH_LOG"; then
  echo "[$(elapsed)] [error] Kaggle kernel push failed. Not waiting for training artifact."
  cat "$PUSH_LOG"
  exit 1
fi
if grep -Eiq "Kernel push error|Maximum .* session count|Permission .* denied|error:" "$PUSH_LOG"; then
  echo "[$(elapsed)] [error] Kaggle reported a push error. Not waiting for training artifact."
  cat "$PUSH_LOG"
  exit 1
fi

echo "[$(elapsed)] [4/5] Wait for training artifact"
mkdir -p "$OUT_DIR" "$ART_DIR"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "[$(elapsed)] Trying to download ${ARTIFACT_ZIP}... attempt ${ATTEMPT}/${MAX_ATTEMPTS}"
  rm -rf "${OUT_DIR:?}/"*
  mkdir -p "$OUT_DIR"

  kaggle kernels output "$KERNEL" -p "$OUT_DIR" -o --file-pattern "^${ARTIFACT_ZIP}$" || true
  if [ -f "${OUT_DIR}/${ARTIFACT_ZIP}" ]; then
    echo "[$(elapsed)] [ok] ${ARTIFACT_ZIP} downloaded."
    break
  fi

  echo "[$(elapsed)] Training artifact not ready yet. Sleeping ${SLEEP_SECONDS}s..."
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(elapsed)] [error] Could not download training artifact. Check Kaggle logs: $KAGGLE_URL"
    exit 1
  fi
  sleep "$SLEEP_SECONDS"
done

echo "[$(elapsed)] [5/5] Extract training artifacts"
rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"
unzip -o "${OUT_DIR}/${ARTIFACT_ZIP}" -d "$ART_DIR"

if [ -f "${ART_DIR}/run_id.txt" ]; then
  GOT_RUN_ID="$(cat "${ART_DIR}/run_id.txt" | tr -d '\n\r')"
  if [ "$GOT_RUN_ID" != "$RUN_ID" ]; then
    echo "[warn] Downloaded training artifact run_id=${GOT_RUN_ID}, expected ${RUN_ID}."
  else
    echo "[$(elapsed)] [ok] run_id verified: ${GOT_RUN_ID}"
  fi
fi

if [ -f "${ART_DIR}/adapter/train_report.json" ]; then
  echo "----- train_report.json -----"
  cat "${ART_DIR}/adapter/train_report.json"
  echo
fi

trap - ERR
finish_success
