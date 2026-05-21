#!/usr/bin/env bash
#SBATCH --job-name=opendocvqa-ocr
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-63%8
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/logs/ocr_%A_%a.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/logs/ocr_%A_%a.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

OCR_ENGINE="${OCR_ENGINE:-tesseract}"
OCR_BIN="${OCR_BIN:-tesseract}"
if [[ "$OCR_ENGINE" == "tesseract" ]]; then
  if ! command -v "$OCR_BIN" >/dev/null 2>&1; then
    echo "OCR binary not found: $OCR_BIN" >&2
    echo "Load/install Tesseract, set OCR_BIN, or use OCR_ENGINE=easyocr." >&2
    exit 1
  fi
elif [[ "$OCR_ENGINE" == "easyocr" ]]; then
  "$PYTHON_BIN" - <<'PY'
import easyocr
print(f"easyocr={easyocr.__version__}")
PY
else
  echo "Unsupported OCR_ENGINE=$OCR_ENGINE. Use tesseract or easyocr." >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-64}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -z "${OCR_LANG:-}" ]]; then
  if [[ "$OCR_ENGINE" == "easyocr" ]]; then
    OCR_LANG="en"
  else
    OCR_LANG="eng"
  fi
fi
OCR_PSM="${OCR_PSM:-}"
OCR_TIMEOUT="${OCR_TIMEOUT:-120}"
EASYOCR_GPU="${EASYOCR_GPU:-0}"
MAX_PAGES="${MAX_PAGES:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"

DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/opendocvqa}"
OUT_DIR="${OUT_DIR:-$LOCAL_OUTPUT_DIR/opendocvqa/ocr_page_text_shards}"
OUTPUT_JSONL="$OUT_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}.jsonl"
OUTPUT_SUMMARY="$OUT_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}_summary.json"

mkdir -p "$OUT_DIR"

echo "ocr_shard index=$SHARD_INDEX num_shards=$NUM_SHARDS data_root=$DATA_ROOT output_jsonl=$OUTPUT_JSONL"

OCR_ARGS=(
  --doc-pages-jsonl "$DATA_ROOT/doc_pages_dev.jsonl"
  --image-root "$DATA_ROOT"
  --ocr-image
  --ocr-engine "$OCR_ENGINE"
  --ocr-bin "$OCR_BIN"
  --ocr-lang "$OCR_LANG"
  --ocr-timeout "$OCR_TIMEOUT"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --max-pages "$MAX_PAGES"
  --progress-every "$PROGRESS_EVERY"
  --require-nonempty
  --output-jsonl "$OUTPUT_JSONL"
  --output-summary-json "$OUTPUT_SUMMARY"
)
if [[ -n "$OCR_PSM" ]]; then
  OCR_ARGS+=(--ocr-psm "$OCR_PSM")
fi
if [[ "$OCR_ENGINE" == "easyocr" && "$EASYOCR_GPU" != "0" ]]; then
  OCR_ARGS+=(--easyocr-gpu)
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/export_converted_page_text.py" "${OCR_ARGS[@]}"
