#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$REPO_ROOT/hpc_vital_paths.generated.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hpc_vital_paths.generated.env"
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

SPLIT="${SPLIT:-dev}"
DATA_ROOT="${M3DOCVQA_DATA_ROOT:-$REPO_ROOT/data/m3-docvqa}"
LABELS_JSONL="${LABELS_JSONL:-$REPO_ROOT/output/m3docvqa_mmqa_evidence_unit_aware_pseudo_page_labels/mmqa_${SPLIT}_evidence_unit_aware_hybrid_v3.jsonl}"
MMQA_IMAGES_JSONL="${MMQA_IMAGES_JSONL:-$DATA_ROOT/multimodalqa/MMQA_images.jsonl}"
MMQA_IMAGE_ROOT="${MMQA_IMAGE_ROOT:-$DATA_ROOT/multimodalqa/final_dataset_images}"
PDF_DIR="${PDF_DIR:-$DATA_ROOT/splits/pdfs_${SPLIT}}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_image_page_vlm_verification_${SPLIT}}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUT_DIR/image_page_verification.jsonl}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$OUT_DIR/image_page_verification.summary.json}"
VLM_MODEL_NAME_OR_PATH="${VLM_MODEL_NAME_OR_PATH:-Qwen2-VL-7B-Instruct}"
VLM_BITS="${VLM_BITS:-16}"
LIMIT="${LIMIT:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-25}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_DIR"

args=(
  --labels-jsonl "$LABELS_JSONL"
  --mmqa-images-jsonl "$MMQA_IMAGES_JSONL"
  --mmqa-image-root "$MMQA_IMAGE_ROOT"
  --pdf-dir "$PDF_DIR"
  --vlm-model-name-or-path "$VLM_MODEL_NAME_OR_PATH"
  --vlm-bits "$VLM_BITS"
  --limit "$LIMIT"
  --sample-mode "$SAMPLE_MODE"
  --seed "$SEED"
  --save-every "$SAVE_EVERY"
  --output-jsonl "$OUTPUT_JSONL"
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
)

if [[ -n "$DOC_PAGES_JSONL" ]]; then
  args+=(--doc-pages-jsonl "$DOC_PAGES_JSONL")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

echo "using_labels_jsonl=$LABELS_JSONL"
echo "using_mmqa_images_jsonl=$MMQA_IMAGES_JSONL"
echo "using_mmqa_image_root=$MMQA_IMAGE_ROOT"
echo "using_pdf_dir=$PDF_DIR"
echo "using_output_jsonl=$OUTPUT_JSONL"
echo "using_limit=$LIMIT"
echo "using_sample_mode=$SAMPLE_MODE"
echo "using_seed=$SEED"
echo "using_dry_run=$DRY_RUN"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_mmqa_image_page_vlm_verifier.py" "${args[@]}"
