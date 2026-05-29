#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_${SPLIT}.jsonl}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_doc_fusion_devsplit_probe}"
LABEL="${LABEL:-mmqa_${SPLIT}_doc_fusion_devsplit}"

DENSE_PRED="${DENSE_PRED:-${M3DOCVQA_DENSE_PRED:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json}}"
SPARSE_PRED="${SPARSE_PRED:-${M3DOCVQA_SPLADE_PRED:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json}}"

mkdir -p "$OUT_DIR"

sources=()
add_source_if_exists() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    sources+=(--source "$label=$path")
  else
    echo "skip_missing_source_${label}=$path" >&2
  fi
}

add_source_if_exists dense "$DENSE_PRED"
add_source_if_exists splade "$SPARSE_PRED"
add_source_if_exists gpp_no_hyperlink "${GPP_NO_HYPERLINK_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"
add_source_if_exists gpp_doc_hyperlink "${GPP_DOC_HYPERLINK_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation/mmqa_dev_gpp_hyperlink_node_docnode_to_hyperlink_docs.prediction.json}"
add_source_if_exists gpp_page_hyperlink "${GPP_PAGE_HYPERLINK_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation/mmqa_dev_gpp_hyperlink_node_pagenode_to_hyperlink_pages.prediction.json}"

source_count=$(( ${#sources[@]} / 2 ))
if [[ "$source_count" -lt 4 ]]; then
  echo "warning_source_count=$source_count ; expected dense, splade, and at least two GPP sources" >&2
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/tune_m3docvqa_doc_fusion_split.py" \
  --gold "$GOLD" \
  "${sources[@]}" \
  --holdout-frac "${HOLDOUT_FRAC:-0.30}" \
  --seed "${SEED:-13}" \
  --top-rows "${TOP_ROWS:-1000}" \
  --source-score-mode "${SOURCE_SCORE_MODE:-rank_score}" \
  --weight-grid "${WEIGHT_GRID:-0,0.25,0.5,1,2,4}" \
  --objective-k "${OBJECTIVE_K:-4}" \
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json" \
  --output-table-md "$OUT_DIR/${LABEL}.table.md" \
  --output-prediction-json "$OUT_DIR/${LABEL}.eval.prediction.json" \
  --prediction-scope eval \
  --output-train-qids "$OUT_DIR/${LABEL}.train_qids.txt" \
  --output-eval-qids "$OUT_DIR/${LABEL}.eval_qids.txt"
