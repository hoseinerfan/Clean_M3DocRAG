#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

unset M3DOCVQA_INTERNAL_ENV_LOADED
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

first_existing_path() {
  local path
  for path in "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

SPLIT="${SPLIT:-dev}"
CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
GOLD="${M3DOCVQA_GOLD:-$GOLD}"
DENSE_PRED="${M3DOCVQA_DENSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_${SPLIT}/mmqa_${SPLIT}_plain_top224_nprobe${FAISS_NPROBE}_effdiag_all.prediction.json}"
SPARSE_PRED="${M3DOCVQA_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_${SPLIT}/mmqa_${SPLIT}_splade.prediction.json}"
DOC_DOC_TOP_DOCS="${DOC_DOC_TOP_DOCS:-20}"
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_doc_doc_edge_ablation}"
LABEL_PREFIX="${LABEL_PREFIX:-mmqa_${SPLIT}_hyperlink_file_sanity}"

PDF_HYPERLINK_EDGES_JSONL="$(
  first_existing_path \
    "${M3DOCVQA_PDF_HYPERLINK_EDGES_JSONL:-}" \
    "${PDF_HYPERLINK_EDGES_JSONL:-}" \
    "${MMDOCIR_PDF_HYPERLINK_EDGES_JSONL:-}" \
    "$LOCAL_OUTPUT_DIR/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl" \
    "$REPO_ROOT/output/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl" \
    "$CUSTOM_ROOT/MMDocIR_M3DocRAG/output/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl" \
    || true
)"

if [[ -z "$PDF_HYPERLINK_EDGES_JSONL" || ! -f "$PDF_HYPERLINK_EDGES_JSONL" ]]; then
  echo "missing_m3docvqa_pdf_hyperlink_edges_jsonl" >&2
  echo "set M3DOCVQA_PDF_HYPERLINK_EDGES_JSONL=/path/to/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl" >&2
  exit 1
fi
if [[ ! -f "$GOLD" ]]; then
  echo "missing_gold: $GOLD" >&2
  exit 1
fi
if [[ ! -f "$DENSE_PRED" ]]; then
  echo "missing_dense_pred: $DENSE_PRED" >&2
  exit 1
fi
if [[ ! -f "$SPARSE_PRED" ]]; then
  echo "missing_sparse_pred: $SPARSE_PRED" >&2
  exit 1
fi

mkdir -p "$GRAPH_OUT_DIR"
ls -lh "$PDF_HYPERLINK_EDGES_JSONL"
echo "first_edges:"
head -n "${SAMPLE_LINES:-5}" "$PDF_HYPERLINK_EDGES_JSONL"

"$PYTHON_BIN" "$REPO_ROOT/scripts/inspect_pdf_hyperlink_graph.py" \
  --hyperlink-edges-jsonl "$PDF_HYPERLINK_EDGES_JSONL" \
  --gold "$GOLD" \
  --dense-prediction-json "$DENSE_PRED" \
  --sparse-prediction-json "$SPARSE_PRED" \
  --dense-top-pages "${DENSE_TOP_PAGES:-1000}" \
  --sparse-top-pages "${SPARSE_TOP_PAGES:-1000}" \
  --doc-doc-top-docs "$DOC_DOC_TOP_DOCS" \
  --sample "${SAMPLE_LINES:-5}" \
  --output-json "$GRAPH_OUT_DIR/${LABEL_PREFIX}.json" \
  --output-md "$GRAPH_OUT_DIR/${LABEL_PREFIX}.md"
