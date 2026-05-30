#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../scripts/m3docvqa_internal_env.sh"

DENSE_PRED="${DENSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_${SPLIT}/mmqa_${SPLIT}_plain_top224_nprobe${FAISS_NPROBE}_effdiag_all.prediction.json}"
SPARSE_PRED="${SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_${SPLIT}/mmqa_${SPLIT}_splade.prediction.json}"
FAISS_TOKEN_INDEX="${FAISS_TOKEN_INDEX:-$LOCAL_EMBEDDINGS_DIR/$INDEX_NAME/index.bin}"
DOC_IDS_JSON="${DOC_IDS_JSON:-$DATASET_ROOT/${SPLIT}_doc_ids.json}"
PAGE_EMBEDDING_DIR="${PAGE_EMBEDDING_DIR:-$LOCAL_EMBEDDINGS_DIR/$EMBEDDING_NAME}"
QUERY_EMBEDDING_DIR="${QUERY_EMBEDDING_DIR:-$LOCAL_EMBEDDINGS_DIR/colpali_query_embeddings_${SPLIT}}"

TOKEN_GRAPH_OUT_DIR="${TOKEN_GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_faiss_token_neighbor_page_graph}"
TOKEN_GRAPH_LABEL="${TOKEN_GRAPH_LABEL:-mmqa_${SPLIT}_faiss_token_neighbor_pages}"
TOKEN_GRAPH_JSONL="${TOKEN_GRAPH_JSONL:-$TOKEN_GRAPH_OUT_DIR/${TOKEN_GRAPH_LABEL}.edges.jsonl}"
TOKEN_GRAPH_SUMMARY="${TOKEN_GRAPH_SUMMARY:-$TOKEN_GRAPH_OUT_DIR/${TOKEN_GRAPH_LABEL}.summary.json}"

mkdir -p "$TOKEN_GRAPH_OUT_DIR"

echo "using_dense_pred=$DENSE_PRED"
echo "using_sparse_pred=$SPARSE_PRED"
echo "using_faiss_token_index=$FAISS_TOKEN_INDEX"
echo "using_doc_ids_json=$DOC_IDS_JSON"
echo "using_page_embedding_dir=$PAGE_EMBEDDING_DIR"
echo "using_query_embedding_dir=$QUERY_EMBEDDING_DIR"
echo "saving_token_graph_jsonl=$TOKEN_GRAPH_JSONL"

TOKEN_GRAPH_REBUILD="${TOKEN_GRAPH_REBUILD:-0}"
if [[ "$TOKEN_GRAPH_REBUILD" == "1" || ! -s "$TOKEN_GRAPH_JSONL" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_faiss_token_neighbor_page_graph.py" \
    --prediction-json "$DENSE_PRED" \
    --query-embedding-dir "$QUERY_EMBEDDING_DIR" \
    --page-embedding-dir "$PAGE_EMBEDDING_DIR" \
    --doc-ids-json "$DOC_IDS_JSON" \
    --faiss-index "$FAISS_TOKEN_INDEX" \
    --faiss-nprobe "${TOKEN_GRAPH_FAISS_NPROBE:-$FAISS_NPROBE}" \
    --output-jsonl "$TOKEN_GRAPH_JSONL" \
    --summary-json "$TOKEN_GRAPH_SUMMARY" \
    --source-page-top-k "${TOKEN_GRAPH_SOURCE_PAGE_TOP_K:-1000}" \
    --target-page-top-k "${TOKEN_GRAPH_TARGET_PAGE_TOP_K:-1000}" \
    --query-faiss-hit-k "${TOKEN_GRAPH_QUERY_FAISS_HIT_K:-224}" \
    --source-token-top-k "${TOKEN_GRAPH_SOURCE_TOKEN_TOP_K:-128}" \
    --neighbor-token-k "${TOKEN_GRAPH_NEIGHBOR_TOKEN_K:-10}" \
    --max-edges-per-source-page "${TOKEN_GRAPH_MAX_EDGES_PER_SOURCE_PAGE:-10}" \
    --min-source-score "${TOKEN_GRAPH_MIN_SOURCE_SCORE:-0.0}" \
    --min-neighbor-score "${TOKEN_GRAPH_MIN_NEIGHBOR_SCORE:-0.0}" \
    --edge-value-mode "${TOKEN_GRAPH_EDGE_VALUE_MODE:-neighbor}" \
    --edge-aggregation "${TOKEN_GRAPH_EDGE_AGGREGATION:-log_count}" \
    --score-normalization "${TOKEN_GRAPH_SCORE_NORMALIZATION:-per_source_page}"
else
  echo "reusing_token_graph_jsonl=$TOKEN_GRAPH_JSONL"
  echo "set TOKEN_GRAPH_REBUILD=1 to rebuild the FAISS token-neighbor edges"
fi

EXTERNAL_PAGE_GRAPH_JSONL="$TOKEN_GRAPH_JSONL" \
EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT="${EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT:-0.10}" \
EXTERNAL_PAGE_GRAPH_DIRECTION="${EXTERNAL_PAGE_GRAPH_DIRECTION:-bidirectional}" \
EXTERNAL_PAGE_GRAPH_WEIGHT_MODE="${EXTERNAL_PAGE_GRAPH_WEIGHT_MODE:-score}" \
EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE="${EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE:-10}" \
EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K="${EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K:-1000}" \
EXTERNAL_PAGE_GRAPH_TARGET_TOP_K="${EXTERNAL_PAGE_GRAPH_TARGET_TOP_K:-1000}" \
DENSE_PRED="$DENSE_PRED" \
SPARSE_PRED="$SPARSE_PRED" \
GRAPH_LABEL="${GRAPH_LABEL:-mmqa_${SPLIT}_gpp_faiss_token_neighbor_pages}" \
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_gpp_faiss_token_neighbor_pages}" \
bash "$REPO_ROOT/scripts/run_m3docvqa_page_preserving_graph_pipeline.sh"

echo "saved_token_graph_jsonl=$TOKEN_GRAPH_JSONL"
echo "saved_token_graph_summary=$TOKEN_GRAPH_SUMMARY"
