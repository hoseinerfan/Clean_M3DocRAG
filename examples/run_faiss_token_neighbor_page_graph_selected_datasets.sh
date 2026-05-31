#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

DATASETS="${DATASETS:-dude sciegqa}"
REPORT_OUT="${REPORT_OUT:-$REPO_ROOT/faiss_token_neighbor_page_graph_results.md}"
REPORT_CSV_OUT="${REPORT_CSV_OUT:-$REPO_ROOT/faiss_token_neighbor_page_graph_results.csv}"

DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}"
RESTART_PROB="${RESTART_PROB:-0.15}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-0.5}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-0.25}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"

TOKEN_GRAPH_QUERY_FAISS_HIT_K="${TOKEN_GRAPH_QUERY_FAISS_HIT_K:-224}"
TOKEN_GRAPH_SOURCE_TOKEN_TOP_K="${TOKEN_GRAPH_SOURCE_TOKEN_TOP_K:-128}"
TOKEN_GRAPH_NEIGHBOR_TOKEN_K="${TOKEN_GRAPH_NEIGHBOR_TOKEN_K:-10}"
TOKEN_GRAPH_MAX_EDGES_PER_SOURCE_PAGE="${TOKEN_GRAPH_MAX_EDGES_PER_SOURCE_PAGE:-10}"
TOKEN_GRAPH_SOURCE_PAGE_TOP_K="${TOKEN_GRAPH_SOURCE_PAGE_TOP_K:-1000}"
TOKEN_GRAPH_TARGET_PAGE_TOP_K="${TOKEN_GRAPH_TARGET_PAGE_TOP_K:-1000}"
TOKEN_GRAPH_FAISS_NPROBE="${TOKEN_GRAPH_FAISS_NPROBE:-4}"
TOKEN_GRAPH_EDGE_VALUE_MODE="${TOKEN_GRAPH_EDGE_VALUE_MODE:-neighbor}"
TOKEN_GRAPH_EDGE_AGGREGATION="${TOKEN_GRAPH_EDGE_AGGREGATION:-log_count}"
TOKEN_GRAPH_SCORE_NORMALIZATION="${TOKEN_GRAPH_SCORE_NORMALIZATION:-per_source_page}"
TOKEN_GRAPH_REBUILD="${TOKEN_GRAPH_REBUILD:-0}"
TOKEN_GRAPH_PREFLIGHT_ONLY="${TOKEN_GRAPH_PREFLIGHT_ONLY:-0}"
TOKEN_GRAPH_AUTO_EXPORT_QUERY_EMBEDDINGS="${TOKEN_GRAPH_AUTO_EXPORT_QUERY_EMBEDDINGS:-1}"
TOKEN_GRAPH_QUERY_BATCH_SIZE="${TOKEN_GRAPH_QUERY_BATCH_SIZE:-16}"
TOKEN_GRAPH_QUERY_TOKEN_FILTER="${TOKEN_GRAPH_QUERY_TOKEN_FILTER:-full}"
TOKEN_GRAPH_QUERY_EMBEDDING_KEY="${TOKEN_GRAPH_QUERY_EMBEDDING_KEY:-embeddings}"
TOKEN_GRAPH_PAGE_EMBEDDING_KEY="${TOKEN_GRAPH_PAGE_EMBEDDING_KEY:-embeddings}"
TOKEN_GRAPH_RETRIEVAL_MODEL_NAME_OR_PATH="${TOKEN_GRAPH_RETRIEVAL_MODEL_NAME_OR_PATH:-colpaligemma-3b-pt-448-base}"
TOKEN_GRAPH_RETRIEVAL_ADAPTER_MODEL_NAME_OR_PATH="${TOKEN_GRAPH_RETRIEVAL_ADAPTER_MODEL_NAME_OR_PATH:-colpali-v1.2}"
TOKEN_GRAPH_QUERY_EXPORT_DEVICE="${TOKEN_GRAPH_QUERY_EXPORT_DEVICE:-auto}"
TOKEN_GRAPH_RUN_EDGE_VARIANTS="${TOKEN_GRAPH_RUN_EDGE_VARIANTS:-1}"
TOKEN_GRAPH_RUN_CANDIDATE_EXPANSION="${TOKEN_GRAPH_RUN_CANDIDATE_EXPANSION:-1}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_MAX_NEW_PAGES="${TOKEN_GRAPH_CANDIDATE_EXPANSION_MAX_NEW_PAGES:-50}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_MIN_SCORE="${TOKEN_GRAPH_CANDIDATE_EXPANSION_MIN_SCORE:-0.0}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_AGGREGATION="${TOKEN_GRAPH_CANDIDATE_EXPANSION_AGGREGATION:-log_count}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_SCORE_MODE="${TOKEN_GRAPH_CANDIDATE_EXPANSION_SCORE_MODE:-below_min}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_APPEND_AFTER_TOP_K="${TOKEN_GRAPH_CANDIDATE_EXPANSION_APPEND_AFTER_TOP_K:-0}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_MODE="${TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_MODE:-none}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_MIN_SCORE="${TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_MIN_SCORE:--1e30}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_CANDIDATE_POOL="${TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_CANDIDATE_POOL:-0}"
TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFIED_SCORE_MODE="${TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFIED_SCORE_MODE:-verified}"
TOKEN_GRAPH_OUTPUT_SUBDIR="${TOKEN_GRAPH_OUTPUT_SUBDIR:-faiss_token_neighbor_page_graph_ablation}"
TOKEN_GRAPH_GRAPH_SUBDIR="${TOKEN_GRAPH_GRAPH_SUBDIR:-faiss_token_neighbor_page_graph}"
TOKEN_GRAPH_LABEL_SUFFIX="${TOKEN_GRAPH_LABEL_SUFFIX:-}"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing_env: $name. Source hpc_vital_paths.generated.env first." >&2
    exit 1
  fi
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

require_dir() {
  local name="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

has_safetensor_files() {
  local dir="$1"
  find "$dir" -maxdepth 1 -name '*.safetensors' -print -quit | read -r _
}

resolve_query_embedding_dir() {
  local dataset="$1"
  local upper="$2"
  local work_root="$3"
  local embedding_root="$4"
  local override_name="${upper}_QUERY_EMBEDDING_DIR"

  if [[ -n "${TOKEN_GRAPH_QUERY_EMBEDDING_DIR:-}" ]]; then
    printf '%s\n' "$TOKEN_GRAPH_QUERY_EMBEDDING_DIR"
    return
  fi
  if [[ -n "${!override_name:-}" ]]; then
    printf '%s\n' "${!override_name}"
    return
  fi

  local candidates=(
    "$embedding_root/colpali_query_embeddings_dev"
    "$embedding_root/colpali_query_embeddings_${dataset}_dev"
    "$embedding_root/${dataset}_query_embeddings_dev"
    "$work_root/embeddings/colpali_query_embeddings_dev"
    "$work_root/embeddings/colpali_query_embeddings_${dataset}_dev"
    "$work_root/embeddings/${dataset}_query_embeddings_dev"
    "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_query_embeddings_${dataset}_dev"
    "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/${dataset}_query_embeddings_dev"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]] && has_safetensor_files "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  printf '%s\n' "$embedding_root/colpali_query_embeddings_dev"
}

resolve_override_path() {
  local upper="$1"
  local suffix="$2"
  local default_value="$3"
  local global_name="TOKEN_GRAPH_${suffix}"
  local dataset_name="${upper}_${suffix}"

  if [[ -n "${!global_name:-}" ]]; then
    printf '%s\n' "${!global_name}"
    return
  fi
  if [[ -n "${!dataset_name:-}" ]]; then
    printf '%s\n' "${!dataset_name}"
    return
  fi
  printf '%s\n' "$default_value"
}

init_reports() {
  : > "$REPORT_OUT"
  : > "$REPORT_CSV_OUT"
}

append_reports() {
  local display_name="$1"
  local recall_md="$2"
  local recall_csv="$3"

  if [[ -f "$recall_md" ]]; then
    {
      echo "## $display_name"
      echo
      cat "$recall_md"
      echo
    } >> "$REPORT_OUT"
  fi

  if [[ -f "$recall_csv" ]]; then
    if [[ ! -s "$REPORT_CSV_OUT" ]]; then
      head -n 1 "$recall_csv" | awk '{print "dataset," $0}' >> "$REPORT_CSV_OUT"
    fi
    tail -n +2 "$recall_csv" | awk -v dataset="$display_name" '{print dataset "," $0}' >> "$REPORT_CSV_OUT"
  fi
}

run_graph_variant() {
  local data_name="$1"
  local data_root="$2"
  local gold="$3"
  local doc_pages="$4"
  local dense_pred="$5"
  local sparse_pred="$6"
  local out_dir="$7"
  local label_prefix="$8"
  local variant="$9"
  local external_graph_jsonl="${10}"
  local edge_weight="${11}"
  local direction="${12}"
  local max_edges_per_source="${13}"

  local label="${label_prefix}_${variant}"
  echo "== $data_name: $variant =="
  DATA_NAME="$data_name" \
  DATA_ROOT="$data_root" \
  GOLD="$gold" \
  DOC_PAGES_JSONL="$doc_pages" \
  DENSE_PRED="$dense_pred" \
  SPARSE_PRED="$sparse_pred" \
  OUT_DIR="$out_dir" \
  GRAPH_PROFILE=page_rank_probe \
  GRAPH_LABEL="$label" \
  FINAL_TOP_PAGES=1000 \
  PER_DOC_PAGE_LIMIT=0 \
  DENSE_WEIGHT="$DENSE_WEIGHT" \
  SPARSE_WEIGHT="$SPARSE_WEIGHT" \
  DOC_SEED_WEIGHT=0.0 \
  DOC_SEED_MODE=rrf \
  SCORE_SEED_WEIGHT=0.0 \
  RESTART_PROB="$RESTART_PROB" \
  PPR_ITERS=30 \
  PPR_ITERATION_MODE=fixed \
  PAGE_DOC_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  PAGE_TO_DOC_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  DOC_TO_PAGE_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  SAME_DOC_WINDOW="$SAME_DOC_WINDOW" \
  ADJACENT_PAGE_EDGE_WEIGHT="$ADJACENT_PAGE_EDGE_WEIGHT" \
  FINAL_PAGE_SEED_WEIGHT="$FINAL_PAGE_SEED_WEIGHT" \
  FINAL_PPR_PAGE_WEIGHT="$FINAL_PPR_PAGE_WEIGHT" \
  FINAL_PPR_DOC_WEIGHT="$FINAL_PPR_DOC_WEIGHT" \
  ADAPTIVE_SOURCE_WEIGHT_MODE=none \
  ADAPTIVE_RESTART_MODE=none \
  ADAPTIVE_TRANSITION_MODE=none \
  ADAPTIVE_ADJACENT_MODE=none \
  EVIDENCE_COMMUNITY_MODE=none \
  POSITION_EVIDENCE_MODE=none \
  QUERY_ANCHOR_EVIDENCE_MODE=none \
  HEADING_BREADCRUMB_MODE=none \
  ENTITY_ALIAS_MODE=none \
  CONSTRAINT_COMPETITION_MODE=none \
  PDF_HYPERLINK_EDGES_JSONL= \
  PDF_HYPERLINK_EDGE_WEIGHT=0.0 \
  EXTERNAL_PAGE_GRAPH_JSONL="$external_graph_jsonl" \
  EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT="$edge_weight" \
  EXTERNAL_PAGE_GRAPH_DIRECTION="$direction" \
  EXTERNAL_PAGE_GRAPH_WEIGHT_MODE=score \
  EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE="$max_edges_per_source" \
  EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K=1000 \
  EXTERNAL_PAGE_GRAPH_TARGET_TOP_K=1000 \
  DOC_DOC_EDGE_MODE=none \
  DOC_DOC_EDGE_WEIGHT=0.0 \
  EXPANSION_TOP_PAGES=0 \
  NEIGHBOR_EXPANSION_WINDOW=0 \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
}

build_expanded_prediction() {
  local dense_pred="$1"
  local token_graph_jsonl="$2"
  local expanded_pred="$3"
  local expanded_summary="$4"
  local query_embedding_dir="$5"
  local page_embedding_dir="$6"
  local verify_args=()

  if [[ "$TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_MODE" != "none" ]]; then
    verify_args=(
      --verification-mode "$TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_MODE"
      --verify-query-embedding-dir "$query_embedding_dir"
      --verify-page-embedding-dir "$page_embedding_dir"
      --verify-query-embedding-key "$TOKEN_GRAPH_QUERY_EMBEDDING_KEY"
      --verify-page-embedding-key "$TOKEN_GRAPH_PAGE_EMBEDDING_KEY"
      --verification-min-score "$TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_MIN_SCORE"
      --verification-candidate-pool "$TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFICATION_CANDIDATE_POOL"
      --verified-score-mode "$TOKEN_GRAPH_CANDIDATE_EXPANSION_VERIFIED_SCORE_MODE"
    )
  fi

  if [[ ! -s "$expanded_pred" || "$TOKEN_GRAPH_REBUILD" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/expand_prediction_with_external_page_graph.py" \
      --prediction-json "$dense_pred" \
      --external-page-graph-jsonl "$token_graph_jsonl" \
      --output-json "$expanded_pred" \
      --summary-json "$expanded_summary" \
      --max-new-pages-per-qid "$TOKEN_GRAPH_CANDIDATE_EXPANSION_MAX_NEW_PAGES" \
      --min-score "$TOKEN_GRAPH_CANDIDATE_EXPANSION_MIN_SCORE" \
      --aggregation "$TOKEN_GRAPH_CANDIDATE_EXPANSION_AGGREGATION" \
      --synthetic-score-mode "$TOKEN_GRAPH_CANDIDATE_EXPANSION_SCORE_MODE" \
      --append-after-top-k "$TOKEN_GRAPH_CANDIDATE_EXPANSION_APPEND_AFTER_TOP_K" \
      "${verify_args[@]}"
  else
    echo "reusing_expanded_prediction=$expanded_pred"
    echo "set TOKEN_GRAPH_REBUILD=1 to rebuild candidate expansion prediction"
  fi
}

build_token_graph() {
  local dense_pred="$1"
  local query_embedding_dir="$2"
  local page_embedding_dir="$3"
  local doc_ids_json="$4"
  local faiss_index="$5"
  local token_graph_jsonl="$6"
  local token_graph_summary="$7"

  if [[ "$TOKEN_GRAPH_REBUILD" == "1" || ! -s "$token_graph_jsonl" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/build_faiss_token_neighbor_page_graph.py" \
      --prediction-json "$dense_pred" \
      --query-embedding-dir "$query_embedding_dir" \
      --page-embedding-dir "$page_embedding_dir" \
      --doc-ids-json "$doc_ids_json" \
      --faiss-index "$faiss_index" \
      --faiss-nprobe "$TOKEN_GRAPH_FAISS_NPROBE" \
      --output-jsonl "$token_graph_jsonl" \
      --summary-json "$token_graph_summary" \
      --source-page-top-k "$TOKEN_GRAPH_SOURCE_PAGE_TOP_K" \
      --target-page-top-k "$TOKEN_GRAPH_TARGET_PAGE_TOP_K" \
      --query-faiss-hit-k "$TOKEN_GRAPH_QUERY_FAISS_HIT_K" \
      --source-token-top-k "$TOKEN_GRAPH_SOURCE_TOKEN_TOP_K" \
      --neighbor-token-k "$TOKEN_GRAPH_NEIGHBOR_TOKEN_K" \
      --max-edges-per-source-page "$TOKEN_GRAPH_MAX_EDGES_PER_SOURCE_PAGE" \
      --edge-value-mode "$TOKEN_GRAPH_EDGE_VALUE_MODE" \
      --edge-aggregation "$TOKEN_GRAPH_EDGE_AGGREGATION" \
      --score-normalization "$TOKEN_GRAPH_SCORE_NORMALIZATION"
  else
    echo "reusing_token_graph_jsonl=$token_graph_jsonl"
    echo "set TOKEN_GRAPH_REBUILD=1 to rebuild the FAISS token-neighbor edges"
  fi
}

ensure_query_embeddings() {
  local gold="$1"
  local query_embedding_dir="$2"
  local label_prefix="$3"

  if [[ -d "$query_embedding_dir" ]] && has_safetensor_files "$query_embedding_dir"; then
    return
  fi

  if [[ "$TOKEN_GRAPH_AUTO_EXPORT_QUERY_EMBEDDINGS" != "1" ]]; then
    require_dir query_embedding_dir "$query_embedding_dir"
    if ! has_safetensor_files "$query_embedding_dir"; then
      echo "missing_query_embedding_files: $query_embedding_dir" >&2
      exit 1
    fi
    return
  fi

  echo "query_embeddings_missing_or_empty=$query_embedding_dir"
  echo "exporting_query_embeddings=$query_embedding_dir"
  mkdir -p "$query_embedding_dir"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/export_colpali_query_embeddings.py" \
    --gold-jsonl "$gold" \
    --output-dir "$query_embedding_dir" \
    --batch-size "$TOKEN_GRAPH_QUERY_BATCH_SIZE" \
    --query-token-filter "$TOKEN_GRAPH_QUERY_TOKEN_FILTER" \
    --embedding-key "$TOKEN_GRAPH_QUERY_EMBEDDING_KEY" \
    --retrieval-model-name-or-path "$TOKEN_GRAPH_RETRIEVAL_MODEL_NAME_OR_PATH" \
    --retrieval-adapter-model-name-or-path "$TOKEN_GRAPH_RETRIEVAL_ADAPTER_MODEL_NAME_OR_PATH" \
    --device "$TOKEN_GRAPH_QUERY_EXPORT_DEVICE" \
    --resume \
    --summary-json "$query_embedding_dir/${label_prefix}_query_embedding_export.summary.json" \
    --metadata-jsonl "$query_embedding_dir/${label_prefix}_query_embedding_export.metadata.jsonl"
}

run_dataset() {
  local display_name="$1"
  local data_name="$2"
  local upper="$3"
  local label_prefix="$4"
  local work_root="$5"
  local output_slug="$6"
  local embedding_name="$7"
  local gold="$8"
  local doc_pages="$9"
  local dense_pred="${10}"
  local sparse_pred="${11}"
  local data_root
  local embedding_root
  local page_embedding_dir
  local faiss_index
  local doc_ids_json
  local query_embedding_dir
  local out_dir
  local token_graph_dir
  local token_graph_jsonl
  local token_graph_summary
  local expanded_dense_pred
  local expanded_dense_summary

  label_prefix="${label_prefix}${TOKEN_GRAPH_LABEL_SUFFIX}"
  data_root="$(dirname "$gold")"
  embedding_root="${TOKEN_GRAPH_EMBEDDINGS_ROOT:-$work_root/embeddings}"
  page_embedding_dir="$(resolve_override_path "$upper" PAGE_EMBEDDING_DIR "$embedding_root/$embedding_name")"
  faiss_index="$(resolve_override_path "$upper" FAISS_INDEX "$embedding_root/${embedding_name}_pageindex_ivfflat/index.bin")"
  doc_ids_json="$(resolve_override_path "$upper" DOC_IDS_JSON "$data_root/dev_doc_ids.json")"
  query_embedding_dir="$(resolve_query_embedding_dir "$data_name" "$upper" "$work_root" "$embedding_root")"
  out_dir="$work_root/output/$output_slug/$TOKEN_GRAPH_OUTPUT_SUBDIR"
  token_graph_dir="$work_root/output/$output_slug/$TOKEN_GRAPH_GRAPH_SUBDIR"
  token_graph_jsonl="$token_graph_dir/${label_prefix}_faiss_token_neighbor_pages.edges.jsonl"
  token_graph_summary="$token_graph_dir/${label_prefix}_faiss_token_neighbor_pages.summary.json"
  expanded_dense_pred="$token_graph_dir/${label_prefix}_faiss_token_neighbor_expanded_dense_top${TOKEN_GRAPH_CANDIDATE_EXPANSION_MAX_NEW_PAGES}.prediction.json"
  expanded_dense_summary="$token_graph_dir/${label_prefix}_faiss_token_neighbor_expanded_dense_top${TOKEN_GRAPH_CANDIDATE_EXPANSION_MAX_NEW_PAGES}.summary.json"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  require_file doc_ids_json "$doc_ids_json"
  require_file faiss_index "$faiss_index"
  require_dir page_embedding_dir "$page_embedding_dir"
  ensure_query_embeddings "$gold" "$query_embedding_dir" "$label_prefix"

  mkdir -p "$out_dir" "$token_graph_dir"

  echo
  echo "== $display_name FAISS token-neighbor page graph =="
  echo "using_dense_pred=$dense_pred"
  echo "using_sparse_pred=$sparse_pred"
  echo "using_faiss_token_index=$faiss_index"
  echo "using_doc_ids_json=$doc_ids_json"
  echo "using_page_embedding_dir=$page_embedding_dir"
  echo "using_query_embedding_dir=$query_embedding_dir"
  echo "saving_token_graph_jsonl=$token_graph_jsonl"

  if [[ "$TOKEN_GRAPH_PREFLIGHT_ONLY" == "1" ]]; then
    echo "preflight_ok=$display_name"
    return
  fi

  build_token_graph "$dense_pred" "$query_embedding_dir" "$page_embedding_dir" \
    "$doc_ids_json" "$faiss_index" "$token_graph_jsonl" "$token_graph_summary"

  if [[ "$TOKEN_GRAPH_RUN_EDGE_VARIANTS" == "1" ]]; then
    run_graph_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" no_external_graph "" 0.0 as_directed 0
    run_graph_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" faiss_w0p05_directed_top5 "$token_graph_jsonl" 0.05 as_directed 5
    run_graph_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" faiss_w0p05_bidir_top5 "$token_graph_jsonl" 0.05 bidirectional 5
  fi

  if [[ "$TOKEN_GRAPH_RUN_CANDIDATE_EXPANSION" == "1" ]]; then
    build_expanded_prediction "$dense_pred" "$token_graph_jsonl" \
      "$expanded_dense_pred" "$expanded_dense_summary" \
      "$query_embedding_dir" "$page_embedding_dir"
    run_graph_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$expanded_dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" "faiss_candidate_expand_top${TOKEN_GRAPH_CANDIDATE_EXPANSION_MAX_NEW_PAGES}" "" 0.0 as_directed 0
  fi

  summary_paths=( "$out_dir/${label_prefix}_"*.summary.json )
  if [[ -e "${summary_paths[0]}" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
      --recall-table \
      --format markdown \
      "${summary_paths[@]}" \
      | tee "$out_dir/${label_prefix}_recall_table.md"

    "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
      --recall-table \
      --format csv \
      "${summary_paths[@]}" \
      > "$out_dir/${label_prefix}_recall_table.csv"
  fi

  append_reports "$display_name" \
    "$out_dir/${label_prefix}_recall_table.md" \
    "$out_dir/${label_prefix}_recall_table.csv"

  echo "saved_token_graph_jsonl=$token_graph_jsonl"
  echo "saved_token_graph_summary=$token_graph_summary"
  if [[ "$TOKEN_GRAPH_RUN_CANDIDATE_EXPANSION" == "1" ]]; then
    echo "saved_expanded_dense_prediction=$expanded_dense_pred"
    echo "saved_expanded_dense_summary=$expanded_dense_summary"
  fi
}

init_reports

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir)
      require_value MMDocIR_WORK_ROOT
      require_value MMDOCIR_GOLD
      require_value MMDOCIR_DOC_PAGES
      require_value MMDOCIR_DENSE_PRED
      require_value MMDOCIR_SPARSE_PRED
      run_dataset "MMDocIR" mmdocir MMDOCIR mmdocir_faiss_token_neighbor \
        "$MMDocIR_WORK_ROOT" mmdocir colpali-v1.2_mm-docir_dev \
        "$MMDOCIR_GOLD" "$MMDOCIR_DOC_PAGES" "$MMDOCIR_DENSE_PRED" "$MMDOCIR_SPARSE_PRED"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      require_value SCIEGQA_GOLD
      require_value SCIEGQA_DOC_PAGES
      require_value SCIEGQA_DENSE_PRED
      require_value SCIEGQA_SPARSE_PRED
      run_dataset "SciEGQA" sciegqa SCIEGQA sciegqa_faiss_token_neighbor \
        "$SciEGQA_WORK_ROOT" sciegqa colpali-v1.2_sci-egqa-bench_dev \
        "$SCIEGQA_GOLD" "$SCIEGQA_DOC_PAGES" "$SCIEGQA_DENSE_PRED" "$SCIEGQA_SPARSE_PRED"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      require_value VIDOSEEK_GOLD
      require_value VIDOSEEK_DOC_PAGES
      require_value VIDOSEEK_DENSE_PRED
      require_value VIDOSEEK_SPARSE_PRED
      run_dataset "ViDoSeek" vidoseek VIDOSEEK vidoseek_faiss_token_neighbor \
        "$VIDOSEEK_WORK_ROOT" vidoseek colpali-v1.2_vidoseek_dev \
        "$VIDOSEEK_GOLD" "$VIDOSEEK_DOC_PAGES" "$VIDOSEEK_DENSE_PRED" "$VIDOSEEK_SPARSE_PRED"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      require_value DUDE_GOLD
      require_value DUDE_DOC_PAGES
      require_value DUDE_DENSE_PRED
      require_value DUDE_SPARSE_PRED
      run_dataset "DUDE" dude DUDE dude_faiss_token_neighbor \
        "$DUDE_WORK_ROOT" dude colpali-v1.2_dude_dev \
        "$DUDE_GOLD" "$DUDE_DOC_PAGES" "$DUDE_DENSE_PRED" "$DUDE_SPARSE_PRED"
      ;;
    vidore|vidore-v3)
      require_value VIDORE_WORK_ROOT
      require_value VIDORE_GOLD
      require_value VIDORE_DOC_PAGES
      require_value VIDORE_DENSE_PRED
      require_value VIDORE_SPARSE_PRED
      run_dataset "ViDoRe-V3" vidore-v3 VIDORE vidore_faiss_token_neighbor \
        "$VIDORE_WORK_ROOT" vidore-v3 colpali-v1.2_vidore-v3_dev \
        "$VIDORE_GOLD" "$VIDORE_DOC_PAGES" "$VIDORE_DENSE_PRED" "$VIDORE_SPARSE_PRED"
      ;;
    opendocvqa)
      require_value OPENDOCVQA_WORK_ROOT
      require_value OPENDOCVQA_GOLD
      require_value OPENDOCVQA_DOC_PAGES
      require_value OPENDOCVQA_DENSE_PRED
      require_value OPENDOCVQA_SPARSE_PRED
      run_dataset "OpenDocVQA" opendocvqa OPENDOCVQA opendocvqa_faiss_token_neighbor \
        "$OPENDOCVQA_WORK_ROOT" opendocvqa colpali-v1.2_opendocvqa_dev \
        "$OPENDOCVQA_GOLD" "$OPENDOCVQA_DOC_PAGES" "$OPENDOCVQA_DENSE_PRED" "$OPENDOCVQA_SPARSE_PRED"
      ;;
    mmlongbench|mmlongbench-docqa)
      require_value MMLONGBENCH_WORK_ROOT
      require_value MMLONGBENCH_GOLD
      require_value MMLONGBENCH_DOC_PAGES
      require_value MMLONGBENCH_DENSE_PRED
      require_value MMLONGBENCH_SPARSE_PRED
      run_dataset "MMLongBench DocQA" mmlongbench-docqa MMLONGBENCH mmlongbench_faiss_token_neighbor \
        "$MMLONGBENCH_WORK_ROOT" mmlongbench-docqa colpali-v1.2_mmlongbench-docqa_dev \
        "$MMLONGBENCH_GOLD" "$MMLONGBENCH_DOC_PAGES" "$MMLONGBENCH_DENSE_PRED" "$MMLONGBENCH_SPARSE_PRED"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

echo "saved_report_md=$REPORT_OUT"
echo "saved_report_csv=$REPORT_CSV_OUT"
