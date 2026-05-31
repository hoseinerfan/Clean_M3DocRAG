#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

declare -A USER_DENSE_PRED_OVERRIDES=()
for name in \
  MMDOCIR_DENSE_PRED \
  SCIEGQA_DENSE_PRED \
  VIDOSEEK_DENSE_PRED \
  DUDE_DENSE_PRED \
  VIDORE_DENSE_PRED \
  OPENDOCVQA_DENSE_PRED \
  MMLONGBENCH_DENSE_PRED; do
  if [[ -n "${!name:-}" ]]; then
    USER_DENSE_PRED_OVERRIDES["$name"]="${!name}"
  fi
done

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi
for name in "${!USER_DENSE_PRED_OVERRIDES[@]}"; do
  export "$name=${USER_DENSE_PRED_OVERRIDES[$name]}"
done
unset name USER_DENSE_PRED_OVERRIDES

DATASETS="${DATASETS:-dude sciegqa mmdocir}"
REPORT_OUT="${REPORT_OUT:-$REPO_ROOT/doc_embedding_cosine_ablation_results.md}"
REPORT_CSV_OUT="${REPORT_CSV_OUT:-$REPO_ROOT/doc_embedding_cosine_ablation_results.csv}"

DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}"
RESTART_PROB="${RESTART_PROB:-0.15}"
PPR_ITERS="${PPR_ITERS:-30}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-0.5}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-0.25}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"

DOC_DOC_EDGE_WEIGHT="${DOC_DOC_EDGE_WEIGHT:-0.10}"
DOC_DOC_TOP_DOCS="${DOC_DOC_TOP_DOCS:-20}"
DOC_DOC_MAX_EDGES_PER_DOC="${DOC_DOC_MAX_EDGES_PER_DOC:-8}"
DOC_DOC_EMBEDDING_POOLING="${DOC_DOC_EMBEDDING_POOLING:-page_seed_weighted_mean}"
DOC_DOC_EMBEDDING_PAGE_TOP_K="${DOC_DOC_EMBEDDING_PAGE_TOP_K:-3}"
DOC_DOC_EMBEDDING_CACHE_DOCS="${DOC_DOC_EMBEDDING_CACHE_DOCS:-128}"
DOC_DOC_MIN_SEMANTIC_SIMILARITY="${DOC_DOC_MIN_SEMANTIC_SIMILARITY:-0.35}"
DOC_DOC_EMBEDDING_MUTUAL_TOP_K="${DOC_DOC_EMBEDDING_MUTUAL_TOP_K:-3}"
DOC_DOC_RESCUE_ANCHOR_TOP_K="${DOC_DOC_RESCUE_ANCHOR_TOP_K:-4}"
DOC_DOC_RESCUE_RANK_MIN="${DOC_DOC_RESCUE_RANK_MIN:-5}"
DOC_DOC_RESCUE_RANK_MAX="${DOC_DOC_RESCUE_RANK_MAX:-20}"
DOC_DOC_CONFIDENCE_MODE="${DOC_DOC_CONFIDENCE_MODE:-top_disagreement_or_margin}"
DOC_DOC_CONFIDENCE_MARGIN="${DOC_DOC_CONFIDENCE_MARGIN:-0.05}"
DOC_EMBED_RUN_BASELINE="${DOC_EMBED_RUN_BASELINE:-1}"
DOC_EMBED_RUN_RAW_VARIANTS="${DOC_EMBED_RUN_RAW_VARIANTS:-1}"
DOC_EMBED_RUN_DENSE_SPARSE_GATED="${DOC_EMBED_RUN_DENSE_SPARSE_GATED:-1}"
DOC_EMBED_RUN_SEMANTIC_GATED="${DOC_EMBED_RUN_SEMANTIC_GATED:-1}"
DOC_EMBED_RUN_RESCUE_GATED="${DOC_EMBED_RUN_RESCUE_GATED:-1}"
DOC_EMBED_OUTPUT_SUBDIR="${DOC_EMBED_OUTPUT_SUBDIR:-doc_embedding_cosine_ablation}"
DOC_EMBED_LABEL_SUFFIX="${DOC_EMBED_LABEL_SUFFIX:-}"

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

resolve_embedding_dir() {
  local upper="$1"
  local default_value="$2"
  local dataset_name="${upper}_DOC_DOC_PAGE_EMBEDDING_DIR"

  if [[ -n "${DOC_EMBED_COSINE_PAGE_EMBEDDING_DIR:-}" ]]; then
    printf '%s\n' "$DOC_EMBED_COSINE_PAGE_EMBEDDING_DIR"
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

run_variant() {
  local data_name="$1"
  local data_root="$2"
  local gold="$3"
  local doc_pages="$4"
  local dense_pred="$5"
  local sparse_pred="$6"
  local out_dir="$7"
  local label_prefix="$8"
  local variant="$9"
  local edge_mode="${10}"
  local edge_weight="${11}"
  local page_embedding_dir="${12}"
  local min_similarity="${13}"
  local splade_index="${14:-}"

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
  PPR_ITERS="$PPR_ITERS" \
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
  EXTERNAL_PAGE_GRAPH_JSONL= \
  EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT=0.0 \
  DOC_DOC_EDGE_MODE="$edge_mode" \
  DOC_DOC_EDGE_WEIGHT="$edge_weight" \
  DOC_DOC_TOP_DOCS="$DOC_DOC_TOP_DOCS" \
  DOC_DOC_MAX_EDGES_PER_DOC="$DOC_DOC_MAX_EDGES_PER_DOC" \
  DOC_DOC_PAGE_EMBEDDING_DIR="$page_embedding_dir" \
  DOC_DOC_EMBEDDING_POOLING="$DOC_DOC_EMBEDDING_POOLING" \
  DOC_DOC_EMBEDDING_PAGE_TOP_K="$DOC_DOC_EMBEDDING_PAGE_TOP_K" \
  DOC_DOC_EMBEDDING_MIN_SIMILARITY="$min_similarity" \
  DOC_DOC_EMBEDDING_CACHE_DOCS="$DOC_DOC_EMBEDDING_CACHE_DOCS" \
  DOC_DOC_EMBEDDING_MUTUAL_TOP_K="$DOC_DOC_EMBEDDING_MUTUAL_TOP_K" \
  DOC_DOC_RESCUE_ANCHOR_TOP_K="$DOC_DOC_RESCUE_ANCHOR_TOP_K" \
  DOC_DOC_RESCUE_RANK_MIN="$DOC_DOC_RESCUE_RANK_MIN" \
  DOC_DOC_RESCUE_RANK_MAX="$DOC_DOC_RESCUE_RANK_MAX" \
  DOC_DOC_CONFIDENCE_MODE="$DOC_DOC_CONFIDENCE_MODE" \
  DOC_DOC_CONFIDENCE_MARGIN="$DOC_DOC_CONFIDENCE_MARGIN" \
  DOC_DOC_MIN_SEMANTIC_SIMILARITY="$DOC_DOC_MIN_SEMANTIC_SIMILARITY" \
  SPLADE_INDEX_PT="$splade_index" \
  EXPANSION_TOP_PAGES=0 \
  NEIGHBOR_EXPANSION_WINDOW=0 \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
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
  local out_dir
  local page_embedding_dir
  local sparse_dir
  local splade_index

  label_prefix="${label_prefix}${DOC_EMBED_LABEL_SUFFIX}"
  data_root="$(dirname "$gold")"
  out_dir="$work_root/output/$output_slug/$DOC_EMBED_OUTPUT_SUBDIR"
  page_embedding_dir="$(resolve_embedding_dir "$upper" "$work_root/embeddings/$embedding_name")"
  sparse_dir="$(cd "$(dirname "$sparse_pred")" && pwd)"
  splade_index="${SPLADE_INDEX_PT:-$sparse_dir/${data_name}_splade_page_index.pt}"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  require_dir page_embedding_dir "$page_embedding_dir"
  mkdir -p "$out_dir"

  echo
  echo "== $display_name doc embedding cosine ablation =="
  echo "using_page_embedding_dir=$page_embedding_dir"
  echo "using_splade_index=$splade_index"
  echo "using_doc_doc_embedding_pooling=$DOC_DOC_EMBEDDING_POOLING"
  echo "using_doc_doc_embedding_page_top_k=$DOC_DOC_EMBEDDING_PAGE_TOP_K"
  echo "using_doc_doc_rescue_gate=mutual_top_k:$DOC_DOC_EMBEDDING_MUTUAL_TOP_K anchor_top_k:$DOC_DOC_RESCUE_ANCHOR_TOP_K rescue_rank:${DOC_DOC_RESCUE_RANK_MIN}-${DOC_DOC_RESCUE_RANK_MAX} confidence:$DOC_DOC_CONFIDENCE_MODE margin:$DOC_DOC_CONFIDENCE_MARGIN"

  if [[ "$DOC_EMBED_RUN_BASELINE" == "1" ]]; then
    run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" no_doc_embed none 0.0 "" 0.0
  fi
  if [[ "$DOC_EMBED_RUN_RAW_VARIANTS" == "1" ]]; then
    run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" doc_embed_cosine_sim0p00 page_embedding_cosine "$DOC_DOC_EDGE_WEIGHT" "$page_embedding_dir" 0.0
    run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" doc_embed_cosine_sim0p50 page_embedding_cosine "$DOC_DOC_EDGE_WEIGHT" "$page_embedding_dir" 0.50
  fi
  if [[ "$DOC_EMBED_RUN_DENSE_SPARSE_GATED" == "1" ]]; then
    run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" doc_embed_cosine_dense_sparse_gated_sim0p50 \
      page_embedding_cosine_dense_sparse_gated "$DOC_DOC_EDGE_WEIGHT" "$page_embedding_dir" 0.50
  fi
  if [[ "$DOC_EMBED_RUN_SEMANTIC_GATED" == "1" ]]; then
    if [[ -f "$splade_index" ]]; then
      run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
        "$out_dir" "$label_prefix" doc_embed_cosine_semantic_gated_sim0p50 \
        page_embedding_cosine_semantic_gated "$DOC_DOC_EDGE_WEIGHT" "$page_embedding_dir" 0.50 "$splade_index"
    else
      echo "skip_doc_embed_cosine_semantic_gated_missing_splade_index: $splade_index" >&2
    fi
  fi
  if [[ "$DOC_EMBED_RUN_RESCUE_GATED" == "1" ]]; then
    run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "$label_prefix" doc_embed_cosine_rescue_gated_sim0p50 \
      page_embedding_cosine_rescue_gated "$DOC_DOC_EDGE_WEIGHT" "$page_embedding_dir" 0.50
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
      run_dataset "MMDocIR" mmdocir MMDOCIR mmdocir_doc_embed_cosine \
        "$MMDocIR_WORK_ROOT" mmdocir colpali-v1.2_mm-docir_dev \
        "$MMDOCIR_GOLD" "$MMDOCIR_DOC_PAGES" "$MMDOCIR_DENSE_PRED" "$MMDOCIR_SPARSE_PRED"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      require_value SCIEGQA_GOLD
      require_value SCIEGQA_DOC_PAGES
      require_value SCIEGQA_DENSE_PRED
      require_value SCIEGQA_SPARSE_PRED
      run_dataset "SciEGQA" sciegqa SCIEGQA sciegqa_doc_embed_cosine \
        "$SciEGQA_WORK_ROOT" sciegqa colpali-v1.2_sci-egqa-bench_dev \
        "$SCIEGQA_GOLD" "$SCIEGQA_DOC_PAGES" "$SCIEGQA_DENSE_PRED" "$SCIEGQA_SPARSE_PRED"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      require_value VIDOSEEK_GOLD
      require_value VIDOSEEK_DOC_PAGES
      require_value VIDOSEEK_DENSE_PRED
      require_value VIDOSEEK_SPARSE_PRED
      run_dataset "ViDoSeek" vidoseek VIDOSEEK vidoseek_doc_embed_cosine \
        "$VIDOSEEK_WORK_ROOT" vidoseek colpali-v1.2_vidoseek_dev \
        "$VIDOSEEK_GOLD" "$VIDOSEEK_DOC_PAGES" "$VIDOSEEK_DENSE_PRED" "$VIDOSEEK_SPARSE_PRED"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      require_value DUDE_GOLD
      require_value DUDE_DOC_PAGES
      require_value DUDE_DENSE_PRED
      require_value DUDE_SPARSE_PRED
      run_dataset "DUDE" dude DUDE dude_doc_embed_cosine \
        "$DUDE_WORK_ROOT" dude colpali-v1.2_dude_dev \
        "$DUDE_GOLD" "$DUDE_DOC_PAGES" "$DUDE_DENSE_PRED" "$DUDE_SPARSE_PRED"
      ;;
    vidore|vidore-v3)
      require_value VIDORE_WORK_ROOT
      require_value VIDORE_GOLD
      require_value VIDORE_DOC_PAGES
      require_value VIDORE_DENSE_PRED
      require_value VIDORE_SPARSE_PRED
      run_dataset "ViDoRe-V3" vidore-v3 VIDORE vidore_doc_embed_cosine \
        "$VIDORE_WORK_ROOT" vidore-v3 colpali-v1.2_vidore-v3_dev \
        "$VIDORE_GOLD" "$VIDORE_DOC_PAGES" "$VIDORE_DENSE_PRED" "$VIDORE_SPARSE_PRED"
      ;;
    opendocvqa)
      require_value OPENDOCVQA_WORK_ROOT
      require_value OPENDOCVQA_GOLD
      require_value OPENDOCVQA_DOC_PAGES
      require_value OPENDOCVQA_DENSE_PRED
      require_value OPENDOCVQA_SPARSE_PRED
      run_dataset "OpenDocVQA" opendocvqa OPENDOCVQA opendocvqa_doc_embed_cosine \
        "$OPENDOCVQA_WORK_ROOT" opendocvqa colpali-v1.2_opendocvqa_dev \
        "$OPENDOCVQA_GOLD" "$OPENDOCVQA_DOC_PAGES" "$OPENDOCVQA_DENSE_PRED" "$OPENDOCVQA_SPARSE_PRED"
      ;;
    mmlongbench|mmlongbench-docqa)
      require_value MMLONGBENCH_WORK_ROOT
      require_value MMLONGBENCH_GOLD
      require_value MMLONGBENCH_DOC_PAGES
      require_value MMLONGBENCH_DENSE_PRED
      require_value MMLONGBENCH_SPARSE_PRED
      run_dataset "MMLongBench DocQA" mmlongbench-docqa MMLONGBENCH mmlongbench_doc_embed_cosine \
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
