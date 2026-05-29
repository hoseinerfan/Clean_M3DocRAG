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

DATASETS="${DATASETS:-vidore opendocvqa}"
MAX_DOCS="${MAX_DOCS:-20}"
SAMPLE="${SAMPLE:-20}"

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

subset_doc_pages() {
  local input_jsonl="$1"
  local output_jsonl="$2"
  local max_docs="$3"

  "$PYTHON_BIN" - "$input_jsonl" "$output_jsonl" "$max_docs" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
max_docs = int(sys.argv[3])
selected = set()
page_count = 0
first_rows = []
output_path.parent.mkdir(parents=True, exist_ok=True)
with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        if doc_id not in selected:
            if max_docs > 0 and len(selected) >= max_docs:
                continue
            selected.add(doc_id)
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        page_count += 1
        if len(first_rows) < 3:
            first_rows.append(row)

print(f"subset_doc_count={len(selected)}")
print(f"subset_page_count={page_count}")
for idx, row in enumerate(first_rows):
    interesting = {
        key: row.get(key)
        for key in [
            "doc_id",
            "page_idx",
            "page_uid",
            "image_path",
            "pdf_path",
            "url",
            "source_url",
            "source_doc_id",
            "repo_id",
            "repo_slug",
            "dataset_name",
        ]
        if key in row
    }
    print(f"sample_row_{idx}_keys={sorted(row)}")
    print(f"sample_row_{idx}_interesting={json.dumps(interesting, ensure_ascii=False)}")
PY
}

run_dataset() {
  local display_name="$1"
  local output_slug="$2"
  local work_root="$3"
  local doc_pages="$4"
  local pdf_root="$5"
  local doc_ids_json="$6"
  local id_url_jsonl="$7"
  local label="$8"
  local out_dir="$work_root/output/$output_slug/pdf_hyperlink_sanity"
  local subset_jsonl="$out_dir/${label}_first_${MAX_DOCS}_docs.doc_pages.jsonl"
  local audit_json="$out_dir/${label}_audit_first_${MAX_DOCS}_docs.json"
  local audit_jsonl="$out_dir/${label}_audit_first_${MAX_DOCS}_docs.jsonl"
  local edges_jsonl="$out_dir/${label}_first_${MAX_DOCS}_docs.edges.jsonl"
  local graph_summary_json="$out_dir/${label}_first_${MAX_DOCS}_docs.edges.summary.json"
  local graph_md="$out_dir/${label}_first_${MAX_DOCS}_docs.edges.md"
  local audit_args=()
  local build_args=()

  require_file doc_pages "$doc_pages"
  mkdir -p "$out_dir"

  echo
  echo "== $display_name PDF hyperlink sanity =="
  echo "doc_pages=$doc_pages"
  echo "pdf_root=$pdf_root"
  echo "max_docs=$MAX_DOCS"

  subset_doc_pages "$doc_pages" "$subset_jsonl" "$MAX_DOCS"

  audit_args=(
    --doc-pages-jsonl "$subset_jsonl"
    --pdf-root "$pdf_root"
    --max-docs "$MAX_DOCS"
    --sample "$SAMPLE"
    --output-json "$audit_json"
    --output-jsonl "$audit_jsonl"
  )
  if [[ -n "$id_url_jsonl" && -f "$id_url_jsonl" ]]; then
    audit_args+=(--id-url-jsonl "$id_url_jsonl")
    echo "id_url_jsonl=$id_url_jsonl"
  else
    echo "id_url_jsonl=NONE"
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_pdf_hyperlinks.py" "${audit_args[@]}"

  if [[ ! -s "$audit_jsonl" ]]; then
    echo "no_hyperlink_records_for_subset"
    : > "$edges_jsonl"
    echo "saved_empty_edges=$edges_jsonl"
    echo "saved_audit_json=$audit_json"
    echo "saved_audit_jsonl=$audit_jsonl"
    return 0
  fi

  build_args=(
    --audit-jsonl "$audit_jsonl"
    --output-edges-jsonl "$edges_jsonl"
    --output-summary-json "$graph_summary_json"
    --output-md "$graph_md"
    --sample "$SAMPLE"
  )
  if [[ -n "$doc_ids_json" && -f "$doc_ids_json" ]]; then
    build_args+=(--valid-doc-ids-json "$doc_ids_json")
    echo "valid_doc_ids_json=$doc_ids_json"
  else
    echo "valid_doc_ids_json=NONE"
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_pdf_hyperlink_graph.py" "${build_args[@]}"
  echo "saved_audit_json=$audit_json"
  echo "saved_audit_jsonl=$audit_jsonl"
  echo "saved_edges_jsonl=$edges_jsonl"
}

for dataset in $DATASETS; do
  case "$dataset" in
    dude)
      require_value DUDE_WORK_ROOT
      DUDE_DATA_ROOT="${DUDE_DATA_ROOT:-$DUDE_WORK_ROOT/data/dude}"
      run_dataset \
        "DUDE" \
        "dude" \
        "$DUDE_WORK_ROOT" \
        "${DUDE_DOC_PAGES:-$DUDE_DATA_ROOT/doc_pages_dev.jsonl}" \
        "${DUDE_PDF_ROOT:-$DUDE_DATA_ROOT/raw/DUDE_train-val-test_binaries/PDF}" \
        "${DUDE_DOC_IDS_JSON:-$DUDE_DATA_ROOT/dev_doc_ids.json}" \
        "${DUDE_ID_URL_JSONL:-}" \
        "dude_pdf_hyperlink_sanity"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      VIDOSEEK_DATA_ROOT="${VIDOSEEK_DATA_ROOT:-$VIDOSEEK_WORK_ROOT/data/vidoseek}"
      run_dataset \
        "ViDoSeek" \
        "vidoseek" \
        "$VIDOSEEK_WORK_ROOT" \
        "${VIDOSEEK_DOC_PAGES:-$VIDOSEEK_DATA_ROOT/doc_pages_dev.jsonl}" \
        "${VIDOSEEK_PDF_ROOT:-$VIDOSEEK_DATA_ROOT/pdfs_raw}" \
        "${VIDOSEEK_DOC_IDS_JSON:-$VIDOSEEK_DATA_ROOT/dev_doc_ids.json}" \
        "${VIDOSEEK_ID_URL_JSONL:-}" \
        "vidoseek_pdf_hyperlink_sanity"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      SCIEGQA_DATA_ROOT="${SCIEGQA_DATA_ROOT:-$SciEGQA_WORK_ROOT/data/sci-egqa-bench}"
      run_dataset \
        "SciEGQA" \
        "sciegqa" \
        "$SciEGQA_WORK_ROOT" \
        "${SCIEGQA_DOC_PAGES:-$SCIEGQA_DATA_ROOT/doc_pages_dev.jsonl}" \
        "${SCIEGQA_PDF_ROOT:-$SCIEGQA_DATA_ROOT/images_raw}" \
        "${SCIEGQA_DOC_IDS_JSON:-$SCIEGQA_DATA_ROOT/dev_doc_ids.json}" \
        "${SCIEGQA_ID_URL_JSONL:-}" \
        "sciegqa_pdf_hyperlink_sanity"
      ;;
    mmdocir)
      require_value MMDocIR_WORK_ROOT
      MMDOCIR_DATA_ROOT="${MMDOCIR_DATA_ROOT:-$MMDocIR_WORK_ROOT/data/mm-docir}"
      run_dataset \
        "MMDocIR" \
        "mmdocir" \
        "$MMDocIR_WORK_ROOT" \
        "${MMDOCIR_DOC_PAGES:-$MMDOCIR_DATA_ROOT/doc_pages_dev.jsonl}" \
        "${MMDOCIR_PDF_ROOT:-$MMDOCIR_DATA_ROOT}" \
        "${MMDOCIR_DOC_IDS_JSON:-$MMDOCIR_DATA_ROOT/dev_doc_ids.json}" \
        "${MMDOCIR_ID_URL_JSONL:-}" \
        "mmdocir_pdf_hyperlink_sanity"
      ;;
    mmlongbench|mmlongbench-docqa)
      require_value MMLONGBENCH_WORK_ROOT
      MMLONGBENCH_DATA_ROOT="${MMLONGBENCH_DATA_ROOT:-$MMLONGBENCH_WORK_ROOT/data/mmlongbench-docqa}"
      run_dataset \
        "MMLongBench DocQA" \
        "mmlongbench-docqa" \
        "$MMLONGBENCH_WORK_ROOT" \
        "${MMLONGBENCH_DOC_PAGES:-$MMLONGBENCH_DATA_ROOT/doc_pages_dev.jsonl}" \
        "${MMLONGBENCH_PDF_ROOT:-$MMLONGBENCH_DATA_ROOT}" \
        "${MMLONGBENCH_DOC_IDS_JSON:-$MMLONGBENCH_DATA_ROOT/dev_doc_ids.json}" \
        "${MMLONGBENCH_ID_URL_JSONL:-}" \
        "mmlongbench_pdf_hyperlink_sanity"
      ;;
    vidore|vidore-v3)
      require_value VIDORE_WORK_ROOT
      require_value VIDORE_DOC_PAGES
      run_dataset \
        "ViDoRe-V3" \
        "vidore-v3" \
        "$VIDORE_WORK_ROOT" \
        "$VIDORE_DOC_PAGES" \
        "${VIDORE_PDF_ROOT:-$(dirname "$VIDORE_DOC_PAGES")}" \
        "${VIDORE_DOC_IDS_JSON:-$(dirname "$VIDORE_DOC_PAGES")/dev_doc_ids.json}" \
        "${VIDORE_ID_URL_JSONL:-}" \
        "vidore_pdf_hyperlink_sanity"
      ;;
    opendocvqa)
      require_value OPENDOCVQA_WORK_ROOT
      require_value OPENDOCVQA_DOC_PAGES
      run_dataset \
        "OpenDocVQA" \
        "opendocvqa" \
        "$OPENDOCVQA_WORK_ROOT" \
        "$OPENDOCVQA_DOC_PAGES" \
        "${OPENDOCVQA_PDF_ROOT:-$(dirname "$OPENDOCVQA_DOC_PAGES")}" \
        "${OPENDOCVQA_DOC_IDS_JSON:-$(dirname "$OPENDOCVQA_DOC_PAGES")/dev_doc_ids.json}" \
        "${OPENDOCVQA_ID_URL_JSONL:-}" \
        "opendocvqa_pdf_hyperlink_sanity"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done
