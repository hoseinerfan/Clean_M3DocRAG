# External Dataset Workflows

This file is the central HPC handoff for the external datasets prepared for M3DocRAG retrieval and `plain_top224` evaluation.

Repo root on the UNC HPC:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
```

Current working branch:

```bash
codex/mmdocir-hpc-workflow
```

Update the HPC checkout before using newly added workflows:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase origin codex/mmdocir-hpc-workflow
```

## Environment Rule

Each dataset has its own scratch workspace and `env_hpc.sh`. Always clear the shared path variables before switching datasets:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
```

For Hugging Face-heavy or gated datasets, also clear cache variables before sourcing the env:

```bash
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
```

Then source the target env:

```bash
source <dataset-folder>/env_hpc.sh
```

The env files set:

- `REPO_ROOT`
- dataset-specific work root, such as `MMDocIR_WORK_ROOT` or `VIDORE_WORK_ROOT`
- `LOCAL_DATA_DIR`
- `LOCAL_EMBEDDINGS_DIR`
- `LOCAL_OUTPUT_DIR`
- `LOCAL_MODEL_DIR`
- `PYTHONPATH`

## Question-Type Failure Analysis

Use `mmdocir/analyze_retrieval_by_question_type.py` to compare full/exact MaxSim retrieval against compact `plain_top224` by metadata-defined question type.

For MMDocIR:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmdocir/env_hpc.sh

"$REPO_ROOT/env/bin/python" mmdocir/analyze_retrieval_by_question_type.py \
  --gold "$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl" \
  --exact-pred "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --compact-pred "$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json" \
  --group-field metadata.type \
  --group-field metadata.domain \
  --min-count 5 \
  --output-md "$LOCAL_OUTPUT_DIR/mmdocir/question_type_failure_exact_vs_compact.md" \
  --output-json "$LOCAL_OUTPUT_DIR/mmdocir/question_type_failure_exact_vs_compact.json"
```

For ViDoRe V3:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source vidore/env_hpc.sh

"$REPO_ROOT/env/bin/python" mmdocir/analyze_retrieval_by_question_type.py \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl" \
  --exact-pred "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --compact-pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --group-field metadata.query_types \
  --group-field metadata.query_format \
  --group-field metadata.content_type \
  --group-field metadata.repo_slug \
  --min-count 25 \
  --output-md "$LOCAL_OUTPUT_DIR/vidore-v3/question_type_failure_exact_vs_compact.md" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/question_type_failure_exact_vs_compact.json"
```

The table is sorted by the highest `both_page_miss@4` rate by default. The JSON output also keeps example questions where both systems miss the gold page in the top 4.

To inspect concrete failed qids and gold page image paths, use `mmdocir/show_failed_gold_pages.py`.

MMDocIR metadata-style failures:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/show_failed_gold_pages.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --exact-pred "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --compact-pred "$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json" \
  --where metadata.type=meta-data \
  --fail-mode both_page_miss \
  --max-examples 10 \
  --top-retrieved 3 \
  --copy-gold-pages-dir "$LOCAL_OUTPUT_DIR/mmdocir/failed_gold_pages/meta-data" \
  --output-md "$LOCAL_OUTPUT_DIR/mmdocir/failed_gold_pages_meta-data.md"
```

ViDoRe finance/table-heavy failures:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/show_failed_gold_pages.py \
  --data-root "$LOCAL_DATA_DIR/vidore-v3" \
  --exact-pred "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --compact-pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --where metadata.repo_slug=finance_fr \
  --where metadata.content_type~=Table \
  --fail-mode both_page_miss \
  --max-examples 10 \
  --top-retrieved 3 \
  --copy-gold-pages-dir "$LOCAL_OUTPUT_DIR/vidore-v3/failed_gold_pages/finance_fr_table" \
  --output-md "$LOCAL_OUTPUT_DIR/vidore-v3/failed_gold_pages_finance_fr_table.md"
```

## Exact Dense + SPLADE Doc-RRF Verification

Use this to test whether the portable dense+sparse method also improves the external datasets. The dense side should be the full/exact MaxSim baseline prediction, not `plain_top224`, so the comparison asks whether document-level RRF can improve over exact dense retrieval.

The reusable driver is:

```bash
scripts/run_external_doc_rrf_pipeline.sh
```

It runs:

1. `scripts/export_converted_page_text.py`
2. `scripts/build_splade_page_index.py`
3. `scripts/run_splade_page_retrieval.py`
4. `scripts/fuse_page_retrieval_predictions.py --fusion-mode doc_rrf`
5. `mmdocir/evaluate_mmdocir_retrieval.py --recall-k 1 2 4 5 10 20`

Default RRF settings match the best current non-heuristic M3DocVQA setup:

```text
dense_top_docs=20
sparse_top_docs=20
final_top_docs=20
rrf_k=10
dense_weight=0.75
sparse_weight=1.25
```

MMDocIR:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmdocir/env_hpc.sh

DATA_NAME=mmdocir \
DATA_ROOT="$LOCAL_DATA_DIR/mm-docir" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/mmdocir/doc_rrf_exact_dense_splade" \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

ViDoRe V3:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source vidore/env_hpc.sh

DATA_NAME=vidore-v3 \
DATA_ROOT="$LOCAL_DATA_DIR/vidore-v3" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/vidore-v3/doc_rrf_exact_dense_splade" \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

SciEGQA-Bench has rendered page images plus source PDFs under the prepared raw image tree. Pass `PDF_ROOT` so the page-text exporter can use `pdftotext`:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source sciegqa/env_hpc.sh

DATA_NAME=sciegqa \
DATA_ROOT="$LOCAL_DATA_DIR/sci-egqa-bench" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/sciegqa/baseline_ret1000.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/sciegqa/doc_rrf_exact_dense_splade" \
PDF_ROOT="$LOCAL_DATA_DIR/sci-egqa-bench/images_raw" \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

ViDoSeek also needs `PDF_ROOT`:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidoseek/env_hpc.sh

DATA_NAME=vidoseek \
DATA_ROOT="$LOCAL_DATA_DIR/vidoseek" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/vidoseek/baseline_ret1000.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/vidoseek/doc_rrf_exact_dense_splade" \
PDF_ROOT="$LOCAL_DATA_DIR/vidoseek/pdfs_raw" \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

Historical OpenDocVQA warning: the first converted `doc_pages_dev.jsonl` had image paths and source IDs but no OCR/markdown text. Any all-empty SPLADE output from that stage is invalid and should still be ignored. OpenDocVQA now has OCR-backed page text and valid SPLADE/Graph-PPR results; keep `--require-nonempty-text` enabled so this failure mode cannot silently recur.

Generate OCR page text first. This wrapper uses Tesseract over the prepared page images and shards the 206k-page workload:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh

command -v tesseract

sbatch \
  --export=ALL,NUM_SHARDS=64,OCR_LANG=eng \
  opendocvqa/sbatch_ocr_page_text_opendocvqa_array.sh
```

If the cluster has no Tesseract module, use the Python EasyOCR fallback. Request a GPU for this path:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh

"$REPO_ROOT/env/bin/python" -c "import easyocr; print(easyocr.__version__)"

export SSL_CERT_FILE="$("$REPO_ROOT/env/bin/python" -c "import certifi; print(certifi.where())")"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export EASYOCR_MODEL_DIR="$LOCAL_OUTPUT_DIR/opendocvqa/easyocr_models"
mkdir -p "$EASYOCR_MODEL_DIR"
"$REPO_ROOT/env/bin/python" -c "import easyocr, os; easyocr.Reader(['en'], gpu=False, model_storage_directory=os.environ['EASYOCR_MODEL_DIR'])"

export EASY_OCR_OUT="$LOCAL_OUTPUT_DIR/opendocvqa/easyocr_page_text_shards"
sbatch \
  --gres=gpu:1 \
  --export=ALL,NUM_SHARDS=64,OCR_ENGINE=easyocr,OCR_LANG=en,EASYOCR_GPU=1,EASYOCR_MODEL_DIR="$EASYOCR_MODEL_DIR",EASYOCR_DOWNLOAD=0,OUT_DIR="$EASY_OCR_OUT" \
  opendocvqa/sbatch_ocr_page_text_opendocvqa_array.sh
```

After all OCR shards complete, merge them into the page-text file consumed by SPLADE:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh

"$REPO_ROOT/env/bin/python" scripts/merge_jsonl_shards.py \
  --input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/ocr_page_text_shards/shard_*_of_64.jsonl" \
  --output-jsonl "$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_page_text_dev.jsonl" \
  --output-summary-json "$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_page_text_dev_merge_summary.json" \
  --dedupe-key page_uid
```

For the EasyOCR path, change the merge `--input-glob` to:

```bash
--input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/easyocr_page_text_shards/shard_*_of_64.jsonl"
```

Then run the same plain_top224 + SPLADE dense-heavy doc-RRF check:

```bash
DATA_NAME=opendocvqa \
DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade" \
PAGE_TEXT_JSONL="$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_page_text_dev.jsonl" \
SKIP_EXPORT=1 \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RRF_K=10 \
RRF_PRED="$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_plain_top224_splade_doc_rrf_denseheavy_k10.prediction.json" \
RRF_SUMMARY="$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_plain_top224_splade_doc_rrf_denseheavy_k10.summary.json" \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

If OCR or VLM text is added directly to the manifest instead, use the same driver with:

```bash
DATA_NAME=opendocvqa \
DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_exact_dense_splade" \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

Compare the RRF output against the exact dense baseline with the same evaluator:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$OUT_DIR/${DATA_NAME}_exact_dense_splade_doc_rrf.prediction.json" \
  --gold "$DATA_ROOT/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20
```

For this verification, treat `doc@4` and `doc@20` as the primary numbers. The fused prediction keeps one representative page row per fused document, so page-level numbers are diagnostic but not a full page-ranking replacement.

Observed doc-RRF results so far. The `doc hit@k` columns use the pipeline summary counts (`reranked_top*_doc_count / qids`), while page notes use evaluator recall. On multi-gold datasets such as ViDoRe V3, evaluator averaged `doc_recall_at_k` can be lower than doc-hit rate.

| Dataset | Method | qids | doc hit@4 | doc hit@20 | page@4 note |
|---|---:|---:|---:|---:|---|
| MMDocIR | SPLADE only | 1658 | 1196 / 1658 = 0.7214 | 1362 / 1658 = 0.8215 | sparse page ranking |
| MMDocIR | exact dense + SPLADE doc-RRF, sparse-heavy `0.75/1.25` | 1658 | 1257 / 1658 = 0.7581 | 1440 / 1658 = 0.8685 | page@4 = 0.4444; doc-fused representative page only |
| MMDocIR | exact dense + SPLADE doc-RRF, dense-heavy `1.25/0.75` | 1658 | 1304 / 1658 = 0.7865 | 1469 / 1658 = 0.8860 | page@4 = 0.4566; doc-fused representative page only |
| MMDocIR | plain_top224 + SPLADE doc-RRF, dense-heavy `1.25/0.75` | 1658 | 1322 / 1658 = 0.7973 | 1481 / 1658 = 0.8932 | page@4 = 0.4649; doc-fused representative page only |
| SciEGQA-Bench | SPLADE only | 1623 | 1421 / 1623 = 0.8755 | 1557 / 1623 = 0.9593 | sparse page ranking |
| SciEGQA-Bench | plain_top224 + SPLADE doc-RRF, dense-heavy `1.25/0.75` | 1623 | 1492 / 1623 = 0.9193 | 1598 / 1623 = 0.9846 | page@4 = 0.5601; doc-fused representative page only |
| ViDoSeek | SPLADE only | 1142 | 1126 / 1142 = 0.9860 | 1135 / 1142 = 0.9939 | sparse page ranking |
| ViDoSeek | plain_top224 + SPLADE doc-RRF, dense-heavy `1.25/0.75` | 1142 | 1141 / 1142 = 0.9991 | 1142 / 1142 = 1.0000 | page@4 = 0.6874; doc-fused representative page only |
| ViDoRe V3 | SPLADE only | 14514 | 9408 / 14514 = 0.6482 | 11658 / 14514 = 0.8032 | sparse page ranking |
| ViDoRe V3 | exact dense + SPLADE doc-RRF, sparse-heavy `0.75/1.25` | 14514 | 10868 / 14514 = 0.7488 | 13825 / 14514 = 0.9525 | page@4 = 0.1699; doc-fused representative page only |
| ViDoRe V3 | exact dense + SPLADE doc-RRF, dense-heavy `1.25/0.75` | 14514 | 12477 / 14514 = 0.8597 | 14162 / 14514 = 0.9757 | page@4 = 0.1963; doc-fused representative page only |
| ViDoRe V3 | plain_top224 + SPLADE doc-RRF, dense-heavy `1.25/0.75` | 14514 | 12547 / 14514 = 0.8645 | 14251 / 14514 = 0.9819 | page@4 = 0.2118; doc-fused representative page only |

Interpretation:

- doc-RRF improves substantially over SPLADE alone on both datasets.
- Dense-heavy RRF is much stronger than sparse-heavy RRF on both datasets, especially ViDoRe V3.
- It still does not beat the existing dense/compact retrieval runs at early rank on these external datasets.
- On SciEGQA-Bench, `plain_top224 + SPLADE` improves document recall over plain_top224 but destroys page recall:
  - plain_top224 doc recall@4: `0.9070`
  - plain_top224 + SPLADE doc recall@4: `0.9193`
  - plain_top224 page recall@4: `0.7394`
  - plain_top224 + SPLADE page recall@4: `0.5601`
- On ViDoSeek, document recall was already saturated, so doc-RRF gains only one top-4 document hit while losing a lot of page recall:
  - plain_top224 doc hit@4: `1140 / 1142 = 0.9982`
  - plain_top224 + SPLADE doc hit@4: `1141 / 1142 = 0.9991`
  - plain_top224 page recall@4: `0.8958`
  - plain_top224 + SPLADE page recall@4: `0.6874`
- On MMDocIR, `plain_top224 + SPLADE` improves doc-hit@20 over plain_top224 alone but remains below plain_top224 at doc-hit@4:
  - plain_top224 doc-hit@4: `1336 / 1658 = 0.8058`
  - plain_top224 + SPLADE doc-hit@4: `1322 / 1658 = 0.7973`
  - plain_top224 doc recall@20: `0.8890`
  - plain_top224 + SPLADE doc recall@20: `0.8932`
- On ViDoRe V3, `plain_top224 + SPLADE` improves over exact-dense RRF in doc-hit count, but it is still weaker than plain_top224 alone on averaged doc recall and page recall:
  - plain_top224 doc recall@4: `0.8854`
  - plain_top224 + SPLADE doc recall@4: `0.8508`
  - plain_top224 doc recall@20: `0.9809`
  - plain_top224 + SPLADE doc recall@20: `0.9791`

The current takeaway is that the M3DocVQA RRF method is not a direct out-of-the-box final page-ranker here. It can help document discovery, as SciEGQA shows, but the fused output needs a within-document page reranker. ViDoRe is mostly a hard page-within-correct-document problem, while MMDocIR metadata failures need document/page-structure signals that sparse text alone does not capture.

## Graph-PPR Verification

Use `scripts/run_external_graph_ppr_pipeline.sh` to test the graph-PPR method on the same dense/SPLADE artifacts. The default profile now matches the best M3DocVQA/MMQA transfer setting from `notes/visual_reranker_handoff_2026-04-27.md`:

- dense source: `plain_top224_ret1000_prediction.json`
- sparse source: SPLADE `*_splade_ret1000.prediction.json`
- graph budget: dense top 1000 + sparse top 1000
- profile: `GRAPH_PROFILE=doc_shortlist_best`
- final output: top 20 pages with `PER_DOC_PAGE_LIMIT=1`, so it behaves as a document shortlist
- PPR: `DOC_SEED_WEIGHT=0.0`, `RESTART_PROB=0.15`, `FINAL_PPR_PAGE_WEIGHT=1.5`, `FINAL_PPR_DOC_WEIGHT=0.75`
- source weights: equal by default

This is the config to use when asking whether Graph-PPR transfers as the best M3DocVQA document retriever. The earlier `page1000_nodocseed` runs are a page-ranking probe, not the exact handoff-best config. To reproduce that page-ranking probe, set:

```bash
GRAPH_PROFILE=page_rank_probe \
FINAL_TOP_PAGES=1000 \
PER_DOC_PAGE_LIMIT=0
```

Optional heading/breadcrumb graph anchors:

```bash
HEADING_BREADCRUMB_MODE=query_gated \
HEADING_BREADCRUMB_FIELD="markdown" \
HEADING_BREADCRUMB_EDGE_WEIGHT=0.15 \
HEADING_BREADCRUMB_RESTART_WEIGHT=0.10 \
HEADING_BREADCRUMB_MAX_PAGE_MATCHES=50 \
HEADING_BREADCRUMB_MAX_DOC_MATCHES=20 \
HEADING_BREADCRUMB_WEIGHT_MODE=local_idf
```

This adds Markdown heading nodes such as `Financial Statements > Notes > Revenue Recognition`
and connects candidate pages sharing the same normalized breadcrumb. Start with `query_gated` when
the query names a section, table, note, topic, or manual subsection. Use `query_gated_shared` as a
stronger ablation when same-document page confusion is high, because it also lets non-query-matched
heading nodes propagate page mass within candidate sections.

Optional entity/alias graph anchors:

```bash
ENTITY_ALIAS_MODE=query_gated \
ENTITY_ALIAS_FIELD="markdown" \
ENTITY_ALIAS_EDGE_WEIGHT=0.10 \
ENTITY_ALIAS_RESTART_WEIGHT=0.05 \
ENTITY_ALIAS_MAX_PAGE_MATCHES=80 \
ENTITY_ALIAS_MAX_DOC_MATCHES=20 \
ENTITY_ALIAS_WEIGHT_MODE=local_idf
```

This extracts corpus entity nodes from the configured page field and normalizes lightweight aliases,
including parenthetical aliases such as `International Business Machines (IBM)` and corpus-level
acronym links when both the long form and acronym appear. The default field is Markdown only; it does
not fall back to OCR, VLM text, or plain text unless those fields are explicitly passed. Start with
`query_gated` for query-named companies, tickers, standards, laws, methods, datasets, chemicals, and
fiscal-year anchors. Use `query_gated_shared` as the stronger graph-propagation ablation after the
gated run is safe.

Run the best-config doc-shortlist check on the smaller datasets first.

Retriever-induced page-page graph edges:

```bash
DATA_NAME=<dataset> \
DATA_ROOT="$LOCAL_DATA_DIR/<dataset-root>" \
DENSE_PRED=/path/to/plain_top224_ret1000_prediction.json \
SPARSE_PRED=/path/to/splade_ret1000.prediction.json \
SPLADE_INDEX_PT=/path/to/splade_page_index.pt \
BM25_PAGE_TEXT_JSONL="$DATA_ROOT/doc_pages_dev.jsonl" \
BM25_KNN_TEXT_FIELD=markdown \
OUT_DIR=/path/to/graph_ppr_output \
BM25_KNN_ENABLE=1 \
BM25_KNN_MUTUAL_ONLY=1 \
bash scripts/run_retriever_induced_graph_track.sh
```

This runs base Graph-PPR, SPLADE page-page kNN Graph-PPR, BM25 page-page mutual-kNN Graph-PPR, and
an unweighted RRF over the graph views. BM25 kNN can also be built standalone with
`scripts/build_bm25_page_knn_graph.py`; it emits the same `--external-page-graph-jsonl` schema as
the SPLADE kNN builder. For the original Markdown-only setup, use `BM25_KNN_TEXT_FIELD=markdown`;
datasets without a real `markdown` field need that field generated upstream before this BM25 view is
meaningful.

SciEGQA-Bench:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source sciegqa/env_hpc.sh

DATA_NAME=sciegqa \
DATA_ROOT="$LOCAL_DATA_DIR/sci-egqa-bench" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/sciegqa/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/sciegqa/doc_rrf_plain_top224_splade/sciegqa_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/sciegqa/graph_ppr_plain_top224_splade" \
GRAPH_LABEL="sciegqa_plain_top224_splade_graph1000_top20_best" \
bash scripts/run_external_graph_ppr_pipeline.sh
```

ViDoSeek:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidoseek/env_hpc.sh

DATA_NAME=vidoseek \
DATA_ROOT="$LOCAL_DATA_DIR/vidoseek" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/vidoseek/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/vidoseek/doc_rrf_plain_top224_splade/vidoseek_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/vidoseek/graph_ppr_plain_top224_splade" \
GRAPH_LABEL="vidoseek_plain_top224_splade_graph1000_top20_best" \
bash scripts/run_external_graph_ppr_pipeline.sh
```

MMDocIR:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmdocir/env_hpc.sh

DATA_NAME=mmdocir \
DATA_ROOT="$LOCAL_DATA_DIR/mm-docir" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/mmdocir/doc_rrf_exact_dense_splade/mmdocir_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/mmdocir/graph_ppr_plain_top224_splade" \
GRAPH_LABEL="mmdocir_plain_top224_splade_graph1000_top20_best" \
bash scripts/run_external_graph_ppr_pipeline.sh
```

ViDoRe V3:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source vidore/env_hpc.sh

DATA_NAME=vidore-v3 \
DATA_ROOT="$LOCAL_DATA_DIR/vidore-v3" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/vidore-v3/doc_rrf_exact_dense_splade/vidore-v3_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/vidore-v3/graph_ppr_plain_top224_splade" \
GRAPH_LABEL="vidore-v3_plain_top224_splade_graph1000_top20_best" \
bash scripts/run_external_graph_ppr_pipeline.sh
```

OpenDocVQA OCR-backed SPLADE and Graph-PPR are now complete. The command shape below remains the reproducibility path for the frozen `denseheavy125_medium_both` page-preserving graph run:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh

DATA_NAME=opendocvqa \
DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/opendocvqa/graph_ppr_plain_top224_splade" \
GRAPH_PROFILE=page_rank_probe \
GRAPH_LABEL="opendocvqa_denseheavy125_medium_both" \
FINAL_TOP_PAGES=1000 \
PER_DOC_PAGE_LIMIT=0 \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RESTART_PROB=0.15 \
PPR_ITERS=30 \
PAGE_DOC_EDGE_WEIGHT=1.0 \
SAME_DOC_WINDOW=1 \
ADJACENT_PAGE_EDGE_WEIGHT=0.25 \
FINAL_PAGE_SEED_WEIGHT=1.0 \
FINAL_PPR_PAGE_WEIGHT=0.5 \
FINAL_PPR_DOC_WEIGHT=0.25 \
bash scripts/run_external_graph_ppr_pipeline.sh
```

Observed page-ranking-probe results from `GRAPH_PROFILE=page_rank_probe`:

| Dataset | qids | page@4 | page@20 | doc@4 | doc@20 | note |
|---|---:|---:|---:|---:|---:|---|
| SciEGQA-Bench | 1623 | 0.7686 | 0.9091 | 0.9298 | 0.9852 | strong page-ranking result; not the doc-shortlist-best profile |
| ViDoSeek | 1142 | 0.8905 | 0.9982 | 0.9991 | 1.0000 | slight page@4 drop vs plain_top224, better deeper page recall |
| OpenDocVQA | 41017 | 0.5863 | 0.7662 | 0.6035 | 0.7922 | OCR-backed `denseheavy125_medium_both`; broad win over `plain_top224` |

Do not mix these with the doc-shortlist-best results; they use different output caps and slightly different PPR weights.

Observed `doc_shortlist_best` transfer results after verifying the saved summaries used the exact intended config:

```text
dense_top_pages = 1000
sparse_top_pages = 1000
final_top_pages = 20
per_doc_page_limit = 1
rrf_k = 10
dense_weight = 1.0
sparse_weight = 1.0
doc_seed_weight = 0.0
restart_prob = 0.15
ppr_iters = 30
page_doc_edge_weight = 1.0
same_doc_window = 1
adjacent_page_edge_weight = 0.25
final_page_seed_weight = 1.0
final_ppr_page_weight = 1.5
final_ppr_doc_weight = 0.75
```

| Dataset | qids | doc@4 | doc@20 | page@4 | page@20 | output |
|---|---:|---:|---:|---:|---:|---|
| MMDocIR | 1658 | 0.8034 | 0.8884 | 0.4562 | 0.4998 | `mmdocir_plain_top224_splade_graph1000_top20_best` |
| SciEGQA-Bench | 1623 | 0.9279 | 0.9846 | 0.5173 | 0.5474 | `sciegqa_plain_top224_splade_graph1000_top20_best` |
| ViDoSeek | 1142 | 0.9991 | 1.0000 | 0.6743 | 0.6751 | `vidoseek_plain_top224_splade_graph1000_top20_best` |
| ViDoRe V3 | 14514 | 0.8725 | 0.9703 | 0.2064 | 0.2285 | `vidore-v3_plain_top224_splade_graph1000_top20_best` |

Conclusion from the external transfer check:

- The M3DocVQA best Graph-PPR document-shortlist config was reproduced correctly on the external datasets, so the negative transfer is not a config mismatch.
- `doc_shortlist_best` is not a strong final page retriever on these page-labeled datasets because it emits one representative page per document.
- On MMDocIR and ViDoRe V3 it does not beat `plain_top224` at early document recall and is much worse at page recall.
- On SciEGQA it improves document recall over `plain_top224`, but the page-ranking probe is substantially better for page recall.
- On ViDoSeek document recall is already saturated, so the small doc gain is not worth the large page-recall loss.
- For external datasets with reliable page labels, continue with a page-preserving Graph-PPR profile (`GRAPH_PROFILE=page_rank_probe` or a tuned page-preserving variant) rather than the one-page-per-doc M3DocVQA shortlist profile.
- Keep `doc_shortlist_best` only for experiments where the downstream stage consumes a document shortlist or one representative page per document.

Follow-up page-preserving runs found a stronger general config for exact page-labeled datasets:

```text
GRAPH_PROFILE=page_rank_probe
FINAL_TOP_PAGES=1000
PER_DOC_PAGE_LIMIT=0
DENSE_WEIGHT=1.25
SPARSE_WEIGHT=0.75
RESTART_PROB=0.15
PPR_ITERS=30
PAGE_DOC_EDGE_WEIGHT=1.0
SAME_DOC_WINDOW=1
ADJACENT_PAGE_EDGE_WEIGHT=0.25
FINAL_PAGE_SEED_WEIGHT=1.0
FINAL_PPR_PAGE_WEIGHT=0.5
FINAL_PPR_DOC_WEIGHT=0.25
```

Short label: `denseheavy125_medium_both`.

| Dataset | qids | best sweep row by page@4 | page@1 | page@4 | conservative boundary gate page hit@4 | page@20 | doc@4 | doc@20 | note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| SciEGQA-Bench | 1623 | `denseheavy125_medium_both` | 0.5508 *(plain 0.5228)* | 0.8152 *(plain 0.7394)* | 0.8152 *(1323 / 1623; native, no net change)* | 0.9248 *(plain 0.8758)* | 0.9291 *(plain 0.9070)* | 0.9871 *(plain 0.9772)* | broad page/doc win |
| MMDocIR | 1658 | `denseheavy125_medium_both` | 0.4596 *(plain 0.4136)* | 0.6719 *(plain 0.6075)* | 0.6737 *(1117 / 1658; +3, 0 lost)* | 0.7889 *(plain 0.7480)* | 0.8160 *(plain 0.8058)* | 0.8938 *(plain 0.8890)* | clear page@1/@4/@20 win |
| ViDoRe V3 | 14514 | `denseheavy125_medium_both` | 0.3902 *(plain 0.1730)* | 0.6465 *(plain 0.3312)* | 0.6465 *(9383 / 14514; heading no-op)* | 0.8227 *(plain 0.5431)* | 0.9099 *(plain 0.8854)* | 0.9788 *(plain 0.9809)* | large page gain; tiny doc@20 loss |
| OpenDocVQA | 41017 | `denseheavy125_medium_both` | 0.3988 *(plain 0.3516)* | 0.5863 *(plain 0.5122)* | N/A *(gate not run)* | 0.7662 *(plain 0.6599)* | 0.6035 *(plain 0.5307)* | 0.7922 *(plain 0.6955)* | OCR-backed SPLADE graph result; broad early-rank win |
| ViDoSeek | 1142 | `denseheavy150_m3best_pagepreserve` | 0.6909 *(plain 0.6830)* | 0.9037 *(plain 0.8958)* | 0.9011* *(1029 / 1142; +6, 0 lost)* | 0.9982 *(plain 0.9842)* | 0.9991 *(plain 0.9982)* | 1.0000 *(plain 1.0000)* | saturated; heavier row best for this dataset |

Use `denseheavy125_medium_both` as the current frozen single page-labeled config. ViDoSeek's best individual row is `denseheavy150_m3best_pagepreserve`, but the `1.25/0.75 + medium_both` setting is the best common setting across SciEGQA, MMDocIR, ViDoRe V3, and OpenDocVQA and remains close on ViDoSeek. Do not claim universal superiority at every metric; report page@1 separately and keep `plain_top224` as the required baseline.

The conservative-boundary column reports measured final top-4 page-hit output after native
heading/bodyguard rescue. It is an end-to-end result only where the gate base is the listed
graph-control branch. `*` For ViDoSeek, the gate starts from its no-heading control
(`1023 -> 1029` hits); it was not applied directly on top of the separately selected
`denseheavy150_m3best_pagepreserve` best row (`page@4 = 0.9037`).

## Graph Structural Attribution Ablation

The page-preserving backbone ranks pages, but its graph signal mixes parent-document support and
adjacent-page continuity. Run this ablation before claiming that its gains are page-local. It
fixes the source fusion to the general backbone (`DENSE_WEIGHT=1.25`, `SPARSE_WEIGHT=0.75`) and
changes only graph structure/final graph components:

| variant | page-doc transitions | adjacent-page transitions | explicit final doc component | question answered |
|---|---:|---:|---:|---|
| `seed_only` | off | off | off | dense/SPLADE page-seed reference |
| `adjacent_only` | off | on | off | can page-local continuity help without document support? |
| `doc_edges_page_score_only` | on | off | off | does document diffusion already alter page ranking? |
| `doc_prior_only` | on | off | on | how much comes from document support alone? |
| `full_no_explicit_doc_score` | on | on | off | do both edge types suffice without direct document boost? |
| `current_full_graph` | on | on | on | frozen `denseheavy125_medium_both` structure |

The default selected-dataset runner covers the five exact-page datasets with available dense and
SPLADE artifacts: MMDocIR, SciEGQA, ViDoSeek, ViDoRe V3, and DUDE. OpenDocVQA is supported as
an opt-in extension, but is not included by default because its sparse graph input is OCR-backed.

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase origin codex/mmdocir-hpc-workflow
source hpc_vital_paths.generated.env

DATASETS="mmdocir sciegqa vidoseek vidore dude" \
bash examples/run_graph_structure_ablation_selected_datasets.sh
```

The wrapper writes each dataset's six predictions/summaries under
`output/<dataset>/graph_structure_ablation/` and writes a combined table to
`graph_structure_ablation_results.md`. Paste that compact report back for interpretation. Read the
outcome as follows:

- if `doc_prior_only` captures most of the `current_full_graph` gain, the backbone is primarily
  document-support reranking with page output;
- if `adjacent_only` increases page metrics without comparable document gains, local page
  continuity has independent evidence;
- if `full_no_explicit_doc_score` matches `current_full_graph`, the explicit parent-document final
  term can be weakened or removed for a cleaner page-level claim.

## Safe Heading/Bodyguard Rescue Gate

This is the precision-oriented rescue layer on top of the heading-augmented graph views. It is not a global reranker. It only accepts narrow rank-window promotions when the promoted page is supported by multiple heading views, beats the displaced boundary page by heading score, and passes a body-evidence guard. It also abstains on layout-sensitive queries such as row/column/right/left questions. The default runner now uses this common gate across datasets; the previous ViDoSeek page-0 block and DUDE document-rank-1 constraint remain available only as explicit audit/reproduction switches.

The `boundary` profile is cutoff-relative. For `HIT_K=k`, the candidate may introduce at most
one page into top-`k`, and that page must have been base rank `k+1`. The default agreement guards
also scale with the cutoff: at least `k-1` base pages remain in candidate top-`k`, support votes
are checked within candidate top-`k`, and the support threshold defaults to all configured
support views (`heuristic` and `strict` for the current PDF-heading gate). The promoted document
must be among the first `k` distinct documents encountered in the base ranking. That
document-rank cap does not require a
page from the same document inside the base top-`k`; the policy ablation below tests that stricter
alternative explicitly.

| `HIT_K` | candidate slot limit | allowed base rescue rank | minimum top-k overlap | default output suffix |
|---:|---:|---:|---:|---|
| 4 | 4 | 5 | 3 | `safe_gate_bodyguard` |
| 6 | 6 | 7 | 5 | `safe_gate_bodyguard_top6` |
| 8 | 8 | 9 | 7 | `safe_gate_bodyguard_top8` |
| 10 | 10 | 11 | 9 | `safe_gate_bodyguard_top10` |

Non-default cutoffs receive a `_top{k}` output suffix so a top-8 experiment does not overwrite the
validated top-4 artifacts.

Recorded safe-gate runs (selected results plus labeled audit/ablation rows):

| Dataset | accepted | base page hit@4 | candidate page hit@4 | gated page hit@4 | recovered | lost | net | body rejects | summary artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MMDocIR | 38 | 1114 | 1113 | 1117 | 3 | 0 | +3 | 25 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/mmdocir/heading_breadcrumb_pdf_markdown_source_ablation/mmdocir_heuristic_strict_safe_gate_bodyguard.summary.json` |
| MMDocIR (`native codeguard` ablation; rejected) | 26 | 1114 | 1117 | 1114 | 1 | 1 | 0 | 25 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/mmdocir/heading_breadcrumb_pdf_markdown_source_ablation/mmdocir_safe_gate_bodyguard_codeguard.summary.json` |
| SciEGQA-Bench | 28 | 1323 | 1328 | 1323 | 0 | 0 | 0 | 23 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/output/sciegqa/heading_breadcrumb_pdf_markdown_source_ablation/sciegqa_safe_gate_bodyguard.summary.json` |
| SciEGQA-Bench (`pymupdf4llm==0.3.4`) | 2 | 1323 | 1323 | 1324 | 1 | 0 | +1 | 5 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/output/sciegqa/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation/sciegqa_safe_gate_bodyguard.summary.json` |
| ViDoSeek (`page-0-block` audit) | 45 | 1023 | 1033 | 1029 | 6 | 0 | +6 | 30 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/heading_breadcrumb_pdf_markdown_source_ablation/vidoseek_strict_support_gate_layoutblock_no_page0_bodyguard.summary.json` |
| ViDoSeek (`pymupdf4llm==0.3.4`, `page-0-block` audit) | 51 | 1023 | 1019 | 1029 | 6 | 0 | +6 | 16 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation/vidoseek_safe_gate_bodyguard_no_page0.summary.json` |
| DUDE (`doc-rank-1` audit) | 6 | 1733 | 1730 | 1733 | 0 | 0 | 0 | 1 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/heading_breadcrumb_pdf_markdown_source_ablation/dude_safe_gate_bodyguard_docrank1.summary.json` |
| ViDoRe V3 (`text-heading` no-op) | 0 | 9383 | 9383 | 9383 | 0 | 0 | 0 | 0 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG/output/vidore-v3/heading_breadcrumb_text_source_ablation/vidore_safe_gate_bodyguard.summary.json` |

Interpretation:

- MMDocIR is the strongest positive-control result: the raw heading candidate is slightly worse than base at page hit@4, but the gate extracts 3 additional hits with zero losses.
- The MMDocIR native codeguard ablation is rejected: it suppressed code-like heading evidence but
  changed the gate result to `1` recovered and `1` lost, eliminating the selected method's
  zero-loss `+3` gain.
- SciEGQA-Bench is a useful negative control: the raw heading candidate improves page hit@4, but the safe gate abstains enough to preserve the baseline with zero losses.
- SciEGQA-Bench PyMuPDF4LLM is a positive extraction ablation: the direct alternative heading
  candidate ties the no-heading control at page hit@4, while the safe gate accepts only `2`
  promotions and obtains `+1` with zero loss.
- The recorded ViDoSeek row shows that an additional page-0 abstention removed observed losses while preserving a positive net gain; it is now an audit variant rather than part of default EvidenceGuard-PPR.
- ViDoSeek PyMuPDF4LLM is a completed exact-page backend ablation. It recovers the same `+6`
  zero-loss final result as native Markdown, but its direct candidate is weaker (`1019` rather
  than `1033` page hits at `@4`) and it accepts more promotions (`51` rather than `45`).
  Native Markdown remains the primary ViDoSeek source.
- DUDE is a negative/neutral transfer case: the common gate previously lost one page hit because a cross-document annual-report heading looked better than a near-empty gold cover page. The recorded doc-rank-1 audit safely abstains (`0` net, `0` lost), but the restriction is no longer part of the default method.
- ViDoRe V3 is a heading-unavailable transfer case: `doc_pages_dev` text produced `0` outline/heuristic/strict heading lines, so full/heuristic/strict graph views were identical to the no-heading control and the safe gate had no heading evidence to accept promotions.

Runner for additional datasets:

```bash
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

This helper can target M3DocVQA, DUDE, and ViDoRe after their `plain_top224` and SPLADE artifacts
exist. All three have now been run: DUDE and ViDoRe are recorded above, while M3DocVQA can be
reported only as a document-level sanity check and optional synthetic page-0 diagnostic because
its gold lacks page labels. It writes per-dataset `*_safe_gate_bodyguard.summary.json`,
`.prediction.json`, and `.cases.json` files under the dataset heading-ablation output directory.
To reproduce the removed dataset-specific audits, set
`VIDOSEEK_REJECT_PROMOTED_PAGE_IDX=0` or `DUDE_PROMOTED_DOC_MAX_BASE_RANK=1`;
those runs write `*_no_page0.*` and `*_docrank1.*` artifacts rather than the common default output.

Rank-window rescue profile:

```bash
SAFE_GATE_PROFILE=window20 \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

`SAFE_GATE_PROFILE=window20` keeps the same safety stack but expands the candidate scan from
the rank-5 boundary to the base rank `5-20` window:

- `candidate_rank_max=20`
- `rescue_rank_min=5`
- `rescue_rank_max=20`
- `support_page_rank_max=20`
- same displaced-boundary heading/body comparison
- same multi-view support, layout-query block, optional promoted page-index block, and doc-rank gate

The default `HIT_K=4 SAFE_GATE_PROFILE=boundary` configuration remains unchanged, so the
validated frozen artifacts above stay reproducible. Higher cutoffs use the adaptive boundary rule
described above; `window20` remains the exploratory absolute rank-window profile.
Set `RUN_GOLD_RANK_AUDIT=1` to also write `*.gold_rank_positions.json` and
`*.gold_rank_positions.md` next to each output. The audit now includes first gold document ranks,
page/doc rank bands, and a page-rank-band by doc-rank-band matrix so rank `6-20` opportunities can
be separated from document-retrieval failures.

Gate-policy ablation after a completed adaptive boundary run:

```bash
HIT_K=8 \
PDF_MARKDOWN_BACKEND=native \
DATASETS="mmdocir sciegqa vidoseek dude" \
bash examples/run_safe_gate_policy_ablation_selected_datasets.sh \
  2>&1 | tee safe_gate_policy_native_boundary_top8_run.log

sed -n '1,240p' \
  output/safe_gate_policy_ablation/native_boundary_top8_policy_ablation.md
```

This is a gate-only ablation: it reads the completed no-heading/full-heading/heuristic/strict
prediction artifacts and does not regenerate Markdown or rerun Graph-PPR. It writes separate
`*_safe_gate_policy_native_boundary_top8_<variant>.*` outputs. The six variants are `control`,
`no_doc_rank_cap`, `require_topk_doc`, `relax_overlap`, `relax_support`, and
`combined_relaxed`. `control` retains the existing distinct-document rank cap, while
`require_topk_doc` is the true same-window document-membership diagnostic. Use the
control-versus-variant page and document hit deltas together with the overlap/document/support/body
rejection counts to decide which constraint is useful.

Top-k case audit after the boundary and policy runs:

```bash
HIT_K=8 \
DATASETS="mmdocir sciegqa vidoseek dude" \
bash examples/audit_safe_gate_topk_boundary_cases.sh

sed -n '1,260p' \
  output/safe_gate_top8_case_audit/safe_gate_top8_case_diagnostics.md
```

This audit reads completed `*.summary.json` and `*.cases.json` files. It aggregates net movement
by variant, lists lost/recovered qids with promoted-page ranks and support votes, and writes
side-by-side Markdown for the selected variants. By default it renders the actual boundary gate
and `policy_require_topk_doc`; set `SIDE_BY_SIDE_VARIANTS` to inspect more variants.

Completed native code-noise ablation (rejected):

```bash
NATIVE_CODEGUARD_ABLATION=1 \
SAFE_GATE_PROFILE=boundary \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="mmdocir" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

This ablation writes `doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl` and uses its graph
view as the strict support vote and heading-relevance safety view. It detected `101` code-dense
pages and suppressed `69` heuristic heading lines, leaving `22,295` codeguard heuristic heading
lines. The final MMDocIR boundary result was `26` accepted promotions, page hit@4 `1114`, `1`
recovered, `1` lost, and net `0`. It therefore fails the admission requirement of retaining the
native gate's zero-loss improvement (`1117`, `+3`, `0` lost). Keep native strict headings in
EvidenceGuard-PPR; retain codeguard only as a recorded negative ablation.

If paths drift across historical output roots, generate and source a canonical path manifest first:

```bash
python scripts/discover_hpc_vital_paths.py \
  --output-json hpc_vital_paths.generated.json \
  --output-env hpc_vital_paths.generated.env

source hpc_vital_paths.generated.env

SAFE_GATE_PROFILE=window20 \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

The generated env exports include variables such as `M3DOCVQA_DENSE_PRED`,
`M3DOCVQA_SPARSE_PRED`, `DUDE_DENSE_PRED`, `DUDE_SPARSE_PRED`, `VIDORE_DENSE_PRED`, and
`VIDORE_SPARSE_PRED`. The selected-dataset runner automatically sources
`hpc_vital_paths.generated.env` from the repo root when present; use `HPC_PATH_ENV=/path/to/file`
to point it somewhere else.

Current M3DocVQA path sanity:

```text
M3DOCVQA_DENSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json
M3DOCVQA_SPARSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json
M3DOCVQA_PAGE_TEXT_JSONL=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl
```

Use the raw `mmqa_dev_splade.prediction.json` for `M3DOCVQA_SPARSE_PRED`; do not substitute historical graph/source-ablation/no-SPLADE outputs.

M3DocVQA annotation limitation:

`MMQA_dev.jsonl` supplies supporting document IDs but no true supporting page indices. For this
dataset, safe-gate `page@4`, `recovered`, and `lost` are unavailable rather than zero; use the
reported document recall for retrieval sanity and run downstream VQA evaluation to assess whether
the promoted pages improve answer quality. Page-level rescue safety remains measurable on
MMDocIR, SciEGQA, ViDoSeek, DUDE, and ViDoRe.

ImageListQ page-0 proxy diagnostic:

For hypothesis exploration only, `scripts/evaluate_first_page_gold_retrieval.py` can assume page
index `0` of every supporting document is relevant on the `ImageListQ` subset. The resulting
`synthetic_page_*` fields are proxy metrics, not annotated page recall, and must not be placed in
the primary page-labeled results table.

```bash
export M3DOCVQA_HEADING_OUT="$PWD/output/m3docvqa_heading_breadcrumb_pdf_markdown_source_ablation"

"$PWD/env/bin/python" scripts/evaluate_first_page_gold_retrieval.py \
  --baseline-pred "$M3DOCVQA_HEADING_OUT/m3docvqa_heading_control_no_heading.prediction.json" \
  --pred "$M3DOCVQA_HEADING_OUT/m3docvqa_safe_window20_gate_bodyguard.prediction.json" \
  --gold "$PWD/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl" \
  --question-type ImageListQ \
  --first-page-idx 0 \
  --hit-k 4 \
  --recall-k 1 2 4 5 10 20 \
  --output-json "$M3DOCVQA_HEADING_OUT/m3docvqa_safe_window20_gate_bodyguard.imagelistq_page0_proxy.json"
```

Completed aligned M3DocVQA document-level comparison:

| Markdown source | heading pages | safe accepted | no-heading doc hit@4 | full-heading doc hit@4 | strict-heading doc hit@4 | safe-gated doc hit@4 | safe doc net |
|---|---:|---:|---:|---:|---:|---:|---:|
| native PDF headings, current rerun | 30,343 | 972 | 2,346 | 2,342 | 2,342 | 2,346 | 0 |
| `pymupdf4llm==0.3.4` | 25,355 | 766 | 2,346 | 2,348 | 2,349 | 2,346 | 0 |

The aligned no-heading controls now match, resolving the earlier run-drift concern. On this
document-only dataset, direct PyMuPDF4LLM heading graph output is slightly better than the native
heading output; the safe gate retains the base document retrieval in both runs. This is not page
rescue evidence because M3DocVQA lacks annotated pages.

The previous native `ImageListQ` page-0 proxy was generated from the older, non-aligned native
prediction tree and must be rerun before comparing it with PyMuPDF4LLM. The existing PyMuPDF4LLM
proxy is negative within its own run (`43` to `42` synthetic page hits at `@4`, `1` recovered,
`2` lost). Prefer the exact-page ViDoSeek experiment below for the backend decision.

Page-labeled PyMuPDF4LLM comparison:

The ViDoSeek exact-page test has completed successfully. MuPDF emitted malformed-content warnings
during conversion, but the export summary reports `backend_error_doc_count=0` and
`unmatched_doc_count=0`.

| ViDoSeek Markdown source | heading pages | raw heading lines | strict heading lines | direct full page hit@4 | direct strict page hit@4 | gated page hit@4 | recovered | lost | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native PyMuPDF | 4,342 | 12,362 | 9,734 | 1,033 | 1,034 | 1,029 | 6 | 0 | +6 |
| `pymupdf4llm==0.3.4` | 3,346 | 4,958 | 3,865 | 1,019 | 1,020 | 1,029 | 6 | 0 | +6 |

PyMuPDF4LLM produces substantially fewer heading-bearing pages and weaker direct heading
rankings, then reaches the same final result only after the conservative gate. It is therefore
not an improvement over native Markdown on ViDoSeek.

The SciEGQA-Bench exact-page backend test has also completed:

| SciEGQA Markdown source | heading pages | raw heading lines | strict heading lines | direct full page hit@4 | direct strict page hit@4 | gated page hit@4 | accepted | recovered | lost | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native PyMuPDF | not recorded here | not recorded here | not recorded here | 1,328 | not recorded here | 1,323 | 28 | 0 | 0 | 0 |
| `pymupdf4llm==0.3.4` | 337 | 505 | 389 | 1,323 | 1,323 | 1,324 | 2 | 1 | 0 | +1 |

PyMuPDF4LLM creates a much smaller promotion set on SciEGQA and the bodyguard identifies one
useful page rescue that native headings did not admit. This provides exact-page evidence that
extractor diversity can matter to the gated method, although the gain is small.

DUDE is already supported by the same runner with `DATASETS="dude"`. MMDocIR remains the
strongest positive native result. Although its prepared Hugging Face artifact provides rendered
pages rather than a declared source-PDF root, the completed native PDF-Markdown export records the
PDF root in `pdf_markdown_summary.json` and matched PDF paths in its JSONL output. The
selected-dataset runner now recovers that provenance automatically for `DATASETS="mmdocir"`; set
`MMDOCIR_PDF_ROOT` explicitly if the recorded directory has moved.

MMDocIR PyMuPDF4LLM exact-page comparison:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
source hpc_vital_paths.generated.env

PDF_MARKDOWN_FORCE_REBUILD=1 \
PDF_MARKDOWN_BACKEND=pymupdf4llm \
SAFE_GATE_PROFILE=boundary \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="mmdocir" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

If the runner reports `missing_mmdocir_pdf_root_directory`, recover or locate the PDF directory
and rerun with:

```bash
MMDOCIR_PDF_ROOT=/path/to/mmdocir/source/pdfs \
PDF_MARKDOWN_FORCE_REBUILD=1 \
PDF_MARKDOWN_BACKEND=pymupdf4llm \
SAFE_GATE_PROFILE=boundary \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="mmdocir" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

Compare completed native and PyMuPDF4LLM outputs across page-labeled PDF datasets:

```bash
DATASETS="vidoseek sciegqa mmdocir" \
bash examples/report_pdf_markdown_backend_comparison.sh
```

This writes one Markdown and one JSON report per available backend pair under
`output/pdf_markdown_backend_comparison/`. The report separates extraction coverage and
heading-disagreement audit samples from downstream page hit@4/gate utility. It skips MMDocIR
until the PyMuPDF4LLM run above has completed, and skips DUDE unless a PyMuPDF4LLM DUDE run is
generated explicitly.

Alternative PDF-to-Markdown quality check:

The native PDF Markdown exporter uses PDF bookmarks and font-size heuristics. To test whether the
heading signal improves with another structured Markdown converter, the same pipeline can use
legacy `PyMuPDF4LLM` `0.3.4`. This backend disables OCR where its API exposes the control, uses
native PDF content, and writes to a distinct output directory without overwriting the validated
native artifacts.

Do not install an unpinned current `pymupdf4llm` for this comparison. Releases introduced in
March 2026 automatically activate the ONNX-based Layout component on import, which emits CPU
affinity errors under the current SLURM binding. A neural-layout run is a separate experiment and
needs its own runtime configuration. Installing even the pinned package may update PyMuPDF in the
experiment environment, so preserve existing native JSONL artifacts and record package versions
before claiming exact reproduction.

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
source hpc_vital_paths.generated.env
source scripts/m3docvqa_internal_env.sh

"$PWD/env/bin/python" -m pip install --force-reinstall "pymupdf4llm==0.3.4"

export ALT_DIR="$LOCAL_OUTPUT_DIR/m3docvqa_heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation"
mkdir -p "$ALT_DIR"

"$PWD/env/bin/python" scripts/export_pdf_page_markdown.py \
  --doc-pages-jsonl "$M3DOCVQA_PAGE_TEXT_JSONL" \
  --pdf-root "$DATASET_ROOT" \
  --backend pymupdf4llm \
  --output-jsonl "$ALT_DIR/doc_pages_dev_with_pdf_markdown.jsonl" \
  --output-summary-json "$ALT_DIR/pdf_markdown_summary.json" \
  --progress-every 1000 \
  --body-char-limit 6000 \
  --require-heading-pages

"$PWD/env/bin/python" scripts/prepare_pdf_markdown_variants.py \
  --input-jsonl "$ALT_DIR/doc_pages_dev_with_pdf_markdown.jsonl" \
  --output-dir "$ALT_DIR/pdf_markdown_variants"
```

Completed alternative artifact directory:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation
```

Compare the extracted heading coverage before interpreting retrieval:

```bash
python - <<'PY'
import json, os, re
from pathlib import Path

alt = Path(os.environ["ALT_DIR"])
for path in [
    alt / "pdf_markdown_summary.json",
    alt / "pdf_markdown_variants" / "pdf_markdown_variants.summary.json",
    alt / "m3docvqa_safe_window20_gate_bodyguard.summary.json",
]:
    if not path.exists():
        print("missing", path)
        continue
    with path.open() as f:
        s = json.load(f)
    print("\n==", path.name, "==")
    for key in [
        "backend", "page_count", "heading_page_count", "source_counts",
        "raw_heuristic_heading_line_count", "strict_heuristic_heading_line_count",
        "accepted_count", "base_page_hit_at_k_count", "candidate_page_hit_at_k_count",
        "page_hit_at_k_count", "recovered", "lost", "net_recovered",
    ]:
        if key in s:
            print(key, s[key])

shown = 0
print("\n== heading samples ==")
with (alt / "doc_pages_dev_with_pdf_markdown.jsonl").open() as f:
    for line in f:
        row = json.loads(line)
        markdown = str(row.get("markdown") or "")
        if not re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", markdown):
            continue
        print("\n", row.get("doc_id"), row.get("page_idx"))
        print(markdown[:800])
        shown += 1
        if shown >= 5:
            break
PY
```

If heading coverage and sampled headings are reasonable, run the graph/gate evaluation. The runner
will reuse the extracted alternative JSONL above rather than regenerate it.

```bash
PDF_MARKDOWN_BACKEND=pymupdf4llm \
SAFE_GATE_PROFILE=window20 \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

After an interrupted extraction, rerun with `PDF_MARKDOWN_FORCE_REBUILD=1`. The exporter now
publishes JSONL and summary files only after conversion completes, so future interrupted runs
leave only `.tmp` files and cannot be reused as completed extraction data.

```bash
PDF_MARKDOWN_FORCE_REBUILD=1 \
PDF_MARKDOWN_BACKEND=pymupdf4llm \
SAFE_GATE_PROFILE=window20 \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

Limitation report / failure taxonomy audit for the frozen graph outputs:

```bash
FAILURE_AUDIT_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/failure_taxonomy
mkdir -p "$FAILURE_AUDIT_DIR"

python scripts/audit_retrieval_failure_taxonomy.py \
  --run vidore "$VIDORE_DATA/MMQA_dev.jsonl" "$VIDORE_OUT/vidore-v3_dev_graph_ppr_base.prediction.json" \
  --run mmdocir "$MMDOCIR_DATA/MMQA_dev.jsonl" "$MMDOCIR_OUT/mmdocir_dev_graph_ppr_base.prediction.json" \
  --run opendocvqa "$OPENDOC_DATA/MMQA_dev.jsonl" "$OPENDOC_OUT/opendocvqa_denseheavy125_medium_both.prediction.json" \
  --hit-k 4 \
  --boundary-k 20 \
  --adjacent-window 2 \
  --topn 50 \
  --output-json "$FAILURE_AUDIT_DIR/graph_page_preserve_limitation_report_rich.json" \
  --output-md "$FAILURE_AUDIT_DIR/graph_page_preserve_limitation_report_rich.md" \
  --output-csv "$FAILURE_AUDIT_DIR/graph_page_preserve_limitation_report_rich_cases.csv"
```

This audit uses gold labels to explain failures, so it is for analysis only. Do not use its categories
as routing features. The main categories are `doc_miss_topk`, `doc_missing_from_pool`,
`right_doc_boundary_page`, `right_doc_adjacent_page`, `right_doc_same_doc_sibling`,
`right_doc_late_page`, and `right_doc_gold_page_missing_from_pool`.

The Markdown output is intended to be the limitation report: it collapses primary categories into
document-retrieval gaps, rank-boundary localization, same-document page confusion, and deep/missing
right-document pages. It also reports retrievability ceilings, primary category-by-limitation
matrices, exact failed gold-page rank histograms, rank-5 gold counts, limitation-by-page-rank and
limitation-by-doc-rank matrices, query-cue slices, gold-label shape, top-k evidence tags,
score-margin diagnostics, metadata hotspots by field, and example failures by limitation group.
Use the CSV for custom pivots.

Observed limitation report for frozen Graph-PPR outputs:

| Dataset | qids | page hit@4 | doc hit@4 | page failures | document retrieval gap | same-document page confusion | rank-boundary localization | right-doc deep/missing page |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViDoRe V3 | 14514 | 9383 | 13206 | 5131 | 1308 / 5131 = 25.5% | 1340 / 5131 = 26.1% | 2228 / 5131 = 43.4% | 255 / 5131 = 5.0% |
| MMDocIR | 1658 | 1114 | 1353 | 544 | 305 / 544 = 56.1% | 74 / 544 = 13.6% | 147 / 544 = 27.0% | 18 / 544 = 3.3% |
| OpenDocVQA | 41017 | 26173 | 26901 | 14844 | 14116 / 14844 = 95.1% | 119 / 14844 = 0.8% | 597 / 14844 = 4.0% | 12 / 14844 = 0.1% |

Primary failure category split from the latest rich run:

| Dataset | doc_miss_topk | doc_missing_from_pool | right_doc_boundary_page | right_doc_adjacent_page | right_doc_same_doc_sibling | right_doc_late_page | right_doc_gold_page_missing_from_pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViDoRe V3 | 1235 | 73 | 2228 | 335 | 1005 | 241 | 14 |
| MMDocIR | 248 | 57 | 147 | 15 | 59 | 15 | 3 |
| OpenDocVQA | 13552 | 564 | 597 | 5 | 114 | 11 | 1 |

Rank and document-position diagnostics:

| Dataset | rank-5 gold failures | right-doc failures | rank-5 gold within right-doc failures | dominant failed gold-doc bucket |
|---|---:|---:|---:|---|
| ViDoRe V3 | 470 / 5131 = 9.2% | 3823 | 449 / 3823 = 11.7% | `top4` = 3823 |
| MMDocIR | 16 / 544 = 2.9% | 239 | 15 / 239 = 6.3% | `top4` = 239 |
| OpenDocVQA | 1184 / 14844 = 8.0% | 728 | 207 / 728 = 28.4% | `doc_5_10` = 5220 |

Operational findings:

1. ViDoRe is mostly a page-local failure problem after the right document is already present. With
   `--boundary-k 20`, rank-boundary localization is the largest bucket, followed by same-document
   page confusion; local evidence, heading/breadcrumb anchors, exact MaxSim boundary checks, and
   content/OCR/layout verifiers are plausible next tests.
2. MMDocIR is mixed, but document discovery is now the largest limitation. Boundary rescue can only
   attack the 239 right-document failures; the 305 document-retrieval-gap failures need stronger
   document/support recall. Heading/breadcrumb anchors are still a good cheap test for financial
   reports and academic papers because section names can bridge pages inside a retrieved document.
3. OpenDocVQA is not primarily a boundary/localization problem under this packed-document setup.
   More than 95% of page failures are document-retrieval gaps, so unconditional page-local reranking
   should not be expected to help and already produced a negative full-dev result.
4. Rank-5 gold is useful but limited: 470 ViDoRe failures, 16 MMDocIR failures, and 1184 OpenDocVQA
   failures have gold exactly at rank 5. The direct right-document rank-5 target is smaller:
   449 ViDoRe, 15 MMDocIR, and 207 OpenDocVQA failures, so top4-vs-rank5 page swaps alone cannot
   solve the dominant issue.
5. Dataset-specific hotspots point to different fixes: ViDoRe finance/table repositories are dominated
   by same-document and boundary errors, MMDocIR financial reports and academic papers mix document
   gaps with page-local errors, and OpenDocVQA needs document discovery/pack selection before page
   localization.

Exact MaxSim boundary verifier for Graph-PPR:

```bash
MAXSIM_BOUNDARY_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/boundary_exact_maxsim
mkdir -p "$MAXSIM_BOUNDARY_DIR"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$VIDORE_DATA/MMQA_dev.jsonl" \
  --base-prediction "$VIDORE_OUT/vidore-v3_dev_graph_ppr_base.prediction.json" \
  --embedding-dir "$VIDORE_ROOT/embeddings/colpali-v1.2_vidore-v3_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/vidore_exact_maxsim_boundary.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/vidore_exact_maxsim_boundary.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/vidore_exact_maxsim_boundary.cases.json"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$MMDOCIR_DATA/MMQA_dev.jsonl" \
  --base-prediction "$MMDOCIR_OUT/mmdocir_dev_graph_ppr_base.prediction.json" \
  --embedding-dir "$MMDOCIR_ROOT/embeddings/colpali-v1.2_mm-docir_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary.cases.json"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$OPENDOC_DATA/MMQA_dev.jsonl" \
  --base-prediction "$OPENDOC_OUT/opendocvqa_denseheavy125_medium_both.prediction.json" \
  --embedding-dir "$OPENDOC_ROOT/embeddings/colpali-v1.2_opendocvqa_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/opendocvqa_exact_maxsim_boundary.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/opendocvqa_exact_maxsim_boundary.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/opendocvqa_exact_maxsim_boundary.cases.json"
```

This is a local, label-free boundary test: exact ColPali MaxSim scores only the current top-4 pages
and rank 5, then swaps rank 5 into top 4 if MaxSim beats the weakest current top-4 page. It should
be reported against the frozen Graph-PPR base; if full-dev losses exceed recoveries, keep Graph-PPR
as the final method and treat the MaxSim boundary run as a diagnostic.

Gated subset smoke test for unsafe full-dev MaxSim runs:

```bash
MAXSIM_BOUNDARY_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/boundary_exact_maxsim
mkdir -p "$MAXSIM_BOUNDARY_DIR"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$MMDOCIR_DATA/MMQA_dev.jsonl" \
  --base-prediction "$MMDOCIR_OUT/mmdocir_dev_graph_ppr_base.prediction.json" \
  --embedding-dir "$MMDOCIR_ROOT/embeddings/colpali-v1.2_mm-docir_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --sample-qids 300 \
  --sample-seed 17 \
  --boundary-doc-policy weakest_doc \
  --min-exact-margin 0.25 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.cases.json"
```

This gate remains non-oracle: it uses only base scores, document agreement, and exact MaxSim. Start
with `weakest_doc` to avoid document swaps. If acceptance is too low, relax to
`--boundary-doc-policy topk_doc --min-boundary-doc-topk-count 1`; if losses remain, add
`--max-base-margin-ratio-4-5 0.02` to require an uncertain base rank-4/rank-5 boundary.

Observed MMDocIR 300-query subset results:

| policy | sample | accepted | base page hit@4 | candidate page hit@4 | recovered | lost | net | page recall@4 | doc recall@4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `weakest_doc`, `min_exact_margin=0.25` | 300 | 43 | 186 | 184 | 0 | 2 | -2 | 0.5884 | 0.7900 |
| `topk_doc`, `min_exact_margin=0.25` | 300 | 57 | 186 | 183 | 0 | 3 | -3 | 0.5834 | 0.7900 |

Do not scale either observed gated setting as-is. Relaxing from `weakest_doc` to `topk_doc`
accepted more swaps but only increased losses, so exact rank-5 MaxSim is not acting as a reliable
MMDocIR page-evidence verifier under these gates. Treat this path as diagnostic unless a stricter
margin/uncertainty subset produces nonzero recoveries and positive net recovery.

## Dataset Summary

| Dataset | Env script | Work root | Data folder | Embedding name | Output subdir | Current/expected scale |
|---|---|---|---|---|---|---|
| MMDocIR | `mmdocir/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG` | `$LOCAL_DATA_DIR/mm-docir` | `colpali-v1.2_mm-docir_dev` | `$LOCAL_OUTPUT_DIR/mmdocir` | 313 docs, 20395 pages, 1658 QAs |
| SciEGQA-Bench | `sciegqa/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG` | `$LOCAL_DATA_DIR/sci-egqa-bench` | `colpali-v1.2_sci-egqa-bench_dev` | `$LOCAL_OUTPUT_DIR/sciegqa` | 80 docs, 1823 pages, 1623 QAs |
| ViDoSeek | `vidoseek/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG` | `$LOCAL_DATA_DIR/vidoseek` | `colpali-v1.2_vidoseek_dev` | `$LOCAL_OUTPUT_DIR/vidoseek` | 290 docs, 5349 pages, 1142 QAs |
| ViDoRe V3 | `vidore/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG` | `$LOCAL_DATA_DIR/vidore-v3` | `colpali-v1.2_vidore-v3_dev` | `$LOCAL_OUTPUT_DIR/vidore-v3` | 189 docs, 19252 pages, 14514 QAs, all languages |
| OpenDocVQA | `opendocvqa/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG` | `$LOCAL_DATA_DIR/opendocvqa` | `colpali-v1.2_opendocvqa_dev` | `$LOCAL_OUTPUT_DIR/opendocvqa` | gated; 3223 packed docs, 206267 pages, 41017 QAs |
| MMLongBench DocQA | `mmlongbench/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG` | `$LOCAL_DATA_DIR/mmlongbench-docqa` | `colpali-v1.2_mmlongbench-docqa_dev` | `$LOCAL_OUTPUT_DIR/mmlongbench-docqa` | DocQA subsets: `longdocurl`, `mmlongdoc`, `slidevqa`; counts written by prepare summary |
| DUDE | `dude/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG` | `$LOCAL_DATA_DIR/dude` | `colpali-v1.2_dude_dev` | `$LOCAL_OUTPUT_DIR/dude` | multi-page DocQA; exact page labels from answer bbox pages; counts written by prepare summary |

Notes:

- OpenDocVQA groups individual corpus images into artificial 64-page packs. Page recall is meaningful; doc recall is only a packing artifact.
- ViDoRe V3 uses input HF split `test` but writes local M3DocRAG split files named `*_dev.*`.
- ViDoRe and OpenDocVQA env files force Hugging Face caches under their scratch work roots to avoid home quota failures.
- MMLongBench DocQA uses `ans_page_list` as exact zero-based page labels. SlideVQA image filenames are one-based, so the converter maps them back to zero-based `page_idx`.
- DUDE uses the official Hugging Face loader, renders source PDFs to page images, and defaults to skipping rows without answer page boxes because they have no exact page retrieval target.
- All embedding sbatch files use `--resume`, so resubmitting after timeout is safe.
- On compute nodes where plain `python` points to base and misses `torch`/`faiss`, use `"$REPO_ROOT/env/bin/python"` for direct commands. The dataset `run_plain_top224_*.sh` wrappers now default to that interpreter through `PYTHON_BIN`.

## MMDocIR

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmdocir/env_hpc.sh
```

Prepare:

```bash
python mmdocir/prepare_mmdocir.py \
  --download \
  --snapshot-dir "$MMDocIR_WORK_ROOT/hf_snapshot/MMDocIR_Evaluation_Dataset" \
  --output-root "$LOCAL_DATA_DIR/mm-docir"
```

Sanity values already observed after doc-id normalization:

```text
docs 313
pages 20395
qas 1658
missing_gold_pages 0
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  mmdocir/sbatch_embed_mmdocir_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" -name "*.safetensors" | wc -l
# expected: 313
```

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/mmdocir"

python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl"
```

`plain_top224`:

```bash
bash mmdocir/run_plain_top224_mmdocir.sh

python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl"
```

Observed full `plain_top224`:

```text
n_qids 1658
page_recall@1 0.4136
page_recall@4 0.6075
page_recall@10 0.6913
page_recall@20 0.7480
page_recall@100 0.8305
page_recall@1000 0.9192
doc_recall@1 0.6852
doc_recall@4 0.8058
doc_recall@20 0.8890
doc_recall@100 0.9692
doc_recall@1000 0.9867
page_hit@4 1068/1658
doc_hit@4 1336/1658
improved_doc_rank_count 326
reranked_top4_doc_count 1336
```

MMDocIR full baseline metrics are not recorded here yet. Re-run the baseline eval command above if the comparison table is needed.

## SciEGQA-Bench

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source sciegqa/env_hpc.sh
```

Prepare:

```bash
python sciegqa/prepare_sciegqa_bench.py \
  --download \
  --snapshot-dir "$SciEGQA_WORK_ROOT/hf_snapshot/SciEGQA-Bench" \
  --output-root "$LOCAL_DATA_DIR/sci-egqa-bench"
```

Sanity values already observed:

```text
docs 80
pages 1823
qas 1623
missing_gold_pages 0
```

Embedding:

```bash
sbatch --time=06:00:00 --array=0-3 --export=ALL,NUM_SHARDS=4,BATCH_SIZE=2 \
  sciegqa/sbatch_embed_sciegqa_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" -name "*.safetensors" | wc -l
# expected: 80
```

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/sciegqa"

python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/sciegqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Observed full baseline:

```text
page_recall@4 0.6898
page_recall@100 0.9227
doc_recall@4 0.8848
doc_recall@100 1.0
```

`plain_top224`:

```bash
bash sciegqa/run_plain_top224_sciegqa.sh

python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/sciegqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/sci-egqa-bench/MMQA_dev.jsonl"
```

Observed full `plain_top224`:

```text
page_recall@4 0.7394
page_recall@100 0.9393
doc_recall@4 0.9070
doc_recall@100 1.0
```

## ViDoSeek

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidoseek/env_hpc.sh
```

Prepare:

```bash
python vidoseek/prepare_vidoseek.py \
  --download \
  --snapshot-dir "$VIDOSEEK_WORK_ROOT/hf_snapshot/ViDoSeek" \
  --output-root "$LOCAL_DATA_DIR/vidoseek"
```

Sanity values already observed:

```text
docs 290
pages 5349
qas 1142
missing_gold_pages 0
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  vidoseek/sbatch_embed_vidoseek_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" -name "*.safetensors" | wc -l
# expected: 290
```

Index:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidoseek" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/vidoseek"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidoseek" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/vidoseek/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidoseek/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl"
```

Observed full baseline:

```text
n_qids 1142
page_recall@1 0.6979
page_recall@4 0.8958
page_recall@10 0.9545
page_recall@20 0.9746
doc_recall@1 0.9921
doc_recall@4 0.9982
doc_recall@10 1.0
page_hit@4 1023/1142
doc_hit@4 1140/1142
```

`plain_top224`:

```bash
bash vidoseek/run_plain_top224_vidoseek.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidoseek/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl"
```

Observed full `plain_top224`:

```text
n_qids 1142
page_recall@1 0.6830
page_recall@4 0.8958
page_recall@10 0.9623
page_recall@20 0.9842
doc_recall@1 0.9939
doc_recall@4 0.9982
doc_recall@20 1.0
page_hit@4 1023/1142
doc_hit@4 1140/1142
improved_doc_rank_count 6
reranked_top4_doc_count 1140
```

## ViDoRe V3

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source vidore/env_hpc.sh
```

Prepare all public V3 domains:

```bash
python vidore/prepare_vidore_v3.py \
  --download \
  --cache-dir "$VIDORE_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/vidore-v3"
```

Optional English-only folder:

```bash
python vidore/prepare_vidore_v3.py \
  --download \
  --cache-dir "$VIDORE_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/vidore-v3-english" \
  --language english
```

Sanity values already observed for all languages:

```text
docs 189
pages 19252
qas 14514
missing_gold_pages 0
missing_qrels 0
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-15 --export=ALL,NUM_SHARDS=16,BATCH_SIZE=2 \
  vidore/sbatch_embed_vidore_v3_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" -name "*.safetensors" | wc -l
# expected: 189
```

If you prepared `vidore-v3-english`, override the data root and output names manually or add a separate wrapper before running the standard pipeline.

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidore-v3" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/vidore-v3"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidore-v3" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

ViDoRe has 14514 queries, so the batch array is safer:

```bash
sbatch --time=24:00:00 --array=0-15%4 \
  --export=ALL,NUM_SHARDS=16,TOP_PAGES=1000,FAISS_NPROBE=4,SAVE_EVERY=25 \
  vidore/sbatch_retrieval_vidore_v3_array.sh
```

After all shards finish, merge them:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000_shards/shard_*_of_16.json" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

`plain_top224`:

```bash
bash vidore/run_plain_top224_vidore_v3.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

For the full ViDoRe V3 set, run `plain_top224` as an array after the merged baseline prediction exists:

```bash
sbatch --time=24:00:00 --array=0-15%4 \
  --export=ALL,NUM_SHARDS=16,TOP_PAGES=1000,BASE_ONLY_PAGE_BATCH_SIZE=64 \
  vidore/sbatch_plain_top224_vidore_v3_array.sh
```

Merge sharded `plain_top224` predictions:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_shards/shard_*_of_16_prediction.json" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

Then evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

Observed full baseline:

```text
n_qids 14514
page_recall@1 0.1555
page_recall@4 0.3001
page_recall@20 0.4925
page_recall@100 0.6689
page_recall@1000 0.8922
doc_recall@1 0.6344
doc_recall@4 0.8700
doc_recall@20 0.9729
doc_recall@100 0.9985
doc_recall@1000 0.9989
page_hit@4 8417/14514
doc_hit@4 12833/14514
```

Observed full `plain_top224`:

```text
n_qids 14514
page_recall@1 0.1730
page_recall@4 0.3312
page_recall@20 0.5431
page_recall@100 0.7246
page_recall@1000 0.8922
doc_recall@1 0.6586
doc_recall@4 0.8854
doc_recall@20 0.9809
doc_recall@100 0.9986
doc_recall@1000 0.9989
page_hit@4 9092/14514
doc_hit@4 13039/14514
```

ViDoRe V3 embeddings, FAISS index, full baseline, and full `plain_top224` are complete.

## OpenDocVQA

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh
```

Authentication is required because `NTT-hil-insight/OpenDocVQA-Corpus` is gated:

```bash
huggingface-cli login
huggingface-cli whoami
```

The login token should be saved under:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/hf_home
```

Quick gated-corpus access test:

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset(
    "NTT-hil-insight/OpenDocVQA-Corpus",
    "all",
    split="test",
    streaming=True,
)
row = next(iter(ds))
print(row.keys())
print("doc_id:", row.get("doc_id"))
print("dataset_name:", row.get("dataset_name"))
PY
```

If the row prints, access is working. A Python shutdown abort after printing the row is an environment cleanup issue; the prep script has a streaming-corpus clean-exit workaround.

Smoke prep:

```bash
"$REPO_ROOT/env/bin/python" opendocvqa/prepare_opendocvqa.py \
  --cache-dir "$OPENDOCVQA_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/opendocvqa_smoke_infovqa_v6" \
  --qa-config infovqa \
  --corpus-config infovqa \
  --qa-split test \
  --corpus-split test \
  --dataset-name infovqa \
  --corpus-scope relevant_only \
  --max-queries 50 \
  --streaming-corpus
```

Observed smoke sanity:

```text
docs 1
pages 15
qas 50
missing_gold_pages 0
```

Observed smoke retrieval results:

```text
baseline page_recall@1 0.92
baseline page_recall@4 0.98
baseline page_recall@20 1.0
plain_top224 page_recall@1 0.94
plain_top224 page_recall@4 0.98
plain_top224 page_recall@10 1.0
doc_recall@1 1.0 for both, because all smoke pages are in one artificial pack
```

Full prep uses a node-local Hugging Face cache to avoid shared-filesystem lock errors such as `OSError: [Errno 37] No locks available`:

```bash
export HF_TOKEN="$(cat "$OPENDOCVQA_WORK_ROOT/hf_home/token")"
export ODVQA_NODE_CACHE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/$USER/opendocvqa_hf_cache"
mkdir -p "$ODVQA_NODE_CACHE"

export HF_HOME="$ODVQA_NODE_CACHE/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$ODVQA_NODE_CACHE/xdg_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

"$REPO_ROOT/env/bin/python" opendocvqa/prepare_opendocvqa.py \
  --cache-dir "$ODVQA_NODE_CACHE/load_dataset_cache" \
  --output-root "$LOCAL_DATA_DIR/opendocvqa" \
  --corpus-config all \
  --corpus-split test \
  --streaming-corpus
```

Observed full prep sanity:

```text
docs 3223
pages 206267
qas 41017
missing_final_gold_page_uids 0
dataset_counts {'chartqa': 20882, 'coyo': 65294, 'docvqa': 12767, 'dude': 27955, 'infovqa': 5485, 'mpmqa': 10018, 'openwikitable': 1257, 'slidevqa': 52380, 'visualmrc': 10229}
```

Embedding:

```bash
sbatch --time=24:00:00 --array=0-63%8 \
  --export=ALL,NUM_SHARDS=64,BATCH_SIZE=2,DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa",EMBEDDING_NAME=colpali-v1.2_opendocvqa_dev \
  opendocvqa/sbatch_embed_opendocvqa_array.sh
```

Smoke embedding batch:

```bash
sbatch --time=01:00:00 --array=0-0 \
  --export=ALL,NUM_SHARDS=1,BATCH_SIZE=2,DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa_smoke_infovqa_v6",EMBEDDING_NAME=colpali-v1.2_opendocvqa_smoke_infovqa_v6_dev \
  opendocvqa/sbatch_embed_opendocvqa_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" -name "*.safetensors" | wc -l
jq length "$LOCAL_DATA_DIR/opendocvqa/dev_doc_ids.json"
# counts should match; expected full count: 3223
```

Observed full embedding/index status:

```text
embedding_count 3223/3223
index_path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/embeddings/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat/index.bin
index_size 103G
index_meta_size 95K
```

Index:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/opendocvqa"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Full OpenDocVQA retrieval is very slow as one foreground process. Run it as a sharded GPU array instead:

```bash
sbatch --time=24:00:00 --array=0-63%4 \
  --export=ALL,NUM_SHARDS=64,TOP_PAGES=1000,FAISS_NPROBE=4,SAVE_EVERY=25 \
  opendocvqa/sbatch_retrieval_opendocvqa_array.sh
```

Each shard writes:

```text
$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000_shards/shard_<idx>_of_64.json
```

After all shards finish, merge them:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000_shards/shard_*_of_64.json" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Observed full-index first-100 baseline sanity:

```text
n_qids 100
page_recall@1 0.46
page_recall@4 0.635
page_recall@20 0.735
page_recall@100 0.845
page_recall@1000 0.96
doc_recall@1 0.47
doc_recall@4 0.66
doc_recall@100 0.9
doc_recall@1000 0.995
page_hit@4 68/100
doc_hit@4 71/100
```

Observed full baseline after merging 64 retrieval shards:

```text
n_qids 41017
page_recall@1 0.4251
page_recall@4 0.5803
page_recall@20 0.6968
page_recall@100 0.7949
page_recall@1000 0.9130
doc_recall@1 0.4334
doc_recall@4 0.5944
doc_recall@20 0.7241
doc_recall@100 0.8401
doc_recall@1000 0.9668
page_hit@4 26161/41017
doc_hit@4 26756/41017
```

`plain_top224`:

```bash
bash opendocvqa/run_plain_top224_opendocvqa.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

For the full OpenDocVQA set, run `plain_top224` as a sharded GPU array after the merged baseline prediction exists:

```bash
sbatch --time=24:00:00 --array=0-63%4 \
  --export=ALL,NUM_SHARDS=64,TOP_PAGES=1000,BASE_ONLY_PAGE_BATCH_SIZE=64 \
  opendocvqa/sbatch_plain_top224_opendocvqa_array.sh
```

Merge sharded `plain_top224` predictions:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_shards/shard_*_of_64_prediction.json" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Then evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Observed full `plain_top224` after merging 64 shards:

```text
n_qids 41017
page_recall@1 0.3516
page_recall@4 0.5122
page_recall@20 0.6599
page_recall@100 0.7932
page_recall@1000 0.9130
doc_recall@1 0.3624
doc_recall@4 0.5307
doc_recall@20 0.6955
doc_recall@100 0.8563
doc_recall@1000 0.9668
page_hit@4 22604/41017
doc_hit@4 23404/41017
```

Compared with full baseline, `plain_top224` underperforms at early ranks on OpenDocVQA:

```text
page_recall@1 0.4251 -> 0.3516
page_recall@4 0.5803 -> 0.5122
page_recall@20 0.6968 -> 0.6599
page_recall@100 0.7949 -> 0.7932
page_recall@1000 0.9130 -> 0.9130
```

## MMLongBench DocQA

This workflow prepares the page-labeled DocQA part of MMLongBench: `longdocurl`, `mmlongdoc`, and `slidevqa` across `K8`, `K16`, `K32`, `K64`, and `K128`. The converter writes the same local files as the other page-labeled datasets, including `MMQA_dev.jsonl`, `doc_pages_dev.jsonl`, `qids_dev.jsonl`, `gold_pages_dev.jsonl`, and `dev_doc_ids.json`.

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmlongbench/env_hpc.sh
```

Prepare:

```bash
"$REPO_ROOT/env/bin/python" mmlongbench/prepare_mmlongbench.py \
  --download \
  --snapshot-dir "$MMLONGBENCH_WORK_ROOT/hf_snapshot/MMLongBench" \
  --output-root "$LOCAL_DATA_DIR/mmlongbench-docqa"
```

Smoke-test prepare before the full run:

```bash
"$REPO_ROOT/env/bin/python" mmlongbench/prepare_mmlongbench.py \
  --download \
  --snapshot-dir "$MMLONGBENCH_WORK_ROOT/hf_snapshot/MMLongBench" \
  --output-root "$LOCAL_DATA_DIR/mmlongbench-docqa-smoke" \
  --length K8 \
  --max-examples-per-file 5
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-31 --export=ALL,NUM_SHARDS=32,BATCH_SIZE=2 \
  mmlongbench/sbatch_embed_mmlongbench_array.sh
```

Expected embedding count equals:

```bash
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["LOCAL_DATA_DIR"]) / "mmlongbench-docqa"
print(len(json.loads((root / "dev_doc_ids.json").read_text())))
PY
```

Index:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mmlongbench-docqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/mmlongbench-docqa"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mmlongbench-docqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/mmlongbench-docqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmlongbench-docqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/mmlongbench-docqa/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

Plain top-224:

```bash
bash mmlongbench/run_plain_top224_mmlongbench.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmlongbench-docqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/mmlongbench-docqa/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

SPLADE:

```bash
DATA_NAME=mmlongbench-docqa \
DATA_ROOT="$LOCAL_DATA_DIR/mmlongbench-docqa" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/plain_top224_ret1000_prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/doc_rrf_plain_top224_splade" \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RRF_K=10 \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

Graph-PPR with the current page-labeled default:

```bash
DATA_NAME=mmlongbench-docqa \
DATA_ROOT="$LOCAL_DATA_DIR/mmlongbench-docqa" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/doc_rrf_plain_top224_splade/mmlongbench-docqa_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/graph_ppr_plain_top224_splade" \
GRAPH_PROFILE=page_rank_probe \
GRAPH_LABEL="mmlongbench_docqa_denseheavy125_medium_both" \
FINAL_TOP_PAGES=1000 \
PER_DOC_PAGE_LIMIT=0 \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RESTART_PROB=0.15 \
PPR_ITERS=30 \
PAGE_DOC_EDGE_WEIGHT=1.0 \
SAME_DOC_WINDOW=1 \
ADJACENT_PAGE_EDGE_WEIGHT=0.25 \
FINAL_PAGE_SEED_WEIGHT=1.0 \
FINAL_PPR_PAGE_WEIGHT=0.5 \
FINAL_PPR_DOC_WEIGHT=0.25 \
bash scripts/run_external_graph_ppr_pipeline.sh
```

## DUDE

DUDE is the replacement target for MP-DocVQA availability issues. It is a multi-page document QA benchmark with PDF/OCR assets and answer page bounding boxes. The converter writes the same local files as the other page-labeled datasets and, by default, excludes rows without exact answer page boxes.

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source dude/env_hpc.sh
```

Prepare:

```bash
"$REPO_ROOT/env/bin/python" dude/prepare_dude.py \
  --output-root "$LOCAL_DATA_DIR/dude" \
  --hf-config Amazon_due \
  --source-split val
```

If the DUDE binaries were already extracted, pass them directly:

```bash
"$REPO_ROOT/env/bin/python" dude/prepare_dude.py \
  --data-dir /path/to/DUDE_train-val-test_binaries \
  --output-root "$LOCAL_DATA_DIR/dude" \
  --hf-config Amazon_due \
  --source-split val
```

Sanity check:

```bash
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
p=os.environ["LOCAL_DATA_DIR"] + "/dude/prepare_dev_summary.json"
s=json.load(open(p))
for k in ["source_row_count","qa_count","doc_count","page_count","answer_page_base","answer_page_base_missing_counts","skipped_no_gold_page_count","missing_gold_page_count","answer_type_counts_kept"]:
    print(k, s.get(k))
PY
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-31 --export=ALL,NUM_SHARDS=32,BATCH_SIZE=2 \
  dude/sbatch_embed_dude_array.sh
```

Index:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/dude" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Dense retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/dude"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/dude" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/dude/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/dude/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/dude/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

Plain Top-224:

```bash
bash dude/run_plain_top224_dude.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/dude/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

SPLADE + page-preserving Graph-PPR:

```bash
DATA_NAME=dude \
DATA_ROOT="$LOCAL_DATA_DIR/dude" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/dude/doc_rrf_plain_top224_splade" \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RRF_K=10 \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh

DATA_NAME=dude \
DATA_ROOT="$LOCAL_DATA_DIR/dude" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/dude/doc_rrf_plain_top224_splade/dude_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/dude/graph_ppr_plain_top224_splade" \
GRAPH_PROFILE=page_rank_probe \
GRAPH_LABEL="dude_denseheavy125_medium_both" \
FINAL_TOP_PAGES=1000 \
PER_DOC_PAGE_LIMIT=0 \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RESTART_PROB=0.15 \
PPR_ITERS=30 \
PAGE_DOC_EDGE_WEIGHT=1.0 \
SAME_DOC_WINDOW=1 \
ADJACENT_PAGE_EDGE_WEIGHT=0.25 \
FINAL_PAGE_SEED_WEIGHT=1.0 \
FINAL_PPR_PAGE_WEIGHT=0.5 \
FINAL_PPR_DOC_WEIGHT=0.25 \
bash scripts/run_external_graph_ppr_pipeline.sh
```


## Common Sanity Check

Use this after any prepare step, replacing `DATASET_DIR` with the local dataset folder name:

```bash
export DATASET_DIR=sci-egqa-bench

python - <<'PY'
import json, os
from pathlib import Path
from PIL import Image

dataset_dir = os.environ["DATASET_DIR"]
root = Path(os.environ["LOCAL_DATA_DIR"]) / dataset_dir
doc_ids = json.loads((root / "dev_doc_ids.json").read_text())
pages = [json.loads(line) for line in (root / "doc_pages_dev.jsonl").open() if line.strip()]
qas = [json.loads(line) for line in (root / "MMQA_dev.jsonl").open() if line.strip()]
page_uids = {row["page_uid"] for row in pages}
missing = []
for row in qas:
    for uid in row.get("metadata", {}).get("gold_page_uids", []):
        if uid not in page_uids:
            missing.append((row["qid"], uid))

print("docs", len(doc_ids))
print("pages", len(pages))
print("qas", len(qas))
print("missing_gold_pages", len(missing))
print("first_missing", missing[:5])
sample = pages[0]
path = root / sample["image_path"]
img = Image.open(path)
print("sample_page", sample)
print("sample_exists", path.exists(), path)
print("sample_size", img.size, img.mode)
PY
```

## Common Job Checks

Inspect Slurm status:

```bash
export JOB_ID=10879877

squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList%30
```

Check embedding logs:

```bash
export WORK_ROOT="$VIDORE_WORK_ROOT"

grep -iE "traceback|error|exception|killed|oom|out of memory|cuda error|time limit" \
  "$WORK_ROOT/logs"/embed_"$JOB_ID"_*.err
```

Examples:

```bash
grep -iE "traceback|error|exception|killed|oom|out of memory|cuda error|time limit" \
  "$SciEGQA_WORK_ROOT/logs"/embed_"$JOB_ID"_*.err

grep -iE "traceback|error|exception|killed|oom|out of memory|cuda error|time limit" \
  "$VIDORE_WORK_ROOT/logs"/embed_"$JOB_ID"_*.err
```

If a job timed out but produced some `.safetensors`, resubmit the same sbatch command. The embedding scripts use `--resume`.

## File-Specific READMEs

Per-dataset implementation details live in:

- `mmdocir/README.md`
- `sciegqa/README.md`
- `vidoseek/README.md`
- `vidore/README.md`
- `opendocvqa/README.md`
- `mmlongbench/README.md`
- `dude/README.md`
