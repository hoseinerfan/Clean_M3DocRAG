#!/bin/bash -l
#SBATCH --job-name=racs-gold-injection
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=output/racs_gold_injection_%j.out
#SBATCH --error=output/racs_gold_injection_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1

# Match the existing reader's data/model-directory setup. Pin experiment
# choices below rather than inheriting old oracle-wrapper defaults.
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh

RACS_PYTHON="$PWD/env/bin/python"
test -x "$RACS_PYTHON"
RACS_GOLD=output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl
RACS_BASE=output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json
RACS_RUN_DIR="output/racs_adaptive_gold_injection_top4_${SLURM_JOB_ID:?}"
# Deliberately fail if this job's directory already exists; never mix runs.
mkdir "$RACS_RUN_DIR"
RACS_INPUT="$RACS_RUN_DIR/pseudo_gold_plus_gpp_fill.prediction.json"
RACS_SUBSET="$RACS_RUN_DIR/pseudo_gold_plus_gpp_fill.gold.jsonl"
RACS_QA="$RACS_RUN_DIR/mmqa_dev_pseudo_gold_plus_gpp_fill_qwen2vl_top4.prediction.json"
RACS_EVAL="$RACS_RUN_DIR/mmqa_dev_pseudo_gold_plus_gpp_fill_qwen2vl_top4.eval.json"

printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
fi

"$RACS_PYTHON" scripts/build_m3docvqa_pseudo_gold_reader_input.py \
  --augmented-gold "$RACS_GOLD" --base-prediction "$RACS_BASE" \
  --top-pages 4 --fill-from-base \
  --output-prediction-json "$RACS_INPUT" \
  --output-filtered-gold "$RACS_SUBSET" \
  --output-summary "$RACS_RUN_DIR/input.summary.json"

# Fail before loading the VLM if the cohort or page budget is wrong.
"$RACS_PYTHON" - "$RACS_GOLD" "$RACS_INPUT" "$RACS_SUBSET" "$RACS_BASE" "$RACS_RUN_DIR/run_manifest.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import build_m3docvqa_pseudo_gold_reader_input as oracle

paper = oracle.load_jsonl_by_qid(Path(sys.argv[1]))
expected = {q: r for q, r in paper.items() if oracle.gold_page_uids(r)}
predictions = oracle.load_prediction(Path(sys.argv[2]))
subset = oracle.load_jsonl_by_qid(Path(sys.argv[3]))
if len(paper) != 2441 or len(expected) != 2188 or set(expected) != set(predictions) or set(expected) != set(subset):
    raise ValueError("Injection cohort must equal the paper's 2188 labeled questions")
for qid, row in predictions.items():
    pages = row["page_retrieval_results"]
    if len(pages) != 4 or len({(p[0], p[1]) for p in pages}) != 4:
        raise ValueError(f"Expected four distinct pages: {qid}")
    prefix = oracle.make_gold_rows(oracle.gold_page_uids(expected[qid]), 4)
    if pages[:len(prefix)] != prefix or subset[qid] != expected[qid]:
        raise ValueError(f"Pseudo-gold prefix or reference labels differ: {qid}")
summary = {
    "questions": len(predictions), "reader_pages_each": 4,
    "questions_with_more_than_four_pseudo_pages": sum(len(oracle.gold_page_uids(r)) > 4 for r in expected.values()),
}
def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}

paths = {"paper_gold": sys.argv[1], "injection_input": sys.argv[2],
         "filtered_gold": sys.argv[3], "gpp_base": sys.argv[4],
         "builder_code": "scripts/build_m3docvqa_pseudo_gold_reader_input.py",
         "reader_code": "scripts/run_m3docvqa_external_retrieval_qa.py",
         "launcher_code": "examples/sbatch_racs_gold_injection_top4.sh"}
manifest = {**summary, "input_validation": "passed", "model_name": "Qwen2-VL-7B-Instruct",
            "bits": 16, "protocol": "pseudo-gold prefix, then distinct GPP fill",
            "fingerprints": {key: fingerprint(Path(value)) for key, value in paths.items()},
            "local_model_dir": os.environ.get("LOCAL_MODEL_DIR"),
            "local_data_dir": os.environ.get("LOCAL_DATA_DIR"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
with Path(sys.argv[5]).open("x") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
print("INJECTION_INPUT_VALIDATED " + json.dumps(summary), flush=True)
PY

# One new diagnostic run only. GPP/CAPP QA answers are reused, not regenerated.
# No resume flag: the output directory must be new and isolated.
"$RACS_PYTHON" scripts/run_m3docvqa_external_retrieval_qa.py \
  --prediction-json "$RACS_INPUT" --gold "$RACS_SUBSET" \
  --data-name m3-docvqa --split dev \
  --model-name-or-path Qwen2-VL-7B-Instruct --bits 16 --qa-top-pages 4 \
  --eval-num-shards 1 --eval-shard-id 0 --doc-image-cache-size 16 --save-every 25 \
  --output-prediction-json "$RACS_QA" --output-eval-json "$RACS_EVAL" --run-eval

"$RACS_PYTHON" - "$RACS_INPUT" "$RACS_QA" "$RACS_EVAL" <<'PY'
import json
import sys
from pathlib import Path
with Path(sys.argv[1]).open() as f:
    inputs = json.load(f)
with Path(sys.argv[2]).open() as f:
    predictions = json.load(f)
if len(predictions) != 2188 or set(predictions) != set(inputs):
    raise ValueError("Reader output does not cover all 2188 injection questions")
for qid, row in predictions.items():
    if row["selected_page_retrieval_results"] != inputs[qid]["page_retrieval_results"]:
        raise ValueError(f"Reader did not consume the intended four pages: {qid}")
with Path(sys.argv[3]).open() as f:
    scores = json.load(f)
print("INJECTION_READER_RESULT " + json.dumps({"questions": len(predictions),
      "reader_pages_each": 4, "overall": scores["overall"]}), flush=True)
PY
printf 'saved_injection_directory=%s\n' "$RACS_RUN_DIR"
