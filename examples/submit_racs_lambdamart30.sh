#!/usr/bin/env bash
# Run on the HPC login node, from the repository root, after pulling code.
set -euo pipefail
[[ -f scripts/run_racs_lambdamart30.py && -x env/bin/python ]] || {
  echo "Run this from the Clean_M3DocRAG root on HPC." >&2; exit 1;
}
mkdir -p output
env/bin/python -B -c 'import lightgbm; print("LightGBM:", lightgbm.__version__)'
env/bin/python -B scripts/run_racs_lambdamart30.py preflight --repo-root "$PWD"
RACS_TRAIN_JOB=$(sbatch --parsable examples/sbatch_racs_lambdamart30_train.sh)
RACS_TRAIN_JOB=${RACS_TRAIN_JOB%%;*}
[[ "$RACS_TRAIN_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected training job ID" >&2; exit 1; }
echo "Training job: $RACS_TRAIN_JOB"
RACS_TRAIN_DIR="$PWD/output/racs_lambdamart30_${RACS_TRAIN_JOB}"
RACS_READER_JOB=$(sbatch --parsable --dependency="afterok:${RACS_TRAIN_JOB}" \
  --kill-on-invalid-dep=yes examples/sbatch_racs_lambdamart30_reader.sh "$RACS_TRAIN_DIR")
RACS_READER_JOB=${RACS_READER_JOB%%;*}
[[ "$RACS_READER_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected reader job ID" >&2; exit 1; }
echo "Reader job (starts after successful training): $RACS_READER_JOB"
echo "Training result: $RACS_TRAIN_DIR/result.json"
echo "QA result: $PWD/output/racs_lambdamart30_reader_${RACS_READER_JOB}/validated_result.json"
echo "Check status: sacct -X -j $RACS_TRAIN_JOB,$RACS_READER_JOB --format=JobID,State,ExitCode,Elapsed -P"
