# Feature-matched LambdaMART baseline — September 26, 2026

## Purpose and scope

The author requested LambdaMART with CAPP's 30 features. This is a **new
controlled baseline**, not a reconstruction or one-factor ablation of the old
40-feature run. No historical predictions, paper tables, or CAPP models are
overwritten. Replace the paper's LambdaMART row only after retrieval and QA
validation complete; update its configuration description at the same time.

Mounted artifact inspection confirmed the old model contains the 30 CAPP
features plus 10 visual-related slots. Nine extra features have zero split
importance; `question_visual_cue` has importance 20. This is evidence that the
old feature lists differ, not that ten useful visual signals were available.

## Matched inputs and new protocol

- Require exact equality to the feature names AND order in the saved full-CAPP
  model, not just a length of 30. Use the shared CAPP feature extractor.
- Use the explicit gold/base/text/source paths from
  `run_racs_controlled_training.input_paths`, the same prospective inputs used
  for controlled CAPP job 15942315. These include three legacy auxiliary source
  rankings, as documented there; source-file names are retained in manifests.
- Same candidate pool: Exact-MaxSim GPP top 1000. Same training-pair builder as
  controlled CAPP, with binary pseudo-page labels and up to 10 negatives per
  rank band. Calculate rank/score/document context from the full pool, not the
  sampled training subset. Standardize using fitting-set statistics.
- Seed 13; 21,206 labeled training questions split into 16,965 fit and 4,241
  validation questions. Validation retains questions with no in-pool positive.
  Select alpha by validation page@4 over 0.00, 0.05, ..., 1.00; ties use the
  smaller alpha. Do NOT reuse the old LambdaMART alpha 0.45 automatically.
- Refit on all eligible training questions (17,929 with an in-pool positive).
  Save the final booster and preprocessing before development evaluation.
  No development relevance labels or answers enter scoring or model selection.
- LightGBM LambdaRank, 300 trees, learning rate 0.05, 31 leaves,
  min_data_in_leaf 30, subsample 0.9, colsample_bytree 0.9, seed 13, 8 threads.
  `subsample_freq` remains the library default (0); do not claim active row
  bagging solely from the `subsample` value. No sklearn fallback.
- Score all 2,441 development questions; retrieval page metrics cover the
  2,188 labeled questions. Validate exact candidate identities/counts.
- Reader: Qwen2-VL-7B-Instruct, 16-bit, four pages, all 2,441 development
  questions, identical existing reader invocation. Validate selected pages
  and independently recompute EM/F1 from saved answers.

Matching features does not equate the losses: LambdaRank is a grouped ranking
objective and CAPP uses weighted binary cross-entropy. CAPP's class weighting
is not transplanted into LambdaRank. The new pair construction, full-pool
feature context, source inputs, and selection protocol also differ from parts
of the legacy LambdaMART pipeline. Accordingly, this is a feature-matched
baseline, not proof that any change from the old LambdaMART score was caused
only by removing ten columns. The controlled CAPP results are the appropriate
reference for a prospective matched-training comparison; historical paper
results retain their separate provenance.

## Files and validation

- Workflow: `scripts/run_racs_lambdamart30.py`.
- Tests: `tests/test_racs_lambdamart30.py`.
- CPU job: `examples/sbatch_racs_lambdamart30_train.sh` (8 CPUs, 128 GiB, 12h).
- GPU job: `examples/sbatch_racs_lambdamart30_reader.sh` (one GPU, 64 GiB, 6h).
- Submission helper: `examples/submit_racs_lambdamart30.sh`.

These are resource/time limits, not predicted runtimes. The reader is submitted
with `afterok` and invalid-dependency cancellation: it runs only if training
and retrieval validation finish successfully. Each run creates a new directory
and refuses to overwrite an existing one.

Training writes `manifest.json` (input hashes, code snapshots and versions),
`split.json`, fit/final booster and model files, `selection.json`,
`dev.prediction.json`, `reader_input.json`, and `result.json`. A persisted-model
reload check precedes evaluation. QA writes its own manifest, predictions,
evaluation and `validated_result.json`. Preparation verifies training hashes;
final QA validation checks input/code hashes, exact four-page consumption,
question coverage and independently calculated answer metrics.

Mounted-drive preflight passed on September 26. The full dataset is intentionally
not loaded over SSHFS for training; computation belongs on a scheduled node.
All six workflow tests passed locally with LightGBM 4.6.0, including actual
fitting, booster serialization/reload, feature-vector equality to CAPP's
training rows, alpha tie-breaking, and candidate-pool validation. All three
shell scripts passed `bash -n`. Full-scale training and QA remain to be run.

## Submission and next step

On HPC, after pulling `codex/mmdocir-hpc-workflow`:

```bash
bash examples/submit_racs_lambdamart30.sh
```

Keep both printed job IDs. After completion, inspect `sacct`, the training
`result.json` and reader `validated_result.json`. Only then replace the old
LambdaMART page@4/EM/F1 row and the feature-count/configuration paragraph.
No new CAPP training or reruns of BGE/monoT5 are required for this baseline.
