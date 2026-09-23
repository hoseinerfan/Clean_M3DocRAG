# Prospective CAPP training and alpha-selection experiment

## Job 15942315: reported completion and next reader job

The author's pasted stdout reports `validated_training_and_retrieval`, a saved
final model and completed development evaluation. Slurm accounting confirmation
has not yet been pasted; `squeue` no longer lists this job.

- 16,965 fit and 4,241 validation questions; no skipped validation questions.
- Selected alpha **0.45**, validation page@4 **0.558358877623202** (2368 hits).
  Alpha 0.40 gives **0.5555293562838953** (2356 hits). These are validation,
  not development scores; the difference does not establish significance.
- Full-refit development page@4 **0.7755941499085923**, versus GPP
  **0.6375685557586838**; historical CAPP was **0.7714808043875686**.
- New page@10 **0.8697440585009141**, below historical CAPP **0.8825**;
  new page@1000 **0.9771480804387569**, equal to GPP.
- These new results validate this prospective protocol, not historical alpha
  selection or an alpha-only causal comparison; the model was refitted.

The author submitted the next GPU job as **15942317**, using standalone bundle
`outputs/racs_controlled_reader_20260923.tar.gz`. Qwen2-VL-7B-Instruct, 16-bit,
four pages, all 2441 dev questions. Uses the known external-retrieval reader and
its usual evaluation, plus checks that each output used the intended ordered
four pages. Existing full training job 15942315 and alpha 0.45 are pinned.
Output: `output/racs_controlled_reader_top4_JOBID/validated_result.json`.
Three local validation tests and shell syntax checks passed. The training
result/model/manifest and gold fingerprints are checked before GPU inference.
Old models, rankings, QA and manuscript remain unchanged.

Prepared September 23, 2026 at the author's request. This is a NEW experiment,
not evidence of the original June selection procedure. Do not replace manuscript
numbers or describe the historical results as train-selected before evaluating
the new model. Historical dev results have already been inspected; this does not
make development a previously untouched test set.

## Fixed protocol

- Existing adaptive exact-only norm05 training/dev labels and Exact MaxSim
  no-hyperlink GPP base candidates; top 1000 candidates.
- All 30 non-visual features; logistic scorer; weighted BCE.
- Three explicitly paired auxiliary GPP rankings: no-hyperlink,
  document-hyperlink, page-hyperlink. Dev paths reproduce the saved full model.
  The `_mmr_target1_train_real` train counterparts are a prospective, explicit
  choice, NOT a verified recovery of the original training sources. This run
  therefore includes hyperlink-derived source features, despite its no-hyperlink
  base. No separate SPLADE auxiliary list is added; the GPP inputs already
  incorporate their retrieval sources.
- No environment-based input fallback or silently omitted source. Missing files,
  missing source questions, missing candidate-page text, unexpected cohort
  counts, and fit/dev QID overlap stop the run.
- Split the 21,206 pseudo-labeled training questions by QID, with seed 13:
  16,965 fitting questions and 4,241 validation questions. Unlabeled training
  questions are excluded before splitting. No in-pool-positive filter is used
  to improve validation scores: labeled questions with no in-pool evidence
  remain in the validation denominator, but cannot supply positive fitting
  examples. Full refit should use 17,929 in-pool-positive questions.
- Standardization is fitted on the fitting matrix only during selection.
- Epochs 80; learning rate 0.01; weight decay 0.0001; batch size 65536;
  positive weight capped at 20. Rank-stratified negatives: up to ten in each
  band, maximum 64 per question. These are fixed for this prospective run.
- Select alpha from 0.00, 0.05, ..., 1.00 by validation page@4. Ties select
  the smallest alpha. There is no requirement that the result be 0.40.
- Refit scaler and scorer from initialization on all eligible training
  questions. Freeze the selected alpha, persist the model, THEN load dev data.
- Apply using only dev question text and retrieval/page features. Dev relevance
  labels enter evaluation, never fitting, alpha selection, or inference features.
- Check unchanged candidate sets and report retrieval metrics over the labeled
  dev subset. Preserve predictions for all 2,441 dev questions for subsequent QA.

## Implementation and evidence

- `scripts/run_racs_controlled_training.py` reuses the existing training module's
  fitting, scaling, tuning, scoring and evaluation functions; the historical
  module is not edited. The upload includes a snapshot of that module.
- `examples/sbatch_racs_controlled_training.sh`: compute partition, 8 CPU cores,
  128 GB RAM, 12-hour limit, no GPU. This is a resource limit, not a duration estimate.
- Five local unit/integration tests passed, including end-to-end miniature
  fit/select/refit/evaluation; missing-input rejection; duplicate QIDs and output
  overwrite rejection; missing-source-QID rejection; and deterministic tie handling.
- The miniature experiment also verifies that the final model is saved before
  dev labels are loaded, and changing only dev labels leaves selected alpha,
  weights, scaler and predicted rankings unchanged.
- This tests software behavior, not the existence/content of HPC inputs. The
  HPC mount was unavailable during preparation. The login-node preflight checks
  file presence; the compute-node job checks content and recorded cohort counts.

Artifacts are written exclusively into `output/racs_controlled_training_JOBID/`:

- `manifest.json`: protocol, full settings, exact paths, SHA-256 of every input,
  Python/NumPy/platform versions, thread environment and code hashes.
- `code/`: runner and training-module snapshots.
- `split.json`: actual fitting/validation QIDs and excluded unlabeled QIDs.
- `fit.model.json`: fitting-split model before alpha selection.
- `selection.json`: validation sample count, every alpha score and chosen alpha.
- `final.model.json`: model after full refit, scaler and recorded training metadata.
- `dev.prediction.json`: new retrieval rankings, not reader-generated answers.
- `result.json`, `retrieval.table.md`: checks and retrieval results.
- `failure.json` if the run fails: not a usable experimental result.

Existing output directories are refused. No historical predictions/models,
Overleaf sources, or Git state are overwritten. The final result status is
`validated_training_and_retrieval`, NOT a claim of completed downstream QA.

## Current code delivery: Git

At the author's request, code updates now use
`https://github.com/hoseinerfan/Clean_M3DocRAG.git`, branch
`codex/mmdocir-hpc-workflow`. The three repository batch scripts in `examples/`
use the versioned Python files in `scripts/`; they no longer require uploaded
archive folders. Existing uploaded snapshots remain unchanged.

On HPC, check the branch and working tree before pulling:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git status --short
git branch --show-current
git pull --ff-only origin codex/mmdocir-hpc-workflow
```

Only pull on the stated branch; do not discard local changes to resolve a
conflict. Jobs 15942317 (alpha 0.45) and 15942394 (alpha 0.40) were already
submitted using standalone snapshots. Do not resubmit them after pulling.
Their last reported status was RUNNING, with no fatal errors in the pasted logs.
Future reader submissions remain explicitly pinned to training job 15942315.
Updating this delivery mechanism does not change the experimental settings.

## Historical archive transfer and submission

Bundle: `outputs/racs_controlled_training_20260923.tar.gz`.
It extracts to a NEW `racs_controlled_training_20260923/` directory containing
the runner, training module, batch script and this protocol.

From the Mac:

```bash
scp -o 'User=aerfanshekooh@jacks.local' \
  /Users/hoseinerfan/Desktop/Clean_M3DocRAG/outputs/racs_controlled_training_20260923.tar.gz \
  innovator.sdstate.edu:/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/
```

From the cluster login terminal:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG &&
tar --keep-old-files -xzf racs_controlled_training_20260923.tar.gz &&
mkdir -p output &&
env/bin/python -B racs_controlled_training_20260923/run_racs_controlled_training.py \
  --repo-root "$PWD" --preflight-only &&
sbatch racs_controlled_training_20260923/sbatch_racs_controlled_training.sh
```

If preflight prints MISSING, stop and share that output; do not substitute
files or submit a fallback run. If the upload folder already exists, tar stops
to avoid replacement. Once the first extraction succeeds, repeat only preflight
and sbatch, not extraction.

## Next after completion

Read Slurm exit status and `result.json`, verify the selected alpha and split
counts, and compare new retrieval with the saved historical results. Then run
the fixed four-page Qwen2-VL reader on the NEW predictions. If using the new model
as the paper's primary model, update dependent QA, budget, ablation and runtime
comparisons as required; old-model analyses must not be relabeled as new-model
results. Reader jobs 15942317 and 15942394 are now awaiting final validation.
