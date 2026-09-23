# Same-model alpha 0.40 versus 0.45 QA comparison

Requested by the author after controlled training job 15942315 selected alpha
0.45 on its training holdout. This is a development-set sensitivity comparison,
NOT another training-only selection run or a reconstruction of historical alpha
choice. Do not switch the claimed train-selected alpha based on development QA.

## Fixed and changed components

- Freeze job15942315's `final.model.json`: same weights, scaler, 30 features.
- Use its recorded dev base, page text, three auxiliary rankings, questions,
  and exact training-code snapshot; verify SHA-256 fingerprints.
- Score each page once. First reblend at 0.45 and require all 2441 complete
  1000-page rankings and base scores to match saved predictions.
- Only then reblend those same scores at 0.40. Preserve candidate sets; report
  changed ordered top4 lists, changed top4 sets, and retrieval for both alphas.
- Run the same external-retrieval Qwen2-VL-7B-Instruct reader, 16-bit, four
  pages, one shard, image cache16, save every25, all2441 dev questions. These
  match the already prepared alpha0.45 reader job's explicit CLI settings.
- Check every completed QA row used exactly the intended four ordered pages,
  has an answer and valid timing, and that all2441 QIDs are present.
- Do not use the historical paper alpha0.40 model: that would confound alpha
  with model differences. No model training is performed in this new job.
- No runtime comparison is claimed across different cluster allocations.

## Files and status

Submitted by the author as **15942394**. Last reported status: RUNNING;
preparation reached 500/2441 questions, before reader inference. The alpha0.45
reader job15942317 had reached 2120/2441 questions. Neither final reader result
has yet been provided.

Subsequent code delivery uses the `codex/mmdocir-hpc-workflow` branch of
`hoseinerfan/Clean_M3DocRAG`. The repository batch script uses `scripts/`
directly. Existing standalone bundles and running jobs are unchanged; do not
submit a duplicate job after pulling updates.

- `scripts/prepare_racs_alpha040_reader.py`
- `examples/sbatch_racs_alpha040_reader_top4.sh`
- `tests/test_racs_alpha040_reader.py`
- Standalone bundle: `outputs/racs_alpha040_reader_20260923.tar.gz`
- Cluster upload directory: `racs_alpha040_reader_20260923/`
- New results: `output/racs_alpha040_reader_top4_JOBID/`
- stdout/stderr: `output/racs_alpha040_reader_JOBID.out/.err`
- Final QA report: `validated_result.json`
- Retrieval report: `retrieval_comparison.json`
- Six combined alpha040/controlled-reader tests pass; shell syntax checked.

The GPU allocation includes a CPU preparation phase. It reuses the original
controlled-run numerical thread settings for replay; reader inference begins
only after successful reproduction of the reference0.45 rankings.

Do not cancel, alter or duplicate an already submitted alpha0.45 reader job.
No historical outputs, model files, manuscript files, or Git state are changed.
