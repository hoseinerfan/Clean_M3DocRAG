# LambdaMART and monoT5 downstream QA

Requested September 23, 2026. The thesis comparison in section5.tex reports
retrieval metrics for these methods. Their saved top-1000 rankings and matching
retrieval tables were found on HPC. No identifiable QA evaluation for either
method was found in a filename scan through depth four (384 output directories).
This search does not establish that a differently named or elsewhere-stored run
never existed. The author has now requested new EM/F1 evaluations.

## Fixed protocol

- Reuse the saved LambdaMART alpha=0.45 and monoT5 alpha=1.00 rankings from
  `output/m3docvqa_standard_reranker_baselines`. No training or reranking.
- Verify exact QID coverage and identical sets of 1000 distinct candidate
  pages against the paper's Exact MaxSim GPP base, for all 2441 questions.
- Validate method metadata, the pinned gold/base paths, and thesis retrieval
  values before starting reader inference. LambdaMART is LightGBM LambdaRank
  with 40 features. monoT5 is the top-1000 trial, not the top-100 trial.
- Qwen2-VL-7B-Instruct, 16-bit, four ordered pages, one shard, image cache16,
  save every25. These explicit settings match the validated BGE reader job.
- Separate array tasks: task0 LambdaMART, task1 monoT5, one GPU each, 6h limit.
- Refuse existing output directories. Fingerprint inputs and code; verify
  intended selected pages, complete questions, answer/timing fields, unchanged
  fingerprints, and independently recomputed EM/F1 after completion.
- Do not infer timing comparisons from job durations or different allocations.
- Do not copy page-level doc@4 into the paper's support-document recall column;
  those are distinct reported metrics.

## Submission

Pull the updated `codex/mmdocir-hpc-workflow` branch, then submit once:

```bash
mkdir -p output
sbatch examples/sbatch_racs_remaining_reranker_readers.sh
```

Logs: `output/racs_reranker_reader_ARRAYID_TASKID.out/.err`.
Results: `output/racs_lambdamart_reader_top4_ARRAYID/validated_result.json`
and `output/racs_monot5_reader_top4_ARRAYID/validated_result.json`.
No QA scores should be entered into the manuscript until validation succeeds.

## Existing jobs

The author reports all earlier jobs finished, including readers 15942317 and
15942394. Their final validated results have not yet been read: the HPC mount
returned Device not configured during this turn. Request their accounting and
validated_result.json from the cluster terminal; disappearance from squeue alone
does not establish success. Do not resubmit those jobs.
