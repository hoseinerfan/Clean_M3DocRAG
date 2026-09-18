# RACS: closing the remaining revision items

Updated 2026-09-17 after the author requested all remaining items.
This plan does not change Overleaf or launch remote jobs automatically.

## 1. BGE reader result

Job 15911612 is submitted; the last reported state was pending for priority.
The author runs HPC commands and shares results. Check accounting, the end of
the job's stdout/stderr, and its `BGE_READER_RESULT` marker before inserting
EM/F1. Do not submit a duplicate. The prepared runtime inventory also reads
`output/racs_bge_reader_top4_15911612/validated_result.json` if it already exists;
absence means unavailable at audit time, not failure.

## 2. Alpha selection: written protocol located

The author asked whether the thesis/paper already explains the procedure. Yes:

- Thesis `sdsu_dissertation_template/section3.tex`, lines 314–318: training-only
  20% holdout, maximizing page@4, then full-training refit.
- Thesis `sdsu_dissertation_template/section4.tex`, lines 230–234: same procedure.
- Thesis `sdsu_dissertation_template/section5.tex`, lines 833–838: repeats it.
- September 15 manuscript `main.tex`, Method and Training paragraphs: held-out
  training selection for page@4 followed by final retraining.

This confirms the **documented protocol**, not an independently recovered
execution trace. The final full-model artifact records fixed alpha 0.40 and no
automatic tuning in that run. This does not disprove an earlier selection run.
The provenance note also records a historical recommendation for separate fixed
alpha trials; a recommended command is not evidence of how the final choice
was made. These distinctions should not be collapsed in either direction.

Next: inventory relevant small model/summary files for prior training-only
tuning records. A candidate must match the label version, training/base inputs,
feature set and selected alpha; a different model's 0.40 choice is not proof.
Matching fit/tune counts alone is also insufficient. If no record is recovered,
ask the author/advisor to confirm the written protocol or qualify it. Do not
silently invent a development-tuned or train-tuned history. No retraining is
authorized by this inventory, and a new tuning run cannot establish history.

## 3. Full-model source description

The manual feature-rationale guide includes the corrected evaluation protocol:
no-hyperlink Exact MaxSim GPP base, plus three legacy auxiliary rankings,
including two hyperlink-enabled variants. The full model and the no-source
ablation remain distinct. This text can be reviewed with the advisor now.
The runtime comparison must charge the full method for whichever auxiliary
generation it needs, rather than assuming saved rankings are free.

## 4. Controlled runtime / memory comparison

The existing 196.2 ms/query result is a valid cached-input CAPP measurement,
not a full query-to-answer benchmark. Do not sum historical timers recorded
on different nodes into a purported end-to-end comparison.

### Immediate prerequisite audit (prepared; not yet run)

`examples/sbatch_racs_runtime_prerequisites.sh` requests one CPU and 64 GB on
`compute`, with a 30-minute cap. It loads the four graph artifacts one at a
time, records their saved configuration, checks qid coverage and referenced
file availability, and inventories the upstream dense/sparse summaries. It
also inventories relevant full-feature CAPP tuning records and any completed
BGE result. Only a new `output/racs_runtime_audit_JOBID/` report is written.

This is a configuration check, **not** a new benchmark or model run. Saved
timing fields remain labeled as unverified in scope. Missing recorded paths
may be optional/unused and must be interpreted with the saved graph mode.
Large metadata files skipped by the size limit are reported as unread, not
absent. The current graph wrapper defaults are not treated as historical truth.

### Benchmark design after prerequisite verification

1. Freeze the exact shared and method-specific inputs. GPP must reproduce the
   paper's base rankings. Full CAPP must reproduce its saved candidate sets
   and ranking order using the verified three auxiliary sources. Rebuild a
   small fixed-qid sample first, failing on differences rather than choosing
   a nearby configuration with favorable scores.
2. Compare GPP+reader and full-CAPP+reader on the same hardware, qids, rendered
   pages, reader checkpoint, precision, prompt and four-page budget. Record
   GPU model/driver, CPU allocation, software/code hashes, cache policy and
   resident models. Select any benchmark subset before looking at outcomes;
   report its size and do not present it as a full-development-set result.
3. Define the start/end boundaries explicitly. A genuine online query-to-answer
   benchmark includes query encoding/search/exact scoring, required graph
   ranking(s), CAPP features/scoring where applicable, image preparation and
   answer generation. Offline corpus indexing/page extraction and one-time
   initialization should be reported separately. If a stage cannot be rebuilt,
   label the result as a partial pipeline benchmark rather than end-to-end.
4. Include auxiliary branches unique to full CAPP, and charge any additional
   retrieval used by those branches. Shared work can be reused within a query
   but must not be double-counted or omitted differently across conditions.
   Do not assume the legacy auxiliary ranks use the same dense retrieval as
   the Exact MaxSim base just because both are GPP outputs.
5. Warm up explicitly, then repeat measured passes with balanced method order.
   Synchronize GPU operations at timing boundaries. Report per-query mean,
   median/p95 and serial throughput, plus initialization/preparation costs.
   Keep validation and result serialization outside the declared query timer.
6. Measure the conditions in separate fresh worker processes on the same
   allocation. Record CPU peak RSS and GPU peak allocated/reserved memory
   separately. Compare matched condition-level peaks; do not subtract two
   monotonically increasing high-water marks within one reused process or
   describe memory for validation-reference JSON as deployed scorer memory.
7. Publish runtime differences only after ranking/cohort checks pass. Explain
   any reader answer variation; do not replace the paper's established QA
   results merely because a timing run generates new answers.

The actual paired benchmark launcher is not yet implemented: reconstructing
the upstream branches depends on the audit. This is a concrete outstanding
step, not evidence that the runtime reviewer request is complete.

Local verification: five audit unit tests cover question-ID conflicts, graph
configuration variation, absent/oversized metadata, output isolation and
tuning-record filtering. The Slurm launcher passes Bash syntax checking.

## 5. Manual manuscript integration

Use the three manual guides in this order: runtime/budget/control;
baselines/ablation; feature rationale. The consolidated Limitations body in
the third guide replaces the earlier piecemeal limitations edits. Preserve
the original main-table cohort and do not paste matched-subset QA into it.
The baseline guide now includes a retrieval-only LaTeX table and BGE model-card
entry; downstream BGE numbers remain pending.

Original Overleaf source/ZIP remains unchanged. The author requested manual
integration; no rewritten or compiled revised manuscript is claimed. After
manual insertion, compile and inspect table widths, numbering, citations,
float placement, and page limit, then share the first draft with the advisor.
If the author wants direct work on a separate manuscript copy, that needs an
explicit change to the earlier manual-only instruction.

## HPC handoff after pushing

~~~bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
if [ "$(git branch --show-current)" = "codex/mmdocir-hpc-workflow" ]; then
  git pull --ff-only origin codex/mmdocir-hpc-workflow &&
  mkdir -p output &&
  sbatch examples/sbatch_racs_runtime_prerequisites.sh
else
  echo "Stop: unexpected branch."
  git branch --show-current
fi
~~~

The new commit changes none of the files used by the active BGE reader job;
pulling it does not require canceling or restarting that job. Return the new
audit job ID, then its stdout/stderr after completion. Keep the full JSON on
HPC for targeted follow-up instead of pasting large prediction files.
