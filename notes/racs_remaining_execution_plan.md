# RACS: closing the remaining revision items

Updated 2026-09-19 after the first online-runtime attempts failed.
This plan does not change Overleaf or launch remote jobs automatically.

## Current completion target

The author has explicitly removed the first-revision stopping point: finish
the reviewer requests, then verify the integrated manuscript. Historical
first-draft wording below describes earlier handoffs, not the present goal.
The remaining experimental gap is R1.2's full online cost. A separate online
benchmark is now implemented in `scripts/benchmark_racs_online.py` with launcher
`examples/sbatch_racs_online_runtime.sh`; see `notes/racs_online_runtime_protocol.md`.
It performs query encoding, FAISS, exact and required approximate MaxSim,
SPLADE, graph/CAPP and Qwen inference, with upstream/output equivalence gates.
Jobs 15911849 and 15911871 both failed. The first log identifies an incorrect
inner-product-only index guard in the new benchmark, before timing. The fix
preserves the saved FAISS metric (L2 or IP), records it separately from the
embedding-dot page scoring, and leaves all ranking replay gates unchanged.
The second detailed log was not supplied. One fresh execution after pulling
the correction was attempted as 15911873. That job confirmed L2 IVF/IP quantizer,
loaded dense assets, then failed resolving the SPLADE tokenizer vocabulary from
the offline Hub-name load. No online runtime result exists. The author has local
SPLADE files and requested using them. Explicit local model/tokenizer directory
support is now implemented, with no download/fallback and unchanged replay gates.
The actual HPC directory path is still needed before a read-only file check and
one new submission. Old failure reports remain untouched.
Do not call R1.2 complete until a successful online report is audited.

The independent raw-record audit of job 15911732 has now passed with the same
3.9422036205 / 4.7029153292 s means and +19.2966% cached-retrieval overhead.
This is not another measured run and does not change that result's scope.
All already-validated QA experiments remain reusable; no duplicate BGE, budget,
injection or feature-ablation run is required. Manual insertion, final PDF QA,
coauthor review and the final point-by-point response remain after the experiment.

## 1. BGE reader result

Job 15911612 is complete. On September 18 the author provided `COMPLETED`,
exit `0:0`, elapsed `02:26:55`, node `gpu009`, and the postflight result:
2441 questions, four pages each, EM 37.85333879557559 / F1 43.8947152806227.
The stdout marker agrees with
`output/racs_bge_reader_top4_15911612/validated_result.json`.
The baseline guide now includes 37.85 EM / 43.89 F1 in its manual table.
Do not submit a duplicate. This is evidence from the completed validated HPC
run, not a local rescore of its full predictions. Tuning provenance and matched
end-to-end runtime remain separate; this result does not resolve either.

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

The transferred inventory was inspected on September 18. Among 98 candidate
records, four match all four recorded main train/eval gold/base paths: the
30-feature fixed-alpha 0.35, 0.40 and 0.45 models, plus the feature-matrix
`all` model at 0.40. All four record automatic tuning disabled and null tuning
summaries. No candidate read errors were reported. This supports the author's
memory of multiple alpha trials, but does not identify the data or criterion
used to select 0.40; other candidates with different inputs cannot establish
that history. Ask the author/advisor to confirm the written protocol or
explicitly qualify it. A fresh tuning run would not prove the original history.

Author response on September 18: **"I am not sure"** which selection split
was used. Do not ask the author to guess or treat the existing prose as
execution evidence. The consolidated revision packet provides advisor-draft
wording that reports the fixed alpha and explicitly states that its historical
selection split is unverified. Advisor review or a genuine historical record
is needed before strengthening that claim; a new tuning run cannot repair
the historical record retroactively.

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

### Prerequisite audit (job 15911623 completed)

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

User-provided audit output now confirms job 15911623 completed with exit 0:0
in 20 seconds on node011. All four graph artifacts exist and their qid sets
match the 2,441-question gold file; no missing direct recorded dependency paths
were reported. Evidence:
`/Users/hoseinerfan/.codex/attachments/edc16373-5d03-4af8-b94c-e65a5b8f14c6/pasted-text.txt`.

Important configuration distinctions from that output:

- The main base uses `mmqa_dev_exact_maxsim_nprobe4_ret1000.prediction.json`
  and final selection `score`.
- All three auxiliary rankings instead use
  `mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json` and final
  selection `mmr_doc_diverse`. Their graph input/output limits are still
  1,000 pages; the filename alone does not establish a different graph limit.
- The shared sparse artifact's saved summary identifies
  `naver/splade-cocondenser-ensembledistil`, top 1,000 pages and 32 query terms.
  This is not the earlier recollection of SPLADE v3. Preserve the recorded
  checkpoint identity unless more direct evidence establishes stale metadata;
  do not silently substitute a v3 artifact in a reproduction or benchmark.
- The legacy dense prediction file exists, but its expected same-stem summary
  is absent. This is a provenance gap, not a missing ranking file.

These checks establish available artifacts and recorded configuration, not
recomputed graph equivalence, transitive model/index availability, or a runtime
result. The full report on HPC is
`output/racs_runtime_audit_15911623/prerequisites.json`. It has now been copied
to `outputs/racs_runtime_prerequisites_15911623.json` and inspected locally.
Every graph has 1000 candidate entries for all 2441 questions. Only its
per-question `graph` metadata varies; other recorded metadata is constant
within each artifact. This does not establish candidate uniqueness or graph
replay equivalence. Alpha findings are above; the optional BGE snapshot in this
earlier report is superseded by the completed-job result in section 1.

The targeted check `scripts/probe_racs_legacy_dense.py` reads the
exact legacy prediction path recorded in the audit, verifies its question-set
digest, and groups embedded configuration across all rows. If a same-stem
detail JSONL exists, it reports only the first nonblank row's configuration,
explicitly without asserting that file is from the same run. No training,
retrieval, prediction modification, or report-file writing occurs. Four local
tests cover read-only behavior, cohort mismatch, configuration variation and
the limited companion-file scope.

The author ran this probe as srun job 15911725 and supplied its output:
`/Users/hoseinerfan/.codex/attachments/b7838225-42ce-47cf-a38f-d7f8603042be/pasted-text.txt`.
All 2441 legacy prediction rows match the gold cohort and have the same four
recorded core settings: `base_score_source=approx_page_maxsim_topk`,
`approx_base_page_token_topk=224`, scorer `query_mean`, selector `global_topk`.
No core field is missing. This is now direct saved-metadata evidence, not an
inference from the filename. The same-stem detail JSONL exists; its first row
additionally records adaptive-k disabled, fp32 coarse scoring, batch size 0
and diagnostics enabled. That first-row-only check does not prove the detail
file's whole-cohort identity with the prediction artifact.

Consequently the main base uses Exact MaxSim, while the reproducing CAPP
auxiliary inputs originate from the legacy approximate 224-page-token-budget
route. Do not silently substitute Exact MaxSim for that route when timing the
historical full system, or relabel the main base as approximate. This resolves
the core scoring identity, not every original invocation/environment setting.

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

The actual paired end-to-end benchmark launcher is not yet implemented.
The legacy dense core configuration is now verified from saved metadata.
The correctness experiment is `validate_racs_graph_replay.py`, launched by
`examples/sbatch_racs_graph_replay.sh`: 16 qids chosen by a fixed hash order,
independent of outcomes, shared by the main and all three auxiliary graphs.
It uses recorded graph settings and cached original dense/sparse predictions;
compares all 1000 ordered page IDs and scores for each query/branch; and
writes only a new diagnostic report. Score tolerance is rtol=1e-6, atol=1e-8,
with exact candidate order required. Any mismatch produces a nonzero exit.
Gold labels/answers are not passed into ranking.

The graph artifacts omit some newer CLI settings. The replay supplies only an
explicit compatibility map, recorded in the report (four applied fields for
the main base, 29 for each legacy branch), including disabled optional extra
inputs and fixed PPR iterations. Unlisted missing settings are rejected.
These are replay hypotheses, not recovered historical settings. A sample
match would validate this sample's outputs under them, not prove the original
command or all-question equivalence. The replay does not regenerate dense or
sparse retrieval, run CAPP/reader inference, or measure full runtime/memory.
Seven local tests and a settings-materialization check on all four audited
configurations passed. The author subsequently supplied job 15911730's result:
`COMPLETED|0:0|00:00:31|node011`. Each of the four branches has 16/16 candidate
set matches, 16/16 complete-order matches and 16/16 score-tolerance matches.
The report is `output/racs_graph_replay_15911730/replay.json`. This is evidence
from pasted accounting/stdout; the full report has not been copied locally.
The 31-second job duration includes preparation and is not a query-time result.

### Completed controlled graph-to-answer experiment (partial pipeline)

`scripts/benchmark_racs_graph_reader.py` and
`examples/sbatch_racs_graph_reader_runtime.sh` now implement an actual paired
timing experiment, rather than another standalone inventory. It starts at
cached dense/sparse rankings, so **it is not the requested full online
query-to-answer benchmark**. This scope is explicit in every report. It does
not close R1.2 or the end-to-end action above by relabeling a partial result.

The experiment selects 128 qids by the same outcome-independent hash rule,
with four disjoint warm-up qids. On one GPU allocation it runs four repeats
per condition in eight fresh, sequential workers with balanced GPP/CAPP order.
GPP recomputes its main graph; CAPP recomputes that graph and all three auxiliary
graphs, builds per-query source maps, and scores/reranks with the saved model.
Both use Qwen2-VL-7B-Instruct, 16-bit weights, exactly four pages, the same prompt,
and explicit CUDA synchronization around the whole query timer. PDF rendering
is inside that timer, with no document-image cache across questions. OS cache
state remains uncontrolled. Initial input/model preparation is separate.

All regenerated 1000-page graph orders/scores and full CAPP orders must match
the saved references, outside the timer. Failed validation prevents a validated
aggregate. Fresh-worker CPU high-water RSS and GPU allocated/reserved peaks are
reported separately, not subtracted into a claimed incremental scorer footprint.
The input bundles contain only sampled upstream rows; these are not production
full-corpus memory estimates. Repeated answer variation is reported, but no
paper QA score is replaced by this timing subset's generations.

Dense/SPLADE query encoding/search and both dense scoring routes remain excluded.
Their faithful online replay/integration is still needed for end-to-end claims.
Nine new local tests plus 15 existing replay/CAPP benchmark tests pass. Live
execution is now complete: job 15911732, exit 0:0, 01:23:12, gpu009, NVIDIA A100
80GB PCIe. The author supplied the validated report status and summary fields
in `/Users/hoseinerfan/.codex/attachments/7b8bc269-b11b-4862-baac-93a20ac3524d/pasted-text.txt`.
On 128 questions with four repeats, mean graph-to-answer time is 3.9422036205 s
for GPP and 4.7029153292 s for CAPP: +0.7607117088 s (+19.2966%). Median/p95 are
3.5649435821/5.9405987030 s and 4.2865021792/7.9158358886 s, respectively.
GPU peak allocated/reserved memory is 19.3656/24.7188 GiB in every worker.
CPU whole-worker peak RSS ranges are GPP 4.9857–5.4267 GiB and CAPP
6.4532–6.8022 GiB; initialization already reached each worker's high-water mark.
No within-method answer variation was reported across repeats. Do not interpret
these as identical answers across methods, isolated scorer memory, or full online
latency. The success status reports the ranking/cohort/hardware gates passed;
raw per-query report files have not been independently recomputed locally.
See `notes/racs_graph_reader_runtime_protocol.md` for exact boundaries and commands.
Step 9 of the consolidated Overleaf packet now includes a manual LaTeX insertion
for this completed experiment. Do not launch another copy for the same summary.

Local verification: five audit unit tests cover question-ID conflicts, graph
configuration variation, absent/oversized metadata, output isolation and
tuning-record filtering. The Slurm launcher passes Bash syntax checking.

## 5. Manual manuscript integration

The single self-contained handoff is now
`notes/racs_overleaf_complete_revision_packet.md`. It consolidates the verified
LaTeX fragments in manuscript order and adds the configuration/tuning edits.
Use that packet rather than inserting it in addition to the older guides.
The three original guides below remain the detailed evidence/context sources.

Use the three manual guides in this order: runtime/budget/control;
baselines/ablation; feature rationale. The consolidated Limitations body in
the third guide replaces the earlier piecemeal limitations edits. Preserve
the original main-table cohort and do not paste matched-subset QA into it.
The baseline guide now includes a combined retrieval/QA LaTeX table and BGE
model-card entry. BGE's validated four-page QA is 37.85 EM / 43.89 F1; QA for
LambdaMART and monoT5 remains unreported. Replace the earlier retrieval-only
draft rather than inserting both versions of the same subsection/table.

Original Overleaf source/ZIP remains unchanged. The author requested manual
integration; no rewritten or compiled revised manuscript is claimed. After
manual insertion, compile and inspect table widths, numbering, citations,
float placement, and page limit, then share the first draft with the advisor.
If the author wants direct work on a separate manuscript copy, that needs an
explicit change to the earlier manual-only instruction.

### Configuration wording before the advisor draft

- Apply the feature-rationale guide's separation of the no-hyperlink main GPP
  base from the three legacy auxiliary sources. Do not relabel the full model
  as the no-source ablation or call the entire pipeline hyperlink-free.
- The sparse input's summary records `naver/splade-cocondenser-ensembledistil`.
  In the paper's Compared Methods paragraph, the generic SPLADE description
  can be made explicit with that checkpoint, attributed to saved metadata.
  The September 15 text does not claim SPLADE v3; do not introduce that claim
  from recollection. Checkpoint bytes have not been independently inspected.
- The original Method and Training paragraphs claim held-out-training alpha
  selection and final refit. Keep this as an explicit advisor-confirmation
  item; the new metadata does not independently prove it. A draft can state
  the verified fixed alpha, but deleting selection details alone does not
  resolve the final paper's selection-protocol obligation.
- Keep new experimental tables separate from the original main-result rows.
  The guides contain draft insertions, not a finished compiled camera-ready.

## Next handoff after the completed graph-to-answer run

Do not repeat the metadata probe, graph precheck, or completed partial benchmark.
Preserve `output/racs_graph_reader_runtime_15911732/` (manifest, runtime summary,
input bundles and eight worker reports). Use the updated manual packet for the
advisor's first draft; the original manuscript remains untouched.

The unresolved experiment is online retrieval/scoring integration, not graph or
reader timing. Historical alpha-selection provenance remains unverified, and
manual Overleaf insertion, compilation/layout checking and advisor review remain.
No new HPC submission is prepared by this result-recording update.
