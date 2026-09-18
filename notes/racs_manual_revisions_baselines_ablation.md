# RACS manual revisions: stronger baselines and feature ablations

Prepared 2026-09-17. This is a manual-edit guide; the September 15 Overleaf ZIP
has not been changed. These additions follow the runtime, reader-budget and
injection-control guide in `notes/racs_manual_revisions_runtime_budget_control.md`.

## 1. Feature ablation: ready to insert

In the September 15 `main.tex`, keep the `Feature Ablation` heading and
`subsec:features` label. Replace the existing table and its three following
paragraphs, stopping before `Promotion Audit`, with the fragment below.
Keep exactly one `tab:feature-ablation` label. The wider table uses `table*`;
check its placement and the page limit after compiling in Overleaf.

The six expanded ablation retrieval reports cover 2,188 labeled questions out
of 2,441 predictions. Their QA scores were independently rescored in HPC job
15911329 on the same 2,441 qids, with four selected pages per question, and
match the saved evaluations. Rank-only, source-only and content-only retrieval
rows are retained from the submitted paper; their QA is not added without a
corresponding audit. A dash does not mean zero or prove that no saved run exists.

~~~latex
\begin{table*}[tbp]
    \centering
    \small
    \caption{Feature-family ablation for CAPP on GPP. Page-level metrics
    use 2,188 pseudo-labeled development questions; QA uses all 2,441
    development questions and a four-page reader budget. EM and F1 are
    percentages. A dash denotes QA not reported in this table.
    Bold marks the largest value in each column among the reported rows.}
    \label{tab:feature-ablation}
    \begin{tabular}{lrrrr}
        \toprule
        Variant & page@4 & page@10 & QA EM & QA F1 \\
        \midrule
        GPP & 0.6376 & 0.8030 & 37.69 & 43.47 \\
        Rank only & 0.6376 & 0.8030 & -- & -- \\
        Source only & 0.6344 & 0.7980 & -- & -- \\
        Structure only & 0.7573 & 0.8844 & 38.71 & 44.75 \\
        Content only & 0.6604 & 0.7980 & -- & -- \\
        Structure + content & \best{0.7779} & 0.8935 & 39.04 & 45.14 \\
        No rank & 0.7692 & \best{0.8958} & 38.92 & 45.04 \\
        No source & 0.7719 & 0.8729 & 39.12 & 45.22 \\
        No structure & 0.6508 & 0.8140 & 37.65 & 43.46 \\
        No content & 0.7573 & 0.8702 & 38.59 & 44.69 \\
        All features & 0.7715 & 0.8825 & \best{39.41} & \best{45.69} \\
        \bottomrule
    \end{tabular}
\end{table*}

Table~\ref{tab:feature-ablation} reports single-family, combined-family,
and leave-one-family-out variants. The six expanded ablations---structure
only, structure + content, no rank, no source, no structure, and no
content---use the same fixed blend weight $\alpha=0.40$ and recorded
training/evaluation gold, base-ranking, and page-text inputs as the full
model. The saved model settings otherwise differ only in feature subset.
Downstream QA uses the same Qwen2-VL-7B-Instruct reader configuration
and four selected pages for every question.

Structure is the strongest individual family under the reported page-level
metrics. Rank-only matches GPP on page@4 and page@10, while source-only
is close to GPP and content-only gives a modest page@4 improvement.
Structure + content obtains the highest page@4 (0.7779), whereas removing
rank features obtains the highest page@10 (0.8958). Among the four
leave-one-family-out variants, removing structure produces the largest
observed reduction in QA F1, from 45.69 to 43.46. These results support
the usefulness of document/page localization cues on this dataset, but
do not establish that the same feature ordering holds on other corpora.

The full model has the highest observed QA EM and F1 among the evaluated
variants. In particular, structure + content reaches 45.14 F1 and no
source reaches 45.22 F1, compared with 45.69 for all features. The
structure + content comparison is therefore explicit in
Table~\ref{tab:feature-ablation}, rather than inferred from the main-results
table. The differing retrieval and QA rankings show that improved
pseudo-page hit rates do not necessarily translate into improved answer
quality. These are descriptive development-set comparisons; we do not
claim statistically significant differences between the close variants.
~~~

### Configuration caveats to retain

- `No rank` maps to `source_structure_content` (25 features); `No content`
  maps to `rank_source_structure` (15). Structure-only has 6 features,
  structure + content 21, no source 26, no structure 24, and full CAPP 30.
- Matching recorded main inputs does not recover the literal auxiliary-source
  filenames used by each historical source-bearing ablation. Do not upgrade
  this audit into a claim of complete historical training reproduction.
- Full CAPP's verified inference replay consumes three legacy auxiliary rankings,
  two generated with active hyperlink edges. The base is no-hyperlink GPP.
  Describe these two levels separately; do not call the full pipeline entirely
  hyperlink-free or relabel its scores as the no-source variant.
- Fixed alpha 0.40 in the saved models does not establish which split was used
  in the earlier manual alpha trials. That selection claim needs author/advisor
  confirmation independently of this ablation table.

## 2. Stronger baselines: confirmed retrieval, pending reader result

The retained thesis rows are confirmed by the saved reports:

| Method | Candidate / rerank depth | Blend alpha | page@4 | Downstream QA |
| --- | --- | ---: | ---: | --- |
| GPP reference | 1000 / not applicable | not applicable | 0.6376 | 37.69 EM / 43.47 F1 |
| LambdaMART | 1000 candidates | 0.45 | 0.6408 | Not yet located by name |
| BGE reranker base | 1000 / 1000 | 0.20 | 0.6705 | Four-page reader job 15911612 submitted; result pending |
| monoT5 base MS MARCO 10k | 1000 / 1000 | 1.00 | 0.6609 | Not yet located by name |
| Full CAPP on GPP | 1000 candidates | 0.40 | 0.7715 | 39.41 EM / 45.69 F1 |

LambdaMART is confirmed as LightGBM with the LambdaRank objective and **40
features**, not the 30-feature logistic CAPP model. Its saved tuning record
selects alpha 0.45. BGE uses maximum sequence length 512 and at most 6,000
page-text characters; monoT5 uses 512 and 4,000. Both neural baseline reports
record 744 empty-text encounters, not 744 missing questions.

These are different reranking systems, not an identical-feature classifier
ablation or a matched total-compute comparison. Full CAPP also consumes its
auxiliary ranking inputs. BGE's alpha-selection split is not established by
the saved report. The filename search found no name-matched QA reports, but
that is not proof that no differently named result exists.

Do not insert invented EM/F1 or support-document metrics for these baselines.
Finalize the baseline manuscript paragraph/table after the new BGE QA result
is checked. The retained top-1000 BGE result is used; the separate top-100 and
unblended BGE trials must not be substituted for it. The BGE run does not
complete matched reader QA for LambdaMART or monoT5, nor resolve tuning provenance.

## 3. One new HPC run: frozen reader on existing BGE rankings

New files:

- `examples/sbatch_racs_bge_reader_top4.sh`: one GPU, 8 CPUs, 64 GB, six-hour limit.
- `scripts/validate_racs_bge_reader.py`: preflight and postflight checks.
- `tests/test_validate_racs_bge_reader.py`: synthetic CPU tests; not an HPC model run.

The job checks the pinned BGE summary and per-question metadata, exactly 2,441
unique gold qids, and the same 1,000 distinct candidate pages as GPP for each
question. It preserves the saved top-four order/scores, makes an isolated
reader input, records file hashes, and runs Qwen2-VL-7B-Instruct with 16-bit
weights. It verifies that all four intended pages were selected for every qid.
It never trains or reranks BGE/CAPP, changes gold labels, silently filters
questions, or overwrites historical results. No Git command runs on the GPU node.

From the HPC repository root, after the changes are pushed:

~~~bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
if [ "$(git branch --show-current)" = "codex/mmdocir-hpc-workflow" ]; then
  git pull --ff-only origin codex/mmdocir-hpc-workflow &&
  mkdir -p output &&
  sbatch examples/sbatch_racs_bge_reader_top4.sh
else
  echo "Stop: unexpected branch. Please share its name."
  git branch --show-current
fi
~~~

Save the returned job ID. Results will be isolated under
`output/racs_bge_reader_top4_JOBID/`. After completion, share `sacct` state and
exit code, the last 12 lines of `output/racs_bge_reader_JOBID.out`, and the last
20 lines of the corresponding `.err`. Success should include
`BGE_READER_RESULT` with 2,441 questions, four pages each, and EM/F1.
If preflight or page rendering fails, inspect the error before changing inputs.

Local checks passed: six validator unit/integration tests, Bash syntax, and
LaTeX brace/environment/replacement-label checks. This guide has not been
compiled as a manuscript; check layout after manual insertion. The HPC job is
submitted as job 15911612 on HPC; its last reported state was pending for
priority. No completion or QA result has yet been verified. Do not resubmit
merely to apply the writing-only updates in the other guides.
