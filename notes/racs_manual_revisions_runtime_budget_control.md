# RACS manual revisions: runtime, reader budget, and injection control

Prepared 2026-09-17 for the first revision shared with the advisor.
This is a copy-and-paste guide, not an edited manuscript. No Overleaf source,
ZIP, model, prediction, or HPC result has been changed.

## Scope and insertion map

The anchors below refer to `main.tex` inside the September 15 manuscript ZIP,
not the older `ACM_Paper` checkout. Use labels and quoted text to locate edits;
table/subsection numbers will change after insertion.

| Location | Manual action |
| --- | --- |
| Method, `subsec:capp` | Replace the sentence claiming another reader budget requires retraining/retuning. |
| Experimental Design, `tab:setup` and Reader paragraph | Distinguish main/budget QA on 2,441 questions from the injection control on 2,188. |
| Results, immediately before `subsec:pseudo-quality` | Insert the runtime and reader-budget subsections below, in that order. |
| Pseudo-Label Quality, `tab:pseudo-audit` | Keep the three coverage rows; remove the two reader-F1 rows that mix question populations. |
| Pseudo-Label Quality, final paragraph | Replace the old 55.82-versus-45.69 comparison with the matched four-page control below. |
| Limitations, pseudo-label and reader-budget discussion | Add the control caveat and replace the claim that changing k requires retraining. |

All CAPP numbers below belong to the original full 30-feature model with
fixed alpha 0.40. Do not relabel them as results of the 26-feature `no_source`
variant. The original k=4 reader answers are reused, not regenerated.

These additions address reader-budget sensitivity and the requested injection
diagnostic. The CPU benchmark addresses stage-level computation, but does not
complete an end-to-end overhead or incremental-memory comparison against GPP.

The manuscript already loads `booktabs` and `tabularx` and defines the `Y`
column type used below. No new citation or LaTeX package is needed for these
experimental additions. Compile after manual insertion to check float placement
and the conference page limit; these fragments are not a compiled layout.

## 1. Small protocol edits

### Method: reader-budget claim

Under `subsec:capp`, replace only:

> If a different reader budget is desired, the model should be retrained or
> retuned for that target k.

with:

~~~latex
Section~\ref{subsec:reader-budget} evaluates different reader budgets
using the same trained scorer, fixed $\alpha=0.40$, and saved rankings,
without budget-specific retraining or retuning.
~~~

The preceding assertion that alpha was selected on held-out training
supervision, and the similar Training-paragraph assertion, remain separate
provenance questions. The saved final run fixes alpha at 0.40; the author's
recollection of testing several values does not establish the selection split.
Do not present the new budget sweep as evidence for that earlier tuning claim.

### Experimental Design: reader paragraph

Replace the existing paragraph headed `\textbf{Reader.}` with:

~~~latex
\textbf{Reader.}
Downstream QA uses Qwen2-VL \cite{Wang2024Qwen2VL}
(7B Instruct, 16-bit weights) over the selected page images.
The main experiments use four pages; the reader-budget analysis varies
the maximum number of pages over $k\in\{1,2,4,8\}$ while keeping the
retrieval rankings and CAPP model fixed. Main and budget-sensitivity QA
results use all 2441 development questions and their original answer
annotations. The pseudo-gold injection diagnostic uses the matched
subset of 2188 questions with pseudo-page labels. Page-level retrieval
metrics also use those 2188 labeled questions.
~~~

In `tab:setup`, replace these two rows:

~~~latex
        Reader budget & 4 pages in main runs; 1, 2, 4, 8 in sensitivity analysis \\
        QA evaluation & 2441 questions for main/budget runs; 2188 for injection control \\
~~~

Replace its caption with:

~~~latex
    \caption{M3DocVQA experimental setup. Page-level retrieval uses
    pseudo-labeled questions; main and budget-sensitivity QA use all
    development questions. The injection control uses the labeled subset.}
~~~

In Implementation protocol, replace the sentence beginning
“For downstream QA, we pass the top four selected page images” with:

~~~latex
For downstream QA, we pass up to $k$ selected page images to the VLM reader
($k=4$ in the main experiments) and evaluate generated answers against
the original M3DocVQA answer annotations.
~~~

The general statement that CAPP stays within its base candidate pool remains
correct. The injection experiment below is a separate privileged diagnostic,
not a new CAPP retrieval method.

## 2. Results addition: runtime

Insert immediately before the existing Pseudo-Label Quality subsection.
This table reports the full benchmark-process memory scope explicitly;
do not shorten that label to “CAPP memory overhead.”

~~~latex
\subsection{Computational Cost}
\label{subsec:capp-cost}

We measured the CPU cost of CAPP with precomputed base and auxiliary
rankings and cached page-text representations. The benchmark reranked
1000 candidate pages for each of 2441 development questions on one CPU
thread of an Intel Xeon Gold 6342 at 2.80\,GHz. We performed one full
warmup pass followed by three measured passes. Every complete ranked
page list matched the saved evaluation artifact in each measured pass.

\begin{table}[tbp]
    \centering
    \small
    \caption{Cached-input CAPP-stage cost. Latency statistics summarize
    each question's mean over three measured passes. Input preparation
    is reported separately from per-question computation.}
    \label{tab:capp-cost}
    \begin{tabularx}{\linewidth}{@{}Yr@{}}
        \toprule
        Measurement & Value \\
        \midrule
        Mean latency & 196.2\,ms \\
        Median latency & 184.1\,ms \\
        95th-percentile latency & 338.2\,ms \\
        Serial throughput & 5.10 questions/s \\
        Observed input preparation & 50.4\,s \\
        Benchmark-process peak RSS & 5.824\,GiB \\
        \bottomrule
    \end{tabularx}
\end{table}

Query-dependent feature extraction averaged 193.5\,ms per question,
approximately 98.6\% of the measured stage time, whereas standardization
and logistic scoring required 0.37\,ms. Thus, feature construction,
rather than the 30-weight logistic layer, dominates the measured cost.
Input preparation includes loading the model, questions, rankings,
and page text, tokenizing page text, and constructing source lookup maps.
Its file-cache state was not controlled, so the reported preparation
time is not a cold-start estimate. Peak RSS covers the whole benchmark
process, including cached inputs, temporary allocations, and reference
validation; it is neither model size nor incremental CAPP-only memory.
The per-question timings exclude page-text extraction, dense/sparse
retrieval, base and auxiliary graph generation, training, answer
generation, and prediction serialization. They therefore characterize
the cached-input CAPP stage, not end-to-end pipeline latency.
~~~

Advisor note: the successful replay's auxiliary inputs are the legacy
no-hyperlink, document-hyperlink, and page-hyperlink GPP rankings. These are
precomputed inputs here; their generation costs are excluded. The base ranking
itself is Exact MaxSim GPP no-hyperlink. The manuscript's overall source
description must reflect this distinction before final submission.

If space is tight, omit the runtime table and put the mean, throughput,
observed preparation, and scoped peak RSS in the text. Preserve the measurement
boundaries. Do not derive a controlled percentage overhead by dividing the CPU
stage time by historical GPU reader time.

## 3. Results addition: reader-budget sensitivity

Insert after the runtime subsection and before Pseudo-Label Quality.

~~~latex
\subsection{Reader-Budget Sensitivity}
\label{subsec:reader-budget}

We varied the maximum reader budget over $k\in\{1,2,4,8\}$ using the
same saved GPP and CAPP rankings and the same Qwen2-VL reader
configuration. The CAPP model and $\alpha=0.40$ were held fixed, with
no budget-specific retraining or retuning. We reused the original
$k=4$ answers and generated answers for the other budgets.
All comparisons cover the same 2441 development questions.

\begin{table}[tbp]
    \centering
    \small
    \caption{Reader-budget sensitivity on all 2441 development questions.
    EM and F1 are percentages; $k$ is the maximum number of reader pages.}
    \label{tab:reader-budget}
    \begin{tabular}{@{}crrrr@{}}
        \toprule
        & \multicolumn{2}{c}{GPP} & \multicolumn{2}{c}{CAPP} \\
        \cmidrule(lr){2-3}\cmidrule(lr){4-5}
        $k$ & EM & F1 & EM & F1 \\
        \midrule
        1 & 32.65 & 37.91 & 32.61 & 37.89 \\
        2 & 35.60 & 41.00 & 35.80 & 41.56 \\
        4 & 37.69 & 43.47 & 39.41 & 45.69 \\
        8 & 37.81 & 44.04 & 39.16 & 45.73 \\
        \bottomrule
    \end{tabular}
\end{table}

CAPP does not improve QA at one page, but gives higher observed F1
at budgets of two, four, and eight pages. The largest within-budget
gain is at four pages (+2.22 F1 points). CAPP with four pages also
exceeds GPP with eight pages by 1.65 F1 points. Increasing CAPP's
budget from four to eight changes F1 by only +0.04 points while EM
decreases from 39.41 to 39.16. These results support four pages as a
practical operating point for this fixed configuration, without
establishing a universally optimal budget. Differences are descriptive;
statistical significance has not been established.
~~~

Do not translate “four rather than eight pages” into half the image tokens,
memory, or latency. Page contents, image processing, and generated answer
lengths can differ.

### Historical reader timings: keep separate from controlled CPU results

These saved `time_qa` observations are available for discussion with the
advisor, but are not proposed as a controlled speedup table in the main text.
GPU models and cross-run hardware/cache comparability have not been established.

| Maximum pages | GPP mean seconds | CAPP mean seconds |
| ---: | ---: | ---: |
| 1 | 1.332 | 1.323 |
| 2 | 2.132 | 2.081 |
| 4 | 3.398 | 3.355 |
| 8 | 7.621 | 10.102 |

These timers include cached document-image loading and answer generation,
but exclude retrieval, CAPP, and initial reader-model loading. The larger
CAPP reader time at k=8 is an observation; its cause has not been established.

## 4. Replace the unmatched pseudo-gold comparison

In `tab:pseudo-audit`, remove only the existing `CAPP reader F1` and
`Pseudo-gold reader F1` rows. Keep the coverage rows and their descriptive
paragraph. The old 55.82 result is not wrong, but its contexts contain one
to four pages and its original comparison used different question populations.

Replace the paragraph starting “The evidence coverage audit further supports
the quality of the constructed labels” and ending “reader-visible top-4
context” with the following text and table:

~~~latex
The evidence-coverage audit reports coverage for 2283 of 2441 questions
(93.53\%). Coverage alone does not establish label correctness.
We therefore also evaluated the answer usefulness of the pseudo-labeled
pages using a matched-question, fixed-page-budget injection control.
For each of the 2188 labeled questions, we placed up to four pseudo-gold
pages first, retaining their stored order, and filled any remaining
slots with the highest-ranked distinct GPP pages. Every reader input
contained exactly four distinct pages. The control used Qwen2-VL
(7B Instruct, 16-bit weights); GPP and CAPP answers were reevaluated
on the identical question subset against the original answer annotations.

\begin{table}[tbp]
    \centering
    \small
    \caption{Matched-question evidence-injection diagnostic.
    All methods use the same 2188 pseudo-labeled development questions
    and four reader pages per question. EM and F1 are percentages.}
    \label{tab:pseudo-injection}
    \begin{tabularx}{\linewidth}{@{}Yrr@{}}
        \toprule
        Reader input & EM & F1 \\
        \midrule
        GPP top four & 36.33 & 42.34 \\
        CAPP top four & 38.16 & 44.54 \\
        Pseudo-gold injection + GPP fill & 44.06 & 51.39 \\
        \bottomrule
    \end{tabularx}
\end{table}

The injection condition exceeds CAPP by 6.85 F1 points and GPP by
9.05 points at the same four-page budget. This supports the usefulness
of the pseudo-labeled pages as answer evidence. It is a privileged
diagnostic, not a deployable retrieval method or a strict upper bound:
injected pages can lie outside the original candidate pool, and
placing them first changes page order. The control does not
independently establish page-label correctness or eliminate possible
coupling between the text-based labeler and CAPP's content features.
~~~

Keep the original main-results table unchanged: its GPP 43.47 and CAPP 45.69
F1 values concern all 2,441 questions. The 42.34 and 44.54 values above concern
only the 2,188-question control cohort. Neither pair should replace the other.
Do not describe these diagnostic gains as statistically significant without
a corresponding analysis.

## 5. Minimal Limitations updates

After the current pseudo-label paragraph, add:

~~~latex
The fixed-budget injection control measures the answer usefulness of
pseudo-labeled pages, but those pages are still selected using heuristic
labels. It does not replace independent human page annotation or rule
out labeler--feature coupling.
~~~

Replace only the first two sentences of the next paragraph, which currently
say that another reader budget requires retraining or retuning, with:

~~~latex
Our budget analysis varies the number of reader pages while keeping
the CAPP model and blend weight fixed. It does not evaluate
budget-specific optimization or establish that the same trade-offs
hold for other readers or datasets.
~~~

Add this cost qualification nearby:

~~~latex
The reported CPU measurements isolate CAPP with cached upstream inputs.
End-to-end latency, the cost of generating auxiliary rankings, and
incremental memory relative to GPP alone remain unmeasured in this analysis.
~~~

Leave the remaining baseline-related limitations for the separate stronger-
baseline revision. Do not remove all comparison limitations merely because
retrieval-only baseline numbers have been located.

## Verification and provenance

- Runtime: job 15908770, completed 0:0, node031, Intel Xeon Gold 6342.
  Report on HPC: `output/racs_capp_runtime_15908770/runtime.json`.
  One warmup plus three measured full passes; all 2,441 ranked lists matched
  on each pass. Mean 0.1962253724 s, median 0.1840725678 s, p95 0.3381803210 s;
  serial throughput 5.0961809253 queries/s. Preparation 50.3945374452 s.
  Process peak RSS 5963.796875 MiB = 5.8240203857 GiB.
- Reader budgets: jobs 15904663 (k=1), 15757964 (k=2), 15757965 (k=8),
  merge/evaluation job 15906481. Original k=4 GPP and CAPP answers reused.
  All eight saved prediction files passed the identical-2,441-qid audit.
- Matched subset: earlier CPU audit checked question IDs, reference answers,
  and current pseudo-label identity for all 2,188 questions. GPP and CAPP
  reader contexts had four pages each.
- Injection: job 15910645, completed 0:0, elapsed 02:15:54 on gpu008.
  Final marker confirmed 2,188 answers and four consumed pages each.
  EM 44.0585009141; F1 51.3875685558. Input validation checked page uniqueness
  and the pseudo-gold prefix; output validation checked the exact intended
  page list for every question. Model not retrained.
  HPC directory: `output/racs_adaptive_gold_injection_top4_15910645/`.
  Evaluation: `mmqa_dev_pseudo_gold_plus_gpp_fill_qwen2vl_top4.eval.json`.
- Matched GPP: EM 36.3345521024, F1 42.3368372943.
  Matched CAPP: EM 38.1627056673, F1 44.5415904936.
  Injection-minus-CAPP F1 = 6.8459780622; injection-minus-GPP F1 = 9.0507312614.
- Evidence for HPC results is the user's pasted accounting and audit outputs;
  this guide does not claim the large remote artifacts were copied locally.
- Original manuscript ZIP SHA-256:
  `9254ac3ed9f4303138df1c5f6edd8f1eb779412d2883d293537866a832c7a719`.

## Before sharing the revised manuscript

- Apply these edits manually, compile, and inspect table widths, numbering,
  references, page count, and float placement.
- Preserve the distinction between full CAPP and the source-free ablation.
  Clarify the full model's auxiliary-ranking inputs with the advisor.
- Resolve or explicitly qualify the original alpha-selection and final-
  retraining claims; neither this profiling run nor the budget sweep verifies
  the original train/validation procedure.
- Do not call the injection control human-gold validation, or call the CPU
  benchmark an end-to-end latency/memory comparison.
- No experiment resubmission is required merely to apply these write-up changes.
