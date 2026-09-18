# RACS: complete manual Overleaf revision packet

Prepared September 18, 2026. This is a single, ordered copy-and-paste packet for
the September 15 manuscript, not a replacement TeX project. The original ZIP
and Overleaf have not been edited. Copy each fragment only at its specified
anchor; do not paste this entire Markdown file into main.tex. Do not also
insert duplicate fragments from the three earlier guides.

## What this packet completes—and what it does not

- It consolidates the completed reader-budget, injection, BGE QA, ablation,
  feature-rationale, scoped CPU-runtime and matched graph-to-answer revisions.
- It gives explicit configuration wording, including the different main and
  auxiliary dense inputs.
- It removes an unsupported certainty about historical alpha selection in the
  proposed advisor draft. The author answered “I am not sure” about the split.
- It does not complete the full runtime experiment. Job 15911730 passed the
  16-question/four-graph correctness replay. Job 15911732 completed controlled
  graph-to-answer timing, but its cached-retrieval scope is not online end-to-end.
- Manual insertion, a compiled PDF, visual/page-limit checking and advisor
  approval still remain. No retraining is required to insert the existing results.

## Verified configuration to preserve

| Component | Recorded configuration / evidence |
| --- | --- |
| Main dense ranking | Exact page-local MaxSim; embedding name colpali-v1.2_m3-docvqa_dev |
| Sparse input | Summary names naver/splade-cocondenser-ensembledistil, 32 query terms, top 1000—not a verified SPLADE-v3 run |
| Main GPP | No hyperlink edges; dense/sparse weights 1.25/0.75; restart 0.15; 30 PPR iterations; score-based final selection |
| Auxiliary rankings | Three legacy GPP variants: no hyperlink, document hyperlink, page hyperlink; approximate MaxSim with a 224-token page budget and document-diverse MMR |
| Full CAPP | Saved 30-feature logistic scorer; 1000 candidates; fixed blend alpha 0.40; four aggregate source features |
| Reader | Qwen2-VL-7B-Instruct, 16-bit weights; main budget four pages |
| Evaluation populations | 2441 questions for main/budget/BGE QA; 2188 labeled questions for page retrieval and matched injection |
| Alpha history | Matching-input 0.35/0.40/0.45 artifacts exist; selection split not verified |
| Existing runtime | Cached-input CAPP only, 196.2 ms/query on one Xeon Gold 6342 CPU thread—not end-to-end |
| Matched graph-to-answer runtime | 128 qids, four repeats, A100 80GB PCIe; GPP 3.942 s vs CAPP 4.703 s; cached upstream retrieval, not full online timing |

The saved-model replay reproduces full CAPP evaluation; it does not recover the
literal original training command. Auxiliary settings are not interchangeable
with main-base settings. A 224-token page budget is not a 224-page candidate pool.

## How to apply

Follow steps 1–15 in order. Keep existing equations, main-result tables and
figures unless a step explicitly says to replace them. Preserve exactly one
instance of each table/subsection label. Proposed alpha wording in steps 4, 6
and 14 is transparent **advisor-draft wording**: the advisor must review it
before final submission. Do not simply remove the caveat while the split is
unknown.


## 1. Replace the feature table

In Content-Aware Page Promotion (subsec:capp), replace the table labeled tab:capp-features. Keep the label and the surrounding scoring equations.

~~~latex
\begin{table}[tbp]
    \centering
    \small
    \caption{The 30 CAPP features, grouped by information source.
    Counts appear in parentheses; examples summarize the implemented features.}
    \label{tab:capp-features}
    \begin{tabularx}{\linewidth}{@{}L{0.22\linewidth}L{0.36\linewidth}Y@{}}
        \toprule
        Group & Examples & Purpose \\
        \midrule
        Rank (5) & Base-rank transforms, normalized score, gap to top score & Retain base-ranking confidence \\
        Source (4) & Source count, best/mean reciprocal rank, best normalized score & Summarize auxiliary ranking support \\
        Structure (6) & Document rank, within-document rank, candidate-page count, page position & Encode document/page localization \\
        Content (15) & Question-token, number, phrase and n-gram overlap; page length & Measure lexical compatibility \\
        \bottomrule
    \end{tabularx}
\end{table}
~~~

## 2. Replace the feature-vector introduction

Replace the paragraph beginning “For each question-page pair” and ending “question-specific evidence cues.” Stop before “The scorer is a single logistic layer.” This clarifies that the scoring vector is standardized.

~~~latex
For each question-page pair $(q,p)$, CAPP constructs 30 explicit features
and standardizes them using the saved training means and standard
deviations. We denote the standardized vector by
$x(q,p) \in \mathbb{R}^{30}$. Table~\ref{tab:capp-features} groups the
features according to four information sources: confidence in the base
ranking, support across auxiliary rankings, document/page localization,
and compatibility between the question and extracted page text.

The design follows the general feature-based learning-to-rank formulation
\cite{Burges2010LambdaMART} and is related to the use of multiple ranked
lists in rank fusion \cite{Cormack2009RRF}. These are established ideas,
not contributions claimed here. CAPP does not use a pairwise LambdaRank
objective or apply reciprocal-rank fusion as its final scoring rule.
Its study-specific design is the selection and grouping of these 30
features for pseudo-page-supervised promotion within a fixed candidate
pool. The individual transforms are not claimed as novel, and the
experiments do not establish an exhaustive or optimal feature search.

Rank features retain information that is already available from retrieval.
Source features summarize whether a page is supported by multiple
auxiliary lists and how highly it is ranked there, without assuming that
those lists provide independent evidence. Structure features describe
page position and a document's representation in the candidate ranking.
Content features provide inexpensive question-dependent lexical signals,
including token, number, phrase and n-gram matches, without invoking a
new neural encoder. Their computation uses the question and page text,
not the reference answer or evidence-metadata strings.
Section~\ref{subsec:features} evaluates single families, structure +
content, and removal of each family from the full model; these comparisons
test the selected design rather than all possible feature combinations.
~~~

## 3. Replace the lightweight-scorer claim

Replace the paragraph beginning “Because the feature vector has 30 dimensions” through “additional VLM call or neural encoder pass.” This describes capacity without claiming that upstream computation is free.

~~~latex
The learned scorer contains 30 weights and one bias. With its inputs
available, scoring consists of feature standardization, one dot product
and a sigmoid per candidate, followed by blending and sorting. The CAPP
scoring stage requires no additional VLM call or neural encoder pass,
but feature construction and provision of upstream rankings still incur
cost. Section~\ref{subsec:capp-cost} measures these stage-level costs
separately rather than treating parameter count as a latency estimate.
The linear logit permits inspection of feature contributions, but the
model cannot learn unrestricted interactions beyond those encoded by
the features. Correlated features also prevent interpreting individual
coefficients as causal importance scores. The scorer itself performs no
direct reasoning over page pixels; visual processing occurs in the
upstream retrieval system and downstream reader.
~~~

## 4. Replace the alpha/budget paragraph in Method

After the blend-score explanation, replace the text beginning “The main model uses fixed” through “retuned for that target k.” Use the fixed-alpha statement below while the original selection split remains unverified. Step 14 adds the associated limitation; do not silently drop it.

~~~latex
The main model uses a fixed blend weight $\alpha=0.40$ in the reported
evaluations. Section~\ref{subsec:reader-budget} evaluates different
reader budgets using the same trained scorer and saved rankings,
without budget-specific retraining or retuning.
~~~

## 5. Clarify sparse and auxiliary inputs

In Experimental Design, add the first sentence below to the Compared methods paragraph immediately after its SPLADE sentence. Then insert the Auxiliary ranking inputs paragraph after Compared methods and before Training. The checkpoint name is taken from saved metadata, not a fresh checkpoint-byte audit.

~~~latex
The saved sparse-retrieval configuration identifies
\texttt{naver/splade-cocondenser-ensembledistil} as the SPLADE
checkpoint, with 32 query terms and a top-1000 page ranking.
~~~

~~~latex
\textbf{Auxiliary ranking inputs.}
For full CAPP on GPP, the base candidate ranking is the Exact MaxSim GPP
variant without hyperlink edges. The source features additionally
summarize three auxiliary GPP rankings: a no-hyperlink variant, a
document-hyperlink variant, and a page-hyperlink variant. These auxiliary
rankings use a legacy approximate MaxSim dense input with a 224-token
page budget (\texttt{query\_mean} token scoring and
\texttt{global\_topk} selection), followed by document-diverse MMR
selection; the main base instead uses Exact MaxSim and score-based
selection. Thus the
no-hyperlink setting describes the base ranking, not the absence of
hyperlink-derived information throughout full CAPP. These auxiliary
rankings supply aggregate support features; they do not enlarge the
base candidate pool. The no-source ablation removes all four such
features and is a separately trained model.
~~~

## 6. Make Training consistent with the available evidence

In the Training paragraph, replace the sentences beginning “We use 50 negatives per labeled question” through “retrain the final scorer on the labeled training questions.” Keep the preceding weighted-BCE description and subsequent pointwise-logistic/variant description. The saved settings specify rank-stratified sampling and 10 negatives per band; the implementation does not guarantee 50 negatives for each question. The old positive-weight cap of 20 is documented in the manuscript, but not recovered from the final model fields inspected here; obtain the original configuration before making a stronger audited claim about that cap. Do not assert a post-selection refit sequence without confirmation.

~~~latex
Negative sampling is rank-stratified, selecting up to 10 eligible
negative pages from each of the rank bands 1--4, 5--20, 21--100,
101--500, and 501--1000. The reported full-model evaluations use
a fixed blend weight $\alpha=0.40$. The historical data split used
to select this value could not be verified from the recovered
experiment records; we report this provenance limitation explicitly
in Section~\ref{sec:limitations}.
~~~

The first band contains only four positions, and pseudo-positive pages are excluded. Thus “50 negatives for every labeled question” is not an accurate description of the inspected implementation. The saved maximum-per-question cap is 64, but does not turn these five bands into 64 sampled negatives. This clarification is based on the saved sampling settings and current implementation, not a newly recovered training invocation.

## 7. Update the Reader paragraph and setup table

Replace the paragraph headed Reader with the first fragment. In tab:setup replace only the Reader budget and QA evaluation rows with the second fragment; replace its caption with the third. In Implementation protocol replace the sentence beginning “For downstream QA, we pass the top four selected page images” with the fourth.

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

~~~latex
        Reader budget & 4 pages in main runs; 1, 2, 4, 8 in sensitivity analysis \\
        QA evaluation & 2441 questions for main/budget runs; 2188 for injection control \\
~~~

~~~latex
    \caption{M3DocVQA experimental setup. Page-level retrieval uses
    pseudo-labeled questions; main and budget-sensitivity QA use all
    development questions. The injection control uses the labeled subset.}
~~~

~~~latex
For downstream QA, we pass up to $k$ selected page images to the VLM reader
($k=4$ in the main experiments) and evaluate generated answers against
the original M3DocVQA answer annotations.
~~~

## 8. Add the stronger-baseline comparison

After the Main Retrieval and QA Results discussion—including its existing question-type breakdown—and before Pseudo-Label Quality, insert this new subsection. This replaces the earlier retrieval-only draft; do not insert both. Step 15 provides its new BGE bibliography entry.

~~~latex
\subsection{Additional Reranking Baselines}
\label{subsec:additional-rerankers}

We additionally compare retained LambdaMART, BGE and monoT5 configurations
using the same recorded top-1000 GPP base-ranking input.
LambdaMART \cite{Burges2010LambdaMART} uses LightGBM's LambdaRank objective,
40 features and blend weight 0.45. The neural baselines use
\texttt{bge-reranker-base} \cite{BAAI2023BGERerankerBase} and
\texttt{monot5-base-msmarco-10k} \cite{Nogueira2020MonoT5}, reranking all
1000 candidates with blend weights 0.20 and 1.00, respectively.
Both use a maximum sequence length of 512; the page-text character caps
are 6000 for BGE and 4000 for monoT5.
For downstream QA, the retained BGE ranking is passed to the same frozen
16-bit Qwen2-VL-7B-Instruct reader with four pages per question. The run
covers the same 2,441 development questions and verifies that BGE reranks
the same 1000 distinct candidate pages as GPP for every question.

\begin{table}[tbp]
    \centering
    \small
    \caption{Additional reranking comparison. Retrieval uses 2,188
    pseudo-labeled development questions; QA uses all 2,441 development
    questions with four pages each. Dashes denote QA not reported here.
    These are system comparisons, not matched-total-compute ablations.}
    \label{tab:additional-rerankers}
    \begin{tabular}{@{}lrrr@{}}
        \toprule
        Method & page@4 & EM & F1 \\
        \midrule
        GPP & 0.6376 & 37.69 & 43.47 \\
        LambdaMART & 0.6408 & -- & -- \\
        BGE reranker base & 0.6705 & 37.85 & 43.89 \\
        monoT5 base & 0.6609 & -- & -- \\
        CAPP on GPP & \best{0.7715} & \best{39.41} & \best{45.69} \\
        \bottomrule
    \end{tabular}
\end{table}

Table~\ref{tab:additional-rerankers} reports higher page@4 for CAPP than
these retained baseline configurations. With the common four-page reader
budget, BGE achieves 37.85 EM and 43.89 F1, compared with 39.41 EM and
45.69 F1 for CAPP. These are descriptive comparisons, not statistical
significance claims or evidence of superiority over all configurations
of these model families.
LambdaMART and CAPP have different feature sets, and full CAPP uses
auxiliary ranking-support features. The neural runs use extracted page
text and report 744 empty-text candidate encounters each, not missing
questions. The comparison therefore does not isolate model capacity,
and downstream QA for LambdaMART and monoT5 is not reported here.
~~~

## 9. Add the scoped runtime analysis

Immediately after step 8, insert this subsection. It now includes two completed experiments: full-cohort cached-input CAPP-stage timing and the 128-question matched graph-to-answer comparison. Neither is full online query-to-answer timing. Keep their different cohorts, hardware, boundaries and memory labels. Median/p95 describe per-question means over three repeats for the first experiment and four for the second, not a serving-system tail-latency guarantee.

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

We additionally compared graph-to-answer execution on 128 development
questions selected by a fixed, outcome-independent hash order, using
one NVIDIA A100 80GB PCIe GPU and one CPU math thread. Each method used
the same questions, four reader pages, and the frozen 16-bit
Qwen2-VL-7B-Instruct reader. Four disjoint questions provided warmup
in each fresh worker process; four measured repetitions per method
alternated execution order. CUDA synchronization bounded each query
timer. Both methods recomputed their main GPP graph; CAPP additionally
recomputed all three auxiliary graphs and its source features and
reranking. PDF rendering, image selection, and reader processing and
generation were timed. There was no rendered-document cache across
questions; filesystem cache state was uncontrolled. Complete graph
orders and scores, and complete CAPP orders, matched the saved
references under the validation checks. Validation and initialization
were outside the query timer.

\begin{table}[tbp]
    \centering
    \small
    \caption{Matched graph-to-answer cost with cached upstream
    retrieval on 128 questions. Latency statistics use each question's
    mean over four repetitions. CPU RSS ranges span fresh-worker
    high-water marks, including initialization, inputs and validation
    references; they are not incremental scorer memory.}
    \label{tab:graph-to-answer-cost}
    \begin{tabularx}{\linewidth}{@{}Yrr@{}}
        \toprule
        Measurement & GPP & CAPP \\
        \midrule
        Mean latency (s) & 3.942 & 4.703 \\
        Median latency (s) & 3.565 & 4.287 \\
        95th-percentile latency (s) & 5.941 & 7.916 \\
        Serial throughput (questions/s) & 0.254 & 0.213 \\
        CPU peak RSS range (GiB) & 4.99--5.43 & 6.45--6.80 \\
        GPU peak allocated (GiB) & 19.37 & 19.37 \\
        GPU peak reserved (GiB) & 24.72 & 24.72 \\
        \bottomrule
    \end{tabularx}
\end{table}

CAPP added 0.761\,s per question, or 19.3\%, within this measured
boundary. Ranking averaged 0.098\,s for GPP and 0.605\,s for CAPP;
image preparation averaged 1.451\,s and 1.717\,s, respectively.
Reader processing and generation averaged 2.393\,s and 2.381\,s.
The additional time therefore is not attributable only to the
logistic scorer. No answer varied across repetitions within either
method. The GPU peak values were equal in this workload, while CPU
high-water marks were higher for CAPP. These observations are not
full-corpus serving-memory requirements or evidence that the methods
have equal GPU requirements under other workloads.

This comparison starts from cached dense and sparse rankings. It
excludes query encoding/search, Exact MaxSim and legacy approximate
MaxSim scoring, and offline indexing. Thus, the measured difference
includes auxiliary graph generation but not the upstream retrieval
needed to supply those graphs; it is not full online query-to-answer
overhead. The timing subset does not replace the full-development-set
QA evaluation.
~~~

## 10. Add reader-budget sensitivity

Immediately after step 9 and before the existing Pseudo-Label Quality heading, insert this subsection. Do not add the historical heterogeneous-GPU time table as a controlled speed comparison.

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

## 11. Replace the unmatched pseudo-gold comparison

In tab:pseudo-audit remove only the existing CAPP reader F1 and Pseudo-gold reader F1 rows. Keep its three coverage rows and descriptive paragraph. Replace the later paragraph starting “The evidence coverage audit further supports the quality of the constructed labels” and ending “reader-visible top-4 context” with the following. The injection table compares the same 2188 questions, not the all-question main scores.

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

## 12. Replace and expand Feature Ablation

Keep the Feature Ablation heading and subsec:features label. Replace its existing table and following three paragraphs up to—but not including—Promotion Audit. The wider table uses table*; check placement and width after compilation.

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

## 13. Replace the final Discussion paragraph

Replace the paragraph beginning “The feature ablation also explains why the final model keeps all feature groups.” Stop before Limitations.

~~~latex
The term content-aware denotes the inclusion of question--page textual
compatibility, not a claim that content is the dominant source of the
observed gain. Structure is the strongest individual feature family
under the reported page-level metrics. Structure + content yields the
highest page@4, while the no-rank variant yields the highest page@10.
The full model has the highest observed QA F1 among the evaluated
variants, including the now-explicit structure + content comparison
in Table~\ref{tab:feature-ablation}. We therefore distinguish page-hit
quality from answer quality and retain the full model for the reported
QA configuration, without claiming that every feature family improves
every metric or that the small QA differences are statistically significant.
~~~

## 14. Replace the entire Limitations body

Keep the Limitations heading and sec:limitations label. Replace its whole body, stopping before Conclusion. This supersedes all earlier piecemeal limitations insertions. The final paragraph is necessary while alpha selection remains unverified; if authentic historical evidence later establishes the procedure, revise that paragraph and steps 4/6 consistently.

~~~latex
Pseudo-page labels are constructed from evidence metadata rather than
complete human page annotations. They can miss visual evidence,
OCR-altered strings, and evidence absent from the metadata. Moreover,
the labeler and CAPP's content features both depend on extracted page
text. The matched four-page injection control tests the answer usefulness
of pseudo-labeled pages, but it neither provides independent human-gold
validation nor rules out labeler--feature coupling. Datasets without
appropriate evidence metadata require another supervision or annotation
procedure.

The logistic scorer uses a fixed set of text and ranking features. It
does not directly analyze page pixels or learn arbitrary interactions,
and structural regularities that help on M3DocVQA may not transfer to
other collections. The full configuration also depends on auxiliary
rankings, including hyperlink-enabled variants; it is not a purely
content-only or entirely hyperlink-free system. The feature ablations
characterize the selected families, not an exhaustive search over
possible representations.

The CPU-only benchmark measures CAPP with cached upstream inputs. The
additional matched graph-to-answer experiment includes auxiliary graph
construction and reader execution, but still excludes dense and sparse
query encoding/search and both dense scoring routes. Neither experiment
establishes full online query-to-answer latency. Fresh-worker CPU RSS
and GPU allocated/reserved peaks characterize the declared benchmark,
not isolated scorer memory or full-corpus deployment requirements.
The timing subset, image-cache policy and uncontrolled filesystem cache
also limit generalization. Broader accuracy--cost comparisons require
matched hardware and timing boundaries across systems.
Our reader-budget analysis varies $k$ while
holding the trained scorer, blend weight, and rankings fixed; it does
not test budget-specific optimization or establish the same behavior
for other readers.

The comparisons cover selected configurations on the M3DocVQA development
set. Small differences among ablations are descriptive rather than
statistically established. Additional reranker configurations and
cross-dataset evaluation remain necessary to assess generality across
layout distributions, OCR quality, document lengths, and evidence density.

The final evaluations use a fixed blend weight, but the historical
selection split for CAPP's $\alpha=0.40$ could not be verified from the
recovered records. The selection split for the retained BGE blend
weight is also unverified. This limits how strongly the reported
development-set comparisons can be interpreted as independent of
hyperparameter selection.
~~~

## 15. Add the BGE bibliography entry

Add this entry once to references.bib. Keep the existing Burges2010LambdaMART, Nogueira2020MonoT5, Cormack2009RRF and Wang2024Qwen2VL entries. The BGE embedding-model reference is not a substitute for the reranker model card.

~~~bibtex
@misc{BAAI2023BGERerankerBase,
  author = {{Beijing Academy of Artificial Intelligence}},
  title = {{BGE Reranker Base}: Model Card},
  year = {2023},
  url = {https://huggingface.co/BAAI/bge-reranker-base},
  note = {Accessed September 17, 2026}
}
~~~

## Final checks before sharing with the advisor

- Confirm that the original main GPP/CAPP QA values remain 37.69/43.47 and
  39.41/45.69 EM/F1, respectively. Do not substitute matched-subset values.
- Confirm BGE QA is 37.85/43.89 on all 2441 questions, four pages each.
- The matched injection comparison uses GPP 36.33/42.34, CAPP 38.16/44.54,
  and injection 44.06/51.39 on 2188 questions.
- Preserve pseudo-label, source-provenance and tuning caveats; do not claim
  significant differences or a measured full-pipeline speedup.
- Check updated table references, citation resolution, duplicate labels,
  table widths, float placement and the applicable camera-ready page limit.
- Recheck Abstract/Introduction/Conclusion for claims contradicted by these
  edits, especially entirely hyperlink-free or all-branches Exact MaxSim.
  Do not insert new unsupported global claims.
- Compile in Overleaf and inspect the resulting PDF visually. This packet has
  source-structure checks only; it is not a compiled revised manuscript.
- Send the advisor the compiled draft with two clearly stated remaining
  issues: controlled end-to-end runtime is unfinished, and historical
  alpha-selection provenance is unresolved. Do not present all reviewer
  requests as fully closed.

## What still requires action outside this packet

1. The graph replay precheck is complete: job 15911730 matched all four branches
   on all 16 sampled questions. It is not a full-cohort or timing result.
2. The graph-to-answer benchmark completed in job 15911732 and its supplied
   summaries are incorporated in step 9. Preserve the raw reports. Online
   retrieval/scoring must still be integrated before claiming a complete
   query-to-answer comparison; this partial result does not close that gap.
3. Have the advisor review the recovered full-model inputs and the explicit
   tuning limitation. The author has already said the selection split is not
   remembered; repeating the question is not a substitute for evidence.
4. Apply these edits manually and return the revised ZIP or PDF for checking.
   Direct manuscript editing remains outside the earlier manual-only request.

## Source snapshots

The original fragments came from the snapshots below. Step 9 and the runtime
limitations now additionally incorporate job 15911732, documented in
`notes/racs_graph_reader_runtime_protocol.md`, from the author's pasted output
`/Users/hoseinerfan/.codex/attachments/7b8bc269-b11b-4862-baac-93a20ac3524d/pasted-text.txt`.
The older guides do not contain this newly completed experiment; use this
consolidated packet for the current runtime insertion. Source hashes identify
the original draft snapshots, not the later packet amendments.


- notes/racs_manual_revisions_runtime_budget_control.md — SHA-256 e5c54767fafe918b7c67bb73fd2afa75a968c51e06de0ebf97467069739d331c

- notes/racs_manual_revisions_baselines_ablation.md — SHA-256 6a14b2f9c6abe2ed5e378b4c5e9a2feff3c3906314eb0f589813b027664d6603

- notes/racs_manual_revisions_feature_rationale.md — SHA-256 39d9e438121055a6dc4e872cb6739188144059f8bf5bc43365a328e58ac30f0c
