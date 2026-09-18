# RACS manual revisions: feature rationale, trade-offs, and limitations

Prepared 2026-09-17 while BGE reader job 15911612 is pending/running on HPC.
Its last user-reported state was pending for priority; no live HPC status is
available locally. This guide contains proposed manual edits, not changes to
Overleaf, the manuscript ZIP, predictions, or the running job.

## Scope and insertion order

1. Apply the runtime/budget/control guide and the expanded ablation guide.
2. In `subsec:capp`, replace `tab:capp-features` and its introductory paragraph
   using sections 1 and 2 below. Keep the existing scoring equations.
3. In that subsection, replace the short paragraph beginning "Because the
   feature vector has 30 dimensions" with section 3.
4. Replace only the final Discussion paragraph using section 4.
5. Section 5 replaces the entire Limitations section body. It consolidates the
   earlier guide's limitations edits: do not paste both versions.

The snippets reuse the manuscript's packages, column types, citation keys, and
labels. Sections referencing `subsec:capp-cost`, `subsec:reader-budget`, or the
expanded `tab:feature-ablation` assume those earlier edits have been inserted.
Manual compilation and page-limit/layout review are still required.

## 1. Correct the feature examples

Replace the existing `tab:capp-features` table, preserving its label:

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

Why these corrections matter: the saved model has aggregate source statistics,
not separate dense/sparse one-hot indicators. Its six structure features do not
directly compare neighboring-page content. Adjacency belongs to upstream GPP.
`doc_page_count_log` counts a document's pages in the candidate list, not the
document's total physical length. Content matches are question-derived, not
matches against the gold answer/evidence strings at inference.

## 2. Explain the rationale and scope of the design

Replace the paragraph beginning "For each question-page pair" and ending
"question-specific evidence cues" with these paragraphs, before "The scorer
is a single logistic layer."

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

The citations provide conceptual context, not proof of the authors' historical
feature-selection process. Do not claim that the exact CAPP feature set was
copied from either paper or that unreported alternatives were tested.

### Full-model auxiliary inputs: necessary protocol clarification

Add the following to Experimental Design, immediately after the Baselines
paragraph and before Training. It describes the verified full-model evaluation
configuration, not a recovered original training command:

~~~latex
\textbf{Auxiliary ranking inputs.}
For full CAPP on GPP, the base candidate ranking is the Exact MaxSim GPP
variant without hyperlink edges. The source features additionally
summarize three auxiliary GPP rankings: a no-hyperlink variant, a
document-hyperlink variant, and a page-hyperlink variant. Thus the
no-hyperlink setting describes the base ranking, not the absence of
hyperlink-derived information throughout full CAPP. These auxiliary
rankings supply aggregate support features; they do not enlarge the
base candidate pool. The no-source ablation removes all four such
features and is a separately trained model.
~~~

Discuss this wording with the advisor: replay verified these inputs against
all 2,441 saved rankings, and the auxiliary summaries confirm active hyperlink
edges. This does not recover the literal historical training-source command
or measure how much the hyperlink-derived information helps. The manuscript
must not silently describe the original full model as the no-source variant.

## 3. State the computational and modeling trade-off precisely

Replace the paragraph beginning "Because the feature vector has 30 dimensions"
with:

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

## 4. Clarify "content-aware" and the full-model choice

In Discussion, replace its final paragraph beginning "The feature ablation
also explains why the final model keeps all feature groups" with:

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

This avoids saying "structure + content is best retrieval" without naming
the cutoff. It also avoids interpreting inclusion of content features as proof
that they explain most of the improvement. No title change is necessary merely
to make this qualification; that choice remains with the authors.

## 5. Consolidated Limitations body

Keep `\section{Limitations}` and `\label{sec:limitations}`. Replace only the
section body with the following. This version makes no claim that the pending
BGE QA job has completed and remains accurate if only the established results
are inserted. Finalize the baseline comparison separately after job validation.

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

The CPU benchmark measures CAPP with cached upstream inputs. It does not
measure end-to-end latency, auxiliary-ranking construction, or incremental
memory relative to GPP alone. The whole-process memory observation must
not be interpreted as the scorer's additional memory requirement.
Broader accuracy--cost comparisons require matched hardware and timing
boundaries across systems. Our reader-budget analysis varies $k$ while
holding the trained scorer, blend weight, and rankings fixed; it does
not test budget-specific optimization or establish the same behavior
for other readers.

The comparisons cover selected configurations on the M3DocVQA development
set. Small differences among ablations are descriptive rather than
statistically established. Additional reranker configurations and
cross-dataset evaluation remain necessary to assess generality across
layout distributions, OCR quality, document lengths, and evidence density.
~~~

## Evidence and validation notes

Assessment: **share with caveats as an advisor-review draft**, not an assertion
that every reviewer request has been fully closed. No new performance numbers
were introduced in these snippets.

- Source implementation: `scripts/train_content_aware_pseudo_page_reranker.py`,
  `BASE_FEATURE_NAMES`, the four family lists, `doc_rank_maps`,
  `question_profile`, `content_features`, `feature_vector`, and
  `standardize_train`/`standardize_eval`. Current code also supports optional
  visual features, but the audited saved full model uses only the 30 base
  features; do not infer a 40-feature CAPP model from current defaults.
- Full-model and ablation facts: user-provided HPC replay, saved-model audits,
  and summaries recorded in `notes/racs_camera_ready_revision_plan.md` and
  `notes/racs_manual_revisions_baselines_ablation.md`. Raw remote artifacts
  were not downloaded for this writing pass.
- The established feature-vector ranking formulation is documented in
  [Burges's learning-to-rank overview, Section 2](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf).
  It is not a source for CAPP's exact features or weighted-BCE training objective.
- Combining multiple ranked lists is documented in the original
  [reciprocal-rank-fusion paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
  CAPP's aggregate statistics and learned blend are distinct from its RRF rule.
- Both citation keys already exist in the September 15 `references.bib`:
  `Burges2010LambdaMART` and `Cormack2009RRF`. No bibliography edit is needed.
- This is a source-text draft. LaTeX compilation, float placement, and final
  page count must be checked after manual integration.
- Local checks passed: feature counts and membership (5/4/6/15) against the
  implementation; balanced braces/environments in all six LaTeX snippets;
  existing bibliography keys and reference targets; no whitespace errors.
  The September 15 ZIP retains SHA-256
  `9254ac3ed9f4303138df1c5f6edd8f1eb779412d2883d293537866a832c7a719`.

## Remaining author confirmation before final submission

The manuscript currently states that alpha was selected on held-out training
data and the final model was retrained afterward. The saved final artifact
fixes alpha at 0.40; the author's recollection of trying several values does
not establish the selection split or that retraining sequence. Do not keep
the precise historical claim without confirmation, and do not invent a
replacement protocol. Ask which data/scores were used in those trials and
whether the model was refit after selecting alpha. If it was development-set
selection, disclose it and discuss the evaluation implications with the advisor.

For the advisor-review draft, the verified statement is simply that the reported
full model uses fixed alpha 0.40. Record the unresolved selection procedure in
the accompanying checklist; do not present omission as a reproducibility fix.
