# RACS 2026: point-by-point revision response (working, not submitted)

This is a completion-oriented response outline, prepared September 18, 2026.
The text describes the available evidence and proposed manual revisions.
Do not submit it as a claim of completed paper changes until those changes are
actually present in the compiled manuscript. R1.2 still needs a validated
online-runtime result. No original manuscript/Overleaf files were changed.

## Reviewer 1

### 1. Feature-selection rationale and relation to prior work

The revision explains the four feature families and separates established
ranking/fusion ideas from this study's empirical feature design. The implemented
30 features comprise rank (5), source (4), structure (6), and content (15).
The expanded feature-family ablations show which choices matter in this setting;
they are not presented as an exhaustive feature search or causal attribution.
The method description also distinguishes the no-hyperlink main GPP base from
the full model's three auxiliary rankings, including hyperlink-enabled inputs.

Integration: packet steps covering the feature table, rationale and setup.
Closure: verify those replacements and their citations in the revised PDF.

### 2. Runtime, throughput, memory and feature overhead

Existing validated evidence includes 196.2 ms/query for cached-input CPU CAPP,
and a paired graph-to-answer comparison including auxiliary-graph generation:
3.942 s for GPP versus 4.703 s for CAPP (+0.761 s; +19.3%), on 128 questions with
four repeats on one A100 80GB. Feature extraction dominates the standalone CPU
stage. These are scoped results, **not full online query-to-answer timing**.

**Open:** execute and validate the new online benchmark, which also includes
query encoders, FAISS, SPLADE and both dense scoring routes. Replace this open
paragraph with its audited latency/serial-throughput/memory results and exact
hardware/cache/preparation boundaries. Do not use Slurm elapsed time or sum
incomparable historical timers. Whole-worker memory is not isolated scorer memory.

### 3. Competitive learned/higher-capacity reranking

The expanded comparison includes BGE-reranker-base under the common 1,000-page
GPP candidate pool and frozen four-page reader. On the same 2,441 development
questions it obtains 37.85 EM / 43.89 F1; full CAPP obtains 39.41 / 45.69.
Saved monoT5 and LambdaMART retrieval results provide additional context, without
inventing reader scores for those methods. Model/feature/budget differences and
unverified historical blend-selection provenance are stated.

Closure: confirm the baseline table, checkpoint citation and matched-budget text.
One matched competitive reranker satisfies the review's “one or more” request;
these results do not establish significance or superiority to every configuration.

### 4. Lightweight scoring benefits and trade-offs

The revision distinguishes the small logistic scoring layer from the cost of
constructing its features and auxiliary rankings. It uses measured stage costs
and the expanded ablations to discuss accuracy/cost trade-offs, without equating
few parameters with negligible pipeline overhead or claiming measured speed
superiority over untimed neural baselines.

Closure: finish this discussion using R1.2's online result, and verify the
Discussion/Limitations edits remove unsupported blanket efficiency claims.

### 5. Systematic family ablation

The table includes removal of each family at fixed alpha=0.40, plus structure-only
and structure+content combinations. Structure is particularly important; removing
it reduces QA F1 to 43.46. Structure+content achieves 45.14 F1 versus 45.69 for all
features, while its page@4 is higher (77.79% versus 77.15%). The interpretation
acknowledges the retrieval/answering trade-off rather than saying every additional
family improves every metric. Historical auxiliary filenames for source-bearing
ablations are not inferred merely from matching main input paths.

Closure: verify family names, counts, table rows, fixed blend and qualified prose.

### 6. Reader-budget sensitivity

The new analysis compares GPP/CAPP at 1, 2, 4 and 8 pages on the same 2,441
questions with the same reader. CAPP F1 is 37.89, 41.56, 45.69 and 45.73,
respectively; corresponding GPP F1 is 37.91, 41.00, 43.47 and 44.04.
The gains are not universal at every budget; the CAPP curve is nearly flat from
four to eight pages. Historical cross-job timings are not used as a controlled
speed comparison.

Closure: confirm budget table and discussion against the evaluated artifacts.

## Reviewer 2

### Labeler/content-feature overlap and gold-page injection

The fixed-budget control inserts available pseudo-gold pages and fills from GPP
to exactly four unique pages on the same 2,188 page-labeled questions. The reader
obtains 44.06 EM / 51.39 F1, versus 38.16 / 44.54 for CAPP and 36.33 / 42.34 for
GPP on that matched subset. This supports the answer usefulness of the selected
evidence. It does not remove pseudo-label coupling, provide independent human
gold, or establish a guaranteed upper bound. The revision states those limits.

Closure: verify matched-subset denominators, exactly-four-page protocol and
the explicit “pseudo-gold” qualifier in the table, caption and discussion.

### Content-aware name and structure+content QA comparison

The revision explains that “content-aware” describes inclusion of content
signals, not their dominance. The previously missing structure+content reader
result is now available (39.04 EM / 45.14 F1), alongside the full model
(39.41 / 45.69). The small observed QA advantage does not establish statistical
superiority; structure+content has stronger page@4, as noted above.

Closure: include the row and align the title/method explanation/conclusion with
the demonstrated role of structure and content.

## Final release gate

1. Online runtime passes replay checks and independent aggregation audit.
2. Every proposed revision is actually integrated into the manually edited paper.
3. All alpha-selection claims are either supported by recovered evidence or
   explicitly qualified; the author's uncertainty is not replaced with a guess.
4. Compile and visually check page limit, figures, tables, citations, labels and
   metric denominators. Confirm paper and relevant thesis descriptions agree.
5. Replace working integration notes with actual section/table references,
   obtain coauthor approval, then prepare the final submission materials.
