#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAGE_REFERENCE_TOKEN_RE = r"(?:pages?|pg\.?|p\.|slides?)"
PAGE_REFERENCE_RANGE_RE = (
    rf"\b{PAGE_REFERENCE_TOKEN_RE}\s*(?:no\.?|number|#)?\s*"
    r"(\d{1,4})\s*(?:-|to|and)\s*(\d{1,4})\b"
)
PAGE_REFERENCE_SINGLE_RE = (
    rf"\b{PAGE_REFERENCE_TOKEN_RE}\s*(?:no\.?|number|#)?\s*(\d{{1,4}})\b"
)
DOCUMENT_COVER_TARGET_RE = (
    r"(?:document|report|paper|article|brochure|guidebook|manual|book|chapter|"
    r"newspaper|presentation|slides?|deck)"
)
FIRST_COVER_REFERENCE_RE = (
    r"\b(?:first|opening)\s+page\b"
    r"|\bfront\s+(?:page|cover)\b"
    r"|\btitle\s+page\b"
    r"|\bcover\s+page\b"
    rf"|\b{DOCUMENT_COVER_TARGET_RE}(?:'s)?\s+cover\b"
    rf"|\bcover\s+of\s+(?:the\s+|this\s+|that\s+|each\s+|a\s+|an\s+)?{DOCUMENT_COVER_TARGET_RE}\b"
)


@dataclass
class PageRecord:
    doc_id: str
    page_idx: int
    dense_rank: int | None = None
    sparse_rank: int | None = None
    expansion_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    expansion_score: float | None = None
    neighbor_seed_score: float = 0.0

    @property
    def page_uid(self) -> str:
        return page_uid(self.doc_id, self.page_idx)


@dataclass
class SourceWeights:
    dense_weight: float
    sparse_weight: float
    metadata: dict[str, object]


@dataclass
class SourceAgreement:
    page_overlap: float
    doc_overlap: float
    dense_top1_in_sparse_lookup: bool
    agreement_score: float
    disagreement_score: float
    metadata: dict[str, object]


@dataclass
class ReciprocalSourceReliability:
    source_top_pages: int
    lookup_pages: int
    doc_weight: float
    page_weight: float
    dense_page_support: float
    dense_doc_support: float
    sparse_page_support: float
    sparse_doc_support: float
    dense_reliability: float
    sparse_reliability: float


@dataclass
class RestartVector:
    seed: dict[str, float]
    doc_ppr_nodes: set[str]
    metadata: dict[str, object]


@dataclass
class TransitionPolicy:
    page_doc_multipliers: dict[str, float]
    global_multiplier: float
    metadata: dict[str, object]


@dataclass
class AdjacentCoherencePolicy:
    page_reliabilities: dict[str, float]
    metadata: dict[str, object]


@dataclass
class EvidenceCommunityPolicy:
    page_source_support: dict[str, float]
    page_local_support: dict[str, float]
    metadata: dict[str, object]


@dataclass
class PositionEvidencePolicy:
    active_role_weights: dict[str, float]
    role_page_weights: dict[str, dict[str, float]]
    metadata: dict[str, object]


@dataclass
class QueryAnchorEvidencePolicy:
    anchor_page_weights: dict[str, dict[str, float]]
    anchor_node_weights: dict[str, float]
    anchor_labels: dict[str, str]
    metadata: dict[str, object]


@dataclass
class PdfHyperlinkEdge:
    source_page_uid: str
    target_doc_id: str
    raw_link_count: int = 1


@dataclass
class PdfHyperlinkGraph:
    by_source_page: dict[str, list[PdfHyperlinkEdge]]
    edge_count: int
    source_page_count: int
    target_doc_count: int


@dataclass
class ExternalPageGraphEdge:
    source_page_uid: str
    target_page_uid: str | None = None
    target_doc_id: str | None = None
    score: float = 1.0
    raw_weight: float = 1.0
    edge_type: str = ""


@dataclass
class ExternalPageGraph:
    by_source_page: dict[str, list[ExternalPageGraphEdge]]
    edge_count: int
    source_page_count: int
    target_page_count: int
    target_doc_count: int


@dataclass
class DocPageCatalog:
    page_counts: dict[str, int]
    page_number_indices: dict[str, dict[int, set[int]]]
    page_texts: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Graph/PPR rerank over dense and SPLADE page-retrieval candidates. "
            "The output uses the same prediction JSON schema as the existing retrieval helpers."
        )
    )
    parser.add_argument("--dense-prediction-json", required=True)
    parser.add_argument("--sparse-prediction-json", required=True)
    parser.add_argument("--gold", help="Optional MMQA-style JSONL for summary metrics.")
    parser.add_argument(
        "--question-type",
        default="",
        help="Optional metadata.type filter applied when --gold is provided, e.g. ImageListQ.",
    )
    parser.add_argument(
        "--doc-pages-jsonl",
        default="",
        help=(
            "Optional converted doc_pages JSONL. When provided, query-position evidence can "
            "use true document page counts for first/last/early/late page roles."
        ),
    )
    parser.add_argument(
        "--dense-top-pages",
        type=int,
        default=1000,
        help="Dense source candidate pages. Use 0 to disable the dense source.",
    )
    parser.add_argument(
        "--sparse-top-pages",
        type=int,
        default=1000,
        help="Sparse/SPLADE source candidate pages. Use 0 to disable the sparse source.",
    )
    parser.add_argument(
        "--final-top-pages",
        type=int,
        default=1000,
        help="Number of final page rows to write. Use 0 to write every graph candidate page.",
    )
    parser.add_argument(
        "--per-doc-page-limit",
        type=int,
        default=0,
        help=(
            "Optional cap on final page rows per document. Use 1 for doc-shortlist style output; "
            "use 0 for no cap."
        ),
    )
    parser.add_argument("--rrf-k", type=float, default=10.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-source-weight-mode",
        choices=["none", "agreement", "reciprocal"],
        default="none",
        help=(
            "Optional query-adaptive source weighting. 'agreement' increases dense weight "
            "and decreases sparse weight when dense/SPLADE overlap is low. 'reciprocal' "
            "estimates dense and sparse reliability separately from rank-weighted reciprocal support."
        ),
    )
    parser.add_argument(
        "--adaptive-source-agreement-top-pages",
        type=int,
        default=20,
        help="Top pages used to estimate dense/SPLADE page and doc overlap. Default: 20.",
    )
    parser.add_argument(
        "--adaptive-source-top1-lookup-pages",
        type=int,
        default=1000,
        help="Sparse top pages used to check whether dense top-1 is supported. Default: 1000.",
    )
    parser.add_argument(
        "--adaptive-source-doc-overlap-weight",
        type=float,
        default=0.7,
        help="Agreement score weight for dense/SPLADE doc-overlap. Default: 0.7.",
    )
    parser.add_argument(
        "--adaptive-source-page-overlap-weight",
        type=float,
        default=0.2,
        help="Agreement score weight for dense/SPLADE page-overlap. Default: 0.2.",
    )
    parser.add_argument(
        "--adaptive-source-top1-weight",
        type=float,
        default=0.1,
        help="Agreement score weight for dense-top1-in-sparse support. Default: 0.1.",
    )
    parser.add_argument(
        "--adaptive-source-strength",
        type=float,
        default=0.5,
        help=(
            "How strongly disagreement shifts mass toward dense. With the defaults, "
            "zero agreement gives dense x1.5 and sparse x0.5 before clamps."
        ),
    )
    parser.add_argument(
        "--adaptive-source-gamma",
        type=float,
        default=1.0,
        help="Exponent applied to disagreement before weighting. Default: 1.0.",
    )
    parser.add_argument("--adaptive-source-min-dense-mult", type=float, default=1.0)
    parser.add_argument("--adaptive-source-max-dense-mult", type=float, default=1.5)
    parser.add_argument("--adaptive-source-min-sparse-mult", type=float, default=0.5)
    parser.add_argument("--adaptive-source-max-sparse-mult", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-source-reciprocal-top-pages",
        type=int,
        default=20,
        help="Top source pages used for reciprocal support reliability. Default: 20.",
    )
    parser.add_argument(
        "--adaptive-source-reciprocal-lookup-pages",
        type=int,
        default=1000,
        help="Target source pages used to look up reciprocal support. Default: 1000.",
    )
    parser.add_argument(
        "--adaptive-source-reciprocal-doc-weight",
        type=float,
        default=0.7,
        help="Reliability weight for reciprocal doc support. Default: 0.7.",
    )
    parser.add_argument(
        "--adaptive-source-reciprocal-page-weight",
        type=float,
        default=0.3,
        help="Reliability weight for reciprocal page support. Default: 0.3.",
    )
    parser.add_argument("--adaptive-source-reciprocal-min-mult", type=float, default=0.75)
    parser.add_argument("--adaptive-source-reciprocal-max-mult", type=float, default=1.25)
    parser.add_argument(
        "--adaptive-source-reciprocal-preserve-total",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve base dense+sparse source weight total in reciprocal mode. Default: true.",
    )
    parser.add_argument(
        "--adaptive-restart-mode",
        choices=["none", "agreement", "reciprocal"],
        default="none",
        help=(
            "Optional query-adaptive PPR restart vector. 'agreement' shifts restart mass "
            "toward dense pages and gates doc-node restart mass by dense/SPLADE agreement. "
            "'reciprocal' estimates dense and sparse restart reliability separately from "
            "rank-weighted reciprocal support."
        ),
    )
    parser.add_argument(
        "--adaptive-restart-source-strength",
        type=float,
        default=0.5,
        help="Strength for dense/sparse restart source reweighting. Default: 0.5.",
    )
    parser.add_argument(
        "--adaptive-restart-gamma",
        type=float,
        default=1.0,
        help="Exponent applied to disagreement for restart source weights. Default: 1.0.",
    )
    parser.add_argument("--adaptive-restart-min-dense-mult", type=float, default=1.0)
    parser.add_argument("--adaptive-restart-max-dense-mult", type=float, default=1.5)
    parser.add_argument("--adaptive-restart-min-sparse-mult", type=float, default=0.5)
    parser.add_argument("--adaptive-restart-max-sparse-mult", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-restart-page-seed-weight",
        type=float,
        default=1.0,
        help="Global multiplier for page-node restart mass in adaptive restart mode.",
    )
    parser.add_argument(
        "--adaptive-restart-doc-seed-weight",
        type=float,
        default=0.25,
        help=(
            "Base doc-node restart mass in adaptive restart mode. It is multiplied by "
            "agreement-gated doc multiplier. Default: 0.25."
        ),
    )
    parser.add_argument("--adaptive-restart-min-doc-mult", type=float, default=0.0)
    parser.add_argument("--adaptive-restart-max-doc-mult", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-restart-neighbor-seed-weight",
        type=float,
        default=1.0,
        help="Multiplier for inherited neighbor restart mass in adaptive restart mode.",
    )
    parser.add_argument(
        "--adaptive-restart-preserve-source-total",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve dense+sparse restart source weight total in reciprocal restart mode. Default: true.",
    )
    parser.add_argument(
        "--adaptive-transition-mode",
        choices=["none", "agreement", "reciprocal"],
        default="none",
        help=(
            "Optional query-adaptive transition matrix. 'agreement' gates all graph edges "
            "by dense/SPLADE agreement. 'reciprocal' also gates page-doc edges by local "
            "rank-weighted reciprocal support."
        ),
    )
    parser.add_argument("--adaptive-transition-agreement-min-mult", type=float, default=0.75)
    parser.add_argument("--adaptive-transition-agreement-max-mult", type=float, default=1.0)
    parser.add_argument("--adaptive-transition-local-min-mult", type=float, default=0.75)
    parser.add_argument("--adaptive-transition-local-max-mult", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-transition-local-weight",
        type=float,
        default=0.5,
        help=(
            "Blend weight for local reciprocal support in reciprocal transition mode. "
            "0 uses only global agreement; 1 uses only local support."
        ),
    )
    parser.add_argument(
        "--adaptive-transition-gate-adjacent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply transition gating to same-doc adjacent-page edges. Default: true.",
    )
    parser.add_argument(
        "--adaptive-transition-gate-page-doc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply transition gating to page-doc edges. Default: true.",
    )
    parser.add_argument(
        "--adaptive-transition-gate-page-to-doc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply transition gating to directed page->doc edges. Default: true.",
    )
    parser.add_argument(
        "--adaptive-transition-gate-doc-to-page",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply transition gating to directed doc->page edges. Default: true.",
    )
    parser.add_argument(
        "--adaptive-adjacent-mode",
        choices=["none", "reciprocal_coherence"],
        default="none",
        help=(
            "Optional query-adaptive same-doc page-page edge gating. "
            "'reciprocal_coherence' gates adjacent-page edges by per-page "
            "dense/SPLADE reciprocal support."
        ),
    )
    parser.add_argument("--adaptive-adjacent-min-mult", type=float, default=0.0)
    parser.add_argument("--adaptive-adjacent-max-mult", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-adjacent-power",
        type=float,
        default=0.5,
        help=(
            "Power for combining endpoint page reliabilities. Default 0.5 gives "
            "sqrt(left_reliability * right_reliability)."
        ),
    )
    parser.add_argument(
        "--evidence-community-mode",
        choices=["none", "source", "local", "source_local"],
        default="none",
        help=(
            "Add query-specific intra-document evidence community nodes. 'source' links "
            "pages to per-doc source-support communities, 'local' links pages to per-doc "
            "local-neighborhood communities, and 'source_local' enables both."
        ),
    )
    parser.add_argument("--evidence-community-source-edge-weight", type=float, default=0.25)
    parser.add_argument("--evidence-community-local-edge-weight", type=float, default=0.25)
    parser.add_argument("--evidence-community-min-page-support", type=float, default=0.05)
    parser.add_argument(
        "--evidence-community-local-window",
        type=int,
        default=2,
        help="Same-document page window used to estimate local community support.",
    )
    parser.add_argument(
        "--evidence-community-both-source-bonus",
        type=float,
        default=0.10,
        help="Raw support bonus for pages present in both dense and sparse sources.",
    )
    parser.add_argument(
        "--evidence-community-support-power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized page support before adding community edges.",
    )
    parser.add_argument(
        "--position-evidence-mode",
        choices=["none", "query_gated"],
        default="none",
        help=(
            "Add query-gated structural position evidence nodes. In query_gated mode, "
            "lexical cues such as first page, last page, cover page, references, or page/slide 17 "
            "seed role nodes that connect to candidate pages with matching document positions."
        ),
    )
    parser.add_argument("--position-evidence-edge-weight", type=float, default=0.10)
    parser.add_argument("--position-evidence-restart-weight", type=float, default=0.10)
    parser.add_argument(
        "--position-evidence-scope",
        choices=["global", "doc_conditioned"],
        default="global",
        help=(
            "Scope of position evidence edges. 'global' connects role nodes to all matching "
            "candidate pages. 'doc_conditioned' only connects pages in high-confidence "
            "candidate documents, with edge weights scaled by dense/SPLADE doc support."
        ),
    )
    parser.add_argument(
        "--position-evidence-doc-top-k",
        type=int,
        default=20,
        help="Candidate documents kept for doc_conditioned position evidence. Default: 20.",
    )
    parser.add_argument(
        "--position-evidence-min-doc-support",
        type=float,
        default=0.0,
        help="Minimum normalized doc support for doc_conditioned position evidence.",
    )
    parser.add_argument(
        "--position-evidence-explicit-page-edge-weight",
        type=float,
        default=0.50,
        help="Edge weight for explicit page-number cues such as 'page 17' or 'slide 17'.",
    )
    parser.add_argument(
        "--position-evidence-explicit-page-restart-weight",
        type=float,
        default=0.25,
        help="Restart weight for explicit page-number cues such as 'page 17' or 'slide 17'.",
    )
    parser.add_argument(
        "--position-evidence-early-frac",
        type=float,
        default=0.15,
        help="Fraction of a document considered early for structural position roles.",
    )
    parser.add_argument(
        "--position-evidence-late-frac",
        type=float,
        default=0.15,
        help="Fraction of a document considered late for structural position roles.",
    )
    parser.add_argument(
        "--query-anchor-evidence-mode",
        choices=["none", "entity_numeric"],
        default="none",
        help=(
            "Add query-anchor evidence nodes from entity-like and numeric query spans. "
            "Requires --doc-pages-jsonl with page text fields."
        ),
    )
    parser.add_argument(
        "--query-anchor-text-field",
        nargs="*",
        default=["ocr_text", "vlm_text", "markdown", "text", "page_text", "content"],
        help="doc_pages JSONL fields concatenated as page text for query-anchor evidence.",
    )
    parser.add_argument(
        "--query-anchor-scope",
        choices=["global", "doc_conditioned"],
        default="doc_conditioned",
        help="Whether query-anchor nodes connect to all matched candidate pages or only top supported docs.",
    )
    parser.add_argument("--query-anchor-doc-top-k", type=int, default=20)
    parser.add_argument("--query-anchor-min-doc-support", type=float, default=0.0)
    parser.add_argument("--query-anchor-edge-weight", type=float, default=0.15)
    parser.add_argument("--query-anchor-restart-weight", type=float, default=0.10)
    parser.add_argument("--query-anchor-max-anchors", type=int, default=16)
    parser.add_argument("--query-anchor-min-entity-len", type=int, default=3)
    parser.add_argument(
        "--query-anchor-weight-mode",
        choices=["uniform", "local_idf"],
        default="uniform",
        help=(
            "Weight matched query-anchor nodes. local_idf downweights anchors that match "
            "many candidate pages for the current query."
        ),
    )
    parser.add_argument(
        "--query-anchor-min-node-weight",
        type=float,
        default=0.10,
        help="Floor for local_idf query-anchor node weights.",
    )
    parser.add_argument(
        "--query-anchor-max-page-matches",
        type=int,
        default=0,
        help="Drop anchors matching more than this many candidate pages. Use 0 to disable.",
    )
    parser.add_argument(
        "--query-anchor-reasoning-mode",
        choices=["none", "financial_slots", "constraint_bundles"],
        default="none",
        help=(
            "Optional query-anchor reasoning layer. financial_slots adds a co-occurrence "
            "node for pages that satisfy multiple financial evidence slots such as metric "
            "terms and fiscal years. constraint_bundles adds a general conjunctive "
            "evidence node that rewards pages satisfying multiple query constraint types."
        ),
    )
    parser.add_argument(
        "--query-anchor-financial-bundle-weight",
        type=float,
        default=1.0,
        help="Node weight for financial slot co-occurrence evidence.",
    )
    parser.add_argument(
        "--query-anchor-financial-min-slot-types",
        type=int,
        default=2,
        help="Minimum financial slot categories a page must match before it receives bundle evidence.",
    )
    parser.add_argument(
        "--query-anchor-financial-max-page-matches",
        type=int,
        default=0,
        help="Drop financial bundle evidence if it matches more than this many pages. Use 0 to disable.",
    )
    parser.add_argument(
        "--query-anchor-financial-table-bonus",
        type=float,
        default=0.0,
        help="Extra multiplier for financial bundle page edges based on numeric/table-like page text.",
    )
    parser.add_argument(
        "--query-anchor-financial-table-min-score",
        type=float,
        default=0.0,
        help="Require this minimum table-likeness score for financial bundle pages. Use 0 to disable.",
    )
    parser.add_argument(
        "--query-anchor-constraint-bundle-weight",
        type=float,
        default=1.0,
        help="Node weight for general query-constraint bundle evidence.",
    )
    parser.add_argument(
        "--query-anchor-constraint-min-slot-types",
        type=int,
        default=2,
        help="Minimum distinct constraint types a page must satisfy before bundle evidence is active.",
    )
    parser.add_argument(
        "--query-anchor-constraint-max-page-matches",
        type=int,
        default=0,
        help="Drop the constraint bundle if it matches more than this many pages. Use 0 to disable.",
    )
    parser.add_argument(
        "--query-anchor-constraint-max-slot-page-matches",
        type=int,
        default=0,
        help=(
            "Ignore individual constraint slots that match more than this many candidate pages. "
            "Use 0 to keep all slots with specificity downweighting."
        ),
    )
    parser.add_argument(
        "--query-anchor-constraint-specificity-floor",
        type=float,
        default=0.10,
        help="Minimum local specificity weight for broad constraint slots.",
    )
    parser.add_argument(
        "--query-anchor-constraint-table-bonus",
        type=float,
        default=0.0,
        help="Extra multiplier for constraint-bundle page edges based on numeric/table-like page text.",
    )
    parser.add_argument(
        "--query-anchor-constraint-table-min-score",
        type=float,
        default=0.0,
        help="Require this minimum table-likeness score for constraint-bundle pages. Use 0 to disable.",
    )
    parser.add_argument(
        "--pdf-hyperlink-edges-jsonl",
        default="",
        help=(
            "Optional edge JSONL from scripts/build_pdf_hyperlink_graph.py. "
            "Edges are used as authored Wikipedia hyperlink transitions."
        ),
    )
    parser.add_argument(
        "--pdf-hyperlink-edge-weight",
        type=float,
        default=0.0,
        help="Transition weight for PDF hyperlink edges. Use 0 to disable.",
    )
    parser.add_argument(
        "--pdf-hyperlink-direction",
        choices=["source_to_target_doc", "bidirectional_doc"],
        default="source_to_target_doc",
        help=(
            "source_to_target_doc adds source_page -> target_doc edges. "
            "bidirectional_doc also adds target_doc -> source_page reverse edges."
        ),
    )
    parser.add_argument(
        "--pdf-hyperlink-weight-mode",
        choices=["uniform", "log_count"],
        default="uniform",
        help="Use uniform hyperlink edge weights or scale by log(1 + raw_link_count).",
    )
    parser.add_argument(
        "--pdf-hyperlink-max-edges-per-source",
        type=int,
        default=0,
        help="Optional cap on hyperlink targets per source page. Use 0 for no cap.",
    )
    parser.add_argument(
        "--pdf-hyperlink-source-top-k",
        type=int,
        default=0,
        help=(
            "Only emit hyperlink edges from pages whose best dense/SPLADE/expansion "
            "rank is within this cutoff. Use 0 for no source-rank gate."
        ),
    )
    parser.add_argument(
        "--pdf-hyperlink-target-doc-top-k",
        type=int,
        default=0,
        help=(
            "Only emit hyperlink edges to docs whose best candidate page rank is within "
            "this cutoff. Use 0 for no target-doc gate."
        ),
    )
    parser.add_argument(
        "--pdf-hyperlink-query-support-weight-mode",
        choices=["none", "source_rank_decay", "source_target_rank_decay"],
        default="none",
        help=(
            "Optionally scale hyperlink edge weights by query support ranks. "
            "source_rank_decay uses the source page rank; source_target_rank_decay "
            "also uses the target doc's best candidate-page rank."
        ),
    )
    parser.add_argument(
        "--external-page-graph-jsonl",
        default="",
        help=(
            "Optional page graph edge JSONL. Rows may contain source_page_uid and "
            "target_page_uid for page-page edges, or target_doc_id for page-doc edges. "
            "This is intended for learned layout/semantic graph edges such as "
            "LayoutLMv3 or DocGraphLM kNN links."
        ),
    )
    parser.add_argument(
        "--external-page-graph-edge-weight",
        type=float,
        default=0.0,
        help="Base transition weight for external page graph edges. Use 0 to disable.",
    )
    parser.add_argument(
        "--external-page-graph-direction",
        choices=["as_directed", "bidirectional"],
        default="as_directed",
        help="Whether to use external edges as directed or add reverse transitions.",
    )
    parser.add_argument(
        "--external-page-graph-weight-mode",
        choices=["uniform", "score", "weight"],
        default="score",
        help=(
            "Scale the base edge weight uniformly, by row score, or by row weight. "
            "Missing score/weight values fall back to 1."
        ),
    )
    parser.add_argument(
        "--external-page-graph-max-edges-per-source",
        type=int,
        default=0,
        help="Optional cap on external targets per source page. Use 0 for no cap.",
    )
    parser.add_argument(
        "--external-page-graph-source-top-k",
        type=int,
        default=0,
        help=(
            "Only emit external edges from pages whose best dense/SPLADE/expansion "
            "rank is within this cutoff. Use 0 for no source-rank gate."
        ),
    )
    parser.add_argument(
        "--external-page-graph-target-top-k",
        type=int,
        default=0,
        help=(
            "Only emit page-target external edges to pages whose best dense/SPLADE/"
            "expansion rank is within this cutoff. Use 0 for no target-rank gate."
        ),
    )
    parser.add_argument(
        "--splade-index-pt",
        default="",
        help=(
            "Optional full SPLADE page index .pt. When set with --expansion-top-pages, "
            "the script performs pseudo-relevance-feedback page expansion before graph/PPR. "
            "It is also used as a page catalog for neighbor-page expansion."
        ),
    )
    parser.add_argument(
        "--expansion-top-pages",
        type=int,
        default=0,
        help="Number of corpus pages to add from SPLADE sparse page-page expansion. Default: disabled.",
    )
    parser.add_argument(
        "--expand-from-top-dense-pages",
        type=int,
        default=50,
        help="Dense seed pages used to build the expansion sparse vector.",
    )
    parser.add_argument(
        "--expand-from-top-sparse-pages",
        type=int,
        default=50,
        help="SPLADE seed pages used to build the expansion sparse vector.",
    )
    parser.add_argument(
        "--expansion-weight",
        type=float,
        default=1.0,
        help="RRF-style source weight assigned to expansion pages.",
    )
    parser.add_argument(
        "--expansion-source-term-topk",
        type=int,
        default=32,
        help="Per-seed page SPLADE terms used for expansion query construction.",
    )
    parser.add_argument(
        "--expansion-query-topk-terms",
        type=int,
        default=256,
        help="Global top terms retained in the combined expansion sparse query.",
    )
    parser.add_argument(
        "--expansion-min-score",
        type=float,
        default=0.0,
        help="Drop expansion retrieved pages whose sparse score is <= this value.",
    )
    parser.add_argument(
        "--score-seed-weight",
        type=float,
        default=0.0,
        help=(
            "Optional within-source min-max score contribution added to rank/RRF page seeds. "
            "Default 0 keeps the graph seed rank-based and comparable across dense/SPLADE."
        ),
    )
    parser.add_argument(
        "--neighbor-expansion-window",
        type=int,
        default=0,
        help=(
            "Add unseen same-document neighboring pages within this distance for strong seed pages. "
            "Use 0 to disable neighbor-page expansion."
        ),
    )
    parser.add_argument(
        "--expand-neighbors-from-top-dense-pages",
        type=int,
        default=50,
        help="Dense seed pages used to add same-document neighboring pages.",
    )
    parser.add_argument(
        "--expand-neighbors-from-top-sparse-pages",
        type=int,
        default=50,
        help="Sparse seed pages used to add same-document neighboring pages.",
    )
    parser.add_argument(
        "--neighbor-seed-weight",
        type=float,
        default=0.25,
        help=(
            "Relative weight for neighbor-page seed mass inherited from a seed page. "
            "The inherited contribution is source_weight * neighbor_seed_weight / (rrf_k + rank) / distance."
        ),
    )
    parser.add_argument(
        "--doc-seed-weight",
        type=float,
        default=1.0,
        help="Weight for doc-node restart mass from dense/SPLADE doc RRF. Default: 1.",
    )
    parser.add_argument("--restart-prob", type=float, default=0.20)
    parser.add_argument("--ppr-iters", type=int, default=30)
    parser.add_argument("--page-doc-edge-weight", type=float, default=1.0)
    parser.add_argument(
        "--page-to-doc-edge-weight",
        type=float,
        default=None,
        help=(
            "Directed page->doc transition weight. Defaults to --page-doc-edge-weight "
            "to preserve the original symmetric graph."
        ),
    )
    parser.add_argument(
        "--doc-to-page-edge-weight",
        type=float,
        default=None,
        help=(
            "Directed doc->page transition weight. Defaults to --page-doc-edge-weight "
            "to preserve the original symmetric graph."
        ),
    )
    parser.add_argument("--adjacent-page-edge-weight", type=float, default=0.25)
    parser.add_argument(
        "--same-doc-window",
        type=int,
        default=1,
        help="Connect candidate pages from the same doc when their page indices differ by <= this value.",
    )
    parser.add_argument(
        "--final-page-seed-weight",
        type=float,
        default=1.0,
        help="Final normalized source page-seed weight.",
    )
    parser.add_argument(
        "--final-ppr-page-weight",
        type=float,
        default=1.0,
        help="Final normalized page-PPR weight.",
    )
    parser.add_argument(
        "--final-ppr-doc-weight",
        type=float,
        default=0.5,
        help="Final normalized owning-doc PPR weight for each page.",
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict] = {}
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object of prediction rows: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid and key is empty: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def load_pdf_hyperlink_graph(path: Path) -> PdfHyperlinkGraph:
    by_source_page: dict[str, list[PdfHyperlinkEdge]] = defaultdict(list)
    target_doc_ids: set[str] = set()
    edge_count = 0
    if not path.exists():
        raise FileNotFoundError(f"PDF hyperlink edge JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_page_uid = str(row.get("source_page_uid", "")).strip()
            target_doc_id = str(row.get("target_doc_id", "")).strip()
            if not source_page_uid or not target_doc_id:
                continue
            try:
                raw_link_count = int(row.get("raw_link_count", 1) or 1)
            except (TypeError, ValueError):
                raw_link_count = 1
            by_source_page[source_page_uid].append(
                PdfHyperlinkEdge(
                    source_page_uid=source_page_uid,
                    target_doc_id=target_doc_id,
                    raw_link_count=max(1, raw_link_count),
                )
            )
            target_doc_ids.add(target_doc_id)
            edge_count += 1
    return PdfHyperlinkGraph(
        by_source_page=dict(by_source_page),
        edge_count=edge_count,
        source_page_count=len(by_source_page),
        target_doc_count=len(target_doc_ids),
    )


def load_external_page_graph(path: Path) -> ExternalPageGraph:
    by_source_page: dict[str, list[ExternalPageGraphEdge]] = defaultdict(list)
    target_page_uids: set[str] = set()
    target_doc_ids: set[str] = set()
    edge_count = 0
    if not path.exists():
        raise FileNotFoundError(f"External page graph JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_page_uid = str(row.get("source_page_uid", "")).strip()
            target_page_uid = str(row.get("target_page_uid", "")).strip() or None
            target_doc_id = str(row.get("target_doc_id", "")).strip() or None
            if not source_page_uid or (not target_page_uid and not target_doc_id):
                continue
            try:
                score = float(row.get("score", 1.0))
            except (TypeError, ValueError):
                score = 1.0
            try:
                raw_weight = float(row.get("weight", row.get("raw_weight", 1.0)))
            except (TypeError, ValueError):
                raw_weight = 1.0
            edge_type = str(row.get("edge_type", row.get("type", ""))).strip()
            by_source_page[source_page_uid].append(
                ExternalPageGraphEdge(
                    source_page_uid=source_page_uid,
                    target_page_uid=target_page_uid,
                    target_doc_id=target_doc_id,
                    score=max(0.0, score),
                    raw_weight=max(0.0, raw_weight),
                    edge_type=edge_type,
                )
            )
            if target_page_uid:
                target_page_uids.add(target_page_uid)
            if target_doc_id:
                target_doc_ids.add(target_doc_id)
            edge_count += 1
    return ExternalPageGraph(
        by_source_page=dict(by_source_page),
        edge_count=edge_count,
        source_page_count=len(by_source_page),
        target_page_count=len(target_page_uids),
        target_doc_count=len(target_doc_ids),
    )


class SparsePageIndex:
    def __init__(self, path: Path, *, load_postings: bool = True) -> None:
        import torch

        payload = torch.load(path, map_location="cpu")
        self.path = path
        self.page_uids: list[str] = [str(value) for value in payload["page_uids"]]
        self.doc_ids: list[str] = [str(value) for value in payload["doc_ids"]]
        self.page_indices = payload["page_indices"].to(torch.int64)
        self.offsets = payload["offsets"].to(torch.int64)
        self.term_ids = payload["term_ids"].to(torch.int64)
        self.term_weights = payload["term_weights"].to(torch.float32)
        self.page_uid_to_idx = {page_uid: idx for idx, page_uid in enumerate(self.page_uids)}
        self.doc_to_page_indices: dict[str, set[int]] = defaultdict(set)
        for doc_id, page_idx in zip(self.doc_ids, self.page_indices.tolist()):
            self.doc_to_page_indices[str(doc_id)].add(int(page_idx))
        self.postings: dict[int, tuple[object, object]] = {}
        if load_postings:
            posting_pages: dict[int, list[int]] = defaultdict(list)
            posting_weights: dict[int, list[float]] = defaultdict(list)
            for page_idx in range(len(self.page_uids)):
                start = int(self.offsets[page_idx].item())
                end = int(self.offsets[page_idx + 1].item())
                for term_id, weight in zip(
                    self.term_ids[start:end].tolist(),
                    self.term_weights[start:end].tolist(),
                ):
                    posting_pages[int(term_id)].append(page_idx)
                    posting_weights[int(term_id)].append(float(weight))
            for term_id, page_ids in posting_pages.items():
                self.postings[int(term_id)] = (
                    torch.tensor(page_ids, dtype=torch.int64),
                    torch.tensor(posting_weights[int(term_id)], dtype=torch.float32),
                )

    def has_page(self, doc_id: str, page_idx: int) -> bool:
        return int(page_idx) in self.doc_to_page_indices.get(str(doc_id), set())

    def top_terms_for_page(self, page_uid_value: str, topk: int) -> list[tuple[int, float]]:
        page_idx = self.page_uid_to_idx.get(page_uid_value)
        if page_idx is None:
            return []
        start = int(self.offsets[page_idx].item())
        end = int(self.offsets[page_idx + 1].item())
        term_ids = self.term_ids[start:end].tolist()
        term_weights = self.term_weights[start:end].tolist()
        pairs = [(int(term_id), float(weight)) for term_id, weight in zip(term_ids, term_weights)]
        if topk > 0:
            pairs = pairs[:topk]
        return pairs

    def retrieve_sparse_query(
        self,
        query_terms: dict[int, float],
        *,
        top_pages: int,
        min_score: float,
    ) -> list[tuple[str, int, float, int]]:
        import torch

        if not query_terms or top_pages <= 0:
            return []
        scores = torch.zeros(len(self.page_uids), dtype=torch.float32)
        for term_id, query_weight in query_terms.items():
            posting = self.postings.get(int(term_id))
            if posting is None:
                continue
            posting_page_ids, posting_doc_weights = posting
            scores.index_add_(0, posting_page_ids, posting_doc_weights * float(query_weight))
        positive_page_ids = torch.nonzero(scores > float(min_score), as_tuple=False).squeeze(-1)
        if positive_page_ids.numel() <= 0:
            return []
        top_count = min(int(top_pages), int(positive_page_ids.numel()))
        top_scores, top_pos = torch.topk(scores[positive_page_ids], k=top_count)
        ranked_page_ids = positive_page_ids[top_pos]
        return [
            (
                str(self.doc_ids[int(page_id)]),
                int(self.page_indices[int(page_id)].item()),
                float(score),
                rank,
            )
            for rank, (page_id, score) in enumerate(
                zip(ranked_page_ids.tolist(), top_scores.tolist()),
                start=1,
            )
        ]


def load_gold_rows(path: Path, question_type: str = "") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    wanted_type = str(question_type).strip()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if not qid:
                continue
            row_type = str(row.get("metadata", {}).get("type", "")).strip()
            if wanted_type and row_type != wanted_type:
                continue
            rows[qid] = row
    return rows


def maybe_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_manifest_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.replace("\x0c", " ").replace("\u0000", " ")).strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(
            part for part in (normalize_manifest_text(item) for item in value) if part
        ).strip()
    if isinstance(value, dict):
        return " ".join(
            part for part in (normalize_manifest_text(item) for item in value.values()) if part
        ).strip()
    return re.sub(r"\s+", " ", str(value)).strip()


def load_doc_page_catalog(
    path: Path,
    *,
    text_fields: Iterable[str] = (),
    load_page_texts: bool = False,
) -> DocPageCatalog:
    counts: dict[str, int] = {}
    page_number_indices: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    page_texts: dict[str, str] = {}
    text_field_names = [str(field).strip() for field in text_fields if str(field).strip()]
    if not path.exists():
        raise FileNotFoundError(f"doc_pages JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            raw_page_idx = row.get("page_idx", row.get("page_id", row.get("page_number")))
            page_idx = maybe_int(raw_page_idx)
            if page_idx is None:
                continue
            counts[doc_id] = max(counts.get(doc_id, 0), page_idx + 1)
            for field_name in ("page_number", "source_page_number"):
                page_number = maybe_int(row.get(field_name))
                if page_number is not None:
                    page_number_indices[doc_id][page_number].add(page_idx)
            if load_page_texts:
                parts = []
                seen_parts = set()
                for field_name in text_field_names:
                    text = normalize_manifest_text(row.get(field_name))
                    if text and text not in seen_parts:
                        parts.append(text)
                        seen_parts.add(text)
                if parts:
                    page_texts[page_uid(doc_id, page_idx)] = " ".join(parts).lower()
    return DocPageCatalog(
        page_counts=counts,
        page_number_indices={doc_id: dict(values) for doc_id, values in page_number_indices.items()},
        page_texts=page_texts,
    )


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_row(row: list[object]) -> tuple[str, int, float] | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    try:
        page_idx = int(row[1])
    except (TypeError, ValueError):
        return None
    score = 0.0
    if len(row) >= 3:
        try:
            score = float(row[2])
        except (TypeError, ValueError):
            score = 0.0
    return doc_id, page_idx, score


def ranked_unique_pages(rows: list[list[object]], top_pages: int) -> list[tuple[str, int, float, int]]:
    if top_pages <= 0:
        return []
    ranked: list[tuple[str, int, float, int]] = []
    seen: set[str] = set()
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen:
            continue
        seen.add(uid)
        ranked.append((doc_id, page_idx, score, len(ranked) + 1))
        if len(ranked) >= top_pages:
            break
    return ranked


def first_doc_rank_map(page_rows: Iterable[tuple[str, int, float, int]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for doc_id, _page_idx, _score, _page_rank in page_rows:
        if doc_id not in ranks:
            ranks[doc_id] = len(ranks) + 1
    return ranks


def minmax_by_uid(rows: list[tuple[str, int, float, int]]) -> dict[str, float]:
    if not rows:
        return {}
    scores = [score for _doc_id, _page_idx, score, _rank in rows]
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return {page_uid(doc_id, page_idx): 1.0 for doc_id, page_idx, _score, _rank in rows}
    return {
        page_uid(doc_id, page_idx): (float(score) - lo) / (hi - lo)
        for doc_id, page_idx, score, _rank in rows
    }


def clamp(value: float, low: float, high: float) -> float:
    if high < low:
        low, high = high, low
    return min(max(float(value), float(low)), float(high))


def effective_page_to_doc_edge_weight(args: argparse.Namespace) -> float:
    value = args.page_to_doc_edge_weight
    if value is None:
        return float(args.page_doc_edge_weight)
    return float(value)


def effective_doc_to_page_edge_weight(args: argparse.Namespace) -> float:
    value = args.doc_to_page_edge_weight
    if value is None:
        return float(args.page_doc_edge_weight)
    return float(value)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def page_rank_map(
    pages: list[tuple[str, int, float, int]],
    limit: int,
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for doc_id, page_idx, _score, rank in pages[: max(0, int(limit))]:
        ranks.setdefault(page_uid(doc_id, page_idx), int(rank))
    return ranks


def doc_rank_map(
    pages: list[tuple[str, int, float, int]],
    limit: int,
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for doc_id, _page_idx, _score, _rank in pages[: max(0, int(limit))]:
        if doc_id not in ranks:
            ranks[doc_id] = len(ranks) + 1
    return ranks


def reciprocal_rank_support(
    *,
    source_pages: list[tuple[str, int, float, int]],
    target_page_ranks: dict[str, int],
    target_doc_ranks: dict[str, int],
    source_top_pages: int,
    rrf_k: float,
) -> tuple[float, float]:
    total_source_importance = 0.0
    page_support = 0.0
    doc_support = 0.0
    reciprocal_scale = float(rrf_k) + 1.0
    for doc_id, page_idx, _score, rank in source_pages[: max(0, int(source_top_pages))]:
        source_importance = 1.0 / (float(rrf_k) + float(rank))
        total_source_importance += source_importance
        page_rank = target_page_ranks.get(page_uid(doc_id, page_idx))
        if page_rank is not None:
            page_support += source_importance * reciprocal_scale / (float(rrf_k) + float(page_rank))
        doc_rank = target_doc_ranks.get(doc_id)
        if doc_rank is not None:
            doc_support += source_importance * reciprocal_scale / (float(rrf_k) + float(doc_rank))
    if total_source_importance <= 0:
        return 0.0, 0.0
    return (
        clamp(page_support / total_source_importance, 0.0, 1.0),
        clamp(doc_support / total_source_importance, 0.0, 1.0),
    )


def reciprocal_rank_value(rank: int | None, rrf_k: float) -> float:
    if rank is None:
        return 0.0
    return clamp((float(rrf_k) + 1.0) / (float(rrf_k) + float(rank)), 0.0, 1.0)


def compute_source_agreement(
    *,
    dense_pages: list[tuple[str, int, float, int]],
    sparse_pages: list[tuple[str, int, float, int]],
    args: argparse.Namespace,
) -> SourceAgreement:
    overlap_top_pages = max(0, int(args.adaptive_source_agreement_top_pages))
    top1_lookup_pages = max(0, int(args.adaptive_source_top1_lookup_pages))
    dense_overlap_pages = {
        page_uid(doc_id, page_idx)
        for doc_id, page_idx, _score, _rank in dense_pages[:overlap_top_pages]
    }
    sparse_overlap_pages = {
        page_uid(doc_id, page_idx)
        for doc_id, page_idx, _score, _rank in sparse_pages[:overlap_top_pages]
    }
    dense_overlap_docs = {doc_id for doc_id, _page_idx, _score, _rank in dense_pages[:overlap_top_pages]}
    sparse_overlap_docs = {doc_id for doc_id, _page_idx, _score, _rank in sparse_pages[:overlap_top_pages]}
    sparse_lookup_pages = {
        page_uid(doc_id, page_idx)
        for doc_id, page_idx, _score, _rank in sparse_pages[:top1_lookup_pages]
    }
    dense_top1_uid = (
        page_uid(dense_pages[0][0], dense_pages[0][1])
        if dense_pages
        else ""
    )
    dense_top1_in_sparse_lookup = bool(dense_top1_uid and dense_top1_uid in sparse_lookup_pages)

    page_overlap = jaccard(dense_overlap_pages, sparse_overlap_pages)
    doc_overlap = jaccard(dense_overlap_docs, sparse_overlap_docs)
    doc_overlap_weight = max(0.0, float(args.adaptive_source_doc_overlap_weight))
    page_overlap_weight = max(0.0, float(args.adaptive_source_page_overlap_weight))
    top1_weight = max(0.0, float(args.adaptive_source_top1_weight))
    agreement_denominator = doc_overlap_weight + page_overlap_weight + top1_weight
    if agreement_denominator <= 0:
        agreement_score = 0.0
    else:
        agreement_score = (
            doc_overlap_weight * doc_overlap
            + page_overlap_weight * page_overlap
            + top1_weight * float(dense_top1_in_sparse_lookup)
        ) / agreement_denominator
    agreement_score = clamp(agreement_score, 0.0, 1.0)
    disagreement = clamp(1.0 - agreement_score, 0.0, 1.0)
    metadata = {
        "adaptive_source_agreement_top_pages": overlap_top_pages,
        "adaptive_source_top1_lookup_pages": top1_lookup_pages,
        "adaptive_source_page_overlap": page_overlap,
        "adaptive_source_doc_overlap": doc_overlap,
        "adaptive_source_dense_top1_in_sparse_lookup": dense_top1_in_sparse_lookup,
        "adaptive_source_agreement_score": agreement_score,
        "adaptive_source_disagreement_score": disagreement,
    }
    return SourceAgreement(
        page_overlap=page_overlap,
        doc_overlap=doc_overlap,
        dense_top1_in_sparse_lookup=dense_top1_in_sparse_lookup,
        agreement_score=agreement_score,
        disagreement_score=disagreement,
        metadata=metadata,
    )


def compute_reciprocal_source_reliability(
    *,
    dense_pages: list[tuple[str, int, float, int]],
    sparse_pages: list[tuple[str, int, float, int]],
    args: argparse.Namespace,
) -> ReciprocalSourceReliability:
    source_top_pages = max(0, int(args.adaptive_source_reciprocal_top_pages))
    lookup_pages = max(0, int(args.adaptive_source_reciprocal_lookup_pages))
    sparse_page_ranks = page_rank_map(sparse_pages, lookup_pages)
    dense_page_ranks = page_rank_map(dense_pages, lookup_pages)
    sparse_doc_ranks = doc_rank_map(sparse_pages, lookup_pages)
    dense_doc_ranks = doc_rank_map(dense_pages, lookup_pages)
    dense_page_support, dense_doc_support = reciprocal_rank_support(
        source_pages=dense_pages,
        target_page_ranks=sparse_page_ranks,
        target_doc_ranks=sparse_doc_ranks,
        source_top_pages=source_top_pages,
        rrf_k=float(args.rrf_k),
    )
    sparse_page_support, sparse_doc_support = reciprocal_rank_support(
        source_pages=sparse_pages,
        target_page_ranks=dense_page_ranks,
        target_doc_ranks=dense_doc_ranks,
        source_top_pages=source_top_pages,
        rrf_k=float(args.rrf_k),
    )
    doc_weight = max(0.0, float(args.adaptive_source_reciprocal_doc_weight))
    page_weight = max(0.0, float(args.adaptive_source_reciprocal_page_weight))
    support_denominator = doc_weight + page_weight
    if support_denominator <= 0:
        dense_reliability = 0.0
        sparse_reliability = 0.0
    else:
        dense_reliability = clamp(
            (doc_weight * dense_doc_support + page_weight * dense_page_support)
            / support_denominator,
            0.0,
            1.0,
        )
        sparse_reliability = clamp(
            (doc_weight * sparse_doc_support + page_weight * sparse_page_support)
            / support_denominator,
            0.0,
            1.0,
        )
    return ReciprocalSourceReliability(
        source_top_pages=source_top_pages,
        lookup_pages=lookup_pages,
        doc_weight=doc_weight,
        page_weight=page_weight,
        dense_page_support=dense_page_support,
        dense_doc_support=dense_doc_support,
        sparse_page_support=sparse_page_support,
        sparse_doc_support=sparse_doc_support,
        dense_reliability=dense_reliability,
        sparse_reliability=sparse_reliability,
    )


def compute_source_weights(
    *,
    agreement: SourceAgreement,
    dense_pages: list[tuple[str, int, float, int]],
    sparse_pages: list[tuple[str, int, float, int]],
    args: argparse.Namespace,
) -> SourceWeights:
    base_dense_weight = float(args.dense_weight)
    base_sparse_weight = float(args.sparse_weight)
    mode = str(args.adaptive_source_weight_mode)
    metadata: dict[str, object] = {
        "adaptive_source_weight_mode": mode,
        "base_dense_weight": base_dense_weight,
        "base_sparse_weight": base_sparse_weight,
        "effective_dense_weight": base_dense_weight,
        "effective_sparse_weight": base_sparse_weight,
        **agreement.metadata,
    }
    if mode == "none":
        return SourceWeights(base_dense_weight, base_sparse_weight, metadata)

    if mode == "reciprocal":
        reliability = compute_reciprocal_source_reliability(
            dense_pages=dense_pages,
            sparse_pages=sparse_pages,
            args=args,
        )
        min_mult = float(args.adaptive_source_reciprocal_min_mult)
        max_mult = float(args.adaptive_source_reciprocal_max_mult)
        dense_mult = clamp(
            min_mult + (max_mult - min_mult) * reliability.dense_reliability,
            min_mult,
            max_mult,
        )
        sparse_mult = clamp(
            min_mult + (max_mult - min_mult) * reliability.sparse_reliability,
            min_mult,
            max_mult,
        )
        effective_dense_weight = base_dense_weight * dense_mult
        effective_sparse_weight = base_sparse_weight * sparse_mult
        if bool(args.adaptive_source_reciprocal_preserve_total):
            raw_total = effective_dense_weight + effective_sparse_weight
            base_total = base_dense_weight + base_sparse_weight
            if raw_total > 0 and base_total > 0:
                total_scale = base_total / raw_total
                effective_dense_weight *= total_scale
                effective_sparse_weight *= total_scale
        else:
            total_scale = 1.0
        metadata.update(
            {
                "adaptive_source_reciprocal_top_pages": reliability.source_top_pages,
                "adaptive_source_reciprocal_lookup_pages": reliability.lookup_pages,
                "adaptive_source_reciprocal_doc_weight": reliability.doc_weight,
                "adaptive_source_reciprocal_page_weight": reliability.page_weight,
                "adaptive_source_dense_page_reciprocal_support": reliability.dense_page_support,
                "adaptive_source_dense_doc_reciprocal_support": reliability.dense_doc_support,
                "adaptive_source_sparse_page_reciprocal_support": reliability.sparse_page_support,
                "adaptive_source_sparse_doc_reciprocal_support": reliability.sparse_doc_support,
                "adaptive_source_dense_reliability": reliability.dense_reliability,
                "adaptive_source_sparse_reliability": reliability.sparse_reliability,
                "adaptive_source_dense_multiplier": dense_mult,
                "adaptive_source_sparse_multiplier": sparse_mult,
                "adaptive_source_reciprocal_preserve_total": bool(
                    args.adaptive_source_reciprocal_preserve_total
                ),
                "adaptive_source_reciprocal_total_scale": total_scale,
                "effective_dense_weight": effective_dense_weight,
                "effective_sparse_weight": effective_sparse_weight,
            }
        )
        return SourceWeights(effective_dense_weight, effective_sparse_weight, metadata)

    disagreement = agreement.disagreement_score
    gamma = max(0.0, float(args.adaptive_source_gamma))
    if gamma != 1.0:
        disagreement = disagreement**gamma
    strength = max(0.0, float(args.adaptive_source_strength))
    dense_mult = clamp(
        1.0 + strength * disagreement,
        float(args.adaptive_source_min_dense_mult),
        float(args.adaptive_source_max_dense_mult),
    )
    sparse_mult = clamp(
        1.0 - strength * disagreement,
        float(args.adaptive_source_min_sparse_mult),
        float(args.adaptive_source_max_sparse_mult),
    )
    effective_dense_weight = base_dense_weight * dense_mult
    effective_sparse_weight = base_sparse_weight * sparse_mult
    metadata.update(
        {
            "adaptive_source_disagreement_score": disagreement,
            "adaptive_source_dense_multiplier": dense_mult,
            "adaptive_source_sparse_multiplier": sparse_mult,
            "effective_dense_weight": effective_dense_weight,
            "effective_sparse_weight": effective_sparse_weight,
        }
    )
    return SourceWeights(effective_dense_weight, effective_sparse_weight, metadata)


def build_expansion_query_terms(
    *,
    sparse_index: SparsePageIndex,
    dense_pages: list[tuple[str, int, float, int]],
    sparse_pages: list[tuple[str, int, float, int]],
    dense_weight: float,
    sparse_weight: float,
    args: argparse.Namespace,
) -> dict[int, float]:
    term_scores: dict[int, float] = defaultdict(float)
    seed_weights: dict[str, float] = defaultdict(float)
    for doc_id, page_idx, _score, rank in dense_pages[: max(0, int(args.expand_from_top_dense_pages))]:
        seed_weights[page_uid(doc_id, page_idx)] += float(dense_weight) / (
            float(args.rrf_k) + float(rank)
        )
    for doc_id, page_idx, _score, rank in sparse_pages[: max(0, int(args.expand_from_top_sparse_pages))]:
        seed_weights[page_uid(doc_id, page_idx)] += float(sparse_weight) / (
            float(args.rrf_k) + float(rank)
        )

    for seed_page_uid, seed_weight in seed_weights.items():
        for term_id, term_weight in sparse_index.top_terms_for_page(
            seed_page_uid,
            int(args.expansion_source_term_topk),
        ):
            term_scores[int(term_id)] += float(seed_weight) * float(term_weight)

    query_topk = int(args.expansion_query_topk_terms)
    if query_topk > 0 and len(term_scores) > query_topk:
        return dict(
            sorted(
                term_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:query_topk]
        )
    return dict(term_scores)


def build_neighbor_expansion_pages(
    *,
    dense_pages: list[tuple[str, int, float, int]],
    sparse_pages: list[tuple[str, int, float, int]],
    sparse_index: SparsePageIndex,
    dense_weight: float,
    sparse_weight: float,
    args: argparse.Namespace,
) -> tuple[list[tuple[str, int, float]], dict[str, float]]:
    window = max(0, int(args.neighbor_expansion_window))
    if window <= 0:
        return [], {}

    seed_specs = [
        (
            dense_pages[: max(0, int(args.expand_neighbors_from_top_dense_pages))],
            float(dense_weight),
        ),
        (
            sparse_pages[: max(0, int(args.expand_neighbors_from_top_sparse_pages))],
            float(sparse_weight),
        ),
    ]
    neighbor_seed: dict[str, float] = defaultdict(float)
    seed_pages = {
        (str(doc_id), int(page_idx))
        for source_pages, _source_weight in seed_specs
        for doc_id, page_idx, _score, _rank in source_pages
    }
    for source_pages, source_weight in seed_specs:
        for doc_id, page_idx, _score, rank in source_pages:
            base_contribution = source_weight * float(args.neighbor_seed_weight) / (
                float(args.rrf_k) + float(rank)
            )
            if base_contribution <= 0:
                continue
            for distance in range(1, window + 1):
                inherited = base_contribution / float(distance)
                for direction in (-1, 1):
                    neighbor_idx = int(page_idx) + direction * distance
                    if neighbor_idx < 0:
                        continue
                    if not sparse_index.has_page(str(doc_id), neighbor_idx):
                        continue
                    if (str(doc_id), neighbor_idx) in seed_pages:
                        continue
                    uid = page_uid(str(doc_id), neighbor_idx)
                    neighbor_seed[uid] += inherited

    neighbor_pages = [
        (str(doc_id), int(page_idx), float(seed_score))
        for uid, seed_score in neighbor_seed.items()
        for doc_id, page_idx in [(uid.rsplit("_page", 1)[0], int(uid.rsplit("_page", 1)[1]))]
    ]
    neighbor_pages.sort(key=lambda item: (-item[2], item[0], item[1]))
    return neighbor_pages, dict(neighbor_seed)


def build_restart_vector(
    *,
    page_seed: dict[str, float],
    doc_seed: dict[str, float],
    dense_pages: list[tuple[str, int, float, int]],
    sparse_pages: list[tuple[str, int, float, int]],
    expansion_pages: list[tuple[str, int, float, int]],
    dense_score_norm: dict[str, float],
    sparse_score_norm: dict[str, float],
    expansion_score_norm: dict[str, float],
    neighbor_seed: dict[str, float],
    dense_doc_ranks: dict[str, int],
    sparse_doc_ranks: dict[str, int],
    source_weights: SourceWeights,
    agreement: SourceAgreement,
    args: argparse.Namespace,
) -> RestartVector:
    mode = str(args.adaptive_restart_mode)
    if mode == "none":
        seed = dict(page_seed)
        for node, value in doc_seed.items():
            seed[node] = seed.get(node, 0.0) + value
        return RestartVector(
            seed=seed,
            doc_ppr_nodes=set(doc_seed),
            metadata={
                "adaptive_restart_mode": mode,
                "restart_vector_page_node_count": len(page_seed),
                "restart_vector_doc_node_count": len(doc_seed),
            },
        )

    if mode == "reciprocal":
        reliability = compute_reciprocal_source_reliability(
            dense_pages=dense_pages,
            sparse_pages=sparse_pages,
            args=args,
        )
        dense_mult = clamp(
            float(args.adaptive_restart_min_dense_mult)
            + (
                float(args.adaptive_restart_max_dense_mult)
                - float(args.adaptive_restart_min_dense_mult)
            )
            * reliability.dense_reliability,
            float(args.adaptive_restart_min_dense_mult),
            float(args.adaptive_restart_max_dense_mult),
        )
        sparse_mult = clamp(
            float(args.adaptive_restart_min_sparse_mult)
            + (
                float(args.adaptive_restart_max_sparse_mult)
                - float(args.adaptive_restart_min_sparse_mult)
            )
            * reliability.sparse_reliability,
            float(args.adaptive_restart_min_sparse_mult),
            float(args.adaptive_restart_max_sparse_mult),
        )
        restart_dense_weight = source_weights.dense_weight * dense_mult
        restart_sparse_weight = source_weights.sparse_weight * sparse_mult
        if bool(args.adaptive_restart_preserve_source_total):
            raw_total = restart_dense_weight + restart_sparse_weight
            base_total = source_weights.dense_weight + source_weights.sparse_weight
            if raw_total > 0 and base_total > 0:
                total_scale = base_total / raw_total
                restart_dense_weight *= total_scale
                restart_sparse_weight *= total_scale
            else:
                total_scale = 1.0
        else:
            total_scale = 1.0

        doc_reliability = (reliability.dense_reliability + reliability.sparse_reliability) / 2.0
        doc_mult = clamp(
            float(args.adaptive_restart_min_doc_mult)
            + (float(args.adaptive_restart_max_doc_mult) - float(args.adaptive_restart_min_doc_mult))
            * doc_reliability,
            float(args.adaptive_restart_min_doc_mult),
            float(args.adaptive_restart_max_doc_mult),
        )
        restart_doc_seed_weight = float(args.adaptive_restart_doc_seed_weight) * doc_mult
        restart_page_seed_weight = float(args.adaptive_restart_page_seed_weight)

        seed: dict[str, float] = defaultdict(float)
        for source_weight, source_pages, source_score_norm in [
            (restart_dense_weight, dense_pages, dense_score_norm),
            (restart_sparse_weight, sparse_pages, sparse_score_norm),
            (float(args.expansion_weight), expansion_pages, expansion_score_norm),
        ]:
            for doc_id, page_idx, _score, rank in source_pages:
                uid = page_uid(doc_id, page_idx)
                seed[uid] += restart_page_seed_weight * source_weight / (
                    float(args.rrf_k) + float(rank)
                )
                seed[uid] += (
                    restart_page_seed_weight
                    * source_weight
                    * float(args.score_seed_weight)
                    * source_score_norm.get(uid, 0.0)
                )
        for uid, seed_score in neighbor_seed.items():
            seed[uid] += (
                restart_page_seed_weight
                * float(args.adaptive_restart_neighbor_seed_weight)
                * float(seed_score)
            )

        doc_ppr_nodes: set[str] = set()
        if restart_doc_seed_weight > 0:
            for doc_id, rank in dense_doc_ranks.items():
                node = f"doc::{doc_id}"
                seed[node] += (
                    restart_doc_seed_weight
                    * restart_dense_weight
                    / (float(args.rrf_k) + float(rank))
                )
                doc_ppr_nodes.add(node)
            for doc_id, rank in sparse_doc_ranks.items():
                node = f"doc::{doc_id}"
                seed[node] += (
                    restart_doc_seed_weight
                    * restart_sparse_weight
                    / (float(args.rrf_k) + float(rank))
                )
                doc_ppr_nodes.add(node)

        return RestartVector(
            seed=dict(seed),
            doc_ppr_nodes=doc_ppr_nodes,
            metadata={
                "adaptive_restart_mode": mode,
                "adaptive_restart_reciprocal_top_pages": reliability.source_top_pages,
                "adaptive_restart_reciprocal_lookup_pages": reliability.lookup_pages,
                "adaptive_restart_reciprocal_doc_weight": reliability.doc_weight,
                "adaptive_restart_reciprocal_page_weight": reliability.page_weight,
                "adaptive_restart_dense_page_reciprocal_support": reliability.dense_page_support,
                "adaptive_restart_dense_doc_reciprocal_support": reliability.dense_doc_support,
                "adaptive_restart_sparse_page_reciprocal_support": reliability.sparse_page_support,
                "adaptive_restart_sparse_doc_reciprocal_support": reliability.sparse_doc_support,
                "adaptive_restart_dense_reliability": reliability.dense_reliability,
                "adaptive_restart_sparse_reliability": reliability.sparse_reliability,
                "adaptive_restart_dense_multiplier": dense_mult,
                "adaptive_restart_sparse_multiplier": sparse_mult,
                "adaptive_restart_doc_multiplier": doc_mult,
                "adaptive_restart_preserve_source_total": bool(
                    args.adaptive_restart_preserve_source_total
                ),
                "adaptive_restart_total_scale": total_scale,
                "effective_restart_dense_weight": restart_dense_weight,
                "effective_restart_sparse_weight": restart_sparse_weight,
                "effective_restart_doc_seed_weight": restart_doc_seed_weight,
                "restart_vector_page_node_count": sum(
                    1 for node in seed if not node.startswith("doc::")
                ),
                "restart_vector_doc_node_count": len(doc_ppr_nodes),
            },
        )

    disagreement = agreement.disagreement_score
    gamma = max(0.0, float(args.adaptive_restart_gamma))
    if gamma != 1.0:
        disagreement = disagreement**gamma
    strength = max(0.0, float(args.adaptive_restart_source_strength))
    dense_mult = clamp(
        1.0 + strength * disagreement,
        float(args.adaptive_restart_min_dense_mult),
        float(args.adaptive_restart_max_dense_mult),
    )
    sparse_mult = clamp(
        1.0 - strength * disagreement,
        float(args.adaptive_restart_min_sparse_mult),
        float(args.adaptive_restart_max_sparse_mult),
    )
    doc_mult = clamp(
        float(args.adaptive_restart_min_doc_mult)
        + (float(args.adaptive_restart_max_doc_mult) - float(args.adaptive_restart_min_doc_mult))
        * agreement.agreement_score,
        float(args.adaptive_restart_min_doc_mult),
        float(args.adaptive_restart_max_doc_mult),
    )
    restart_dense_weight = source_weights.dense_weight * dense_mult
    restart_sparse_weight = source_weights.sparse_weight * sparse_mult
    restart_doc_seed_weight = float(args.adaptive_restart_doc_seed_weight) * doc_mult
    restart_page_seed_weight = float(args.adaptive_restart_page_seed_weight)

    seed: dict[str, float] = defaultdict(float)
    for source_weight, source_pages, source_score_norm in [
        (restart_dense_weight, dense_pages, dense_score_norm),
        (restart_sparse_weight, sparse_pages, sparse_score_norm),
        (float(args.expansion_weight), expansion_pages, expansion_score_norm),
    ]:
        for doc_id, page_idx, _score, rank in source_pages:
            uid = page_uid(doc_id, page_idx)
            seed[uid] += restart_page_seed_weight * source_weight / (
                float(args.rrf_k) + float(rank)
            )
            seed[uid] += (
                restart_page_seed_weight
                * source_weight
                * float(args.score_seed_weight)
                * source_score_norm.get(uid, 0.0)
            )
    for uid, seed_score in neighbor_seed.items():
        seed[uid] += (
            restart_page_seed_weight
            * float(args.adaptive_restart_neighbor_seed_weight)
            * float(seed_score)
        )

    doc_ppr_nodes: set[str] = set()
    if restart_doc_seed_weight > 0:
        for doc_id, rank in dense_doc_ranks.items():
            node = f"doc::{doc_id}"
            seed[node] += (
                restart_doc_seed_weight
                * restart_dense_weight
                / (float(args.rrf_k) + float(rank))
            )
            doc_ppr_nodes.add(node)
        for doc_id, rank in sparse_doc_ranks.items():
            node = f"doc::{doc_id}"
            seed[node] += (
                restart_doc_seed_weight
                * restart_sparse_weight
                / (float(args.rrf_k) + float(rank))
            )
            doc_ppr_nodes.add(node)

    return RestartVector(
        seed=dict(seed),
        doc_ppr_nodes=doc_ppr_nodes,
        metadata={
            "adaptive_restart_mode": mode,
            "adaptive_restart_agreement_score": agreement.agreement_score,
            "adaptive_restart_disagreement_score": disagreement,
            "adaptive_restart_dense_multiplier": dense_mult,
            "adaptive_restart_sparse_multiplier": sparse_mult,
            "adaptive_restart_doc_multiplier": doc_mult,
            "effective_restart_dense_weight": restart_dense_weight,
            "effective_restart_sparse_weight": restart_sparse_weight,
            "effective_restart_doc_seed_weight": restart_doc_seed_weight,
            "restart_vector_page_node_count": sum(1 for node in seed if not node.startswith("doc::")),
            "restart_vector_doc_node_count": len(doc_ppr_nodes),
        },
    )


def record_reciprocal_transition_reliability(
    *,
    record: PageRecord,
    dense_doc_ranks: dict[str, int],
    sparse_doc_ranks: dict[str, int],
    args: argparse.Namespace,
) -> float:
    doc_weight = max(0.0, float(args.adaptive_source_reciprocal_doc_weight))
    page_weight = max(0.0, float(args.adaptive_source_reciprocal_page_weight))
    denominator = doc_weight + page_weight
    if denominator <= 0:
        return 0.0

    values: list[float] = []
    if record.dense_rank is not None:
        page_support = reciprocal_rank_value(record.sparse_rank, float(args.rrf_k))
        doc_support = reciprocal_rank_value(sparse_doc_ranks.get(record.doc_id), float(args.rrf_k))
        values.append((doc_weight * doc_support + page_weight * page_support) / denominator)
    if record.sparse_rank is not None:
        page_support = reciprocal_rank_value(record.dense_rank, float(args.rrf_k))
        doc_support = reciprocal_rank_value(dense_doc_ranks.get(record.doc_id), float(args.rrf_k))
        values.append((doc_weight * doc_support + page_weight * page_support) / denominator)
    if record.expansion_rank is not None and not values:
        values.append(0.5)
    if record.neighbor_seed_score > 0 and not values:
        values.append(0.5)
    if not values:
        return 0.0
    return clamp(statistics.fmean(values), 0.0, 1.0)


def build_transition_policy(
    *,
    records: dict[str, PageRecord],
    agreement: SourceAgreement,
    dense_doc_ranks: dict[str, int],
    sparse_doc_ranks: dict[str, int],
    args: argparse.Namespace,
) -> TransitionPolicy:
    mode = str(args.adaptive_transition_mode)
    global_multiplier = 1.0
    page_doc_multipliers = {uid: 1.0 for uid in records}
    metadata: dict[str, object] = {
        "adaptive_transition_mode": mode,
        "adaptive_transition_gate_adjacent": bool(args.adaptive_transition_gate_adjacent),
        "adaptive_transition_gate_page_doc": bool(args.adaptive_transition_gate_page_doc),
        "adaptive_transition_gate_page_to_doc": bool(args.adaptive_transition_gate_page_to_doc),
        "adaptive_transition_gate_doc_to_page": bool(args.adaptive_transition_gate_doc_to_page),
        "adaptive_transition_global_multiplier": 1.0,
        "mean_adaptive_transition_page_doc_multiplier": 1.0 if records else None,
    }
    if mode == "none":
        return TransitionPolicy(page_doc_multipliers, global_multiplier, metadata)

    agreement_min = float(args.adaptive_transition_agreement_min_mult)
    agreement_max = float(args.adaptive_transition_agreement_max_mult)
    global_multiplier = clamp(
        agreement_min + (agreement_max - agreement_min) * agreement.agreement_score,
        agreement_min,
        agreement_max,
    )
    if mode == "agreement":
        page_doc_multipliers = {uid: global_multiplier for uid in records}
        metadata.update(
            {
                "adaptive_transition_agreement_min_mult": agreement_min,
                "adaptive_transition_agreement_max_mult": agreement_max,
                "adaptive_transition_global_multiplier": global_multiplier,
                "mean_adaptive_transition_page_doc_multiplier": global_multiplier if records else None,
                "mean_adaptive_transition_local_reliability": None,
            }
        )
        return TransitionPolicy(page_doc_multipliers, global_multiplier, metadata)

    local_min = float(args.adaptive_transition_local_min_mult)
    local_max = float(args.adaptive_transition_local_max_mult)
    local_weight = clamp(float(args.adaptive_transition_local_weight), 0.0, 1.0)
    local_reliabilities: dict[str, float] = {}
    multipliers: dict[str, float] = {}
    for uid, record in records.items():
        local_reliability = record_reciprocal_transition_reliability(
            record=record,
            dense_doc_ranks=dense_doc_ranks,
            sparse_doc_ranks=sparse_doc_ranks,
            args=args,
        )
        local_reliabilities[uid] = local_reliability
        local_multiplier = clamp(
            local_min + (local_max - local_min) * local_reliability,
            local_min,
            local_max,
        )
        multipliers[uid] = (
            (1.0 - local_weight) * global_multiplier
            + local_weight * local_multiplier
        )
    metadata.update(
        {
            "adaptive_transition_agreement_min_mult": agreement_min,
            "adaptive_transition_agreement_max_mult": agreement_max,
            "adaptive_transition_local_min_mult": local_min,
            "adaptive_transition_local_max_mult": local_max,
            "adaptive_transition_local_weight": local_weight,
            "adaptive_transition_global_multiplier": global_multiplier,
            "mean_adaptive_transition_local_reliability": (
                statistics.fmean(local_reliabilities.values()) if local_reliabilities else None
            ),
            "mean_adaptive_transition_page_doc_multiplier": (
                statistics.fmean(multipliers.values()) if multipliers else None
            ),
        }
    )
    return TransitionPolicy(multipliers, global_multiplier, metadata)


def build_adjacent_coherence_policy(
    *,
    records: dict[str, PageRecord],
    dense_doc_ranks: dict[str, int],
    sparse_doc_ranks: dict[str, int],
    args: argparse.Namespace,
) -> AdjacentCoherencePolicy:
    mode = str(args.adaptive_adjacent_mode)
    page_reliabilities = {uid: 1.0 for uid in records}
    metadata: dict[str, object] = {
        "adaptive_adjacent_mode": mode,
        "adaptive_adjacent_min_mult": float(args.adaptive_adjacent_min_mult),
        "adaptive_adjacent_max_mult": float(args.adaptive_adjacent_max_mult),
        "adaptive_adjacent_power": float(args.adaptive_adjacent_power),
        "mean_adaptive_adjacent_page_reliability": 1.0 if records else None,
    }
    if mode == "none":
        return AdjacentCoherencePolicy(page_reliabilities=page_reliabilities, metadata=metadata)

    reliabilities = {
        uid: record_reciprocal_transition_reliability(
            record=record,
            dense_doc_ranks=dense_doc_ranks,
            sparse_doc_ranks=sparse_doc_ranks,
            args=args,
        )
        for uid, record in records.items()
    }
    metadata["mean_adaptive_adjacent_page_reliability"] = (
        statistics.fmean(reliabilities.values()) if reliabilities else None
    )
    return AdjacentCoherencePolicy(page_reliabilities=reliabilities, metadata=metadata)


def adjacent_edge_multiplier(
    *,
    left_uid: str,
    right_uid: str,
    policy: AdjacentCoherencePolicy,
    args: argparse.Namespace,
) -> float:
    if str(args.adaptive_adjacent_mode) == "none":
        return 1.0
    left_reliability = clamp(policy.page_reliabilities.get(left_uid, 0.0), 0.0, 1.0)
    right_reliability = clamp(policy.page_reliabilities.get(right_uid, 0.0), 0.0, 1.0)
    power = max(0.0, float(args.adaptive_adjacent_power))
    coherence = clamp((left_reliability * right_reliability) ** power, 0.0, 1.0)
    return clamp(
        float(args.adaptive_adjacent_min_mult)
        + (float(args.adaptive_adjacent_max_mult) - float(args.adaptive_adjacent_min_mult))
        * coherence,
        float(args.adaptive_adjacent_min_mult),
        float(args.adaptive_adjacent_max_mult),
    )


def build_evidence_community_policy(
    *,
    records: dict[str, PageRecord],
    pages_by_doc: dict[str, list[PageRecord]],
    source_weights: SourceWeights,
    args: argparse.Namespace,
) -> EvidenceCommunityPolicy:
    mode = str(args.evidence_community_mode)
    raw_source_support: dict[str, float] = {}
    for uid, record in records.items():
        dense_component = source_weights.dense_weight * reciprocal_rank_value(
            record.dense_rank,
            float(args.rrf_k),
        )
        sparse_component = source_weights.sparse_weight * reciprocal_rank_value(
            record.sparse_rank,
            float(args.rrf_k),
        )
        both_bonus = (
            float(args.evidence_community_both_source_bonus)
            if record.dense_rank is not None and record.sparse_rank is not None
            else 0.0
        )
        raw_source_support[uid] = dense_component + sparse_component + both_bonus

    page_source_support = max_scale(raw_source_support)
    raw_local_support: dict[str, float] = {uid: 0.0 for uid in records}
    local_window = max(0, int(args.evidence_community_local_window))
    if local_window > 0:
        for doc_records in pages_by_doc.values():
            by_page_idx = {int(record.page_idx): record for record in doc_records}
            for record in doc_records:
                support = 0.0
                for delta in range(1, local_window + 1):
                    for neighbor_idx in (int(record.page_idx) - delta, int(record.page_idx) + delta):
                        neighbor = by_page_idx.get(neighbor_idx)
                        if neighbor is None:
                            continue
                        support += raw_source_support.get(neighbor.page_uid, 0.0) / float(delta + 1)
                raw_local_support[record.page_uid] = support
    page_local_support = max_scale(raw_local_support)
    metadata: dict[str, object] = {
        "evidence_community_mode": mode,
        "evidence_community_source_edge_weight": float(args.evidence_community_source_edge_weight),
        "evidence_community_local_edge_weight": float(args.evidence_community_local_edge_weight),
        "evidence_community_min_page_support": float(args.evidence_community_min_page_support),
        "evidence_community_local_window": int(args.evidence_community_local_window),
        "evidence_community_both_source_bonus": float(args.evidence_community_both_source_bonus),
        "evidence_community_support_power": float(args.evidence_community_support_power),
        "mean_evidence_community_source_support": (
            statistics.fmean(page_source_support.values()) if page_source_support else None
        ),
        "mean_evidence_community_local_support": (
            statistics.fmean(page_local_support.values()) if page_local_support else None
        ),
    }
    return EvidenceCommunityPolicy(
        page_source_support=page_source_support,
        page_local_support=page_local_support,
        metadata=metadata,
    )


def add_evidence_community_edges(
    *,
    graph: dict[str, dict[str, float]],
    records: dict[str, PageRecord],
    policy: EvidenceCommunityPolicy,
    args: argparse.Namespace,
) -> dict[str, object]:
    mode = str(args.evidence_community_mode)
    if mode == "none":
        return {
            "evidence_community_node_count": 0,
            "evidence_community_edge_count_directed": 0,
            "evidence_community_source_edge_count_directed": 0,
            "evidence_community_local_edge_count_directed": 0,
        }

    enabled_source = mode in {"source", "source_local"}
    enabled_local = mode in {"local", "source_local"}
    min_support = clamp(float(args.evidence_community_min_page_support), 0.0, 1.0)
    support_power = max(0.0, float(args.evidence_community_support_power))
    community_nodes: set[str] = set()
    source_edge_count = 0
    local_edge_count = 0

    for uid, record in records.items():
        if enabled_source:
            support = policy.page_source_support.get(uid, 0.0)
            if support >= min_support:
                weighted_support = support**support_power if support_power != 1.0 else support
                node = f"community::source::{record.doc_id}"
                community_nodes.add(node)
                edge_weight = float(args.evidence_community_source_edge_weight) * weighted_support
                if edge_weight > 0:
                    add_undirected_edge(graph, uid, node, edge_weight)
                    source_edge_count += 2
        if enabled_local:
            support = policy.page_local_support.get(uid, 0.0)
            if support >= min_support:
                weighted_support = support**support_power if support_power != 1.0 else support
                node = f"community::local::{record.doc_id}"
                community_nodes.add(node)
                edge_weight = float(args.evidence_community_local_edge_weight) * weighted_support
                if edge_weight > 0:
                    add_undirected_edge(graph, uid, node, edge_weight)
                    local_edge_count += 2

    return {
        "evidence_community_node_count": len(community_nodes),
        "evidence_community_edge_count_directed": source_edge_count + local_edge_count,
        "evidence_community_source_edge_count_directed": source_edge_count,
        "evidence_community_local_edge_count_directed": local_edge_count,
    }


def add_query_role(role_weights: dict[str, float], role: str, weight: float) -> None:
    if weight <= 0:
        return
    role_weights[role] = max(role_weights.get(role, 0.0), float(weight))


def add_explicit_position_page(
    *,
    role_weights: dict[str, float],
    explicit_page_numbers: list[int],
    raw_page: int,
) -> None:
    if raw_page < 0:
        return
    if raw_page not in explicit_page_numbers:
        explicit_page_numbers.append(raw_page)
    target_idx = raw_page - 1 if raw_page > 0 else 0
    add_query_role(role_weights, f"page_{target_idx}", 1.0)


def extract_position_roles_from_question(question: str) -> tuple[dict[str, float], list[int], list[int]]:
    query = str(question or "").lower()
    role_weights: dict[str, float] = {}

    if re.search(FIRST_COVER_REFERENCE_RE, query):
        add_query_role(role_weights, "first", 1.0)
        add_query_role(role_weights, "early", 0.5)
    if re.search(r"\btable\s+of\s+contents\b|\bcontents?\s+page\b", query):
        add_query_role(role_weights, "early", 0.75)
    if re.search(
        r"\b(last|final|ending|back)\s+page\b|\bend\s+of\s+(the\s+)?(document|report|paper)\b",
        query,
    ):
        add_query_role(role_weights, "last", 1.0)
        add_query_role(role_weights, "late", 0.5)
    if re.search(r"\b(signature|signatures|signed|leadership\s+signature)\b", query):
        add_query_role(role_weights, "last", 0.75)
        add_query_role(role_weights, "late", 1.0)
    if re.search(
        r"\b(how\s+many\s+pages|number\s+of\s+pages|page\s+count|total\s+pages)\b",
        query,
    ):
        add_query_role(role_weights, "last", 1.0)
        add_query_role(role_weights, "late", 0.5)
    if re.search(r"\b(references|bibliography|appendix|appendices|acknowledg(e)?ments?)\b", query):
        add_query_role(role_weights, "late", 1.0)

    explicit_page_numbers: list[int] = []
    for match in re.finditer(PAGE_REFERENCE_RANGE_RE, query):
        start_page = int(match.group(1))
        end_page = int(match.group(2))
        lo, hi = sorted((start_page, end_page))
        if hi - lo > 20:
            continue
        for raw_page in range(lo, hi + 1):
            add_explicit_position_page(
                role_weights=role_weights,
                explicit_page_numbers=explicit_page_numbers,
                raw_page=raw_page,
            )

    for match in re.finditer(PAGE_REFERENCE_SINGLE_RE, query):
        raw_page = int(match.group(1))
        add_explicit_position_page(
            role_weights=role_weights,
            explicit_page_numbers=explicit_page_numbers,
            raw_page=raw_page,
        )

    explicit_page_indices = [raw_page - 1 if raw_page > 0 else 0 for raw_page in explicit_page_numbers]
    return role_weights, explicit_page_indices, explicit_page_numbers


def record_position_roles(
    *,
    record: PageRecord,
    doc_page_counts: dict[str, int],
    doc_page_number_indices: dict[str, dict[int, set[int]]],
    early_frac: float,
    late_frac: float,
    explicit_page_numbers: set[int],
) -> set[str]:
    roles: set[str] = set()
    page_idx = int(record.page_idx)
    if page_idx == 0:
        roles.add("first")
        roles.add("early")

    page_count = int(doc_page_counts.get(record.doc_id, 0))
    if page_count > 0:
        last_idx = max(0, page_count - 1)
        early_cutoff = max(0, int(last_idx * clamp(early_frac, 0.0, 1.0)))
        late_start = min(
            last_idx,
            max(0, int((last_idx + 1) * (1.0 - clamp(late_frac, 0.0, 1.0)))),
        )
        if page_idx <= early_cutoff:
            roles.add("early")
        if page_idx == last_idx:
            roles.add("last")
        if page_idx >= late_start:
            roles.add("late")

    page_number_lookup = doc_page_number_indices.get(record.doc_id, {})
    for raw_page in explicit_page_numbers:
        target_idx = raw_page - 1 if raw_page > 0 else 0
        if page_idx == target_idx or page_idx in page_number_lookup.get(raw_page, set()):
            roles.add(f"page_{target_idx}")
    return roles


def build_position_evidence_policy(
    *,
    question: str,
    records: dict[str, PageRecord],
    doc_page_counts: dict[str, int],
    doc_page_number_indices: dict[str, dict[int, set[int]]],
    doc_position_weights: dict[str, float],
    args: argparse.Namespace,
) -> PositionEvidencePolicy:
    mode = str(args.position_evidence_mode)
    active_role_weights: dict[str, float] = {}
    explicit_page_indices: list[int] = []
    explicit_page_numbers: list[int] = []
    role_page_weights: dict[str, dict[str, float]] = {}
    if mode != "none":
        active_role_weights, explicit_page_indices, explicit_page_numbers = (
            extract_position_roles_from_question(question)
        )
        explicit_number_set = set(explicit_page_numbers)
        early_frac = float(args.position_evidence_early_frac)
        late_frac = float(args.position_evidence_late_frac)
        scope = str(args.position_evidence_scope)
        min_doc_support = clamp(float(args.position_evidence_min_doc_support), 0.0, 1.0)
        for uid, record in records.items():
            doc_support = 1.0
            if scope == "doc_conditioned":
                doc_support = doc_position_weights.get(record.doc_id, 0.0)
                if doc_support < min_doc_support or doc_support <= 0:
                    continue
            page_roles = record_position_roles(
                record=record,
                doc_page_counts=doc_page_counts,
                doc_page_number_indices=doc_page_number_indices,
                early_frac=early_frac,
                late_frac=late_frac,
                explicit_page_numbers=explicit_number_set,
            )
            for role in page_roles:
                if role in active_role_weights:
                    role_page_weights.setdefault(role, {})[uid] = doc_support

    page_match_count = sum(len(pages) for pages in role_page_weights.values())
    metadata: dict[str, object] = {
        "position_evidence_mode": mode,
        "position_evidence_scope": str(args.position_evidence_scope),
        "position_evidence_active_roles": sorted(active_role_weights),
        "position_evidence_active_role_count": len(active_role_weights),
        "position_evidence_explicit_page_indices": explicit_page_indices,
        "position_evidence_explicit_page_numbers": explicit_page_numbers,
        "position_evidence_role_node_count": len(role_page_weights),
        "position_evidence_page_match_count": page_match_count,
        "position_evidence_doc_support_count": len(doc_position_weights),
        "position_evidence_doc_top_k": int(args.position_evidence_doc_top_k),
        "position_evidence_min_doc_support": float(args.position_evidence_min_doc_support),
        "mean_position_evidence_doc_support": (
            statistics.fmean(doc_position_weights.values()) if doc_position_weights else None
        ),
        "position_evidence_doc_page_count_available": bool(doc_page_counts),
        "position_evidence_edge_weight": float(args.position_evidence_edge_weight),
        "position_evidence_restart_weight": float(args.position_evidence_restart_weight),
        "position_evidence_explicit_page_edge_weight": float(
            args.position_evidence_explicit_page_edge_weight
        ),
        "position_evidence_explicit_page_restart_weight": float(
            args.position_evidence_explicit_page_restart_weight
        ),
        "position_evidence_early_frac": float(args.position_evidence_early_frac),
        "position_evidence_late_frac": float(args.position_evidence_late_frac),
    }
    return PositionEvidencePolicy(
        active_role_weights=active_role_weights,
        role_page_weights=role_page_weights,
        metadata=metadata,
    )


def is_explicit_position_role(role: str) -> bool:
    return role.startswith("page_")


def build_position_doc_weights(
    *,
    dense_doc_ranks: dict[str, int],
    sparse_doc_ranks: dict[str, int],
    source_weights: SourceWeights,
    args: argparse.Namespace,
) -> dict[str, float]:
    raw_doc_weights: dict[str, float] = defaultdict(float)
    for doc_id, rank in dense_doc_ranks.items():
        raw_doc_weights[doc_id] += source_weights.dense_weight / (
            float(args.rrf_k) + float(rank)
        )
    for doc_id, rank in sparse_doc_ranks.items():
        raw_doc_weights[doc_id] += source_weights.sparse_weight / (
            float(args.rrf_k) + float(rank)
        )
    if not raw_doc_weights:
        return {}
    top_k = max(0, int(args.position_evidence_doc_top_k))
    sorted_docs = sorted(
        raw_doc_weights.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if top_k > 0:
        sorted_docs = sorted_docs[:top_k]
    return max_scale(dict(sorted_docs))


def add_position_evidence_edges(
    *,
    graph: dict[str, dict[str, float]],
    policy: PositionEvidencePolicy,
    args: argparse.Namespace,
) -> dict[str, object]:
    if str(args.position_evidence_mode) == "none":
        return {
            "position_evidence_edge_count_directed": 0,
            "position_evidence_seed_node_count": 0,
        }

    edge_count = 0
    seeded_node_count = 0
    for role, pages in policy.role_page_weights.items():
        if not pages:
            continue
        role_weight = policy.active_role_weights.get(role, 0.0)
        if role_weight <= 0:
            continue
        base_edge_weight = (
            float(args.position_evidence_explicit_page_edge_weight)
            if is_explicit_position_role(role)
            else float(args.position_evidence_edge_weight)
        )
        role_node = f"query_position::{role}"
        seeded_node_count += 1
        for uid, page_weight in pages.items():
            edge_weight = base_edge_weight * role_weight * float(page_weight)
            if edge_weight > 0:
                add_directed_edge(graph, role_node, uid, edge_weight)
                edge_count += 1

    return {
        "position_evidence_edge_count_directed": edge_count,
        "position_evidence_seed_node_count": seeded_node_count,
    }


def add_position_evidence_restart(
    *,
    seed: dict[str, float],
    policy: PositionEvidencePolicy,
    args: argparse.Namespace,
) -> int:
    if str(args.position_evidence_mode) == "none":
        return 0
    added = 0
    for role, pages in policy.role_page_weights.items():
        if not pages:
            continue
        role_weight = policy.active_role_weights.get(role, 0.0)
        if role_weight <= 0:
            continue
        restart_weight = (
            float(args.position_evidence_explicit_page_restart_weight)
            if is_explicit_position_role(role)
            else float(args.position_evidence_restart_weight)
        )
        if restart_weight <= 0:
            continue
        seed[f"query_position::{role}"] = seed.get(f"query_position::{role}", 0.0) + (
            restart_weight * role_weight
        )
        added += 1
    return added


QUERY_ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "list",
    "many",
    "of",
    "on",
    "or",
    "page",
    "pages",
    "please",
    "report",
    "section",
    "the",
    "there",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "write",
}

FINANCIAL_METRIC_TERMS = {
    "asset",
    "assets",
    "capitalization",
    "cash",
    "compensation",
    "cost",
    "costs",
    "debt",
    "depreciation",
    "dividend",
    "dividends",
    "ebitda",
    "emission",
    "emissions",
    "employee",
    "employees",
    "expense",
    "expenses",
    "flow",
    "ghg",
    "gross",
    "income",
    "interest",
    "liabilities",
    "liability",
    "margin",
    "market",
    "operating",
    "profit",
    "purchase",
    "ratio",
    "repurchase",
    "revenue",
    "revenues",
    "sales",
    "share",
    "shareholder",
    "shareholders",
    "shares",
    "stock",
    "tax",
    "total",
}

FINANCIAL_METRIC_PHRASES = {
    "board of directors",
    "cash flow",
    "common stock",
    "core revenue growth",
    "cumulative shareholder return",
    "days payable outstanding",
    "dividend income",
    "effective tax rate",
    "executive compensation",
    "fixed asset turnover",
    "full-time employees",
    "ghg emissions",
    "gross margin",
    "gross margin percentage",
    "gross profit",
    "interest expense",
    "long-term debt",
    "market capitalization",
    "net discrete tax gains",
    "net income",
    "net revenues",
    "operating cash flow",
    "operating expenses",
    "operating profit margin",
    "repurchasing of common stock",
    "return on equity",
    "sales and revenues",
    "shareholder return",
    "total assets",
    "total debt",
    "total debt ratio",
    "total ghg emissions",
    "total liabilities",
    "total revenue",
}

FINANCIAL_REASONING_TRIGGERS = FINANCIAL_METRIC_TERMS | {
    "fy",
    "fiscal",
    "million",
    "billion",
    "percentage",
    "percent",
    "round",
    "year",
}

FINANCIAL_ENTITY_STOPWORDS = QUERY_ANCHOR_STOPWORDS | FINANCIAL_METRIC_TERMS | {
    "answer",
    "before",
    "company",
    "companies",
    "financial",
    "fiscal",
    "long-term",
    "round",
    "year",
}


def normalize_query_anchor(anchor: str) -> str:
    normalized = re.sub(r"\s+", " ", str(anchor or "").strip(" \t\n\r,.;:!?()[]{}\"'`"))
    normalized = normalized.replace("’", "'")
    if normalized.lower().endswith("'s"):
        normalized = normalized[:-2]
    return normalized.strip()


def add_query_anchor(anchors: list[str], anchor: str, *, min_len: int) -> None:
    normalized = normalize_query_anchor(anchor)
    if not normalized:
        return
    lower = normalized.lower()
    if len(lower) < min_len and not re.search(r"\d", lower):
        return
    if lower in QUERY_ANCHOR_STOPWORDS:
        return
    if lower not in {item.lower() for item in anchors}:
        anchors.append(normalized)


def extract_query_anchors(question: str, *, max_anchors: int, min_entity_len: int) -> list[str]:
    query = str(question or "")
    anchors: list[str] = []

    for match in re.finditer(r"['\"]([^'\"]{3,80})['\"]", query):
        add_query_anchor(anchors, match.group(1), min_len=min_entity_len)

    numeric_pattern = (
        r"\b(?:FY)?\d{4}\b"
        r"|\b\d+(?:\.\d+)?\s*(?:%|percent|percentage|million|billion|thousand)\b"
        r"|\b\d+\.\d+\b"
    )
    for match in re.finditer(numeric_pattern, query, flags=re.IGNORECASE):
        add_query_anchor(anchors, match.group(0), min_len=1)

    entity_token = r"[A-Z][A-Za-z0-9&./+-]*(?:'[sS])?"
    entity_phrase_pattern = rf"\b{entity_token}(?:\s+{entity_token}){{0,4}}\b"
    for match in re.finditer(entity_phrase_pattern, query):
        phrase = normalize_query_anchor(match.group(0))
        if not phrase:
            continue
        tokens = [normalize_query_anchor(token) for token in phrase.split()]
        tokens = [token for token in tokens if token and token.lower() not in QUERY_ANCHOR_STOPWORDS]
        if not tokens:
            continue
        if len(tokens) > 1:
            add_query_anchor(anchors, " ".join(tokens), min_len=min_entity_len)
        for token in tokens:
            if re.search(r"[A-Z].*[A-Z]|\d", token) or len(token) >= max(4, min_entity_len):
                add_query_anchor(anchors, token, min_len=min_entity_len)
        if len(anchors) >= max_anchors:
            break

    return anchors[: max(0, int(max_anchors))]


def anchor_matches_page_text(anchor: str, page_text: str) -> bool:
    if not anchor or not page_text:
        return False
    normalized_anchor = normalize_query_anchor(anchor).lower()
    if not normalized_anchor:
        return False
    if re.match(r"^[a-z0-9_.+-]+$", normalized_anchor):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_anchor)}(?![a-z0-9])",
                page_text,
            )
        )
    return normalized_anchor in page_text


def query_has_financial_reasoning_cue(question: str) -> bool:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9&./+-]*", str(question or ""))
    }
    return bool(tokens & FINANCIAL_REASONING_TRIGGERS)


def add_financial_slot_value(
    slots: dict[str, list[str]],
    seen: dict[str, set[str]],
    slot: str,
    value: str,
) -> None:
    normalized = normalize_query_anchor(value)
    lower = normalized.lower()
    if normalized and lower not in seen[slot]:
        slots[slot].append(normalized)
        seen[slot].add(lower)


def dedupe_specific_financial_values(values: list[str]) -> list[str]:
    ordered = sorted(
        [normalize_query_anchor(value) for value in values if normalize_query_anchor(value)],
        key=lambda value: (-len(value.split()), -len(value), value.lower()),
    )
    kept: list[str] = []
    for value in ordered:
        lower = value.lower()
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(lower)}(?![a-z0-9])", other.lower())
            for other in kept
            if other.lower() != lower
        ):
            continue
        kept.append(value)
    return kept


def extract_financial_year_values(question: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\bFY\s*'?(\d{2}|\d{4})\b", str(question or ""), flags=re.IGNORECASE):
        raw_digits = match.group(1)
        normalized = f"FY{raw_digits}"
        for value in [normalized, f"20{raw_digits}" if len(raw_digits) == 2 else raw_digits]:
            if value.lower() not in seen:
                values.append(value)
                seen.add(value.lower())
    for match in re.finditer(r"\b20\d{2}\b", str(question or "")):
        value = match.group(0)
        if value.lower() not in seen:
            values.append(value)
            seen.add(value.lower())
    return values


def extract_financial_metric_values(question: str) -> list[str]:
    query = str(question or "")
    query_lower = query.lower()
    values: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = normalize_query_anchor(value)
        lower = normalized.lower()
        if normalized and lower not in seen:
            values.append(normalized)
            seen.add(lower)

    for phrase in sorted(FINANCIAL_METRIC_PHRASES, key=lambda item: (-len(item.split()), item)):
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", query_lower):
            add(phrase)

    tokens = re.findall(r"[a-z0-9&./+-]+", query_lower)
    stopwords = FINANCIAL_ENTITY_STOPWORDS | {"did", "does", "much", "many"}
    for ngram_len in range(4, 0, -1):
        for start in range(0, max(0, len(tokens) - ngram_len + 1)):
            ngram = tokens[start : start + ngram_len]
            if any(re.fullmatch(r"(?:fy)?\d{2,4}", token) for token in ngram):
                continue
            if not (set(ngram) & FINANCIAL_METRIC_TERMS):
                continue
            while ngram and ngram[0] in stopwords:
                ngram = ngram[1:]
            while ngram and ngram[-1] in stopwords:
                ngram = ngram[:-1]
            if not ngram or not (set(ngram) & FINANCIAL_METRIC_TERMS):
                continue
            if len(ngram) == 1 and ngram[0] in {"total", "gross", "net", "operating"}:
                continue
            add(" ".join(ngram))

    return dedupe_specific_financial_values(values)[:12]


def financial_anchor_slots(question: str, anchors: list[str]) -> dict[str, list[str]]:
    slots: dict[str, list[str]] = {"metric": [], "year": [], "entity": []}
    seen: dict[str, set[str]] = {key: set() for key in slots}
    for value in extract_financial_metric_values(question):
        add_financial_slot_value(slots, seen, "metric", value)
    for value in extract_financial_year_values(question):
        add_financial_slot_value(slots, seen, "year", value)
    for anchor in anchors:
        normalized = normalize_query_anchor(anchor)
        lower = normalized.lower()
        if not normalized:
            continue
        anchor_tokens = set(re.findall(r"[a-z][a-z0-9&./+-]*", lower))
        if re.fullmatch(r"(?:fy\s*'?(\d{2}|\d{4})|20\d{2})", lower):
            slot = "year"
        elif re.search(r"\b(?:fy\s*'?(\d{2}|\d{4})|20\d{2})\b", lower):
            continue
        elif anchor_tokens & FINANCIAL_METRIC_TERMS:
            slot = "metric"
        elif re.search(r"[A-Z]", normalized) and lower not in FINANCIAL_ENTITY_STOPWORDS:
            slot = "entity"
        else:
            continue
        add_financial_slot_value(slots, seen, slot, normalized)
    slots["metric"] = dedupe_specific_financial_values(slots["metric"])[:12]
    slots["year"] = dedupe_specific_financial_values(slots["year"])[:6]
    slots["entity"] = dedupe_specific_financial_values(slots["entity"])[:6]
    return slots


def financial_reasoning_node_id(anchors: list[str]) -> str:
    normalized = "|".join(sorted(normalize_query_anchor(anchor).lower() for anchor in anchors))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"financial_reasoning::{digest}"


def financial_table_likeness(page_text: str) -> float:
    text = str(page_text or "")
    if not text:
        return 0.0
    numeric_hits = len(
        re.findall(
            r"(?:\$|€|£)?\(?\d[\d,]*(?:\.\d+)?\)?%?|\b20\d{2}\b|\bFY\s*'?[\d]{2,4}\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    percent_hits = len(re.findall(r"\d(?:\.\d+)?\s*%", text))
    currency_hits = len(re.findall(r"[$€£]\s*\(?\d", text))
    delimiter_hits = text.count("|") + text.count("\t")
    financial_term_hits = sum(
        1
        for term in FINANCIAL_METRIC_TERMS
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
    )
    numeric_score = clamp(numeric_hits / 45.0, 0.0, 1.0)
    marker_score = clamp((percent_hits + currency_hits + delimiter_hits) / 20.0, 0.0, 1.0)
    term_score = clamp(financial_term_hits / 12.0, 0.0, 1.0)
    return clamp(0.55 * numeric_score + 0.25 * marker_score + 0.20 * term_score, 0.0, 1.0)


def build_financial_reasoning_bundle(
    *,
    question: str,
    anchors: list[str],
    records: dict[str, PageRecord],
    page_texts: dict[str, str],
    doc_anchor_weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[str | None, dict[str, float], dict[str, object]]:
    if str(args.query_anchor_reasoning_mode) != "financial_slots":
        return None, {}, {}
    slots = financial_anchor_slots(question, anchors)
    active_slots = {slot: values for slot, values in slots.items() if values}
    min_slot_types = max(1, int(args.query_anchor_financial_min_slot_types))
    base_metadata: dict[str, object] = {
        "query_anchor_reasoning_mode": str(args.query_anchor_reasoning_mode),
        "query_anchor_financial_bundle_weight": float(args.query_anchor_financial_bundle_weight),
        "query_anchor_financial_min_slot_types": min_slot_types,
        "query_anchor_financial_max_page_matches": int(
            args.query_anchor_financial_max_page_matches
        ),
    }
    if len(active_slots) < min_slot_types:
        return None, {}, {
            **base_metadata,
            "query_anchor_financial_reasoning_active": False,
            "query_anchor_financial_reasoning_reason": "too_few_slot_types",
            "query_anchor_financial_metric_anchor_count": len(slots["metric"]),
            "query_anchor_financial_year_anchor_count": len(slots["year"]),
            "query_anchor_financial_entity_anchor_count": len(slots["entity"]),
            "query_anchor_financial_slot_type_count": len(active_slots),
            "query_anchor_financial_bundle_page_match_count": 0,
            "query_anchor_financial_bundle_dropped_broad": False,
        }

    matched_pages: dict[str, float] = {}
    matched_table_scores: list[float] = []
    table_bonus = max(0.0, float(args.query_anchor_financial_table_bonus))
    table_min_score = clamp(float(args.query_anchor_financial_table_min_score), 0.0, 1.0)
    for uid, record in records.items():
        doc_support = doc_anchor_weights.get(record.doc_id, 0.0)
        if str(args.query_anchor_scope) == "doc_conditioned" and doc_support <= 0:
            continue
        page_text = page_texts.get(uid, "")
        if not page_text:
            continue
        matched_slot_count = 0
        matched_anchor_count = 0
        total_anchor_count = sum(len(values) for values in active_slots.values())
        for slot, values in active_slots.items():
            slot_matches = sum(1 for value in values if anchor_matches_page_text(value, page_text))
            if slot_matches > 0:
                matched_slot_count += 1
                matched_anchor_count += slot_matches
        if matched_slot_count < min_slot_types:
            continue
        slot_score = matched_slot_count / max(1.0, float(len(active_slots)))
        anchor_score = matched_anchor_count / max(1.0, float(total_anchor_count))
        support = doc_support if str(args.query_anchor_scope) == "doc_conditioned" else 1.0
        table_score = financial_table_likeness(page_text)
        if table_score < table_min_score:
            continue
        matched_table_scores.append(table_score)
        base_score = support * (0.5 * slot_score + 0.5 * anchor_score)
        matched_pages[uid] = base_score * (1.0 + table_bonus * table_score)

    max_matches = max(0, int(args.query_anchor_financial_max_page_matches))
    dropped = bool(max_matches > 0 and len(matched_pages) > max_matches)
    if dropped:
        matched_pages = {}

    node_id = financial_reasoning_node_id(
        slots["metric"][:8] + slots["year"][:4] + slots["entity"][:4]
    )
    label_parts = []
    for slot in ("metric", "year", "entity"):
        if slots[slot]:
            label_parts.append(f"{slot}={','.join(slots[slot][:4])}")
    metadata = {
        **base_metadata,
        "query_anchor_financial_reasoning_active": bool(matched_pages),
        "query_anchor_financial_reasoning_reason": "matched" if matched_pages else "no_page_matches",
        "query_anchor_financial_metric_anchor_count": len(slots["metric"]),
        "query_anchor_financial_year_anchor_count": len(slots["year"]),
        "query_anchor_financial_entity_anchor_count": len(slots["entity"]),
        "query_anchor_financial_slot_type_count": len(active_slots),
        "query_anchor_financial_bundle_page_match_count": len(matched_pages),
        "query_anchor_financial_bundle_dropped_broad": dropped,
        "query_anchor_financial_bundle_label": " | ".join(label_parts),
        "query_anchor_financial_table_bonus": table_bonus,
        "query_anchor_financial_table_min_score": table_min_score,
        "query_anchor_financial_mean_table_score": (
            statistics.fmean(matched_table_scores) if matched_table_scores else None
        ),
    }
    return node_id if matched_pages else None, matched_pages, metadata


CONSTRAINT_ROLE_TRIGGERS = {
    "table": {
        "table",
        "tabular",
        "row",
        "column",
        "financial statement",
        "balance sheet",
        "income statement",
        "cash flow",
    },
    "chart": {"chart", "graph", "plot", "diagram", "figure"},
    "cover": {"cover", "front page", "title page", "first page"},
    "references": {"reference", "references", "bibliography", "citation", "cited"},
    "appendix": {"appendix", "appendices", "supplement"},
    "signature": {"signature", "signed", "signatory"},
    "date": {"date", "dated", "published", "publication", "released", "release"},
}


def extract_constraint_numeric_values(question: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    pattern = (
        r"\bFY\s*'?\d{2,4}\b"
        r"|\b20\d{2}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b(?:\$|€|£)?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|percentage|million|billion|thousand)?\b"
    )
    for match in re.finditer(pattern, str(question or ""), flags=re.IGNORECASE):
        value = normalize_query_anchor(match.group(0))
        lower = value.lower()
        if value and lower not in seen:
            values.append(value)
            seen.add(lower)
    return values[:12]


def extract_constraint_role_values(question: str) -> list[str]:
    query_lower = str(question or "").lower()
    roles: list[str] = []
    for role, triggers in CONSTRAINT_ROLE_TRIGGERS.items():
        if any(trigger in query_lower for trigger in triggers):
            roles.append(role)
    return roles


def query_constraint_slots(question: str, anchors: list[str]) -> dict[str, list[str]]:
    slots: dict[str, list[str]] = {
        "entity": [],
        "numeric": [],
        "metric": [],
        "role": [],
    }
    seen: dict[str, set[str]] = {slot: set() for slot in slots}

    def add(slot: str, value: str) -> None:
        normalized = normalize_query_anchor(value)
        lower = normalized.lower()
        if normalized and lower not in seen[slot]:
            slots[slot].append(normalized)
            seen[slot].add(lower)

    for value in extract_constraint_numeric_values(question):
        add("numeric", value)
    for value in extract_financial_metric_values(question):
        add("metric", value)
    for role in extract_constraint_role_values(question):
        add("role", role)

    for anchor in anchors:
        normalized = normalize_query_anchor(anchor)
        lower = normalized.lower()
        if not normalized:
            continue
        if re.search(r"\d", lower):
            add("numeric", normalized)
            continue
        anchor_tokens = set(re.findall(r"[a-z][a-z0-9&./+-]*", lower))
        if anchor_tokens & FINANCIAL_METRIC_TERMS:
            add("metric", normalized)
            continue
        if lower in FINANCIAL_ENTITY_STOPWORDS:
            continue
        if re.search(r"[A-Z]", normalized) or len(normalized.split()) > 1:
            add("entity", normalized)

    slots["entity"] = dedupe_specific_financial_values(slots["entity"])[:8]
    slots["numeric"] = dedupe_specific_financial_values(slots["numeric"])[:8]
    slots["metric"] = dedupe_specific_financial_values(slots["metric"])[:8]
    slots["role"] = slots["role"][:6]
    return slots


def query_constraint_node_id(slots: dict[str, list[str]]) -> str:
    raw_parts = []
    for slot in sorted(slots):
        if slots[slot]:
            raw_parts.append(f"{slot}={','.join(value.lower() for value in slots[slot])}")
    digest = hashlib.sha1("|".join(raw_parts).encode("utf-8")).hexdigest()[:12]
    return f"query_constraint_bundle::{digest}"


def role_matches_page(role: str, page_text: str, record: PageRecord) -> float:
    text = str(page_text or "")
    role = str(role or "").lower()
    if not text and role != "cover":
        return 0.0
    if role == "table":
        score = financial_table_likeness(text)
        if "|" in text or "\t" in text:
            score = max(score, 0.5)
        return score if score >= 0.10 else 0.0
    if role == "chart":
        return 1.0 if re.search(r"\b(?:chart|graph|plot|figure|diagram)\b", text) else 0.0
    if role == "cover":
        return 1.0 if int(record.page_idx) == 0 else 0.0
    if role == "references":
        return 1.0 if re.search(r"\b(?:references|bibliography|works cited)\b", text) else 0.0
    if role == "appendix":
        return 1.0 if re.search(r"\bappendix\b|\bappendices\b", text) else 0.0
    if role == "signature":
        return 1.0 if re.search(r"\b(?:signature|signed|signatory)\b", text) else 0.0
    if role == "date":
        return 1.0 if re.search(r"\b20\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text) else 0.0
    return 0.0


def constraint_slot_match_count(
    *,
    slot: str,
    values: list[str],
    page_text: str,
    record: PageRecord,
) -> int:
    if slot == "role":
        return sum(1 for value in values if role_matches_page(value, page_text, record) > 0)
    return sum(1 for value in values if anchor_matches_page_text(value, page_text))


def constraint_slot_specificity_weight(
    *, match_count: int, candidate_count: int, floor: float
) -> float:
    if match_count <= 0 or candidate_count <= 1:
        return 1.0
    try:
        raw = math.log((float(candidate_count) + 1.0) / (float(match_count) + 1.0)) / math.log(
            float(candidate_count) + 1.0
        )
    except (ValueError, ZeroDivisionError):
        raw = 1.0
    return clamp(raw, floor, 1.0)


def build_query_constraint_bundle(
    *,
    question: str,
    anchors: list[str],
    records: dict[str, PageRecord],
    page_texts: dict[str, str],
    doc_anchor_weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[str | None, dict[str, float], dict[str, object]]:
    if str(args.query_anchor_reasoning_mode) != "constraint_bundles":
        return None, {}, {}

    slots = query_constraint_slots(question, anchors)
    active_slots = {slot: values for slot, values in slots.items() if values}
    min_slot_types = max(1, int(args.query_anchor_constraint_min_slot_types))
    max_slot_page_matches = max(0, int(args.query_anchor_constraint_max_slot_page_matches))
    max_page_matches = max(0, int(args.query_anchor_constraint_max_page_matches))
    specificity_floor = clamp(float(args.query_anchor_constraint_specificity_floor), 0.0, 1.0)
    table_bonus = max(0.0, float(args.query_anchor_constraint_table_bonus))
    table_min_score = clamp(float(args.query_anchor_constraint_table_min_score), 0.0, 1.0)
    base_metadata: dict[str, object] = {
        "query_anchor_constraint_bundle_weight": float(
            args.query_anchor_constraint_bundle_weight
        ),
        "query_anchor_constraint_min_slot_types": min_slot_types,
        "query_anchor_constraint_max_page_matches": max_page_matches,
        "query_anchor_constraint_max_slot_page_matches": max_slot_page_matches,
        "query_anchor_constraint_specificity_floor": specificity_floor,
        "query_anchor_constraint_table_bonus": table_bonus,
        "query_anchor_constraint_table_min_score": table_min_score,
    }
    if len(active_slots) < min_slot_types:
        return None, {}, {
            **base_metadata,
            "query_anchor_constraint_bundle_active": False,
            "query_anchor_constraint_bundle_reason": "too_few_slot_types",
            "query_anchor_constraint_active_slot_count": len(active_slots),
            "query_anchor_constraint_bundle_page_match_count": 0,
            "query_anchor_constraint_dropped_broad_slot_count": 0,
            "query_anchor_constraint_bundle_dropped_broad": False,
        }

    scope = str(args.query_anchor_scope)
    min_doc_support = clamp(float(args.query_anchor_min_doc_support), 0.0, 1.0)
    candidate_records: dict[str, PageRecord] = {}
    for uid, record in records.items():
        if scope == "doc_conditioned":
            doc_support = doc_anchor_weights.get(record.doc_id, 0.0)
            if doc_support < min_doc_support or doc_support <= 0:
                continue
        candidate_records[uid] = record
    candidate_count = len(candidate_records)

    slot_page_match_counts: dict[str, int] = {}
    active_slots_filtered: dict[str, list[str]] = {}
    dropped_broad_slot_count = 0
    for slot, values in active_slots.items():
        match_count = 0
        for uid, record in candidate_records.items():
            page_text = page_texts.get(uid, "")
            if constraint_slot_match_count(slot=slot, values=values, page_text=page_text, record=record) > 0:
                match_count += 1
        if max_slot_page_matches > 0 and match_count > max_slot_page_matches:
            dropped_broad_slot_count += 1
            continue
        if match_count > 0:
            slot_page_match_counts[slot] = match_count
            active_slots_filtered[slot] = values

    if len(active_slots_filtered) < min_slot_types:
        return None, {}, {
            **base_metadata,
            "query_anchor_constraint_bundle_active": False,
            "query_anchor_constraint_bundle_reason": "too_few_matched_slot_types",
            "query_anchor_constraint_active_slot_count": len(active_slots),
            "query_anchor_constraint_matched_slot_count": len(active_slots_filtered),
            "query_anchor_constraint_bundle_page_match_count": 0,
            "query_anchor_constraint_dropped_broad_slot_count": dropped_broad_slot_count,
            "query_anchor_constraint_bundle_dropped_broad": False,
            "query_anchor_constraint_slot_page_match_counts": slot_page_match_counts,
        }

    slot_specificities = {
        slot: constraint_slot_specificity_weight(
            match_count=count,
            candidate_count=candidate_count,
            floor=specificity_floor,
        )
        for slot, count in slot_page_match_counts.items()
    }
    total_value_count = sum(len(values) for values in active_slots_filtered.values())
    matched_pages: dict[str, float] = {}
    matched_table_scores: list[float] = []
    for uid, record in candidate_records.items():
        page_text = page_texts.get(uid, "")
        if not page_text and "role" not in active_slots_filtered:
            continue
        matched_slots: list[str] = []
        matched_value_count = 0
        for slot, values in active_slots_filtered.items():
            count = constraint_slot_match_count(
                slot=slot,
                values=values,
                page_text=page_text,
                record=record,
            )
            if count > 0:
                matched_slots.append(slot)
                matched_value_count += count
        required_slots = {
            slot for slot in ("entity", "numeric") if slot in active_slots_filtered
        }
        if required_slots and not required_slots.issubset(set(matched_slots)):
            continue
        if len(matched_slots) < min_slot_types:
            continue
        table_score = financial_table_likeness(page_text)
        if table_score < table_min_score:
            continue
        doc_support = doc_anchor_weights.get(record.doc_id, 1.0) if scope == "doc_conditioned" else 1.0
        slot_score = len(matched_slots) / max(1.0, float(len(active_slots_filtered)))
        value_score = matched_value_count / max(1.0, float(total_value_count))
        specificity = statistics.fmean(slot_specificities[slot] for slot in matched_slots)
        base_score = doc_support * specificity * (0.65 * slot_score + 0.35 * value_score)
        matched_table_scores.append(table_score)
        matched_pages[uid] = base_score * (1.0 + table_bonus * table_score)

    dropped_bundle = bool(max_page_matches > 0 and len(matched_pages) > max_page_matches)
    if dropped_bundle:
        matched_pages = {}

    label_parts = []
    for slot in ("entity", "numeric", "metric", "role"):
        if active_slots_filtered.get(slot):
            label_parts.append(f"{slot}={','.join(active_slots_filtered[slot][:4])}")
    metadata = {
        **base_metadata,
        "query_anchor_constraint_bundle_active": bool(matched_pages),
        "query_anchor_constraint_bundle_reason": "matched" if matched_pages else "no_page_matches",
        "query_anchor_constraint_active_slot_count": len(active_slots),
        "query_anchor_constraint_matched_slot_count": len(active_slots_filtered),
        "query_anchor_constraint_entity_count": len(slots["entity"]),
        "query_anchor_constraint_numeric_count": len(slots["numeric"]),
        "query_anchor_constraint_metric_count": len(slots["metric"]),
        "query_anchor_constraint_role_count": len(slots["role"]),
        "query_anchor_constraint_bundle_page_match_count": len(matched_pages),
        "query_anchor_constraint_dropped_broad_slot_count": dropped_broad_slot_count,
        "query_anchor_constraint_bundle_dropped_broad": dropped_bundle,
        "query_anchor_constraint_bundle_label": " | ".join(label_parts),
        "query_anchor_constraint_required_slots": [
            slot for slot in ("entity", "numeric") if slot in active_slots_filtered
        ],
        "query_anchor_constraint_slot_page_match_counts": slot_page_match_counts,
        "query_anchor_constraint_mean_slot_specificity": (
            statistics.fmean(slot_specificities.values()) if slot_specificities else None
        ),
        "query_anchor_constraint_mean_table_score": (
            statistics.fmean(matched_table_scores) if matched_table_scores else None
        ),
    }
    return query_constraint_node_id(active_slots_filtered) if matched_pages else None, matched_pages, metadata


def query_anchor_node_id(anchor: str) -> str:
    digest = hashlib.sha1(anchor.lower().encode("utf-8")).hexdigest()[:12]
    return f"query_anchor::{digest}"


def query_anchor_specificity_weight(
    *, match_count: int, candidate_count: int, args: argparse.Namespace
) -> float:
    if str(args.query_anchor_weight_mode) != "local_idf":
        return 1.0
    if match_count <= 0 or candidate_count <= 1:
        return 1.0
    try:
        raw = math.log((float(candidate_count) + 1.0) / (float(match_count) + 1.0)) / math.log(
            float(candidate_count) + 1.0
        )
    except (ValueError, ZeroDivisionError):
        raw = 1.0
    return clamp(raw, float(args.query_anchor_min_node_weight), 1.0)


def build_query_anchor_doc_weights(
    *,
    dense_doc_ranks: dict[str, int],
    sparse_doc_ranks: dict[str, int],
    source_weights: SourceWeights,
    args: argparse.Namespace,
) -> dict[str, float]:
    raw_doc_weights: dict[str, float] = defaultdict(float)
    for doc_id, rank in dense_doc_ranks.items():
        raw_doc_weights[doc_id] += source_weights.dense_weight / (
            float(args.rrf_k) + float(rank)
        )
    for doc_id, rank in sparse_doc_ranks.items():
        raw_doc_weights[doc_id] += source_weights.sparse_weight / (
            float(args.rrf_k) + float(rank)
        )
    if not raw_doc_weights:
        return {}
    sorted_docs = sorted(raw_doc_weights.items(), key=lambda item: (-item[1], item[0]))
    top_k = max(0, int(args.query_anchor_doc_top_k))
    if top_k > 0:
        sorted_docs = sorted_docs[:top_k]
    return max_scale(dict(sorted_docs))


def build_query_anchor_evidence_policy(
    *,
    question: str,
    records: dict[str, PageRecord],
    page_texts: dict[str, str],
    doc_anchor_weights: dict[str, float],
    args: argparse.Namespace,
) -> QueryAnchorEvidencePolicy:
    mode = str(args.query_anchor_evidence_mode)
    anchor_page_weights: dict[str, dict[str, float]] = {}
    anchor_node_weights: dict[str, float] = {}
    anchor_labels: dict[str, str] = {}
    anchors: list[str] = []
    dropped_broad_anchor_count = 0
    financial_metadata: dict[str, object] = {
        "query_anchor_reasoning_mode": str(args.query_anchor_reasoning_mode),
        "query_anchor_financial_reasoning_active": False,
        "query_anchor_financial_reasoning_reason": "disabled",
        "query_anchor_financial_metric_anchor_count": 0,
        "query_anchor_financial_year_anchor_count": 0,
        "query_anchor_financial_entity_anchor_count": 0,
        "query_anchor_financial_bundle_page_match_count": 0,
    }
    constraint_metadata: dict[str, object] = {
        "query_anchor_constraint_bundle_active": False,
        "query_anchor_constraint_bundle_reason": "disabled",
        "query_anchor_constraint_active_slot_count": 0,
        "query_anchor_constraint_matched_slot_count": 0,
        "query_anchor_constraint_entity_count": 0,
        "query_anchor_constraint_numeric_count": 0,
        "query_anchor_constraint_metric_count": 0,
        "query_anchor_constraint_role_count": 0,
        "query_anchor_constraint_bundle_page_match_count": 0,
        "query_anchor_constraint_dropped_broad_slot_count": 0,
        "query_anchor_constraint_bundle_dropped_broad": False,
    }
    if mode != "none" and page_texts:
        anchors = extract_query_anchors(
            question,
            max_anchors=int(args.query_anchor_max_anchors),
            min_entity_len=int(args.query_anchor_min_entity_len),
        )
        scope = str(args.query_anchor_scope)
        min_doc_support = clamp(float(args.query_anchor_min_doc_support), 0.0, 1.0)
        max_page_matches = max(0, int(args.query_anchor_max_page_matches))
        if scope == "doc_conditioned":
            candidate_count = sum(
                1
                for record in records.values()
                if doc_anchor_weights.get(record.doc_id, 0.0) >= min_doc_support
                and doc_anchor_weights.get(record.doc_id, 0.0) > 0
            )
        else:
            candidate_count = len(records)
        for anchor in anchors:
            node_id = query_anchor_node_id(anchor)
            anchor_labels[node_id] = anchor
            matched_pages: dict[str, float] = {}
            for uid, record in records.items():
                doc_support = 1.0
                if scope == "doc_conditioned":
                    doc_support = doc_anchor_weights.get(record.doc_id, 0.0)
                    if doc_support < min_doc_support or doc_support <= 0:
                        continue
                page_text = page_texts.get(uid, "")
                if anchor_matches_page_text(anchor, page_text):
                    matched_pages[uid] = doc_support
            if max_page_matches > 0 and len(matched_pages) > max_page_matches:
                dropped_broad_anchor_count += 1
                continue
            if matched_pages:
                anchor_page_weights[node_id] = matched_pages
                anchor_node_weights[node_id] = query_anchor_specificity_weight(
                    match_count=len(matched_pages),
                    candidate_count=candidate_count,
                    args=args,
                )
        if str(args.query_anchor_reasoning_mode) == "financial_slots":
            if query_has_financial_reasoning_cue(question):
                bundle_node, bundle_pages, financial_metadata = build_financial_reasoning_bundle(
                    question=question,
                    anchors=anchors,
                    records=records,
                    page_texts=page_texts,
                    doc_anchor_weights=doc_anchor_weights,
                    args=args,
                )
                if bundle_node and bundle_pages:
                    anchor_page_weights[bundle_node] = bundle_pages
                    anchor_node_weights[bundle_node] = max(
                        0.0, float(args.query_anchor_financial_bundle_weight)
                    )
                    anchor_labels[bundle_node] = (
                        "financial_slots: "
                        + str(financial_metadata.get("query_anchor_financial_bundle_label", ""))
                    )
            else:
                financial_metadata.update(
                    {
                        "query_anchor_financial_reasoning_reason": "no_financial_cue",
                        "query_anchor_financial_bundle_weight": float(
                            args.query_anchor_financial_bundle_weight
                        ),
                        "query_anchor_financial_min_slot_types": int(
                            args.query_anchor_financial_min_slot_types
                        ),
                        "query_anchor_financial_max_page_matches": int(
                            args.query_anchor_financial_max_page_matches
                        ),
                    }
                )
        if str(args.query_anchor_reasoning_mode) == "constraint_bundles":
            bundle_node, bundle_pages, constraint_metadata = build_query_constraint_bundle(
                question=question,
                anchors=anchors,
                records=records,
                page_texts=page_texts,
                doc_anchor_weights=doc_anchor_weights,
                args=args,
            )
            if bundle_node and bundle_pages:
                anchor_page_weights[bundle_node] = bundle_pages
                anchor_node_weights[bundle_node] = max(
                    0.0, float(args.query_anchor_constraint_bundle_weight)
                )
                anchor_labels[bundle_node] = (
                    "constraint_bundles: "
                    + str(constraint_metadata.get("query_anchor_constraint_bundle_label", ""))
                )

    page_match_count = sum(len(pages) for pages in anchor_page_weights.values())
    metadata: dict[str, object] = {
        "query_anchor_evidence_mode": mode,
        "query_anchor_scope": str(args.query_anchor_scope),
        "query_anchor_active_anchor_count": len(anchors),
        "query_anchor_matched_anchor_count": len(anchor_page_weights),
        "query_anchor_dropped_broad_anchor_count": dropped_broad_anchor_count,
        "query_anchor_page_match_count": page_match_count,
        "query_anchor_doc_support_count": len(doc_anchor_weights),
        "query_anchor_doc_top_k": int(args.query_anchor_doc_top_k),
        "query_anchor_min_doc_support": float(args.query_anchor_min_doc_support),
        "query_anchor_edge_weight": float(args.query_anchor_edge_weight),
        "query_anchor_restart_weight": float(args.query_anchor_restart_weight),
        "query_anchor_weight_mode": str(args.query_anchor_weight_mode),
        "query_anchor_min_node_weight": float(args.query_anchor_min_node_weight),
        "query_anchor_max_page_matches": int(args.query_anchor_max_page_matches),
        "mean_query_anchor_node_weight": (
            statistics.fmean(anchor_node_weights.values()) if anchor_node_weights else None
        ),
        "query_anchor_page_text_available": bool(page_texts),
        "query_anchor_text_fields": list(args.query_anchor_text_field),
        "query_anchor_labels": [anchor_labels[node_id] for node_id in sorted(anchor_labels)],
        **financial_metadata,
        **constraint_metadata,
    }
    return QueryAnchorEvidencePolicy(
        anchor_page_weights=anchor_page_weights,
        anchor_node_weights=anchor_node_weights,
        anchor_labels=anchor_labels,
        metadata=metadata,
    )


def add_query_anchor_evidence_edges(
    *,
    graph: dict[str, dict[str, float]],
    policy: QueryAnchorEvidencePolicy,
    args: argparse.Namespace,
) -> dict[str, object]:
    if str(args.query_anchor_evidence_mode) == "none":
        return {
            "query_anchor_edge_count_directed": 0,
            "query_anchor_seed_node_count": 0,
        }
    edge_count = 0
    seeded_node_count = 0
    for anchor_node, pages in policy.anchor_page_weights.items():
        if not pages:
            continue
        seeded_node_count += 1
        node_weight = policy.anchor_node_weights.get(anchor_node, 1.0)
        for uid, page_weight in pages.items():
            edge_weight = float(args.query_anchor_edge_weight) * node_weight * float(page_weight)
            if edge_weight > 0:
                add_directed_edge(graph, anchor_node, uid, edge_weight)
                edge_count += 1
    return {
        "query_anchor_edge_count_directed": edge_count,
        "query_anchor_seed_node_count": seeded_node_count,
    }


def add_query_anchor_evidence_restart(
    *,
    seed: dict[str, float],
    policy: QueryAnchorEvidencePolicy,
    args: argparse.Namespace,
) -> int:
    if str(args.query_anchor_evidence_mode) == "none":
        return 0
    restart_weight = float(args.query_anchor_restart_weight)
    if restart_weight <= 0:
        return 0
    added = 0
    for anchor_node, pages in policy.anchor_page_weights.items():
        if not pages:
            continue
        node_weight = policy.anchor_node_weights.get(anchor_node, 1.0)
        seed[anchor_node] = seed.get(anchor_node, 0.0) + restart_weight * node_weight
        added += 1
    return added


def page_record_best_rank(record: PageRecord) -> int | None:
    ranks = [
        rank
        for rank in (record.dense_rank, record.sparse_rank, record.expansion_rank)
        if rank is not None
    ]
    return min(ranks) if ranks else None


def rank_decay_factor(rank: int | None) -> float:
    if rank is None or int(rank) <= 0:
        return 0.0
    return 1.0 / math.log2(float(rank) + 1.0)


def pdf_hyperlink_query_support_multiplier(
    *,
    source_rank: int | None,
    target_doc_rank: int | None,
    args: argparse.Namespace,
) -> float:
    mode = str(args.pdf_hyperlink_query_support_weight_mode)
    if mode == "source_rank_decay":
        return rank_decay_factor(source_rank)
    if mode == "source_target_rank_decay":
        return rank_decay_factor(source_rank) * rank_decay_factor(target_doc_rank)
    return 1.0


def pdf_hyperlink_edge_weight(
    raw_link_count: int,
    args: argparse.Namespace,
    *,
    source_rank: int | None = None,
    target_doc_rank: int | None = None,
) -> float:
    base_weight = float(args.pdf_hyperlink_edge_weight)
    if base_weight <= 0:
        return 0.0
    if str(args.pdf_hyperlink_weight_mode) == "log_count":
        base_weight *= math.log1p(max(1, int(raw_link_count)))
    return base_weight * pdf_hyperlink_query_support_multiplier(
        source_rank=source_rank,
        target_doc_rank=target_doc_rank,
        args=args,
    )


def pdf_hyperlink_doc_best_ranks(records: dict[str, PageRecord]) -> dict[str, int]:
    doc_best_ranks: dict[str, int] = {}
    for record in records.values():
        rank = page_record_best_rank(record)
        if rank is None:
            continue
        doc_best_ranks[record.doc_id] = min(doc_best_ranks.get(record.doc_id, rank), rank)
    return doc_best_ranks


def pdf_hyperlink_passes_rank_gate(rank: int | None, cutoff: int) -> bool:
    if cutoff <= 0:
        return True
    return rank is not None and rank <= cutoff


def add_pdf_hyperlink_edges(
    *,
    graph: dict[str, dict[str, float]],
    records: dict[str, PageRecord],
    pdf_hyperlink_graph: PdfHyperlinkGraph | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    if (
        pdf_hyperlink_graph is None
        or not str(args.pdf_hyperlink_edges_jsonl)
        or float(args.pdf_hyperlink_edge_weight) <= 0
    ):
        return {
            "pdf_hyperlink_edge_count_directed": 0,
            "pdf_hyperlink_source_page_count": 0,
            "pdf_hyperlink_target_doc_count": 0,
            "pdf_hyperlink_raw_link_count": 0,
            "pdf_hyperlink_source_top_k": int(args.pdf_hyperlink_source_top_k),
            "pdf_hyperlink_target_doc_top_k": int(args.pdf_hyperlink_target_doc_top_k),
            "pdf_hyperlink_query_support_weight_mode": str(
                args.pdf_hyperlink_query_support_weight_mode
            ),
            "pdf_hyperlink_skipped_source_rank_gate": 0,
            "pdf_hyperlink_skipped_target_doc_rank_gate": 0,
            "pdf_hyperlink_loaded_edge_count": (
                pdf_hyperlink_graph.edge_count if pdf_hyperlink_graph is not None else 0
            ),
        }

    candidate_doc_ids = {record.doc_id for record in records.values()}
    doc_best_ranks = pdf_hyperlink_doc_best_ranks(records)
    used_source_pages: set[str] = set()
    used_target_docs: set[str] = set()
    edge_count = 0
    raw_link_count = 0
    max_edges_per_source = max(0, int(args.pdf_hyperlink_max_edges_per_source))
    source_top_k = max(0, int(args.pdf_hyperlink_source_top_k))
    target_doc_top_k = max(0, int(args.pdf_hyperlink_target_doc_top_k))
    skipped_source_rank_gate = 0
    skipped_target_doc_rank_gate = 0

    for source_uid in sorted(records):
        source_edges = pdf_hyperlink_graph.by_source_page.get(source_uid, [])
        if not source_edges:
            continue
        source_rank = page_record_best_rank(records[source_uid])
        if not pdf_hyperlink_passes_rank_gate(source_rank, source_top_k):
            skipped_source_rank_gate += 1
            continue
        weighted_edges: list[tuple[PdfHyperlinkEdge, float, int | None]] = []
        for edge in source_edges:
            target_doc_rank = doc_best_ranks.get(edge.target_doc_id)
            if edge.target_doc_id not in candidate_doc_ids:
                continue
            if not pdf_hyperlink_passes_rank_gate(target_doc_rank, target_doc_top_k):
                skipped_target_doc_rank_gate += 1
                continue
            weight = pdf_hyperlink_edge_weight(
                edge.raw_link_count,
                args,
                source_rank=source_rank,
                target_doc_rank=target_doc_rank,
            )
            if weight <= 0:
                continue
            weighted_edges.append((edge, weight, target_doc_rank))
        ordered_edges = sorted(
            weighted_edges,
            key=lambda item: (-item[1], -int(item[0].raw_link_count), item[0].target_doc_id),
        )
        if max_edges_per_source > 0:
            ordered_edges = ordered_edges[:max_edges_per_source]
        for edge, weight, _target_doc_rank in ordered_edges:
            target_doc_node = f"doc::{edge.target_doc_id}"
            add_directed_edge(graph, source_uid, target_doc_node, weight)
            edge_count += 1
            raw_link_count += int(edge.raw_link_count)
            used_source_pages.add(source_uid)
            used_target_docs.add(edge.target_doc_id)
            if str(args.pdf_hyperlink_direction) == "bidirectional_doc":
                add_directed_edge(graph, target_doc_node, source_uid, weight)
                edge_count += 1

    return {
        "pdf_hyperlink_edge_count_directed": edge_count,
        "pdf_hyperlink_source_page_count": len(used_source_pages),
        "pdf_hyperlink_target_doc_count": len(used_target_docs),
        "pdf_hyperlink_raw_link_count": raw_link_count,
        "pdf_hyperlink_loaded_edge_count": pdf_hyperlink_graph.edge_count,
        "pdf_hyperlink_loaded_source_page_count": pdf_hyperlink_graph.source_page_count,
        "pdf_hyperlink_loaded_target_doc_count": pdf_hyperlink_graph.target_doc_count,
        "pdf_hyperlink_edge_weight": float(args.pdf_hyperlink_edge_weight),
        "pdf_hyperlink_direction": str(args.pdf_hyperlink_direction),
        "pdf_hyperlink_weight_mode": str(args.pdf_hyperlink_weight_mode),
        "pdf_hyperlink_max_edges_per_source": int(args.pdf_hyperlink_max_edges_per_source),
        "pdf_hyperlink_source_top_k": int(args.pdf_hyperlink_source_top_k),
        "pdf_hyperlink_target_doc_top_k": int(args.pdf_hyperlink_target_doc_top_k),
        "pdf_hyperlink_query_support_weight_mode": str(
            args.pdf_hyperlink_query_support_weight_mode
        ),
        "pdf_hyperlink_skipped_source_rank_gate": skipped_source_rank_gate,
        "pdf_hyperlink_skipped_target_doc_rank_gate": skipped_target_doc_rank_gate,
    }


def external_page_graph_edge_weight(
    edge: ExternalPageGraphEdge,
    args: argparse.Namespace,
) -> float:
    base_weight = float(args.external_page_graph_edge_weight)
    if base_weight <= 0:
        return 0.0
    mode = str(args.external_page_graph_weight_mode)
    if mode == "score":
        return base_weight * max(0.0, float(edge.score))
    if mode == "weight":
        return base_weight * max(0.0, float(edge.raw_weight))
    return base_weight


def external_page_graph_record_rank(record: PageRecord) -> int | None:
    return page_record_best_rank(record)


def external_page_graph_passes_rank_gate(record: PageRecord, cutoff: int) -> bool:
    if cutoff <= 0:
        return True
    rank = external_page_graph_record_rank(record)
    return rank is not None and rank <= cutoff


def add_external_page_graph_edges(
    *,
    graph: dict[str, dict[str, float]],
    records: dict[str, PageRecord],
    external_page_graph: ExternalPageGraph | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    if (
        external_page_graph is None
        or not str(args.external_page_graph_jsonl)
        or float(args.external_page_graph_edge_weight) <= 0
    ):
        return {
            "external_page_graph_edge_count_directed": 0,
            "external_page_graph_source_page_count": 0,
            "external_page_graph_target_page_count": 0,
            "external_page_graph_target_doc_count": 0,
            "external_page_graph_source_top_k": int(args.external_page_graph_source_top_k),
            "external_page_graph_target_top_k": int(args.external_page_graph_target_top_k),
            "external_page_graph_skipped_source_rank_gate": 0,
            "external_page_graph_skipped_target_rank_gate": 0,
            "external_page_graph_loaded_edge_count": (
                external_page_graph.edge_count if external_page_graph is not None else 0
            ),
        }

    candidate_doc_ids = {record.doc_id for record in records.values()}
    used_source_pages: set[str] = set()
    used_target_pages: set[str] = set()
    used_target_docs: set[str] = set()
    edge_count = 0
    max_edges_per_source = max(0, int(args.external_page_graph_max_edges_per_source))
    source_top_k = max(0, int(args.external_page_graph_source_top_k))
    target_top_k = max(0, int(args.external_page_graph_target_top_k))
    skipped_source_rank_gate = 0
    skipped_target_rank_gate = 0

    for source_uid in sorted(records):
        source_edges = external_page_graph.by_source_page.get(source_uid, [])
        if not source_edges:
            continue
        source_record = records[source_uid]
        if not external_page_graph_passes_rank_gate(source_record, source_top_k):
            skipped_source_rank_gate += 1
            continue
        ordered_edges = sorted(
            source_edges,
            key=lambda edge: (
                -external_page_graph_edge_weight(edge, args),
                edge.target_page_uid or "",
                edge.target_doc_id or "",
            ),
        )
        if max_edges_per_source > 0:
            ordered_edges = ordered_edges[:max_edges_per_source]
        for edge in ordered_edges:
            weight = external_page_graph_edge_weight(edge, args)
            if weight <= 0:
                continue
            target_node = None
            if edge.target_page_uid:
                target_record = records.get(edge.target_page_uid)
                if target_record is None:
                    continue
                if not external_page_graph_passes_rank_gate(target_record, target_top_k):
                    skipped_target_rank_gate += 1
                    continue
                target_node = edge.target_page_uid
                used_target_pages.add(edge.target_page_uid)
                used_target_docs.add(target_record.doc_id)
            elif edge.target_doc_id and edge.target_doc_id in candidate_doc_ids:
                target_node = f"doc::{edge.target_doc_id}"
                used_target_docs.add(edge.target_doc_id)
            if target_node is None:
                continue
            add_directed_edge(graph, source_uid, target_node, weight)
            edge_count += 1
            used_source_pages.add(source_uid)
            if str(args.external_page_graph_direction) == "bidirectional":
                add_directed_edge(graph, target_node, source_uid, weight)
                edge_count += 1

    return {
        "external_page_graph_edge_count_directed": edge_count,
        "external_page_graph_source_page_count": len(used_source_pages),
        "external_page_graph_target_page_count": len(used_target_pages),
        "external_page_graph_target_doc_count": len(used_target_docs),
        "external_page_graph_loaded_edge_count": external_page_graph.edge_count,
        "external_page_graph_loaded_source_page_count": external_page_graph.source_page_count,
        "external_page_graph_loaded_target_page_count": external_page_graph.target_page_count,
        "external_page_graph_loaded_target_doc_count": external_page_graph.target_doc_count,
        "external_page_graph_edge_weight": float(args.external_page_graph_edge_weight),
        "external_page_graph_direction": str(args.external_page_graph_direction),
        "external_page_graph_weight_mode": str(args.external_page_graph_weight_mode),
        "external_page_graph_max_edges_per_source": int(
            args.external_page_graph_max_edges_per_source
        ),
        "external_page_graph_source_top_k": int(args.external_page_graph_source_top_k),
        "external_page_graph_target_top_k": int(args.external_page_graph_target_top_k),
        "external_page_graph_skipped_source_rank_gate": skipped_source_rank_gate,
        "external_page_graph_skipped_target_rank_gate": skipped_target_rank_gate,
    }


def add_undirected_edge(
    graph: dict[str, dict[str, float]],
    left: str,
    right: str,
    weight: float,
) -> None:
    if weight <= 0 or left == right:
        return
    add_directed_edge(graph, left, right, weight)
    add_directed_edge(graph, right, left, weight)


def add_directed_edge(
    graph: dict[str, dict[str, float]],
    source: str,
    target: str,
    weight: float,
) -> None:
    if weight <= 0 or source == target:
        return
    graph.setdefault(source, {})
    graph.setdefault(target, {})
    graph[source][target] = graph[source].get(target, 0.0) + float(weight)


def normalize_nonnegative(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        if not values:
            return {}
        uniform = 1.0 / len(values)
        return {key: uniform for key in values}
    return {key: max(0.0, value) / total for key, value in values.items()}


def max_scale(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: float(value) / max_value for key, value in values.items()}


def run_ppr(
    *,
    graph: dict[str, dict[str, float]],
    seed: dict[str, float],
    restart_prob: float,
    iters: int,
) -> dict[str, float]:
    nodes = sorted(set(graph) | set(seed))
    if not nodes:
        return {}
    normalized_seed = normalize_nonnegative({node: seed.get(node, 0.0) for node in nodes})
    rank = dict(normalized_seed)
    outgoing_totals = {
        node: sum(max(0.0, weight) for weight in graph.get(node, {}).values())
        for node in nodes
    }

    for _ in range(max(0, int(iters))):
        next_rank = {node: float(restart_prob) * normalized_seed[node] for node in nodes}
        dangling_mass = 0.0
        for src in nodes:
            src_rank = rank.get(src, 0.0)
            total = outgoing_totals.get(src, 0.0)
            if total <= 0:
                dangling_mass += src_rank
                continue
            share = (1.0 - float(restart_prob)) * src_rank / total
            for dst, weight in graph.get(src, {}).items():
                if weight > 0:
                    next_rank[dst] = next_rank.get(dst, 0.0) + share * float(weight)
        if dangling_mass > 0:
            for node in nodes:
                next_rank[node] += (1.0 - float(restart_prob)) * dangling_mass * normalized_seed[node]
        rank = next_rank
    return rank


def gold_doc_ids(row: dict) -> set[str]:
    return {
        str(item.get("doc_id", "")).strip()
        for item in row.get("supporting_context", [])
        if str(item.get("doc_id", "")).strip()
    }


def gold_page_uids(row: dict) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for item in row.get("supporting_context", []):
        doc_id = str(item.get("doc_id", "")).strip()
        page_idx = item.get("page_idx", item.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def first_rank(items: Iterable[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def ranked_docs_from_rows(rows: list[list[object]]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, _page_idx, _score = parsed
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def ranked_pages_from_rows(rows: list[list[object]]) -> list[str]:
    pages: list[str] = []
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, _score = parsed
        pages.append(page_uid(doc_id, page_idx))
    return pages


def page_rows_from_ranked_unique_pages(
    rows: list[list[object]],
    top_pages: int,
) -> list[list[object]]:
    return [
        [doc_id, int(page_idx), float(score)]
        for doc_id, page_idx, score, _rank in ranked_unique_pages(rows, top_pages)
    ]


def median_or_none(values: list[int | None]) -> float | None:
    filtered = sorted(int(value) for value in values if value is not None)
    if not filtered:
        return None
    return float(statistics.median(filtered))


def build_qid_graph_ranking(
    *,
    qid: str,
    dense_row: dict,
    sparse_row: dict,
    args: argparse.Namespace,
    sparse_index: SparsePageIndex | None = None,
    gold_row: dict | None = None,
    doc_page_catalog: DocPageCatalog | None = None,
    pdf_hyperlink_graph: PdfHyperlinkGraph | None = None,
    external_page_graph: ExternalPageGraph | None = None,
) -> tuple[list[list[object]], dict]:
    question = str(dense_row.get("question") or sparse_row.get("question", ""))
    dense_pages = ranked_unique_pages(
        dense_row.get("page_retrieval_results", []),
        int(args.dense_top_pages),
    )
    sparse_pages = ranked_unique_pages(
        sparse_row.get("page_retrieval_results", []),
        int(args.sparse_top_pages),
    )
    source_agreement = compute_source_agreement(
        dense_pages=dense_pages,
        sparse_pages=sparse_pages,
        args=args,
    )
    source_weights = compute_source_weights(
        agreement=source_agreement,
        dense_pages=dense_pages,
        sparse_pages=sparse_pages,
        args=args,
    )
    expansion_pages: list[tuple[str, int, float, int]] = []
    neighbor_pages: list[tuple[str, int, float]] = []
    neighbor_seed: dict[str, float] = {}
    if sparse_index is not None and int(args.expansion_top_pages) > 0:
        expansion_query_terms = build_expansion_query_terms(
            sparse_index=sparse_index,
            dense_pages=dense_pages,
            sparse_pages=sparse_pages,
            dense_weight=source_weights.dense_weight,
            sparse_weight=source_weights.sparse_weight,
            args=args,
        )
        expansion_pages = sparse_index.retrieve_sparse_query(
            expansion_query_terms,
            top_pages=int(args.expansion_top_pages),
            min_score=float(args.expansion_min_score),
        )
    if sparse_index is not None and int(args.neighbor_expansion_window) > 0:
        neighbor_pages, neighbor_seed = build_neighbor_expansion_pages(
            dense_pages=dense_pages,
            sparse_pages=sparse_pages,
            sparse_index=sparse_index,
            dense_weight=source_weights.dense_weight,
            sparse_weight=source_weights.sparse_weight,
            args=args,
        )
    dense_score_norm = minmax_by_uid(dense_pages)
    sparse_score_norm = minmax_by_uid(sparse_pages)
    expansion_score_norm = minmax_by_uid(expansion_pages)
    dense_doc_ranks = first_doc_rank_map(dense_pages)
    sparse_doc_ranks = first_doc_rank_map(sparse_pages)

    records: dict[str, PageRecord] = {}
    page_seed: dict[str, float] = defaultdict(float)
    doc_seed: dict[str, float] = defaultdict(float)

    for source_name, source_weight, source_pages, source_score_norm in [
        ("dense", source_weights.dense_weight, dense_pages, dense_score_norm),
        ("sparse", source_weights.sparse_weight, sparse_pages, sparse_score_norm),
        ("expansion", float(args.expansion_weight), expansion_pages, expansion_score_norm),
    ]:
        for doc_id, page_idx, score, rank in source_pages:
            uid = page_uid(doc_id, page_idx)
            record = records.get(uid)
            if record is None:
                record = PageRecord(doc_id=doc_id, page_idx=page_idx)
                records[uid] = record
            if source_name == "dense":
                record.dense_rank = rank
                record.dense_score = score
            elif source_name == "sparse":
                record.sparse_rank = rank
                record.sparse_score = score
            else:
                record.expansion_rank = rank
                record.expansion_score = score
            page_seed[uid] += source_weight / (float(args.rrf_k) + float(rank))
            page_seed[uid] += source_weight * float(args.score_seed_weight) * source_score_norm.get(uid, 0.0)

    for doc_id, page_idx, seed_score in neighbor_pages:
        uid = page_uid(doc_id, page_idx)
        record = records.get(uid)
        if record is None:
            record = PageRecord(doc_id=doc_id, page_idx=page_idx)
            records[uid] = record
        record.neighbor_seed_score += float(seed_score)
        page_seed[uid] += float(seed_score)

    for doc_id, rank in dense_doc_ranks.items():
        doc_seed[f"doc::{doc_id}"] += (
            float(args.doc_seed_weight)
            * source_weights.dense_weight
            / (float(args.rrf_k) + float(rank))
        )
    for doc_id, rank in sparse_doc_ranks.items():
        doc_seed[f"doc::{doc_id}"] += (
            float(args.doc_seed_weight)
            * source_weights.sparse_weight
            / (float(args.rrf_k) + float(rank))
        )

    transition_policy = build_transition_policy(
        records=records,
        agreement=source_agreement,
        dense_doc_ranks=dense_doc_ranks,
        sparse_doc_ranks=sparse_doc_ranks,
        args=args,
    )
    adjacent_policy = build_adjacent_coherence_policy(
        records=records,
        dense_doc_ranks=dense_doc_ranks,
        sparse_doc_ranks=sparse_doc_ranks,
        args=args,
    )
    pages_by_doc: dict[str, list[PageRecord]] = defaultdict(list)
    for record in records.values():
        pages_by_doc[record.doc_id].append(record)
    evidence_policy = build_evidence_community_policy(
        records=records,
        pages_by_doc=pages_by_doc,
        source_weights=source_weights,
        args=args,
    )
    position_doc_weights = build_position_doc_weights(
        dense_doc_ranks=dense_doc_ranks,
        sparse_doc_ranks=sparse_doc_ranks,
        source_weights=source_weights,
        args=args,
    )
    position_policy = build_position_evidence_policy(
        question=question,
        records=records,
        doc_page_counts=doc_page_catalog.page_counts if doc_page_catalog is not None else {},
        doc_page_number_indices=(
            doc_page_catalog.page_number_indices if doc_page_catalog is not None else {}
        ),
        doc_position_weights=position_doc_weights,
        args=args,
    )
    query_anchor_doc_weights = build_query_anchor_doc_weights(
        dense_doc_ranks=dense_doc_ranks,
        sparse_doc_ranks=sparse_doc_ranks,
        source_weights=source_weights,
        args=args,
    )
    query_anchor_policy = build_query_anchor_evidence_policy(
        question=question,
        records=records,
        page_texts=doc_page_catalog.page_texts if doc_page_catalog is not None else {},
        doc_anchor_weights=query_anchor_doc_weights,
        args=args,
    )

    graph: dict[str, dict[str, float]] = defaultdict(dict)
    base_page_to_doc_weight = effective_page_to_doc_edge_weight(args)
    base_doc_to_page_weight = effective_doc_to_page_edge_weight(args)
    for uid, record in records.items():
        doc_node = f"doc::{record.doc_id}"
        graph.setdefault(uid, {})
        graph.setdefault(doc_node, {})
        transition_multiplier = transition_policy.page_doc_multipliers.get(uid, 1.0)
        page_to_doc_weight = base_page_to_doc_weight
        doc_to_page_weight = base_doc_to_page_weight
        if bool(args.adaptive_transition_gate_page_doc):
            if bool(args.adaptive_transition_gate_page_to_doc):
                page_to_doc_weight *= transition_multiplier
            if bool(args.adaptive_transition_gate_doc_to_page):
                doc_to_page_weight *= transition_multiplier
        add_directed_edge(graph, uid, doc_node, page_to_doc_weight)
        add_directed_edge(graph, doc_node, uid, doc_to_page_weight)

    evidence_community_metadata = add_evidence_community_edges(
        graph=graph,
        records=records,
        policy=evidence_policy,
        args=args,
    )
    position_evidence_metadata = add_position_evidence_edges(
        graph=graph,
        policy=position_policy,
        args=args,
    )
    query_anchor_evidence_metadata = add_query_anchor_evidence_edges(
        graph=graph,
        policy=query_anchor_policy,
        args=args,
    )
    pdf_hyperlink_metadata = add_pdf_hyperlink_edges(
        graph=graph,
        records=records,
        pdf_hyperlink_graph=pdf_hyperlink_graph,
        args=args,
    )
    external_page_graph_metadata = add_external_page_graph_edges(
        graph=graph,
        records=records,
        external_page_graph=external_page_graph,
        args=args,
    )
    same_doc_window = int(args.same_doc_window)
    adjacent_edge_multipliers: list[float] = []
    if same_doc_window > 0 and float(args.adjacent_page_edge_weight) > 0:
        for doc_records in pages_by_doc.values():
            doc_records.sort(key=lambda item: item.page_idx)
            for left_idx, left in enumerate(doc_records):
                for right in doc_records[left_idx + 1 :]:
                    if right.page_idx - left.page_idx > same_doc_window:
                        break
                    adjacent_weight = float(args.adjacent_page_edge_weight)
                    if bool(args.adaptive_transition_gate_adjacent):
                        adjacent_weight *= (
                            transition_policy.page_doc_multipliers.get(left.page_uid, 1.0)
                            + transition_policy.page_doc_multipliers.get(right.page_uid, 1.0)
                        ) / 2.0
                    adaptive_adjacent_multiplier = adjacent_edge_multiplier(
                        left_uid=left.page_uid,
                        right_uid=right.page_uid,
                        policy=adjacent_policy,
                        args=args,
                    )
                    adjacent_weight *= adaptive_adjacent_multiplier
                    adjacent_edge_multipliers.append(adaptive_adjacent_multiplier)
                    add_undirected_edge(
                        graph,
                        left.page_uid,
                        right.page_uid,
                        adjacent_weight,
                    )

    restart_vector = build_restart_vector(
        page_seed=page_seed,
        doc_seed=doc_seed,
        dense_pages=dense_pages,
        sparse_pages=sparse_pages,
        expansion_pages=expansion_pages,
        dense_score_norm=dense_score_norm,
        sparse_score_norm=sparse_score_norm,
        expansion_score_norm=expansion_score_norm,
        neighbor_seed=neighbor_seed,
        dense_doc_ranks=dense_doc_ranks,
        sparse_doc_ranks=sparse_doc_ranks,
        source_weights=source_weights,
        agreement=source_agreement,
        args=args,
    )
    position_seed_node_count = add_position_evidence_restart(
        seed=restart_vector.seed,
        policy=position_policy,
        args=args,
    )
    query_anchor_seed_node_count = add_query_anchor_evidence_restart(
        seed=restart_vector.seed,
        policy=query_anchor_policy,
        args=args,
    )
    ppr = run_ppr(
        graph=graph,
        seed=restart_vector.seed,
        restart_prob=float(args.restart_prob),
        iters=int(args.ppr_iters),
    )

    page_seed_scaled = max_scale({uid: float(score) for uid, score in page_seed.items()})
    page_ppr_scaled = max_scale({uid: ppr.get(uid, 0.0) for uid in records})
    doc_ppr_scaled = max_scale({node: ppr.get(node, 0.0) for node in restart_vector.doc_ppr_nodes})

    ranked_records: list[tuple[PageRecord, float, float, float, float]] = []
    for uid, record in records.items():
        doc_node = f"doc::{record.doc_id}"
        seed_component = page_seed_scaled.get(uid, 0.0)
        page_ppr_component = page_ppr_scaled.get(uid, 0.0)
        doc_ppr_component = doc_ppr_scaled.get(doc_node, 0.0)
        final_score = (
            float(args.final_page_seed_weight) * seed_component
            + float(args.final_ppr_page_weight) * page_ppr_component
            + float(args.final_ppr_doc_weight) * doc_ppr_component
        )
        ranked_records.append(
            (record, float(final_score), seed_component, page_ppr_component, doc_ppr_component)
        )

    ranked_records.sort(
        key=lambda item: (
            -item[1],
            min(item[0].dense_rank or 10**9, item[0].sparse_rank or 10**9),
            item[0].dense_rank or 10**9,
            item[0].sparse_rank or 10**9,
            item[0].doc_id,
            item[0].page_idx,
        )
    )

    final_rows: list[list[object]] = []
    per_doc_counts: dict[str, int] = defaultdict(int)
    for record, final_score, _seed_component, _page_ppr_component, _doc_ppr_component in ranked_records:
        if int(args.per_doc_page_limit) > 0 and per_doc_counts[record.doc_id] >= int(args.per_doc_page_limit):
            continue
        per_doc_counts[record.doc_id] += 1
        final_rows.append([record.doc_id, int(record.page_idx), float(final_score)])
        if int(args.final_top_pages) > 0 and len(final_rows) >= int(args.final_top_pages):
            break

    trace_top = [
        {
            "page_uid": record.page_uid,
            "doc_id": record.doc_id,
            "page_idx": int(record.page_idx),
            "final_score": final_score,
            "page_seed_norm": seed_component,
            "page_ppr_norm": page_ppr_component,
            "doc_ppr_norm": doc_ppr_component,
            "dense_rank": record.dense_rank,
            "sparse_rank": record.sparse_rank,
            "expansion_rank": record.expansion_rank,
            "neighbor_seed_score": record.neighbor_seed_score,
        }
        for record, final_score, seed_component, page_ppr_component, doc_ppr_component in ranked_records[:20]
    ]
    candidate_doc_ids = {record.doc_id for record in records.values()}
    candidate_page_uids = set(records)
    metadata = {
        "qid": qid,
        "candidate_page_count": len(records),
        "candidate_doc_count": len(pages_by_doc),
        "dense_candidate_page_count": len(dense_pages),
        "sparse_candidate_page_count": len(sparse_pages),
        "expansion_candidate_page_count": len(expansion_pages),
        "neighbor_candidate_page_count": len(neighbor_pages),
        "output_page_count": len(final_rows),
        "graph_node_count": len(graph),
        "graph_edge_count_directed": sum(len(neighbors) for neighbors in graph.values()),
        "graph_edge_count_undirected": sum(len(neighbors) for neighbors in graph.values()) // 2,
        "graph_has_asymmetric_page_doc_edges": (
            abs(base_page_to_doc_weight - base_doc_to_page_weight) > 1e-12
            or (
                bool(args.adaptive_transition_gate_page_doc)
                and bool(args.adaptive_transition_gate_page_to_doc)
                != bool(args.adaptive_transition_gate_doc_to_page)
            )
        ),
        "top_graph_pages": trace_top,
        **source_weights.metadata,
        **restart_vector.metadata,
        **transition_policy.metadata,
        **adjacent_policy.metadata,
        **evidence_policy.metadata,
        **evidence_community_metadata,
        **position_policy.metadata,
        **position_evidence_metadata,
        **query_anchor_policy.metadata,
        **query_anchor_evidence_metadata,
        **pdf_hyperlink_metadata,
        **external_page_graph_metadata,
        "position_evidence_restart_seed_node_count": position_seed_node_count,
        "query_anchor_restart_seed_node_count": query_anchor_seed_node_count,
        "adaptive_adjacent_edge_count": len(adjacent_edge_multipliers),
        "mean_adaptive_adjacent_edge_multiplier": (
            statistics.fmean(adjacent_edge_multipliers) if adjacent_edge_multipliers else None
        ),
    }
    if gold_row is not None:
        doc_gold = gold_doc_ids(gold_row)
        page_gold = gold_page_uids(gold_row)
        metadata["candidate_has_gold_doc"] = bool(candidate_doc_ids & doc_gold)
        metadata["candidate_has_gold_page"] = bool(candidate_page_uids & page_gold) if page_gold else None
    return final_rows, metadata


def summarize_prediction_rows(rows: list[list[object]], gold_row: dict | None) -> dict:
    ranked_docs = ranked_docs_from_rows(rows)
    ranked_pages = ranked_pages_from_rows(rows)
    summary: dict[str, object] = {
        "doc_count": len(ranked_docs),
        "page_count": len(ranked_pages),
        "top_doc_ids": ranked_docs[:20],
    }
    if gold_row is not None:
        doc_gold = gold_doc_ids(gold_row)
        page_gold = gold_page_uids(gold_row)
        summary["gold_doc_ids"] = sorted(doc_gold)
        summary["gold_page_uids"] = sorted(page_gold)
        summary["reranked_first_gold_doc_rank"] = first_rank(ranked_docs, doc_gold)
        summary["reranked_first_gold_page_rank"] = first_rank(ranked_pages, page_gold) if page_gold else None
        summary["contains_gold_doc"] = summary["reranked_first_gold_doc_rank"] is not None
        summary["contains_gold_page"] = (
            summary["reranked_first_gold_page_rank"] is not None if page_gold else None
        )
    return summary


def prefixed_gold_ranks(prefix: str, summary: dict) -> dict[str, object]:
    return {
        f"{prefix}_first_gold_doc_rank": summary.get("reranked_first_gold_doc_rank"),
        f"{prefix}_first_gold_page_rank": summary.get("reranked_first_gold_page_rank"),
        f"{prefix}_contains_gold_doc": summary.get("contains_gold_doc"),
        f"{prefix}_contains_gold_page": summary.get("contains_gold_page"),
    }


def main() -> None:
    args = parse_args()

    dense_pred = load_prediction(Path(args.dense_prediction_json))
    sparse_pred = load_prediction(Path(args.sparse_prediction_json))
    sparse_index = None
    need_sparse_index = bool(args.splade_index_pt) and (
        int(args.expansion_top_pages) > 0 or int(args.neighbor_expansion_window) > 0
    )
    if int(args.neighbor_expansion_window) > 0 and not args.splade_index_pt:
        raise ValueError("--neighbor-expansion-window requires --splade-index-pt for page-catalog lookup.")
    if need_sparse_index:
        sparse_index = SparsePageIndex(
            Path(args.splade_index_pt),
            load_postings=int(args.expansion_top_pages) > 0,
        )
    pdf_hyperlink_graph = None
    if args.pdf_hyperlink_edges_jsonl:
        pdf_hyperlink_graph = load_pdf_hyperlink_graph(Path(args.pdf_hyperlink_edges_jsonl))
    external_page_graph = None
    if args.external_page_graph_jsonl:
        external_page_graph = load_external_page_graph(Path(args.external_page_graph_jsonl))
    common_qids = sorted(set(dense_pred) & set(sparse_pred))
    if not common_qids:
        raise ValueError("Dense and sparse predictions have no qids in common.")

    gold_rows = load_gold_rows(Path(args.gold), args.question_type) if args.gold else {}
    if gold_rows:
        qids = sorted(set(common_qids) & set(gold_rows))
        if not qids:
            raise ValueError("No qids remain after intersecting predictions with gold/filter.")
    else:
        qids = common_qids

    doc_page_catalog: DocPageCatalog | None = None
    if args.doc_pages_jsonl:
        doc_page_catalog = load_doc_page_catalog(
            Path(args.doc_pages_jsonl),
            text_fields=args.query_anchor_text_field,
            load_page_texts=str(args.query_anchor_evidence_mode) != "none",
        )

    fused_payload: dict[str, dict] = {}
    per_qid: list[dict] = []
    for qid in qids:
        gold_row = gold_rows.get(qid)
        final_rows, graph_metadata = build_qid_graph_ranking(
            qid=qid,
            dense_row=dense_pred[qid],
            sparse_row=sparse_pred[qid],
            args=args,
            sparse_index=sparse_index,
            gold_row=gold_row,
            doc_page_catalog=doc_page_catalog,
            pdf_hyperlink_graph=pdf_hyperlink_graph,
            external_page_graph=external_page_graph,
        )
        question = dense_pred[qid].get("question") or sparse_pred[qid].get("question", "")
        dense_source_summary = summarize_prediction_rows(
            page_rows_from_ranked_unique_pages(
                dense_pred[qid].get("page_retrieval_results", []),
                int(args.dense_top_pages),
            ),
            gold_row,
        )
        sparse_source_summary = summarize_prediction_rows(
            page_rows_from_ranked_unique_pages(
                sparse_pred[qid].get("page_retrieval_results", []),
                int(args.sparse_top_pages),
            ),
            gold_row,
        )
        row_summary = {
            "qid": qid,
            "question": question,
            **summarize_prediction_rows(final_rows, gold_row),
            **prefixed_gold_ranks("dense", dense_source_summary),
            **prefixed_gold_ranks("sparse", sparse_source_summary),
            "graph": {
                key: value
                for key, value in graph_metadata.items()
                if key not in {"top_graph_pages"}
            },
        }
        per_qid.append(row_summary)
        fused_payload[qid] = {
            "pred_answer": dense_pred[qid].get("pred_answer", ""),
            "page_retrieval_results": final_rows,
            "qid": qid,
            "question": question,
            "top_retrieved_docs": ranked_docs_from_rows(final_rows)[:10],
            "reranker_metadata": {
                "fusion_method": "graph_ppr",
                "dense_prediction_json": args.dense_prediction_json,
                "sparse_prediction_json": args.sparse_prediction_json,
                "dense_top_pages": int(args.dense_top_pages),
                "sparse_top_pages": int(args.sparse_top_pages),
                "final_top_pages": int(args.final_top_pages),
                "per_doc_page_limit": int(args.per_doc_page_limit),
                "rrf_k": float(args.rrf_k),
                "dense_weight": float(args.dense_weight),
                "sparse_weight": float(args.sparse_weight),
                "adaptive_source_weight_mode": args.adaptive_source_weight_mode,
                "adaptive_source_agreement_top_pages": int(args.adaptive_source_agreement_top_pages),
                "adaptive_source_top1_lookup_pages": int(args.adaptive_source_top1_lookup_pages),
                "adaptive_source_doc_overlap_weight": float(args.adaptive_source_doc_overlap_weight),
                "adaptive_source_page_overlap_weight": float(args.adaptive_source_page_overlap_weight),
                "adaptive_source_top1_weight": float(args.adaptive_source_top1_weight),
                "adaptive_source_strength": float(args.adaptive_source_strength),
                "adaptive_source_gamma": float(args.adaptive_source_gamma),
                "adaptive_source_min_dense_mult": float(args.adaptive_source_min_dense_mult),
                "adaptive_source_max_dense_mult": float(args.adaptive_source_max_dense_mult),
                "adaptive_source_min_sparse_mult": float(args.adaptive_source_min_sparse_mult),
                "adaptive_source_max_sparse_mult": float(args.adaptive_source_max_sparse_mult),
                "adaptive_source_reciprocal_top_pages": int(
                    args.adaptive_source_reciprocal_top_pages
                ),
                "adaptive_source_reciprocal_lookup_pages": int(
                    args.adaptive_source_reciprocal_lookup_pages
                ),
                "adaptive_source_reciprocal_doc_weight": float(
                    args.adaptive_source_reciprocal_doc_weight
                ),
                "adaptive_source_reciprocal_page_weight": float(
                    args.adaptive_source_reciprocal_page_weight
                ),
                "adaptive_source_reciprocal_min_mult": float(
                    args.adaptive_source_reciprocal_min_mult
                ),
                "adaptive_source_reciprocal_max_mult": float(
                    args.adaptive_source_reciprocal_max_mult
                ),
                "adaptive_source_reciprocal_preserve_total": bool(
                    args.adaptive_source_reciprocal_preserve_total
                ),
                "adaptive_restart_mode": args.adaptive_restart_mode,
                "adaptive_restart_source_strength": float(args.adaptive_restart_source_strength),
                "adaptive_restart_gamma": float(args.adaptive_restart_gamma),
                "adaptive_restart_min_dense_mult": float(args.adaptive_restart_min_dense_mult),
                "adaptive_restart_max_dense_mult": float(args.adaptive_restart_max_dense_mult),
                "adaptive_restart_min_sparse_mult": float(args.adaptive_restart_min_sparse_mult),
                "adaptive_restart_max_sparse_mult": float(args.adaptive_restart_max_sparse_mult),
                "adaptive_restart_page_seed_weight": float(args.adaptive_restart_page_seed_weight),
                "adaptive_restart_doc_seed_weight": float(args.adaptive_restart_doc_seed_weight),
                "adaptive_restart_min_doc_mult": float(args.adaptive_restart_min_doc_mult),
                "adaptive_restart_max_doc_mult": float(args.adaptive_restart_max_doc_mult),
                "adaptive_restart_neighbor_seed_weight": float(
                    args.adaptive_restart_neighbor_seed_weight
                ),
                "adaptive_restart_preserve_source_total": bool(
                    args.adaptive_restart_preserve_source_total
                ),
                "adaptive_transition_mode": args.adaptive_transition_mode,
                "adaptive_transition_agreement_min_mult": float(
                    args.adaptive_transition_agreement_min_mult
                ),
                "adaptive_transition_agreement_max_mult": float(
                    args.adaptive_transition_agreement_max_mult
                ),
                "adaptive_transition_local_min_mult": float(args.adaptive_transition_local_min_mult),
                "adaptive_transition_local_max_mult": float(args.adaptive_transition_local_max_mult),
                "adaptive_transition_local_weight": float(args.adaptive_transition_local_weight),
                "adaptive_transition_gate_adjacent": bool(args.adaptive_transition_gate_adjacent),
                "adaptive_transition_gate_page_doc": bool(args.adaptive_transition_gate_page_doc),
                "adaptive_transition_gate_page_to_doc": bool(
                    args.adaptive_transition_gate_page_to_doc
                ),
                "adaptive_transition_gate_doc_to_page": bool(
                    args.adaptive_transition_gate_doc_to_page
                ),
                "adaptive_adjacent_mode": args.adaptive_adjacent_mode,
                "adaptive_adjacent_min_mult": float(args.adaptive_adjacent_min_mult),
                "adaptive_adjacent_max_mult": float(args.adaptive_adjacent_max_mult),
                "adaptive_adjacent_power": float(args.adaptive_adjacent_power),
                "evidence_community_mode": args.evidence_community_mode,
                "evidence_community_source_edge_weight": float(
                    args.evidence_community_source_edge_weight
                ),
                "evidence_community_local_edge_weight": float(
                    args.evidence_community_local_edge_weight
                ),
                "evidence_community_min_page_support": float(
                    args.evidence_community_min_page_support
                ),
                "evidence_community_local_window": int(args.evidence_community_local_window),
                "evidence_community_both_source_bonus": float(
                    args.evidence_community_both_source_bonus
                ),
                "evidence_community_support_power": float(
                    args.evidence_community_support_power
                ),
                "doc_pages_jsonl": args.doc_pages_jsonl,
                "position_evidence_mode": args.position_evidence_mode,
                "position_evidence_scope": args.position_evidence_scope,
                "position_evidence_doc_top_k": int(args.position_evidence_doc_top_k),
                "position_evidence_min_doc_support": float(
                    args.position_evidence_min_doc_support
                ),
                "position_evidence_edge_weight": float(args.position_evidence_edge_weight),
                "position_evidence_restart_weight": float(args.position_evidence_restart_weight),
                "position_evidence_explicit_page_edge_weight": float(
                    args.position_evidence_explicit_page_edge_weight
                ),
                "position_evidence_explicit_page_restart_weight": float(
                    args.position_evidence_explicit_page_restart_weight
                ),
                "position_evidence_early_frac": float(args.position_evidence_early_frac),
                "position_evidence_late_frac": float(args.position_evidence_late_frac),
                "query_anchor_evidence_mode": args.query_anchor_evidence_mode,
                "query_anchor_text_field": list(args.query_anchor_text_field),
                "query_anchor_scope": args.query_anchor_scope,
                "query_anchor_doc_top_k": int(args.query_anchor_doc_top_k),
                "query_anchor_min_doc_support": float(args.query_anchor_min_doc_support),
                "query_anchor_edge_weight": float(args.query_anchor_edge_weight),
                "query_anchor_restart_weight": float(args.query_anchor_restart_weight),
                "query_anchor_max_anchors": int(args.query_anchor_max_anchors),
                "query_anchor_min_entity_len": int(args.query_anchor_min_entity_len),
                "query_anchor_weight_mode": args.query_anchor_weight_mode,
                "query_anchor_min_node_weight": float(args.query_anchor_min_node_weight),
                "query_anchor_max_page_matches": int(args.query_anchor_max_page_matches),
                "query_anchor_reasoning_mode": args.query_anchor_reasoning_mode,
                "query_anchor_financial_bundle_weight": float(
                    args.query_anchor_financial_bundle_weight
                ),
                "query_anchor_financial_min_slot_types": int(
                    args.query_anchor_financial_min_slot_types
                ),
                "query_anchor_financial_max_page_matches": int(
                    args.query_anchor_financial_max_page_matches
                ),
                "query_anchor_financial_table_bonus": float(
                    args.query_anchor_financial_table_bonus
                ),
                "query_anchor_financial_table_min_score": float(
                    args.query_anchor_financial_table_min_score
                ),
                "query_anchor_constraint_bundle_weight": float(
                    args.query_anchor_constraint_bundle_weight
                ),
                "query_anchor_constraint_min_slot_types": int(
                    args.query_anchor_constraint_min_slot_types
                ),
                "query_anchor_constraint_max_page_matches": int(
                    args.query_anchor_constraint_max_page_matches
                ),
                "query_anchor_constraint_max_slot_page_matches": int(
                    args.query_anchor_constraint_max_slot_page_matches
                ),
                "query_anchor_constraint_specificity_floor": float(
                    args.query_anchor_constraint_specificity_floor
                ),
                "query_anchor_constraint_table_bonus": float(
                    args.query_anchor_constraint_table_bonus
                ),
                "query_anchor_constraint_table_min_score": float(
                    args.query_anchor_constraint_table_min_score
                ),
                "pdf_hyperlink_edges_jsonl": args.pdf_hyperlink_edges_jsonl,
                "pdf_hyperlink_edge_weight": float(args.pdf_hyperlink_edge_weight),
                "pdf_hyperlink_direction": args.pdf_hyperlink_direction,
                "pdf_hyperlink_weight_mode": args.pdf_hyperlink_weight_mode,
                "pdf_hyperlink_max_edges_per_source": int(args.pdf_hyperlink_max_edges_per_source),
                "pdf_hyperlink_source_top_k": int(args.pdf_hyperlink_source_top_k),
                "pdf_hyperlink_target_doc_top_k": int(args.pdf_hyperlink_target_doc_top_k),
                "pdf_hyperlink_query_support_weight_mode": (
                    args.pdf_hyperlink_query_support_weight_mode
                ),
                "external_page_graph_jsonl": args.external_page_graph_jsonl,
                "external_page_graph_edge_weight": float(args.external_page_graph_edge_weight),
                "external_page_graph_direction": args.external_page_graph_direction,
                "external_page_graph_weight_mode": args.external_page_graph_weight_mode,
                "external_page_graph_max_edges_per_source": int(
                    args.external_page_graph_max_edges_per_source
                ),
                "external_page_graph_source_top_k": int(args.external_page_graph_source_top_k),
                "external_page_graph_target_top_k": int(args.external_page_graph_target_top_k),
                "splade_index_pt": args.splade_index_pt,
                "expansion_top_pages": int(args.expansion_top_pages),
                "expand_from_top_dense_pages": int(args.expand_from_top_dense_pages),
                "expand_from_top_sparse_pages": int(args.expand_from_top_sparse_pages),
                "expansion_weight": float(args.expansion_weight),
                "expansion_source_term_topk": int(args.expansion_source_term_topk),
                "expansion_query_topk_terms": int(args.expansion_query_topk_terms),
                "expansion_min_score": float(args.expansion_min_score),
                "score_seed_weight": float(args.score_seed_weight),
                "doc_seed_weight": float(args.doc_seed_weight),
                "restart_prob": float(args.restart_prob),
                "ppr_iters": int(args.ppr_iters),
                "page_doc_edge_weight": float(args.page_doc_edge_weight),
                "page_to_doc_edge_weight": effective_page_to_doc_edge_weight(args),
                "doc_to_page_edge_weight": effective_doc_to_page_edge_weight(args),
                "adjacent_page_edge_weight": float(args.adjacent_page_edge_weight),
                "same_doc_window": int(args.same_doc_window),
                "final_page_seed_weight": float(args.final_page_seed_weight),
                "final_ppr_page_weight": float(args.final_ppr_page_weight),
                "final_ppr_doc_weight": float(args.final_ppr_doc_weight),
                "graph": {
                    key: value
                    for key, value in graph_metadata.items()
                    if key not in {"candidate_has_gold_doc", "candidate_has_gold_page"}
                },
            },
        }

    summary: dict[str, object] = {
        "fusion_method": "graph_ppr",
        "qid_count": len(qids),
        "dense_prediction_json": args.dense_prediction_json,
        "sparse_prediction_json": args.sparse_prediction_json,
        "gold": args.gold,
        "question_type": args.question_type,
        "dense_top_pages": int(args.dense_top_pages),
        "sparse_top_pages": int(args.sparse_top_pages),
        "final_top_pages": int(args.final_top_pages),
        "per_doc_page_limit": int(args.per_doc_page_limit),
        "rrf_k": float(args.rrf_k),
        "dense_weight": float(args.dense_weight),
        "sparse_weight": float(args.sparse_weight),
        "adaptive_source_weight_mode": args.adaptive_source_weight_mode,
        "adaptive_source_agreement_top_pages": int(args.adaptive_source_agreement_top_pages),
        "adaptive_source_top1_lookup_pages": int(args.adaptive_source_top1_lookup_pages),
        "adaptive_source_doc_overlap_weight": float(args.adaptive_source_doc_overlap_weight),
        "adaptive_source_page_overlap_weight": float(args.adaptive_source_page_overlap_weight),
        "adaptive_source_top1_weight": float(args.adaptive_source_top1_weight),
        "adaptive_source_strength": float(args.adaptive_source_strength),
        "adaptive_source_gamma": float(args.adaptive_source_gamma),
        "adaptive_source_min_dense_mult": float(args.adaptive_source_min_dense_mult),
        "adaptive_source_max_dense_mult": float(args.adaptive_source_max_dense_mult),
        "adaptive_source_min_sparse_mult": float(args.adaptive_source_min_sparse_mult),
        "adaptive_source_max_sparse_mult": float(args.adaptive_source_max_sparse_mult),
        "adaptive_source_reciprocal_top_pages": int(args.adaptive_source_reciprocal_top_pages),
        "adaptive_source_reciprocal_lookup_pages": int(args.adaptive_source_reciprocal_lookup_pages),
        "adaptive_source_reciprocal_doc_weight": float(args.adaptive_source_reciprocal_doc_weight),
        "adaptive_source_reciprocal_page_weight": float(args.adaptive_source_reciprocal_page_weight),
        "adaptive_source_reciprocal_min_mult": float(args.adaptive_source_reciprocal_min_mult),
        "adaptive_source_reciprocal_max_mult": float(args.adaptive_source_reciprocal_max_mult),
        "adaptive_source_reciprocal_preserve_total": bool(
            args.adaptive_source_reciprocal_preserve_total
        ),
        "adaptive_restart_mode": args.adaptive_restart_mode,
        "adaptive_restart_source_strength": float(args.adaptive_restart_source_strength),
        "adaptive_restart_gamma": float(args.adaptive_restart_gamma),
        "adaptive_restart_min_dense_mult": float(args.adaptive_restart_min_dense_mult),
        "adaptive_restart_max_dense_mult": float(args.adaptive_restart_max_dense_mult),
        "adaptive_restart_min_sparse_mult": float(args.adaptive_restart_min_sparse_mult),
        "adaptive_restart_max_sparse_mult": float(args.adaptive_restart_max_sparse_mult),
        "adaptive_restart_page_seed_weight": float(args.adaptive_restart_page_seed_weight),
        "adaptive_restart_doc_seed_weight": float(args.adaptive_restart_doc_seed_weight),
        "adaptive_restart_min_doc_mult": float(args.adaptive_restart_min_doc_mult),
        "adaptive_restart_max_doc_mult": float(args.adaptive_restart_max_doc_mult),
        "adaptive_restart_neighbor_seed_weight": float(args.adaptive_restart_neighbor_seed_weight),
        "adaptive_restart_preserve_source_total": bool(args.adaptive_restart_preserve_source_total),
        "adaptive_transition_mode": args.adaptive_transition_mode,
        "adaptive_transition_agreement_min_mult": float(args.adaptive_transition_agreement_min_mult),
        "adaptive_transition_agreement_max_mult": float(args.adaptive_transition_agreement_max_mult),
        "adaptive_transition_local_min_mult": float(args.adaptive_transition_local_min_mult),
        "adaptive_transition_local_max_mult": float(args.adaptive_transition_local_max_mult),
        "adaptive_transition_local_weight": float(args.adaptive_transition_local_weight),
        "adaptive_transition_gate_adjacent": bool(args.adaptive_transition_gate_adjacent),
        "adaptive_transition_gate_page_doc": bool(args.adaptive_transition_gate_page_doc),
        "adaptive_transition_gate_page_to_doc": bool(args.adaptive_transition_gate_page_to_doc),
        "adaptive_transition_gate_doc_to_page": bool(args.adaptive_transition_gate_doc_to_page),
        "adaptive_adjacent_mode": args.adaptive_adjacent_mode,
        "adaptive_adjacent_min_mult": float(args.adaptive_adjacent_min_mult),
        "adaptive_adjacent_max_mult": float(args.adaptive_adjacent_max_mult),
        "adaptive_adjacent_power": float(args.adaptive_adjacent_power),
        "evidence_community_mode": args.evidence_community_mode,
        "evidence_community_source_edge_weight": float(args.evidence_community_source_edge_weight),
        "evidence_community_local_edge_weight": float(args.evidence_community_local_edge_weight),
        "evidence_community_min_page_support": float(args.evidence_community_min_page_support),
        "evidence_community_local_window": int(args.evidence_community_local_window),
        "evidence_community_both_source_bonus": float(args.evidence_community_both_source_bonus),
        "evidence_community_support_power": float(args.evidence_community_support_power),
        "doc_pages_jsonl": args.doc_pages_jsonl,
        "doc_page_count_doc_count": (
            len(doc_page_catalog.page_counts) if doc_page_catalog is not None else 0
        ),
        "doc_page_number_lookup_doc_count": (
            len(doc_page_catalog.page_number_indices) if doc_page_catalog is not None else 0
        ),
        "doc_page_text_page_count": (
            len(doc_page_catalog.page_texts) if doc_page_catalog is not None else 0
        ),
        "position_evidence_mode": args.position_evidence_mode,
        "position_evidence_scope": args.position_evidence_scope,
        "position_evidence_doc_top_k": int(args.position_evidence_doc_top_k),
        "position_evidence_min_doc_support": float(args.position_evidence_min_doc_support),
        "position_evidence_edge_weight": float(args.position_evidence_edge_weight),
        "position_evidence_restart_weight": float(args.position_evidence_restart_weight),
        "position_evidence_explicit_page_edge_weight": float(
            args.position_evidence_explicit_page_edge_weight
        ),
        "position_evidence_explicit_page_restart_weight": float(
            args.position_evidence_explicit_page_restart_weight
        ),
        "position_evidence_early_frac": float(args.position_evidence_early_frac),
        "position_evidence_late_frac": float(args.position_evidence_late_frac),
        "query_anchor_evidence_mode": args.query_anchor_evidence_mode,
        "query_anchor_text_field": list(args.query_anchor_text_field),
        "query_anchor_scope": args.query_anchor_scope,
        "query_anchor_doc_top_k": int(args.query_anchor_doc_top_k),
        "query_anchor_min_doc_support": float(args.query_anchor_min_doc_support),
        "query_anchor_edge_weight": float(args.query_anchor_edge_weight),
        "query_anchor_restart_weight": float(args.query_anchor_restart_weight),
        "query_anchor_max_anchors": int(args.query_anchor_max_anchors),
        "query_anchor_min_entity_len": int(args.query_anchor_min_entity_len),
        "query_anchor_weight_mode": args.query_anchor_weight_mode,
        "query_anchor_min_node_weight": float(args.query_anchor_min_node_weight),
        "query_anchor_max_page_matches": int(args.query_anchor_max_page_matches),
        "query_anchor_reasoning_mode": args.query_anchor_reasoning_mode,
        "query_anchor_financial_bundle_weight": float(args.query_anchor_financial_bundle_weight),
        "query_anchor_financial_min_slot_types": int(args.query_anchor_financial_min_slot_types),
        "query_anchor_financial_max_page_matches": int(args.query_anchor_financial_max_page_matches),
        "query_anchor_financial_table_bonus": float(args.query_anchor_financial_table_bonus),
        "query_anchor_financial_table_min_score": float(args.query_anchor_financial_table_min_score),
        "query_anchor_constraint_bundle_weight": float(args.query_anchor_constraint_bundle_weight),
        "query_anchor_constraint_min_slot_types": int(args.query_anchor_constraint_min_slot_types),
        "query_anchor_constraint_max_page_matches": int(args.query_anchor_constraint_max_page_matches),
        "query_anchor_constraint_max_slot_page_matches": int(
            args.query_anchor_constraint_max_slot_page_matches
        ),
        "query_anchor_constraint_specificity_floor": float(
            args.query_anchor_constraint_specificity_floor
        ),
        "query_anchor_constraint_table_bonus": float(args.query_anchor_constraint_table_bonus),
        "query_anchor_constraint_table_min_score": float(
            args.query_anchor_constraint_table_min_score
        ),
        "pdf_hyperlink_edges_jsonl": args.pdf_hyperlink_edges_jsonl,
        "pdf_hyperlink_edge_weight": float(args.pdf_hyperlink_edge_weight),
        "pdf_hyperlink_direction": args.pdf_hyperlink_direction,
        "pdf_hyperlink_weight_mode": args.pdf_hyperlink_weight_mode,
        "pdf_hyperlink_max_edges_per_source": int(args.pdf_hyperlink_max_edges_per_source),
        "pdf_hyperlink_source_top_k": int(args.pdf_hyperlink_source_top_k),
        "pdf_hyperlink_target_doc_top_k": int(args.pdf_hyperlink_target_doc_top_k),
        "pdf_hyperlink_query_support_weight_mode": args.pdf_hyperlink_query_support_weight_mode,
        "pdf_hyperlink_loaded_edge_count": (
            pdf_hyperlink_graph.edge_count if pdf_hyperlink_graph is not None else 0
        ),
        "pdf_hyperlink_loaded_source_page_count": (
            pdf_hyperlink_graph.source_page_count if pdf_hyperlink_graph is not None else 0
        ),
        "pdf_hyperlink_loaded_target_doc_count": (
            pdf_hyperlink_graph.target_doc_count if pdf_hyperlink_graph is not None else 0
        ),
        "external_page_graph_jsonl": args.external_page_graph_jsonl,
        "external_page_graph_edge_weight": float(args.external_page_graph_edge_weight),
        "external_page_graph_direction": args.external_page_graph_direction,
        "external_page_graph_weight_mode": args.external_page_graph_weight_mode,
        "external_page_graph_max_edges_per_source": int(
            args.external_page_graph_max_edges_per_source
        ),
        "external_page_graph_source_top_k": int(args.external_page_graph_source_top_k),
        "external_page_graph_target_top_k": int(args.external_page_graph_target_top_k),
        "external_page_graph_loaded_edge_count": (
            external_page_graph.edge_count if external_page_graph is not None else 0
        ),
        "external_page_graph_loaded_source_page_count": (
            external_page_graph.source_page_count if external_page_graph is not None else 0
        ),
        "external_page_graph_loaded_target_page_count": (
            external_page_graph.target_page_count if external_page_graph is not None else 0
        ),
        "external_page_graph_loaded_target_doc_count": (
            external_page_graph.target_doc_count if external_page_graph is not None else 0
        ),
        "splade_index_pt": args.splade_index_pt,
        "expansion_top_pages": int(args.expansion_top_pages),
        "expand_from_top_dense_pages": int(args.expand_from_top_dense_pages),
        "expand_from_top_sparse_pages": int(args.expand_from_top_sparse_pages),
        "expansion_weight": float(args.expansion_weight),
        "expansion_source_term_topk": int(args.expansion_source_term_topk),
        "expansion_query_topk_terms": int(args.expansion_query_topk_terms),
        "expansion_min_score": float(args.expansion_min_score),
        "score_seed_weight": float(args.score_seed_weight),
        "doc_seed_weight": float(args.doc_seed_weight),
        "restart_prob": float(args.restart_prob),
        "ppr_iters": int(args.ppr_iters),
        "page_doc_edge_weight": float(args.page_doc_edge_weight),
        "page_to_doc_edge_weight": effective_page_to_doc_edge_weight(args),
        "doc_to_page_edge_weight": effective_doc_to_page_edge_weight(args),
        "adjacent_page_edge_weight": float(args.adjacent_page_edge_weight),
        "same_doc_window": int(args.same_doc_window),
        "final_page_seed_weight": float(args.final_page_seed_weight),
        "final_ppr_page_weight": float(args.final_ppr_page_weight),
        "final_ppr_doc_weight": float(args.final_ppr_doc_weight),
        "mean_candidate_page_count": (
            statistics.fmean(float(row["graph"]["candidate_page_count"]) for row in per_qid)
            if per_qid
            else None
        ),
        "mean_candidate_doc_count": (
            statistics.fmean(float(row["graph"]["candidate_doc_count"]) for row in per_qid)
            if per_qid
            else None
        ),
        "mean_effective_dense_weight": (
            statistics.fmean(float(row["graph"]["effective_dense_weight"]) for row in per_qid)
            if per_qid
            else None
        ),
        "mean_effective_sparse_weight": (
            statistics.fmean(float(row["graph"]["effective_sparse_weight"]) for row in per_qid)
            if per_qid
            else None
        ),
        "mean_adaptive_source_agreement_score": (
            statistics.fmean(
                float(row["graph"].get("adaptive_source_agreement_score", 1.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_source_doc_overlap": (
            statistics.fmean(
                float(row["graph"].get("adaptive_source_doc_overlap", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_source_page_overlap": (
            statistics.fmean(
                float(row["graph"].get("adaptive_source_page_overlap", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "adaptive_source_dense_top1_in_sparse_lookup_count": sum(
            1
            for row in per_qid
            if row["graph"].get("adaptive_source_dense_top1_in_sparse_lookup") is True
        ),
        "mean_adaptive_source_dense_reliability": (
            statistics.fmean(
                float(row["graph"].get("adaptive_source_dense_reliability", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_source_sparse_reliability": (
            statistics.fmean(
                float(row["graph"].get("adaptive_source_sparse_reliability", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_source_dense_doc_reciprocal_support": (
            statistics.fmean(
                float(row["graph"].get("adaptive_source_dense_doc_reciprocal_support", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_source_sparse_doc_reciprocal_support": (
            statistics.fmean(
                float(row["graph"].get("adaptive_source_sparse_doc_reciprocal_support", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_restart_dense_reliability": (
            statistics.fmean(
                float(row["graph"].get("adaptive_restart_dense_reliability", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_restart_sparse_reliability": (
            statistics.fmean(
                float(row["graph"].get("adaptive_restart_sparse_reliability", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_restart_dense_doc_reciprocal_support": (
            statistics.fmean(
                float(row["graph"].get("adaptive_restart_dense_doc_reciprocal_support", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_restart_sparse_doc_reciprocal_support": (
            statistics.fmean(
                float(row["graph"].get("adaptive_restart_sparse_doc_reciprocal_support", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_effective_restart_dense_weight": (
            statistics.fmean(
                float(row["graph"].get("effective_restart_dense_weight", args.dense_weight))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_effective_restart_sparse_weight": (
            statistics.fmean(
                float(row["graph"].get("effective_restart_sparse_weight", args.sparse_weight))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_effective_restart_doc_seed_weight": (
            statistics.fmean(
                float(row["graph"].get("effective_restart_doc_seed_weight", args.doc_seed_weight))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_restart_doc_multiplier": (
            statistics.fmean(
                float(row["graph"].get("adaptive_restart_doc_multiplier", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_restart_vector_doc_node_count": (
            statistics.fmean(float(row["graph"].get("restart_vector_doc_node_count", 0)) for row in per_qid)
            if per_qid
            else None
        ),
        "mean_adaptive_transition_global_multiplier": (
            statistics.fmean(
                float(row["graph"].get("adaptive_transition_global_multiplier", 1.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_transition_local_reliability": (
            statistics.fmean(
                float(row["graph"].get("mean_adaptive_transition_local_reliability", 0.0))
                for row in per_qid
                if row["graph"].get("mean_adaptive_transition_local_reliability") is not None
            )
            if any(
                row["graph"].get("mean_adaptive_transition_local_reliability") is not None
                for row in per_qid
            )
            else None
        ),
        "mean_adaptive_transition_page_doc_multiplier": (
            statistics.fmean(
                float(row["graph"].get("mean_adaptive_transition_page_doc_multiplier", 1.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_adaptive_adjacent_page_reliability": (
            statistics.fmean(
                float(row["graph"].get("mean_adaptive_adjacent_page_reliability", 1.0))
                for row in per_qid
                if row["graph"].get("mean_adaptive_adjacent_page_reliability") is not None
            )
            if any(
                row["graph"].get("mean_adaptive_adjacent_page_reliability") is not None
                for row in per_qid
            )
            else None
        ),
        "mean_adaptive_adjacent_edge_multiplier": (
            statistics.fmean(
                float(row["graph"].get("mean_adaptive_adjacent_edge_multiplier", 1.0))
                for row in per_qid
                if row["graph"].get("mean_adaptive_adjacent_edge_multiplier") is not None
            )
            if any(
                row["graph"].get("mean_adaptive_adjacent_edge_multiplier") is not None
                for row in per_qid
            )
            else None
        ),
        "mean_adaptive_adjacent_edge_count": (
            statistics.fmean(
                float(row["graph"].get("adaptive_adjacent_edge_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_evidence_community_source_support": (
            statistics.fmean(
                float(row["graph"].get("mean_evidence_community_source_support", 0.0))
                for row in per_qid
                if row["graph"].get("mean_evidence_community_source_support") is not None
            )
            if any(
                row["graph"].get("mean_evidence_community_source_support") is not None
                for row in per_qid
            )
            else None
        ),
        "mean_evidence_community_local_support": (
            statistics.fmean(
                float(row["graph"].get("mean_evidence_community_local_support", 0.0))
                for row in per_qid
                if row["graph"].get("mean_evidence_community_local_support") is not None
            )
            if any(
                row["graph"].get("mean_evidence_community_local_support") is not None
                for row in per_qid
            )
            else None
        ),
        "mean_evidence_community_node_count": (
            statistics.fmean(
                float(row["graph"].get("evidence_community_node_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_evidence_community_edge_count_directed": (
            statistics.fmean(
                float(row["graph"].get("evidence_community_edge_count_directed", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "position_evidence_qid_count": sum(
            1
            for row in per_qid
            if int(row["graph"].get("position_evidence_active_role_count", 0)) > 0
        ),
        "mean_position_evidence_active_role_count": (
            statistics.fmean(
                float(row["graph"].get("position_evidence_active_role_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_position_evidence_page_match_count": (
            statistics.fmean(
                float(row["graph"].get("position_evidence_page_match_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_position_evidence_edge_count_directed": (
            statistics.fmean(
                float(row["graph"].get("position_evidence_edge_count_directed", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_position_evidence_doc_support_count": (
            statistics.fmean(
                float(row["graph"].get("position_evidence_doc_support_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_position_evidence_doc_support": (
            statistics.fmean(
                float(row["graph"].get("mean_position_evidence_doc_support", 0.0))
                for row in per_qid
                if row["graph"].get("mean_position_evidence_doc_support") is not None
            )
            if any(
                row["graph"].get("mean_position_evidence_doc_support") is not None
                for row in per_qid
            )
            else None
        ),
        "mean_position_evidence_restart_seed_node_count": (
            statistics.fmean(
                float(row["graph"].get("position_evidence_restart_seed_node_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "query_anchor_evidence_qid_count": sum(
            1
            for row in per_qid
            if int(row["graph"].get("query_anchor_matched_anchor_count", 0)) > 0
        ),
        "mean_query_anchor_active_anchor_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_active_anchor_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_matched_anchor_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_matched_anchor_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_page_match_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_page_match_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_edge_count_directed": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_edge_count_directed", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_node_weight": (
            statistics.fmean(
                float(row["graph"].get("mean_query_anchor_node_weight", 0.0))
                for row in per_qid
                if row["graph"].get("mean_query_anchor_node_weight") is not None
            )
            if any(row["graph"].get("mean_query_anchor_node_weight") is not None for row in per_qid)
            else None
        ),
        "mean_query_anchor_dropped_broad_anchor_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_dropped_broad_anchor_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_restart_seed_node_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_restart_seed_node_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "pdf_hyperlink_qid_count": sum(
            1
            for row in per_qid
            if int(row["graph"].get("pdf_hyperlink_edge_count_directed", 0)) > 0
        ),
        "mean_pdf_hyperlink_edge_count_directed": (
            statistics.fmean(
                float(row["graph"].get("pdf_hyperlink_edge_count_directed", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_pdf_hyperlink_source_page_count": (
            statistics.fmean(
                float(row["graph"].get("pdf_hyperlink_source_page_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_pdf_hyperlink_target_doc_count": (
            statistics.fmean(
                float(row["graph"].get("pdf_hyperlink_target_doc_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_pdf_hyperlink_raw_link_count": (
            statistics.fmean(
                float(row["graph"].get("pdf_hyperlink_raw_link_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_pdf_hyperlink_skipped_source_rank_gate": (
            statistics.fmean(
                float(row["graph"].get("pdf_hyperlink_skipped_source_rank_gate", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_pdf_hyperlink_skipped_target_doc_rank_gate": (
            statistics.fmean(
                float(row["graph"].get("pdf_hyperlink_skipped_target_doc_rank_gate", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "external_page_graph_qid_count": sum(
            1
            for row in per_qid
            if int(row["graph"].get("external_page_graph_edge_count_directed", 0)) > 0
        ),
        "mean_external_page_graph_edge_count_directed": (
            statistics.fmean(
                float(row["graph"].get("external_page_graph_edge_count_directed", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_external_page_graph_source_page_count": (
            statistics.fmean(
                float(row["graph"].get("external_page_graph_source_page_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_external_page_graph_target_page_count": (
            statistics.fmean(
                float(row["graph"].get("external_page_graph_target_page_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_external_page_graph_target_doc_count": (
            statistics.fmean(
                float(row["graph"].get("external_page_graph_target_doc_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "query_anchor_financial_reasoning_qid_count": sum(
            1
            for row in per_qid
            if bool(row["graph"].get("query_anchor_financial_reasoning_active", False))
        ),
        "mean_query_anchor_financial_bundle_page_match_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_financial_bundle_page_match_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_financial_table_score": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_financial_mean_table_score", 0.0))
                for row in per_qid
                if row["graph"].get("query_anchor_financial_mean_table_score") is not None
            )
            if any(
                row["graph"].get("query_anchor_financial_mean_table_score") is not None
                for row in per_qid
            )
            else None
        ),
        "query_anchor_constraint_bundle_qid_count": sum(
            1
            for row in per_qid
            if bool(row["graph"].get("query_anchor_constraint_bundle_active", False))
        ),
        "mean_query_anchor_constraint_bundle_page_match_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_constraint_bundle_page_match_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_constraint_active_slot_count": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_constraint_active_slot_count", 0.0))
                for row in per_qid
            )
            if per_qid
            else None
        ),
        "mean_query_anchor_constraint_slot_specificity": (
            statistics.fmean(
                float(row["graph"].get("query_anchor_constraint_mean_slot_specificity", 0.0))
                for row in per_qid
                if row["graph"].get("query_anchor_constraint_mean_slot_specificity") is not None
            )
            if any(
                row["graph"].get("query_anchor_constraint_mean_slot_specificity") is not None
                for row in per_qid
            )
            else None
        ),
        "per_qid": per_qid,
    }
    if gold_rows:
        doc_ranks = [row.get("reranked_first_gold_doc_rank") for row in per_qid]
        page_ranks = [row.get("reranked_first_gold_page_rank") for row in per_qid]
        dense_doc_ranks = [row.get("dense_first_gold_doc_rank") for row in per_qid]
        sparse_doc_ranks = [row.get("sparse_first_gold_doc_rank") for row in per_qid]
        dense_page_ranks = [row.get("dense_first_gold_page_rank") for row in per_qid]
        sparse_page_ranks = [row.get("sparse_first_gold_page_rank") for row in per_qid]
        candidate_doc_hits = [row["graph"].get("candidate_has_gold_doc") for row in per_qid]
        candidate_page_hits = [row["graph"].get("candidate_has_gold_page") for row in per_qid]
        dense_doc_hits = [row.get("dense_contains_gold_doc") for row in per_qid]
        sparse_doc_hits = [row.get("sparse_contains_gold_doc") for row in per_qid]
        dense_page_hits = [row.get("dense_contains_gold_page") for row in per_qid]
        sparse_page_hits = [row.get("sparse_contains_gold_page") for row in per_qid]
        summary["reranked_top4_doc_count"] = sum(
            1 for rank in doc_ranks if rank is not None and int(rank) <= 4
        )
        summary["reranked_top20_doc_count"] = sum(
            1 for rank in doc_ranks if rank is not None and int(rank) <= 20
        )
        summary["reranked_top4_page_count"] = sum(
            1 for rank in page_ranks if rank is not None and int(rank) <= 4
        )
        summary["reranked_top20_page_count"] = sum(
            1 for rank in page_ranks if rank is not None and int(rank) <= 20
        )
        summary["dense_top4_doc_count"] = sum(
            1 for rank in dense_doc_ranks if rank is not None and int(rank) <= 4
        )
        summary["dense_top20_doc_count"] = sum(
            1 for rank in dense_doc_ranks if rank is not None and int(rank) <= 20
        )
        summary["sparse_top4_doc_count"] = sum(
            1 for rank in sparse_doc_ranks if rank is not None and int(rank) <= 4
        )
        summary["sparse_top20_doc_count"] = sum(
            1 for rank in sparse_doc_ranks if rank is not None and int(rank) <= 20
        )
        summary["dense_top4_page_count"] = sum(
            1 for rank in dense_page_ranks if rank is not None and int(rank) <= 4
        )
        summary["dense_top20_page_count"] = sum(
            1 for rank in dense_page_ranks if rank is not None and int(rank) <= 20
        )
        summary["sparse_top4_page_count"] = sum(
            1 for rank in sparse_page_ranks if rank is not None and int(rank) <= 4
        )
        summary["sparse_top20_page_count"] = sum(
            1 for rank in sparse_page_ranks if rank is not None and int(rank) <= 20
        )
        summary["candidate_gold_doc_count"] = sum(1 for hit in candidate_doc_hits if hit is True)
        summary["candidate_gold_doc_miss_count"] = len(qids) - int(summary["candidate_gold_doc_count"])
        summary["candidate_gold_page_count"] = sum(1 for hit in candidate_page_hits if hit is True)
        summary["dense_candidate_gold_doc_count"] = sum(1 for hit in dense_doc_hits if hit is True)
        summary["sparse_candidate_gold_doc_count"] = sum(1 for hit in sparse_doc_hits if hit is True)
        summary["dense_candidate_gold_page_count"] = sum(1 for hit in dense_page_hits if hit is True)
        summary["sparse_candidate_gold_page_count"] = sum(1 for hit in sparse_page_hits if hit is True)
        summary["source_gold_doc_both_count"] = sum(
            1
            for dense_hit, sparse_hit in zip(dense_doc_hits, sparse_doc_hits)
            if dense_hit is True and sparse_hit is True
        )
        summary["source_gold_doc_dense_only_count"] = sum(
            1
            for dense_hit, sparse_hit in zip(dense_doc_hits, sparse_doc_hits)
            if dense_hit is True and sparse_hit is not True
        )
        summary["source_gold_doc_sparse_only_count"] = sum(
            1
            for dense_hit, sparse_hit in zip(dense_doc_hits, sparse_doc_hits)
            if dense_hit is not True and sparse_hit is True
        )
        summary["source_gold_doc_neither_count"] = sum(
            1
            for dense_hit, sparse_hit in zip(dense_doc_hits, sparse_doc_hits)
            if dense_hit is not True and sparse_hit is not True
        )
        summary["candidate_gold_doc_top4_count"] = summary["reranked_top4_doc_count"]
        summary["candidate_gold_doc_top20_count"] = summary["reranked_top20_doc_count"]
        summary["candidate_gold_doc_ranker_miss_top20_count"] = sum(
            1
            for candidate_hit, graph_rank in zip(candidate_doc_hits, doc_ranks)
            if candidate_hit is True and not (graph_rank is not None and int(graph_rank) <= 20)
        )
        summary["candidate_gold_doc_promotion_miss_top4_count"] = sum(
            1
            for graph_rank in doc_ranks
            if graph_rank is not None and 4 < int(graph_rank) <= 20
        )
        summary["graph_recovers_top4_doc_vs_dense_count"] = sum(
            1
            for dense_rank, graph_rank in zip(dense_doc_ranks, doc_ranks)
            if not (dense_rank is not None and int(dense_rank) <= 4)
            and graph_rank is not None
            and int(graph_rank) <= 4
        )
        summary["graph_loses_top4_doc_vs_dense_count"] = sum(
            1
            for dense_rank, graph_rank in zip(dense_doc_ranks, doc_ranks)
            if dense_rank is not None
            and int(dense_rank) <= 4
            and not (graph_rank is not None and int(graph_rank) <= 4)
        )
        summary["reranked_doc_rank_median"] = median_or_none(doc_ranks)  # type: ignore[arg-type]
        summary["reranked_page_rank_median"] = median_or_none(page_ranks)  # type: ignore[arg-type]
        summary["dense_doc_rank_median"] = median_or_none(dense_doc_ranks)  # type: ignore[arg-type]
        summary["sparse_doc_rank_median"] = median_or_none(sparse_doc_ranks)  # type: ignore[arg-type]

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(fused_payload, indent=2) + "\n", encoding="utf-8")

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(qids)}")
    if gold_rows:
        print(f"reranked_top4_doc_count: {summary['reranked_top4_doc_count']}")
        print(f"reranked_top20_doc_count: {summary['reranked_top20_doc_count']}")
        print(f"reranked_top4_page_count: {summary['reranked_top4_page_count']}")
        print(f"reranked_top20_page_count: {summary['reranked_top20_page_count']}")


if __name__ == "__main__":
    main()
