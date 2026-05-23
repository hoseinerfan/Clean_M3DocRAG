#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
            "lexical cues such as first page, last page, cover, references, or page/slide 17 "
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


def load_doc_page_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
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
            if raw_page_idx is None:
                continue
            try:
                page_idx = int(raw_page_idx)
            except (TypeError, ValueError):
                continue
            counts[doc_id] = max(counts.get(doc_id, 0), page_idx + 1)
    return counts


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


def extract_position_roles_from_question(question: str) -> tuple[dict[str, float], list[int]]:
    query = str(question or "").lower()
    role_weights: dict[str, float] = {}

    if re.search(r"\b(first|opening)\s+page\b|\bfront\s+page\b|\btitle\s+page\b|\bcover\b", query):
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

    explicit_page_indices: list[int] = []
    for match in re.finditer(
        r"\b(?:pages?|pg|p|slides?)\.?\s*(?:no\.?|number|#)?\s*(\d{1,4})\b",
        query,
    ):
        raw_page = int(match.group(1))
        target_idx = raw_page - 1 if raw_page > 0 else 0
        if target_idx not in explicit_page_indices:
            explicit_page_indices.append(target_idx)
        add_query_role(role_weights, f"page_{target_idx}", 1.0)

    return role_weights, explicit_page_indices


def record_position_roles(
    *,
    record: PageRecord,
    doc_page_counts: dict[str, int],
    early_frac: float,
    late_frac: float,
    explicit_page_indices: set[int],
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

    if page_idx in explicit_page_indices:
        roles.add(f"page_{page_idx}")
    return roles


def build_position_evidence_policy(
    *,
    question: str,
    records: dict[str, PageRecord],
    doc_page_counts: dict[str, int],
    doc_position_weights: dict[str, float],
    args: argparse.Namespace,
) -> PositionEvidencePolicy:
    mode = str(args.position_evidence_mode)
    active_role_weights: dict[str, float] = {}
    explicit_page_indices: list[int] = []
    role_page_weights: dict[str, dict[str, float]] = {}
    if mode != "none":
        active_role_weights, explicit_page_indices = extract_position_roles_from_question(question)
        explicit_set = set(explicit_page_indices)
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
                early_frac=early_frac,
                late_frac=late_frac,
                explicit_page_indices=explicit_set,
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
    doc_page_counts: dict[str, int] | None = None,
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
        doc_page_counts=doc_page_counts or {},
        doc_position_weights=position_doc_weights,
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
        "position_evidence_restart_seed_node_count": position_seed_node_count,
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

    doc_page_counts: dict[str, int] = {}
    if args.doc_pages_jsonl:
        doc_page_counts = load_doc_page_counts(Path(args.doc_pages_jsonl))

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
            doc_page_counts=doc_page_counts,
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
        "doc_page_count_doc_count": len(doc_page_counts),
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
