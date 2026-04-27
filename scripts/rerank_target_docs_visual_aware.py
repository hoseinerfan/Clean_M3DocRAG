#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QUERY_TOKEN_FILTER_CHOICES = ("full", "drop_pad_like", "semantic_only")
BIG_RANK = 10**12


@dataclass(frozen=True)
class WeightConfig:
    base: float
    visual: float
    non_visual: float
    balance: float


@dataclass
class PageFeature:
    doc_id: str
    page_idx: int
    page_uid: str
    base_page_score: float
    visual_page_score: float
    non_visual_page_score: float
    visual_avg_score: float
    non_visual_avg_score: float
    balance_score: float
    visual_alignment_count: int
    visual_alignment_ratio: float
    non_visual_alignment_count: int
    non_visual_alignment_ratio: float
    visual_query_token_count: int
    non_visual_query_token_count: int
    visual_patch_count: int
    non_visual_patch_count: int


def parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError(f"No numeric values found in {raw!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental visual-aware reranker over an explicit target-doc subset. "
            "This helper does not modify the baseline retrieval code; it recomputes "
            "page-local MaxSim features and reranks candidate docs/pages offline."
        )
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Free-form query text")
    query_group.add_argument("--qid", help="Benchmark qid to load from MMQA_<split>.jsonl")

    parser.add_argument("--data_name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--gold", help="Optional MMQA_<split>.jsonl override for --qid mode")
    parser.add_argument("--embedding_name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--query_token_filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
        help="Match the retrieval ablation whose query tokens you want to rerank.",
    )
    parser.add_argument(
        "--ignore-pad-scores-in-final-ranking",
        action="store_true",
        help="Keep PAD tokens in the encoded query but exclude them from the final page-score sum.",
    )
    parser.add_argument(
        "--nonspatial-token-position",
        default="suffix",
        choices=["prefix", "suffix"],
        help=(
            "How to interpret the extra non-spatial tokens in a page embedding. "
            "Use 'suffix' for the current ColPali page layout."
        ),
    )
    parser.add_argument("--retrieval_model_name_or_path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval_adapter_model_name_or_path", default="colpali-v1.2")

    parser.add_argument(
        "--splice-query-token-labels",
        required=True,
        help="Query-token visual/non-visual label file used for the reranker channels.",
    )
    parser.add_argument(
        "--splice-patch-labels-jsonl",
        required=True,
        help="Patch-level visual/non-visual label JSONL used for page-channel masking.",
    )

    parser.add_argument(
        "--candidate-doc-id",
        action="append",
        default=[],
        help="Target doc_id to rerank; pass multiple times.",
    )
    parser.add_argument(
        "--candidate-doc-ids-file",
        default="",
        help="Optional text/JSON file containing target doc_ids.",
    )
    parser.add_argument(
        "--candidate-page-uid",
        action="append",
        default=[],
        help="Optional explicit page_uid to include; pass multiple times.",
    )
    parser.add_argument(
        "--candidate-page-uids-file",
        default="",
        help="Optional text/JSON file containing explicit page_uids.",
    )
    parser.add_argument(
        "--baseline-pred",
        default="",
        help=(
            "Optional baseline prediction JSON from run_rag_m3docvqa.py. "
            "If provided, the helper can bootstrap candidate docs/pages from that retrieval output."
        ),
    )
    parser.add_argument(
        "--from-baseline-top-unique-docs",
        type=int,
        default=0,
        help="How many unique docs to pull from --baseline-pred. Set >0 to enable.",
    )
    parser.add_argument(
        "--from-baseline-top-pages",
        type=int,
        default=0,
        help="How many page rows to pull from --baseline-pred. Set >0 to enable exact page-pool reranking.",
    )
    parser.add_argument(
        "--replace-last-page-with",
        action="append",
        default=[],
        help=(
            "When bootstrapping a baseline page pool, replace the lowest-ranked baseline page(s) "
            "with these explicit page_uids while keeping the pool size fixed."
        ),
    )
    parser.add_argument(
        "--gold-doc-id",
        action="append",
        default=[],
        help="Optional gold doc_id override(s). If omitted in --qid mode, uses supporting_context doc_ids.",
    )
    parser.add_argument(
        "--gold-page-uid",
        action="append",
        default=[],
        help=(
            "Optional manual gold page_uid(s), e.g. <doc_id>_page<idx>. "
            "When provided, summaries include page-level ranks and grid search optimizes page rank."
        ),
    )

    parser.add_argument("--weight-base", type=float, default=1.0)
    parser.add_argument("--weight-visual", type=float, default=0.0)
    parser.add_argument("--weight-non-visual", type=float, default=0.0)
    parser.add_argument("--weight-balance", type=float, default=0.0)
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help=(
            "Search over weight combinations and pick the configuration that gives the "
            "best first-gold doc/page rank inside the target set."
        ),
    )
    parser.add_argument("--grid-base-values", default="1.0")
    parser.add_argument("--grid-visual-values", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--grid-non-visual-values", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--grid-balance-values", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument(
        "--report-topn",
        type=int,
        default=20,
        help="How many top reranked docs/pages to keep in the JSON summary.",
    )
    parser.add_argument("--output-json", default="", help="Optional summary JSON output path.")
    parser.add_argument(
        "--output-prediction-json",
        default="",
        help=(
            "Optional run_rag-style prediction JSON path. "
            "Writes page_retrieval_results ordered by the reranked page score."
        ),
    )
    return parser.parse_args()


def resolve_model_path(name_or_path: str) -> Path:
    from m3docrag.utils.paths import LOCAL_MODEL_DIR

    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    local_candidate = Path(LOCAL_MODEL_DIR) / name_or_path
    if local_candidate.exists():
        return local_candidate
    return candidate


def parse_string_list_file(path_str: str) -> list[str]:
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict):
            for key in ("doc_ids", "page_uids", "items", "values"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError(f"Unsupported JSON structure in {path}")

    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if item:
                values.append(item)
    return values


def load_baseline_candidate_pool(
    pred_path: str,
    qid: str | None,
    top_unique_docs: int,
    top_pages: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    if not pred_path:
        return [], [], {}
    if qid is None:
        raise ValueError("--baseline-pred requires --qid so the helper can select one query.")
    if top_unique_docs <= 0 and top_pages <= 0:
        return [], [], {}

    payload = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    if qid not in payload:
        raise KeyError(f"QID missing in baseline prediction file: {qid}")

    rows = payload[qid].get("page_retrieval_results", [])
    baseline_page_uids: list[str] = []
    ordered_doc_ids: list[str] = []
    doc_rank_map: dict[str, int] = {}
    for row_idx, row in enumerate(rows, start=1):
        doc_id = str(row[0]).strip()
        page_idx = int(row[1])
        page_uid = f"{doc_id}_page{page_idx}"
        if top_pages > 0 and row_idx <= top_pages:
            baseline_page_uids.append(page_uid)
        if not doc_id or doc_id in doc_rank_map:
            continue
        doc_rank_map[doc_id] = len(doc_rank_map) + 1
        ordered_doc_ids.append(doc_id)
        if top_unique_docs > 0 and len(ordered_doc_ids) >= top_unique_docs and (top_pages <= 0 or row_idx >= top_pages):
            break
    return ordered_doc_ids[:top_unique_docs] if top_unique_docs > 0 else [], baseline_page_uids, doc_rank_map


def apply_page_pool_replacements(
    baseline_page_uids: list[str],
    replacement_page_uids: list[str],
) -> list[str]:
    if not replacement_page_uids:
        return baseline_page_uids
    page_uids = list(baseline_page_uids)
    page_uid_set = set(page_uids)
    for replacement in replacement_page_uids:
        replacement = replacement.strip()
        if not replacement:
            continue
        if replacement in page_uid_set:
            continue
        if page_uids:
            dropped = page_uids.pop()
            page_uid_set.remove(dropped)
        page_uids.append(replacement)
        page_uid_set.add(replacement)
    return page_uids


def collect_candidate_sources(args: argparse.Namespace) -> tuple[list[str], set[str], dict[str, int]]:
    doc_ids: list[str] = []
    explicit_page_uids: set[str] = set()

    def add_doc_id(doc_id: str) -> None:
        doc_id = doc_id.strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)

    for value in args.candidate_doc_id:
        add_doc_id(value)
    for value in parse_string_list_file(args.candidate_doc_ids_file):
        add_doc_id(value)

    for value in args.candidate_page_uid:
        value = value.strip()
        if value:
            explicit_page_uids.add(value)
            add_doc_id(value.split("_page")[0])
    for value in parse_string_list_file(args.candidate_page_uids_file):
        value = value.strip()
        if value:
            explicit_page_uids.add(value)
            add_doc_id(value.split("_page")[0])

    baseline_doc_ids, baseline_page_uids, baseline_doc_rank_map = load_baseline_candidate_pool(
        pred_path=args.baseline_pred,
        qid=args.qid,
        top_unique_docs=args.from_baseline_top_unique_docs,
        top_pages=args.from_baseline_top_pages,
    )
    for doc_id in baseline_doc_ids:
        add_doc_id(doc_id)
    if baseline_page_uids:
        baseline_page_uids = apply_page_pool_replacements(
            baseline_page_uids=baseline_page_uids,
            replacement_page_uids=args.replace_last_page_with,
        )
        for page_uid in baseline_page_uids:
            explicit_page_uids.add(page_uid)
            add_doc_id(page_uid.split("_page")[0])

    if not doc_ids:
        raise ValueError(
            "No candidate docs/pages were provided. Use --candidate-doc-id, "
            "--candidate-doc-ids-file, --candidate-page-uid, --candidate-page-uids-file, "
            "or --baseline-pred with --from-baseline-top-unique-docs / --from-baseline-top-pages."
        )

    return doc_ids, explicit_page_uids, baseline_doc_rank_map


def load_doc_embeddings_for_doc_ids(
    doc_ids: list[str],
    embedding_name: str,
) -> dict[str, torch.Tensor]:
    import safetensors
    import torch

    from m3docrag.utils.paths import LOCAL_EMBEDDINGS_DIR

    emb_dir = Path(LOCAL_EMBEDDINGS_DIR) / embedding_name
    docid2embs: dict[str, torch.Tensor] = {}
    missing: list[str] = []

    for doc_id in doc_ids:
        emb_path = emb_dir / f"{doc_id}.safetensors"
        if not emb_path.exists():
            missing.append(doc_id)
            continue
        with safetensors.safe_open(emb_path, framework="pt", device="cpu") as handle:
            doc_embs = handle.get_tensor("embeddings")
        docid2embs[doc_id] = doc_embs.bfloat16()

    if missing:
        raise FileNotFoundError(
            f"Missing embeddings for {len(missing)} candidate docs in {emb_dir}: "
            f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}"
        )

    return docid2embs


def load_query_text_and_gold_doc_ids(args: argparse.Namespace) -> tuple[str, list[str]]:
    from scripts.plot_page_query_token_heatmaps import load_gold_row_from_qid

    gold_doc_ids = [str(value).strip() for value in args.gold_doc_id if str(value).strip()]
    if args.query:
        return args.query, gold_doc_ids

    gold_row = load_gold_row_from_qid(
        SimpleNamespace(
            qid=args.qid,
            gold=args.gold,
            data_name=args.data_name,
            split=args.split,
        )
    )
    if not gold_doc_ids:
        gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})
    return gold_row["question"], gold_doc_ids


def build_page_id_metadata(
    docid2embs: dict[str, torch.Tensor],
    *,
    explicit_page_uids: set[str],
    nonspatial_token_position: str,
) -> tuple[list[tuple[str, int]], dict[str, dict]]:
    from scripts.plot_page_query_token_heatmaps import infer_patch_grid

    page_specs: list[tuple[str, int]] = []
    page_meta: dict[str, dict] = {}
    for doc_id, doc_embs in docid2embs.items():
        for page_idx in range(len(doc_embs)):
            page_uid = f"{doc_id}_page{page_idx}"
            if explicit_page_uids and page_uid not in explicit_page_uids:
                continue
            page_token_count = int(doc_embs[page_idx].view(-1, doc_embs[page_idx].shape[-1]).shape[0])
            extra_tokens, grid_side = infer_patch_grid(page_token_count)
            n_spatial_patches = grid_side * grid_side
            page_specs.append((doc_id, page_idx))
            page_meta[page_uid] = {
                "page_uid": page_uid,
                "page_id": f"{doc_id}:{page_idx}",
                "page_token_count": page_token_count,
                "extra_tokens": extra_tokens,
                "grid_side": grid_side,
                "n_spatial_patches": n_spatial_patches,
                "nonspatial_token_position": nonspatial_token_position,
            }
    if not page_specs:
        raise ValueError("No candidate pages remain after applying the explicit page filters.")
    return page_specs, page_meta


def load_patch_axis_classes_for_pages(
    labels_jsonl: str,
    page_meta: dict[str, dict],
) -> dict[str, list[str]]:
    from scripts.plot_page_query_patch_dot_grid import classify_patch_from_splice_row

    page_id_to_uid = {meta["page_id"]: page_uid for page_uid, meta in page_meta.items()}
    classes_by_uid = {
        page_uid: ["unknown"] * meta["n_spatial_patches"]
        for page_uid, meta in page_meta.items()
    }

    path = Path(labels_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            page_id = str(row.get("page_id", "")).strip()
            page_uid = page_id_to_uid.get(page_id)
            if page_uid is None:
                continue
            patch_index = int(row.get("patch_index", -1))
            classes = classes_by_uid[page_uid]
            if 0 <= patch_index < len(classes):
                classes[patch_index] = classify_patch_from_splice_row(row)
    return classes_by_uid


def build_page_token_classes(
    *,
    page_meta: dict,
    patch_axis_classes: list[str],
) -> list[str]:
    token_classes = ["unknown"] * page_meta["page_token_count"]
    offset = page_meta["extra_tokens"] if page_meta["nonspatial_token_position"] == "prefix" else 0
    for patch_idx, patch_class in enumerate(patch_axis_classes):
        token_idx = offset + patch_idx
        if 0 <= token_idx < len(token_classes):
            token_classes[token_idx] = patch_class
    return token_classes


def make_query_score_mask(
    *,
    query_raw_tokens: list[str],
    ignore_pad_scores_in_final_ranking: bool,
) -> torch.Tensor:
    import torch

    if not ignore_pad_scores_in_final_ranking:
        return torch.ones(len(query_raw_tokens), dtype=torch.bool)
    keep_mask = torch.tensor([token != "<pad>" for token in query_raw_tokens], dtype=torch.bool)
    if not keep_mask.any():
        raise ValueError("Ignoring PAD scores removed every scoring query token.")
    return keep_mask


def compute_page_feature(
    *,
    page_emb: torch.Tensor,
    query_emb: torch.Tensor,
    query_axis_classes: list[str],
    query_score_mask: torch.Tensor,
    page_token_classes: list[str],
    doc_id: str,
    page_idx: int,
) -> PageFeature:
    import torch

    score_matrix = page_emb @ query_emb.T
    full_best_scores, full_best_indices = score_matrix.max(dim=0)
    active_best_scores = full_best_scores[query_score_mask.to(full_best_scores.device)]
    base_page_score = float(active_best_scores.sum().item())

    visual_query_indices = [
        idx for idx, axis_class in enumerate(query_axis_classes)
        if axis_class == "visual" and bool(query_score_mask[idx])
    ]
    non_visual_query_indices = [
        idx for idx, axis_class in enumerate(query_axis_classes)
        if axis_class == "non_visual" and bool(query_score_mask[idx])
    ]

    visual_page_indices = [
        idx for idx, axis_class in enumerate(page_token_classes)
        if axis_class == "visual"
    ]
    non_visual_page_indices = [
        idx for idx, axis_class in enumerate(page_token_classes)
        if axis_class == "non_visual"
    ]

    def masked_channel_score(page_indices: list[int], query_indices: list[int]) -> float:
        if not page_indices or not query_indices:
            return 0.0
        page_index_tensor = torch.tensor(page_indices, device=score_matrix.device, dtype=torch.long)
        query_index_tensor = torch.tensor(query_indices, device=score_matrix.device, dtype=torch.long)
        masked_scores = score_matrix.index_select(0, page_index_tensor).index_select(1, query_index_tensor)
        return float(masked_scores.max(dim=0).values.sum().item())

    visual_page_score = masked_channel_score(visual_page_indices, visual_query_indices)
    non_visual_page_score = masked_channel_score(non_visual_page_indices, non_visual_query_indices)

    visual_avg_score = (
        visual_page_score / len(visual_query_indices) if visual_query_indices else 0.0
    )
    non_visual_avg_score = (
        non_visual_page_score / len(non_visual_query_indices) if non_visual_query_indices else 0.0
    )
    balance_score = (
        min(visual_avg_score, non_visual_avg_score)
        if visual_query_indices and non_visual_query_indices
        else 0.0
    )

    visual_alignment_count = 0
    for query_idx in visual_query_indices:
        best_page_token_idx = int(full_best_indices[query_idx])
        if page_token_classes[best_page_token_idx] == "visual":
            visual_alignment_count += 1
    non_visual_alignment_count = 0
    for query_idx in non_visual_query_indices:
        best_page_token_idx = int(full_best_indices[query_idx])
        if page_token_classes[best_page_token_idx] == "non_visual":
            non_visual_alignment_count += 1

    visual_alignment_ratio = (
        visual_alignment_count / len(visual_query_indices) if visual_query_indices else 0.0
    )
    non_visual_alignment_ratio = (
        non_visual_alignment_count / len(non_visual_query_indices) if non_visual_query_indices else 0.0
    )

    return PageFeature(
        doc_id=doc_id,
        page_idx=page_idx,
        page_uid=f"{doc_id}_page{page_idx}",
        base_page_score=base_page_score,
        visual_page_score=visual_page_score,
        non_visual_page_score=non_visual_page_score,
        visual_avg_score=visual_avg_score,
        non_visual_avg_score=non_visual_avg_score,
        balance_score=balance_score,
        visual_alignment_count=visual_alignment_count,
        visual_alignment_ratio=visual_alignment_ratio,
        non_visual_alignment_count=non_visual_alignment_count,
        non_visual_alignment_ratio=non_visual_alignment_ratio,
        visual_query_token_count=len(visual_query_indices),
        non_visual_query_token_count=len(non_visual_query_indices),
        visual_patch_count=len(visual_page_indices),
        non_visual_patch_count=len(non_visual_page_indices),
    )


def fused_page_score(feature: PageFeature, weights: WeightConfig) -> float:
    return (
        weights.base * feature.base_page_score
        + weights.visual * feature.visual_page_score
        + weights.non_visual * feature.non_visual_page_score
        + weights.balance * feature.balance_score
    )


def build_rankings(
    page_features: list[PageFeature],
    weights: WeightConfig,
    baseline_doc_rank_map: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    ranked_pages = sorted(
        page_features,
        key=lambda item: (
            fused_page_score(item, weights),
            item.base_page_score,
            item.visual_page_score,
            item.non_visual_page_score,
        ),
        reverse=True,
    )

    reranked_pages = []
    for rank, item in enumerate(ranked_pages, start=1):
        payload = asdict(item)
        payload["fused_page_score"] = fused_page_score(item, weights)
        payload["rank"] = rank
        reranked_pages.append(payload)

    best_page_by_doc: dict[str, dict] = {}
    for item in reranked_pages:
        doc_id = item["doc_id"]
        current = best_page_by_doc.get(doc_id)
        if current is None or item["fused_page_score"] > current["fused_doc_score"]:
            best_page_by_doc[doc_id] = {
                "doc_id": doc_id,
                "rank": None,
                "fused_doc_score": item["fused_page_score"],
                "best_page_uid": item["page_uid"],
                "best_page_idx": item["page_idx"],
                "best_page_base_score": item["base_page_score"],
                "best_page_visual_score": item["visual_page_score"],
                "best_page_non_visual_score": item["non_visual_page_score"],
                "best_page_balance_score": item["balance_score"],
                "baseline_doc_rank": baseline_doc_rank_map.get(doc_id),
            }

    reranked_docs = sorted(
        best_page_by_doc.values(),
        key=lambda item: (item["fused_doc_score"], item["best_page_base_score"]),
        reverse=True,
    )
    for rank, item in enumerate(reranked_docs, start=1):
        item["rank"] = rank

    return reranked_docs, reranked_pages


def summarize_gold_doc_ranks(
    reranked_docs: list[dict],
    gold_doc_ids: list[str],
) -> dict:
    gold_doc_id_set = set(gold_doc_ids)
    gold_hits = [item for item in reranked_docs if item["doc_id"] in gold_doc_id_set]
    first_gold_doc_rank = gold_hits[0]["rank"] if gold_hits else None
    return {
        "gold_doc_ids": gold_doc_ids,
        "first_gold_doc_rank": first_gold_doc_rank,
        "gold_doc_hits_at_4": sum(item["rank"] <= 4 for item in gold_hits),
        "gold_doc_hits_at_10": sum(item["rank"] <= 10 for item in gold_hits),
        "gold_doc_ranks": [
            {
                "doc_id": item["doc_id"],
                "rank": item["rank"],
                "best_page_uid": item["best_page_uid"],
                "fused_doc_score": item["fused_doc_score"],
            }
            for item in gold_hits
        ],
    }


def summarize_gold_page_ranks(
    reranked_pages: list[dict],
    gold_page_uids: list[str],
) -> dict:
    gold_page_uid_set = set(gold_page_uids)
    gold_hits = [item for item in reranked_pages if item["page_uid"] in gold_page_uid_set]
    first_gold_page_rank = gold_hits[0]["rank"] if gold_hits else None
    return {
        "gold_page_uids": gold_page_uids,
        "first_gold_page_rank": first_gold_page_rank,
        "gold_page_hits_at_4": sum(item["rank"] <= 4 for item in gold_hits),
        "gold_page_hits_at_10": sum(item["rank"] <= 10 for item in gold_hits),
        "gold_page_ranks": [
            {
                "page_uid": item["page_uid"],
                "rank": item["rank"],
                "doc_id": item["doc_id"],
                "page_idx": item["page_idx"],
                "fused_page_score": item["fused_page_score"],
            }
            for item in gold_hits
        ],
    }


def grid_search_weights(
    *,
    page_features: list[PageFeature],
    baseline_doc_rank_map: dict[str, int],
    gold_doc_ids: list[str],
    gold_page_uids: list[str],
    base_values: list[float],
    visual_values: list[float],
    non_visual_values: list[float],
    balance_values: list[float],
) -> tuple[WeightConfig, dict, list[dict]]:
    if not gold_doc_ids and not gold_page_uids:
        raise ValueError("--grid-search needs gold doc ids or gold page uids.")

    best_weights: WeightConfig | None = None
    best_summary: dict | None = None
    leaderboard: list[dict] = []

    for base, visual, non_visual, balance in itertools.product(
        base_values, visual_values, non_visual_values, balance_values
    ):
        weights = WeightConfig(
            base=base,
            visual=visual,
            non_visual=non_visual,
            balance=balance,
        )
        reranked_docs, _reranked_pages = build_rankings(
            page_features=page_features,
            weights=weights,
            baseline_doc_rank_map=baseline_doc_rank_map,
        )
        gold_doc_summary = summarize_gold_doc_ranks(reranked_docs, gold_doc_ids) if gold_doc_ids else None
        gold_page_summary = summarize_gold_page_ranks(_reranked_pages, gold_page_uids) if gold_page_uids else None

        if gold_page_summary is not None:
            first_rank = gold_page_summary["first_gold_page_rank"]
            hits_at_4 = gold_page_summary["gold_page_hits_at_4"]
            hits_at_10 = gold_page_summary["gold_page_hits_at_10"]
            rank_key = first_rank if first_rank is not None else BIG_RANK
            objective = (
                0 if rank_key <= 4 else 1,
                rank_key,
                -hits_at_4,
                -hits_at_10,
            )
            record = {
                "weights": asdict(weights),
                "first_gold_page_rank": first_rank,
                "gold_page_hits_at_4": hits_at_4,
                "gold_page_hits_at_10": hits_at_10,
                "objective": objective,
            }
            if gold_doc_summary is not None:
                record["first_gold_doc_rank"] = gold_doc_summary["first_gold_doc_rank"]
                record["gold_doc_hits_at_4"] = gold_doc_summary["gold_doc_hits_at_4"]
                record["gold_doc_hits_at_10"] = gold_doc_summary["gold_doc_hits_at_10"]
        else:
            assert gold_doc_summary is not None
            first_rank = gold_doc_summary["first_gold_doc_rank"]
            hits_at_4 = gold_doc_summary["gold_doc_hits_at_4"]
            hits_at_10 = gold_doc_summary["gold_doc_hits_at_10"]
            rank_key = first_rank if first_rank is not None else BIG_RANK
            objective = (
                0 if rank_key <= 4 else 1,
                rank_key,
                -hits_at_4,
                -hits_at_10,
            )
            record = {
                "weights": asdict(weights),
                "first_gold_doc_rank": first_rank,
                "gold_doc_hits_at_4": hits_at_4,
                "gold_doc_hits_at_10": hits_at_10,
                "objective": objective,
            }
        leaderboard.append(record)

        if best_summary is None or objective < tuple(best_summary["objective"]):
            best_summary = record
            best_weights = weights

    assert best_weights is not None
    assert best_summary is not None
    leaderboard = sorted(leaderboard, key=lambda item: tuple(item["objective"]))[:20]
    return best_weights, best_summary, leaderboard


def build_prediction_payload(
    *,
    qid: str | None,
    query: str,
    reranked_pages: list[dict],
    metadata: dict,
) -> dict:
    qkey = qid if qid is not None else "__custom_query__"
    rows = [
        [item["doc_id"], item["page_idx"], item["fused_page_score"]]
        for item in reranked_pages
    ]
    return {
        qkey: {
            "query": query,
            "page_retrieval_results": rows,
            "pred_answer": "",
            "time_retrieval": None,
            "time_qa": None,
            "reranker_metadata": metadata,
        }
    }


def main() -> None:
    args = parse_args()

    import torch

    from m3docrag.retrieval import ColPaliRetrievalModel
    from scripts.plot_page_query_patch_dot_grid import axis_class_counts, load_splice_query_axis_classes
    from scripts.plot_page_query_token_heatmaps import clean_token_label

    query_text, gold_doc_ids = load_query_text_and_gold_doc_ids(args)
    gold_page_uids = [str(value).strip() for value in args.gold_page_uid if str(value).strip()]
    candidate_doc_ids, explicit_page_uids, baseline_doc_rank_map = collect_candidate_sources(args)
    docid2embs = load_doc_embeddings_for_doc_ids(candidate_doc_ids, args.embedding_name)
    page_specs, page_meta = build_page_id_metadata(
        docid2embs=docid2embs,
        explicit_page_uids=explicit_page_uids,
        nonspatial_token_position=args.nonspatial_token_position,
    )
    patch_axis_classes_by_uid = load_patch_axis_classes_for_pages(
        labels_jsonl=args.splice_patch_labels_jsonl,
        page_meta=page_meta,
    )

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    query_meta = retrieval_model.encode_query_with_metadata(
        query=query_text,
        to_cpu=True,
        query_token_filter=args.query_token_filter,
    )
    query_emb = query_meta["embeddings"].float()
    query_raw_tokens = query_meta.get("raw_tokens", [])
    query_token_labels = [clean_token_label(token) for token in query_raw_tokens]
    query_axis_classes = load_splice_query_axis_classes(
        query_labels_path=args.splice_query_token_labels,
        qid=args.qid,
        query_token_labels=query_token_labels,
        query_raw_tokens=query_raw_tokens,
    )
    query_score_mask = make_query_score_mask(
        query_raw_tokens=query_raw_tokens,
        ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_emb = query_emb.to(device=device, dtype=torch.float32)

    page_features: list[PageFeature] = []
    with torch.no_grad():
        for doc_id, page_idx in page_specs:
            page_uid = f"{doc_id}_page{page_idx}"
            page_emb = docid2embs[doc_id][page_idx].view(-1, docid2embs[doc_id][page_idx].shape[-1]).to(
                device=device,
                dtype=torch.float32,
            )
            page_token_classes = build_page_token_classes(
                page_meta=page_meta[page_uid],
                patch_axis_classes=patch_axis_classes_by_uid[page_uid],
            )
            page_features.append(
                compute_page_feature(
                    page_emb=page_emb,
                    query_emb=query_emb,
                    query_axis_classes=query_axis_classes,
                    query_score_mask=query_score_mask,
                    page_token_classes=page_token_classes,
                    doc_id=doc_id,
                    page_idx=page_idx,
                )
            )

    if not page_features:
        raise ValueError("No candidate pages were scored.")

    if args.grid_search:
        best_weights, best_grid_record, grid_leaderboard = grid_search_weights(
            page_features=page_features,
            baseline_doc_rank_map=baseline_doc_rank_map,
            gold_doc_ids=gold_doc_ids,
            gold_page_uids=gold_page_uids,
            base_values=parse_float_list(args.grid_base_values),
            visual_values=parse_float_list(args.grid_visual_values),
            non_visual_values=parse_float_list(args.grid_non_visual_values),
            balance_values=parse_float_list(args.grid_balance_values),
        )
        weights = best_weights
    else:
        weights = WeightConfig(
            base=args.weight_base,
            visual=args.weight_visual,
            non_visual=args.weight_non_visual,
            balance=args.weight_balance,
        )
        best_grid_record = None
        grid_leaderboard = []

    reranked_docs, reranked_pages = build_rankings(
        page_features=page_features,
        weights=weights,
        baseline_doc_rank_map=baseline_doc_rank_map,
    )
    gold_summary = summarize_gold_doc_ranks(reranked_docs, gold_doc_ids)
    gold_page_summary = summarize_gold_page_ranks(reranked_pages, gold_page_uids)

    summary = {
        "qid": args.qid,
        "query": query_text,
        "embedding_name": args.embedding_name,
        "query_token_filter": args.query_token_filter,
        "ignore_pad_scores_in_final_ranking": args.ignore_pad_scores_in_final_ranking,
        "nonspatial_token_position": args.nonspatial_token_position,
        "candidate_doc_count": len(candidate_doc_ids),
        "candidate_page_count": len(page_features),
        "candidate_doc_ids": candidate_doc_ids,
        "explicit_page_uids": sorted(explicit_page_uids),
        "baseline_doc_rank_map": baseline_doc_rank_map,
        "query_token_labels": query_token_labels,
        "query_axis_classes": query_axis_classes,
        "query_axis_class_counts": axis_class_counts(query_axis_classes),
        "query_score_active_token_count": int(query_score_mask.sum().item()),
        "weights": asdict(weights),
        "grid_search": {
            "enabled": args.grid_search,
            "best": best_grid_record,
            "leaderboard": grid_leaderboard,
        },
        "gold_summary": gold_summary,
        "gold_page_summary": gold_page_summary,
        "top_reranked_docs": reranked_docs[: args.report_topn],
        "top_reranked_pages": reranked_pages[: args.report_topn],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_prediction_json:
        prediction_payload = build_prediction_payload(
            qid=args.qid,
            query=query_text,
            reranked_pages=reranked_pages,
            metadata={
                "weights": asdict(weights),
                "query_token_filter": args.query_token_filter,
                "ignore_pad_scores_in_final_ranking": args.ignore_pad_scores_in_final_ranking,
                "query_label_path": args.splice_query_token_labels,
                "patch_label_path": args.splice_patch_labels_jsonl,
                "grid_search": args.grid_search,
                "candidate_doc_count": len(candidate_doc_ids),
                "candidate_page_count": len(page_features),
                "gold_doc_ids": gold_doc_ids,
                "gold_page_uids": gold_page_uids,
            },
        )
        prediction_path = Path(args.output_prediction_json)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps(prediction_payload, indent=2) + "\n", encoding="utf-8")

    print(f"query_token_filter: {args.query_token_filter}")
    print(f"ignore_pad_scores_in_final_ranking: {args.ignore_pad_scores_in_final_ranking}")
    print(f"candidate_doc_count: {len(candidate_doc_ids)}")
    print(f"candidate_page_count: {len(page_features)}")
    print(f"query_axis_class_counts: {axis_class_counts(query_axis_classes)}")
    if gold_doc_ids:
        print(f"gold_doc_ids: {gold_doc_ids}")
        print(f"first_gold_doc_rank: {gold_summary['first_gold_doc_rank']}")
        print(f"gold_doc_hits_at_4: {gold_summary['gold_doc_hits_at_4']}")
    if gold_page_uids:
        print(f"gold_page_uids: {gold_page_uids}")
        print(f"first_gold_page_rank: {gold_page_summary['first_gold_page_rank']}")
        print(f"gold_page_hits_at_4: {gold_page_summary['gold_page_hits_at_4']}")
    print(f"weights: {asdict(weights)}")
    print("top_reranked_docs:")
    for item in reranked_docs[: min(args.report_topn, 10)]:
        print(
            f"  {item['rank']:>4} {item['doc_id']} "
            f"score={item['fused_doc_score']:.4f} page={item['best_page_uid']}"
        )
    if args.output_json:
        print(f"saved_summary: {args.output_json}")
    if args.output_prediction_json:
        print(f"saved_prediction: {args.output_prediction_json}")


if __name__ == "__main__":
    main()
