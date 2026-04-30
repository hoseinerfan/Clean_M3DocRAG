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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

QUERY_TOKEN_FILTER_CHOICES = ("full", "drop_pad_like", "semantic_only")
BASE_SCORE_SOURCE_CHOICES = (
    "exact_page_maxsim",
    "baseline_pred",
    "approx_page_maxsim_topk",
    "two_stage_page_maxsim",
)
APPROX_BASE_PAGE_TOKEN_SCORER_CHOICES = ("query_mean", "query_token_max")
APPROX_BASE_PAGE_TOKEN_SELECTOR_CHOICES = (
    "global_topk",
    "spatial_quadrant_mix",
    "query_label_mix",
    "soft_label_prior",
)
COARSE_SCORE_DTYPE_CHOICES = ("fp32", "bf16", "fp16")
BIG_RANK = 10**12
SOFT_VISUAL_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
}


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


def is_base_only_weights(weights: WeightConfig) -> bool:
    return (
        float(weights.visual) == 0.0
        and float(weights.non_visual) == 0.0
        and float(weights.balance) == 0.0
    )


def make_base_only_page_feature(
    *,
    doc_id: str,
    page_idx: int,
    base_page_score: float,
) -> PageFeature:
    return PageFeature(
        doc_id=doc_id,
        page_idx=page_idx,
        page_uid=f"{doc_id}_page{page_idx}",
        base_page_score=float(base_page_score),
        visual_page_score=0.0,
        non_visual_page_score=0.0,
        visual_avg_score=0.0,
        non_visual_avg_score=0.0,
        balance_score=0.0,
        visual_alignment_count=0,
        visual_alignment_ratio=0.0,
        non_visual_alignment_count=0,
        non_visual_alignment_ratio=0.0,
        visual_query_token_count=0,
        non_visual_query_token_count=0,
        visual_patch_count=0,
        non_visual_patch_count=0,
    )


def compute_coarse_page_token_scores(
    *,
    page_emb: torch.Tensor,
    query_emb: torch.Tensor,
    approx_page_token_scorer: str,
    coarse_score_dtype: str = "fp32",
) -> torch.Tensor:
    import torch

    if coarse_score_dtype == "fp32" or page_emb.device.type != "cuda":
        page_emb_coarse = page_emb
        query_emb_coarse = query_emb
    else:
        dtype = torch.bfloat16 if coarse_score_dtype == "bf16" else torch.float16
        page_emb_coarse = page_emb.to(dtype=dtype)
        query_emb_coarse = query_emb.to(dtype=dtype)

    if approx_page_token_scorer == "query_mean":
        coarse_query = query_emb_coarse.mean(dim=0)
        return (page_emb_coarse @ coarse_query).to(dtype=torch.float32)
    if approx_page_token_scorer == "query_token_max":
        score_matrix = page_emb_coarse @ query_emb_coarse.T
        return score_matrix.max(dim=1).values.to(dtype=torch.float32)
    raise ValueError(f"Unsupported approx_page_token_scorer: {approx_page_token_scorer!r}")


def compute_exact_base_page_score(
    *,
    page_emb: torch.Tensor,
    query_emb: torch.Tensor,
    query_score_mask: torch.Tensor,
) -> float:
    score_matrix = page_emb @ query_emb.T
    full_best_scores = score_matrix.max(dim=0).values
    active_best_scores = full_best_scores[query_score_mask.to(full_best_scores.device)]
    return float(active_best_scores.sum().item())


def compute_approx_base_page_score(
    *,
    page_emb: torch.Tensor,
    query_emb: torch.Tensor,
    query_score_mask: torch.Tensor,
    approx_page_token_topk: int,
    approx_page_token_scorer: str,
    approx_page_token_selector: str,
    approx_page_token_spatial_reserve: int,
    query_axis_classes: list[str] | None,
    query_token_labels: list[str] | None,
    page_token_classes: list[str] | None,
    approx_page_token_label_reserve: int,
    approx_page_token_soft_visual_query_weight: float,
    approx_page_token_soft_patch_visual_bonus: float,
    coarse_score_dtype: str = "fp32",
) -> float:
    pruned_page_emb = maybe_prune_page_tokens_for_base_only(
        page_emb=page_emb,
        query_emb=query_emb,
        approx_page_token_topk=approx_page_token_topk,
        approx_page_token_scorer=approx_page_token_scorer,
        approx_page_token_selector=approx_page_token_selector,
        approx_page_token_spatial_reserve=approx_page_token_spatial_reserve,
        query_axis_classes=query_axis_classes,
        query_token_labels=query_token_labels,
        page_token_classes=page_token_classes,
        approx_page_token_label_reserve=approx_page_token_label_reserve,
        approx_page_token_soft_visual_query_weight=approx_page_token_soft_visual_query_weight,
        approx_page_token_soft_patch_visual_bonus=approx_page_token_soft_patch_visual_bonus,
        coarse_score_dtype=coarse_score_dtype,
    )
    return compute_exact_base_page_score(
        page_emb=pruned_page_emb,
        query_emb=query_emb,
        query_score_mask=query_score_mask,
    )


def _topk_indices_from_scores(
    *,
    scores: torch.Tensor,
    k: int,
) -> list[int]:
    import torch

    if k <= 0:
        return []
    k = min(int(k), int(scores.shape[0]))
    if k <= 0:
        return []
    return torch.topk(scores, k=k, largest=True, sorted=False).indices.tolist()


def _infer_spatial_quadrant_groups(page_token_count: int) -> tuple[int, list[list[int]]]:
    prefix_tokens, grid_side = infer_patch_grid(page_token_count)
    if grid_side < 2:
        return prefix_tokens, []

    row_mid = grid_side // 2
    col_mid = grid_side // 2
    quadrants = [[] for _ in range(4)]

    for row in range(grid_side):
        for col in range(grid_side):
            quadrant_idx = 0
            if row >= row_mid:
                quadrant_idx += 2
            if col >= col_mid:
                quadrant_idx += 1
            quadrants[quadrant_idx].append(prefix_tokens + row * grid_side + col)

    return prefix_tokens, [group for group in quadrants if group]


def _is_informative_visual_query_token(token_label: str) -> bool:
    normalized = _normalize_query_axis_text(token_label)
    if not normalized:
        return False
    if normalized in {"[bos]", "[eos]", "[pad]", "[ws]"}:
        return False
    if normalized in SOFT_VISUAL_QUERY_STOPWORDS:
        return False
    if not any(ch.isalnum() for ch in normalized):
        return False
    return True


def _select_informative_visual_query_indices(
    *,
    query_axis_classes: list[str] | None,
    query_token_labels: list[str] | None,
) -> list[int]:
    if (
        query_axis_classes is None
        or query_token_labels is None
        or len(query_axis_classes) != len(query_token_labels)
    ):
        return []
    return [
        idx
        for idx, axis_class in enumerate(query_axis_classes)
        if axis_class == "visual" and _is_informative_visual_query_token(query_token_labels[idx])
    ]


def select_page_token_indices_for_base_only(
    *,
    page_emb: torch.Tensor,
    query_emb: torch.Tensor,
    approx_page_token_topk: int,
    approx_page_token_scorer: str,
    approx_page_token_selector: str,
    approx_page_token_spatial_reserve: int,
    query_axis_classes: list[str] | None,
    query_token_labels: list[str] | None,
    page_token_classes: list[str] | None,
    approx_page_token_label_reserve: int,
    approx_page_token_soft_visual_query_weight: float,
    approx_page_token_soft_patch_visual_bonus: float,
    coarse_score_dtype: str = "fp32",
) -> list[int]:
    import torch

    coarse_scores = compute_coarse_page_token_scores(
        page_emb=page_emb,
        query_emb=query_emb,
        approx_page_token_scorer=approx_page_token_scorer,
        coarse_score_dtype=coarse_score_dtype,
    )
    topk = min(int(approx_page_token_topk), int(page_emb.shape[0]))
    if topk >= int(page_emb.shape[0]):
        return list(range(int(page_emb.shape[0])))

    if approx_page_token_selector == "global_topk":
        return _topk_indices_from_scores(scores=coarse_scores, k=topk)

    if approx_page_token_selector == "soft_label_prior":
        combined_scores = coarse_scores.clone()
        informative_visual_query_indices = _select_informative_visual_query_indices(
            query_axis_classes=query_axis_classes,
            query_token_labels=query_token_labels,
        )

        if informative_visual_query_indices and float(approx_page_token_soft_visual_query_weight) != 0.0:
            visual_query_index_tensor = torch.tensor(
                informative_visual_query_indices,
                device=query_emb.device,
                dtype=torch.long,
            )
            visual_query_emb = query_emb.index_select(0, visual_query_index_tensor)
            visual_query_scores = compute_coarse_page_token_scores(
                page_emb=page_emb,
                query_emb=visual_query_emb,
                approx_page_token_scorer="query_token_max",
                coarse_score_dtype=coarse_score_dtype,
            )
            combined_scores = combined_scores + float(approx_page_token_soft_visual_query_weight) * visual_query_scores

        if (
            page_token_classes is not None
            and len(page_token_classes) == int(page_emb.shape[0])
            and float(approx_page_token_soft_patch_visual_bonus) != 0.0
        ):
            patch_prior = torch.zeros_like(combined_scores)
            visual_bonus = float(approx_page_token_soft_patch_visual_bonus)
            for idx, axis_class in enumerate(page_token_classes):
                if axis_class == "visual":
                    patch_prior[idx] = visual_bonus
            combined_scores = combined_scores + patch_prior

        return _topk_indices_from_scores(scores=combined_scores, k=topk)

    if approx_page_token_selector == "query_label_mix":
        visual_query_indices = []
        if query_axis_classes is not None and len(query_axis_classes) == int(query_emb.shape[0]):
            visual_query_indices = [
                idx for idx, axis_class in enumerate(query_axis_classes) if axis_class == "visual"
            ]
        label_reserve = min(max(int(approx_page_token_label_reserve), 0), topk)
        if label_reserve <= 0 or not visual_query_indices:
            return _topk_indices_from_scores(scores=coarse_scores, k=topk)

        global_budget = topk - label_reserve
        selected: set[int] = set()
        if global_budget > 0:
            selected.update(_topk_indices_from_scores(scores=coarse_scores, k=global_budget))

        visual_query_index_tensor = torch.tensor(
            visual_query_indices,
            device=query_emb.device,
            dtype=torch.long,
        )
        visual_query_emb = query_emb.index_select(0, visual_query_index_tensor)
        label_scores = compute_coarse_page_token_scores(
            page_emb=page_emb,
            query_emb=visual_query_emb,
            approx_page_token_scorer=approx_page_token_scorer,
            coarse_score_dtype=coarse_score_dtype,
        )
        selected.update(_topk_indices_from_scores(scores=label_scores, k=label_reserve))

        if len(selected) < topk:
            ranked_all = _topk_indices_from_scores(scores=coarse_scores, k=int(coarse_scores.shape[0]))
            for idx in ranked_all:
                selected.add(idx)
                if len(selected) >= topk:
                    break

        ranked_selected = sorted(selected, key=lambda idx: float(coarse_scores[idx].item()), reverse=True)
        return ranked_selected[:topk]

    if approx_page_token_selector != "spatial_quadrant_mix":
        raise ValueError(f"Unsupported approx_page_token_selector: {approx_page_token_selector!r}")

    _prefix_tokens, quadrant_groups = _infer_spatial_quadrant_groups(int(page_emb.shape[0]))
    if not quadrant_groups:
        return _topk_indices_from_scores(scores=coarse_scores, k=topk)

    spatial_reserve = min(max(int(approx_page_token_spatial_reserve), 0), topk)
    global_budget = topk - spatial_reserve

    selected: set[int] = set()
    if global_budget > 0:
        selected.update(_topk_indices_from_scores(scores=coarse_scores, k=global_budget))

    if spatial_reserve > 0:
        base_per_group = spatial_reserve // len(quadrant_groups)
        remainder = spatial_reserve % len(quadrant_groups)
        for group_idx, group in enumerate(quadrant_groups):
            group_budget = base_per_group + (1 if group_idx < remainder else 0)
            if group_budget <= 0:
                continue
            group_tensor = torch.tensor(group, device=coarse_scores.device, dtype=torch.long)
            group_scores = coarse_scores.index_select(0, group_tensor)
            local_top = _topk_indices_from_scores(scores=group_scores, k=min(group_budget, len(group)))
            for local_idx in local_top:
                selected.add(group[local_idx])

    if len(selected) < topk:
        ranked_all = _topk_indices_from_scores(scores=coarse_scores, k=int(coarse_scores.shape[0]))
        for idx in ranked_all:
            selected.add(idx)
            if len(selected) >= topk:
                break

    ranked_selected = sorted(selected, key=lambda idx: float(coarse_scores[idx].item()), reverse=True)
    return ranked_selected[:topk]


def maybe_prune_page_tokens_for_base_only(
    *,
    page_emb: torch.Tensor,
    query_emb: torch.Tensor,
    approx_page_token_topk: int,
    approx_page_token_scorer: str,
    approx_page_token_selector: str,
    approx_page_token_spatial_reserve: int,
    query_axis_classes: list[str] | None,
    query_token_labels: list[str] | None,
    page_token_classes: list[str] | None,
    approx_page_token_label_reserve: int,
    approx_page_token_soft_visual_query_weight: float,
    approx_page_token_soft_patch_visual_bonus: float,
    coarse_score_dtype: str = "fp32",
) -> torch.Tensor:
    import torch

    if approx_page_token_topk <= 0:
        return page_emb
    topk = min(int(approx_page_token_topk), int(page_emb.shape[0]))
    if topk >= int(page_emb.shape[0]):
        return page_emb
    top_indices = select_page_token_indices_for_base_only(
        page_emb=page_emb,
        query_emb=query_emb,
        approx_page_token_topk=topk,
        approx_page_token_scorer=approx_page_token_scorer,
        approx_page_token_selector=approx_page_token_selector,
        approx_page_token_spatial_reserve=approx_page_token_spatial_reserve,
        query_axis_classes=query_axis_classes,
        query_token_labels=query_token_labels,
        page_token_classes=page_token_classes,
        approx_page_token_label_reserve=approx_page_token_label_reserve,
        approx_page_token_soft_visual_query_weight=approx_page_token_soft_visual_query_weight,
        approx_page_token_soft_patch_visual_bonus=approx_page_token_soft_patch_visual_bonus,
        coarse_score_dtype=coarse_score_dtype,
    )
    top_index_tensor = torch.tensor(top_indices, device=page_emb.device, dtype=torch.long)
    return page_emb.index_select(0, top_index_tensor)


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


def load_gold_row_from_qid(args: argparse.Namespace) -> dict:
    from m3docrag.utils.paths import LOCAL_DATA_DIR

    if args.gold:
        candidate_paths = [Path(args.gold)]
    else:
        candidate_paths = [
            Path(LOCAL_DATA_DIR) / args.data_name / "multimodalqa" / f"MMQA_{args.split}.jsonl",
            REPO_ROOT / "data" / args.data_name / "multimodalqa" / f"MMQA_{args.split}.jsonl",
        ]

    gold_path = next((path for path in candidate_paths if path.exists()), None)
    if gold_path is None:
        searched = "\n".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            "Could not find MMQA gold file. Checked:\n"
            f"{searched}\n"
            "Pass --gold /path/to/MMQA_<split>.jsonl explicitly."
        )

    with open(gold_path, "r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if obj["qid"] == args.qid:
                return obj
    raise KeyError(f"QID not found in gold file: {args.qid}")


def clean_token_label(token: str) -> str:
    token = token.replace("▁", " ")
    token = token.replace("<pad>", "[PAD]")
    token = token.replace("<bos>", "[BOS]")
    token = token.replace("<eos>", "[EOS]")
    token = token.strip()
    return token if token else "[WS]"


def infer_patch_grid(page_token_count: int) -> tuple[int, int]:
    for side in range(int(page_token_count**0.5), 0, -1):
        patch_count = side * side
        if patch_count <= page_token_count:
            prefix_tokens = page_token_count - patch_count
            if prefix_tokens >= 0:
                return prefix_tokens, side
    raise ValueError(f"Unable to infer patch grid from page_token_count={page_token_count}")


def classify_patch_from_splice_row(row: dict) -> str:
    patch_class = str(row.get("patch_class", "")).strip().lower()
    if patch_class in {"visual", "non_visual", "unknown"}:
        return patch_class
    if patch_class == "neutral":
        return "unknown"
    concepts = row.get("top_concepts", []) or []
    concept_names = {str(item.get("concept", "")).strip() for item in concepts}
    if "visual_region" in concept_names:
        return "visual"
    if {"ocr_text", "table_text", "table_structure"} & concept_names:
        return "non_visual"
    return "unknown"


def axis_class_counts(axis_classes: list[str]) -> dict[str, int]:
    counts = {"visual": 0, "non_visual": 0, "unknown": 0}
    for value in axis_classes:
        counts[value if value in counts else "unknown"] += 1
    return counts


def _query_label_record_id(record: dict) -> str:
    for key in ("qid", "query_id", "question_id"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return ""


def _query_label_record_from_path(path: Path, qid: str | None) -> dict | None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if qid is not None and qid in payload and isinstance(payload[qid], dict):
                return payload[qid]
            if qid is None:
                return payload
            values = payload.values()
        elif isinstance(payload, list):
            values = payload
        else:
            return None
        for row in values:
            if isinstance(row, dict):
                row_qid = _query_label_record_id(row)
                if qid is not None and row_qid == qid:
                    return row
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_qid = _query_label_record_id(row)
            if qid is not None and row_qid == qid:
                return row
    return None


def _normalize_query_axis_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(clean_token_label(str(value)).strip().lower().split())


def _extract_query_axis_strings(values: object) -> list[str]:
    extracted: list[str] = []

    def visit(value: object) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, str):
            normalized = _normalize_query_axis_text(value)
            if normalized:
                extracted.append(normalized)
            return
        if isinstance(value, dict):
            for key in (
                "token",
                "text",
                "value",
                "phrase",
                "label",
                "content",
                "surface_form",
                "matched_text",
                "norm",
                "norm_phrase",
                "token_text",
                "raw_text",
            ):
                if key in value:
                    visit(value.get(key))
                    return
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    visit(values)
    return extracted


def _extract_query_axis_indices(values: object) -> set[int]:
    extracted: set[int] = set()

    def visit(value: object) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            extracted.add(value)
            return
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.lstrip("-").isdigit():
                extracted.add(int(candidate))
            return
        if isinstance(value, dict):
            for key in ("index", "token_index", "query_token_idx", "idx"):
                if key in value:
                    visit(value.get(key))
                    return
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    visit(values)
    return extracted


def _apply_query_axis_mask(classes: list[str], values: object, axis_class: str) -> bool:
    if not isinstance(values, list) or len(values) != len(classes):
        return False

    applied = False
    for idx, value in enumerate(values):
        is_positive = False
        if isinstance(value, bool):
            is_positive = value
        elif isinstance(value, (int, float)):
            is_positive = bool(value)
        if is_positive:
            classes[idx] = axis_class
            applied = True
    return applied


def _build_query_axis_text_and_spans(
    query_token_labels: list[str],
    query_raw_tokens: list[str] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    use_raw_tokens = query_raw_tokens is not None and len(query_raw_tokens) == len(query_token_labels)

    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, token_label in enumerate(query_token_labels):
        token_source = query_raw_tokens[idx] if use_raw_tokens else token_label
        token_text = _normalize_query_axis_text(token_source)
        if not token_text:
            spans.append((-1, -1))
            continue

        needs_space = bool(parts)
        if use_raw_tokens:
            needs_space = bool(parts) and str(query_raw_tokens[idx]).startswith("▁")
        if needs_space:
            parts.append(" ")
            cursor += 1

        start = cursor
        parts.append(token_text)
        cursor += len(token_text)
        spans.append((start, cursor))

    return "".join(parts), spans


def _mark_query_axis_text_matches(
    classes: list[str],
    *,
    query_text: str,
    query_token_spans: list[tuple[int, int]],
    texts: list[str],
    axis_class: str,
    overwrite_unknown_only: bool = False,
) -> None:
    normalized_texts = []
    seen_texts: set[str] = set()
    for value in texts:
        normalized = _normalize_query_axis_text(value)
        if normalized and normalized not in seen_texts:
            seen_texts.add(normalized)
            normalized_texts.append(normalized)

    for text in normalized_texts:
        target_len = len(text)
        for start_idx, (start_char, _start_end) in enumerate(query_token_spans):
            if start_char < 0:
                continue
            for end_idx in range(start_idx, len(query_token_spans)):
                _end_start, end_char = query_token_spans[end_idx]
                if end_char < 0:
                    continue
                segment = query_text[start_char:end_char]
                if segment == text:
                    for idx in range(start_idx, end_idx + 1):
                        if overwrite_unknown_only and classes[idx] != "unknown":
                            continue
                        classes[idx] = axis_class
                    break
                if len(segment) >= target_len:
                    break


def _uses_visual_needed_schema(path: Path, record: dict | None) -> bool:
    if "visual_needed" in path.name.lower():
        return True
    if record is None:
        return False
    return any("visual_needed" in str(key).lower() for key in record)


def _uses_binary_token_label_schema(record: dict | None) -> bool:
    if record is None:
        return False
    return any(key in record for key in ("token_labels", "phrase_labels", "visual_token_count", "non_visual_token_count"))


def _binary_label_to_axis_class(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text.startswith("visual"):
        return "visual"
    if text.startswith("non_visual"):
        return "non_visual"
    if text == "neutral":
        return "unknown"
    return "unknown"


def _collect_binary_token_label_texts(record: dict, axis_class: str) -> list[str]:
    texts: list[str] = []
    for item in record.get("token_labels", []) or []:
        if not isinstance(item, dict):
            continue
        if _binary_label_to_axis_class(item.get("label")) != axis_class:
            continue
        texts.extend(_extract_query_axis_strings([item.get("norm"), item.get("token"), item.get("token_text")]))
    return texts


def _collect_binary_phrase_label_texts(record: dict, axis_class: str) -> list[str]:
    texts: list[str] = []
    for item in record.get("phrase_labels", []) or []:
        if not isinstance(item, dict):
            continue
        if _binary_label_to_axis_class(item.get("label")) != axis_class:
            continue
        texts.extend(
            _extract_query_axis_strings(
                [
                    item.get("norm_phrase"),
                    item.get("raw_examples"),
                    item.get("norm_tokens"),
                    item.get("phrase"),
                ]
            )
        )
    return texts


def _load_binary_token_label_query_axis_classes(
    *,
    record: dict | None,
    query_token_labels: list[str],
    query_raw_tokens: list[str] | None,
) -> list[str]:
    classes = ["non_visual"] * len(query_token_labels)
    if record is None:
        return classes

    query_text, query_token_spans = _build_query_axis_text_and_spans(
        query_token_labels=query_token_labels,
        query_raw_tokens=query_raw_tokens,
    )

    visual_token_texts = _collect_binary_token_label_texts(record, "visual")
    non_visual_token_texts = _collect_binary_token_label_texts(record, "non_visual")
    visual_phrase_texts = _collect_binary_phrase_label_texts(record, "visual")

    token_labels = record.get("token_labels", []) or []
    if isinstance(token_labels, list) and len(token_labels) == len(query_token_labels):
        for idx, item in enumerate(token_labels):
            if not isinstance(item, dict):
                continue
            axis_class = _binary_label_to_axis_class(item.get("label"))
            if axis_class == "visual":
                classes[idx] = "visual"
            elif axis_class == "non_visual" and classes[idx] != "visual":
                classes[idx] = "non_visual"

    if isinstance(token_labels, list) and len(token_labels) == len(query_token_labels):
        visual_token_indices = {int(x) for x in record.get("visual_token_indices", [])}
        non_visual_token_indices = {int(x) for x in record.get("non_visual_token_indices", [])}
        for idx in visual_token_indices:
            if 0 <= idx < len(classes):
                classes[idx] = "visual"
        for idx in non_visual_token_indices:
            if 0 <= idx < len(classes) and classes[idx] != "visual":
                classes[idx] = "non_visual"

    visual_token_text_set = set(visual_token_texts)
    non_visual_token_text_set = set(non_visual_token_texts)
    for idx, token_label in enumerate(query_token_labels):
        token_key = _normalize_query_axis_text(token_label)
        if token_key and token_key in visual_token_text_set:
            classes[idx] = "visual"
        elif token_key and token_key in non_visual_token_text_set and classes[idx] != "visual":
            classes[idx] = "non_visual"

    _mark_query_axis_text_matches(
        classes,
        query_text=query_text,
        query_token_spans=query_token_spans,
        texts=visual_token_texts,
        axis_class="visual",
    )
    _mark_query_axis_text_matches(
        classes,
        query_text=query_text,
        query_token_spans=query_token_spans,
        texts=visual_phrase_texts,
        axis_class="visual",
    )

    return classes


def _load_visual_needed_query_axis_classes(
    *,
    record: dict | None,
    query_token_labels: list[str],
    query_raw_tokens: list[str] | None,
) -> list[str]:
    classes = ["non_visual"] * len(query_token_labels)
    if record is None:
        return classes

    query_text, query_token_spans = _build_query_axis_text_and_spans(
        query_token_labels=query_token_labels,
        query_raw_tokens=query_raw_tokens,
    )

    for field in ("visual_needed_mask", "visual_needed_token_mask", "visual_token_mask"):
        _apply_query_axis_mask(classes, record.get(field), "visual")

    for field in ("visual_needed_token_indices", "visual_needed_indices", "visual_token_indices"):
        for idx in _extract_query_axis_indices(record.get(field)):
            if 0 <= idx < len(classes):
                classes[idx] = "visual"

    visual_needed_texts: list[str] = []
    for field in (
        "visual_needed_tokens",
        "visual_needed_token_texts",
        "visual_needed_words",
        "visual_needed_terms",
        "visual_needed",
        "visual_tokens",
    ):
        visual_needed_texts.extend(_extract_query_axis_strings(record.get(field)))

    for idx, token_label in enumerate(query_token_labels):
        token_key = _normalize_query_axis_text(token_label)
        if token_key and token_key in visual_needed_texts:
            classes[idx] = "visual"

    _mark_query_axis_text_matches(
        classes,
        query_text=query_text,
        query_token_spans=query_token_spans,
        texts=visual_needed_texts,
        axis_class="visual",
    )
    _mark_query_axis_text_matches(
        classes,
        query_text=query_text,
        query_token_spans=query_token_spans,
        texts=_extract_query_axis_strings(record.get("visual_needed_phrases")),
        axis_class="visual",
    )

    return classes


def load_splice_query_axis_classes(
    query_labels_path: str,
    qid: str | None,
    query_token_labels: list[str],
    query_raw_tokens: list[str] | None = None,
) -> list[str]:
    classes = ["unknown"] * len(query_token_labels)
    if not query_labels_path:
        return classes

    path = Path(query_labels_path)
    if not path.exists():
        raise FileNotFoundError(path)

    record = _query_label_record_from_path(path, qid)
    if _uses_visual_needed_schema(path, record):
        return _load_visual_needed_query_axis_classes(
            record=record,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
    if _uses_binary_token_label_schema(record):
        return _load_binary_token_label_query_axis_classes(
            record=record,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
    if record is None:
        return classes

    explicit_classes = record.get("query_token_classes")
    if isinstance(explicit_classes, list) and explicit_classes:
        if all(isinstance(value, dict) for value in explicit_classes):
            for value in explicit_classes:
                idx = int(value.get("index", -1))
                if not (0 <= idx < len(classes)):
                    continue
                text = str(value.get("class", "")).strip().lower()
                if text == "neutral":
                    text = "unknown"
                if text in {"visual", "non_visual", "unknown"}:
                    classes[idx] = text
            if any(x != "unknown" for x in classes):
                return classes
        elif len(explicit_classes) == len(query_token_labels):
            normalized = []
            for value in explicit_classes:
                text = str(value).strip().lower()
                if text == "neutral":
                    text = "unknown"
                if text in {"visual", "non_visual", "unknown"}:
                    normalized.append(text)
                else:
                    normalized.append("unknown")
            return normalized

    visual_token_indices = {int(x) for x in record.get("visual_token_indices", [])}
    non_visual_token_indices = {int(x) for x in record.get("non_visual_token_indices", [])}
    for idx in visual_token_indices:
        if 0 <= idx < len(classes):
            classes[idx] = "visual"
    for idx in non_visual_token_indices:
        if 0 <= idx < len(classes) and classes[idx] == "unknown":
            classes[idx] = "non_visual"

    visual_tokens = {str(x).strip().lower() for x in record.get("visual_tokens", [])}
    non_visual_tokens = {str(x).strip().lower() for x in record.get("non_visual_tokens", [])}
    if visual_tokens or non_visual_tokens:
        for idx, token in enumerate(query_token_labels):
            token_key = token.strip().lower()
            if token_key in visual_tokens:
                classes[idx] = "visual"
            elif token_key in non_visual_tokens and classes[idx] == "unknown":
                classes[idx] = "non_visual"

        query_text, query_token_spans = _build_query_axis_text_and_spans(
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
        _mark_query_axis_text_matches(
            classes,
            query_text=query_text,
            query_token_spans=query_token_spans,
            texts=list(visual_tokens),
            axis_class="visual",
        )
        _mark_query_axis_text_matches(
            classes,
            query_text=query_text,
            query_token_spans=query_token_spans,
            texts=list(non_visual_tokens),
            axis_class="non_visual",
            overwrite_unknown_only=True,
        )

    return classes


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
        "--base-score-source",
        default="exact_page_maxsim",
        choices=BASE_SCORE_SOURCE_CHOICES,
        help=(
            "Source used for the base term in the fusion. "
            "'exact_page_maxsim' recomputes page-local MaxSim on the fixed candidate pool. "
            "'baseline_pred' reuses the page score already stored in --baseline-pred. "
            "'approx_page_maxsim_topk' uses query-guided top-K page-token pruning before MaxSim "
            "in base-only mode. "
            "'two_stage_page_maxsim' uses approximate top-K pruning on all pages, then "
            "recomputes exact MaxSim only on the top-N stage-1 pages."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-topk",
        type=int,
        default=0,
        help=(
            "Top-K page tokens kept for query-guided pruning when "
            "--base-score-source=approx_page_maxsim_topk in base-only mode."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-scorer",
        default="query_mean",
        choices=APPROX_BASE_PAGE_TOKEN_SCORER_CHOICES,
        help=(
            "Coarse scorer used to select page tokens before top-K pruning in "
            "approx_page_maxsim_topk mode. 'query_mean' is the original fast mean-query scorer; "
            "'query_token_max' uses per-page-token max similarity over query tokens and is usually "
            "stronger but less efficient."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-selector",
        default="global_topk",
        choices=APPROX_BASE_PAGE_TOKEN_SELECTOR_CHOICES,
        help=(
            "Token-selection strategy used before top-K approximate MaxSim pruning. "
            "'global_topk' is the current global-only selection. "
            "'spatial_quadrant_mix' reserves part of the token budget across page quadrants. "
            "'soft_label_prior' keeps global top-K selection but adds soft bonuses from "
            "informative visual query tokens and visual-labeled page patches."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-spatial-reserve",
        type=int,
        default=64,
        help=(
            "When --approx-base-page-token-selector=spatial_quadrant_mix, reserve this many "
            "token slots across spatial quadrants before filling the rest globally."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-label-reserve",
        type=int,
        default=64,
        help=(
            "When --approx-base-page-token-selector=query_label_mix, reserve this many "
            "token slots for pruning against the query's visual-token subset."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-soft-visual-query-weight",
        type=float,
        default=0.5,
        help=(
            "When --approx-base-page-token-selector=soft_label_prior, add this weight times "
            "the informative visual-query-token alignment score to the pruning score."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-soft-patch-visual-bonus",
        type=float,
        default=0.2,
        help=(
            "When --approx-base-page-token-selector=soft_label_prior, add this bonus to "
            "page tokens whose patch label is visual before top-K selection."
        ),
    )
    parser.add_argument(
        "--base-only-page-batch-size",
        type=int,
        default=0,
        help=(
            "Optional batch size for base-only page scoring. Currently accelerates exact and "
            "global-topK approximate scoring paths when page token counts match."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-coarse-dtype",
        default="fp32",
        choices=COARSE_SCORE_DTYPE_CHOICES,
        help=(
            "Numeric precision used only for the coarse pruning-score computation in approximate "
            "base-only modes. Final exact MaxSim remains fp32."
        ),
    )
    parser.add_argument(
        "--two-stage-exact-top-pages",
        type=int,
        default=0,
        help=(
            "For --base-score-source=two_stage_page_maxsim, recompute exact MaxSim only on the "
            "top-N pages from the approximate stage-1 ranking."
        ),
    )
    parser.add_argument(
        "--visual-rerank-top-pages",
        type=int,
        default=0,
        help=(
            "Experimental staged mode. First compute a base-only ranking over all candidate pages, "
            "then recompute full visual-aware features only for the top-N stage-1 pages."
        ),
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
    parser.add_argument("--weight-visual", type=float, default=1.0)
    parser.add_argument("--weight-non-visual", type=float, default=0.0)
    parser.add_argument("--weight-balance", type=float, default=8.0)
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
    raise FileNotFoundError(
        "Could not resolve model path in offline mode. Checked:\n"
        f"  explicit: {candidate}\n"
        f"  LOCAL_MODEL_DIR: {local_candidate}\n"
        "Set LOCAL_MODEL_DIR correctly or pass an explicit local path via "
        "--retrieval_model_name_or_path /path/to/model and "
        "--retrieval_adapter_model_name_or_path /path/to/adapter."
    )


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
) -> tuple[list[str], list[str], dict[str, int], dict[str, float]]:
    if not pred_path:
        return [], [], {}, {}
    if qid is None:
        raise ValueError("--baseline-pred requires --qid so the helper can select one query.")
    if top_unique_docs <= 0 and top_pages <= 0:
        return [], [], {}, {}

    payload = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    if qid not in payload:
        raise KeyError(f"QID missing in baseline prediction file: {qid}")

    rows = payload[qid].get("page_retrieval_results", [])
    baseline_page_uids: list[str] = []
    ordered_doc_ids: list[str] = []
    doc_rank_map: dict[str, int] = {}
    page_score_map: dict[str, float] = {}
    for row_idx, row in enumerate(rows, start=1):
        doc_id = str(row[0]).strip()
        page_idx = int(row[1])
        page_uid = f"{doc_id}_page{page_idx}"
        page_score_map[page_uid] = float(row[2])
        if top_pages > 0 and row_idx <= top_pages:
            baseline_page_uids.append(page_uid)
        if not doc_id or doc_id in doc_rank_map:
            continue
        doc_rank_map[doc_id] = len(doc_rank_map) + 1
        ordered_doc_ids.append(doc_id)
        if top_unique_docs > 0 and len(ordered_doc_ids) >= top_unique_docs and (top_pages <= 0 or row_idx >= top_pages):
            break
    return (
        ordered_doc_ids[:top_unique_docs] if top_unique_docs > 0 else [],
        baseline_page_uids,
        doc_rank_map,
        page_score_map,
    )


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


def collect_candidate_sources(args: argparse.Namespace) -> tuple[list[str], set[str], dict[str, int], dict[str, float]]:
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

    baseline_doc_ids, baseline_page_uids, baseline_doc_rank_map, baseline_page_score_map = load_baseline_candidate_pool(
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

    return doc_ids, explicit_page_uids, baseline_doc_rank_map, baseline_page_score_map


def load_doc_embeddings_for_doc_ids(
    doc_ids: list[str],
    embedding_name: str,
) -> dict[str, torch.Tensor]:
    import safetensors
    import torch

    from m3docrag.utils.paths import LOCAL_EMBEDDINGS_DIR

    candidate_dirs = [
        Path(LOCAL_EMBEDDINGS_DIR) / embedding_name,
        REPO_ROOT / "embeddings" / embedding_name,
    ]
    emb_dir = next((path for path in candidate_dirs if path.exists()), None)
    if emb_dir is None:
        searched = "\n".join(f"  {path}" for path in candidate_dirs)
        raise FileNotFoundError(
            "Could not resolve embedding directory in offline mode. Checked:\n"
            f"{searched}\n"
            "Set LOCAL_EMBEDDINGS_DIR correctly or place embeddings under "
            "REPO_ROOT/embeddings/<embedding_name>."
        )

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
    base_score_override: float | None = None,
) -> PageFeature:
    import torch

    score_matrix = page_emb @ query_emb.T
    full_best_scores, full_best_indices = score_matrix.max(dim=0)
    active_best_scores = full_best_scores[query_score_mask.to(full_best_scores.device)]
    exact_base_page_score = float(active_best_scores.sum().item())
    base_page_score = exact_base_page_score if base_score_override is None else float(base_score_override)

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


def compute_base_only_page_feature(
    *,
    page_emb: torch.Tensor,
    query_emb: torch.Tensor,
    query_score_mask: torch.Tensor,
    doc_id: str,
    page_idx: int,
    base_score_override: float | None = None,
    approx_page_token_topk: int = 0,
    approx_page_token_scorer: str = "query_mean",
    approx_page_token_selector: str = "global_topk",
    approx_page_token_spatial_reserve: int = 64,
    query_axis_classes: list[str] | None = None,
    query_token_labels: list[str] | None = None,
    page_token_classes: list[str] | None = None,
    approx_page_token_label_reserve: int = 64,
    approx_page_token_soft_visual_query_weight: float = 0.5,
    approx_page_token_soft_patch_visual_bonus: float = 0.2,
    coarse_score_dtype: str = "fp32",
) -> PageFeature:
    exact_base_page_score = compute_approx_base_page_score(
        page_emb=page_emb,
        query_emb=query_emb,
        query_score_mask=query_score_mask,
        approx_page_token_topk=approx_page_token_topk,
        approx_page_token_scorer=approx_page_token_scorer,
        approx_page_token_selector=approx_page_token_selector,
        approx_page_token_spatial_reserve=approx_page_token_spatial_reserve,
        query_axis_classes=query_axis_classes,
        query_token_labels=query_token_labels,
        page_token_classes=page_token_classes,
        approx_page_token_label_reserve=approx_page_token_label_reserve,
        approx_page_token_soft_visual_query_weight=approx_page_token_soft_visual_query_weight,
        approx_page_token_soft_patch_visual_bonus=approx_page_token_soft_patch_visual_bonus,
        coarse_score_dtype=coarse_score_dtype,
    )
    base_page_score = exact_base_page_score if base_score_override is None else float(base_score_override)
    return make_base_only_page_feature(
        doc_id=doc_id,
        page_idx=page_idx,
        base_page_score=base_page_score,
    )


def compute_base_only_page_feature_scores_batched(
    *,
    batch_page_embs: torch.Tensor,
    query_emb: torch.Tensor,
    query_score_mask: torch.Tensor,
    base_score_source: str,
    approx_page_token_topk: int,
    approx_page_token_scorer: str,
    coarse_score_dtype: str,
) -> list[float]:
    import torch

    if base_score_source == "exact_page_maxsim":
        score_matrix = batch_page_embs @ query_emb.T
        full_best_scores = score_matrix.max(dim=1).values
        active_best_scores = full_best_scores[:, query_score_mask.to(full_best_scores.device)]
        return active_best_scores.sum(dim=1).to(dtype=torch.float32).tolist()

    if base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim"}:
        raise ValueError(f"Unsupported batched base score source: {base_score_source!r}")

    topk = min(int(approx_page_token_topk), int(batch_page_embs.shape[1]))
    if topk <= 0:
        raise ValueError("Batched approximate base scoring requires approx_page_token_topk > 0.")

    if coarse_score_dtype == "fp32" or batch_page_embs.device.type != "cuda":
        batch_page_embs_coarse = batch_page_embs
        query_emb_coarse = query_emb
    else:
        dtype = torch.bfloat16 if coarse_score_dtype == "bf16" else torch.float16
        batch_page_embs_coarse = batch_page_embs.to(dtype=dtype)
        query_emb_coarse = query_emb.to(dtype=dtype)

    if approx_page_token_scorer == "query_mean":
        coarse_query = query_emb_coarse.mean(dim=0)
        coarse_scores = (batch_page_embs_coarse @ coarse_query).to(dtype=torch.float32)
    elif approx_page_token_scorer == "query_token_max":
        coarse_score_matrix = batch_page_embs_coarse @ query_emb_coarse.T
        coarse_scores = coarse_score_matrix.max(dim=2).values.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported batched approx_page_token_scorer: {approx_page_token_scorer!r}")

    top_indices = torch.topk(coarse_scores, k=topk, dim=1, largest=True, sorted=False).indices
    gather_index = top_indices.unsqueeze(-1).expand(-1, -1, batch_page_embs.shape[-1])
    pruned_page_embs = batch_page_embs.gather(dim=1, index=gather_index)

    score_matrix = pruned_page_embs @ query_emb.T
    full_best_scores = score_matrix.max(dim=1).values
    active_best_scores = full_best_scores[:, query_score_mask.to(full_best_scores.device)]
    return active_best_scores.sum(dim=1).to(dtype=torch.float32).tolist()


def compute_base_only_page_features(
    *,
    page_specs: list[tuple[str, int]],
    docid2embs: dict[str, object],
    query_emb: torch.Tensor,
    query_score_mask: torch.Tensor,
    base_score_source: str,
    baseline_page_score_map: dict[str, float],
    approx_page_token_topk: int,
    approx_page_token_scorer: str,
    approx_page_token_selector: str,
    approx_page_token_spatial_reserve: int,
    query_axis_classes: list[str] | None,
    query_token_labels: list[str] | None,
    page_token_classes_by_uid: dict[str, list[str]] | None,
    approx_page_token_label_reserve: int,
    approx_page_token_soft_visual_query_weight: float,
    approx_page_token_soft_patch_visual_bonus: float,
    coarse_score_dtype: str,
    page_batch_size: int,
) -> list[PageFeature]:
    import torch

    features: list[PageFeature] = []
    can_batch = (
        page_batch_size > 1
        and approx_page_token_selector == "global_topk"
        and base_score_source in {"exact_page_maxsim", "approx_page_maxsim_topk", "two_stage_page_maxsim"}
    )

    batch_meta: list[tuple[str, int, float | None]] = []
    batch_embs: list[torch.Tensor] = []
    batch_token_count: int | None = None

    def flush_batch() -> None:
        nonlocal batch_meta, batch_embs, batch_token_count
        if not batch_meta:
            return
        if len(batch_meta) == 1 or not can_batch:
            for (doc_id, page_idx, base_score_override), page_emb in zip(batch_meta, batch_embs):
                features.append(
                    compute_base_only_page_feature(
                        page_emb=page_emb,
                        query_emb=query_emb,
                        query_score_mask=query_score_mask,
                        doc_id=doc_id,
                        page_idx=page_idx,
                        base_score_override=base_score_override,
                        approx_page_token_topk=approx_page_token_topk,
                        approx_page_token_scorer=approx_page_token_scorer,
                        approx_page_token_selector=approx_page_token_selector,
                        approx_page_token_spatial_reserve=approx_page_token_spatial_reserve,
                        query_axis_classes=query_axis_classes,
                        query_token_labels=query_token_labels,
                        page_token_classes=(
                            None
                            if page_token_classes_by_uid is None
                            else page_token_classes_by_uid.get(f"{doc_id}_page{page_idx}")
                        ),
                        approx_page_token_label_reserve=approx_page_token_label_reserve,
                        approx_page_token_soft_visual_query_weight=approx_page_token_soft_visual_query_weight,
                        approx_page_token_soft_patch_visual_bonus=approx_page_token_soft_patch_visual_bonus,
                        coarse_score_dtype=coarse_score_dtype,
                    )
                )
        else:
            batch_tensor = torch.stack(batch_embs, dim=0)
            batch_scores = compute_base_only_page_feature_scores_batched(
                batch_page_embs=batch_tensor,
                query_emb=query_emb,
                query_score_mask=query_score_mask,
                base_score_source=base_score_source,
                approx_page_token_topk=approx_page_token_topk,
                approx_page_token_scorer=approx_page_token_scorer,
                coarse_score_dtype=coarse_score_dtype,
            )
            for (doc_id, page_idx, base_score_override), batch_score in zip(batch_meta, batch_scores):
                base_page_score = float(batch_score) if base_score_override is None else float(base_score_override)
                features.append(
                    make_base_only_page_feature(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        base_page_score=base_page_score,
                    )
                )

        batch_meta = []
        batch_embs = []
        batch_token_count = None

    for doc_id, page_idx in page_specs:
        page_uid = f"{doc_id}_page{page_idx}"
        page_emb = docid2embs[doc_id][page_idx].view(-1, docid2embs[doc_id][page_idx].shape[-1]).to(
            device=query_emb.device,
            dtype=query_emb.dtype,
        )
        base_score_override = (
            baseline_page_score_map.get(page_uid)
            if base_score_source == "baseline_pred"
            else None
        )

        if not can_batch:
            batch_meta = [(doc_id, page_idx, base_score_override)]
            batch_embs = [page_emb]
            flush_batch()
            continue

        page_token_count = int(page_emb.shape[0])
        if (
            batch_meta
            and (
                batch_token_count != page_token_count
                or len(batch_meta) >= page_batch_size
            )
        ):
            flush_batch()

        batch_meta.append((doc_id, page_idx, base_score_override))
        batch_embs.append(page_emb)
        batch_token_count = page_token_count

    flush_batch()
    return features


def apply_two_stage_exact_rerank_to_page_features(
    *,
    page_features: list[PageFeature],
    docid2embs: dict[str, object],
    query_emb: torch.Tensor,
    query_score_mask: torch.Tensor,
    top_pages: int,
) -> list[PageFeature]:
    if top_pages <= 0 or not page_features:
        return page_features

    _stage1_ranked_docs, stage1_ranked_pages = build_rankings(
        page_features=page_features,
        weights=WeightConfig(base=1.0, visual=0.0, non_visual=0.0, balance=0.0),
        baseline_doc_rank_map={},
    )
    top_page_uids = {item["page_uid"] for item in stage1_ranked_pages[:top_pages]}

    updated_features: list[PageFeature] = []
    for feature in page_features:
        if feature.page_uid not in top_page_uids:
            updated_features.append(feature)
            continue

        page_tensor = docid2embs[feature.doc_id][feature.page_idx]
        page_emb = page_tensor.view(-1, page_tensor.shape[-1]).to(
            device=query_emb.device,
            dtype=query_emb.dtype,
        )
        exact_base_page_score = compute_exact_base_page_score(
            page_emb=page_emb,
            query_emb=query_emb,
            query_score_mask=query_score_mask,
        )
        updated_features.append(
            make_base_only_page_feature(
                doc_id=feature.doc_id,
                page_idx=feature.page_idx,
                base_page_score=exact_base_page_score,
            )
        )

    return updated_features


def apply_visual_rerank_to_top_pages(
    *,
    page_features: list[PageFeature],
    docid2embs: dict[str, object],
    query_emb: torch.Tensor,
    query_axis_classes: list[str],
    query_score_mask: torch.Tensor,
    page_token_classes_by_uid: dict[str, list[str]],
    top_pages: int,
) -> list[PageFeature]:
    if top_pages <= 0 or not page_features:
        return page_features

    _stage1_ranked_docs, stage1_ranked_pages = build_rankings(
        page_features=page_features,
        weights=WeightConfig(base=1.0, visual=0.0, non_visual=0.0, balance=0.0),
        baseline_doc_rank_map={},
    )
    selected_page_uids = {item["page_uid"] for item in stage1_ranked_pages[:top_pages]}

    updated_features: list[PageFeature] = []
    for feature in page_features:
        if feature.page_uid not in selected_page_uids:
            updated_features.append(feature)
            continue

        page_tensor = docid2embs[feature.doc_id][feature.page_idx]
        page_emb = page_tensor.view(-1, page_tensor.shape[-1]).to(
            device=query_emb.device,
            dtype=query_emb.dtype,
        )
        page_token_classes = page_token_classes_by_uid.get(feature.page_uid)
        if page_token_classes is None:
            raise KeyError(f"Missing page token classes for selected page_uid={feature.page_uid}")
        updated_features.append(
            compute_page_feature(
                page_emb=page_emb,
                query_emb=query_emb,
                query_axis_classes=query_axis_classes,
                query_score_mask=query_score_mask,
                page_token_classes=page_token_classes,
                doc_id=feature.doc_id,
                page_idx=feature.page_idx,
                base_score_override=None,
            )
        )

    return updated_features


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

    if args.base_score_source == "approx_page_maxsim_topk" and args.approx_base_page_token_topk <= 0:
        raise ValueError(
            "--base-score-source=approx_page_maxsim_topk requires "
            "--approx-base-page-token-topk > 0."
        )
    if args.base_score_source == "two_stage_page_maxsim" and args.approx_base_page_token_topk <= 0:
        raise ValueError(
            "--base-score-source=two_stage_page_maxsim requires "
            "--approx-base-page-token-topk > 0."
        )
    if args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim"} and args.approx_base_page_token_topk > 0:
        raise ValueError(
            "--approx-base-page-token-topk is only valid with "
            "--base-score-source=approx_page_maxsim_topk or two_stage_page_maxsim."
        )
    if (
        args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim"}
        and args.approx_base_page_token_scorer != "query_mean"
    ):
        raise ValueError(
            "--approx-base-page-token-scorer is only valid with "
            "--base-score-source=approx_page_maxsim_topk or two_stage_page_maxsim."
        )
    if (
        args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim"}
        and args.approx_base_page_token_selector != "global_topk"
    ):
        raise ValueError(
            "--approx-base-page-token-selector is only valid with "
            "--base-score-source=approx_page_maxsim_topk or two_stage_page_maxsim."
        )
    if (
        args.approx_base_page_token_selector != "spatial_quadrant_mix"
        and args.approx_base_page_token_spatial_reserve != 64
    ):
        raise ValueError(
            "--approx-base-page-token-spatial-reserve is only valid with "
            "--approx-base-page-token-selector=spatial_quadrant_mix."
        )
    if (
        args.approx_base_page_token_selector != "query_label_mix"
        and args.approx_base_page_token_label_reserve != 64
    ):
        raise ValueError(
            "--approx-base-page-token-label-reserve is only valid with "
            "--approx-base-page-token-selector=query_label_mix."
        )
    if (
        args.approx_base_page_token_selector != "soft_label_prior"
        and args.approx_base_page_token_soft_visual_query_weight != 0.5
    ):
        raise ValueError(
            "--approx-base-page-token-soft-visual-query-weight is only valid with "
            "--approx-base-page-token-selector=soft_label_prior."
        )
    if (
        args.approx_base_page_token_selector != "soft_label_prior"
        and args.approx_base_page_token_soft_patch_visual_bonus != 0.2
    ):
        raise ValueError(
            "--approx-base-page-token-soft-patch-visual-bonus is only valid with "
            "--approx-base-page-token-selector=soft_label_prior."
        )
    if args.base_only_page_batch_size < 0:
        raise ValueError("--base-only-page-batch-size must be >= 0.")
    if (
        args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim"}
        and args.approx_base_page_token_coarse_dtype != "fp32"
    ):
        raise ValueError(
            "--approx-base-page-token-coarse-dtype is only valid with "
            "--base-score-source=approx_page_maxsim_topk or two_stage_page_maxsim."
        )
    if args.base_score_source == "two_stage_page_maxsim" and args.two_stage_exact_top_pages <= 0:
        raise ValueError(
            "--base-score-source=two_stage_page_maxsim requires "
            "--two-stage-exact-top-pages > 0."
        )
    if args.base_score_source != "two_stage_page_maxsim" and args.two_stage_exact_top_pages > 0:
        raise ValueError(
            "--two-stage-exact-top-pages is only valid with "
            "--base-score-source=two_stage_page_maxsim."
        )
    if args.visual_rerank_top_pages < 0:
        raise ValueError("--visual-rerank-top-pages must be >= 0.")

    import torch

    from m3docrag.retrieval import ColPaliRetrievalModel

    query_text, gold_doc_ids = load_query_text_and_gold_doc_ids(args)
    gold_page_uids = [str(value).strip() for value in args.gold_page_uid if str(value).strip()]
    candidate_doc_ids, explicit_page_uids, baseline_doc_rank_map, baseline_page_score_map = collect_candidate_sources(args)
    fixed_weights = WeightConfig(
        base=args.weight_base,
        visual=args.weight_visual,
        non_visual=args.weight_non_visual,
        balance=args.weight_balance,
    )
    fixed_base_only = (not args.grid_search) and is_base_only_weights(fixed_weights)
    staged_visual_rerank = (not args.grid_search) and args.visual_rerank_top_pages > 0
    if (
        args.base_score_source in {"approx_page_maxsim_topk", "two_stage_page_maxsim"}
        and not (fixed_base_only or staged_visual_rerank)
    ):
        raise ValueError(
            f"--base-score-source={args.base_score_source} is currently only supported in base-only mode."
        )
    if args.base_only_page_batch_size > 0 and not (fixed_base_only or staged_visual_rerank):
        raise ValueError("--base-only-page-batch-size is currently only supported in base-only mode.")
    if args.visual_rerank_top_pages > 0 and fixed_base_only:
        raise ValueError("--visual-rerank-top-pages requires non-base-only fusion weights.")
    if args.visual_rerank_top_pages > 0 and args.grid_search:
        raise ValueError("--visual-rerank-top-pages is currently only supported with fixed weights.")

    if fixed_base_only and args.base_score_source == "baseline_pred":
        page_features = []
        for page_uid in sorted(explicit_page_uids):
            doc_id, page_suffix = page_uid.rsplit("_page", 1)
            page_idx = int(page_suffix)
            score = baseline_page_score_map.get(page_uid)
            if score is None:
                continue
            page_features.append(
                make_base_only_page_feature(
                    doc_id=doc_id,
                    page_idx=page_idx,
                    base_page_score=score,
                )
            )
        if not page_features:
            raise ValueError("No candidate pages were available for baseline_pred base-only reranking.")
        query_axis_classes = []
        query_token_labels = []
        query_raw_tokens = []
        best_grid_record = None
        grid_leaderboard = []
        weights = fixed_weights
        reranked_docs, reranked_pages = build_rankings(
            page_features=page_features,
            weights=weights,
            baseline_doc_rank_map=baseline_doc_rank_map,
        )
        gold_doc_summary = summarize_gold_doc_ranks(reranked_docs, gold_doc_ids)
        gold_page_summary = summarize_gold_page_ranks(reranked_pages, gold_page_uids)
        summary = {
            "qid": args.qid,
            "query": query_text,
            "query_token_filter": args.query_token_filter,
            "ignore_pad_scores_in_final_ranking": args.ignore_pad_scores_in_final_ranking,
            "base_score_source": args.base_score_source,
            "approx_base_page_token_topk": args.approx_base_page_token_topk,
            "approx_base_page_token_scorer": args.approx_base_page_token_scorer,
            "approx_base_page_token_selector": args.approx_base_page_token_selector,
            "approx_base_page_token_spatial_reserve": args.approx_base_page_token_spatial_reserve,
            "approx_base_page_token_label_reserve": args.approx_base_page_token_label_reserve,
            "base_only_page_batch_size": args.base_only_page_batch_size,
            "approx_base_page_token_coarse_dtype": args.approx_base_page_token_coarse_dtype,
            "two_stage_exact_top_pages": args.two_stage_exact_top_pages,
            "candidate_doc_count": len(candidate_doc_ids),
            "candidate_page_count": len(page_features),
            "query_axis_class_counts": axis_class_counts(query_axis_classes),
            "gold_doc_ids": gold_doc_ids,
            **gold_doc_summary,
            "gold_page_uids": gold_page_uids,
            **gold_page_summary,
            "weights": asdict(weights),
            "grid_search_enabled": args.grid_search,
            "grid_search_best": best_grid_record,
            "grid_search_leaderboard": grid_leaderboard,
            "top_reranked_docs": reranked_docs[:10],
            "top_reranked_pages": reranked_pages[:10],
            "query_token_labels": query_token_labels,
            "query_axis_classes": query_axis_classes,
            "query_label_path": args.splice_query_token_labels,
            "patch_label_path": args.splice_patch_labels_jsonl,
            "explicit_page_uids": sorted(explicit_page_uids),
            "baseline_doc_rank_map": baseline_doc_rank_map,
        }
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"saved_summary: {output_path}")
        if args.output_prediction_json:
            prediction_payload = {
                args.qid: {
                    "pred_answer": "",
                    "page_retrieval_results": [
                        [item["doc_id"], item["page_idx"], item["fused_page_score"]]
                        for item in reranked_pages
                    ],
                    "qid": args.qid,
                    "question": query_text,
                    "top_retrieved_docs": [item["doc_id"] for item in reranked_docs[:10]],
                }
            }
            prediction_path = Path(args.output_prediction_json)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(json.dumps(prediction_payload, indent=2) + "\n", encoding="utf-8")
            print(f"saved_prediction: {prediction_path}")
        return

    docid2embs = load_doc_embeddings_for_doc_ids(candidate_doc_ids, args.embedding_name)
    page_specs, page_meta = build_page_id_metadata(
        docid2embs=docid2embs,
        explicit_page_uids=explicit_page_uids,
        nonspatial_token_position=args.nonspatial_token_position,
    )
    needs_patch_axis_classes = (
        not fixed_base_only
        or staged_visual_rerank
        or args.approx_base_page_token_selector == "soft_label_prior"
    )
    patch_axis_classes_by_uid = (
        load_patch_axis_classes_for_pages(
            labels_jsonl=args.splice_patch_labels_jsonl,
            page_meta=page_meta,
        )
        if needs_patch_axis_classes
        else None
    )
    page_token_classes_by_uid = (
        None
        if patch_axis_classes_by_uid is None
        else {
            page_uid: build_page_token_classes(
                page_meta=page_meta[page_uid],
                patch_axis_classes=patch_axis_classes,
            )
            for page_uid, patch_axis_classes in patch_axis_classes_by_uid.items()
        }
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
    if staged_visual_rerank:
        query_axis_classes = load_splice_query_axis_classes(
            query_labels_path=args.splice_query_token_labels,
            qid=args.qid,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
    elif fixed_base_only and args.approx_base_page_token_selector in {"query_label_mix", "soft_label_prior"}:
        query_axis_classes = load_splice_query_axis_classes(
            query_labels_path=args.splice_query_token_labels,
            qid=args.qid,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
    elif fixed_base_only:
        query_axis_classes = []
    else:
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
        if fixed_base_only:
            page_features = compute_base_only_page_features(
                page_specs=page_specs,
                docid2embs=docid2embs,
                query_emb=query_emb,
                query_score_mask=query_score_mask,
                base_score_source=args.base_score_source,
                baseline_page_score_map=baseline_page_score_map,
                approx_page_token_topk=(
                    args.approx_base_page_token_topk
                    if args.base_score_source in {"approx_page_maxsim_topk", "two_stage_page_maxsim"}
                    else 0
                ),
                approx_page_token_scorer=args.approx_base_page_token_scorer,
                approx_page_token_selector=args.approx_base_page_token_selector,
                approx_page_token_spatial_reserve=args.approx_base_page_token_spatial_reserve,
                query_axis_classes=query_axis_classes,
                query_token_labels=query_token_labels,
                page_token_classes_by_uid=page_token_classes_by_uid,
                approx_page_token_label_reserve=args.approx_base_page_token_label_reserve,
                approx_page_token_soft_visual_query_weight=args.approx_base_page_token_soft_visual_query_weight,
                approx_page_token_soft_patch_visual_bonus=args.approx_base_page_token_soft_patch_visual_bonus,
                coarse_score_dtype=args.approx_base_page_token_coarse_dtype,
                page_batch_size=args.base_only_page_batch_size,
            )
        elif staged_visual_rerank:
            page_features = compute_base_only_page_features(
                page_specs=page_specs,
                docid2embs=docid2embs,
                query_emb=query_emb,
                query_score_mask=query_score_mask,
                base_score_source=args.base_score_source,
                baseline_page_score_map=baseline_page_score_map,
                approx_page_token_topk=(
                    args.approx_base_page_token_topk
                    if args.base_score_source in {"approx_page_maxsim_topk", "two_stage_page_maxsim"}
                    else 0
                ),
                approx_page_token_scorer=args.approx_base_page_token_scorer,
                approx_page_token_selector=args.approx_base_page_token_selector,
                approx_page_token_spatial_reserve=args.approx_base_page_token_spatial_reserve,
                query_axis_classes=query_axis_classes,
                query_token_labels=query_token_labels,
                page_token_classes_by_uid=page_token_classes_by_uid,
                approx_page_token_label_reserve=args.approx_base_page_token_label_reserve,
                approx_page_token_soft_visual_query_weight=args.approx_base_page_token_soft_visual_query_weight,
                approx_page_token_soft_patch_visual_bonus=args.approx_base_page_token_soft_patch_visual_bonus,
                coarse_score_dtype=args.approx_base_page_token_coarse_dtype,
                page_batch_size=args.base_only_page_batch_size,
            )
        else:
            for doc_id, page_idx in page_specs:
                page_uid = f"{doc_id}_page{page_idx}"
                page_emb = docid2embs[doc_id][page_idx].view(-1, docid2embs[doc_id][page_idx].shape[-1]).to(
                    device=device,
                    dtype=torch.float32,
                )
                assert page_token_classes_by_uid is not None
                page_token_classes = page_token_classes_by_uid[page_uid]
                page_features.append(
                    compute_page_feature(
                        page_emb=page_emb,
                        query_emb=query_emb,
                        query_axis_classes=query_axis_classes,
                        query_score_mask=query_score_mask,
                        page_token_classes=page_token_classes,
                        doc_id=doc_id,
                        page_idx=page_idx,
                        base_score_override=(
                            baseline_page_score_map.get(page_uid)
                            if args.base_score_source == "baseline_pred"
                            else None
                        ),
                    )
                )

    if not page_features:
        raise ValueError("No candidate pages were scored.")

    if args.base_score_source == "two_stage_page_maxsim":
        page_features = apply_two_stage_exact_rerank_to_page_features(
            page_features=page_features,
            docid2embs=docid2embs,
            query_emb=query_emb,
            query_score_mask=query_score_mask,
            top_pages=args.two_stage_exact_top_pages,
        )

    if staged_visual_rerank:
        assert page_token_classes_by_uid is not None
        page_features = apply_visual_rerank_to_top_pages(
            page_features=page_features,
            docid2embs=docid2embs,
            query_emb=query_emb,
            query_axis_classes=query_axis_classes,
            query_score_mask=query_score_mask,
            page_token_classes_by_uid=page_token_classes_by_uid,
            top_pages=args.visual_rerank_top_pages,
        )

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
        weights = fixed_weights
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
        "base_score_source": args.base_score_source,
        "approx_base_page_token_topk": args.approx_base_page_token_topk,
        "approx_base_page_token_scorer": args.approx_base_page_token_scorer,
        "approx_base_page_token_selector": args.approx_base_page_token_selector,
        "approx_base_page_token_spatial_reserve": args.approx_base_page_token_spatial_reserve,
        "approx_base_page_token_label_reserve": args.approx_base_page_token_label_reserve,
        "approx_base_page_token_soft_visual_query_weight": args.approx_base_page_token_soft_visual_query_weight,
        "approx_base_page_token_soft_patch_visual_bonus": args.approx_base_page_token_soft_patch_visual_bonus,
        "base_only_page_batch_size": args.base_only_page_batch_size,
        "approx_base_page_token_coarse_dtype": args.approx_base_page_token_coarse_dtype,
        "two_stage_exact_top_pages": args.two_stage_exact_top_pages,
        "visual_rerank_top_pages": args.visual_rerank_top_pages,
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
                "base_score_source": args.base_score_source,
                "approx_base_page_token_topk": args.approx_base_page_token_topk,
                "approx_base_page_token_scorer": args.approx_base_page_token_scorer,
                "approx_base_page_token_selector": args.approx_base_page_token_selector,
                "approx_base_page_token_spatial_reserve": args.approx_base_page_token_spatial_reserve,
                "approx_base_page_token_label_reserve": args.approx_base_page_token_label_reserve,
                "base_only_page_batch_size": args.base_only_page_batch_size,
                "approx_base_page_token_coarse_dtype": args.approx_base_page_token_coarse_dtype,
                "two_stage_exact_top_pages": args.two_stage_exact_top_pages,
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
    print(f"base_score_source: {args.base_score_source}")
    print(f"approx_base_page_token_topk: {args.approx_base_page_token_topk}")
    print(f"approx_base_page_token_scorer: {args.approx_base_page_token_scorer}")
    print(f"approx_base_page_token_selector: {args.approx_base_page_token_selector}")
    print(f"approx_base_page_token_spatial_reserve: {args.approx_base_page_token_spatial_reserve}")
    print(f"approx_base_page_token_label_reserve: {args.approx_base_page_token_label_reserve}")
    print(f"base_only_page_batch_size: {args.base_only_page_batch_size}")
    print(f"approx_base_page_token_coarse_dtype: {args.approx_base_page_token_coarse_dtype}")
    print(f"two_stage_exact_top_pages: {args.two_stage_exact_top_pages}")
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
