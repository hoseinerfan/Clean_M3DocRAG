#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for root in (REPO_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.run_boundary_vlm_verifier import (  # noqa: E402
    load_page_image_paths,
    load_vqa_model,
    open_image,
)


FINAL_DECISIONS = {"verified", "rejected", "unclear"}
PROXY_TIERS = {"visual_proxy", "mixed_direct_proxy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify MMQA image-title proxy page labels by showing a frozen VLM the original "
            "MMQA evidence image and the candidate rendered PDF page. The prompt does not "
            "include the question or answer."
        )
    )
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--mmqa-images-jsonl", required=True)
    parser.add_argument("--mmqa-image-root", required=True)
    parser.add_argument(
        "--doc-pages-jsonl",
        default="",
        help="Optional page JSONL containing image_path/source_image_path fields.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="",
        help="Fallback directory containing <doc_id>.pdf files.",
    )
    parser.add_argument("--vlm-model-name-or-path", default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--vlm-model-type", default="")
    parser.add_argument("--vlm-bits", type=int, default=16)
    parser.add_argument("--no-accelerate", action="store_true")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve cases and assets without loading the VLM or rendering PDF pages.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id", "")).strip(): row
        for row in load_jsonl(path)
        if str(row.get("id", "")).strip()
    }


def resolve_mmqa_image_path(image_row: dict[str, Any], image_root: Path) -> Path | None:
    raw = str(image_row.get("path", "") or "").strip()
    if not raw:
        return None
    raw_path = Path(raw)
    candidates = [raw_path] if raw_path.is_absolute() else []
    candidates.extend(
        [
            image_root / raw_path,
            image_root / raw_path.name,
            image_root / "final_dataset_images" / raw_path,
            image_root / "images" / raw_path,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def parse_page_uid(uid: str) -> tuple[str, int] | None:
    if "_page" not in uid:
        return None
    doc_id, raw_page = uid.rsplit("_page", 1)
    try:
        return doc_id, int(raw_page)
    except ValueError:
        return None


def build_cases(
    labels: list[dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    image_root: Path,
    qids: set[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in labels:
        qid = str(row.get("qid", "")).strip()
        if qids and qid not in qids:
            continue
        for page in row.get("pseudo_gold_pages", []) or []:
            if not isinstance(page, dict):
                continue
            page_uid = str(page.get("page_uid", "")).strip()
            if not page_uid:
                continue
            for unit in page.get("evidence_units", []) or []:
                if not isinstance(unit, dict):
                    continue
                if str(unit.get("unit_type", "")) != "image_instance":
                    continue
                if str(unit.get("supervision_tier", "")) not in PROXY_TIERS:
                    continue
                unit_id = str(unit.get("unit_id", "")).strip()
                image_id = str(unit.get("doc_id", "")).strip()
                case_id = f"{qid}::{unit_id}::{page_uid}"
                if not unit_id or not image_id or case_id in seen:
                    continue
                seen.add(case_id)
                image_row = images_by_id.get(image_id, {})
                reference_path = resolve_mmqa_image_path(image_row, image_root)
                cases.append(
                    {
                        "case_id": case_id,
                        "qid": qid,
                        "question_type": str(row.get("question_type", "")),
                        "unit_id": unit_id,
                        "image_id": image_id,
                        "image_title": str(image_row.get("title", "") or ""),
                        "image_relative_path": str(image_row.get("path", "") or ""),
                        "reference_image_path": str(reference_path) if reference_path else "",
                        "page_uid": page_uid,
                        "page_doc_id": str(page.get("doc_id", "")),
                        "page_idx": page.get("page_idx"),
                        "proxy_score": page.get("score"),
                        "proxy_confidence": str(page.get("confidence", "")),
                        "decision": "pending",
                        "reason": "",
                    }
                )
    return cases


def image_presence_prompt() -> str:
    return (
        "The first image is a reference evidence image from a dataset.\n"
        "The second image is a complete rendered PDF page.\n"
        "Determine whether the PDF page visibly contains the same underlying image as the "
        "reference. Allow resizing, cropping, compression, or minor color changes. Do not "
        "decide from matching titles, captions, or surrounding text alone.\n"
        "Reply with exactly one token: yes, no, or unclear."
    )


def parse_verification_response(response: str) -> tuple[str, str]:
    normalized = " ".join(str(response or "").strip().lower().split())
    if not normalized:
        return "unclear", normalized
    first = normalized.strip(" .,:;()[]{}").split()[0]
    if first == "yes":
        return "verified", normalized
    if first == "no":
        return "rejected", normalized
    if first in {"unclear", "unsure", "maybe", "possibly"}:
        return "unclear", normalized
    if "yes" in normalized and "no" not in normalized:
        return "verified", normalized
    if "no" in normalized and "yes" not in normalized:
        return "rejected", normalized
    return "unclear", normalized


class PageImageLoader:
    def __init__(self, *, page_image_paths: dict[str, Path], pdf_dir: Path | None, dpi: int):
        self.page_image_paths = page_image_paths
        self.pdf_dir = pdf_dir
        self.dpi = int(dpi)

    def preflight(self, page_uid: str) -> tuple[bool, str]:
        image_path = self.page_image_paths.get(page_uid)
        if image_path and image_path.is_file():
            return True, "page_image_path"
        parsed = parse_page_uid(page_uid)
        if parsed is None or self.pdf_dir is None:
            return False, "missing_page_source"
        pdf_path = self.pdf_dir / f"{parsed[0]}.pdf"
        return (pdf_path.is_file(), "pdf" if pdf_path.is_file() else "missing_pdf")

    def load(self, page_uid: str) -> tuple[Image.Image, str]:
        image_path = self.page_image_paths.get(page_uid)
        if image_path and image_path.is_file():
            return open_image(image_path), str(image_path)
        parsed = parse_page_uid(page_uid)
        if parsed is None or self.pdf_dir is None:
            raise FileNotFoundError(f"No page image source for {page_uid}")
        doc_id, page_idx = parsed
        pdf_path = self.pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        from pdf2image import convert_from_path

        images = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            first_page=page_idx + 1,
            last_page=page_idx + 1,
        )
        if len(images) != 1:
            raise RuntimeError(f"Expected one rendered page for {page_uid}, got {len(images)}")
        return images[0].convert("RGB"), f"{pdf_path}#page={page_idx + 1}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    decision_counts = Counter(str(row.get("decision", "")) for row in rows)
    qtype_counts = Counter(str(row.get("question_type", "")) for row in rows)
    verified_qtypes = Counter(
        str(row.get("question_type", "")) for row in rows if row.get("decision") == "verified"
    )
    return {
        "labels_jsonl": str(args.labels_jsonl),
        "mmqa_images_jsonl": str(args.mmqa_images_jsonl),
        "mmqa_image_root": str(args.mmqa_image_root),
        "pdf_dir": str(args.pdf_dir),
        "case_count": len(rows),
        "qid_count": len({str(row.get("qid", "")) for row in rows}),
        "unit_count": len({str(row.get("unit_id", "")) for row in rows}),
        "decision_counts": dict(sorted(decision_counts.items())),
        "question_type_counts": dict(sorted(qtype_counts.items())),
        "verified_by_question_type": dict(sorted(verified_qtypes.items())),
        "dry_run": bool(args.dry_run),
        "model_name_or_path": str(args.vlm_model_name_or_path),
        "vlm_bits": int(args.vlm_bits),
    }


def main() -> None:
    args = parse_args()
    if not args.doc_pages_jsonl and not args.pdf_dir:
        raise ValueError("Provide --doc-pages-jsonl and/or --pdf-dir for PDF page images.")
    labels = load_jsonl(Path(args.labels_jsonl))
    images_by_id = load_by_id(Path(args.mmqa_images_jsonl))
    image_root = Path(args.mmqa_image_root)
    qids = {str(value).strip() for value in args.qid if str(value).strip()}
    cases = build_cases(labels, images_by_id, image_root, qids)
    if int(args.limit) > 0:
        cases = cases[: int(args.limit)]

    output_path = Path(args.output_jsonl)
    existing: dict[str, dict[str, Any]] = {}
    if bool(args.resume) and output_path.is_file():
        existing = {
            str(row.get("case_id", "")): row
            for row in load_jsonl(output_path)
            if str(row.get("case_id", ""))
        }
    rows = [existing.get(str(case["case_id"]), case) for case in cases]

    page_paths = load_page_image_paths(Path(args.doc_pages_jsonl)) if args.doc_pages_jsonl else {}
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
    page_loader = PageImageLoader(page_image_paths=page_paths, pdf_dir=pdf_dir, dpi=int(args.dpi))

    pending_indices: list[int] = []
    for idx, case in enumerate(rows):
        if str(case.get("decision", "")) in FINAL_DECISIONS:
            continue
        reference_path = Path(str(case.get("reference_image_path", "")))
        if not reference_path.is_file():
            case["decision"] = "missing_reference_image"
            case["reason"] = "MMQA image path could not be resolved"
            continue
        page_ready, page_source = page_loader.preflight(str(case["page_uid"]))
        case["page_source"] = page_source
        if not page_ready:
            case["decision"] = "missing_page_image"
            case["reason"] = page_source
            continue
        if bool(args.dry_run):
            case["decision"] = "ready"
            case["reason"] = "dry_run_preflight_passed"
            continue
        pending_indices.append(idx)

    vqa_model = None if bool(args.dry_run) or not pending_indices else load_vqa_model(args)
    prompt = image_presence_prompt()
    processed = 0
    for idx in pending_indices:
        case = rows[idx]
        try:
            reference_image = open_image(Path(str(case["reference_image_path"])))
            page_image, page_source_path = page_loader.load(str(case["page_uid"]))
            response = vqa_model.generate(images=[reference_image, page_image], question=prompt)
            decision, normalized = parse_verification_response(response)
            case.update(
                {
                    "decision": decision,
                    "reason": "vlm_image_presence_decision",
                    "vlm_response": str(response).strip(),
                    "vlm_response_normalized": normalized,
                    "page_source_path": page_source_path,
                    "prompt_version": "image_presence_v1_no_qa_text",
                }
            )
        except Exception as exc:
            case["decision"] = "error"
            case["reason"] = f"{type(exc).__name__}: {exc}"
        processed += 1
        if processed % max(1, int(args.save_every)) == 0:
            write_jsonl(output_path, rows)
            print(f"processed_cases={processed}/{len(pending_indices)}")

    write_jsonl(output_path, rows)
    summary = summarize(rows, args)
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_output={output_path}")
    print(f"saved_summary={summary_path}")
    print(f"case_count={summary['case_count']}")
    print(f"qid_count={summary['qid_count']}")
    print(f"decision_counts={summary['decision_counts']}")


if __name__ == "__main__":
    main()
