#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a query-subset M3DocVQA dataset root that reuses the full PDF corpus "
            "and full dev_doc_ids.json, while narrowing MMQA_<split>.jsonl to selected qids."
        )
    )
    parser.add_argument("--source-root", required=True, help="Existing m3-docvqa dataset root")
    parser.add_argument("--output-root", required=True, help="New subset dataset root to create")
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--qid", dest="qids", action="append", required=True, help="QID to keep; pass multiple times")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    split = args.split
    wanted_qids = list(dict.fromkeys(args.qids))

    mmqa_src = source_root / "multimodalqa" / f"MMQA_{split}.jsonl"
    if not mmqa_src.exists():
        raise FileNotFoundError(mmqa_src)

    pdf_src = source_root / "splits" / f"pdfs_{split}"
    if not pdf_src.exists():
        raise FileNotFoundError(pdf_src)

    split_doc_ids_src = source_root / f"{split}_doc_ids.json"
    if not split_doc_ids_src.exists():
        raise FileNotFoundError(split_doc_ids_src)

    output_mmqa_dir = output_root / "multimodalqa"
    output_pdf_dir = output_root / "splits" / f"pdfs_{split}"

    output_mmqa_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_dir.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    found_qids = set()
    with open(mmqa_src, "r", encoding="utf-8") as reader:
        for line in reader:
            obj = json.loads(line)
            qid = obj.get("qid")
            if qid in wanted_qids:
                kept.append(obj)
                found_qids.add(qid)

    missing = [qid for qid in wanted_qids if qid not in found_qids]
    if missing:
        raise ValueError(f"QIDs not found in {mmqa_src}: {missing}")

    subset_mmqa_path = output_mmqa_dir / f"MMQA_{split}.jsonl"
    with open(subset_mmqa_path, "w", encoding="utf-8") as writer:
        for obj in kept:
            writer.write(json.dumps(obj) + "\n")

    split_doc_ids_dst = output_root / f"{split}_doc_ids.json"
    if split_doc_ids_dst.exists() or split_doc_ids_dst.is_symlink():
        split_doc_ids_dst.unlink()
    shutil.copy2(split_doc_ids_src, split_doc_ids_dst)

    if output_pdf_dir.exists() or output_pdf_dir.is_symlink():
        if output_pdf_dir.is_symlink() or output_pdf_dir.is_file():
            output_pdf_dir.unlink()
        else:
            raise FileExistsError(
                f"{output_pdf_dir} already exists as a real directory; remove it first."
            )
    output_pdf_dir.symlink_to(pdf_src, target_is_directory=True)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split": split,
        "n_qids": len(kept),
        "qids": wanted_qids,
        "subset_mmqa_path": str(subset_mmqa_path),
        "split_doc_ids_path": str(split_doc_ids_dst),
        "pdf_dir_symlink": str(output_pdf_dir),
    }
    with open(output_root / "subset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created subset dataset at {output_root}")
    print(f"Subset questions: {len(kept)}")
    print(f"MMQA file: {subset_mmqa_path}")
    print(f"Reused full doc-id universe: {split_doc_ids_dst}")
    print(f"Symlinked PDFs: {output_pdf_dir} -> {pdf_src}")


if __name__ == "__main__":
    main()
