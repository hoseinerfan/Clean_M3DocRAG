#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    export: str
    patterns: tuple[str, ...]
    required_terms: tuple[str, ...] = ()
    preferred_terms: tuple[str, ...] = ()
    reject_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    aliases: tuple[str, ...]
    env_script: str
    work_root_export: str
    work_root_default: str
    artifacts: tuple[ArtifactSpec, ...]


CUSTOM_ROOT = "/mmfs1/scratch/jacks.local/aerfanshekooh/custom"


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        label="m3docvqa",
        aliases=("m3docvqa", "m3dovqa", "m3-docvqa", "mmqa"),
        env_script="scripts/m3docvqa_internal_env.sh",
        work_root_export="REPO_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/Clean_M3DocRAG",
        artifacts=(
            ArtifactSpec(
                "gold",
                "M3DOCVQA_GOLD",
                ("MMQA_dev.jsonl",),
                required_terms=("m3-docvqa", "multimodalqa"),
            ),
            ArtifactSpec(
                "dense_pred",
                "M3DOCVQA_DENSE_PRED",
                ("*plain_top224*.prediction.json",),
                required_terms=("mmqa",),
                preferred_terms=("nprobe4", "effdiag", "all"),
                reject_terms=("shard_",),
            ),
            ArtifactSpec(
                "sparse_pred",
                "M3DOCVQA_SPARSE_PRED",
                ("*splade*.prediction.json",),
                required_terms=("mmqa", "splade"),
                preferred_terms=("m3docvqa_splade_mmqa_dev",),
            ),
            ArtifactSpec(
                "page_text_jsonl",
                "M3DOCVQA_PAGE_TEXT_JSONL",
                ("*m3docvqa*page_text*.jsonl", "*m3docvqa*dev_page_text.jsonl"),
                preferred_terms=("m3docvqa_page_text",),
            ),
            ArtifactSpec(
                "baseline_pred",
                "M3DOCVQA_BASELINE_PRED",
                ("*baseline*.prediction.json",),
                required_terms=("mmqa",),
                preferred_terms=("ret1000", "nprobe4"),
            ),
        ),
    ),
    DatasetSpec(
        label="mmdocir",
        aliases=("mmdocir", "mm-docir"),
        env_script="mmdocir/env_hpc.sh",
        work_root_export="MMDocIR_WORK_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/MMDocIR_M3DocRAG",
        artifacts=(
            ArtifactSpec("gold", "MMDOCIR_GOLD", ("MMQA_dev.jsonl",), required_terms=("mm-docir",)),
            ArtifactSpec("doc_pages", "MMDOCIR_DOC_PAGES", ("doc_pages_dev.jsonl",), required_terms=("mm-docir",)),
            ArtifactSpec("dense_pred", "MMDOCIR_DENSE_PRED", ("plain_top224_ret1000_prediction.json",), required_terms=("mmdocir",)),
            ArtifactSpec(
                "sparse_pred",
                "MMDOCIR_SPARSE_PRED",
                ("mmdocir_splade_ret1000.prediction.json",),
                required_terms=("mmdocir", "splade"),
            ),
            ArtifactSpec("baseline_pred", "MMDOCIR_BASELINE_PRED", ("baseline_ret1000.json",), required_terms=("mmdocir",)),
            ArtifactSpec(
                "pdf_markdown_jsonl",
                "MMDOCIR_PDF_MD_JSONL",
                ("doc_pages_dev_with_pdf_markdown.jsonl",),
                required_terms=("mmdocir",),
            ),
            ArtifactSpec(
                "variant_dir_summary",
                "MMDOCIR_VARIANT_SUMMARY",
                ("pdf_markdown_variants.summary.json",),
                required_terms=("mmdocir",),
            ),
        ),
    ),
    DatasetSpec(
        label="sciegqa",
        aliases=("sciegqa", "sci-egqa"),
        env_script="sciegqa/env_hpc.sh",
        work_root_export="SciEGQA_WORK_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/SciEGQA_M3DocRAG",
        artifacts=(
            ArtifactSpec("gold", "SCIEGQA_GOLD", ("MMQA_dev.jsonl",), required_terms=("sci-egqa",)),
            ArtifactSpec("doc_pages", "SCIEGQA_DOC_PAGES", ("doc_pages_dev.jsonl",), required_terms=("sci-egqa",)),
            ArtifactSpec("dense_pred", "SCIEGQA_DENSE_PRED", ("plain_top224_ret1000_prediction.json",), required_terms=("sciegqa",)),
            ArtifactSpec(
                "sparse_pred",
                "SCIEGQA_SPARSE_PRED",
                ("sciegqa_splade_ret1000.prediction.json",),
                required_terms=("sciegqa", "splade"),
            ),
            ArtifactSpec("baseline_pred", "SCIEGQA_BASELINE_PRED", ("baseline_ret1000.json",), required_terms=("sciegqa",)),
            ArtifactSpec(
                "pdf_markdown_jsonl",
                "SCIEGQA_PDF_MD_JSONL",
                ("doc_pages_dev_with_pdf_markdown.jsonl",),
                required_terms=("sciegqa",),
            ),
            ArtifactSpec(
                "variant_dir_summary",
                "SCIEGQA_VARIANT_SUMMARY",
                ("pdf_markdown_variants.summary.json",),
                required_terms=("sciegqa",),
            ),
        ),
    ),
    DatasetSpec(
        label="vidoseek",
        aliases=("vidoseek",),
        env_script="vidoseek/env_hpc.sh",
        work_root_export="VIDOSEEK_WORK_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/ViDoSeek_M3DocRAG",
        artifacts=(
            ArtifactSpec("gold", "VIDOSEEK_GOLD", ("MMQA_dev.jsonl",), required_terms=("vidoseek",)),
            ArtifactSpec("doc_pages", "VIDOSEEK_DOC_PAGES", ("doc_pages_dev.jsonl",), required_terms=("vidoseek",)),
            ArtifactSpec("dense_pred", "VIDOSEEK_DENSE_PRED", ("plain_top224_ret1000_prediction.json",), required_terms=("vidoseek",)),
            ArtifactSpec(
                "sparse_pred",
                "VIDOSEEK_SPARSE_PRED",
                ("vidoseek_splade_ret1000.prediction.json",),
                required_terms=("vidoseek", "splade"),
            ),
            ArtifactSpec("baseline_pred", "VIDOSEEK_BASELINE_PRED", ("baseline_ret1000.json",), required_terms=("vidoseek",)),
            ArtifactSpec(
                "pdf_markdown_jsonl",
                "VIDOSEEK_PDF_MD_JSONL",
                ("doc_pages_dev_with_pdf_markdown.jsonl",),
                required_terms=("vidoseek",),
            ),
            ArtifactSpec(
                "variant_dir_summary",
                "VIDOSEEK_VARIANT_SUMMARY",
                ("pdf_markdown_variants.summary.json",),
                required_terms=("vidoseek",),
            ),
        ),
    ),
    DatasetSpec(
        label="vidore",
        aliases=("vidore", "vidore-v3"),
        env_script="vidore/env_hpc.sh",
        work_root_export="VIDORE_WORK_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/ViDoRe_M3DocRAG",
        artifacts=(
            ArtifactSpec("gold", "VIDORE_GOLD", ("MMQA_dev.jsonl",), required_terms=("vidore-v3",)),
            ArtifactSpec("doc_pages", "VIDORE_DOC_PAGES", ("doc_pages_dev.jsonl",), required_terms=("vidore-v3",)),
            ArtifactSpec("dense_pred", "VIDORE_DENSE_PRED", ("plain_top224_ret1000_prediction.json",), required_terms=("vidore-v3",)),
            ArtifactSpec(
                "sparse_pred",
                "VIDORE_SPARSE_PRED",
                ("vidore-v3_splade_ret1000.prediction.json",),
                required_terms=("vidore-v3", "splade"),
            ),
            ArtifactSpec("baseline_pred", "VIDORE_BASELINE_PRED", ("baseline_ret1000.json",), required_terms=("vidore-v3",)),
            ArtifactSpec(
                "variant_dir_summary",
                "VIDORE_VARIANT_SUMMARY",
                ("pdf_markdown_variants.summary.json", "markdown_variants/pdf_markdown_variants.summary.json"),
                required_terms=("vidore-v3",),
            ),
        ),
    ),
    DatasetSpec(
        label="dude",
        aliases=("dude",),
        env_script="dude/env_hpc.sh",
        work_root_export="DUDE_WORK_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/DUDE_M3DocRAG",
        artifacts=(
            ArtifactSpec("gold", "DUDE_GOLD", ("MMQA_dev.jsonl",), required_terms=("dude",)),
            ArtifactSpec("doc_pages", "DUDE_DOC_PAGES", ("doc_pages_dev.jsonl",), required_terms=("dude",)),
            ArtifactSpec("dense_pred", "DUDE_DENSE_PRED", ("plain_top224_ret1000_prediction.json",), required_terms=("dude",)),
            ArtifactSpec(
                "sparse_pred",
                "DUDE_SPARSE_PRED",
                ("dude_splade_ret1000.prediction.json",),
                required_terms=("dude", "splade"),
            ),
            ArtifactSpec("baseline_pred", "DUDE_BASELINE_PRED", ("baseline_ret1000.json",), required_terms=("dude",)),
            ArtifactSpec(
                "pdf_markdown_jsonl",
                "DUDE_PDF_MD_JSONL",
                ("doc_pages_dev_with_pdf_markdown.jsonl",),
                required_terms=("dude",),
            ),
            ArtifactSpec(
                "variant_dir_summary",
                "DUDE_VARIANT_SUMMARY",
                ("pdf_markdown_variants.summary.json",),
                required_terms=("dude",),
            ),
        ),
    ),
    DatasetSpec(
        label="opendocvqa",
        aliases=("opendocvqa",),
        env_script="opendocvqa/env_hpc.sh",
        work_root_export="OPENDOCVQA_WORK_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/OpenDocVQA_M3DocRAG",
        artifacts=(
            ArtifactSpec("gold", "OPENDOCVQA_GOLD", ("MMQA_dev.jsonl",), required_terms=("opendocvqa",)),
            ArtifactSpec("doc_pages", "OPENDOCVQA_DOC_PAGES", ("doc_pages_dev.jsonl",), required_terms=("opendocvqa",)),
            ArtifactSpec("dense_pred", "OPENDOCVQA_DENSE_PRED", ("plain_top224_ret1000_prediction.json",), required_terms=("opendocvqa",)),
            ArtifactSpec(
                "sparse_pred",
                "OPENDOCVQA_SPARSE_PRED",
                ("opendocvqa_splade_ret1000.prediction.json",),
                required_terms=("opendocvqa", "splade"),
            ),
            ArtifactSpec(
                "page_text_jsonl",
                "OPENDOCVQA_PAGE_TEXT_JSONL",
                ("opendocvqa_page_text_dev.jsonl",),
                required_terms=("opendocvqa",),
            ),
        ),
    ),
    DatasetSpec(
        label="mmlongbench",
        aliases=("mmlongbench", "mmlongbench-docqa"),
        env_script="mmlongbench/env_hpc.sh",
        work_root_export="MMLONGBENCH_WORK_ROOT",
        work_root_default=f"{CUSTOM_ROOT}/MMLongBench_M3DocRAG",
        artifacts=(
            ArtifactSpec("gold", "MMLONGBENCH_GOLD", ("MMQA_dev.jsonl",), required_terms=("mmlongbench-docqa",)),
            ArtifactSpec("doc_pages", "MMLONGBENCH_DOC_PAGES", ("doc_pages_dev.jsonl",), required_terms=("mmlongbench-docqa",)),
            ArtifactSpec(
                "dense_pred",
                "MMLONGBENCH_DENSE_PRED",
                ("plain_top224_ret1000_prediction.json",),
                required_terms=("mmlongbench-docqa",),
            ),
            ArtifactSpec(
                "sparse_pred",
                "MMLONGBENCH_SPARSE_PRED",
                ("mmlongbench-docqa_splade_ret1000.prediction.json",),
                required_terms=("mmlongbench-docqa", "splade"),
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover important HPC dataset/artifact paths and write a reusable JSON "
            "manifest plus a shell export file."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root to scan. Repeatable. Defaults to known /mmfs1 and /ces custom roots when present.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset label to include. Repeatable. Defaults to all known datasets.",
    )
    parser.add_argument(
        "--output-json",
        default="hpc_vital_paths.generated.json",
        help="Path for JSON manifest.",
    )
    parser.add_argument(
        "--output-env",
        default="hpc_vital_paths.generated.env",
        help="Path for shell export file.",
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def default_roots() -> list[Path]:
    custom_root = Path(os.environ.get("HPC_CUSTOM_ROOT", CUSTOM_ROOT))
    ces_root = Path("/ces/scratch/jacks.local/aerfanshekooh/custom")
    roots = [
        custom_root / "outputs",
        custom_root / "Clean_M3DocRAG",
        *(Path(dataset.work_root_default) for dataset in DATASETS),
        ces_root / "outputs",
        ces_root / "Clean_M3DocRAG",
        *(
            Path(str(dataset.work_root_default).replace(str(custom_root), str(ces_root)))
            for dataset in DATASETS
        ),
        Path.cwd(),
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not str(root):
            continue
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        key = str(resolved)
        if key not in seen and root.exists():
            seen.add(key)
            out.append(root)
    return out


def find_by_patterns(roots: list[Path], patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for pattern in patterns:
            if "/" in pattern:
                cmd = ["find", str(root), "-path", f"*{pattern}", "-type", "f"]
            else:
                cmd = ["find", str(root), "-type", "f", "-name", pattern]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                continue
            for line in result.stdout.splitlines():
                path = line.strip()
                if path and path not in seen:
                    seen.add(path)
                    found.append(path)
    return found


def score_path(path: str, dataset: DatasetSpec, spec: ArtifactSpec) -> int:
    lower = path.lower()
    score = 0
    for term in dataset.aliases:
        if term.lower() in lower:
            score += 10
    for term in spec.required_terms:
        if term.lower() in lower:
            score += 20
        else:
            score -= 100
    for term in spec.preferred_terms:
        if term.lower() in lower:
            score += 5
    for term in spec.reject_terms:
        if term.lower() in lower:
            score -= 50
    if "/output/" in lower:
        score += 2
    if "/data/" in lower and spec.key in {"gold", "doc_pages"}:
        score += 2
    return score


def choose_candidates(
    paths: list[str],
    dataset: DatasetSpec,
    spec: ArtifactSpec,
    max_candidates: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    scored = [
        {"path": path, "score": score_path(path, dataset, spec)}
        for path in paths
    ]
    scored.sort(key=lambda row: (-int(row["score"]), str(row["path"])))
    viable = [row for row in scored if int(row["score"]) >= 0]
    candidates = viable[:max_candidates] if viable else scored[:max_candidates]
    selected = str(candidates[0]["path"]) if candidates and int(candidates[0]["score"]) >= 0 else None
    return selected, candidates


def dataset_specs(selected: list[str]) -> list[DatasetSpec]:
    if not selected:
        return list(DATASETS)
    selected_lower = {item.lower() for item in selected}
    specs: list[DatasetSpec] = []
    for dataset in DATASETS:
        names = {dataset.label.lower(), *(alias.lower() for alias in dataset.aliases)}
        if names & selected_lower:
            specs.append(dataset)
    missing = selected_lower - {
        name
        for dataset in specs
        for name in {dataset.label.lower(), *(alias.lower() for alias in dataset.aliases)}
    }
    if missing:
        raise SystemExit(f"Unknown dataset(s): {', '.join(sorted(missing))}")
    return specs


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.root] if args.root else default_roots()
    roots = [root for root in roots if root.exists()]
    if not roots:
        raise SystemExit("No scan roots exist. Pass --root /path/to/custom.")

    manifest: dict[str, Any] = {
        "scan_roots": [str(root) for root in roots],
        "datasets": {},
        "exports": {},
    }
    env_lines = [
        "# Generated by scripts/discover_hpc_vital_paths.py",
        "# Source this file before running cross-dataset helpers:",
        "#   source hpc_vital_paths.generated.env",
        "",
    ]

    for dataset in dataset_specs(list(args.dataset)):
        work_root = os.environ.get(dataset.work_root_export, dataset.work_root_default)
        dataset_row: dict[str, Any] = {
            "env_script": dataset.env_script,
            "work_root_export": dataset.work_root_export,
            "work_root": work_root,
            "artifacts": {},
        }
        env_lines.append(f"export {dataset.work_root_export}={shell_quote(work_root)}")
        manifest["exports"][dataset.work_root_export] = work_root
        for spec in dataset.artifacts:
            matches = find_by_patterns(roots, spec.patterns)
            selected, candidates = choose_candidates(
                matches,
                dataset,
                spec,
                int(args.max_candidates),
            )
            dataset_row["artifacts"][spec.key] = {
                "export": spec.export,
                "selected": selected,
                "candidates": candidates,
                "patterns": list(spec.patterns),
            }
            if selected:
                manifest["exports"][spec.export] = selected
                env_lines.append(f"export {spec.export}={shell_quote(selected)}")
        env_lines.append("")
        manifest["datasets"][dataset.label] = dataset_row

    output_json = Path(args.output_json)
    output_env = Path(args.output_env)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_env.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_env.write_text("\n".join(env_lines).rstrip() + "\n", encoding="utf-8")

    print(f"saved_json: {output_json}")
    print(f"saved_env: {output_env}")
    print("")
    for label, row in manifest["datasets"].items():
        print(f"== {label} ==")
        for key, artifact in row["artifacts"].items():
            selected = artifact.get("selected")
            status = "ok" if selected else "missing"
            print(f"{key}: {status} {selected or ''}")


if __name__ == "__main__":
    main()
