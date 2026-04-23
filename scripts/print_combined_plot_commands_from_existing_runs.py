#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "scripts" / "render_page_query_heatmaps_and_patch_dots.py"
QID_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{32})(?![0-9a-f])")


@dataclass(frozen=True)
class RunSpec:
    source_dir: Path
    qid: str
    mode: str  # "ret4" or "gold"
    query_token_filter: str
    ignore_pad_scores_in_final_ranking: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an existing page_query_heatmaps directory tree and print the proper combined "
            "heatmap+patch-dot wrapper command for each recognized run directory."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Existing output/page_query_heatmaps root to scan.",
    )
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--embedding-name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--query-labels",
        default=(
            "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/"
            "deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v2.jsonl"
        ),
        help="Query-token visual/non-visual label file.",
    )
    parser.add_argument(
        "--patch-labels",
        default="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far.jsonl",
        help="Patch visual/non-visual label file.",
    )
    parser.add_argument(
        "--output-suffix",
        default="allplots",
        help="Suffix appended to the source run directory name for the fresh combined-output folder.",
    )
    parser.add_argument(
        "--selection",
        default="all",
        choices=["all", "ret4", "gold"],
        help="Only print commands for this run type.",
    )
    parser.add_argument(
        "--only-qid",
        action="append",
        default=[],
        help="Optional qid filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Include commands even if the target allplots folder already has both summaries.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the commands instead of only printing them.",
    )
    return parser.parse_args()


def infer_qid(path: Path) -> str | None:
    for value in (path.name, path.parent.name, str(path)):
        match = QID_RE.search(value)
        if match:
            return match.group(1)
    return None


def infer_mode(run_name: str) -> str | None:
    if run_name.startswith("ret4_top4"):
        return "ret4"
    if run_name.startswith("gold_doc_pages_ret1000"):
        return "gold"
    return None


def infer_query_token_filter(run_name: str) -> str:
    if "semantic_only" in run_name:
        return "semantic_only"
    if "drop_pad" in run_name:
        return "drop_pad_like"
    return "full"


def infer_ignore_pad(run_name: str, query_token_filter: str) -> bool:
    if query_token_filter != "full":
        return False
    return "ignorepadscore" in run_name


def is_already_combined_output(run_dir: Path) -> bool:
    return (
        (run_dir / "heatmaps_summary.json").exists()
        or (run_dir / "patch_dots_summary.json").exists()
        or run_dir.name.endswith("_allplots")
    )


def discover_run_specs(root: Path) -> list[RunSpec]:
    discovered: list[RunSpec] = []
    for parent_dir in sorted(root.iterdir()):
        if not parent_dir.is_dir():
            continue
        for run_dir in sorted(parent_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            mode = infer_mode(run_dir.name)
            if mode is None:
                continue
            qid = infer_qid(run_dir)
            if qid is None:
                continue
            query_token_filter = infer_query_token_filter(run_dir.name)
            discovered.append(
                RunSpec(
                    source_dir=run_dir,
                    qid=qid,
                    mode=mode,
                    query_token_filter=query_token_filter,
                    ignore_pad_scores_in_final_ranking=infer_ignore_pad(
                        run_dir.name,
                        query_token_filter,
                    ),
                )
            )
    return discovered


def should_keep(spec: RunSpec, selection: str, only_qids: set[str]) -> bool:
    if selection != "all" and spec.mode != selection:
        return False
    if only_qids and spec.qid not in only_qids:
        return False
    return True


def build_output_dir(spec: RunSpec, output_suffix: str) -> Path:
    return spec.source_dir.parent / f"{spec.source_dir.name}_{output_suffix}"


def build_command(args: argparse.Namespace, spec: RunSpec, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(WRAPPER_PATH),
        "--qid",
        spec.qid,
        "--data_name",
        args.data_name,
        "--split",
        args.split,
        "--embedding_name",
        args.embedding_name,
        "--query_token_filter",
        spec.query_token_filter,
        "--shared-output-dir",
        "--heatmap-summary-name",
        "heatmaps_summary.json",
        "--patch-dot-summary-name",
        "patch_dots_summary.json",
        "--splice-query-token-labels",
        args.query_labels,
        "--splice-patch-labels-jsonl",
        args.patch_labels,
        "--output-dir",
        str(output_dir),
    ]
    if spec.ignore_pad_scores_in_final_ranking:
        cmd.append("--ignore-pad-scores-in-final-ranking")
    if spec.mode == "ret4":
        cmd.extend(
            [
                "--n_retrieval_pages",
                "4",
                "--plot-rank-start",
                "1",
                "--plot-rank-count",
                "4",
            ]
        )
    else:
        cmd.extend(
            [
                "--n_retrieval_pages",
                "1000",
                "--gold-doc-pages",
                "--explicit-page-mode",
                "retrieved_contrib",
            ]
        )
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not WRAPPER_PATH.exists():
        raise FileNotFoundError(WRAPPER_PATH)
    if not root.exists():
        raise FileNotFoundError(root)

    only_qids = set(args.only_qid)
    planned = 0
    skipped = 0
    failed = 0

    for spec in discover_run_specs(root):
        if not should_keep(spec, args.selection, only_qids):
            continue
        if is_already_combined_output(spec.source_dir):
            continue

        output_dir = build_output_dir(spec, args.output_suffix)
        heatmaps_summary = output_dir / "heatmaps_summary.json"
        patch_dots_summary = output_dir / "patch_dots_summary.json"
        if heatmaps_summary.exists() and patch_dots_summary.exists() and not args.overwrite:
            skipped += 1
            print(f"skip (already exists): {output_dir}")
            continue

        cmd = build_command(args, spec, output_dir)
        planned += 1
        print(
            f"[{planned}] qid={spec.qid} mode={spec.mode} "
            f"query_token_filter={spec.query_token_filter} "
            f"ignore_pad_scores_in_final_ranking={spec.ignore_pad_scores_in_final_ranking}"
        )
        print("  source: " + str(spec.source_dir))
        print("  output: " + str(output_dir))
        print("  " + " ".join(shlex.quote(part) for part in cmd))

        if not args.execute:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"  failed: exit_code={exc.returncode}")

    print(
        f"planned={planned} skipped={skipped} failed={failed} execute={args.execute} "
        f"selection={args.selection} root={root}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
