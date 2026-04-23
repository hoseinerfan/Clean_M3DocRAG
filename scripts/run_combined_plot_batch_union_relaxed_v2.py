#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "scripts" / "render_page_query_heatmaps_and_patch_dots.py"


@dataclass(frozen=True)
class RunSpec:
    alias_dir: str
    qid: str
    mode: str  # "ret4" or "gold"


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec("imgq_nogold__hot_fuzz__6105bed6e83b2189f0e2fd0fb1092822", "6105bed6e83b2189f0e2fd0fb1092822", "ret4"),
    RunSpec("imgq_nogold__golden_sphere__e9047078de74d8240ce689bdc504aea7", "e9047078de74d8240ce689bdc504aea7", "ret4"),
    RunSpec("imgq_gold__warbler_eye__d4d6487894e25da9ba73d0ffb39385b1", "d4d6487894e25da9ba73d0ffb39385b1", "ret4"),
    RunSpec("imgq_gold__trident_neptune__c4bf958b6b268f7026de91e8d26da7e4", "c4bf958b6b268f7026de91e8d26da7e4", "ret4"),
    RunSpec("hard_mixed__ahl_wild_cat__8a36aeac84a462745c5fa5ffacc45c1b", "8a36aeac84a462745c5fa5ffacc45c1b", "ret4"),
    RunSpec("hard__lgbt_billionaires__e783cba0b3df36372d11823e378e5437", "e783cba0b3df36372d11823e378e5437", "ret4"),
    RunSpec("hard__lgbt_billionaires__e783cba0b3df36372d11823e378e5437", "e783cba0b3df36372d11823e378e5437", "gold"),
    RunSpec("hard__flower_logo__e1e6ed53f9ad11813845088f4cf2f6b1", "e1e6ed53f9ad11813845088f4cf2f6b1", "ret4"),
    RunSpec("hard__flower_logo__e1e6ed53f9ad11813845088f4cf2f6b1", "e1e6ed53f9ad11813845088f4cf2f6b1", "gold"),
    RunSpec("hard__ben_piazza_poster__18ecd2ac6c0ac69993b92dc4b30137e8", "18ecd2ac6c0ac69993b92dc4b30137e8", "ret4"),
    RunSpec("hard__ben_piazza_poster__18ecd2ac6c0ac69993b92dc4b30137e8", "18ecd2ac6c0ac69993b92dc4b30137e8", "gold"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run both page-query heatmaps/overlays and patch-dot plots for the common qids "
            "discussed in the audit. By default this prints the planned commands; add "
            "--execute to actually run them."
        )
    )
    parser.add_argument(
        "--root",
        default="output/page_query_heatmaps",
        help="Base output/page_query_heatmaps directory.",
    )
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--embedding-name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--query-token-filter",
        default="full",
        choices=["full", "drop_pad_like", "semantic_only"],
        help="Query-token filter passed through to the shared wrapper.",
    )
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
        "--suffix",
        default="visual_union_relaxed_v2_broadphrases_allplots",
        help="Suffix added to fresh sibling output folders.",
    )
    parser.add_argument(
        "--selection",
        default="all",
        choices=["all", "ret4", "gold"],
        help="Which predefined run set to launch.",
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
        help="Rerun even if the target summaries already exist.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the wrapper. Without this flag, only print the planned commands.",
    )
    return parser.parse_args()


def build_output_dir(root: Path, spec: RunSpec, suffix: str) -> Path:
    stem = "ret4_top4" if spec.mode == "ret4" else "gold_doc_pages_ret1000"
    return root / spec.alias_dir / f"{stem}_{suffix}"


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
        args.query_token_filter,
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


def should_run(spec: RunSpec, selection: str, only_qids: set[str]) -> bool:
    if selection != "all" and spec.mode != selection:
        return False
    if only_qids and spec.qid not in only_qids:
        return False
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not WRAPPER_PATH.exists():
        raise FileNotFoundError(WRAPPER_PATH)

    only_qids = set(args.only_qid)
    planned = 0
    skipped = 0
    failed = 0

    for spec in RUN_SPECS:
        if not should_run(spec, args.selection, only_qids):
            continue

        output_dir = build_output_dir(root, spec, args.suffix)
        heatmap_summary_path = output_dir / "heatmaps_summary.json"
        patch_dot_summary_path = output_dir / "patch_dots_summary.json"
        if (
            heatmap_summary_path.exists()
            and patch_dot_summary_path.exists()
            and not args.overwrite
        ):
            skipped += 1
            print(
                "skip (already exists): "
                f"{heatmap_summary_path} and {patch_dot_summary_path}"
            )
            continue

        cmd = build_command(args, spec, output_dir)
        planned += 1
        print(f"[{planned}] {spec.mode} {spec.alias_dir}")
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
        f"selection={args.selection} root={root} suffix={args.suffix} "
        f"query_token_filter={args.query_token_filter}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
