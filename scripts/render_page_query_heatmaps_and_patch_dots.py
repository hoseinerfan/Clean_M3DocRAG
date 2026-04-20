#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HEATMAP_PLOTTER = REPO_ROOT / "scripts" / "plot_page_query_token_heatmaps.py"
PATCH_DOT_PLOTTER = REPO_ROOT / "scripts" / "plot_page_query_patch_dot_grid.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the heatmap/overlay plotter and the patch-dot plotter with one shared command. "
            "By default outputs are written to separate subdirectories, but a shared-output mode "
            "is available when you want both plotters to write into the same folder."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Free-form query text")
    group.add_argument("--qid", help="Benchmark qid to load from MMQA_<split>.jsonl")

    parser.add_argument("--data_name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--gold", help="Optional MMQA_<split>.jsonl override for --qid mode")
    parser.add_argument("--embedding_name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--faiss_index_type",
        default="ivfflat",
        choices=["flatip", "ivfflat", "ivfpq"],
    )
    parser.add_argument("--retrieval_model_type", default="colpali", choices=["colpali"])
    parser.add_argument("--retrieval_model_name_or_path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval_adapter_model_name_or_path", default="colpali-v1.2")
    parser.add_argument("--n_retrieval_pages", type=int, default=4)
    parser.add_argument(
        "--query_token_filter",
        default="full",
        choices=["full", "drop_pad_like", "semantic_only"],
    )
    parser.add_argument(
        "--ignore-pad-scores-in-final-ranking",
        action="store_true",
        help="Pass through to both plotters.",
    )
    parser.add_argument("--plot-rank-start", type=int, default=1)
    parser.add_argument("--plot-rank-count", type=int, default=4)
    parser.add_argument(
        "--page-uid",
        action="append",
        default=[],
        help="Explicit page_uid to render. Can be passed multiple times.",
    )
    parser.add_argument(
        "--gold-doc-pages",
        action="store_true",
        help="When used with --qid, expand all pages from the qid's gold supporting docs.",
    )
    parser.add_argument(
        "--explicit-page-mode",
        default="direct_page_maxsim",
        choices=["direct_page_maxsim", "retrieved_contrib"],
    )
    parser.add_argument(
        "--nonspatial-token-position",
        default="suffix",
        choices=["prefix", "suffix"],
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory.",
    )
    parser.add_argument(
        "--shared-output-dir",
        action="store_true",
        help=(
            "Write both plotter outputs directly into --output-dir. The wrapper renames each "
            "plotter's summary.json to keep them from clobbering each other."
        ),
    )
    parser.add_argument("--heatmap-subdir", default="heatmaps")
    parser.add_argument("--patch-dot-subdir", default="patch_dots")
    parser.add_argument("--heatmap-summary-name", default="heatmaps_summary.json")
    parser.add_argument("--patch-dot-summary-name", default="patch_dots_summary.json")
    parser.add_argument("--skip-heatmaps", action="store_true")
    parser.add_argument("--skip-patch-dots", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )

    parser.add_argument(
        "--overlay-on-page",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Heatmap-only flag.",
    )
    parser.add_argument(
        "--overlay-mode",
        default="original_exact",
        choices=["aspectfit", "processor_exact", "original_exact"],
        help="Heatmap-only flag.",
    )
    parser.add_argument(
        "--overlay-image-size",
        type=int,
        default=1280,
        help="Heatmap-only flag.",
    )
    parser.add_argument(
        "--overlay-clean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Heatmap-only flag.",
    )
    parser.add_argument(
        "--save-patch-crops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Heatmap-only flag.",
    )
    parser.add_argument(
        "--contrib-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Heatmap-only flag.",
    )
    parser.add_argument(
        "--swap-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Heatmap-only flag.",
    )
    parser.add_argument("--heatmap-cell-width", type=int, default=28)
    parser.add_argument("--heatmap-cell-height", type=int, default=4)
    parser.add_argument("--page-token-tick-step", type=int, default=64)

    parser.add_argument(
        "--splice-patch-labels-jsonl",
        default="",
        help="Patch-dot-only flag.",
    )
    parser.add_argument(
        "--splice-query-token-labels",
        default="",
        help="Patch-dot-only flag.",
    )
    parser.add_argument("--dot-cell-width", type=int, default=12)
    parser.add_argument("--dot-cell-height", type=int, default=38)
    parser.add_argument("--patch-tick-step", type=int, default=32)
    parser.add_argument("--dot-radius", type=int, default=4)

    return parser.parse_args()


def append_shared_args(cmd: list[str], args: argparse.Namespace, output_dir: Path) -> list[str]:
    if args.query is not None:
        cmd.extend(["--query", args.query])
    else:
        cmd.extend(["--qid", args.qid])

    cmd.extend(
        [
            "--data_name",
            args.data_name,
            "--split",
            args.split,
            "--embedding_name",
            args.embedding_name,
            "--faiss_index_type",
            args.faiss_index_type,
            "--retrieval_model_type",
            args.retrieval_model_type,
            "--retrieval_model_name_or_path",
            args.retrieval_model_name_or_path,
            "--retrieval_adapter_model_name_or_path",
            args.retrieval_adapter_model_name_or_path,
            "--n_retrieval_pages",
            str(args.n_retrieval_pages),
            "--query_token_filter",
            args.query_token_filter,
            "--plot-rank-start",
            str(args.plot_rank_start),
            "--plot-rank-count",
            str(args.plot_rank_count),
            "--explicit-page-mode",
            args.explicit_page_mode,
            "--nonspatial-token-position",
            args.nonspatial_token_position,
            "--output-dir",
            str(output_dir),
        ]
    )
    if args.gold:
        cmd.extend(["--gold", args.gold])
    if args.ignore_pad_scores_in_final_ranking:
        cmd.append("--ignore-pad-scores-in-final-ranking")
    if args.gold_doc_pages:
        cmd.append("--gold-doc-pages")
    for page_uid in args.page_uid:
        cmd.extend(["--page-uid", page_uid])
    return cmd


def build_heatmap_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = append_shared_args([sys.executable, str(HEATMAP_PLOTTER)], args, output_dir)
    cmd.extend(
        [
            "--cell-width",
            str(args.heatmap_cell_width),
            "--cell-height",
            str(args.heatmap_cell_height),
            "--page-token-tick-step",
            str(args.page_token_tick_step),
            "--overlay-mode",
            args.overlay_mode,
            "--overlay-image-size",
            str(args.overlay_image_size),
        ]
    )
    cmd.append("--contrib-only" if args.contrib_only else "--no-contrib-only")
    cmd.append("--swap-axes" if args.swap_axes else "--no-swap-axes")
    cmd.append("--overlay-on-page" if args.overlay_on_page else "--no-overlay-on-page")
    cmd.append("--overlay-clean" if args.overlay_clean else "--no-overlay-clean")
    cmd.append("--save-patch-crops" if args.save_patch_crops else "--no-save-patch-crops")
    return cmd


def build_patch_dot_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = append_shared_args([sys.executable, str(PATCH_DOT_PLOTTER)], args, output_dir)
    cmd.extend(
        [
            "--cell-width",
            str(args.dot_cell_width),
            "--cell-height",
            str(args.dot_cell_height),
            "--patch-tick-step",
            str(args.patch_tick_step),
            "--dot-radius",
            str(args.dot_radius),
        ]
    )
    if args.splice_patch_labels_jsonl:
        cmd.extend(["--splice-patch-labels-jsonl", args.splice_patch_labels_jsonl])
    if args.splice_query_token_labels:
        cmd.extend(["--splice-query-token-labels", args.splice_query_token_labels])
    return cmd


def run_command(
    label: str,
    cmd: list[str],
    output_dir: Path,
    summary_name: str,
    dry_run: bool,
) -> None:
    print(f"[{label}]")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    summary_path = output_dir / "summary.json"
    target_summary_path = output_dir / summary_name
    if summary_name != "summary.json":
        print(f"  rename {summary_path.name} -> {target_summary_path.name}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if summary_name != "summary.json":
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary_path.replace(target_summary_path)


def main() -> None:
    args = parse_args()

    if args.skip_heatmaps and args.skip_patch_dots:
        raise ValueError("At least one plotter must remain enabled.")
    if (
        args.shared_output_dir
        and not args.skip_heatmaps
        and not args.skip_patch_dots
        and args.heatmap_summary_name == args.patch_dot_summary_name
    ):
        raise ValueError(
            "--shared-output-dir requires distinct --heatmap-summary-name and "
            "--patch-dot-summary-name values."
        )
    if not HEATMAP_PLOTTER.exists():
        raise FileNotFoundError(HEATMAP_PLOTTER)
    if not PATCH_DOT_PLOTTER.exists():
        raise FileNotFoundError(PATCH_DOT_PLOTTER)

    base_output_dir = Path(args.output_dir)
    planned: list[tuple[str, list[str], Path, str]] = []

    if not args.skip_heatmaps:
        heatmap_output_dir = base_output_dir if args.shared_output_dir else (base_output_dir / args.heatmap_subdir)
        planned.append(
            (
                "heatmaps",
                build_heatmap_command(args, heatmap_output_dir),
                heatmap_output_dir,
                args.heatmap_summary_name,
            )
        )

    if not args.skip_patch_dots:
        patch_dot_output_dir = base_output_dir if args.shared_output_dir else (base_output_dir / args.patch_dot_subdir)
        planned.append(
            (
                "patch_dots",
                build_patch_dot_command(args, patch_dot_output_dir),
                patch_dot_output_dir,
                args.patch_dot_summary_name,
            )
        )

    for label, cmd, output_dir, summary_name in planned:
        run_command(label, cmd, output_dir=output_dir, summary_name=summary_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
