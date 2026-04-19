#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTTER_PATH = REPO_ROOT / "scripts" / "plot_page_query_token_heatmaps.py"
QID_PATTERN = re.compile(r"([0-9a-f]{32})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each qid-named directory under output/page_query_heatmaps, create a nested "
            "subdirectory and render page-query heatmaps into it."
        )
    )
    parser.add_argument(
        "--root",
        default="output/page_query_heatmaps",
        help="Directory containing qid or alias__qid subdirectories.",
    )
    parser.add_argument(
        "--subdir",
        default="heatmaps",
        help="Name of the nested directory to create under each qid folder.",
    )
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--embedding-name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument("--n-retrieval-pages", type=int, default=4)
    parser.add_argument("--plot-rank-start", type=int, default=1)
    parser.add_argument("--plot-rank-count", type=int, default=4)
    parser.add_argument(
        "--query-token-filter",
        default="full",
        help="Passed through to plot_page_query_token_heatmaps.py",
    )
    parser.add_argument(
        "--overlay-on-page",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also render overlay PNGs in the nested heatmap folder. Defaults off.",
    )
    parser.add_argument(
        "--save-patch-crops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save patch crops in the nested heatmap folder.",
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
        help="Rerun even if <subdir>/summary.json already exists.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the plotter. Without this flag, only print the planned commands.",
    )
    return parser.parse_args()


def extract_qid(dirname: str) -> str | None:
    match = QID_PATTERN.search(dirname)
    if match is None:
        return None
    return match.group(1)


def build_command(args: argparse.Namespace, qid: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(PLOTTER_PATH),
        "--qid",
        qid,
        "--data_name",
        args.data_name,
        "--split",
        args.split,
        "--embedding_name",
        args.embedding_name,
        "--n_retrieval_pages",
        str(args.n_retrieval_pages),
        "--plot-rank-start",
        str(args.plot_rank_start),
        "--plot-rank-count",
        str(args.plot_rank_count),
        "--query_token_filter",
        args.query_token_filter,
        "--output-dir",
        str(output_dir),
    ]
    cmd.append("--overlay-on-page" if args.overlay_on_page else "--no-overlay-on-page")
    cmd.append("--save-patch-crops" if args.save_patch_crops else "--no-save-patch-crops")
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not PLOTTER_PATH.exists():
        raise FileNotFoundError(PLOTTER_PATH)

    only_qids = set(args.only_qid)
    planned = 0
    skipped = 0
    failed = 0

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue

        qid = extract_qid(entry.name)
        if qid is None:
            skipped += 1
            print(f"skip (no qid suffix): {entry}")
            continue
        if only_qids and qid not in only_qids:
            skipped += 1
            print(f"skip (filtered): {entry}")
            continue

        nested_dir = entry / args.subdir
        summary_path = nested_dir / "summary.json"
        if summary_path.exists() and not args.overwrite:
            skipped += 1
            print(f"skip (already exists): {summary_path}")
            continue

        cmd = build_command(args, qid=qid, output_dir=nested_dir)
        planned += 1
        print(f"[{planned}] {entry.name}")
        print("  " + " ".join(shlex.quote(part) for part in cmd))

        if not args.execute:
            continue

        nested_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"  failed: exit_code={exc.returncode}")

    print(
        f"planned={planned} skipped={skipped} failed={failed} execute={args.execute} "
        f"root={root} subdir={args.subdir}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
