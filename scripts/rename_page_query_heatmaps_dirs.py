#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


RENAMES = {
    "00c6daf76670c32dccf8effff5b552a7": "easy_ret4_success__who_sings_are_you_gonna_go_my_way__00c6daf76670c32dccf8effff5b552a7",
    "e783cba0b3df36372d11823e378e5437": "hard_audit__lgbt_billionaires_bald_thick_glasses__e783cba0b3df36372d11823e378e5437",
    "8a36aeac84a462745c5fa5ffacc45c1b": "hard_audit_mixed__ahl_pouncing_wild_cat_jersey__8a36aeac84a462745c5fa5ffacc45c1b",
    "e1e6ed53f9ad11813845088f4cf2f6b1": "hard_audit__dzenis_beganovic_flower_logo__e1e6ed53f9ad11813845088f4cf2f6b1",
    "18ecd2ac6c0ac69993b92dc4b30137e8": "hard_audit__ben_piazza_left_half_womans_face_poster__18ecd2ac6c0ac69993b92dc4b30137e8",
    "d4d6487894e25da9ba73d0ffb39385b1": "imagelistq_ret4_correct_with_gold__bush_warbler_orange_stripe_eye__d4d6487894e25da9ba73d0ffb39385b1",
    "c4bf958b6b268f7026de91e8d26da7e4": "imagelistq_ret4_correct_with_gold__trident_symbol_planet_neptune__c4bf958b6b268f7026de91e8d26da7e4",
    "e9047078de74d8240ce689bdc504aea7": "imagelistq_ret4_correct_without_gold__trophy_golden_sphere_player__e9047078de74d8240ce689bdc504aea7",
    "6105bed6e83b2189f0e2fd0fb1092822": "imagelistq_ret4_correct_without_gold__garth_jennings_two_police_officers__6105bed6e83b2189f0e2fd0fb1092822",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename known qid directories under output/page_query_heatmaps to "
            "human-readable names while preserving the original qid suffix."
        )
    )
    parser.add_argument(
        "--root",
        default="output/page_query_heatmaps",
        help="Directory containing qid-named subdirectories.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename directories. Without this flag, only print the planned renames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    renamed = 0
    skipped = 0

    for old_name, new_name in RENAMES.items():
        old_path = root / old_name
        new_path = root / new_name

        if not old_path.exists():
            skipped += 1
            print(f"skip (missing): {old_path}")
            continue

        if new_path.exists():
            skipped += 1
            print(f"skip (target exists): {new_path}")
            continue

        print(f"{old_path} -> {new_path}")
        if args.execute:
            old_path.rename(new_path)
        renamed += 1

    print(f"planned_or_renamed={renamed} skipped={skipped} execute={args.execute}")


if __name__ == "__main__":
    main()
