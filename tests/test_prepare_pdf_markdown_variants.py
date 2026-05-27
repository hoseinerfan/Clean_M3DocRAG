import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pdf_markdown_variants.py"
SPEC = importlib.util.spec_from_file_location("prepare_pdf_markdown_variants", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CodeguardVariantTests(unittest.TestCase):
    def test_codeguard_suppresses_heuristic_headings_on_native_code_page(self) -> None:
        row = {
            "markdown_source": "pdf_heading",
            "markdown_backend": "native",
            "pdf_outline_heading_count": 1,
            "pdf_heuristic_heading_count": 2,
            "markdown": "\n".join(
                [
                    "# Tutorial 1: Learn about Configs",
                    "## Size of RoI features",
                    "## Whether to use sigmoid",
                    "",
                    "MMDetection, Release 2.18.0",
                    "roi_feat_size=7,",
                    "# Size of RoI features",
                    "num_classes=80,",
                    "# Number of classes for classification",
                    "bbox_coder=dict(",
                    "type='DeltaXYWHBBoxCoder',",
                    "loss_cls=dict(",
                    "use_sigmoid=False,",
                ]
            ),
        }

        detail = MODULE.codeguard_page_detail(
            row, min_code_lines=4, min_code_line_ratio=0.10
        )

        self.assertTrue(detail["code_dense"])
        output = MODULE.variant_markdown(
            "strict_heading_codeguard",
            ["# Tutorial 1: Learn about Configs"],
            ["## Size of RoI features", "## Whether to use sigmoid"],
            ["## Size of RoI features", "## Whether to use sigmoid"],
            [],
        )
        self.assertEqual(output, "# Tutorial 1: Learn about Configs")

    def test_codeguard_preserves_native_report_headings(self) -> None:
        row = {
            "markdown_source": "pdf_heading",
            "markdown_backend": "native",
            "pdf_outline_heading_count": 0,
            "pdf_heuristic_heading_count": 2,
            "markdown": "\n".join(
                [
                    "# Software and Connected Initiatives",
                    "### Progress on Connected Cars and Connected Technologies",
                    "",
                    "TOYOTA MOTOR CORPORATION INTEGRATED REPORT",
                    "Software is becoming an important factor in determining product appeal.",
                    "Connected cars will be applied to a variety of areas.",
                ]
            ),
        }

        detail = MODULE.codeguard_page_detail(
            row, min_code_lines=4, min_code_line_ratio=0.10
        )

        self.assertFalse(detail["code_dense"])
        strict_lines = [
            "# Software and Connected Initiatives",
            "### Progress on Connected Cars and Connected Technologies",
        ]
        output = MODULE.variant_markdown(
            "strict_heading_codeguard", [], strict_lines, strict_lines, strict_lines
        )
        self.assertEqual(output, "\n".join(strict_lines))

    def test_codeguard_does_not_reinterpret_pymupdf4llm_pages(self) -> None:
        row = {
            "markdown_source": "pymupdf4llm_heading",
            "markdown_backend": "pymupdf4llm",
            "pdf_outline_heading_count": 0,
            "pdf_heuristic_heading_count": 1,
            "markdown": "\n".join(
                [
                    "## Configuration",
                    "```",
                    "roi_feat_size=7,",
                    "num_classes=80,",
                    "bbox_coder=dict(",
                    "type='DeltaXYWHBBoxCoder',",
                    "```",
                ]
            ),
        }

        detail = MODULE.codeguard_page_detail(
            row, min_code_lines=4, min_code_line_ratio=0.10
        )

        self.assertFalse(detail["code_dense"])

    def test_cli_writes_codeguard_variant_and_suppression_counters(self) -> None:
        row = {
            "doc_id": "code-doc",
            "page_idx": 0,
            "markdown_source": "pdf_heading",
            "markdown_backend": "native",
            "pdf_outline_heading_count": 1,
            "pdf_heuristic_heading_count": 2,
            "markdown": "\n".join(
                [
                    "# Config Tutorial",
                    "## Size of RoI features",
                    "## Whether to use sigmoid",
                    "",
                    "roi_feat_size=7,",
                    "# Size of RoI features",
                    "num_classes=80,",
                    "# Number of classes for classification",
                    "bbox_coder=dict(",
                    "type='DeltaXYWHBBoxCoder',",
                    "loss_cls=dict(",
                    "use_sigmoid=False,",
                ]
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "pages.jsonl"
            output_dir = root / "variants"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            saved_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT_PATH),
                    "--input-jsonl",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.main()
            finally:
                sys.argv = saved_argv

            summary = json.loads(
                (output_dir / "pdf_markdown_variants.summary.json").read_text(encoding="utf-8")
            )
            output_row = json.loads(
                (output_dir / "doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )

        self.assertEqual(summary["codeguard_code_dense_page_count"], 1)
        self.assertEqual(summary["codeguard_suppressed_heuristic_heading_line_count"], 2)
        self.assertEqual(output_row["markdown"], "# Config Tutorial")
        self.assertTrue(output_row["pdf_variant_codeguard"]["code_dense"])


if __name__ == "__main__":
    unittest.main()
