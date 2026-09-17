import importlib.util
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_m3docvqa_pseudo_gold_reader_input.py"
SPEC = importlib.util.spec_from_file_location("build_m3docvqa_pseudo_gold_reader_input", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildM3DocVQAPseudoGoldReaderInputTests(unittest.TestCase):
    def test_injection_launcher_validation_blocks(self) -> None:
        launcher = SCRIPT.parent.parent / "examples" / "sbatch_racs_gold_injection_top4.sh"
        blocks = [s.split("\nPY\n", 1)[0] for s in launcher.read_text().split("<<'PY'\n")[1:]]
        self.assertEqual(len(blocks), 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_path, input_path, subset_path, base_path, manifest_path = (
                root / name for name in ("paper.jsonl", "input.json", "subset.jsonl", "base.json", "manifest.json"))
            paper = [{"qid": f"q{i}", "question": "Where?", "answers": [{"answer": "test"}],
                      "metadata": {"gold_page_uids": ["d_page0"] if i < 2188 else []}} for i in range(2441)]
            paper_path.write_text("".join(json.dumps(r) + "\n" for r in paper))
            subset_path.write_text("".join(json.dumps(r) + "\n" for r in paper[:2188]))
            pages = [["d", 0, 999999.0], ["d", 1, 3.0], ["d", 2, 2.0], ["d", 3, 1.0]]
            predictions = {r["qid"]: {"page_retrieval_results": pages} for r in paper[:2188]}
            input_path.write_text(json.dumps(predictions))
            base_path.write_text(json.dumps({r["qid"]: {"page_retrieval_results": pages} for r in paper}))
            argv = ["validate", str(paper_path), str(input_path), str(subset_path), str(base_path), str(manifest_path)]
            # The launcher runs from the repository root, which supplies code fingerprints.
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(blocks[0], str(launcher), "exec"), {"__name__": "__main__"})
            self.assertEqual(json.loads(manifest_path.read_text())["questions"], 2188)
            predictions["q0"] = {"page_retrieval_results": pages + [["d", 4, 0.0]]}
            input_path.write_text(json.dumps(predictions))
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "four distinct pages"):
                exec(compile(blocks[0], str(launcher), "exec"), {"__name__": "__main__"})
            predictions["q0"] = {"page_retrieval_results": pages}
            input_path.write_text(json.dumps(predictions))
            qa_path, eval_path = root / "qa.json", root / "eval.json"
            qa_rows = {q: {"selected_page_retrieval_results": pages} for q in predictions}
            qa_path.write_text(json.dumps(qa_rows))
            eval_path.write_text(json.dumps({"overall": {"list_em": 0.0, "list_f1": 0.0}}))
            argv = ["validate", str(input_path), str(qa_path), str(eval_path)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(blocks[1], str(launcher), "exec"), {"__name__": "__main__"})
            self.assertIn("INJECTION_READER_RESULT", output.getvalue())
            del qa_rows["q0"]
            qa_path.write_text(json.dumps(qa_rows))
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "all 2188"):
                exec(compile(blocks[1], str(launcher), "exec"), {"__name__": "__main__"})

    def test_base_fill_respects_already_full_budget(self) -> None:
        base = {"page_retrieval_results": [["other", 0, 1.0]]}
        rows = [["gold", i, 100.0 - i] for i in range(5)]
        self.assertEqual(MODULE.append_base_fill(rows=rows[:4], base_row=base, top_pages=4), rows[:4])
        self.assertEqual(MODULE.append_base_fill(rows=rows, base_row=base, top_pages=4), rows[:4])
        self.assertEqual(MODULE.append_base_fill(rows=rows, base_row=base, top_pages=0), [])
        self.assertEqual(len(rows), 5)  # The caller's list is unchanged.

    def test_base_fill_skips_duplicates_and_preserves_order(self) -> None:
        base = {"page_retrieval_results": [["d", i, 10.0 - i] for i in range(5)]}
        rows = [["d", 2, 100.0]]
        filled = MODULE.append_base_fill(rows=rows, base_row=base, top_pages=4)
        self.assertEqual([r[:2] for r in filled], [["d", 2], ["d", 0], ["d", 1], ["d", 3]])

    def test_cli_fill_produces_four_pages_for_all_labeled_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            augmented = root / "gold.jsonl"
            base = root / "base.json"
            output = root / "input.json"
            summary = root / "summary.json"
            rows = [{"qid": f"q{n}", "question": f"Question {n}",
                     "metadata": {"gold_page_uids": [f"d_page{i}" for i in range(n)]}}
                    for n in range(6)]
            augmented.write_text("".join(json.dumps(r) + "\n" for r in rows))
            base.write_text(json.dumps({r["qid"]: {"page_retrieval_results":
                            [["d", i, 10.0 - i] for i in range(10)]} for r in rows}))
            before = (augmented.read_bytes(), base.read_bytes())
            subprocess.run([sys.executable, str(SCRIPT), "--augmented-gold", str(augmented),
                            "--base-prediction", str(base), "--top-pages", "4", "--fill-from-base",
                            "--output-prediction-json", str(output), "--output-summary", str(summary)],
                           check=True, capture_output=True, text=True)
            predictions = json.loads(output.read_text())
            self.assertEqual(set(predictions), {f"q{i}" for i in range(1, 6)})
            for row in predictions.values():
                self.assertEqual(row["page_retrieval_results"], row["selected_page_retrieval_results"])
                self.assertEqual(len(row["page_retrieval_results"]), 4)
            self.assertEqual(json.loads(summary.read_text())["shorter_than_top_pages_qids"], 0)
            self.assertEqual(before, (augmented.read_bytes(), base.read_bytes()))

    def test_visual_proxy_control_replaces_only_proxy_pages(self) -> None:
        rows, missing = MODULE.make_visual_proxy_same_doc_non_gold_rows(
            gold_uids=["d1_page1", "d2_page2"],
            supervision_tiers={"d1_page1": "direct", "d2_page2": "visual_proxy"},
            pages_by_doc={"d1": [0, 1], "d2": [1, 2, 3]},
            top_pages=4,
        )
        self.assertEqual(missing, 0)
        self.assertEqual([(row[0], row[1]) for row in rows], [("d1", 1), ("d2", 1)])

    def test_filters_qids_by_supervision_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            augmented = root / "augmented.jsonl"
            original = root / "original.jsonl"
            output = root / "prediction.json"
            filtered = root / "gold.jsonl"
            summary = root / "summary.json"
            rows = [
                {
                    "qid": "q_direct",
                    "question": "direct",
                    "supporting_context": [{"doc_id": "d1"}],
                    "metadata": {
                        "gold_page_uids": ["d1_page0"],
                        "pseudo_gold_qid_supervision_tier": "complete_direct",
                    },
                },
                {
                    "qid": "q_proxy",
                    "question": "proxy",
                    "supporting_context": [{"doc_id": "d2"}],
                    "metadata": {
                        "gold_page_uids": ["d2_page1"],
                        "pseudo_gold_qid_supervision_tier": "complete_hybrid",
                    },
                },
            ]
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            augmented.write_text(payload, encoding="utf-8")
            original.write_text(payload, encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--augmented-gold",
                    str(augmented),
                    "--original-gold",
                    str(original),
                    "--qid-supervision-tier",
                    "complete_direct",
                    "--output-prediction-json",
                    str(output),
                    "--output-filtered-gold",
                    str(filtered),
                    "--output-summary",
                    str(summary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            prediction = json.loads(output.read_text(encoding="utf-8"))
            stats = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(set(prediction), {"q_direct"})
            self.assertEqual(stats["written_qids"], 1)
            self.assertEqual(stats["skipped_supervision_tier_mismatch"], 1)
            self.assertEqual(stats["qid_supervision_tiers"], ["complete_direct"])


if __name__ == "__main__":
    unittest.main()
