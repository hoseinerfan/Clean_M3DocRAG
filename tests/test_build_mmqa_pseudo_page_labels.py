import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_mmqa_pseudo_page_labels.py"
SPEC = importlib.util.spec_from_file_location("build_mmqa_pseudo_page_labels", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildMMQAPseudoPageLabelsTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_text_table_and_image_evidence_localize_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "MMQA_dev.jsonl"
            pages = root / "page_text.jsonl"
            tables = root / "MMQA_tables.jsonl"
            texts = root / "MMQA_texts.jsonl"
            images = root / "MMQA_images.jsonl"
            mapping = root / "id_url_mapping.jsonl"
            out = root / "labels.jsonl"
            summary = root / "summary.json"
            augmented = root / "augmented.jsonl"

            self.write_jsonl(
                gold,
                [
                    {
                        "qid": "q_text",
                        "question": "When was Hillaryland?",
                        "answers": [
                            {
                                "answer": "2008 U.S. election",
                                "text_instances": [
                                    {
                                        "doc_id": "text_doc",
                                        "text": "2008 U.S. election",
                                        "start_byte": 10,
                                    }
                                ],
                                "table_indices": [],
                                "image_instances": [],
                            }
                        ],
                        "metadata": {"type": "TextQ", "modalities": ["text"]},
                        "supporting_context": [{"doc_id": "text_doc", "doc_part": "text"}],
                    },
                    {
                        "qid": "q_table",
                        "question": "For which film did Ben Piazza play Mr. Simms?",
                        "answers": [
                            {
                                "answer": "Mask",
                                "text_instances": [],
                                "table_indices": [[0, 1]],
                                "image_instances": [],
                            }
                        ],
                        "metadata": {
                            "type": "TableQ",
                            "modalities": ["table"],
                            "table_id": "table_doc",
                            "pseudo_language_question": "In [Filmography] of [Ben Piazza], title for [Mr. Simms]",
                        },
                        "supporting_context": [{"doc_id": "table_doc", "doc_part": "table"}],
                    },
                    {
                        "qid": "q_image",
                        "question": "What sport is shown?",
                        "answers": [
                            {
                                "answer": "baseball",
                                "text_instances": [],
                                "table_indices": [],
                                "image_instances": [{"doc_id": "image_doc", "doc_part": "image"}],
                            }
                        ],
                        "metadata": {"type": "ImageQ", "modalities": ["image"]},
                        "supporting_context": [{"doc_id": "image_doc", "doc_part": "image"}],
                    },
                ],
            )
            self.write_jsonl(
                pages,
                [
                    {"doc_id": "text_doc", "page_idx": 0, "text": "Unrelated Clinton advisors."},
                    {
                        "doc_id": "text_doc",
                        "page_idx": 1,
                        "text": "Hillaryland was active in the 2008 U.S. election.",
                    },
                    {"doc_id": "table_doc", "page_idx": 0, "text": "Ben Piazza filmography role Mr. Simms Mask."},
                    {"doc_id": "image_doc", "page_idx": 2, "text": "The Bad News Bears is a baseball film."},
                ],
            )
            self.write_jsonl(
                tables,
                [
                    {
                        "id": "table_doc",
                        "title": "Ben Piazza",
                        "url": "https://en.wikipedia.org/wiki/Ben_Piazza",
                        "table": {
                            "table_rows": [
                                [
                                    {"text": "Mr. Simms", "links": []},
                                    {
                                        "text": "Mask",
                                        "links": [
                                            {
                                                "text": "Mask",
                                                "wiki_title": "Mask (1985 film)",
                                                "url": "https://en.wikipedia.org/wiki/Mask_(1985_film)",
                                            }
                                        ],
                                    },
                                ]
                            ]
                        },
                    }
                ],
            )
            self.write_jsonl(
                texts,
                [
                    {
                        "id": "text_doc",
                        "title": "Hillaryland",
                        "url": "https://en.wikipedia.org/wiki/Hillaryland",
                        "text": "Hillaryland was a group.",
                    }
                ],
            )
            self.write_jsonl(
                images,
                [
                    {
                        "id": "image_doc",
                        "title": "The Bad News Bears",
                        "url": "https://en.wikipedia.org/wiki/The_Bad_News_Bears",
                        "path": "image_doc.jpg",
                    }
                ],
            )
            self.write_jsonl(
                mapping,
                [
                    {"id": "text_doc", "url": "https://en.wikipedia.org/wiki/Hillaryland"},
                    {"id": "table_doc", "url": "https://en.wikipedia.org/wiki/Ben_Piazza"},
                    {"id": "image_doc", "url": "https://en.wikipedia.org/wiki/The_Bad_News_Bears"},
                ],
            )

            MODULE.main_with_args = None
            argv = [
                "build_mmqa_pseudo_page_labels.py",
                "--gold",
                str(gold),
                "--doc-pages-jsonl",
                str(pages),
                "--mmqa-texts-jsonl",
                str(texts),
                "--mmqa-tables-jsonl",
                str(tables),
                "--mmqa-images-jsonl",
                str(images),
                "--id-url-mapping-jsonl",
                str(mapping),
                "--output-jsonl",
                str(out),
                "--output-summary-json",
                str(summary),
                "--output-augmented-gold-jsonl",
                str(augmented),
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                MODULE.main()
            finally:
                sys.argv = old_argv

            labels = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            by_qid = {row["qid"]: row for row in labels}
            self.assertEqual(by_qid["q_text"]["pseudo_gold_page_uids"], ["text_doc_page1"])
            self.assertEqual(by_qid["q_table"]["pseudo_gold_page_uids"], ["table_doc_page0"])
            self.assertEqual(by_qid["q_image"]["pseudo_gold_page_uids"], ["image_doc_page2"])

            summary_row = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(summary_row["matched_qid_count"], 3)
            augmented_rows = [
                json.loads(line) for line in augmented.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                augmented_rows[0]["metadata"]["gold_page_uids"],
                ["text_doc_page1"],
            )

    def test_evidence_coverage_policy_selects_pages_with_new_doc_evidence(self) -> None:
        first = MODULE.PageScore(page_uid="docA_page0", doc_id="docA", page_idx=0)
        first.score = 12.0
        first.exact_matches = [
            {"source": "answer_text", "text": "alpha", "weight": 5.0},
            {"source": "question_entity", "text": "weak shared", "weight": 1.0},
        ]
        second = MODULE.PageScore(page_uid="docA_page1", doc_id="docA", page_idx=1)
        second.score = 10.0
        second.exact_matches = [
            {"source": "text_instance", "text": "beta", "weight": 9.0},
        ]
        duplicate = MODULE.PageScore(page_uid="docA_page2", doc_id="docA", page_idx=2)
        duplicate.score = 9.0
        duplicate.exact_matches = [
            {"source": "answer_text", "text": "alpha", "weight": 5.0},
        ]

        selected = MODULE.select_labels_by_evidence_coverage(
            [first, second, duplicate],
            min_score=8.0,
            top_pages_per_doc=3,
            top_pages_per_qid=4,
            coverage_min_match_weight=3.0,
        )

        self.assertEqual(
            {item.page_uid for item in selected},
            {"docA_page0", "docA_page1"},
        )
        self.assertNotIn("docA_page2", {item.page_uid for item in selected})

    def test_evidence_weight_overrides_update_selected_sources_only(self) -> None:
        weights = MODULE.parse_evidence_weight_overrides(
            "question_entity=0,pseudo_question_slot=0,supporting_doc_title=2.5"
        )

        self.assertEqual(weights["question_entity"], 0.0)
        self.assertEqual(weights["pseudo_question_slot"], 0.0)
        self.assertEqual(weights["supporting_doc_title"], 2.5)
        self.assertEqual(
            weights["text_instance"],
            MODULE.DEFAULT_EVIDENCE_WEIGHTS["text_instance"],
        )

    def test_evidence_weight_overrides_reject_unknown_source(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_evidence_weight_overrides("unknown_signal=1")


if __name__ == "__main__":
    unittest.main()
