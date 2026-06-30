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

    def test_text_instance_start_byte_context_breaks_page_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "MMQA_dev.jsonl"
            pages = root / "page_text.jsonl"
            texts = root / "MMQA_texts.jsonl"
            out = root / "labels.jsonl"
            summary = root / "summary.json"
            augmented = root / "augmented.jsonl"

            full_text = (
                "Introductory material. target answer alpha river mountain canyon "
                "contains the local evidence context."
            )
            start_byte = len("Introductory material. ".encode("utf-8"))

            self.write_jsonl(
                gold,
                [
                    {
                        "qid": "q_context",
                        "question": "Which page contains the target answer?",
                        "answers": [
                            {
                                "answer": "target answer",
                                "text_instances": [
                                    {
                                        "doc_id": "text_doc",
                                        "text": "target answer",
                                        "start_byte": start_byte,
                                    }
                                ],
                                "table_indices": [],
                                "image_instances": [],
                            }
                        ],
                        "metadata": {"type": "TextQ"},
                        "supporting_context": [{"doc_id": "text_doc", "doc_part": "text"}],
                    }
                ],
            )
            self.write_jsonl(
                pages,
                [
                    {"doc_id": "text_doc", "page_idx": 0, "text": "target answer appears in a generic note."},
                    {
                        "doc_id": "text_doc",
                        "page_idx": 1,
                        "text": "target answer alpha river mountain canyon contains the local evidence context.",
                    },
                ],
            )
            self.write_jsonl(
                texts,
                [
                    {
                        "id": "text_doc",
                        "title": "Text Document",
                        "url": "https://en.wikipedia.org/wiki/Text_Document",
                        "text": full_text,
                    }
                ],
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_mmqa_pseudo_page_labels.py",
                    "--gold",
                    str(gold),
                    "--doc-pages-jsonl",
                    str(pages),
                    "--mmqa-texts-jsonl",
                    str(texts),
                    "--min-score",
                    "8",
                    "--top-pages-per-doc",
                    "1",
                    "--top-pages-per-qid",
                    "4",
                    "--text-instance-context-window-chars",
                    "80",
                    "--text-instance-context-max-phrases",
                    "2",
                    "--text-instance-context-phrase-token-count",
                    "4",
                    "--output-jsonl",
                    str(out),
                    "--output-summary-json",
                    str(summary),
                    "--output-augmented-gold-jsonl",
                    str(augmented),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["pseudo_gold_page_uids"], ["text_doc_page1"])
            sources = row["evidence_source_counts"]
            self.assertGreater(sources.get("text_instance_context", 0), 0)

    def test_zero_weight_start_byte_context_verifies_text_instance_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "MMQA_dev.jsonl"
            pages = root / "page_text.jsonl"
            texts = root / "MMQA_texts.jsonl"
            out = root / "labels.jsonl"
            summary = root / "summary.json"

            full_text = (
                "Header. target answer alpha river mountain canyon contains local evidence."
            )
            start_byte = len("Header. ".encode("utf-8"))

            self.write_jsonl(
                gold,
                [
                    {
                        "qid": "q_context_verify",
                        "question": "Which page contains the target answer?",
                        "answers": [
                            {
                                "answer": "target answer",
                                "text_instances": [
                                    {
                                        "doc_id": "text_doc",
                                        "text": "target answer",
                                        "start_byte": start_byte,
                                    }
                                ],
                                "table_indices": [],
                                "image_instances": [],
                            }
                        ],
                        "metadata": {"type": "TextQ"},
                        "supporting_context": [{"doc_id": "text_doc", "doc_part": "text"}],
                    }
                ],
            )
            self.write_jsonl(
                pages,
                [
                    {"doc_id": "text_doc", "page_idx": 0, "text": "target answer appears in a generic note."},
                    {
                        "doc_id": "text_doc",
                        "page_idx": 1,
                        "text": "target answer alpha river mountain canyon contains local evidence.",
                    },
                ],
            )
            self.write_jsonl(
                texts,
                [
                    {
                        "id": "text_doc",
                        "title": "Text Document",
                        "url": "https://en.wikipedia.org/wiki/Text_Document",
                        "text": full_text,
                    }
                ],
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_mmqa_pseudo_page_labels.py",
                    "--gold",
                    str(gold),
                    "--doc-pages-jsonl",
                    str(pages),
                    "--mmqa-texts-jsonl",
                    str(texts),
                    "--min-score",
                    "8",
                    "--top-pages-per-doc",
                    "1",
                    "--top-pages-per-qid",
                    "4",
                    "--evidence-weight-overrides",
                    (
                        "answer_text=0,text_instance=10,text_instance_context=0,"
                        "image_title=10,image_doc_title=0,table_title=0,"
                        "table_answer_cell=10,table_row_cell=0,table_row_link_text=0,"
                        "table_row_link_title=0,supporting_doc_title=0,answer_entity=0,"
                        "question_entity=0,pseudo_question_slot=0"
                    ),
                    "--text-instance-context-window-chars",
                    "80",
                    "--text-instance-context-verification-bonus",
                    "5",
                    "--output-jsonl",
                    str(out),
                    "--output-summary-json",
                    str(summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["pseudo_gold_page_uids"], ["text_doc_page1"])
            page = row["pseudo_gold_pages"][0]
            self.assertEqual(page["score"], 15.0)
            self.assertEqual(
                [match["source"] for match in page["positive_exact_matches"]],
                ["text_instance"],
            )
            self.assertIn(
                "text_instance_context",
                {match["source"] for match in page["diagnostic_exact_matches"]},
            )
            self.assertTrue(
                all(float(match["weight"]) == 0.0 for match in page["diagnostic_exact_matches"])
            )
            self.assertEqual(
                page["verification_matches"][0]["source"],
                "text_instance_context_verification",
            )
            self.assertEqual(row["positive_evidence_source_counts"], {"text_instance": 1})
            summary_row = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(
                summary_row["selected_verification_source_counts"],
                {"text_instance_context_verification": 1},
            )
            self.assertEqual(summary_row["positive_evidence_source_counts"], {"text_instance": 1})

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

    def test_adaptive_page_caps_follow_direct_evidence_units(self) -> None:
        row = {
            "qid": "q_adaptive",
            "answers": [
                {
                    "answer": "unused",
                    "text_instances": [
                        {"doc_id": "doc_text", "text": "first evidence"},
                        {"doc_id": "doc_text", "text": "second evidence"},
                    ],
                    "table_indices": [[0, 1]],
                    "image_instances": [{"doc_id": "doc_image"}],
                }
            ],
            "metadata": {"table_id": "doc_table"},
        }
        tables = {
            "doc_table": {
                "table": {
                    "table_rows": [
                        [
                            {"text": "row label", "links": []},
                            {"text": "table answer cell", "links": []},
                        ]
                    ]
                }
            }
        }
        images = {"doc_image": {"title": "Reference Image Title"}}

        raw_counts, meta = MODULE.direct_evidence_cap_counts_for_row(
            row,
            tables_by_id=tables,
            images_by_id=images,
        )
        self.assertEqual(raw_counts, {"doc_image": 1, "doc_table": 1, "doc_text": 2})
        self.assertEqual(meta["adaptive_cap_direct_evidence_unit_count"], 4)
        self.assertEqual(
            meta["adaptive_cap_source_counts"],
            {"image_title": 1, "table_answer_cell": 1, "text_instance": 2},
        )

        doc_caps, qid_cap = MODULE.bounded_adaptive_caps(
            raw_counts,
            max_pages_per_doc=10,
            max_pages_per_qid=20,
        )
        self.assertEqual(doc_caps, {"doc_image": 1, "doc_table": 1, "doc_text": 2})
        self.assertEqual(qid_cap, 4)

        bounded_doc_caps, bounded_qid_cap = MODULE.bounded_adaptive_caps(
            raw_counts,
            max_pages_per_doc=1,
            max_pages_per_qid=2,
        )
        self.assertEqual(bounded_doc_caps, {"doc_image": 1, "doc_table": 1, "doc_text": 1})
        self.assertEqual(bounded_qid_cap, 2)

    def test_adaptive_doc_caps_allow_multiple_evidence_pages_in_same_doc(self) -> None:
        first = MODULE.PageScore(page_uid="docA_page0", doc_id="docA", page_idx=0)
        first.score = 1.0
        first.exact_matches = [
            {"source": "text_instance", "text": "alpha evidence", "weight": 1.0},
        ]
        second = MODULE.PageScore(page_uid="docA_page1", doc_id="docA", page_idx=1)
        second.score = 1.0
        second.exact_matches = [
            {"source": "text_instance", "text": "beta evidence", "weight": 1.0},
        ]

        fixed = MODULE.select_labels_by_evidence_coverage(
            [first, second],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=10,
            coverage_min_match_weight=1.0,
        )
        adaptive = MODULE.select_labels_by_evidence_coverage(
            [first, second],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=10,
            coverage_min_match_weight=1.0,
            doc_caps={"docA": 2},
        )

        self.assertEqual(len(fixed), 1)
        self.assertEqual({item.page_uid for item in adaptive}, {"docA_page0", "docA_page1"})

    def test_question_overlap_tie_breaker_prefers_more_relevant_same_score_page(self) -> None:
        earlier = MODULE.PageScore(
            page_uid="docA_page0",
            doc_id="docA",
            page_idx=0,
            question_overlap=0.1,
        )
        earlier.score = 1.0
        earlier.exact_matches = [
            {"source": "text_instance", "text": "shared evidence", "weight": 1.0},
        ]
        later = MODULE.PageScore(
            page_uid="docA_page3",
            doc_id="docA",
            page_idx=3,
            question_overlap=0.8,
        )
        later.score = 1.0
        later.exact_matches = [
            {"source": "text_instance", "text": "shared evidence", "weight": 1.0},
        ]

        default_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            doc_caps={"docA": 1},
        )
        overlap_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="question_overlap",
            doc_caps={"docA": 1},
        )

        self.assertEqual([item.page_uid for item in default_selected], ["docA_page0"])
        self.assertEqual([item.page_uid for item in overlap_selected], ["docA_page3"])

    def test_context_verification_tie_breaker_uses_answer_diagnostic_only_after_direct_tie(self) -> None:
        earlier = MODULE.PageScore(page_uid="docA_page0", doc_id="docA", page_idx=0)
        earlier.score = 1.0
        earlier.exact_matches = [
            {"source": "text_instance", "text": "shared evidence", "weight": 1.0},
        ]
        later = MODULE.PageScore(page_uid="docA_page2", doc_id="docA", page_idx=2)
        later.score = 1.0
        later.exact_matches = [
            {"source": "text_instance", "text": "shared evidence", "weight": 1.0},
            {"source": "answer_text", "text": "answer phrase", "weight": 0.0},
            {"source": "answer_entity", "text": "answer entity", "weight": 0.0},
        ]
        diagnostic_only = MODULE.PageScore(page_uid="docA_page3", doc_id="docA", page_idx=3)
        diagnostic_only.score = 0.0
        diagnostic_only.exact_matches = [
            {"source": "answer_text", "text": "answer phrase", "weight": 0.0},
        ]
        MODULE.apply_context_verification_tie_score(earlier)
        MODULE.apply_context_verification_tie_score(later)
        MODULE.apply_context_verification_tie_score(diagnostic_only)

        default_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later, diagnostic_only],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            doc_caps={"docA": 1},
        )
        context_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later, diagnostic_only],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="context_verification",
            doc_caps={"docA": 1},
        )

        self.assertEqual(earlier.context_verification_score, 0.0)
        self.assertGreater(later.context_verification_score, 0.0)
        self.assertEqual(diagnostic_only.context_verification_score, 0.0)
        self.assertEqual([item.page_uid for item in default_selected], ["docA_page0"])
        self.assertEqual([item.page_uid for item in context_selected], ["docA_page2"])

    def test_context_verification_tie_breaker_uses_table_and_image_context(self) -> None:
        table_plain = MODULE.PageScore(page_uid="docT_page0", doc_id="docT", page_idx=0)
        table_plain.score = 1.0
        table_plain.exact_matches = [
            {"source": "table_answer_cell", "text": "shared answer", "weight": 1.0},
        ]
        table_context = MODULE.PageScore(page_uid="docT_page4", doc_id="docT", page_idx=4)
        table_context.score = 1.0
        table_context.exact_matches = [
            {"source": "table_answer_cell", "text": "shared answer", "weight": 1.0},
            {"source": "table_title", "text": "relevant table", "weight": 0.0},
            {"source": "table_row_header", "text": "row label", "weight": 0.0},
            {"source": "table_row_cell", "text": "neighbor value", "weight": 0.0},
        ]
        image_plain = MODULE.PageScore(page_uid="docI_page0", doc_id="docI", page_idx=0)
        image_plain.score = 1.0
        image_plain.exact_matches = [
            {"source": "image_title", "text": "shared caption", "weight": 1.0},
        ]
        image_context = MODULE.PageScore(page_uid="docI_page5", doc_id="docI", page_idx=5)
        image_context.score = 1.0
        image_context.exact_matches = [
            {"source": "image_title", "text": "shared caption", "weight": 1.0},
            {"source": "image_doc_title", "text": "image article title", "weight": 0.0},
        ]
        for item in [table_plain, table_context, image_plain, image_context]:
            MODULE.apply_context_verification_tie_score(item)

        table_selected = MODULE.select_labels_by_evidence_coverage(
            [table_plain, table_context],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="context_verification",
            doc_caps={"docT": 1},
        )
        image_selected = MODULE.select_labels_by_evidence_coverage(
            [image_plain, image_context],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="context_verification",
            doc_caps={"docI": 1},
        )

        self.assertGreater(table_context.context_verification_score, 0.0)
        self.assertGreater(image_context.context_verification_score, 0.0)
        self.assertEqual([item.page_uid for item in table_selected], ["docT_page4"])
        self.assertEqual([item.page_uid for item in image_selected], ["docI_page5"])

    def test_indirect_verification_tie_breaker_uses_table_context_diagnostics(self) -> None:
        earlier = MODULE.PageScore(page_uid="docA_page0", doc_id="docA", page_idx=0)
        earlier.score = 1.0
        earlier.exact_matches = [
            {"source": "table_answer_cell", "text": "shared answer", "weight": 1.0},
        ]
        later = MODULE.PageScore(page_uid="docA_page2", doc_id="docA", page_idx=2)
        later.score = 1.0
        later.exact_matches = [
            {"source": "table_answer_cell", "text": "shared answer", "weight": 1.0},
            {"source": "table_title", "text": "relevant table", "weight": 0.0},
            {"source": "table_row_cell", "text": "row context", "weight": 0.0},
        ]
        MODULE.apply_indirect_verification_tie_score(earlier)
        MODULE.apply_indirect_verification_tie_score(later)

        default_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            doc_caps={"docA": 1},
        )
        indirect_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="indirect_verification",
            doc_caps={"docA": 1},
        )

        self.assertEqual(earlier.indirect_verification_score, 0.0)
        self.assertGreater(later.indirect_verification_score, 0.0)
        self.assertEqual([item.page_uid for item in default_selected], ["docA_page0"])
        self.assertEqual([item.page_uid for item in indirect_selected], ["docA_page2"])

    def test_table_context_tie_breaker_uses_table_headers_without_answer_duplicate(self) -> None:
        earlier = MODULE.PageScore(page_uid="docA_page0", doc_id="docA", page_idx=0)
        earlier.score = 1.0
        earlier.exact_matches = [
            {"source": "table_answer_cell", "text": "shared answer", "weight": 1.0},
            {"source": "table_row_cell", "text": "shared answer", "weight": 0.0},
        ]
        later = MODULE.PageScore(page_uid="docA_page2", doc_id="docA", page_idx=2)
        later.score = 1.0
        later.exact_matches = [
            {"source": "table_answer_cell", "text": "shared answer", "weight": 1.0},
            {"source": "table_row_cell", "text": "shared answer", "weight": 0.0},
            {"source": "table_title", "text": "relevant table", "weight": 0.0},
            {"source": "table_row_header", "text": "row label", "weight": 0.0},
            {"source": "table_column_header", "text": "column label", "weight": 0.0},
            {"source": "table_row_cell", "text": "neighbor value", "weight": 0.0},
        ]
        MODULE.apply_table_context_tie_score(earlier)
        MODULE.apply_table_context_tie_score(later)

        default_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            doc_caps={"docA": 1},
        )
        table_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="table_context",
            doc_caps={"docA": 1},
        )

        self.assertEqual(earlier.table_context_score, 0.0)
        self.assertGreater(later.table_context_score, 0.0)
        self.assertEqual([item.page_uid for item in default_selected], ["docA_page0"])
        self.assertEqual([item.page_uid for item in table_selected], ["docA_page2"])

    def test_image_presence_tie_breaker_only_uses_visual_metadata_for_image_evidence(self) -> None:
        earlier = MODULE.PageScore(
            page_uid="docA_page0",
            doc_id="docA",
            page_idx=0,
            visual_image_tie_score=0.0,
            visual_has_image=False,
        )
        earlier.score = 1.0
        earlier.exact_matches = [
            {"source": "image_title", "text": "shared caption", "weight": 1.0},
        ]
        later = MODULE.PageScore(
            page_uid="docA_page3",
            doc_id="docA",
            page_idx=3,
            visual_image_tie_score=11.25,
            visual_image_count=2,
            visual_large_image_count=1,
            visual_image_area_ratio=0.15,
            visual_has_image=True,
            visual_has_large_image=True,
        )
        later.score = 1.0
        later.exact_matches = [
            {"source": "image_title", "text": "shared caption", "weight": 1.0},
        ]
        text_page = MODULE.PageScore(
            page_uid="docB_page3",
            doc_id="docB",
            page_idx=3,
            visual_image_tie_score=99.0,
            visual_has_image=True,
        )
        text_page.score = 1.0
        text_page.exact_matches = [
            {"source": "text_instance", "text": "shared text", "weight": 1.0},
        ]
        text_earlier = MODULE.PageScore(page_uid="docB_page0", doc_id="docB", page_idx=0)
        text_earlier.score = 1.0
        text_earlier.exact_matches = [
            {"source": "text_instance", "text": "shared text", "weight": 1.0},
        ]

        default_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            doc_caps={"docA": 1},
        )
        image_selected = MODULE.select_labels_by_evidence_coverage(
            [earlier, later],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="image_presence",
            doc_caps={"docA": 1},
        )
        text_selected = MODULE.select_labels_by_evidence_coverage(
            [text_earlier, text_page],
            min_score=1.0,
            top_pages_per_doc=1,
            top_pages_per_qid=1,
            coverage_min_match_weight=1.0,
            evidence_coverage_tie_breaker="image_presence",
            doc_caps={"docB": 1},
        )

        self.assertEqual([item.page_uid for item in default_selected], ["docA_page0"])
        self.assertEqual([item.page_uid for item in image_selected], ["docA_page3"])
        self.assertEqual([item.page_uid for item in text_selected], ["docB_page0"])


if __name__ == "__main__":
    unittest.main()
