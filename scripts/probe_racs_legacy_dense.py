#!/usr/bin/env python3
"""Print stored legacy-dense configuration; no model runs or file writes.

The earlier inventory inspected upstream summaries, not prediction metadata.
Use the exact legacy dense path from that inventory, never wrapper defaults.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_racs_runtime_prerequisites import prediction_rows, qid_digest, read_json


CORE_KEYS = ("base_score_source", "approx_base_page_token_topk",
             "approx_base_page_token_scorer", "approx_base_page_token_selector")
CONFIG_KEYS = {"baseline_pred", "embedding_name", "data_name", "split",
               "input_qid_jsonl", "qid_jsonl", "from_baseline_top_pages"}
CONFIG_PREFIXES = ("approx_base_", "base_score_", "base_only_", "weight_",
                   "two_stage_", "report_")


def configuration(row):
    return {k: v for k, v in row.items()
            if (k in CONFIG_KEYS or k.startswith(CONFIG_PREFIXES))
            and isinstance(v, (str, int, float, bool, type(None)))}


def inspect_prediction(path, expected_digest):
    rows = prediction_rows(read_json(path))
    if not rows:
        raise ValueError("Empty legacy prediction")
    if qid_digest(rows) != expected_digest:
        raise ValueError("Legacy prediction question IDs differ from audit gold")
    configs = Counter()
    missing = Counter()
    metadata_keys = set()
    for row in rows.values():
        metadata = row.get("reranker_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Non-object reranker_metadata")
        metadata_keys.update(metadata)
        saved = configuration(metadata)
        configs[json.dumps(saved, sort_keys=True, allow_nan=False)] += 1
        missing.update(k for k in CORE_KEYS if k not in saved or saved[k] is None)
    return {"prediction_path": str(path), "questions": len(rows),
            "qids_match_audit_gold": True,
            "first_row_keys": sorted(next(iter(rows.values()))),
            "metadata_keys": sorted(metadata_keys),
            "configuration_groups": [{"configuration": json.loads(k), "questions": n}
                                     for k, n in sorted(configs.items())],
            "missing_core_field_counts": dict(missing)}


def probe(audit_path):
    audit = read_json(audit_path)
    legacy_path = Path(audit["graphs"]["no_hyperlink"]
                       ["compact_configuration"]["dense_prediction_json"])
    if not legacy_path.is_absolute() or not legacy_path.name.endswith(".prediction.json"):
        raise ValueError("Expected an absolute recorded prediction path")
    result = inspect_prediction(legacy_path, audit["gold"]["qid_sha256"])
    # A first-row JSONL may preserve richer configuration than the prediction.
    # It is only supporting evidence, not a verified same-run companion.
    companion = legacy_path.with_name(legacy_path.name[:-len(".prediction.json")] + ".jsonl")
    detail = {"path": str(companion), "exists": companion.is_file(),
              "scope": "first nonblank row only; same-run identity not verified"}
    if companion.is_file():
        with companion.open() as handle:
            first = next((json.loads(line) for line in handle if line.strip()), {})
        detail["first_row_configuration"] = configuration(first)
    result["possible_detail_jsonl"] = detail
    result["status"] = "metadata_probe_only_not_replay_or_timing"
    result["not_established"] = ["equivalence to current wrapper defaults",
                                 "full original invocation or environment",
                                 "replayed ranking equivalence", "end-to-end runtime"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.audit_json), indent=2, allow_nan=False))
