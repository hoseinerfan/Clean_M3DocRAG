#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

mkdir -p "$QIDS_OUT_DIR"

echo "using_gold=$GOLD"
echo "using_qids_jsonl=$QIDS_JSONL"
echo "using_question_type_filter=${QUESTION_TYPE_FILTER:-ALL}"

"$PYTHON_BIN" - "$GOLD" "$QIDS_JSONL" "$QUESTION_TYPE_FILTER" <<'PY'
import json
import sys
from pathlib import Path

gold_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
question_type = str(sys.argv[3]).strip()

count = 0
with gold_path.open("r", encoding="utf-8") as handle, out_path.open("w", encoding="utf-8") as out:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if question_type:
            row_type = str(row.get("metadata", {}).get("type", "")).strip()
            if row_type != question_type:
                continue
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        out.write(
            json.dumps(
                {
                    "qid": qid,
                    "question": row.get("question", ""),
                    "question_type": row.get("metadata", {}).get("type", "UNKNOWN"),
                },
                ensure_ascii=True,
            )
            + "\n"
        )
        count += 1

print(f"saved_qids: {out_path}")
print(f"qid_count: {count}")
PY
