from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact_xai.io_utils import load_json, save_json
from exact_xai.schemas import AnswerRequest


def main():
    data = load_json("data/fraction_dataset.json")
    requests = []
    for rid, rec in enumerate(data):
        for qid, q in enumerate(rec.get("questions", [])):
            requests.append(AnswerRequest(
                id=f"{rid}:{qid}",
                premises_nl=rec.get("premises-NL", []),
                premises_fol=rec.get("premises-FOL", []),
                question=q,
                gold_answer=(rec.get("answers") or [None])[qid],
            ).model_dump())
    save_json({"records": requests}, "artifacts/input_requests.json")
    print("Wrote artifacts/input_requests.json")

if __name__ == "__main__":
    main()
