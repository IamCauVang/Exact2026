from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from exact_xai.io_utils import load_json, save_json
from exact_xai.pipeline import AnswerPipeline
from exact_xai.dataset_utils import records_from_exact_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/fraction_dataset.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--input-mode", choices=["auto", "fol", "nl"], default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load_json(args.dataset)
    records, loader_warnings = records_from_exact_dataset(data, input_mode=args.input_mode)
    if args.limit:
        records = records[:args.limit]

    pipe = AnswerPipeline(llm=None, input_mode=args.input_mode)
    rows = []
    correct = total = 0
    for req in records:
        ans = pipe.answer(req)
        row = ans.model_dump()
        row["gold_answer"] = req.gold_answer
        row["question_type"] = req.question_type
        rows.append(row)
        if req.gold_answer is not None:
            total += 1
            correct += int(str(ans.answer).strip().lower() == str(req.gold_answer).strip().lower())
        print(f"{req.id}: pred={ans.answer} gold={req.gold_answer} mode={ans.mode} warnings={ans.warnings}")

    report = {
        "num_records": len(rows),
        "num_with_gold": total,
        "exact_match": correct / total if total else None,
        "loader_warnings": loader_warnings,
    }
    print(report)
    if args.out:
        out = Path(args.out)
        save_json(rows, out / "answers.json")
        save_json(report, out / "eval_report.json")


if __name__ == "__main__":
    main()
