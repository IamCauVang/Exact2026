from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_questions(record: dict[str, Any]) -> list[Any]:
    for key in ["questions", "Questions", "question", "Question"]:
        if key in record:
            value = record[key]
            return value if isinstance(value, list) else [value]
    return []


def get_premises(record: dict[str, Any]) -> list[str]:
    for key in ["premises-NL", "premises_nl", "premises", "Premises", "context"]:
        if key in record:
            value = record[key]
            if isinstance(value, list):
                return [str(x) for x in value]
            return [str(value)]
    return []


def question_to_text(q: Any) -> str:
    if isinstance(q, str):
        return q
    if isinstance(q, dict):
        for key in ["question", "Question", "text", "prompt"]:
            if key in q:
                return str(q[key])
        return json.dumps(q, ensure_ascii=False)
    return str(q)


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}m:{s:05.2f}s"


def is_correct(pred: Any, gold: Any) -> bool:
    if gold is None:
        return False
    return str(pred).strip().lower() == str(gold).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--answers", default="artifacts/answers.json")
    ap.add_argument("--out", default="artifacts/qa_report.md")
    ap.add_argument("--show-premises", action="store_true")
    args = ap.parse_args()

    dataset = load_json(args.dataset)
    answers = load_json(args.answers)

    lines: list[str] = []
    correct = 0
    with_gold = 0
    warning_cases = 0

    for item in answers:
        gold = item.get("gold_answer")
        if gold is not None:
            with_gold += 1
            correct += int(is_correct(item.get("answer"), gold))
        if item.get("warnings"):
            warning_cases += 1

    lines.append("# QA Report")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Answers: `{args.answers}`")
    lines.append(f"- Total answer cases: `{len(answers)}`")
    lines.append(f"- Correct: `{correct}/{with_gold}`")
    lines.append(f"- Warning cases: `{warning_cases}`")
    lines.append("")

    for item in answers:
        case_id = str(item.get("id", "?"))
        record_idx = question_idx = None
        try:
            record_idx_s, question_idx_s = case_id.split(":")
            record_idx = int(record_idx_s)
            question_idx = int(question_idx_s)
        except Exception:
            pass

        record = dataset[record_idx] if record_idx is not None and record_idx < len(dataset) else {}
        questions = get_questions(record)
        q = questions[question_idx] if question_idx is not None and question_idx < len(questions) else ""
        premises = get_premises(record)

        answer = item.get("answer")
        gold = item.get("gold_answer")
        status = "N/A" if gold is None else ("✅ CORRECT" if is_correct(answer, gold) else "❌ WRONG")
        warnings = item.get("warnings") or []
        used = item.get("used_premises") or []
        raw = item.get("raw") or {}
        parsed = item.get("parsed_question") or {}

        lines.append(f"## Case {case_id} — {status}")
        lines.append("")
        lines.append(f"**Type:** `{item.get('question_type')}`")
        lines.append("")
        lines.append(f"**Question:** {question_to_text(q)}")
        lines.append("")
        lines.append(f"**Answer:** `{answer}`")
        lines.append(f"**Gold:** `{gold}`")
        lines.append(f"**Runtime:** `{format_duration(item.get('runtime_seconds'))}`")
        lines.append("")
        lines.append(f"**Warnings:** `{'; '.join(map(str, warnings)) if warnings else 'none'}`")
        lines.append("")
        lines.append(f"**Used premises:** `{used}`")
        lines.append("")

        if args.show_premises and premises:
            lines.append("**Premises:**")
            lines.append("")
            for i, p in enumerate(premises, start=1):
                mark = "✅" if i in used else "  "
                lines.append(f"{mark} P{i}. {p}")
            lines.append("")

        lines.append("**Parsed question:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

        if raw:
            lines.append("**Raw option results:**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(raw, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

        lines.append(f"**Explanation:** {item.get('explanation', '')}")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
