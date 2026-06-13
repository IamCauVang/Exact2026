#!/usr/bin/env python3
"""Build SFT data for an EXACT NL -> FOL/IR compiler.

The model is trained as a compiler, not an answer predictor.
It receives natural-language premises/questions/options and returns JSON-only logic IR.

Outputs:
  train.jsonl / valid.jsonl / test.jsonl
  split_report.json
  noisy_cases.json

Sample families:
  premise_translation:  single NL premise -> provided FOL premise
  record_premise_compiler: full NL premise list -> provided FOL premise list
  question_intent:      question text -> type/intent/options text, no answer solving
  silver_option_compiler: MCQ options -> conservative silver FOL when regex parser can parse
  full_compiler:        full record -> provided premises FOL + question IR + silver options when available
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a strict NL-to-FOL compiler for an explainable AI system. "
    "Return JSON only. Do not solve the question. Do not choose A/B/C/D. "
    "Translate natural-language rules, questions, and options into formal logic IR."
)

MCQ_RE = re.compile(r"(?ms)(?:^|\n)\s*([A-E])\s*[\.)]\s*(.*?)(?=(?:\n\s*[A-E]\s*[\.)]\s*)|\Z)")
DIRECT_MCQ_RE = re.compile(r"(?m)^\s*A\s*[\.)]\s+.+\n\s*B\s*[\.)]\s+", re.I)

YESNO_STARTERS = (
    "does ", "do ", "is ", "are ", "can ", "could ", "should ",
    "would ", "will ", "did ", "has ", "have ", "was ", "were ",
)
OPEN_STARTERS = (
    "what ", "how ", "why ", "explain ", "describe ", "calculate ",
    "determine ", "identify ", "list ", "when ", "where ", "who ",
)


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["records", "data", "items", "examples", "train"]:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Dataset top-level must be a list or a dict containing records/data/items/examples.")


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def get_field(record: dict[str, Any], keys: list[str]) -> list[Any]:
    for k in keys:
        if k in record:
            return as_list(record[k])
    return []


def get_questions(record: dict[str, Any]) -> list[Any]:
    return get_field(record, ["questions", "Questions", "question", "Question"])


def get_answers(record: dict[str, Any]) -> list[Any]:
    return get_field(record, ["answers", "Answers", "answer", "Answer", "gold_answers", "gold"])


def get_explanations(record: dict[str, Any]) -> list[Any]:
    return get_field(record, ["explanation", "explanations", "Explanation", "Explanations"])


def get_premises_nl(record: dict[str, Any]) -> list[str]:
    return [str(x) for x in get_field(record, ["premises-NL", "premises_nl", "premises", "Premises", "context"])]


def get_premises_fol(record: dict[str, Any]) -> list[str]:
    return [str(x) for x in get_field(record, ["premises-FOL", "premises_fol", "fol", "FOL"])]


def q_text(q: Any) -> str:
    if isinstance(q, str):
        return q
    if isinstance(q, dict):
        for k in ["question", "Question", "text", "prompt"]:
            if k in q:
                return str(q[k])
    return str(q)


def detect_question_type(question: Any) -> str:
    text = q_text(question).strip()
    lower = text.lower()
    if isinstance(question, dict):
        for key in ["type", "question_type", "kind"]:
            if key in question:
                val = str(question[key]).lower()
                if "multiple" in val or "mcq" in val or "choice" in val:
                    return "multiple_choice"
                if "yes" in val or "no" in val or "uncertain" in val:
                    return "yes_no"
                if "open" in val:
                    return "open"
        for key in ["choices", "options", "Choices", "Options"]:
            if question.get(key):
                return "multiple_choice"
    if DIRECT_MCQ_RE.search(text):
        return "multiple_choice"
    if lower.startswith(OPEN_STARTERS):
        # "Which conclusion..." with A/B/C/D was already caught as MCQ.
        return "open"
    if lower.startswith(YESNO_STARTERS):
        return "yes_no"
    return "open"


def detect_intent(question: str) -> str:
    lower = question.lower()
    if "fewest premise" in lower or "least premise" in lower or "minimum premise" in lower:
        return "fewest_premises"
    if "strongest conclusion" in lower or "strongest" in lower:
        return "strongest_conclusion"
    if "which conclusion" in lower or "which statement" in lower or "which of the following" in lower:
        return "select_supported_option"
    if lower.startswith(YESNO_STARTERS) or "does it follow" in lower:
        return "entailment_yes_no"
    return "open_answer"


def split_mcq_options(question: str) -> dict[str, str]:
    matches = MCQ_RE.findall(question)
    return {label.upper(): text.strip() for label, text in matches}


def normalize_predicate_phrase(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the|all|every|any|python|project|projects|code|student|students|model|models)\b", " ", s)
    s = re.sub(r"\b(is|are|be|being|been|must|can|does|do|has|have|with|from|based|on|to|it|they|them|that|which)\b", " ", s)
    s = s.replace("pep 8", "pep8")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_predicate"


def parse_simple_clause(text: str) -> str | None:
    """Conservative silver parser for common option/premise clauses.

    This is intentionally small. It only creates silver labels when the shape is clear.
    The labels are not gold, so downstream reports mark them as silver.
    """
    raw = text.strip().rstrip(".")
    lower = raw.lower()

    # If X then Y, including comma after condition.
    m = re.match(r"if\s+(.+?)\s*,?\s+then\s+(.+)$", raw, re.I)
    if m:
        left, right = m.group(1), m.group(2)
        return f"ForAll(x, {parse_predicate_expr(left)} -> {parse_predicate_expr(right)})"

    # If all X are Y, then all X are Z.
    m = re.match(r"if\s+all\s+(.+?)\s+are\s+(.+?)\s*,?\s+then\s+all\s+.+?\s+are\s+(.+)$", raw, re.I)
    if m:
        a, b = m.group(2), m.group(3)
        return f"ForAll(x, {parse_predicate_expr(a)} -> {parse_predicate_expr(b)})"

    # All X are Y / Every X is Y.
    m = re.match(r"(?:all|every)\s+(.+?)\s+(?:is|are)\s+(.+)$", raw, re.I)
    if m:
        return f"ForAll(x, {parse_predicate_expr(m.group(2))})"

    # There exists at least one X that/is Y.
    m = re.match(r"there\s+exists\s+at\s+least\s+one\s+(.+?)\s+(?:that|which|who|is|are)\s+(.+)$", raw, re.I)
    if m:
        return f"Exists(x, {parse_predicate_expr(m.group(2))})"

    return None


def parse_predicate_expr(phrase: str) -> str:
    s = phrase.strip().rstrip(".")
    neg = False
    # Negative forms.
    if re.search(r"\b(not|does not|do not|is not|are not|cannot|can't|doesn't|don't)\b", s, re.I):
        neg = True
        s = re.sub(r"\b(does not|do not|is not|are not|cannot|can't|doesn't|don't|not)\b", " ", s, flags=re.I)
    pred = normalize_predicate_phrase(s)
    atom = f"{pred}(x)"
    return f"not {atom}" if neg else atom


def make_messages(user_content: str, target_obj: Any) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(target_obj, ensure_ascii=False, separators=(",", ":"))},
        ]
    }


def record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_samples_for_record(record: dict[str, Any], rid: int, split: str, include_silver_options: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    samples: list[dict[str, Any]] = []
    noisy: list[dict[str, Any]] = []
    stats = Counter()
    premises_nl = get_premises_nl(record)
    premises_fol = get_premises_fol(record)
    questions = [q_text(q) for q in get_questions(record)]
    answers = [str(a) for a in get_answers(record)]
    explanations = [str(e) for e in get_explanations(record)]
    rh = record_hash(record)

    # A. single premise translation, gold.
    for i, (nl, fol) in enumerate(zip(premises_nl, premises_fol), start=1):
        user = f"Translate this natural-language premise into FOL. Return JSON only.\n\nPremise {i}: {nl}"
        target = {"task": "premise_translation", "premise_id": i, "fol": fol}
        row = make_messages(user, target)
        row.update({"meta": {"record_id": rid, "record_hash": rh, "split": split, "sample_type": "premise_translation", "quality": "gold"}})
        samples.append(row)
        stats["premise_translation"] += 1

    # B. record premise compiler, gold.
    if premises_nl and premises_fol:
        user_lines = ["Compile all natural-language premises into FOL. Return JSON only.", "", "Premises-NL:"]
        for i, nl in enumerate(premises_nl, start=1):
            user_lines.append(f"{i}. {nl}")
        target = {
            "task": "record_premise_compiler",
            "premises_fol": [{"id": i, "fol": fol} for i, fol in enumerate(premises_fol, start=1)],
        }
        row = make_messages("\n".join(user_lines), target)
        row.update({"meta": {"record_id": rid, "record_hash": rh, "split": split, "sample_type": "record_premise_compiler", "quality": "gold"}})
        samples.append(row)
        stats["record_premise_compiler"] += 1

    # C/D. question/intents and conservative silver option compiler.
    for qid, q in enumerate(questions):
        qtype = detect_question_type(q)
        intent = detect_intent(q)
        options_text = split_mcq_options(q) if qtype == "multiple_choice" else {}
        gold = answers[qid] if qid < len(answers) else None
        exp = explanations[qid] if qid < len(explanations) else None

        # question intent sample. Does not train final answer.
        user = "Identify question type and reasoning intent. Do not answer the question. Return JSON only.\n\nQuestion:\n" + q
        target = {
            "task": "question_intent",
            "question_type": qtype,
            "intent": intent,
            "options_text": options_text,
        }
        row = make_messages(user, target)
        row.update({"meta": {"record_id": rid, "question_id": qid, "record_hash": rh, "split": split, "sample_type": "question_intent", "quality": "derived"}})
        samples.append(row)
        stats["question_intent"] += 1

        options_fol: dict[str, str] = {}
        unparsed: dict[str, str] = {}
        if qtype == "multiple_choice" and options_text:
            for label, text in options_text.items():
                fol = parse_simple_clause(text)
                if fol:
                    options_fol[label] = fol
                else:
                    unparsed[label] = text
            if options_fol and include_silver_options:
                user = "Translate these MCQ options into FOL. Do not choose an answer. Return JSON only.\n\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in options_text.items())
                target = {"task": "silver_option_compiler", "question_type": qtype, "intent": intent, "options_fol": options_fol}
                row = make_messages(user, target)
                row.update({"meta": {"record_id": rid, "question_id": qid, "record_hash": rh, "split": split, "sample_type": "silver_option_compiler", "quality": "silver", "unparsed_options": sorted(unparsed)}})
                samples.append(row)
                stats["silver_option_compiler"] += 1
            if unparsed:
                noisy.append({"record_id": rid, "question_id": qid, "type": "unparsed_options", "options": unparsed})
                stats["unparsed_options"] += len(unparsed)

        # Full compiler sample: gold premises + derived question IR + optional silver options.
        user_lines = [SYSTEM_PROMPT, "", "Compile the record into canonical logic JSON. Do not answer the question.", "", "Premises-NL:"]
        for i, nl in enumerate(premises_nl, start=1):
            user_lines.append(f"{i}. {nl}")
        user_lines += ["", "Question:", q]
        target = {
            "task": "full_compiler",
            "premises_fol": [{"id": i, "fol": fol} for i, fol in enumerate(premises_fol, start=1)],
            "question": {
                "type": qtype,
                "intent": intent,
                "options_text": options_text,
                "options_fol": options_fol if include_silver_options else {},
            },
        }
        row = make_messages("\n".join(user_lines), target)
        row.update({"meta": {"record_id": rid, "question_id": qid, "record_hash": rh, "split": split, "sample_type": "full_compiler", "quality": "gold_premises_silver_options" if options_fol else "gold_premises"}})
        samples.append(row)
        stats["full_compiler"] += 1

        # Keep labels only as metadata for analysis; not as assistant target.
        if gold is not None or exp is not None:
            row["meta"]["gold_answer"] = gold
            row["meta"]["has_explanation"] = exp is not None

    return samples, noisy, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/full_data.json")
    ap.add_argument("--out-dir", default="data/sft_nl2fol")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-silver-options", action="store_true", help="Disable conservative silver labels for MCQ option FOL.")
    args = ap.parse_args()

    data = load_json(args.dataset)
    records = get_records(data)
    n = len(records)
    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    split_by_idx: dict[int, str] = {}
    for i in indices[:n_train]:
        split_by_idx[i] = "train"
    for i in indices[n_train:n_train + n_valid]:
        split_by_idx[i] = "valid"
    for i in indices[n_train + n_valid:]:
        split_by_idx[i] = "test"

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    noisy_cases: list[dict[str, Any]] = []
    stats_by_split: dict[str, Counter] = defaultdict(Counter)
    qtype_counter = Counter()

    include_silver = not args.no_silver_options
    for rid, rec in enumerate(records):
        split = split_by_idx[rid]
        for q in get_questions(rec):
            qtype_counter[detect_question_type(q)] += 1
        samples, noisy, stats = build_samples_for_record(rec, rid, split, include_silver)
        rows_by_split[split].extend(samples)
        noisy_cases.extend(noisy)
        stats_by_split[split].update(stats)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        append_jsonl(out / f"{split}.jsonl", rows_by_split[split])

    report = {
        "dataset": args.dataset,
        "num_records": n,
        "question_type_counter": dict(qtype_counter),
        "splits": {
            split: {
                "num_records": sum(1 for v in split_by_idx.values() if v == split),
                "num_samples": len(rows_by_split[split]),
                "sample_type_counter": dict(stats_by_split[split]),
            }
            for split in ["train", "valid", "test"]
        },
        "include_silver_options": include_silver,
        "num_noisy_cases": len(noisy_cases),
    }
    dump_json(out / "split_report.json", report)
    dump_json(out / "noisy_cases.json", noisy_cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
