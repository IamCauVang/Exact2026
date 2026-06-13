from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import subprocess
import textwrap
import time
from collections import Counter, defaultdict
from tqdm import tqdm

from exact_xai.io_utils import load_json, save_json
from exact_xai.schemas import AnswerRequest
from exact_xai.pipeline import AnswerPipeline
from exact_xai.llm_qwen import maybe_load_qwen
from exact_xai.fol import parse_fol_premises, parse_atom
from exact_xai.reasoner import Reasoner
from exact_xai.explanation import proof_to_explanation
from exact_xai.dataset_utils import records_from_exact_dataset, infer_question_kind


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}m:{s:05.2f}s"


def one_line(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return textwrap.shorten(text, width=limit, placeholder="...") if len(text) > limit else text


def print_case_start(index: int, total: int, req: AnswerRequest) -> None:
    qtype = req.question_type or infer_question_kind(req.question)
    print("", flush=True)
    print(f"[case {index}/{total}] id={req.id} type={qtype}", flush=True)
    print(f"Question: {one_line(req.question)}", flush=True)
    if req.gold_answer is not None:
        print(f"Gold: {req.gold_answer}", flush=True)


def print_case_end(index: int, total: int, req: AnswerRequest, row: dict, elapsed_s: float) -> None:
    pred = row.get("answer")
    gold = req.gold_answer
    status = "N/A"
    if gold is not None:
        status = "OK" if _is_correct(pred, gold) else "WRONG"
    warnings = row.get("warnings") or []
    warn_txt = "; ".join(map(str, warnings)) if warnings else "none"
    print(
        f"[case {index}/{total} done in {fmt_seconds(elapsed_s)}] "
        f"answer={pred} gold={gold} status={status} warnings={warn_txt}",
        flush=True,
    )
    raw = row.get("raw") or {}
    option_results = raw.get("option_results")
    if option_results:
        print(f"Option results: {json.dumps(option_results, ensure_ascii=False)}", flush=True)


def records_from_batch(path: str):
    obj = load_json(path)
    for r in obj.get("records", []):
        yield AnswerRequest(**r)


def run_unit_smoke(out: Path):
    kb = parse_fol_premises([
        "ForAll(x, A(x) -> B(x))",
        "ForAll(x, B(x) -> C(x))",
        "A(Sophia)",
    ])
    t0 = time.perf_counter()
    rr = Reasoner(kb).prove_atom(parse_atom("C(Sophia)"))
    runtime = time.perf_counter() - t0
    explanation = proof_to_explanation(rr.answer, rr.proof, explanation_style="short")
    row = {
        "id": "smoke:0",
        "answer": rr.answer,
        "target": "C(Sophia)",
        "proof": [step.model_dump() for step in rr.proof],
        "explanation": explanation,
        "gold_answer": "Yes",
        "runtime_seconds": runtime,
    }
    report = {
        "task": "smoke",
        "num_question_cases": 1,
        "num_with_gold": 1,
        "exact_match": 1.0 if rr.answer == "Yes" else 0.0,
        "model_mode": "symbolic_unit_smoke_no_llm",
        "total_runtime_seconds": runtime,
    }
    save_json([row], out / "answers.json")
    save_json(report, out / "eval_report.json")
    save_json([], out / "bad_cases.json")
    save_json([], out / "warning_cases.json")
    save_json([{"id": "smoke:0", "runtime_seconds": runtime}], out / "case_timings.json")
    print(report, flush=True)


def _is_correct(pred, gold) -> bool:
    if gold is None:
        return False
    return str(pred).strip().lower() == str(gold).strip().lower()


def save_readable_qa_report(records: list[AnswerRequest], results: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# QA Report")
    lines.append("")
    lines.append(f"- Total question cases: `{len(results)}`")
    correct = 0
    with_gold = 0
    warning_cases = 0
    for row, req in zip(results, records):
        if req.gold_answer is not None:
            with_gold += 1
            correct += int(_is_correct(row.get("answer"), req.gold_answer))
        if row.get("warnings"):
            warning_cases += 1
    lines.append(f"- Correct: `{correct}/{with_gold}`")
    lines.append(f"- Warning cases: `{warning_cases}`")
    lines.append("")

    for row, req in zip(results, records):
        pred = row.get("answer")
        gold = req.gold_answer
        status = "N/A" if gold is None else ("✅ CORRECT" if _is_correct(pred, gold) else "❌ WRONG")
        qtype = row.get("question_type") or req.question_type or infer_question_kind(req.question)
        warnings = row.get("warnings") or []
        used = row.get("used_premises") or []
        raw = row.get("raw") or {}
        parsed = row.get("parsed_question") or {}

        lines.append(f"## Case {row.get('id') or req.id} — {status}")
        lines.append("")
        lines.append(f"**Type:** `{qtype}`")
        lines.append("")
        lines.append(f"**Question:** {req.question}")
        lines.append("")
        lines.append(f"**Answer:** `{pred}`")
        lines.append(f"**Gold:** `{gold}`")
        lines.append(f"**Runtime:** `{fmt_seconds(float(row.get('runtime_seconds') or 0.0))}`")
        lines.append("")
        if warnings:
            lines.append(f"**Warnings:** `{'; '.join(map(str, warnings))}`")
        else:
            lines.append("**Warnings:** none")
        lines.append("")
        lines.append(f"**Used premises:** `{used}`")
        lines.append("")
        if req.premises_nl:
            lines.append("**Premises:**")
            lines.append("")
            for i, prem in enumerate(req.premises_nl, start=1):
                mark = "✅" if i in used else "  "
                lines.append(f"{mark} P{i}. {prem}")
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
        lines.append(f"**Explanation:** {row.get('explanation', '')}")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_answer_batch(
    records: list[AnswerRequest],
    pipe: AnswerPipeline,
    out: Path,
    task: str,
    batch_size: int | None = None,
    loader_warnings: list[str] | None = None,
    log_cases: bool = False,
):
    results = []
    bad_cases = []
    warning_cases = []
    case_timings = []
    metrics_by_type = defaultdict(lambda: {"total": 0, "with_gold": 0, "correct": 0})
    warning_counter = Counter()
    parse_counter = Counter()

    correct = total_with_gold = 0
    if batch_size is None or batch_size <= 0:
        batch_size = len(records) or 1

    total_cases = len(records)
    run_t0 = time.perf_counter()
    batch_reports = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        for offset, req in enumerate(tqdm(batch, desc=f"EXACT {task} {start}:{start+len(batch)}"), start=0):
            case_index = start + offset + 1
            if log_cases:
                print_case_start(case_index, total_cases, req)
            case_t0 = time.perf_counter()
            ans = pipe.answer(req)
            runtime_s = time.perf_counter() - case_t0
            row = ans.model_dump()
            row["gold_answer"] = req.gold_answer
            row["question_type"] = req.question_type or infer_question_kind(req.question)
            row["runtime_seconds"] = runtime_s
            results.append(row)

            qtype = row["question_type"]
            metrics_by_type[qtype]["total"] += 1
            if req.gold_answer is not None:
                total_with_gold += 1
                metrics_by_type[qtype]["with_gold"] += 1
                if _is_correct(ans.answer, req.gold_answer):
                    correct += 1
                    metrics_by_type[qtype]["correct"] += 1
                else:
                    bad_cases.append(row)

            if row.get("warnings"):
                warning_cases.append(row)
                for w in row["warnings"]:
                    warning_counter[w.split(":")[0]] += 1
            if ans.parsed_question:
                parse_counter[ans.parsed_question.parser] += 1

            timing_row = {
                "id": req.id,
                "question_type": row["question_type"],
                "answer": ans.answer,
                "gold_answer": req.gold_answer,
                "correct": _is_correct(ans.answer, req.gold_answer) if req.gold_answer is not None else None,
                "runtime_seconds": runtime_s,
                "warnings": row.get("warnings", []),
            }
            case_timings.append(timing_row)
            if log_cases:
                print_case_end(case_index, total_cases, req, row, runtime_s)

        elapsed_so_far = time.perf_counter() - run_t0
        batch_report = {
            "start": start,
            "end": start + len(batch),
            "num_question_cases_done": len(results),
            "exact_match_so_far": correct / total_with_gold if total_with_gold else None,
            "num_bad_cases_so_far": len(bad_cases),
            "num_warning_cases_so_far": len(warning_cases),
            "elapsed_seconds_so_far": elapsed_so_far,
        }
        batch_reports.append(batch_report)
        save_json(batch_report, out / "batch_progress" / f"batch_{start:06d}.json")
        save_json(results, out / "answers.partial.json")
        save_json(case_timings, out / "case_timings.partial.json")

    by_type = {}
    for k, v in metrics_by_type.items():
        by_type[k] = {
            **v,
            "exact_match": v["correct"] / v["with_gold"] if v["with_gold"] else None,
        }

    total_runtime = time.perf_counter() - run_t0
    runtimes = [x["runtime_seconds"] for x in case_timings]
    report = {
        "task": task,
        "num_question_cases": len(records),
        # Backward-compatible alias. Prefer num_question_cases in new code.
        "num_records": len(records),
        "num_with_gold": total_with_gold,
        "exact_match": correct / total_with_gold if total_with_gold else None,
        "by_question_type": by_type,
        "num_bad_cases": len(bad_cases),
        "num_warning_cases": len(warning_cases),
        "warning_counter": dict(warning_counter),
        "parser_counter": dict(parse_counter),
        "loader_warnings": loader_warnings or [],
        "model_mode": results[0].get("mode") if results else "none",
        "total_runtime_seconds": total_runtime,
        "avg_case_runtime_seconds": (sum(runtimes) / len(runtimes)) if runtimes else None,
        "max_case_runtime_seconds": max(runtimes) if runtimes else None,
        "min_case_runtime_seconds": min(runtimes) if runtimes else None,
    }

    save_json(results, out / "answers.json")
    save_json(report, out / "eval_report.json")
    save_json(bad_cases, out / "bad_cases.json")
    save_json(warning_cases, out / "warning_cases.json")
    save_json(batch_reports, out / "batch_reports.json")
    save_json(case_timings, out / "case_timings.json")
    save_readable_qa_report(records, results, out / "qa_report.md")
    print(report, flush=True)


def run_translation_benchmark(records: list[AnswerRequest], pipe: AnswerPipeline, out: Path, log_cases: bool = False):
    rows = []
    ok = 0
    total = 0
    timings = []
    t0_all = time.perf_counter()
    for i, req in enumerate(tqdm(records, desc="NL->Logic benchmark"), start=1):
        if log_cases:
            print_case_start(i, len(records), req)
        t0 = time.perf_counter()
        kb, warnings, raw = pipe.build_kb(req)
        elapsed = time.perf_counter() - t0
        generated = raw.get("generated_premises_fol", req.premises_fol)
        total += 1
        valid = bool(generated)
        ok += int(valid)
        row = {
            "id": req.id,
            "valid": valid,
            "warnings": warnings,
            "generated_premises_fol": generated,
            "gold_premises_fol": req.premises_fol,
            "runtime_seconds": elapsed,
        }
        rows.append(row)
        timings.append({"id": req.id, "runtime_seconds": elapsed, "valid": valid, "warnings": warnings})
        if log_cases:
            print(f"[translate case {i}/{len(records)} done in {fmt_seconds(elapsed)}] valid={valid} warnings={warnings or 'none'}", flush=True)
    report = {
        "task": "translate",
        "num_question_cases": total,
        "num_records": total,
        "translation_nonempty_rate": ok / total if total else None,
        "total_runtime_seconds": time.perf_counter() - t0_all,
        "avg_case_runtime_seconds": sum(x["runtime_seconds"] for x in timings) / len(timings) if timings else None,
    }
    save_json(rows, out / "translations.json")
    save_json(report, out / "translation_report.json")
    save_json(timings, out / "case_timings.json")
    print(report, flush=True)



def _run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_v47_build_silver(dataset: Path, out: Path) -> None:
    silver_dir_env = os.environ.get("SILVER_OUT_DIR", "data/silver_v47")
    silver_dir = Path(silver_dir_env)
    if not silver_dir.is_absolute():
        # Build inside artifact dir so outputs are always downloaded.
        silver_dir = out / silver_dir_env
    _run_cmd([
        sys.executable, str(Path(__file__).resolve().parent / "build_silver_nl2fol_dataset.py"),
        "--dataset", str(dataset),
        "--out-dir", str(silver_dir),
    ])
    print(f"[v4.7] silver dataset ready: {silver_dir}", flush=True)


def run_v47_train_parser(dataset: Path, out: Path) -> None:
    # Build silver data first unless explicit train/valid files are supplied.
    silver_dir_env = os.environ.get("SILVER_OUT_DIR", "data/silver_v47")
    silver_dir = Path(silver_dir_env)
    if not silver_dir.is_absolute():
        silver_dir = out / silver_dir_env
    train_file = os.environ.get("TRAIN_FILE", "")
    valid_file = os.environ.get("VALID_FILE", "")
    if not train_file:
        if not (silver_dir / "all.train.jsonl").exists():
            run_v47_build_silver(dataset, out)
        # Default to all.train so question-parse weak samples teach target/choice schema too.
        train_file = str(silver_dir / "all.train.jsonl")
        valid_file = str(silver_dir / "all.valid.jsonl")
    output_adapter = os.environ.get("OUTPUT_ADAPTER", "train_artifacts/adapter_v47")
    adapter_out = Path(output_adapter)
    if not adapter_out.is_absolute():
        adapter_out = out / output_adapter
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "train_lora_nl2fol_v47.py"),
        "--train", train_file,
        "--valid", valid_file,
        "--model", os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        "--out", str(adapter_out),
        "--epochs", os.environ.get("TRAIN_EPOCHS", "2"),
        "--max-steps", os.environ.get("TRAIN_MAX_STEPS", "-1"),
        "--batch-size", os.environ.get("TRAIN_BATCH_SIZE", "1"),
        "--grad-accum", os.environ.get("GRAD_ACCUM", "8"),
        "--lr", os.environ.get("LEARNING_RATE", "2e-4"),
        "--min-confidence", os.environ.get("MIN_CONFIDENCE", "0.0"),
        "--use-4bit", os.environ.get("USE_4BIT", "1"),
    ]
    _run_cmd(cmd)
    print(f"[v4.7] adapter ready: {adapter_out}", flush=True)


def run_v47_audit(out: Path) -> None:
    answers = out / "answers.json"
    if not answers.exists():
        raise FileNotFoundError(f"answers.json not found for audit: {answers}. Run benchmark first or pass artifacts back.")
    _run_cmd([
        sys.executable, str(Path(__file__).resolve().parent / "audit_dirty_cases.py"),
        "--answers", str(answers),
        "--out-dir", str(out / "dirty_audit"),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["smoke", "batch", "eval", "benchmark", "translate", "batch_nl", "build_silver", "train_parser_v47", "workflow_v47", "audit_dirty"], default="batch")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--requests", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--case-ids", default=None, help="Comma-separated AnswerRequest ids to run, e.g. 20:1,23:0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--input-mode", choices=["auto", "fol", "nl"], default="auto")
    ap.add_argument("--no-z3", action="store_true")
    ap.add_argument("--log-cases", action="store_true", help="Print each question, answer, warning, and runtime to terminal.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "smoke":
        run_unit_smoke(out)
        return

    loader_warnings = []
    if args.requests:
        records = list(records_from_batch(args.requests))
    elif args.dataset:
        data = load_json(args.dataset)
        input_mode = "nl" if args.task == "batch_nl" else args.input_mode
        records, loader_warnings = records_from_exact_dataset(data, input_mode=input_mode)
    else:
        raise SystemExit("Provide --dataset or --requests")

    # v4.7 data-driven parser workflow tasks. These do not need AnswerPipeline/LLM benchmark loop.
    if args.task == "build_silver":
        run_v47_build_silver(Path(args.dataset), out)
        return
    if args.task == "train_parser_v47":
        run_v47_train_parser(Path(args.dataset), out)
        return
    if args.task == "workflow_v47":
        run_v47_build_silver(Path(args.dataset), out)
        run_v47_train_parser(Path(args.dataset), out)
        return
    if args.task == "audit_dirty":
        run_v47_audit(out)
        return

    if args.case_ids:
        wanted = {x.strip() for x in str(args.case_ids).split(",") if x.strip()}
        before = len(records)
        records = [r for r in records if str(r.id) in wanted]
        found = {str(r.id) for r in records}
        missing = sorted(wanted - found)
        if missing:
            loader_warnings.append(f"case_ids_missing:{missing}")
        print(f"[case-ids] selected {len(records)}/{before}: {sorted(found)}", flush=True)
    if args.start:
        records = records[args.start:]
    if args.limit:
        records = records[: args.limit]

    llm = maybe_load_qwen()
    input_mode = "nl" if args.task == "batch_nl" else args.input_mode
    pipe = AnswerPipeline(llm=llm, input_mode=input_mode, use_z3=not args.no_z3)

    if args.task == "translate":
        pipe.input_mode = "nl"
        run_translation_benchmark(records, pipe, out, log_cases=args.log_cases)
    else:
        run_answer_batch(
            records,
            pipe,
            out,
            args.task,
            batch_size=args.batch_size,
            loader_warnings=loader_warnings,
            log_cases=args.log_cases,
        )


if __name__ == "__main__":
    main()
