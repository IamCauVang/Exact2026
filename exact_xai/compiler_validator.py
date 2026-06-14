from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

PRED_BAD_RE = re.compile(r"[A-Za-z_]\w*\s+[A-Za-z_]\w*\s*\(")

@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] | None = None


def load_compiler_json(text: str) -> ValidationResult:
    """Parse a JSON-only compiler response and perform shape checks.

    This validator is intentionally conservative. It should be run before sending
    LLM output to the solver.
    """
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        data = json.loads(cleaned)
    except Exception as e:
        return ValidationResult(False, errors=[f"invalid_json:{type(e).__name__}:{e}"])
    return validate_compiler_ir(data)


def validate_compiler_ir(data: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    premises = data.get("premises_fol", [])
    if premises and not isinstance(premises, list):
        errors.append("premises_fol_must_be_list")
    for i, p in enumerate(premises):
        fol = p.get("fol") if isinstance(p, dict) else str(p)
        if not fol:
            errors.append(f"premise_{i}_empty_fol")
        if PRED_BAD_RE.search(fol):
            errors.append(f"premise_{i}_predicate_contains_space")

    question = data.get("question", {})

    if question and not isinstance(question, dict):
        errors.append("question_must_be_object")
        question = {}

    qtype = (
        question.get("type")
        or data.get("question_type")
        or data.get("kind")
    )

    if qtype not in {None, "multiple_choice", "yes_no", "open"}:
        warnings.append(f"unknown_question_type:{qtype}")

    options_text = question.get("options_text", {}) if isinstance(question, dict) else {}
    options_fol = question.get("options_fol", {}) if isinstance(question, dict) else {}
    if qtype == "multiple_choice":
        if options_text and not isinstance(options_text, dict):
            errors.append("options_text_must_be_object")
        if options_fol and not isinstance(options_fol, dict):
            errors.append("options_fol_must_be_object")
        for label, opt_text in (options_text or {}).items():
            fol = (options_fol or {}).get(label)
            if not fol:
                warnings.append(f"missing_option_fol:{label}")
                continue
            if str(opt_text).strip().lower().startswith("if ") and "->" not in fol:
                errors.append(f"conditional_option_without_implication:{label}")
            if PRED_BAD_RE.search(fol):
                errors.append(f"option_{label}_predicate_contains_space")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, data=data)
