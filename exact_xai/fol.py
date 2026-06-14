from __future__ import annotations

from dataclasses import dataclass, field
import re
from .fol_repair import repair_fol_string, normalize_predicate_name, normalize_entity_name

VAR_NAMES = {"x", "y", "z", "s", "student", "project", "code"}

@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple[str, ...] = ()
    negated: bool = False

    def __str__(self) -> str:
        prefix = "not " if self.negated else ""
        return f"{prefix}{self.pred}({', '.join(self.args)})"

    def substitute(self, env: dict[str, str]) -> "Atom":
        return Atom(self.pred, tuple(env.get(a, a) for a in self.args), self.negated)

    @property
    def variables(self) -> set[str]:
        return {a for a in self.args if is_var(a)}

@dataclass
class Rule:
    antecedents: list[Atom]
    consequent: Atom
    source_id: int
    source_text: str = ""

@dataclass
class KnowledgeBase:
    facts: set[Atom] = field(default_factory=set)
    rules: list[Rule] = field(default_factory=list)
    premise_texts: list[str] = field(default_factory=list)
    fact_sources: dict[Atom, list[int]] = field(default_factory=dict)

    def constants(self) -> set[str]:
        cs: set[str] = set()
        for f in self.facts:
            for a in f.args:
                if not is_var(a):
                    cs.add(a)
        if not cs:
            cs.add("GENERIC")
        return cs

def is_var(s: str) -> bool:
    return s in VAR_NAMES or (len(s) == 1 and s.islower())

def normalize_fol(s: str) -> str:
    s = s.strip()
    replacements = {
        "ForAll": "forall",
        "Exists": "exists",
        "∀": "forall ",
        "∃": "exists ",
        "→": "->",
        "=>": "->",
        "¬": "not ",
        "~": "not ",
        "∧": "&",
        " and ": " & ",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s

def strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s

def unwrap_quantifier(s: str) -> tuple[str, str | None]:
    s = normalize_fol(s)
    m = re.match(r"^(forall|exists)\s*\(\s*([a-zA-Z_]\w*)\s*,\s*(.*)\)\s*$", s)
    if m:
        return strip_outer_parens(m.group(3)), m.group(1)
    m = re.match(r"^(forall|exists)\s*([a-zA-Z_]\w*)\s*(.*)$", s)
    if m:
        return strip_outer_parens(m.group(3)), m.group(1)
    return s, None

def split_top_level(s: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[start:i].strip())
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(s[start:].strip())
    return [p for p in parts if p]

ATOM_RE = re.compile(r"^(not\s+)?([A-Za-z_]\w*)\s*\((.*)\)\s*$")

def parse_atom(s: str) -> Atom:
    s = repair_fol_string(strip_outer_parens(s.strip()))
    s = s.replace("¬", "not ")
    m = ATOM_RE.match(s)
    if not m:
        name = re.sub(r"\W+", "_", s).strip("_") or "unknown"
        name = normalize_predicate_name(name)
        if name.startswith("not_"):
            return Atom(normalize_predicate_name(name[4:]), (), True)
        return Atom(name, ())
    neg = bool(m.group(1))
    pred = normalize_predicate_name(m.group(2))
    if pred.startswith("not_"):
        neg = True
        pred = normalize_predicate_name(pred[4:])
    args_raw = m.group(3).strip()
    args = tuple(normalize_entity_name(a.strip()) for a in args_raw.split(",") if a.strip()) if args_raw else ()
    return Atom(pred, args, neg)

def parse_conjunction(s: str) -> list[Atom]:
    s = strip_outer_parens(s)
    return [parse_atom(p) for p in split_top_level(s, "&")]

def unwrap_all_quantifiers(text: str) -> tuple[str, str | None]:
    body = text
    quant: str | None = None

    while True:
        next_body, next_quant = unwrap_quantifier(body)

        if next_quant not in {"forall", "exists"}:
            break

        if next_quant == "exists":
            quant = "exists"
        elif quant is None:
            quant = "forall"

        body = strip_outer_parens(next_body)

    return body, quant

def parse_fol_statement(text: str, source_id: int) -> tuple[list[Atom], list[Rule]]:
    text = repair_fol_string(text)

    body, quant = unwrap_all_quantifiers(text)
    body = strip_outer_parens(body)

    facts: list[Atom] = []
    rules: list[Rule] = []

    parts = split_top_level(body, "->")

    if len(parts) >= 2:
        left, right = parts[:2]
        ants = parse_conjunction(left)
        cons = parse_atom(right)
        rules.append(Rule(ants, cons, source_id, text))
    else:
        atoms = parse_conjunction(body) if "&" in body else [parse_atom(body)]

        for atom in atoms:
            if quant == "forall" and atom.variables:
                rules.append(Rule([], atom, source_id, text))
            elif quant == "exists" and atom.variables:
                env = {v: f"EXISTS_{source_id}" for v in atom.variables}
                facts.append(atom.substitute(env))
            else:
                facts.append(atom)

    return facts, rules

def parse_fol_premises(premises_fol: list[str], premises_nl: list[str] | None = None) -> KnowledgeBase:
    kb = KnowledgeBase(premise_texts=premises_nl or [])
    for i, text in enumerate(premises_fol, start=1):
        facts, rules = parse_fol_statement(text, i)
        for f in facts:
            kb.facts.add(f)
            kb.fact_sources.setdefault(f, []).append(i)
        kb.rules.extend(rules)
    return kb
