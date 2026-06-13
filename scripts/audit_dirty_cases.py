#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_json(path: str | Path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Audit suspicious dirty labels / parser failures from answers.json.')
    ap.add_argument('--answers', required=True)
    ap.add_argument('--out-dir', default='artifacts/dirty_audit')
    args = ap.parse_args()
    rows = load_json(args.answers)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dirty = []
    families = defaultdict(list)
    for row in rows:
        rid = str(row.get('id', ''))
        fam = rid.split(':')[0]
        families[fam].append(row)
        pred = str(row.get('answer')).strip().lower()
        gold = str(row.get('gold_answer')).strip().lower()
        parsed = row.get('parsed_question') or {}
        warnings = row.get('warnings') or []
        reasons = []
        if parsed.get('target') is None and parsed.get('kind') == 'yes_no' and gold in {'yes', 'no'}:
            reasons.append('target_null_but_gold_yes_no')
        if 'no_multiple_choice_option_provable' in warnings and gold in {'a','b','c','d'}:
            reasons.append('gold_option_unprovable')
        if any(str(w).startswith('multiple_provable_options') for w in warnings):
            reasons.append('multiple_provable_options_ambiguous')
        if pred != gold and gold not in {'none', 'null'}:
            reasons.append('prediction_gold_mismatch')
        if reasons:
            dirty.append({'id': rid, 'answer': row.get('answer'), 'gold_answer': row.get('gold_answer'), 'reasons': reasons, 'warnings': warnings})

    # pairwise contradiction heuristic: same family has one correct MCQ saying Yes-ish and a yes/no mismatch.
    for fam, fam_rows in families.items():
        mismatches = [r for r in fam_rows if str(r.get('answer')).lower() != str(r.get('gold_answer')).lower()]
        if len(mismatches) >= 1 and len(fam_rows) > 1:
            for r in mismatches:
                dirty.append({'id': r.get('id'), 'answer': r.get('answer'), 'gold_answer': r.get('gold_answer'), 'reasons': ['family_has_mismatch_check_manually'], 'warnings': r.get('warnings', [])})

    # de-duplicate by id + reasons
    seen = set(); uniq = []
    for d in dirty:
        key = (d['id'], tuple(d['reasons']))
        if key not in seen:
            seen.add(key); uniq.append(d)

    write_jsonl(out / 'dirty_candidates.jsonl', uniq)
    md = ['# Dirty / suspicious candidates', '', f'- Total: `{len(uniq)}`', '']
    for d in uniq:
        md.append(f"## {d['id']}")
        md.append(f"- answer: `{d['answer']}`")
        md.append(f"- gold: `{d['gold_answer']}`")
        md.append(f"- reasons: `{', '.join(d['reasons'])}`")
        if d.get('warnings'):
            md.append(f"- warnings: `{'; '.join(map(str, d['warnings']))}`")
        md.append('')
    (out / 'dirty_candidates.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({'dirty_candidates': len(uniq), 'out_dir': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
