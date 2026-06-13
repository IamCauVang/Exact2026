#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact_xai.silver_dataset import (
    read_json, write_json, write_jsonl, unwrap_dataset,
    build_premise_translation_sample, build_question_parse_samples,
    split_by_family,
)


def main() -> None:
    ap = argparse.ArgumentParser(description='Build v4.7 silver NL->FOL/parser dataset from EXACT records.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out-dir', default='data/silver_v47')
    ap.add_argument('--valid-ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--include-question-parse', action='store_true', default=True)
    ap.add_argument('--no-question-parse', dest='include_question_parse', action='store_false')
    args = ap.parse_args()

    records = unwrap_dataset(read_json(args.dataset))
    all_rows = []
    premise_rows = []
    question_rows = []
    weak_rows = []
    skipped = 0

    for idx, rec in enumerate(records):
        prem = build_premise_translation_sample(rec, idx)
        if prem is not None:
            premise_rows.append(prem)
            all_rows.append(prem)
        else:
            skipped += 1
        if args.include_question_parse:
            qs = build_question_parse_samples(rec, idx)
            question_rows.extend(qs)
            weak_rows.extend(qs)
            all_rows.extend(qs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clean_rows = premise_rows
    clean_train, clean_valid = split_by_family(clean_rows, args.valid_ratio, args.seed)
    weak_train, weak_valid = split_by_family(weak_rows, args.valid_ratio, args.seed)
    all_train, all_valid = split_by_family(all_rows, args.valid_ratio, args.seed)

    counts = {
        'records': len(records),
        'skipped_no_premises': skipped,
        'premise_samples': len(premise_rows),
        'question_parse_samples': len(question_rows),
        'all_samples': len(all_rows),
        'clean_train': len(clean_train),
        'clean_valid': len(clean_valid),
        'weak_train': len(weak_train),
        'weak_valid': len(weak_valid),
        'all_train': len(all_train),
        'all_valid': len(all_valid),
    }

    write_jsonl(out / 'clean.train.jsonl', clean_train)
    write_jsonl(out / 'clean.valid.jsonl', clean_valid)
    write_jsonl(out / 'weak.train.jsonl', weak_train)
    write_jsonl(out / 'weak.valid.jsonl', weak_valid)
    write_jsonl(out / 'all.train.jsonl', all_train)
    write_jsonl(out / 'all.valid.jsonl', all_valid)
    write_jsonl(out / 'all.jsonl', all_rows)
    write_json(out / 'stats.json', counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
