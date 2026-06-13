#!/usr/bin/env python3
import argparse, json, re
from collections import Counter
from pathlib import Path

MCQ_RE = re.compile(r"(?m)(^|\n)\s*A[\.)]\s+.+(^|\n)\s*B[\.)]\s+", re.I)
YESNO_STARTERS = ("does ","do ","is ","are ","can ","could ","should ","would ","will ","did ","has ","have ","was ","were ")
OPEN_STARTERS = ("what ","how ","why ","explain ","describe ","calculate ","determine ","identify ","list ","when ","where ","who ")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def records(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for k in ["records","data","items","examples","train"]:
            if isinstance(data.get(k), list): return data[k]
    raise ValueError("Unknown dataset format")

def as_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def field(r, keys):
    for k in keys:
        if k in r: return as_list(r[k])
    return []

def qtext(q):
    if isinstance(q, str): return q
    if isinstance(q, dict):
        for k in ["question","Question","text","prompt"]:
            if k in q: return str(q[k])
    return str(q)

def atext(a):
    if isinstance(a, str): return a.strip()
    if isinstance(a, dict):
        for k in ["answer","Answer","label","gold","value"]:
            if k in a: return str(a[k]).strip()
        return json.dumps(a, ensure_ascii=False)
    return str(a).strip()

def qtype(q):
    text = qtext(q).strip(); lower = text.lower()
    if isinstance(q, dict):
        for k in ["type","question_type","kind"]:
            if k in q:
                v = str(q[k]).lower()
                if "multiple" in v or "mcq" in v or "choice" in v: return "multiple_choice"
                if "yes" in v or "no" in v or "uncertain" in v: return "yes_no"
                if "open" in v: return "open"
        for k in ["choices","options","Choices","Options"]:
            if q.get(k): return "multiple_choice"
    if MCQ_RE.search(text): return "multiple_choice"
    if lower.startswith(OPEN_STARTERS): return "open"
    if lower.startswith(YESNO_STARTERS): return "yes_no"
    return "open"

def atype(a):
    t = atext(a)
    if re.fullmatch(r"[A-E]", t, re.I): return "multiple_choice_answer"
    if t.lower() in {"yes","no","uncertain","unknown"}: return "closed_label_answer"
    return "open_answer"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/full_data.json")
    ap.add_argument("--show-warnings", action="store_true")
    ap.add_argument("--show-first", type=int, default=0)
    args = ap.parse_args()
    recs = records(load_json(args.dataset))
    qc=Counter(); ac=Counter(); warnings=[]
    total_q=total_a=total_e=total_nl=total_fol=0; rec_fol=rec_exp=0
    for ri,r in enumerate(recs):
        qs=field(r,["questions","Questions","question","Question"])
        ans=field(r,["answers","Answers","answer","Answer","gold_answers","gold"])
        exps=field(r,["explanation","explanations","Explanation","Explanations"])
        nlp=field(r,["premises-NL","premises_nl","premises","Premises","context"])
        fol=field(r,["premises-FOL","premises_fol","fol","FOL"])
        total_q += len(qs); total_a += len(ans); total_e += len(exps); total_nl += len(nlp); total_fol += len(fol)
        rec_fol += bool(fol); rec_exp += bool(exps)
        if len(ans)!=len(qs): warnings.append(f"record {ri}: questions={len(qs)} answers={len(ans)}")
        for qi,q in enumerate(qs):
            qt=qtype(q); qc[qt]+=1
            if qi < len(ans):
                at=atype(ans[qi]); ac[at]+=1
                if qt=="multiple_choice" and at!="multiple_choice_answer": warnings.append(f"record {ri}, question {qi}: MCQ but answer_type={at}, answer={atext(ans[qi])[:80]}")
                elif qt=="yes_no" and at!="closed_label_answer": warnings.append(f"record {ri}, question {qi}: yes_no but answer_type={at}, answer={atext(ans[qi])[:80]}")
                elif qt=="open" and at=="closed_label_answer": warnings.append(f"record {ri}, question {qi}: open question has closed-label answer={atext(ans[qi])[:80]}")
    print("="*72); print("DATASET SUMMARY"); print("="*72)
    print(f"Dataset file              : {args.dataset}")
    print(f"Source records            : {len(recs)}")
    print(f"Total question cases      : {total_q}")
    print(f"Total answers             : {total_a}")
    print(f"Total explanations        : {total_e}\n")
    print("Question types:")
    for k,v in qc.most_common(): print(f"  {k:18s}: {v}")
    print("\nAnswer types:")
    for k,v in ac.most_common(): print(f"  {k:24s}: {v}")
    print("\nPremises:")
    print(f"  Records with FOL         : {rec_fol}/{len(recs)}")
    print(f"  Records with explanation : {rec_exp}/{len(recs)}")
    print(f"  Total NL premises        : {total_nl}")
    print(f"  Total FOL premises       : {total_fol}")
    if recs:
        print(f"  Avg questions / record   : {total_q/len(recs):.2f}")
        print(f"  Avg NL premises / record : {total_nl/len(recs):.2f}")
        print(f"  Avg FOL premises / record: {total_fol/len(recs):.2f}")
    print(f"\nWarnings                  : {len(warnings)}")
    if args.show_warnings:
        for w in warnings: print("-", w)
    if args.show_first:
        for ri,r in enumerate(recs[:args.show_first]):
            qs=field(r,["questions","Questions","question","Question"]); ans=field(r,["answers","Answers","answer","Answer"])
            print(f"\nRecord {ri}: questions={len(qs)}, answers={len(ans)}")
            for qi,q in enumerate(qs): print(f"  Q{qi}: type={qtype(q)} answer={atext(ans[qi]) if qi < len(ans) else '<missing>'} | {qtext(q)[:180].replace(chr(10),' ')}")
if __name__ == '__main__': main()
