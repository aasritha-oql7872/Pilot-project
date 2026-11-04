"""Phase A audit: why do agents stop after one search on two-hop questions?

    python -m src.audit retrieval --split twohop_dev          # no model: can two BM25 queries get both docs?
    python -m src.audit episodes outputs/main_v2/eval_twohop_trained.json [...]   # failure taxonomy
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from data.generate import DATA_DIR, read_jsonl
from src.search_tool import Retriever

PERSON_RE = re.compile(r"([A-Z]\w+ [A-Z]\w+) leads|is ([A-Z]\w+ [A-Z]\w+),")


def person_from(text):
    m = PERSON_RE.search(text)
    return (m.group(1) or m.group(2)) if m else None


def retrieval_audit(split):
    """Scripted two-query chain: q1 about the project's PI, q2 about the PI named in the result."""
    retriever = Retriever.for_split(split)
    rows = []
    for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl"):
        pi_id, aff_id = t["evidence_doc_ids"]
        d1 = retriever.search(f"Project {t['subject']} principal investigator")
        d_q = retriever.search(t["question"])
        pi_doc = next((d for d in d1 if d["doc_id"] == pi_id), None)
        person = person_from(pi_doc["text"]) if pi_doc else None
        d2 = retriever.search(f"{person} affiliation") if person else []
        rows.append({"q1_gets_pi": pi_doc is not None, "question_gets_pi": any(d["doc_id"] == pi_id for d in d_q),
                     "question_gets_aff": any(d["doc_id"] == aff_id for d in d_q),
                     "person_extracted": person is not None, "q2_gets_aff": any(d["doc_id"] == aff_id for d in d2)})
    n = len(rows)
    out = {k: round(sum(r[k] for r in rows) / n, 4) for k in rows[0]}
    out["chain_complete"] = round(sum(r["q1_gets_pi"] and r["q2_gets_aff"] for r in rows) / n, 4)
    out["n"] = n
    return out


def episode_audit(report_path):
    """Failure taxonomy over the two-hop episodes of one evaluation report."""
    rep = json.loads(Path(report_path).read_text())
    split = rep["split"]
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(DATA_DIR / f"{split}_corpus.jsonl")}
    eps = [e for e in rep["episodes"] if e["kind"] == "twohop"]
    c = Counter()
    for e in eps:
        got = {d for ids in e["retrieved_doc_ids"] for d in ids}
        pi_id, aff_id = e["evidence_doc_ids"]
        n = e["num_searches"]
        answer = (e["final_answer"] or "").lower()
        c["episodes"] += 1
        c[f"searches={n}"] += 1
        c["first_doc_retrieved"] += pi_id in got
        c["second_doc_retrieved"] += aff_id in got
        c["complete_chain"] += {pi_id, aff_id} <= got
        c["stopped_after_partial_evidence"] += (n == 1 and pi_id in got and aff_id not in got)
        c["answered_with_intermediate_entity"] += bool(answer) and answer in docs[pi_id].lower()
        c["invalid_action"] += e["num_invalid"] > 0 and e["final_answer"] is not None
        c["no_answer_or_malformed"] += e["final_answer"] is None
        c["correct"] += e["correct"]
    n = c["episodes"]
    return {"report": str(report_path), "n_twohop": n, **{k: (round(v / n, 4) if n else None) for k, v in sorted(c.items()) if k != "episodes"}}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval"); r.add_argument("--split", default="twohop_dev")
    e = sub.add_parser("episodes"); e.add_argument("reports", nargs="+")
    args = parser.parse_args()
    if args.cmd == "retrieval":
        print(json.dumps(retrieval_audit(args.split), indent=2))
    else:
        for path in args.reports:
            print(json.dumps(episode_audit(path), indent=2))


if __name__ == "__main__":
    main()
