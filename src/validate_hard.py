"""Validate the hard benchmark: ground truth is derivable in <= 2 searches from OBSERVED text, splits are disjoint, and
distractors really compete. No model is involved.

    python -m src.validate_hard                    # structural checks on all three splits + naive baselines on hard_dev
"""

import json
import re
import sys

from data.generate import DATA_DIR, read_jsonl
from src.audit import PERSON_RE
from src.reward import ABSTAIN_ANSWER, normalize
from src.search_tool import Retriever

SPLITS = ["hard_train", "hard_dev", "hard_final"]
AFF_RE = [re.compile(r"^(?P<p>.+?) is affiliated with (?P<v>.+?), the institution"),
          re.compile(r"^(?P<p>.+?) works at (?P<v>.+?) and is affiliated")]
ATTR_WORDS = {"city": "city based located", "funder": "funded funding agency", "year": "started launched year"}


def person_of_project(docs, name):
    """PI named by the document about exactly `Project <name>` (not a decoy 'Project <name> Prime')."""
    for d in docs:
        if re.search(rf"Project {re.escape(name)}(?! [A-Z])", d["text"]) and ("principal investigator" in d["text"] or "leads" in d["text"]):
            m = PERSON_RE.search(d["text"])
            if m:
                return m.group(1) or m.group(2), d
    return None, None


def affiliation_of(docs, person, top1=False):
    for d in docs[:1] if top1 else docs:
        for rx in AFF_RE:
            m = rx.match(d["text"])
            if m and m.group("p") == person:
                return m.group("v"), d
    return None, None


def project_of(docs, person):
    for d in docs:
        m = re.search(rf"{re.escape(person)} leads Project (\w+)", d["text"]) or re.search(rf"Project (\w+) principal investigator is {re.escape(person)}", d["text"])
        if m:
            return m.group(1), d
    return None, None


def solve(task, retriever, top1=False):
    """Scripted two-search agent that only reads retrieved text. top1=True reads just the first-ranked record
    (the naive 'trust the top hit' policy). Returns (answer, retrieved doc ids)."""
    got = []
    run = lambda q: (lambda r: (got.extend(d["doc_id"] for d in r), r)[1])(retriever.search(q))
    if task["category"] == "reverse_two_hop":
        project, _ = project_of(run(f"{task['subject']} leads principal investigator")[:1 if top1 else 2], task["subject"])
        attr = next(a for a, kw in {"city": "city", "funder": "agency", "year": "year"}.items() if kw in task["question"])
        docs = run(f"Project {project} {ATTR_WORDS[attr]}") if project else []
        target = f"{project.lower()}_{attr}" if project else None
        d = next((x for x in docs if x["doc_id"] == target), docs[0] if (top1 and docs) else None)
        text = d["text"] if d else ""
        m = re.search(r"(?:city of|located in|is funded by the|provided by the|started in|launched in) ([\w ]+?)(?:,|\.| where| the)", text)
        return (m.group(1) if m else None), got
    person, _ = person_of_project(run(f"Project {task['subject']} principal investigator")[:1 if top1 else 2], task["subject"])
    if person is None:
        return None, got
    value, _ = affiliation_of(run(f"{person} affiliation"), person, top1)
    return (value if value is not None else ABSTAIN_ANSWER), got


def validate_split(split, naive=False):
    retriever = Retriever.for_split(split)
    tasks = read_jsonl(DATA_DIR / f"{split}_tasks.jsonl")
    corpus = {d["doc_id"]: d["text"] for d in read_jsonl(DATA_DIR / f"{split}_corpus.jsonl")}
    out = {}
    for t in tasks:
        row = out.setdefault(t["category"], {"n": 0, "oracle_correct": 0, "evidence_reachable": 0, "naive_top1_correct": 0, "gold_rank1_q1": 0})
        row["n"] += 1
        ans, got = solve(t, retriever)
        row["oracle_correct"] += normalize(ans or "") == normalize(t["gold"])
        row["evidence_reachable"] += set(t["evidence_doc_ids"]) <= set(got)
        if naive:
            nans, _ = solve(t, retriever, top1=True)
            row["naive_top1_correct"] += normalize(nans or "") == normalize(t["gold"])
            r1 = retriever.search(f"Project {t['subject']} principal investigator" if t["category"] != "reverse_two_hop"
                                  else f"{t['subject']} leads principal investigator")
            row["gold_rank1_q1"] += bool(r1) and r1[0]["doc_id"] == t["evidence_doc_ids"][0]
        # a defensible ground truth: the gold string is inside the final evidence document (answerable tasks)
        if t["answerable"]:
            assert t["gold"] in corpus[t["evidence_doc_ids"][-1]], f"gold not in evidence doc: {t['task_id']}"
        else:
            person = PERSON_RE.search(corpus[t["evidence_doc_ids"][0]])
            name = person.group(1) or person.group(2)
            assert not any(AFF_RE[0].match(x) or AFF_RE[1].match(x) for x in corpus.values()
                           if x.startswith(name + " ")), f"unanswerable task has an affiliation record: {t['task_id']}"
    return out


def is_pseudo_word(word):
    """True iff the word is a concatenation of the generator's name syllables (and an optional city suffix)."""
    from data.generate import CITY_SUFFIXES, SYLLABLES
    w = word.lower()
    for suffix in CITY_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix):
            w = w[: -len(suffix)]
            break
    ok = [True] + [False] * len(w)
    for i in range(1, len(w) + 1):
        ok[i] = any(ok[i - len(sy)] for sy in SYLLABLES if w.endswith(sy, 0, i) and i >= len(sy))
    return ok[len(w)]


def entity_words(split):
    """Generated entity names only (closed-class words like first names, org suffixes, 'Alpha' are shared by design)."""
    words = set()
    for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl"):
        words |= set(re.findall(r"[A-Z][a-z]{3,}", t["question"] + " " + t["gold"]))
    for d in read_jsonl(DATA_DIR / f"{split}_corpus.jsonl"):
        words |= set(re.findall(r"[A-Z][a-z]{3,}", d["text"]))
    return {w for w in words if is_pseudo_word(w)}


def main():
    report = {s: validate_split(s, naive=(s == "hard_dev")) for s in SPLITS}
    print(json.dumps(report, indent=2))
    bad = [(s, c) for s, rows in report.items() for c, r in rows.items() if r["evidence_reachable"] != r["n"] or r["oracle_correct"] != r["n"]]
    if bad:
        print("INVALID categories:", bad); sys.exit(1)


if __name__ == "__main__":
    main()
