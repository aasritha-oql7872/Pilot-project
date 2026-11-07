"""Headroom benchmark ("hard"): answerable and unanswerable questions with distractors, in the same BM25 environment.

Splits (disjoint entities, own corpus each): hard_train (GRPO training only), hard_dev (model selection / headroom
check), hard_final (LOCKED: do not evaluate until the method is frozen; hashes in data/manifest_hard.json).

Categories (all solvable with <= 2 searches; MAX_SEARCHES = 2 in this environment):
  standard         plain two-hop control (project -> PI -> PI's institution)
  distractor_name  a decoy project "Project <N> Prime" shares the target's name: its PI/affiliation docs compete
  distractor_person a decoy person shares the PI's surname; both affiliation docs match the second query
  stale_affiliation a document says the PI was *previously* affiliated with another institution (gold = current one)
  reverse_two_hop  "city/funder/year of the project led by <PI>": PI doc first, then the project's attribute doc
  unanswerable     the PI's affiliation record does not exist; correct behaviour is <answer>insufficient evidence</answer>

Not generated (documented in reports/evaluation_design.md): three-hop / alias chains (need > 2 searches), and
"failed first search" reformulation (the rare project-name token always makes the first query succeed in BM25).

    python -m data.make_hard --seed 20261106
"""

import argparse
import json
import random
import re

from data.generate import (CITY_SUFFIXES, DATA_DIR, DOC_TEMPLATES, FIRST_NAMES, ORG_SUFFIXES, TWOHOP_QUESTIONS, file_hash,
                           make_word, read_jsonl, write_jsonl)
from src.reward import ABSTAIN_ANSWER

EXISTING = ["train", "dev", "test", "twohop", "twohop_train", "twohop_dev", "final", "twohop_final"]
CATEGORIES = ["standard", "distractor_name", "distractor_person", "stale_affiliation", "reverse_two_hop", "unanswerable"]
SIZES = {"hard_train": 20, "hard_dev": 10, "hard_final": 20}      # tasks per category
DECOY_SUFFIX = ["Prime", "Alpha", "Next", "Two"]
# The word "affiliation" is what a searcher naturally types; the current-affiliation templates say "affiliated", so
# the stale record outranks the current one for the query "<person> affiliation" (a misleading first result).
FORMER = ["{p}'s previous affiliation was {v}, an institution where {p} no longer works.",
          "The former affiliation of {p} was {v}; {p} no longer works there."]
REVERSE_Q = {"city": "In which city is the project led by {pi} based?",
             "funder": "Which agency funds the project led by {pi}?",
             "year": "In what year did the project led by {pi} start?"}


def used_words():
    words = set()
    for split in EXISTING:
        for d in read_jsonl(DATA_DIR / f"{split}_corpus.jsonl"):
            words |= set(re.findall(r"[A-Z][a-z]+", d["text"]))
        for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl"):
            words |= set(re.findall(r"[A-Z][a-z]+", t["question"] + " " + t["gold"]))
    for w in list(words):
        for suffix in CITY_SUFFIXES:
            if w.endswith(suffix):
                words.add(w[: -len(suffix)])
    return words


def make_project(rng, used):
    """As data.generate.make_project but with 3-syllable surnames (the 2-syllable pool is exhausted by earlier splits)."""
    return {"name": make_word(rng, used), "pi": f"{rng.choice(FIRST_NAMES)} {make_word(rng, used)}",
            "year": str(rng.randint(1985, 2025)), "funder": f"{make_word(rng, used)} {rng.choice(ORG_SUFFIXES)}",
            "city": make_word(rng, used) + rng.choice(CITY_SUFFIXES),
            "affiliation": f"{make_word(rng, used)} {rng.choice(ORG_SUFFIXES)}"}


def docs_for(rng, proj, attrs, tag=None):
    """Corpus docs for selected attributes of a project; ids follow the repository convention."""
    out = []
    for attr in attrs:
        subj = proj["pi"] if attr == "affiliation" else proj["name"]
        out.append({"doc_id": f"{proj['name'].lower().replace(' ', '_')}_{attr}",
                    "text": rng.choice(DOC_TEMPLATES[attr]).format(p=subj, v=proj[attr])})
    return out


ALL_ATTRS = list(DOC_TEMPLATES)
pid = lambda proj, attr: f"{proj['name'].lower().replace(' ', '_')}_{attr}"


def task(category, proj, question, gold, evidence, decoys, answerable=True, kind="twohop", subject=None):
    return {"kind": kind, "category": category, "answerable": answerable, "attribute": "pi_affiliation",
            "subject": subject or proj["name"], "gold": gold, "question": question,
            "evidence_doc_ids": evidence, "decoy_doc_ids": decoys}


def make_one(category, rng, used):
    proj = make_project(rng, used)
    docs = docs_for(rng, proj, ALL_ATTRS)
    q2 = rng.choice(TWOHOP_QUESTIONS).format(p=proj["name"])
    ev = [pid(proj, "pi"), pid(proj, "affiliation")]
    if category == "standard":
        return task(category, proj, q2, proj["affiliation"], ev, []), docs
    if category == "distractor_name":
        decoy = make_project(rng, used)
        decoy["name"] = f"{proj['name']} {rng.choice(DECOY_SUFFIX)}"
        dd = docs_for(rng, decoy, ALL_ATTRS)
        return task(category, proj, q2, proj["affiliation"], ev, [pid(decoy, "pi"), pid(decoy, "affiliation")]), docs + dd
    if category == "distractor_person":
        decoy = make_project(rng, used)
        surname = proj["pi"].split()[-1]
        first = rng.choice([f for f in FIRST_NAMES if f != proj["pi"].split()[0]])
        decoy["pi"] = f"{first} {surname}"
        dd = docs_for(rng, decoy, ["pi", "affiliation"])
        return task(category, proj, q2, proj["affiliation"], ev, [pid(decoy, "pi"), pid(decoy, "affiliation")]), docs + dd
    if category == "stale_affiliation":
        old = f"{make_word(rng, used)} {rng.choice(['University', 'Institute', 'Foundation', 'Council', 'Laboratory'])}"
        former = {"doc_id": f"{proj['name'].lower()}_former", "text": rng.choice(FORMER).format(p=proj["pi"], v=old)}
        return task(category, proj, q2, proj["affiliation"], ev, [former["doc_id"]]), docs + [former]
    if category == "reverse_two_hop":
        attr = rng.choice(list(REVERSE_Q))
        return task(category, proj, REVERSE_Q[attr].format(pi=proj["pi"]), proj[attr],
                    [pid(proj, "pi"), pid(proj, attr)], [], subject=proj["pi"]), docs
    if category == "unanswerable":
        docs = [d for d in docs if d["doc_id"] != pid(proj, "affiliation")]
        return task(category, proj, q2, ABSTAIN_ANSWER, [pid(proj, "pi")], [], answerable=False, kind="unanswerable"), docs
    raise ValueError(category)


def build(rng, used, split):
    tasks, corpus = [], []
    for category in CATEGORIES:
        for _ in range(SIZES[split]):
            t, docs = make_one(category, rng, used)
            tasks.append(t); corpus += docs
    rng.shuffle(tasks)
    for i, t in enumerate(tasks):
        t["task_id"] = f"{split}_{t['category']}_{i:03d}"
        t["split"] = split
    ids = [d["doc_id"] for d in corpus]
    assert len(ids) == len(set(ids)), "duplicate doc ids"
    rng.shuffle(corpus)
    return tasks, corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--regenerate-unevaluated", action="store_true",
                        help="allowed only while NO model has been evaluated on any hard split; records a revision note")
    args = parser.parse_args()
    revision = 1
    if (DATA_DIR / "manifest_hard.json").exists():
        if not args.regenerate_unevaluated:
            raise SystemExit("hard splits already exist; refusing to regenerate (hard_final is locked)")
        revision = json.loads((DATA_DIR / "manifest_hard.json").read_text()).get("revision", 1) + 1
    rng, used = random.Random(args.seed), used_words()
    manifest = {"seed": args.seed, "revision": revision,
                "revision_note": None if revision == 1 else "regenerated before any model evaluation to make the stale_affiliation distractor outrank the gold record",
                "categories": CATEGORIES, "per_category": SIZES, "splits": {}}
    for split in SIZES:
        tasks, corpus = build(rng, used, split)
        write_jsonl(DATA_DIR / f"{split}_tasks.jsonl", tasks)
        write_jsonl(DATA_DIR / f"{split}_corpus.jsonl", corpus)
        manifest["splits"][split] = {"tasks": len(tasks), "corpus_docs": len(corpus),
                                     "tasks_sha256": file_hash(DATA_DIR / f"{split}_tasks.jsonl"),
                                     "corpus_sha256": file_hash(DATA_DIR / f"{split}_corpus.jsonl")}
    (DATA_DIR / "manifest_hard.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
