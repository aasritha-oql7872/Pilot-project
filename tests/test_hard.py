import json
import re
from collections import Counter

import pytest

from data.generate import DATA_DIR, read_jsonl
from data.make_hard import CATEGORIES, SIZES
from src.manifest import file_hash
from src.reward import ABSTAIN_ANSWER
from src.validate_hard import SPLITS, entity_words, validate_split

OTHER = ["train", "dev", "test", "twohop", "twohop_train", "twohop_dev", "final", "twohop_final"]


def test_category_counts_and_schema():
    for split in SPLITS:
        tasks = read_jsonl(DATA_DIR / f"{split}_tasks.jsonl")
        assert Counter(t["category"] for t in tasks) == {c: SIZES[split] for c in CATEGORIES}
        assert all(t["split"] == split and t["task_id"].startswith(split) for t in tasks)
        assert all((t["kind"] == "unanswerable") == (not t["answerable"]) for t in tasks)
        assert all(t["gold"] == ABSTAIN_ANSWER for t in tasks if not t["answerable"])
        assert all(len(t["evidence_doc_ids"]) == (1 if not t["answerable"] else 2) for t in tasks)


@pytest.mark.parametrize("split", SPLITS)
def test_ground_truth_is_derivable_from_observed_text_in_two_searches(split):
    for cat, r in validate_split(split).items():
        assert r["oracle_correct"] == r["n"] and r["evidence_reachable"] == r["n"], (split, cat, r)


def test_hard_splits_are_disjoint_from_each_other_and_from_every_other_split():
    ents = {s: entity_words(s) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:] + OTHER:
            assert not (ents[a] & (ents[b] if b in ents else entity_words(b))), (a, b)


def test_no_duplicate_questions_or_golds_leak_into_questions():
    qs = [t["question"] for s in SPLITS for t in read_jsonl(DATA_DIR / f"{s}_tasks.jsonl")]
    assert len(qs) == len(set(qs))
    for s in SPLITS:
        for t in read_jsonl(DATA_DIR / f"{s}_tasks.jsonl"):
            if t["answerable"]:
                assert t["gold"].lower() not in t["question"].lower()


def test_decoys_exist_and_are_never_the_gold_evidence():
    for s in SPLITS:
        corpus = {d["doc_id"] for d in read_jsonl(DATA_DIR / f"{s}_corpus.jsonl")}
        for t in read_jsonl(DATA_DIR / f"{s}_tasks.jsonl"):
            assert set(t["evidence_doc_ids"]) <= corpus and set(t["decoy_doc_ids"]) <= corpus
            assert not set(t["decoy_doc_ids"]) & set(t["evidence_doc_ids"])
            if t["category"] in ("distractor_name", "distractor_person", "stale_affiliation"):
                assert t["decoy_doc_ids"]


def test_locked_files_match_manifest():
    m = json.loads((DATA_DIR / "manifest_hard.json").read_text())["splits"]
    for s in SPLITS:
        assert file_hash(DATA_DIR / f"{s}_tasks.jsonl") == m[s]["tasks_sha256"]
        assert file_hash(DATA_DIR / f"{s}_corpus.jsonl") == m[s]["corpus_sha256"]


def test_stale_affiliation_record_outranks_the_gold_record():
    from src.search_tool import Retriever
    from src.validate_hard import person_of_project
    r = Retriever.for_split("hard_dev")
    for t in read_jsonl(DATA_DIR / "hard_dev_tasks.jsonl"):
        if t["category"] == "stale_affiliation":
            person, _ = person_of_project(r.search(f"Project {t['subject']} principal investigator"), t["subject"])
            top = r.search(f"{person} affiliation")
            assert top[0]["doc_id"] in t["decoy_doc_ids"] and t["evidence_doc_ids"][1] in [d["doc_id"] for d in top]
