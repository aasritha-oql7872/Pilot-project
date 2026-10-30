"""Leakage and split-boundary checks on the generated data."""

import json

import pytest

from data.generate import DATA_DIR, read_jsonl

SPLITS = ["train", "dev", "test"]
TASKS = {s: read_jsonl(DATA_DIR / f"{s}_tasks.jsonl") for s in SPLITS}
CORPUS = {s: read_jsonl(DATA_DIR / f"{s}_corpus.jsonl") for s in SPLITS}


def test_sizes_and_mix():
    assert len(TASKS["train"]) >= 200
    for split, tasks in TASKS.items():
        frac = sum(t["kind"] == "no_search" for t in tasks) / len(tasks)
        assert 0.35 <= frac <= 0.45, split


@pytest.mark.parametrize("split", SPLITS)
def test_lookup_gold_not_in_prompt(split):
    for t in TASKS[split]:
        if t["kind"] == "lookup":
            assert t["gold"].lower() not in t["question"].lower(), t["task_id"]


@pytest.mark.parametrize("split", SPLITS)
def test_no_search_fact_in_prompt_not_corpus(split):
    corpus_text = " ".join(d["text"] for d in CORPUS[split])
    for t in TASKS[split]:
        if t["kind"] == "no_search":
            assert t["gold"] in t["question"], t["task_id"]
            assert t["subject"] not in corpus_text, t["task_id"]  # entity is never indexed


@pytest.mark.parametrize("split", SPLITS)
def test_lookup_evidence_in_own_corpus(split):
    docs = {d["doc_id"]: d["text"] for d in CORPUS[split]}
    for t in TASKS[split]:
        for doc_id in t["evidence_doc_ids"]:
            assert t["gold"] in docs[doc_id], t["task_id"]


def test_splits_disjoint():
    ids = [t["task_id"] for s in SPLITS for t in TASKS[s]]
    assert len(ids) == len(set(ids))
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not {d["doc_id"] for d in CORPUS[a]} & {d["doc_id"] for d in CORPUS[b]}
                assert not {t["gold"] for t in TASKS[a]} & {t["gold"] for t in TASKS[b]} - _years()


def _years():
    return {str(y) for y in range(1985, 2026)}  # years are the only answers allowed to repeat


def test_model_visible_prompt_has_no_private_fields():
    from src.environment import make_prompt
    for t in TASKS["train"]:
        visible = json.dumps(make_prompt(t["question"]))
        assert t["task_id"] not in visible
        for doc_id in t["evidence_doc_ids"]:
            assert doc_id not in visible
        if t["kind"] == "lookup":
            assert t["gold"] not in visible


def test_twohop_split_is_isolated_and_answerable():
    tasks = read_jsonl(DATA_DIR / "twohop_tasks.jsonl")
    corpus = read_jsonl(DATA_DIR / "twohop_corpus.jsonl")
    docs = {d["doc_id"]: d["text"] for d in corpus}
    assert len(tasks) >= 50
    for t in tasks:
        assert t["gold"].lower() not in t["question"].lower()
        assert t["gold"] in docs[t["evidence_doc_ids"][1]]       # answer lives in the second hop only
        assert t["gold"] not in docs[t["evidence_doc_ids"][0]]
    train_test = {d["doc_id"] for s in SPLITS for d in CORPUS[s]}
    assert not {d["doc_id"] for d in corpus} & train_test
    assert not {t["gold"] for t in tasks} & {t["gold"] for s in SPLITS for t in TASKS[s]} - _years()
