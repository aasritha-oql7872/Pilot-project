from data.generate import DATA_DIR, read_jsonl
from src.search_tool import MAX_CHARS, TOP_K, Retriever, format_results


def test_deterministic_and_bounded():
    r = Retriever.for_split("train")
    first = r.search("Project principal investigator")
    assert first == r.search("Project principal investigator")
    assert first == Retriever.for_split("train").search("Project principal investigator")
    assert len(first) <= TOP_K and all(len(d["text"]) <= MAX_CHARS for d in first)


def test_empty_and_unmatched_queries():
    r = Retriever.for_split("train")
    assert r.search("") == [] and r.search("   ?! the of") == []
    assert r.search("zzzqqq") == []
    assert format_results([]) == "No matching records."


def test_oracle_recall_at_2():
    """Using the question itself as the query must surface the gold doc >= 95% of the time."""
    for split in ["train", "dev", "test"]:
        r = Retriever.for_split(split)
        lookup = [t for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl") if t["kind"] == "lookup"]
        hits = sum(t["evidence_doc_ids"][0] in [d["doc_id"] for d in r.search(t["question"])] for t in lookup)
        assert hits / len(lookup) >= 0.95, (split, hits, len(lookup))


def test_train_index_cannot_reach_test_records():
    train = Retriever.for_split("train")
    test_ids = {d["doc_id"] for d in read_jsonl(DATA_DIR / "test_corpus.jsonl")}
    for t in read_jsonl(DATA_DIR / "test_tasks.jsonl"):
        assert not {d["doc_id"] for d in train.search(t["question"])} & test_ids
