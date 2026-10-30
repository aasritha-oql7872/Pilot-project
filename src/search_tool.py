"""Deterministic local BM25 retriever over one split's corpus."""

import re

from rank_bm25 import BM25Okapi

from data.generate import DATA_DIR, read_jsonl

TOP_K = 2
MAX_CHARS = 300
STOPWORDS = {"the", "a", "an", "of", "is", "in", "on", "by", "for", "with", "and", "its",
             "who", "what", "which", "where", "when", "does", "did", "was", "as", "to", "that"}


def stem(token):
    """Crude suffix stripping so funds/funded/funding all match."""
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text):
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

    @classmethod
    def for_split(cls, split):
        """`split` may join several splits with "+" (mixed training); doc ids are globally unique."""
        return cls([d for name in split.split("+") for d in read_jsonl(DATA_DIR / f"{name}_corpus.jsonl")])

    def search(self, query, k=TOP_K):
        """Top-k docs by BM25 score; ties broken by corpus order so results are stable."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(self.docs)), key=lambda i: (-scores[i], i))
        return [{"doc_id": self.docs[i]["doc_id"], "text": self.docs[i]["text"][:MAX_CHARS]}
                for i in order[:k] if scores[i] > 0]


def format_results(results):
    if not results:
        return "No matching records."
    return "\n".join(f"[{r['doc_id']}] {r['text']}" for r in results)
