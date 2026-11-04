"""Create fresh, never-inspected holdouts: `final` (100 one-hop/no-search) and `twohop_final` (100 two-hop).

Independent seed, and every name already used by an existing split is excluded, so no entity or document
overlaps with train/dev/test/twohop*. Existing files are never modified. Do not evaluate on these until the
method and hyperparameters are frozen; the hashes in manifest_sealed.json prove they were not edited.

    python -m data.make_sealed --seed 20261102
"""

import argparse
import json
import random
import re

from data.generate import CITY_SUFFIXES, DATA_DIR, build_split, build_twohop, file_hash, read_jsonl, write_jsonl

EXISTING = ["train", "dev", "test", "twohop", "twohop_train", "twohop_dev"]


def used_words():
    words = set()
    for split in EXISTING:
        for d in read_jsonl(DATA_DIR / f"{split}_corpus.jsonl"):
            words |= set(re.findall(r"[A-Z][a-z]+", d["text"]))
        for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl"):
            words |= set(re.findall(r"[A-Z][a-z]+", t["question"] + " " + t["gold"]))
    for w in list(words):  # cities are a name plus a suffix ("Melvelzanport"): also block the bare name
        for suffix in CITY_SUFFIXES:
            if w.endswith(suffix):
                words.add(w[: -len(suffix)])
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if (DATA_DIR / "manifest_sealed.json").exists():
        raise SystemExit("sealed holdouts already exist; refusing to regenerate")
    rng, used = random.Random(args.seed), used_words()
    manifest = {"seed": args.seed, "splits": {}}
    for split, builder in [("final", lambda: build_split(rng, used, "final", 100)),
                           ("twohop_final", lambda: build_twohop(rng, used, "twohop_final", 100))]:
        tasks, corpus = builder()
        write_jsonl(DATA_DIR / f"{split}_tasks.jsonl", tasks)
        write_jsonl(DATA_DIR / f"{split}_corpus.jsonl", corpus)
        manifest["splits"][split] = {"tasks": len(tasks), "corpus_docs": len(corpus),
                                     "tasks_sha256": file_hash(DATA_DIR / f"{split}_tasks.jsonl"),
                                     "corpus_sha256": file_hash(DATA_DIR / f"{split}_corpus.jsonl")}
    (DATA_DIR / "manifest_sealed.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
