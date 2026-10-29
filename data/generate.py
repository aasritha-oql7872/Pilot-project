"""Generate the fictional SearchLite knowledge base and question splits.

Every split gets its own entities, its own corpus and its own questions, so train
search can never reach dev/test facts. Gold answers live only in task metadata.

    python -m data.generate --seed 42 --train 240 --dev 60 --test 100
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
NO_SEARCH_FRACTION = 0.4
QUESTIONS_PER_PROJECT = 3

FIRST_NAMES = ["Maya", "Ravi", "Lena", "Omar", "Iris", "Theo", "Nadia", "Felix", "Priya", "Jonas",
               "Elena", "Kofi", "Sana", "Marco", "Yuki", "Aria", "Dmitri", "Leila", "Tomas", "Zara"]
SYLLABLES = ["vel", "mor", "an", "tri", "ka", "lo", "ser", "dun", "bra", "quin", "eth", "ol",
             "zan", "pir", "cor", "mel", "tav", "ris", "gal", "ond", "bel", "fen", "thar", "ux"]
ORG_SUFFIXES = ["University", "Institute", "Foundation", "Council", "Laboratory"]
CITY_SUFFIXES = ["haven", "ford", "mouth", "field", "port"]

# Corpus phrasings: two per attribute, each covering the vocabulary of every question
# template for that attribute so a lexical retriever can match them. {p}=subject, {v}=value.
DOC_TEMPLATES = {
    "pi": ["Project {p} principal investigator is {v}, who leads the project as lead researcher.",
           "{v} leads Project {p} as its principal investigator and lead researcher."],
    "year": ["Project {p} started in {v}, the year it was launched.",
             "Project {p} was launched in {v}, the year work on it started."],
    "funder": ["Project {p} is funded by the {v}, the agency that provides its funding.",
               "Funding for Project {p} is provided by the {v}, its funding agency."],
    "city": ["Project {p} is based in the city of {v}, where it is located.",
             "Project {p} is located in {v}, the city where the project is based."],
    "affiliation": ["{p} is affiliated with {v}, the institution where they work.",
                    "{p} works at {v} and is affiliated with that institution."],
}
LOOKUP_QUESTIONS = {
    "pi": ["Who is the principal investigator of Project {p}?", "Who leads Project {p}?",
           "Name the lead researcher on Project {p}."],
    "year": ["In what year did Project {p} start?", "When was Project {p} launched?"],
    "funder": ["Which agency funds Project {p}?", "Who provides funding for Project {p}?"],
    "city": ["In which city is Project {p} based?", "Where is Project {p} located?"],
    "affiliation": ["Which institution is {p} affiliated with?", "Where does {p} work?"],
}
# No-search questions state the fact in the prompt, then ask for it.
NO_SEARCH_QUESTIONS = {
    "pi": "The archive says Project {p} is led by {v}. Who leads Project {p}?",
    "year": "The archive says Project {p} started in {v}. What year did Project {p} start?",
    "funder": "The archive says Project {p} is funded by the {v}. Which agency funds Project {p}?",
    "city": "The archive says Project {p} is based in {v}. In which city is Project {p} based?",
    "affiliation": "The archive says {p} is affiliated with {v}. Which institution is {p} affiliated with?",
}


def make_word(rng, used, n_syllables=3):
    """A fresh capitalized pseudo-word, unique across the whole generation run."""
    for _ in range(10_000):
        word = "".join(rng.choice(SYLLABLES) for _ in range(n_syllables)).capitalize()
        if word not in used:
            used.add(word)
            return word
    raise RuntimeError(f"name pool for {n_syllables}-syllable words exhausted")


def make_project(rng, used):
    """One fictional project with its PI (who has an affiliation)."""
    return {
        "name": make_word(rng, used),
        "pi": f"{rng.choice(FIRST_NAMES)} {make_word(rng, used, 2)}",
        "year": str(rng.randint(1985, 2025)),
        "funder": f"{make_word(rng, used)} {rng.choice(ORG_SUFFIXES)}",
        "city": make_word(rng, used) + rng.choice(CITY_SUFFIXES),
        "affiliation": f"{make_word(rng, used)} {rng.choice(ORG_SUFFIXES)}",
    }


def subject_and_value(project, attr):
    """Affiliation is a fact about the PI; every other attribute is about the project."""
    return (project["pi"] if attr == "affiliation" else project["name"]), project[attr]


def build_corpus(rng, projects):
    docs = []
    for proj in projects:
        for attr, templates in DOC_TEMPLATES.items():
            subj, value = subject_and_value(proj, attr)
            docs.append({"doc_id": f"{proj['name'].lower()}_{attr}",
                         "text": rng.choice(templates).format(p=subj, v=value)})
    return docs


def build_split(rng, used, split, n_tasks):
    n_no_search = round(n_tasks * NO_SEARCH_FRACTION)
    n_lookup = n_tasks - n_no_search
    n_projects = -(-n_lookup // QUESTIONS_PER_PROJECT)  # ceil
    projects = [make_project(rng, used) for _ in range(n_projects)]
    corpus = build_corpus(rng, projects)

    tasks = []
    for proj in projects:
        for attr in rng.sample(list(LOOKUP_QUESTIONS), QUESTIONS_PER_PROJECT):
            subj, value = subject_and_value(proj, attr)
            tasks.append({"kind": "lookup", "attribute": attr, "subject": subj, "gold": value,
                          "question": rng.choice(LOOKUP_QUESTIONS[attr]).format(p=subj),
                          "evidence_doc_ids": [f"{proj['name'].lower()}_{attr}"]})
    tasks = tasks[:n_lookup]

    # No-search entities are never indexed: the prompt is the only source of the fact.
    for _ in range(n_no_search):
        proj = make_project(rng, used)
        attr = rng.choice(list(NO_SEARCH_QUESTIONS))
        subj, value = subject_and_value(proj, attr)
        tasks.append({"kind": "no_search", "attribute": attr, "subject": subj, "gold": value,
                      "question": NO_SEARCH_QUESTIONS[attr].format(p=subj, v=value),
                      "evidence_doc_ids": []})

    rng.shuffle(tasks)
    for i, task in enumerate(tasks):
        task["task_id"] = f"{split}_{task['kind']}_{i:03d}"
        task["split"] = split
    return tasks, corpus


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--dev", type=int, default=60)
    parser.add_argument("--test", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    used = set()
    manifest = {"seed": args.seed, "splits": {}}
    for split, n in [("train", args.train), ("dev", args.dev), ("test", args.test)]:
        tasks, corpus = build_split(rng, used, split, n)
        write_jsonl(DATA_DIR / f"{split}_tasks.jsonl", tasks)
        write_jsonl(DATA_DIR / f"{split}_corpus.jsonl", corpus)
        manifest["splits"][split] = {
            "tasks": len(tasks),
            "no_search": sum(t["kind"] == "no_search" for t in tasks),
            "lookup": sum(t["kind"] == "lookup" for t in tasks),
            "corpus_docs": len(corpus),
            "corpus_sha256": file_hash(DATA_DIR / f"{split}_corpus.jsonl"),
        }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
