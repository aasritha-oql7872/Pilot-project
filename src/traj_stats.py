"""Exploration and reward-variance statistics from a GRPO run's trajectories.jsonl (real rollouts only).

    python -m src.traj_stats outputs/study/F0 [more run dirs] --out summary.json

Rollouts are grouped by (step, task_id): the GRPO group. Reported separately for two-hop tasks:
  second_search / valid_second_search : rollouts with 2 executed searches (valid = no invalid action)
  nonzero_std_groups / orig_nonzero_std_groups : groups whose shaped / ORIGINAL reward has nonzero std
  bonus_created_variance : groups where the original reward is constant but the shaped reward is not
and for the controls: search frequency on lookup / no_search tasks (unnecessary-search check).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import pstdev

EPS = 1e-9


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def frac(rows, fn):
    return round(sum(map(bool, map(fn, rows))) / len(rows), 4) if rows else None


def window_rates(rows, size):
    """Second-search rate over consecutive blocks of `size` training steps (two-hop rollouts)."""
    if not rows:
        return []
    last = max(r["step"] for r in rows)
    out = []
    for lo in range(0, last + 1, size):
        block = [r for r in rows if lo <= r["step"] < lo + size]
        if block:
            out.append({"steps": f"{lo}-{lo + size - 1}", "n": len(block), "second_search": frac(block, lambda r: r["num_searches"] >= 2)})
    return out


def stats(rows, window=10):
    two = [r for r in rows if r["kind"] == "twohop"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["step"], r["task_id"])].append(r)

    def nonzero(g, key):
        return pstdev([x[key] for x in g]) > EPS if len(g) > 1 else False

    tg = [g for g in groups.values() if g[0]["kind"] == "twohop"]
    og = [g for g in groups.values() if g[0]["kind"] != "twohop"]
    got = lambda r: {d for ids in r["retrieved_doc_ids"] for d in ids}
    key_orig = "original_reward" if rows and "original_reward" in rows[0] else "reward"
    return {
        "rollouts": len(rows), "groups": len(groups), "steps": len({r["step"] for r in rows}),
        "twohop": {
            "rollouts": len(two),
            "searches_distribution": {k: frac(two, lambda r, k=k: r["num_searches"] == k) for k in (0, 1, 2)},
            "second_search": frac(two, lambda r: r["num_searches"] >= 2),
            "valid_second_search": frac(two, lambda r: r["num_searches"] >= 2 and r["num_invalid"] == 0),
            "first_evidence": frac(two, lambda r: r["evidence_doc_ids"][0] in got(r)),
            "complete_chain": frac(two, lambda r: set(r["evidence_doc_ids"]) <= got(r)),
            "distinct_docs_mean": round(sum(len(got(r)) for r in two) / len(two), 3) if two else None,
            "correct": frac(two, lambda r: r["correct"]),
            "mean_reward": round(sum(r["reward"] for r in two) / len(two), 4) if two else None,
            "second_search_by_window": window_rates(two, window),
            "groups": len(tg),
            "nonzero_std_groups": frac(tg, lambda g: nonzero(g, "reward")),
            "orig_nonzero_std_groups": frac(tg, lambda g: nonzero(g, key_orig)),
            "bonus_created_variance": frac(tg, lambda g: nonzero(g, "reward") and not nonzero(g, key_orig)),
        },
        "controls": {
            "lookup_search_rate": frac([r for r in rows if r["kind"] == "lookup"], lambda r: r["num_searches"] >= 1),
            "no_search_search_rate": frac([r for r in rows if r["kind"] == "no_search"], lambda r: r["num_searches"] >= 1),
            "lookup_second_search_rate": frac([r for r in rows if r["kind"] == "lookup"], lambda r: r["num_searches"] >= 2),
            "onehop_correct": frac([r for r in rows if r["kind"] != "twohop"], lambda r: r["correct"]),
            "nonzero_std_groups_onehop": frac(og, lambda g: nonzero(g, "reward")),
        },
        "invalid_rate": frac(rows, lambda r: r["num_invalid"] > 0),
        "all_groups_nonzero_std": frac(list(groups.values()), lambda g: nonzero(g, "reward")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run directories containing trajectories.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()
    result = {Path(d).name: stats(load(Path(d) / "trajectories.jsonl"), args.window) for d in args.runs}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
