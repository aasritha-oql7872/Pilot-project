"""Compare a base evaluation with one or more trained evaluations on the same tasks.

    python -m src.compare --base outputs/main_v2/eval_base.json \
        --trained outputs/main_v2/eval_trained.json outputs/seed43/eval_trained.json
"""

import argparse
import json
import random
from math import comb
from statistics import mean, stdev


def load(path):
    episodes = json.load(open(path))["episodes"]
    return {e["task_id"]: e for e in episodes}


def bootstrap_ci(values, n_boot=10_000, seed=0):
    """95% percentile bootstrap CI of the mean over tasks."""
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def sign_test_p(wins, losses):
    """Two-sided exact McNemar / sign test on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--trained", nargs="+", required=True)
    args = parser.parse_args()

    base = load(args.base)
    def acc(run, ids, kind=None):
        return [float(run[i]["correct"]) for i in ids if kind in (None, run[i]["kind"])]

    print(f"{'run':38} {'n':>4} {'accuracy [95% CI]':>24}   vs base: wins/losses  p")
    lo, hi = bootstrap_ci(acc(base, base))
    print(f"{args.base:38} {len(base):>4} {mean(acc(base, base)):>8.3f} [{lo:.3f}, {hi:.3f}]")
    overall = []
    for path in args.trained:
        run = load(path)
        ids = sorted(set(base) & set(run))
        if len(ids) != len(base) or len(ids) != len(run):
            raise SystemExit(f"{path}: task IDs differ from base; refusing an unpaired comparison.")
        a = acc(run, ids)
        lo, hi = bootstrap_ci(a)
        wins = sum(run[i]["correct"] and not base[i]["correct"] for i in ids)
        losses = sum(base[i]["correct"] and not run[i]["correct"] for i in ids)
        overall.append(mean(a))
        print(f"{path:38} {len(ids):>4} {mean(a):>8.3f} [{lo:.3f}, {hi:.3f}]   {wins:>4}/{losses:<4}  {sign_test_p(wins, losses):.2g}")
        for kind in sorted({e['kind'] for e in run.values()}):
            ka = acc(run, ids, kind); kb = acc(base, ids, kind)
            print(f"    {kind:10} base {mean(kb):.3f} -> trained {mean(ka):.3f}  (n={len(ka)})")
    if len(overall) > 1:
        print(f"\ntrained accuracy across {len(overall)} runs: mean {mean(overall):.3f}, sd {stdev(overall):.3f}, "
              f"min {min(overall):.3f}, max {max(overall):.3f}")


if __name__ == "__main__":
    main()
