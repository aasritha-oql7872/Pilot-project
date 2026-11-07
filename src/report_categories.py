"""Per-category evaluation report and paired model comparison on the hard benchmark.

    python -m src.report_categories --out summary.md --reference "M2 SFT=eval_hard_dev_M2.json" \
        "M2 SFT=eval_hard_dev_M2.json" "M3 SFT+GRPO=eval_hard_dev_M3.json" "M3 sampled=a.json,b.json,c.json"

Rows come only from saved evaluation JSONs. A label with several comma-separated files is a repeated stochastic
evaluation: accuracy is reported as the mean over runs (+/- sd over runs); k/n are the pooled counts and the pooled
interval ignores that repeats of one task are not independent, so only the per-run mean is used for comparisons.

Metrics (per category and overall):
  accuracy               exact final answer (the abstention string counts as the gold answer of unanswerable tasks)
  evidence-supported     correct AND every required evidence document was retrieved
  evidence recall        mean fraction of the required documents that were retrieved
  unsupported answer     a non-abstaining answer found neither in the question nor in any retrieved record
  false abstention       abstained on an answerable task
  searches / invalid     mean executed searches; share of episodes with an invalid action
Paired comparison against --reference: exact two-sided sign test on discordant tasks, and a paired bootstrap 95% CI
for the accuracy difference (same task ids required).
"""

import argparse
import json
import random
import statistics
from math import comb
from pathlib import Path

from src.reward import ABSTAIN_ANSWER, normalize
from src.summarize_study import wilson


def retrieved_texts(ep):
    """Record texts shown to the agent (tool messages without the trailing answer-format reminder)."""
    return [m["content"].split("\n\n")[0] for m in ep["messages"] if m.get("role") == "tool"]


def episode_metrics(ep):
    got = {d for ids in ep["retrieved_doc_ids"] for d in ids}
    req = ep["evidence_doc_ids"]
    ans = ep["final_answer"]
    abstained = ans is not None and normalize(ans) == normalize(ABSTAIN_ANSWER)
    question = next((m["content"] for m in ep["messages"] if m.get("role") == "user"), "")
    hay = normalize(" ".join([question] + retrieved_texts(ep)))
    supported = ans is not None and bool(normalize(ans)) and normalize(ans) in hay
    answerable = ep["kind"] != "unanswerable"
    return {"correct": bool(ep["correct"]),
            "evidence_supported": bool(ep["correct"]) and set(req) <= got,
            "recall": (len(got & set(req)) / len(req)) if req else None,
            "unsupported": ans is not None and not abstained and not supported,
            "false_abstain": abstained and answerable,
            "searches": ep["num_searches"], "invalid": ep["num_invalid"] > 0}


def load(path):
    rep = json.loads(Path(path).read_text())
    return {e["task_id"]: e for e in rep["episodes"]}, rep


def agg(eps, key, cats=None):
    rows = [episode_metrics(e) for e in eps if cats is None or e.get("category") in cats]
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2**n)


def paired_boot(a, b, n_boot=5000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    ms = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


def fmt(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entries", nargs="+", help='"label=path.json" or "label=a.json,b.json,c.json"')
    parser.add_argument("--reference", default=None, help='"label=path.json" for paired comparisons')
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    models = {}
    for e in args.entries:
        label, paths = e.split("=", 1)
        runs = [load(p)[0] for p in paths.split(",")]
        models[label] = runs
    cats = sorted({c for runs in models.values() for r in runs for e in r.values() for c in [e.get("category")] if c})
    lines = ["| Model | Category | k/n | Accuracy [95% Wilson] | Evidence-supported | Evidence recall | Unsupported answers | False abstention | Searches | Invalid |",
             "|" + "---|" * 10]
    for label, runs in models.items():
        for cat in ["ALL"] + cats:
            per_run = [agg(list(r.values()), None, None if cat == "ALL" else {cat}) for r in runs]
            per_run = [rows for rows in per_run if rows]
            if not per_run:
                continue
            k = sum(sum(x["correct"] for x in rows) for rows in per_run)
            n = sum(len(rows) for rows in per_run)
            lo, hi = wilson(k, n)
            accs = [mean([x["correct"] for x in rows]) for rows in per_run]
            acc_txt = fmt(mean(accs)) + (f" ± {100 * statistics.stdev(accs):.1f} (sd over {len(accs)} runs)" if len(accs) > 1 else "")
            f = lambda key: fmt(mean([mean([x[key] for x in rows]) for rows in per_run]))
            lines.append(f"| {label} | {cat} | {k}/{n} | {acc_txt} [{fmt(lo)}, {fmt(hi)}] | {f('evidence_supported')} | {f('recall')} | "
                         f"{f('unsupported')} | {f('false_abstain')} | {mean([mean([x['searches'] for x in rows]) for rows in per_run]):.2f} | {f('invalid')} |")
    text = "\n".join(lines)
    if args.reference:
        rl, rp = args.reference.split("=", 1)
        ref = load(rp)[0]
        text += f"\n\nPaired comparison against **{rl}** (accuracy; exact sign test on discordant tasks; paired bootstrap CI of the difference)\n"
        text += "\n| Model | Category | model-only correct | reference-only correct | difference [95% CI] | sign-test p |\n|---|---|---|---|---|---|"
        for label, runs in models.items():
            if len(runs) != 1 or label == rl:
                continue
            run = runs[0]
            if set(run) != set(ref):
                text += f"\n| {label} | – | task ids differ from the reference: no paired comparison | | | |"
                continue
            for cat in ["ALL"] + cats:
                ids = [i for i in ref if cat == "ALL" or ref[i].get("category") == cat]
                a = [float(run[i]["correct"]) for i in ids]; b = [float(ref[i]["correct"]) for i in ids]
                w = sum(x > y for x, y in zip(a, b)); l = sum(x < y for x, y in zip(a, b))
                lo, hi = paired_boot(a, b)
                text += f"\n| {label} | {cat} | {w} | {l} | {100 * (mean(a) - mean(b)):+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] | {sign_p(w, l):.3g} |"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
