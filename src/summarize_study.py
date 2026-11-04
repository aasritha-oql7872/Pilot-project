"""Comparison table over evaluation reports (one row per condition x split), from real eval JSONs only.

    python -m src.summarize_study --out outputs/study/summary.md \
        "A one-hop GRPO=outputs/study/eval_twohop_dev_A.json" "C mixed, orig reward=..."
"""

import argparse
import json
from pathlib import Path

from src.compare import bootstrap_ci


def row(label, path):
    rep = json.loads(Path(path).read_text())
    eps = rep["episodes"]
    two = [e for e in eps if e["kind"] == "twohop"]
    need = [e for e in eps if e["kind"] != "no_search"]
    got = lambda e: {d for ids in e["retrieved_doc_ids"] for d in ids}
    frac = lambda xs, f: (sum(map(f, xs)) / len(xs)) if xs else None
    lo, hi = bootstrap_ci([float(e["correct"]) for e in eps])
    return {"condition": label, "split": rep["split"], "n": len(eps),
            "accuracy": frac(eps, lambda e: e["correct"]), "ci": (lo, hi),
            "first_evidence": frac(need, lambda e: e["evidence_doc_ids"][0] in got(e)),
            "complete_chain": frac(need, lambda e: set(e["evidence_doc_ids"]) <= got(e)),
            "second_search": frac(two, lambda e: e["num_searches"] >= 2),
            "premature_stop": frac(two, lambda e: e["num_searches"] == 1 and not set(e["evidence_doc_ids"]) <= got(e)),
            "use_second_record": frac([e for e in two if set(e["evidence_doc_ids"]) <= got(e)], lambda e: e["correct"]),
            "premature_answer": frac(need, lambda e: not set(e["evidence_doc_ids"]) <= got(e)),
            "decision_acc": rep["metrics"]["search_decision_accuracy"],
            "invalid": rep["metrics"]["invalid_rate"], "mean_searches": rep["metrics"]["mean_searches"],
            "mean_reward": rep["metrics"]["mean_reward"]}


def fmt(x, pct=True):
    return "–" if x is None else (f"{100 * x:.1f}%" if pct else f"{x:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entries", nargs="+", help='"label=path/to/eval.json"')
    parser.add_argument("--out", default=None)
    parser.add_argument("--compute", nargs="*", default=[], help="label=training run dir")
    args = parser.parse_args()
    rows = [row(*e.split("=", 1)) for e in args.entries]
    head = ("| Condition | Split | n | Accuracy [95% CI] | 1st evidence | Full chain | 2nd search | Premature stop "
            "| Correct given full chain | Decision acc | Invalid | Searches | Reward |\n|" + "---|" * 12)
    lines = [head]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['split']} | {r['n']} | {fmt(r['accuracy'])} "
                     f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}] | {fmt(r['first_evidence'])} | {fmt(r['complete_chain'])} | "
                     f"{fmt(r['second_search'])} | {fmt(r['premature_stop'])} | {fmt(r['use_second_record'])} | {fmt(r['decision_acc'])} | "
                     f"{fmt(r['invalid'])} | {fmt(r['mean_searches'], False)} | {fmt(r['mean_reward'], False)} |")
    text = "\n".join(lines)
    for entry in args.compute:  # label=run_dir -> training time / GPU memory from system_info.txt
        label, d = entry.split("=", 1)
        info = dict(l.split(": ", 1) for l in (Path(d) / "system_info.txt").read_text().splitlines() if ": " in l)
        text += f"\n* compute {label}: {info.get('train_seconds', '?')} s, peak VRAM {info.get('peak_vram_gb', '?')} GB, steps {info.get('optimizer_steps', '?')}"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
