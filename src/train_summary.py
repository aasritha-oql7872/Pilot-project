"""Training-diagnostics table from each run's own train_metrics.csv (raw logged values, no smoothing).

    python -m src.train_summary --out train_summary.md "A original GRPO=outputs/main_v2" "M3 s42=outputs/study_hard/M3_s42"

Early = first third of logged steps, late = last third. A change in training reward between thirds is NOT evidence of
generalisation (the training prompts repeat across epochs); it is reported next to the held-out accuracies, never instead of them.
"""

import argparse
from pathlib import Path

import pandas as pd

COLS = [("reward", "reward"), ("searchlite/correct", "train correct"), ("searchlite/searches_per_episode", "searches/ep"),
        ("searchlite/twohop_searches", "two-hop searches"), ("entropy", "entropy")]


def thirds(s):
    s = s.dropna()
    if len(s) < 3:
        return None, None
    k = max(1, len(s) // 3)
    return s.iloc[:k].mean(), s.iloc[-k:].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+", help='"label=run_dir"')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    head = ["Run", "logged steps"] + [f"{n} early→late" for _, n in COLS] + ["grad_norm mean", "steps with gradient", "zero-std groups (mean)", "loss min / max"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in args.entries:
        label, d = e.split("=", 1)
        p = Path(d) / "train_metrics.csv"
        if not p.exists():
            lines.append(f"| {label} | missing {p} |" + " |" * (len(head) - 2)); continue
        df = pd.read_csv(p)
        cells = []
        for c, _ in COLS:
            a, b = thirds(df[c]) if c in df else (None, None)
            cells.append("–" if a is None else f"{a:.3f} → {b:.3f}")
        g = df["grad_norm"] if "grad_norm" in df else None
        z = df["frac_reward_zero_std"] if "frac_reward_zero_std" in df else None
        cells += [f"{g.mean():.3f}" if g is not None else "–", f"{(g > 0).mean():.0%}" if g is not None else "–",
                  f"{z.mean():.0%}" if z is not None else "–", f"{df['loss'].min():.3f} / {df['loss'].max():.3f}"]
        lines.append(f"| {label} | {len(df)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
