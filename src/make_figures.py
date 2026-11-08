"""Training and evaluation figures drawn ONLY from saved logs/evaluations. Missing inputs are skipped and listed, never invented.

    python -m src.make_figures --out submission/figures --original outputs/main_v2 --study outputs/study_hard \
        --splits hard_dev hard_final

Smoothed lines are labelled "moving avg"; raw per-step values are always drawn too.
Why GRPO policy loss is not a supervised cross-entropy loss: with beta=0 the loss is the mean over tokens of
-(clipped importance ratio x advantage). Advantages are group-normalised, so they sum to ~0 within a group; at the first
gradient step the ratio is 1 and the loss is ~0 by construction. It can be positive or negative, a near-zero value
(especially on steps where every rollout in a group got the same reward and the advantage is exactly 0) says nothing
about convergence. Use reward, accuracy, the share of groups with nonzero reward std, and behaviour metrics instead.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.report_categories import load
from src.summarize_study import wilson

LOSS_NOTE = ("GRPO policy loss is not a supervised cross-entropy: it is -(ratio x group-normalised advantage), is ~0 at the first "
             "gradient step by construction and exactly 0 on steps where all rollouts in a group tie. Its level does not measure convergence.")
COLORS = {"M2": "#555555", "M3": "#1f77b4", "M4": "#d62728", "M4g": "#2ca02c", "A": "#9467bd"}
MODEL = re.compile(r"^(M2|M3|M4g|M4)(_s(\d+))?$")


def smooth(s, w=5):
    return s.rolling(w, min_periods=1, center=True).mean()


def curve(ax, df, col, label, color, ls="-"):
    if col not in df or df[col].isna().all():
        return False
    ax.plot(df["step"], df[col], color=color, alpha=0.25, lw=0.8, ls=ls)
    ax.plot(df["step"], smooth(df[col]), color=color, lw=1.8, ls=ls, label=f"{label} (moving avg, w=5)")
    return True


def metrics(run):
    p = Path(run) / "train_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def fam(label):
    return MODEL.match(label.split()[0]).group(1) if MODEL.match(label.split()[0]) else "A"


def panel_runs(runs, cols, titles, path, suptitle, note=None):
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4), squeeze=False)
    drawn = False
    for ax, col, title in zip(axes[0], cols, titles):
        for label, d in runs.items():
            df = metrics(d)
            if df is not None:
                drawn |= curve(ax, df, col, label, COLORS[fam(label)], "-" if label.endswith("42") or "_s" not in label else "--")
        ax.set_title(title, fontsize=10); ax.set_xlabel("GRPO optimizer step"); ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=7, wrap=True)
    fig.tight_layout()
    if drawn:
        fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return drawn


def acc_table(study, split, labels):
    out = {}
    for lab in labels:
        p = Path(study) / f"eval_{split}_{lab}.json"
        if p.exists():
            eps, _ = load(p)
            out[lab] = eps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--original", default="outputs/main_v2", help="run dir of the original (RL-only) GRPO, condition A")
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--splits", nargs="+", default=["hard_dev", "hard_final"])
    ap.add_argument("--seeds", nargs="+", default=["42", "43"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S = Path(args.study)
    made, skipped = [], []
    grpo = {f"{v}_s{sd}": S / f"{v}_s{sd}" for v in ("M3", "M4", "M4g") for sd in args.seeds if (S / f"{v}_s{sd}" / "train_metrics.csv").exists()}
    orig = {"A (original GRPO, no SFT)": Path(args.original)} if (Path(args.original) / "train_metrics.csv").exists() else {}

    jobs = [("fig1_original_grpo_reward_loss.png", orig, ["reward", "loss"], ["mean episode reward", "GRPO policy loss"],
             "Original GRPO from the base model (no SFT)", LOSS_NOTE),
            ("fig2_m2_grpo_reward_loss.png", grpo, ["reward", "loss"], ["mean episode reward", "GRPO policy loss"],
             "GRPO initialised from the SFT checkpoint M2 (M3 original / M4 evidence / M4g grounded reward)", LOSS_NOTE),
            ("fig3_reward_variation.png", {**orig, **grpo}, ["reward_std", "frac_reward_zero_std", "grad_norm"],
             ["reward std within the batch", "share of groups with zero reward std (no gradient)", "gradient norm"],
             "Reward variation and learning signal during training", "A group whose rollouts all tie gives zero advantage and zero gradient."),
            ("fig4_search_frequency.png", {**orig, **grpo}, ["searchlite/searches_per_episode", "searchlite/twohop_searches"],
             ["searches per episode (all training tasks)", "searches per episode (two-hop tasks only)"],
             "Search behaviour during training", None)]
    for name, runs, cols, titles, sup, note in jobs:
        if runs and panel_runs(runs, cols, titles, out / name, sup, note):
            made.append(name)
        else:
            skipped.append(f"{name}: no train_metrics.csv found")

    mlabels = ["M2"] + [f"{v}_s{sd}" for v in ("M3", "M4", "M4g") for sd in args.seeds]
    for split in args.splits:  # accuracy with Wilson intervals
        t = acc_table(S, split, mlabels)
        if len(t) < 2:
            skipped.append(f"accuracy on {split}: fewer than two evaluations"); continue
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for i, (lab, eps) in enumerate(t.items()):
            k, n = sum(bool(e["correct"]) for e in eps.values()), len(eps)
            lo, hi = wilson(k, n)
            ax.bar(i, 100 * k / n, color=COLORS[fam(lab)], alpha=0.85)
            ax.errorbar(i, 100 * k / n, yerr=[[max(0.0, 100 * k / n - 100 * lo)], [max(0.0, 100 * hi - 100 * k / n)]], color="k", capsize=3)
            ax.text(i, 100 * hi + 1, f"{k}/{n}", ha="center", fontsize=7)
        ax.set_xticks(range(len(t))); ax.set_xticklabels(list(t), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 112); ax.set_title(f"{split}: accuracy with 95% Wilson intervals", fontsize=10); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"fig5_accuracy_{split}.png", dpi=170); plt.close(fig); made.append(f"fig5_accuracy_{split}.png")

        cats = sorted({e.get("category") for eps in t.values() for e in eps.values() if e.get("category")})
        fams = {}
        for lab, eps in t.items():
            fams.setdefault(fam(lab), []).append(eps)
        fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.8 / len(fams)
        for j, (f, runs) in enumerate(fams.items()):
            vals = []
            for c in cats:
                accs = [sum(bool(e["correct"]) for e in r.values() if e.get("category") == c) / max(1, sum(e.get("category") == c for e in r.values())) for r in runs]
                vals.append(100 * sum(accs) / len(accs))
            ax.bar([i + j * w for i in range(len(cats))], vals, w, label=f"{f} (mean of {len(runs)} run{'s' * (len(runs) > 1)})", color=COLORS[f])
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(cats))]); ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("accuracy (%)"); ax.set_title(f"{split}: accuracy by category (n per category = {max(sum(e.get('category') == c for e in next(iter(t.values())).values()) for c in cats)})", fontsize=10)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig(out / f"fig6_per_category_{split}.png", dpi=170); plt.close(fig)
        made.append(f"fig6_per_category_{split}.png")

        fig, ax = plt.subplots(figsize=(5, 3.6))
        for i, v in enumerate(("M2", "M3", "M4", "M4g")):
            pts = [(lab, 100 * sum(bool(e["correct"]) for e in eps.values()) / len(eps)) for lab, eps in t.items() if fam(lab) == v]
            ax.scatter([i] * len(pts), [p[1] for p in pts], color=COLORS[v], s=40, zorder=3)
            for lab, y in pts:
                ax.annotate(lab.split("_s")[-1] if "_s" in lab else "", (i, y), textcoords="offset points", xytext=(6, -3), fontsize=7)
        ax.set_xticks(range(4)); ax.set_xticklabels(["M2", "M3", "M4", "M4g"]); ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{split}: each point is one training seed", fontsize=10); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"fig7_seeds_{split}.png", dpi=170); plt.close(fig); made.append(f"fig7_seeds_{split}.png")

    cap = ["# Figure captions", "", LOSS_NOTE, "", "Smoothed curves are centred moving averages (window 5) drawn over the raw per-step values.", ""]
    cap += [f"- `{m}`" for m in made] + ["", "Skipped (inputs missing):"] + [f"- {s}" for s in skipped]
    (out / "figure_captions.md").write_text("\n".join(cap) + "\n")
    print("made:", *made, sep="\n  "); print("skipped:", *skipped, sep="\n  ")


if __name__ == "__main__":
    main()
