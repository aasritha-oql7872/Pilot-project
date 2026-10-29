"""Plot logged loss and reward vs optimizer step from a real run's train_metrics.csv.

    python -m src.plot_metrics --run outputs/main --output outputs/main/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

PLOTS = [  # (column, y label, file name)
    ("loss", "GRPO policy loss", "loss_vs_step.png"),
    ("reward", "mean episode reward", "reward_vs_step.png"),
    ("searchlite/correct", "fraction of rollouts correct", "correct_vs_step.png"),
    ("searchlite/searches_per_episode", "searches per episode", "searches_vs_step.png"),
]


def load_metrics(run_dir):
    path = Path(run_dir) / "train_metrics.csv"
    if not path.exists():
        sys.exit(f"error: {path} not found; plots are only made from logged training runs.")
    df = pd.read_csv(path)
    if df.empty or "loss" not in df or "reward" not in df:
        sys.exit(f"error: {path} has no logged loss/reward rows.")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = load_metrics(args.run)
    cfg = yaml.safe_load((Path(args.run) / "config_snapshot.yaml").read_text())
    tag = f"{cfg['model']} | LoRA r={cfg['lora']['r']} | seed {cfg['seed']} | run {Path(args.run).name}"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for column, ylabel, name in PLOTS:
        if column not in df:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["step"], df[column], marker="o", markersize=3, linewidth=1)
        if column == "reward" and "reward_std" in df:  # within-batch spread, unsmoothed
            ax.fill_between(df["step"], df[column] - df["reward_std"], df[column] + df["reward_std"],
                            alpha=0.2, label="± reward std (batch)")
            ax.legend()
        ax.set_xlabel("optimizer step (1 step = one GRPO update)")
        ax.set_ylabel(ylabel)
        ax.set_title(tag, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
