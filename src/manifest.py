"""Experiment manifest: one table mapping run -> initialization, reward, seed, checkpoint, decoding, split.

    python -m src.manifest --roots outputs/study_sft outputs/study_early outputs/study_rl_only --out outputs/experiment_manifest.json
    python -m src.manifest --verify-sealed        # exit 1 if a sealed holdout file differs from data/manifest_sealed.json

Built only from files the runs themselves wrote (config_snapshot.yaml, eval_*.json); nothing is filled in by hand.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from data.generate import DATA_DIR, file_hash


def run_entry(d):
    cfg = yaml.safe_load((d / "config_snapshot.yaml").read_text())
    g, sft = cfg.get("grpo"), cfg.get("sft")
    return {"run": str(d), "kind": "sft" if sft else "grpo", "model": cfg.get("model"), "seed": cfg.get("seed"),
            "init_adapter": cfg.get("init_adapter"), "reward": (cfg.get("reward") or {}).get("mode", "original") if g else None,
            "reward_params": cfg.get("reward") if g else None, "train_splits": cfg.get("train_splits", ["train", "twohop_train"] if sft else None),
            "answer_format": cfg.get("answer_format", "default"), "grpo": g, "sft": sft,
            "adapter": str(d / "adapter") if (d / "adapter" / "adapter_config.json").exists() else None}


def eval_entry(path):
    rep = json.loads(path.read_text())
    return {"file": str(path), "split": rep.get("split"), "checkpoint": rep.get("checkpoint"), "decoding": rep.get("decoding"),
            "answer_format": rep.get("answer_format"), "prompt_variant": rep.get("prompt_variant"),
            "corpus_sha256": rep.get("corpus_sha256"), "n": rep.get("metrics", {}).get("n"),
            "accuracy": rep.get("metrics", {}).get("accuracy")}


def verify_sealed(name="manifest_sealed.json"):
    manifest = json.loads((DATA_DIR / name).read_text())["splits"]
    bad = []
    for split, h in manifest.items():
        for kind in ("tasks", "corpus"):
            if file_hash(DATA_DIR / f"{split}_{kind}.jsonl") != h[f"{kind}_sha256"]:
                bad.append(f"{split}_{kind}")
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", default=[])
    parser.add_argument("--out", default=None)
    parser.add_argument("--verify-sealed", action="store_true")
    parser.add_argument("--verify-hard", action="store_true", help="check hard_train/dev/final against data/manifest_hard.json")
    args = parser.parse_args()
    if args.verify_sealed or args.verify_hard:
        bad = verify_sealed("manifest_hard.json" if args.verify_hard else "manifest_sealed.json")
        print("sealed files modified: " + ", ".join(bad) if bad else "sealed files match manifest_sealed.json")
        sys.exit(1 if bad else 0)
    runs, evals = [], []
    for root in map(Path, args.roots):
        runs += [run_entry(p.parent) for p in sorted(root.rglob("config_snapshot.yaml"))]
        evals += [eval_entry(p) for p in sorted(root.rglob("eval_*.json"))]
    text = json.dumps({"runs": runs, "evaluations": evals}, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(f"{len(runs)} runs, {len(evals)} evaluations")


if __name__ == "__main__":
    main()
