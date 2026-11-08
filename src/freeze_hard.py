"""Freeze the headline model BEFORE the sealed hard_final evaluation, using development results only.

    python -m src.freeze_hard --study outputs/study_hard --sft outputs/study_sft [--seeds 42 43] [--dry-run]

Decision procedure (declared here, in the repository, before hard_final is evaluated. Disclosure: when the thresholds
were written, the author had seen only the user's summary of two-hop dev accuracy (M2 14/30; M3, M4, M4g about 29-30/30)
and had NOT seen the per-seed hard_dev tables, the dev/twohop_dev regression tables, or any hard_final result):

  1. Eligible variant: mean hard_dev accuracy over seeds beats M2 by >= 5 points, AND dev and twohop_dev each lose at most
     3 points relative to M2 (the pre-registered rule from run_headroom_study.sh).
  2. Competitive set: eligible variants whose mean hard_dev accuracy is within 5.0 points (about 3 of 60 questions) of the
     best eligible variant, AND whose worst seed is still above M2. A one-question gain in one seed never decides anything.
  3. Headline = the SIMPLEST competitive variant. Simplicity order: M3 (original reward, no extra terms) < M4g (adds one
     unsupported-answer penalty, no gold ids) < M4 (adds two shaping bonuses that use privileged gold doc ids). Ties after
     simplicity go to the variant with the smaller hard_dev spread across seeds.
  4. If nothing is eligible the headline stays M2 (negative result) and no GRPO model is promoted.
  5. The headline checkpoint is seed 42 (declared default). The other seed is reported as a replication, never swapped in.

The manifest is written once (mode 'x', then read-only). If FROZEN_HARD.json already exists the script refuses.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from data.generate import DATA_DIR, file_hash
from src.environment import ANSWER_FORMATS, MAX_SEARCHES, PROMPT_VARIANTS, SYSTEM_PROMPT
from src.manifest import verify_sealed

SIMPLICITY = {"M3": 0, "M4g": 1, "M4": 2}
MIN_GAIN, MAX_REGRESSION, TOLERANCE = 5.0, 3.0, 5.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def acc(path):
    eps = json.loads(Path(path).read_text())["episodes"]
    return 100.0 * sum(bool(e["correct"]) for e in eps) / len(eps), len(eps), sum(bool(e["correct"]) for e in eps)


def searches(path):
    eps = json.loads(Path(path).read_text())["episodes"]
    return sum(e["num_searches"] for e in eps) / len(eps)


def code_hash(root="src"):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*.py")):
        h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def mean(xs):
    return sum(xs) / len(xs)


def run_stats(run_dir):
    """Training stability read from the run's own files."""
    import csv
    rows = list(csv.DictReader(open(Path(run_dir) / "train_metrics.csv")))
    g = lambda k: [float(r[k]) for r in rows if r.get(k) not in (None, "")]
    secs = None
    info = Path(run_dir) / "system_info.txt"
    if info.exists():
        m = re.search(r"train_seconds:\s*([\d.]+)", info.read_text())
        secs = float(m.group(1)) if m else None
    grad = g("grad_norm")
    return {"logged_steps": len(rows), "finite": all(v == v and abs(v) < 1e9 for k in ("loss", "reward", "grad_norm") for v in g(k)),
            "frac_steps_with_gradient": sum(x > 0 for x in grad) / max(1, len(grad)),
            "mean_frac_zero_std_groups": mean(g("frac_reward_zero_std")) if g("frac_reward_zero_std") else None,
            "train_seconds": secs}


def select(table, m2):
    """table[variant] = {seed: {hard_dev, dev, twohop_dev}} in percent; m2 = same dict for M2. Returns (headline, reasons, per-variant)."""
    info, reasons = {}, []
    for v, by_seed in table.items():
        hd = [s["hard_dev"] for s in by_seed.values()]
        info[v] = {"hard_dev_mean": mean(hd), "hard_dev_min": min(hd), "hard_dev_spread": max(hd) - min(hd),
                   "gain_vs_M2": mean(hd) - m2["hard_dev"],
                   "dev_change": mean([s["dev"] for s in by_seed.values()]) - m2["dev"],
                   "twohop_dev_change": mean([s["twohop_dev"] for s in by_seed.values()]) - m2["twohop_dev"]}
        i = info[v]
        i["eligible"] = i["gain_vs_M2"] >= MIN_GAIN and i["dev_change"] >= -MAX_REGRESSION and i["twohop_dev_change"] >= -MAX_REGRESSION
    elig = {v: i for v, i in info.items() if i["eligible"]}
    if not elig:
        return "M2", ["no GRPO variant met the pre-registered rule (>=5 pts on hard_dev, <=3 pts regression on dev and twohop_dev)"], info
    best = max(i["hard_dev_mean"] for i in elig.values())
    comp = {v: i for v, i in elig.items() if i["hard_dev_mean"] >= best - TOLERANCE and i["hard_dev_min"] > m2["hard_dev"]}
    if not comp:
        return "M2", ["eligible variants had a seed that did not beat M2; headline stays M2"], info
    headline = sorted(comp, key=lambda v: (SIMPLICITY[v], comp[v]["hard_dev_spread"]))[0]
    reasons.append(f"eligible: {sorted(elig)}; competitive (within {TOLERANCE} pts of best mean {best:.1f}% and every seed above M2): {sorted(comp)}")
    reasons.append(f"headline = simplest competitive variant by reward complexity (M3 < M4g < M4): {headline}")
    return headline, reasons, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--sft", default="outputs/study_sft")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--headline-seed", type=int, default=42)
    ap.add_argument("--config", default="configs/train_hard_grpo.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print the decision, write nothing")
    args = ap.parse_args()
    S, out = Path(args.study), Path(args.study) / "FROZEN_HARD.json"
    if out.exists() and not args.dry_run:
        sys.exit(f"{out} already exists: the freeze is immutable. Create nothing new; use the existing manifest.")

    def get(label, split):
        p = S / f"eval_{split}_{label}.json"
        if not p.exists():
            sys.exit(f"missing {p}: finish run_headroom_study.sh before freezing")
        return acc(p)[0]

    m2 = {s: get("M2", s) for s in ("hard_dev", "dev", "twohop_dev")}
    table = {}
    for v in SIMPLICITY:
        by = {}
        for sd in args.seeds:
            if not (S / f"{v}_s{sd}" / "adapter" / "adapter_config.json").exists():
                sys.exit(f"missing adapter {v}_s{sd}")
            by[sd] = {s: get(f"{v}_s{sd}", s) for s in ("hard_dev", "dev", "twohop_dev")}
        table[v] = by
    headline, reasons, info = select(table, m2)
    for v in info:
        info[v]["mean_searches_hard_dev"] = mean([searches(S / f"eval_hard_dev_{v}_s{sd}.json") for sd in args.seeds])
        info[v]["training"] = {sd: run_stats(S / f"{v}_s{sd}") for sd in args.seeds}
    if headline == "M2":
        ckpt, seed, run_dir = str(Path(args.sft) / "B" / "adapter"), None, Path(args.sft) / "B"
    else:
        run_dir = S / f"{headline}_s{args.headline_seed}"
        ckpt, seed = str(run_dir / "adapter"), args.headline_seed
    bad = verify_sealed("manifest_hard.json") + verify_sealed("manifest_sealed.json")
    if bad:
        sys.exit("sealed/hard dataset files differ from their manifests: " + ", ".join(bad))
    prior = sorted(str(p) for p in list(S.glob("eval_hard_final_*.json")) + list(S.glob("final_*.log")))
    sealed_old = sorted(str(p) for p in Path(args.sft).glob("**/eval_*final*.json"))
    hard, old = json.loads((DATA_DIR / "manifest_hard.json").read_text()), json.loads((DATA_DIR / "manifest_sealed.json").read_text())
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline_model": headline, "headline_label": f"{headline}" + (f" seed {seed}" if seed else " (SFT only, study_sft/B)"),
        "checkpoint_path": ckpt, "adapter_sha256": sha256_file(Path(ckpt) / "adapter_model.safetensors") if (Path(ckpt) / "adapter_model.safetensors").exists() else None,
        "training_seed": seed, "replication_seeds": [s for s in args.seeds if s != seed],
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")), "source_sha256_src": code_hash(),
        "config": args.config, "config_sha256": sha256_file(args.config),
        "run_config_snapshot_sha256": sha256_file(run_dir / "config_snapshot.yaml") if (run_dir / "config_snapshot.yaml").exists() else None,
        "evaluation": {"answer_format": "typed_abstain", "answer_format_text": ANSWER_FORMATS["typed_abstain"],
                       "system_prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + PROMPT_VARIANTS.get("default", "")).encode()).hexdigest(),
                       "max_searches": MAX_SEARCHES, "decoding": "greedy (temperature 0)", "top_k_retrieval": 2},
        "selection_rule": __doc__.split("Decision procedure")[1].split('The manifest is written')[0].strip(),
        "reasons": reasons, "dev_evidence": {"M2": m2, "variants": info, "per_seed": {v: {str(k): s for k, s in b.items()} for v, b in table.items()}},
        "sealed_dataset_hashes": {"hard": {k: v for k, v in hard["splits"].items() if k == "hard_final"},
                                  "one_and_two_hop": old["splits"]},
        "hard_final_viewed_before_freeze": bool(prior), "hard_final_prior_artifacts": prior,
        "earlier_sealed_artifacts_for_final_and_twohop_final": sealed_old,
        "sealed_view_statement": ("hard_final had NOT been evaluated before this freeze (no eval_hard_final_* or final_*.log found)."
                                  if not prior else "WARNING: hard_final artifacts existed before the freeze; hard_final is NOT an untouched holdout."),
        "note": "final/twohop_final are a different benchmark from hard_final. If they were evaluated earlier (see run_sealed.sh logs on Drive) they are not untouched either.",
    }
    text = json.dumps(manifest, indent=2, default=str)
    print(text)
    if args.dry_run:
        return
    with open(out, "x") as f:
        f.write(text)
    os.chmod(out, 0o444)
    (S / "FROZEN_HARD").write_text(f"headline: {manifest['headline_label']}; frozen {manifest['frozen_at_utc']}; see FROZEN_HARD.json\n")
    os.chmod(S / "FROZEN_HARD", 0o444)
    print(f"\nFROZEN -> {out} (read-only)")


if __name__ == "__main__":
    main()
