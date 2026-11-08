"""Build the generated blocks the report includes, and copy raw artifacts into the submission package.

    python -m src.package_submission --study outputs/study_hard --sft outputs/study_sft --sub submission

Every block is derived from files on disk. A block for an experiment that has not run says so instead of showing numbers.
"""

import argparse
import json
import shutil
from pathlib import Path


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def frozen_block(fz):
    if not fz:
        return None
    ev = fz["dev_evidence"]
    lines = [f"**Headline model: {fz['headline_label']}** (frozen {fz['frozen_at_utc']} UTC, before `hard_final` was evaluated)",
             f"- Checkpoint: `{fz['checkpoint_path']}` (adapter sha256 `{(fz['adapter_sha256'] or 'n/a')[:16]}...`), training seed {fz['training_seed']}, replication seeds {fz['replication_seeds']}",
             f"- Code: git `{(fz['git_commit'] or 'n/a')[:10]}` (working tree dirty: {fz['git_dirty']}); `src/` sha256 `{fz['source_sha256_src'][:16]}...`; config `{fz['config']}` sha256 `{fz['config_sha256'][:16]}...`",
             f"- Evaluation: answer format `{fz['evaluation']['answer_format']}`, {fz['evaluation']['decoding']}, max searches {fz['evaluation']['max_searches']}, system-prompt sha256 `{fz['evaluation']['system_prompt_sha256'][:16]}...`",
             f"- Sealed-set statement: {fz['sealed_view_statement']}", f"- {fz['note']}", "",
             "Selection rule applied to development data (no hard_final data used):", "```", fz["selection_rule"], "```", "",
             f"Reasons: {'; '.join(fz['reasons'])}", "",
             "| Variant | mean hard_dev | min seed | gain vs M2 (pts) | dev change | twohop_dev change | eligible | mean searches | steps with gradient | zero-std groups |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v, i in ev["variants"].items():
        tr = i.get("training", {})
        fg = ", ".join(f"{t['frac_steps_with_gradient']:.0%}" for t in tr.values())
        zs = ", ".join(f"{t['mean_frac_zero_std_groups']:.0%}" if t["mean_frac_zero_std_groups"] is not None else "–" for t in tr.values())
        lines.append(f"| {v} | {i['hard_dev_mean']:.1f}% | {i['hard_dev_min']:.1f}% | {i['gain_vs_M2']:+.1f} | {i['dev_change']:+.1f} | {i['twohop_dev_change']:+.1f} | "
                     f"{'yes' if i['eligible'] else 'no'} | {i['mean_searches_hard_dev']:.2f} | {fg} | {zs} |")
    lines += ["", f"M2 on development splits: hard_dev {ev['M2']['hard_dev']:.1f}%, dev {ev['M2']['dev']:.1f}%, twohop_dev {ev['M2']['twohop_dev']:.1f}%. "
              "Per-seed percentages are in `FROZEN_HARD.json`."]
    return "\n".join(lines)


def figures_block(figdir):
    figs = sorted(Path(figdir).glob("*.png"))
    if not figs:
        return None
    cap = {"fig1": "Original GRPO (no SFT): reward and policy loss per optimizer step.",
           "fig2": "GRPO initialised from the M2 SFT checkpoint: reward and policy loss, by reward variant and seed.",
           "fig3": "Reward spread, share of groups with zero reward std (no gradient) and gradient norm during training.",
           "fig4": "Searches per episode during training (all tasks and two-hop tasks).",
           "fig5": "Accuracy with 95% Wilson intervals (k/n above bars).",
           "fig6": "Accuracy by category (mean over seeds for each family).",
           "fig7": "Per-seed accuracy: spread across seeds vs differences between variants."}
    return "\n\n".join(f"![{f.stem}](figures/{f.name})\n*{f.stem.split('_')[0]}: {cap.get(f.stem.split('_')[0], '')}*" for f in figs)


def status_block(dirs, what, script):
    found = [d for d in dirs if Path(d).exists() and any(Path(d).rglob("decision_*.json"))]
    if not found:
        return (f"**Not run.** The {what} experiment (`{script}`) has not produced results in this submission, so no number is reported for it and it is "
                "listed as future work. The staged design, gates and metrics are implemented and unit-tested.")
    out = [f"Decision records found for the {what} experiment:"]
    for d in found:
        for p in sorted(Path(d).rglob("decision_*.json")):
            out += [f"- `{p}`", "```", json.dumps(jload(p), indent=1)[:1500], "```"]
    return "\n".join(out)


def lineage_block(study):
    L = jload(Path(study) / "lineage_BCD.json")
    if not L:
        return None
    lines = ["| Pair | byte-identical adapters | max abs diff | relative L2 diff |", "|---|---|---|---|"]
    for k, v in L["pairs"].items():
        lines.append(f"| {k} | {v['byte_identical']} | {v.get('max_abs_diff')} | {v.get('relative_l2_diff')} |")
    for lab, r in L["runs"].items():
        m = r.get("metrics")
        if m:
            lines.append(f"\n{lab}: logged steps {m.get('logged_steps')}, steps with non-zero gradient {m.get('steps_nonzero_grad')}")
    ov = {k: v for k, v in L["pairs"].items() if "rollouts" in v}
    if ov:
        lines.append("\nSampled-rollout overlap between runs: " + "; ".join(f"{k}: {json.dumps(v['rollouts'])}" for k, v in ov.items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--sft", default="outputs/study_sft")
    ap.add_argument("--sub", default="submission")
    args = ap.parse_args()
    S, sub = Path(args.study), Path(args.sub)
    (sub / "tables").mkdir(parents=True, exist_ok=True)
    (sub / "raw").mkdir(exist_ok=True)
    fz = jload(S / "FROZEN_HARD.json")
    blocks = {"tables/frozen_summary.md": frozen_block(fz), "figures_block.md": figures_block(sub / "figures"),
              "tables/lineage_status.md": lineage_block(S),
              "tables/early_status.md": status_block([Path(args.study).parent / "study_early"], "partial-SFT (E0/E1/E2)", "run_early_study.sh"),
              "tables/rl_only_status.md": status_block([Path(args.study).parent / "study_rl_only"], "RL-only (F0-F4)", "run_rl_only_study.sh")}
    for name, text in blocks.items():
        if text:
            (sub / name).write_text(text + "\n")
    if fz:
        shutil.copy(S / "FROZEN_HARD.json", sub / "FROZEN_HARD.json")
    copied = 0
    for pat in ("eval_hard_dev_*.json", "eval_hard_final_*.json", "eval_twohop_dev_*.json", "decision_*.json", "diag_*.json", "traj_stats*.json", "lineage*.json",
                "experiment_manifest*.json", "validate_hard.json", "HARD_FINAL_FIRST_EVALUATION"):
        for p in S.glob(pat):
            shutil.copy(p, sub / "raw" / p.name); copied += 1
    for run in sorted(S.glob("M[34]*_s*")):
        for f in ("train_metrics.csv", "config_snapshot.yaml", "system_info.txt", "param_drift.csv"):
            if (run / f).exists():
                (sub / "raw" / run.name).mkdir(exist_ok=True); shutil.copy(run / f, sub / "raw" / run.name / f); copied += 1
    print(f"wrote {[k for k, v in blocks.items() if v]}; copied {copied} raw files into {sub}/raw")


if __name__ == "__main__":
    main()
