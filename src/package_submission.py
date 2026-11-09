"""Build the generated blocks the report includes, and copy raw artifacts into the submission package.

    python -m src.package_submission --study outputs/study_hard --sft outputs/study_sft --sub submission

Every block is derived from files on disk. A block for an experiment that has not run says so instead of showing numbers.
"""

import argparse
import hashlib
import json
import re
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


def figures_block(figdir, sub=None):
    figs = [f for f in sorted(Path(figdir).glob("*.png")) if not f.stem.startswith(("fig8", "fig9"))]
    if not figs:
        return None
    cap = {"fig1": "Original GRPO from the pretrained model (no SFT): reward and GRPO policy loss per optimizer step.",
           "fig2": "GRPO initialised from the M2 SFT checkpoint on hard_train: reward and GRPO policy loss, by reward variant (M3, M4, M4g) and seed (solid 42, dashed 43).",
           "fig3": "Reward spread, share of groups with zero reward spread (no gradient) and gradient norm during training.",
           "fig4": "Searches per training episode (all tasks and two-hop tasks).",
           "fig5": "Accuracy with 95% Wilson intervals (k/n above bars); M2 is shown under both reminders.",
           "fig6": "Accuracy by category (mean over seeds for each GRPO family).",
           "fig7": "Per-seed accuracy: spread across seeds vs differences between variants.",
           "fig8": "Before/after accuracy with both SFT baselines (Comparison A: same reminder; Comparison B: M2 under the reminder it was trained with). Not an RL-only effect.",
           "fig9": "Per-category accuracy: M2 under typed_abstain, M2 under typed, and the frozen GRPO model."}
    out = "\n\n".join(f"![{f.stem}](figures/{f.name})\n*{f.stem.split('_')[0]}: {cap.get(f.stem.split('_')[0], '')}*" for f in figs)
    if not any(f.name.startswith("fig1_") for f in figs):
        note = Path(sub) / "tables" / "original_grpo_log_recovery.md" if sub else None
        reason = note.read_text().strip() if note and note.exists() else "No per-step log of the original from-base GRPO run was found among the saved artifacts."
        out = f"> **Figure 1 (original GRPO reward and loss): unavailable.** {reason}\n\n" + out
    return out


def before_after_block(figdir):
    figs = sorted(Path(figdir).glob("fig8*.png")) + sorted(Path(figdir).glob("fig9*.png"))
    if not figs:
        return None
    cap = {"fig8": "Before/after accuracy with both SFT baselines (A: same reminder; B: M2 under the reminder it was trained with). Not an RL-only effect.",
           "fig9": "Per-category accuracy: M2 under typed_abstain, M2 under typed, and the frozen GRPO model."}
    return "\n\n".join(f"![{f.stem}](figures/{f.name})\n*{f.stem.split('_')[0]}: {cap[f.stem.split('_')[0]]} ({f.stem.split('_')[-2]}_{f.stem.split('_')[-1]})*" for f in figs)


def status_block(dirs, what, script, summary_glob):
    found = [Path(d) for d in dirs if Path(d).exists()]
    summaries = [p for d in found for p in sorted(d.glob(summary_glob))]
    decisions = [p for d in found for p in sorted(d.rglob("decision_*.json"))]
    if not summaries and not decisions:
        return (f"**Not run in this submission.** The {what} experiment (`{script}`) has no results among the saved artifacts, so no number is reported for it and it is "
                "listed as future work.")
    out = [f"Results of the {what} experiment (`{script}`; development splits only, exploratory), copied from the saved summary files:"]
    for p in summaries:
        out += ["", f"`{p.name}`:", "", p.read_text().strip()]
    for p in decisions:
        out += ["", f"Gate decision `{p.name}`:", "```", json.dumps(jload(p), indent=1)[:1500], "```"]
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

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".DS_Store", "*.safetensors", "*.bin", "*.pt")
CODE_DIRS = ("src", "data", "configs", "tests")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_code(sub, notebooks, fz):
    """Source, configs, datasets+hash manifests, requirements, run scripts, executed notebooks, and ONLY the frozen headline adapter."""
    code = sub / "code"
    for d in CODE_DIRS:
        if Path(d).exists():
            if (code / d).exists():
                shutil.rmtree(code / d)  # rebuilt from the working tree; this is the package's own copy, not an archived submission
            shutil.copytree(d, code / d, ignore=IGNORE)
    code.mkdir(exist_ok=True)
    for pat in ("run_*.sh", "requirements*.txt", "requirements.lock", "README.md", "*.ipynb"):
        for p in Path(".").glob(pat):
            if p.is_file():
                shutil.copy(p, code / p.name)
    (code / "reports").mkdir(exist_ok=True)
    if Path("reports/FINAL_PILOT_REPORT.template.md").exists():
        shutil.copy("reports/FINAL_PILOT_REPORT.template.md", code / "reports" / "FINAL_PILOT_REPORT.template.md")
    nb = sub / "notebooks"
    nb.mkdir(exist_ok=True)
    got = []
    for n in notebooks:
        n = Path(n)
        if n.exists() and not (nb / n.name).exists():
            shutil.copy(n, nb / n.name); got.append(n.name)
    ad = None
    if fz and fz.get("checkpoint_path"):
        src = Path(fz["checkpoint_path"])
        if (src / "adapter_model.safetensors").exists():
            ad = sub / "checkpoint" / "frozen_headline_adapter"
            ad.mkdir(parents=True, exist_ok=True)
            for f in ("adapter_model.safetensors", "adapter_config.json"):
                if (src / f).exists():
                    shutil.copy(src / f, ad / f)
    return got, ad


def readme(sub, fz, two):
    sysinfo = ""
    for p in sorted((sub / "raw").glob("M3_s*/system_info.txt"))[:1]:
        sysinfo = "\n".join(p.read_text().splitlines()[:14])
    lines = ["# SearchLite-RL submission package (CS 496 Agent AI pilot): how to read and reproduce", "",
             "Start with `FINAL_PILOT_REPORT.md`. Every number in it comes from a file in this package (`raw/`, `tables/`, `FROZEN_HARD.json`) or is labelled with its source.", "",
             "## Course requirements", "",
             "| Requirement | Where |", "|---|---|",
             "| 1. Working RL training framework (TRL GRPO, LoRA, BM25 tool environment) | `code/src/train_grpo.py`, `code/src/environment.py`; training logs `raw/M3_s42/` etc.; tests in `code/tests/` |",
             "| 2. Reward and policy-loss curves | `figures/fig2_m2_grpo_reward_loss.png` (M2-initialised GRPO); `figures/fig1_original_grpo_reward_loss.png` if present (see the report for its status); `figures/fig3`, `fig4` |",
             "| 3. Case studies from real tool traces | `case_studies.md` (verbatim saved traces) and report Section 13 |", "",
             "Bonus material (not required): reward-variant ablation, format control, zero-variance diagnostics, RL-only and partial-SFT experiments.", ""]
    if fz:
        lines += ["## Frozen headline model", "",
                  f"- {fz['headline_label']}, frozen {fz['frozen_at_utc']} UTC before the sealed evaluation; manifest `FROZEN_HARD.json` (read-only at creation).",
                  f"- Adapter sha256 `{fz['adapter_sha256']}`; the adapter itself is in `checkpoint/frozen_headline_adapter/` (hash re-checked by the status checklist).",
                  f"- Git commit at freeze `{fz['git_commit']}`, `src/` sha256 `{fz['source_sha256_src']}`.", ""]
    if two:
        oc = two["overall_counts"]
        lines += [f"## Sealed `{two['split']}` accuracy (from the raw evaluation JSONs)", "", "| model | correct | accuracy |", "|---|---|---|"]
        lines += [f"| {k} | {v[0]}/{v[1]} | {100 * v[0] / v[1]:.1f}% |" for k, v in oc.items()]
        lines += ["", "Read these with the two paired comparisons in the report (fixed prompt vs stronger SFT baseline): the SFT checkpoint is strongly sensitive to the evaluation reminder.", ""]
    lines += ["## Contents", "", "- `FINAL_PILOT_REPORT.md`: the report; `STATUS_CHECKLIST.md`: automated checks; `FROZEN_HARD.json`: freeze manifest",
              "- `raw/`: saved evaluation JSONs (per-question traces), training logs, gate decisions, hashes of the first sealed evaluation",
              "- `tables/`, `figures/`, `case_studies.md`: generated from `raw/` by `code/src/*.py`",
              "- `code/`: source, configs, datasets and their hash manifests, tests, run scripts, requirements; `notebooks/`: executed Colab notebooks (with outputs)",
              "- `MANIFEST_SHA256.txt`: sha256 of every file in this package", "",
              "## Reproduce", "", "```", "pip install -r code/requirements.lock && cd code && python -m pytest -q       # CPU, no GPU needed",
              "# GPU stages (free Colab L4) are in the notebooks: SearchLite_RL_Headroom_Resume.ipynb (training) and SearchLite_RL_Final_Submission.ipynb (freeze, sealed evaluation)",
              "# CPU analysis from the saved evaluations: python -m src.two_comparisons --study <dir with eval_*.json> --split hard_final --headline M3_s42 ...", "```", ""]
    if sysinfo:
        lines += ["## Environment of the headline run (`raw/M3_s*/system_info.txt`, first lines)", "", "```", sysinfo, "```", ""]
    lines += ["Compute: free Google Colab GPUs only (L4 / A100); no paid APIs; no heavy training on the author's laptop. Per-run wall-clock is in each `system_info.txt` / the freeze manifest."]
    (sub / "README_SUBMISSION.md").write_text("\n".join(lines) + "\n")


def write_manifest(sub):
    files = sorted(p for p in sub.rglob("*") if p.is_file() and p.name != "MANIFEST_SHA256.txt")
    (sub / "MANIFEST_SHA256.txt").write_text("".join(f"{sha(p)}  {p.relative_to(sub)}\n" for p in files))
    return len(files)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--sft", default="outputs/study_sft")
    ap.add_argument("--sub", default="submission")
    ap.add_argument("--notebooks", nargs="*", default=[], help="executed notebooks (with outputs) to include")
    args = ap.parse_args()
    S, sub = Path(args.study), Path(args.sub)
    (sub / "tables").mkdir(parents=True, exist_ok=True)
    (sub / "raw").mkdir(exist_ok=True)
    fz = jload(S / "FROZEN_HARD.json")
    blocks = {"tables/frozen_summary.md": frozen_block(fz), "figures_block.md": figures_block(sub / "figures", sub), "figures_before_after.md": before_after_block(sub / "figures"),
              "tables/lineage_status.md": lineage_block(S),
              "tables/early_status.md": status_block([Path(args.study).parent / "study_early"], "partial-SFT (E0/E1/E2)", "run_early_study.sh", "summary_early*.md"),
              "tables/rl_only_status.md": status_block([Path(args.study).parent / "study_rl_only"], "RL-only (F0-F4)", "run_rl_only_study.sh", "summary_rl_only*.md")}
    for name, text in blocks.items():
        if text:
            (sub / name).write_text(text + "\n")
    if fz:
        shutil.copy(S / "FROZEN_HARD.json", sub / "FROZEN_HARD.json")
    copied = 0
    for pat in ("eval_hard_dev_*.json", "eval_hard_final_*.json", "eval_twohop_dev_*.json", "decision_*.json", "diag_*.json", "traj_stats*.json", "lineage*.json",
                "experiment_manifest*.json", "validate_hard.json", "HARD_FINAL_FIRST_EVALUATION", "eval_dev_*.json", "summary_hard_*.md"):
        for p in S.glob(pat):
            shutil.copy(p, sub / "raw" / p.name); copied += 1
    for run in sorted(S.glob("M[34]*_s*")):
        for f in ("train_metrics.csv", "config_snapshot.yaml", "system_info.txt", "param_drift.csv"):
            if (run / f).exists():
                (sub / "raw" / run.name).mkdir(exist_ok=True); shutil.copy(run / f, sub / "raw" / run.name / f); copied += 1
    for extra in ("study_early", "study_rl_only"):
        d = Path(args.study).parent / extra
        if d.exists():
            (sub / "raw" / extra).mkdir(exist_ok=True)
            for pat in ("summary_*.md", "decision_*.json", "eval_*.json"):
                for p in d.glob(pat):
                    shutil.copy(p, sub / "raw" / extra / p.name); copied += 1
            for run in sorted(d.glob("[EF]*_s*")) + sorted(d.glob("F*_pilot")):
                if (run / "train_metrics.csv").exists():
                    (sub / "raw" / extra / run.name).mkdir(exist_ok=True); shutil.copy(run / "train_metrics.csv", sub / "raw" / extra / run.name / "train_metrics.csv"); copied += 1
    orig = Path("outputs/main_v2")
    for f in ("train_metrics.csv", "train_metrics.source.txt"):
        if (orig / f).exists():
            (sub / "raw" / "main_v2_original_grpo").mkdir(exist_ok=True); shutil.copy(orig / f, sub / "raw" / "main_v2_original_grpo" / f); copied += 1
    got, ad = package_code(sub, args.notebooks, fz)
    readme(sub, fz, jload(sub / "tables" / "two_comparisons.json"))
    print(f"wrote {[k for k, v in blocks.items() if v]}; copied {copied} raw files into {sub}/raw; code + notebooks {got}; frozen adapter {'included' if ad else 'NOT found'}")


if __name__ == "__main__":
    main()
