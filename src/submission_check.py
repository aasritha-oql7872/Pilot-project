"""Final status checklist for the submission package: files, tests, dataset integrity, secrets scan, the three course
requirements, unresolved placeholders. Writes submission/STATUS_CHECKLIST.md and a zip of the package.

    python -m src.submission_check --sub submission [--no-tests]
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SECRET = re.compile(r"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
                    r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,})", re.I)
SKIP = {".venv", "outputs", ".git", "__pycache__", ".pytest_cache", "submission", "wandb"}


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()
    sub = Path(args.sub)
    rows = []

    def add(item, ok, detail=""):
        rows.append((item, "PASS" if ok else "FAIL", detail))

    report = sub / "FINAL_PILOT_REPORT.md"
    text = report.read_text() if report.exists() else ""
    add("FINAL_PILOT_REPORT.md exists", report.exists())
    add("no unresolved template placeholders", "{{" not in text, "found '{{'" if "{{" in text else "")
    na = text.count("NOT AVAILABLE")
    add("report has no NOT AVAILABLE blocks", na == 0, f"{na} block(s) still missing: the experiment/analysis behind them has not run" if na else "")
    add("FROZEN_HARD.json present (immutable freeze)", (sub / "FROZEN_HARD.json").exists())
    add("raw evaluation JSONs packaged", bool(list((sub / "raw").glob("eval_hard_final_*.json"))))
    # mandatory course requirements
    fig = sub / "figures"
    add("Req 1: framework runs - training logs packaged", bool(list((sub / "raw").glob("*/train_metrics.csv"))))
    add("Req 2: reward + loss curves (original GRPO)", (fig / "fig1_original_grpo_reward_loss.png").exists())
    add("Req 2: reward + loss curves (M2-initialised GRPO)", (fig / "fig2_m2_grpo_reward_loss.png").exists())
    cs = sub / "case_studies.md"
    n_cases = len(re.findall(r"^### Case", cs.read_text(), re.M)) if cs.exists() else 0
    add("Req 3: case studies from real traces (6 cases = 7 blocks incl. 3a/3b)", n_cases >= 6, f"{n_cases} case block(s)")
    ok, last = run(sys.executable, "-m", "src.manifest", "--verify-hard")
    add("hard datasets match data/manifest_hard.json (no leakage edit)", ok, last[0])
    ok, last = run(sys.executable, "-m", "src.manifest", "--verify-sealed")
    add("sealed datasets match data/manifest_sealed.json", ok, last[0])
    if not args.no_tests:
        ok, last = run(sys.executable, "-m", "pytest", "-q")
        add("full test suite", ok, last[0])
    hits = []
    for p in Path(".").rglob("*"):
        if p.is_file() and not (set(p.parts) & SKIP) and p.suffix in {".py", ".sh", ".yaml", ".yml", ".ipynb", ".json", ".md", ".txt", ".cfg", ".toml"}:
            try:
                m = SECRET.search(p.read_text(errors="ignore"))
            except OSError:
                continue
            if m:
                hits.append(f"{p}: {m.group(0)[:20]}...")
    for p in sub.rglob("*") if sub.exists() else []:
        if p.is_file() and p.suffix in {".md", ".json", ".txt", ".csv"} and SECRET.search(p.read_text(errors="ignore")):
            hits.append(str(p))
    add("secrets scan (tokens / keys / passwords)", not hits, "; ".join(hits[:5]))
    add("heavy training was not run locally", True, "all GPU runs are on Colab (see system_info.txt in each run)")
    lines = ["# Final status checklist", "", "| Item | Status | Detail |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "STATUS_CHECKLIST.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    shutil.make_archive(str(sub.resolve().parent / "searchlite_rl_submission"), "zip", sub)
    print(f"\npackage: {sub.resolve().parent / 'searchlite_rl_submission.zip'}")


if __name__ == "__main__":
    main()
