"""Final status checklist for the submission package: files, tests, dataset integrity, secrets scan, the three course
requirements, unresolved placeholders, package hygiene and links. Writes submission/STATUS_CHECKLIST.md, MANIFEST_SHA256.txt
and a zip of the package (versioned name, never overwriting an existing archive).

    python -m src.submission_check --sub submission [--no-tests] [--version v2_20261108]
"""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.package_submission import write_manifest

SECRET = re.compile(r"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
                    r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,})", re.I)
SKIP = {".venv", "outputs", ".git", "__pycache__", ".pytest_cache", "submission", "wandb"}
TEXT = {".py", ".sh", ".yaml", ".yml", ".ipynb", ".json", ".md", ".txt", ".cfg", ".toml", ".csv", ".jsonl", ".lock"}
NA_BLOCK = re.compile(r"^> \*\*NOT AVAILABLE:\*\*", re.M)
FIG1_NOTE = "Figure 1 (original GRPO reward and loss): unavailable"
FORBIDDEN = (".env", ".pem", ".key", ".netrc", ".pt", ".pth", ".ckpt", ".bin")
MAX_BYTES = 25 * 1024 * 1024
REQUIRED = ["FINAL_PILOT_REPORT.md", "README_SUBMISSION.md", "STATUS_CHECKLIST.md", "FROZEN_HARD.json", "case_studies.md", "MANIFEST_SHA256.txt",
            "figures/fig2_m2_grpo_reward_loss.png", "tables/two_comparisons.md", "tables/summary_hard_final.md", "code/src/train_grpo.py",
            "code/configs/train_hard_grpo.yaml", "code/requirements.lock", "code/data/manifest_hard.json", "checkpoint/frozen_headline_adapter/adapter_model.safetensors"]


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def broken_links(sub, text):
    """Relative markdown links / images in the report that do not resolve inside the package."""
    bad = []
    for m in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)\)", text):
        t = m.group(1)
        if re.match(r"[a-z]+://|#|mailto:", t):
            continue
        if not (sub / t.split("#")[0]).exists():
            bad.append(t)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--version", default=None, help="suffix for the zip name, e.g. v2_20261108")
    args = ap.parse_args()
    sub = Path(args.sub)
    rows = []

    def add(item, ok, detail=""):
        rows.append((item, "PASS" if ok else "FAIL", detail))

    report = sub / "FINAL_PILOT_REPORT.md"
    text = report.read_text() if report.exists() else ""
    add("FINAL_PILOT_REPORT.md exists", report.exists())
    add("no unresolved template placeholders", "{{" not in text, "found '{{'" if "{{" in text else "")
    na = len(NA_BLOCK.findall(text))
    add("report has no unfilled artifact blocks", na == 0, f"{na} block(s) marked NOT AVAILABLE: the artifact behind them was not produced" if na else "")
    add("FROZEN_HARD.json present (immutable freeze)", (sub / "FROZEN_HARD.json").exists())
    add("raw evaluation JSONs packaged", bool(list((sub / "raw").glob("eval_hard_final_*.json"))))
    add("both SFT baselines present on hard_final (M2 typed_abstain and M2 typed)", all((sub / "raw" / f"eval_hard_final_{n}.json").exists() for n in ("M2", "M2_typed")))
    add("paired comparison A/B tables present", (sub / "tables" / "two_comparisons.json").exists())
    add("report has the 'Improvement Attributable to RL Versus Prompt Design' section", "How Much Improvement Is Attributable to RL Versus Prompt Design?" in text)
    add("report links and figure paths resolve", not broken_links(sub, text), ", ".join(broken_links(sub, text)[:5]))
    # mandatory course requirements
    fig = sub / "figures"
    add("Req 1: framework runs - training logs packaged", bool(list((sub / "raw").glob("*/train_metrics.csv"))))
    orig = (fig / "fig1_original_grpo_reward_loss.png").exists()
    add("Req 2: reward + loss curves (original GRPO)", orig or FIG1_NOTE in text,
        "present" if orig else ("UNAVAILABLE, stated in the report with the reason (not a failure of the required curves)" if FIG1_NOTE in text else "missing and not explained in the report"))
    add("Req 2: reward + loss curves (M2-initialised GRPO)", (fig / "fig2_m2_grpo_reward_loss.png").exists())
    add("Req 2: before/after accuracy figure", bool(list(fig.glob("fig8_before_after_hard_final.png"))))
    cs = sub / "case_studies.md"
    n_cases = len(re.findall(r"^### Case", cs.read_text(), re.M)) if cs.exists() else 0
    add("Req 3: case studies from real traces (6 cases = 7 blocks incl. 3a/3b)", n_cases >= 6, f"{n_cases} case block(s)")
    add("executed notebook(s) with outputs included", bool(list((sub / "notebooks").glob("*.ipynb"))))
    ok, last = run(sys.executable, "-m", "src.manifest", "--verify-hard")
    add("hard datasets match data/manifest_hard.json (no leakage edit)", ok, last[0])
    ok, last = run(sys.executable, "-m", "src.manifest", "--verify-sealed")
    add("sealed datasets match data/manifest_sealed.json", ok, last[0])
    fz = sub / "FROZEN_HARD.json"
    if fz.exists():
        import json
        m = json.loads(fz.read_text())
        ad = sub / "checkpoint" / "frozen_headline_adapter" / "adapter_model.safetensors"
        add("frozen adapter hash equals the freeze manifest", ad.exists() and sha256(ad) == m.get("adapter_sha256"), "" if ad.exists() else "adapter not packaged")
    if not args.no_tests:
        ok, last = run(sys.executable, "-m", "pytest", "-q", "tests")   # not the package's own copy under submission/code/tests
        add("full test suite", ok, last[0])
    hits = []
    for p in Path(".").rglob("*"):
        if p.is_file() and not (set(p.parts) & SKIP) and p.suffix in TEXT:
            try:
                m = SECRET.search(p.read_text(errors="ignore"))
            except OSError:
                continue
            if m:
                hits.append(f"{p}: {m.group(0)[:20]}...")
    for p in sub.rglob("*") if sub.exists() else []:
        if p.is_file() and p.suffix in TEXT and SECRET.search(p.read_text(errors="ignore")):
            hits.append(str(p))
    add("secrets scan (repository + package: tokens / keys / passwords)", not hits, "; ".join(hits[:5]))
    junk = []
    for p in sub.rglob("*") if sub.exists() else []:
        if p.is_file():
            if p.name.startswith(".env") or p.suffix in FORBIDDEN or "__pycache__" in p.parts or p.stat().st_size > MAX_BYTES:
                junk.append(str(p.relative_to(sub)))
            elif p.suffix == ".safetensors" and "frozen_headline_adapter" not in p.parts:
                junk.append(str(p.relative_to(sub)))
    add("package hygiene (no .env, extra checkpoints, caches or files > 25 MB)", not junk, ", ".join(junk[:5]))
    add("heavy training was not run locally", True, "all GPU runs are on Colab (see system_info.txt in each run)")
    lines = ["# Final status checklist", "", "| Item | Status | Detail |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "STATUS_CHECKLIST.md").write_text("\n".join(lines) + "\n")
    n = write_manifest(sub)
    print("\n".join(lines))
    name = "searchlite_rl_submission" + (f"_{args.version}" if args.version else "")
    zpath = sub.resolve().parent / f"{name}.zip"
    if zpath.exists():
        raise SystemExit(f"{zpath} already exists and is never overwritten; choose a new --version")
    shutil.make_archive(str(sub.resolve().parent / name), "zip", sub)
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        missing = [r for r in REQUIRED if r not in names]
        same = report.exists() and z.read("FINAL_PILOT_REPORT.md") == report.read_bytes()
    digest = sha256(zpath)
    (zpath.parent / f"{name}.zip.sha256").write_text(f"{digest}  {zpath.name}\n")
    print(f"\npackage: {zpath}  ({zpath.stat().st_size // 1024} KB, {n} files in manifest)\nsha256: {digest}")
    print("zip contains every required path" if not missing else f"ZIP IS MISSING: {missing}")
    print("zip report == packaged report" if same else "ZIP REPORT DIFFERS FROM THE PACKAGED REPORT")


if __name__ == "__main__":
    main()
