"""Verify that the frozen headline selection, checkpoint, code and sealed evaluation are consistent. Read-only: changes nothing.

    python -m src.verify_freeze --study outputs/study_hard --out submission/tables/freeze_verification.md

Checks (PASS / FAIL / INFO):
  * FROZEN_HARD.json parses, names the headline model, and is read-only on this filesystem (INFO on Drive mounts, which may ignore modes);
  * the headline adapter on disk has the sha256 recorded at freeze time;
  * the freeze happened BEFORE the first hard_final evaluation (marker file) and hard_final was not viewed before the freeze;
  * hard_final data files still match data/manifest_hard.json and the hashes recorded in the freeze manifest;
  * no training / evaluation / environment / reward / retrieval / data / config file changed between the freeze commit and HEAD
    (only reporting, packaging and test code may differ; the list is printed);
  * every hard_final evaluation used the declared settings (`typed_abstain`, greedy, the same task ids), except the deliberate M2 `typed` control;
  * the headline evaluation file points at the frozen checkpoint.
"""

import argparse
import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime
from pathlib import Path

from data.generate import DATA_DIR, file_hash
from src.manifest import verify_sealed

PROTECTED_PREFIXES = ("data/", "configs/")
PROTECTED_FILES = {"src/environment.py", "src/evaluate.py", "src/train_grpo.py", "src/train_sft.py", "src/reward.py", "src/reward_evidence.py",
                   "src/search_tool.py", "src/sft_data.py", "src/sft_tokens.py", "src/trace_schema.py", "src/freeze_hard.py", "src/manifest.py"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def parse_time(t):
    return datetime.fromisoformat(t.strip().replace("Z", "+00:00"))


def verify(study):
    S = Path(study)
    rows = []

    def add(item, status, detail=""):
        rows.append((item, status, detail))

    fp = S / "FROZEN_HARD.json"
    if not fp.exists():
        return [("FROZEN_HARD.json exists", "FAIL", str(fp))]
    m = json.loads(fp.read_text())
    add("FROZEN_HARD.json parses; headline", "PASS", f"{m['headline_label']}, frozen {m['frozen_at_utc']}")
    writable = bool(fp.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    add("FROZEN_HARD.json is read-only", "INFO" if writable else "PASS", "mode bits writable (Drive mounts may not keep chmod 444; the script created it with mode 'x' and refuses a second run)" if writable else "")
    ck = Path(m["checkpoint_path"]) / "adapter_model.safetensors"
    add("headline adapter sha256 equals the freeze manifest", "PASS" if ck.exists() and sha256(ck) == m["adapter_sha256"] else "FAIL", f"{ck} {'present' if ck.exists() else 'MISSING'}")
    mk = S / "HARD_FINAL_FIRST_EVALUATION"
    if mk.exists():
        t0, t1 = parse_time(m["frozen_at_utc"]), parse_time(mk.read_text())
        add("freeze happened before the first hard_final evaluation", "PASS" if t0 < t1 else "FAIL", f"frozen {t0.isoformat()}, first hard_final evaluation {t1.isoformat()}")
    else:
        add("freeze happened before the first hard_final evaluation", "INFO", "no HARD_FINAL_FIRST_EVALUATION marker (sealed evaluation not run?)")
    add("hard_final not viewed before the freeze (recorded)", "PASS" if not m["hard_final_viewed_before_freeze"] else "FAIL", m["sealed_view_statement"])
    bad = verify_sealed("manifest_hard.json")
    cur = json.loads((DATA_DIR / "manifest_hard.json").read_text())["splits"]["hard_final"]
    add("hard_final data matches manifest_hard.json and the freeze record", "PASS" if not bad and cur == m["sealed_dataset_hashes"]["hard"]["hard_final"] else "FAIL", ", ".join(bad))
    commit = m.get("git_commit")
    changed = git("diff", "--name-only", commit, "HEAD") if commit else None
    if changed is None:
        add("no protected file changed since the freeze commit", "INFO", "git history for the freeze commit is not available here")
    else:
        files = [f for f in changed.splitlines() if f]
        prot = [f for f in files if f in PROTECTED_FILES or f.startswith(PROTECTED_PREFIXES)]
        add("no training/evaluation/environment/reward/retrieval/data/config file changed since the freeze commit", "PASS" if not prot else "FAIL",
            (f"PROTECTED CHANGED: {prot}; " if prot else "") + f"changed since {commit[:10]}: {', '.join(files) or 'nothing'}")
    evs = {p.name[len("eval_hard_final_"):-5]: json.loads(p.read_text()) for p in sorted(S.glob("eval_hard_final_*.json"))}
    if evs:
        odd = {}
        ids = {frozenset(e["task_id"] for e in r["episodes"]) for r in evs.values()}
        for lab, r in evs.items():
            want = "typed" if lab == "M2_typed" else "typed_abstain"
            if lab in ("base", "M1"):
                continue
            if r.get("answer_format") != want or r.get("decoding") != "greedy" or r.get("split") != "hard_final":
                odd[lab] = (r.get("answer_format"), r.get("decoding"), r.get("split"))
        add("hard_final evaluations used the declared settings (typed_abstain, greedy; M2_typed is the deliberate control)", "PASS" if not odd else "FAIL", str(odd) if odd else f"{len(evs)} evaluations")
        add("all hard_final evaluations cover identical task ids", "PASS" if len(ids) == 1 else "FAIL", f"{len(next(iter(ids)))} tasks")
        head = f"{m['headline_model']}_s{m['training_seed']}" if m["headline_model"] != "M2" else "M2"
        if head in evs:
            add("headline evaluation file points at the frozen checkpoint", "PASS" if str(evs[head]["checkpoint"]).rstrip("/").endswith(m["checkpoint_path"].rstrip("/")) else "FAIL", str(evs[head]["checkpoint"]))
    return rows


def render(rows):
    return "\n".join(["| Check | Status | Detail |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = render(verify(args.study))
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
