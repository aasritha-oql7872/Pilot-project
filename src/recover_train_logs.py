"""Locate (never invent) the training log of the original from-base GRPO run, condition A / M1.

    python -m src.recover_train_logs --run outputs/main_v2 --search /content/drive/MyDrive/SearchLite-RL --report submission/tables/original_grpo_log_recovery.md

Order of attempts, stopping at the first that works:
  1. <run>/train_metrics.csv already present.
  2. A train_metrics.csv written by that run, found under the --search roots in a directory named main_v2 or main. Copied
     byte-for-byte, never overwriting (the notebook only restored the adapter, not the CSV, so the file was left on Drive).
  3. The trainer's own console log (run_all.log / train.log): TRL prints one dict per logged step. The first training segment of a
     `run_all.log` is the original run (run_all.sh trains main_v2 first). The values are exactly what the trainer logged; the step
     index is the log-line order (logging_steps=1). The source and this caveat are recorded next to the file.
  4. Otherwise the result is "unavailable", with the list of places searched. Nothing is approximated from aggregate accuracy.
"""

import argparse
import ast
import csv
import hashlib
import re
import shutil
from pathlib import Path

DICT = re.compile(r"\{'(?:loss|train_runtime)'[^{}]*\}")
RUN_ID = re.compile(r'"run_id":\s*"([^"]+)"')
NAMES = ("main_v2", "main")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def parse_segments(text):
    """List of (run_id or None, [row dicts]) for every training run printed in a console log."""
    segs, cur = [], []
    pos = 0
    pending = []  # segments waiting for the run_id JSON printed after training
    for m in DICT.finditer(text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "train_runtime" in d:
            pending.append([cur, m.end()]); cur = []
            continue
        if "reward" in d:
            cur.append({k: num(v) for k, v in d.items()})
    if cur:
        pending.append([cur, len(text)])
    for rows, end in pending:
        nxt = RUN_ID.search(text, end)
        segs.append((nxt.group(1) if nxt else None, rows))
    return [s for s in segs if s[1]]


def write_csv(rows, path):
    rows = [{"step": i, **r} for i, r in enumerate(rows, 1)]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def find_csv(roots):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for name in NAMES:
            for p in [root / "outputs" / name / "train_metrics.csv", root / name / "train_metrics.csv"] + sorted(root.glob(f"*/{name}/train_metrics.csv")):
                if p.exists():
                    return p
    return None


def find_logs(roots):
    out = []
    for root in roots:
        root = Path(root)
        if root.exists():
            out += [p for p in [root / "run_all.log", root / "outputs" / "run_all.log"] if p.exists()]
            out += sorted(root.glob("main*/train.log")) + sorted(root.glob("outputs/main*/train.log"))
    return out


def recover(run, roots, run_id=None):
    run = Path(run)
    dest = run / "train_metrics.csv"
    searched = [str(r) for r in roots]
    if dest.exists():
        return {"status": "present", "path": str(dest), "sha256": sha256(dest), "searched": searched}
    src = find_csv(roots)
    if src:
        run.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)  # dest did not exist: nothing is overwritten
        (run / "train_metrics.source.txt").write_text(f"copied from {src} (sha256 {sha256(src)})\n")
        return {"status": "copied", "path": str(dest), "origin": str(src), "sha256": sha256(dest), "searched": searched}
    for log in find_logs(roots):
        segs = parse_segments(log.read_text(errors="ignore"))
        pick = [s for s in segs if run_id and s[0] == run_id] or (segs[:1] if log.name == "run_all.log" or log.name == "train.log" else [])
        if pick:
            run.mkdir(parents=True, exist_ok=True)
            write_csv(pick[0][1], dest)
            note = (f"reconstructed from the trainer's console log {log} (training segment 1 of {len(segs)}; run_id printed after it: {pick[0][0]}). "
                    f"Every value is what the trainer logged; the step index is the log-line order (logging_steps=1), not read from trainer state.\n")
            (run / "train_metrics.source.txt").write_text(note)
            return {"status": "reconstructed", "path": str(dest), "origin": str(log), "rows": len(pick[0][1]), "run_id": pick[0][0], "caveat": note.strip(), "sha256": sha256(dest), "searched": searched}
    return {"status": "unavailable", "searched": searched}


def describe(res):
    st = res["status"]
    if st == "present":
        return f"Original GRPO training log: found at `{res['path']}` (sha256 `{res['sha256'][:16]}...`)."
    if st == "copied":
        return (f"Original GRPO training log: recovered from Google Drive (`{res['origin']}`), copied unchanged to `{res['path']}` (sha256 `{res['sha256'][:16]}...`). "
                "The earlier figure was missing because the setup cell restored only the adapter from Drive, not the CSV.")
    if st == "reconstructed":
        return (f"Original GRPO training log: reconstructed from the trainer's own console log `{res['origin']}` ({res['rows']} logged steps). {res['caveat']}")
    return ("**Original GRPO (no SFT) reward/loss figure: UNAVAILABLE.** No `train_metrics.csv` for that run and no console log containing its per-step values were found "
            f"under: {', '.join(res['searched']) or 'no search roots'}. Nothing was reconstructed from aggregate accuracy or estimated. The mandatory reward and loss curves are "
            "shown for the M2-initialised GRPO runs (Figure 2), whose full logs are packaged. The original run's *evaluation* results remain available and are unaffected.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/main_v2")
    ap.add_argument("--search", nargs="*", default=["/content/drive/MyDrive/SearchLite-RL"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    res = recover(args.run, args.search, args.run_id)
    text = describe(res)
    print(res["status"].upper() + ":", text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n")


if __name__ == "__main__":
    main()
