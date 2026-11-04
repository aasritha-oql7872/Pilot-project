"""Stage decisions for the exploration studies. Prints a JSON decision and writes it to --out; always exits 0
(the calling shell script reads `proceed` / `selected`), so a "no" is recorded, never silently skipped.

    python -m src.explore_gate temps  --out d.json  explore_T0.7.json explore_T1.0.json ...
    python -m src.explore_gate pilot  --out d.json  outputs/study_rl_only/F2_pilot
    python -m src.explore_gate early  --out d.json  outputs/study_early/probe_step*.json

temps: choose the sampling temperature with the highest second-search rate among non-degenerate settings
       (answered_rate >= 0.5), ties -> lowest temperature; proceed = some rollout ever made a 2nd search.
pilot: proceed to the full run only if the short pilot produced at least one two-hop rollout with 2 searches.
early: choose the checkpoint whose SAMPLED (T=0.7) second-search rate is in [lo, hi] and closest to the midpoint;
       selected = null if none qualifies (nothing is forced).
"""

import argparse
import json
import re
from pathlib import Path

from src.traj_stats import load, stats


def decide_temps(paths, min_rate=0.01):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    table = [{"temperature": r["temperature"], "rollouts": r["rollouts"], "second_search_rate": r["second_search_rate"],
              "tasks_with_any_two_search_rollout": r["tasks_with_any_two_search_rollout"],
              "answered_rate": r.get("answered_rate"), "invalid_rate": r.get("invalid_rate")} for r in rows]
    ok = [t for t in table if (t["answered_rate"] is None or t["answered_rate"] >= 0.5)]
    best = max(ok, key=lambda t: (t["second_search_rate"], -t["temperature"])) if ok else None
    return {"table": table, "selected_temperature": best["temperature"] if best else None,
            "max_second_search_rate": max((t["second_search_rate"] for t in table), default=0.0),
            "proceed": bool(best and best["second_search_rate"] >= min_rate), "min_rate": min_rate}


def decide_pilot(run_dir):
    s = stats(load(Path(run_dir) / "trajectories.jsonl"))["twohop"]
    return {"run": str(run_dir), "twohop_rollouts": s["rollouts"], "second_search": s["second_search"],
            "valid_second_search": s["valid_second_search"], "nonzero_std_groups": s["nonzero_std_groups"],
            "proceed": bool(s["second_search"])}


def decide_early(paths, lo=0.2, hi=0.7):
    table = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        step = int(re.search(r"step(\d+)", str(p)).group(1))
        table.append({"step": step, "second_search_rate": round(r["second_search_rate"], 4), "correct_rate": round(r["correct_rate"], 4),
                      "tasks_with_any_two_search_rollout": r["tasks_with_any_two_search_rollout"]})
    table.sort(key=lambda t: t["step"])
    ok = [t for t in table if lo <= t["second_search_rate"] <= hi]
    mid = (lo + hi) / 2
    pick = min(ok, key=lambda t: (abs(t["second_search_rate"] - mid), t["step"])) if ok else None
    return {"table": table, "target_range": [lo, hi], "selected": pick["step"] if pick else None,
            "note": None if pick else "no checkpoint has a sampled second-search rate in the target range; nothing is forced"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["temps", "pilot", "early"])
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    res = (decide_temps(args.inputs) if args.mode == "temps" else decide_pilot(args.inputs[0]) if args.mode == "pilot"
           else decide_early(args.inputs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
