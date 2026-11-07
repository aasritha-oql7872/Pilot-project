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


def decide_signal(paths, min_answered=0.8, min_informative=0.15):
    """Pick (temperature, group size) with the most informative GRPO groups under the ORIGINAL reward (training split
    probe), among settings whose rollouts still produce answers. Stop if even the best has too few informative groups."""
    rows = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        for key in ("group_size_4", "group_size_8"):
            if key in r:
                o, e, g = (r[key][m] for m in ("original", "evidence", "grounded"))
                rows.append({"temperature": r["temperature"], "group_size": int(key[-1]), "answered_rate": round(r["answered_rate"], 3),
                             "accuracy": round(r["accuracy"], 3), "informative_groups_original": o["frac_nonzero_std"],
                             "informative_groups_evidence": e["frac_nonzero_std"], "informative_groups_grounded": g["frac_nonzero_std"],
                             "bonus_created_variance_evidence": e["frac_bonus_created_variance"],
                             "bonus_created_variance_grounded": g["frac_bonus_created_variance"],
                             "distinct_trajectories_per_group": o["distinct_trajectories_per_group"]})
    ok = [r for r in rows if r["answered_rate"] >= min_answered]
    best = max(ok, key=lambda r: (r["informative_groups_original"], -r["temperature"], -r["group_size"])) if ok else None
    return {"table": rows, "selected": best, "proceed": bool(best and best["informative_groups_original"] >= min_informative),
            "min_informative": min_informative}


def decide_headroom(path, max_overall=0.85, max_category=0.8):
    eps = json.loads(Path(path).read_text())["episodes"]
    cats = {}
    for e in eps:
        cats.setdefault(e.get("category"), []).append(bool(e["correct"]))
    acc = {c: round(sum(v) / len(v), 3) for c, v in cats.items()}
    overall = sum(bool(e["correct"]) for e in eps) / len(eps)
    return {"overall": round(overall, 3), "by_category": acc, "proceed": overall <= max_overall or min(acc.values()) <= max_category,
            "rule": f"proceed iff overall <= {max_overall} or some category <= {max_category}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["temps", "pilot", "early", "signal", "headroom"])
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    res = (decide_temps(args.inputs) if args.mode == "temps" else decide_pilot(args.inputs[0]) if args.mode == "pilot"
           else decide_signal(args.inputs) if args.mode == "signal" else decide_headroom(args.inputs[0]) if args.mode == "headroom"
           else decide_early(args.inputs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
