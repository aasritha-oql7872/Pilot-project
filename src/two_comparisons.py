"""The two comparisons that bound how much of the GRPO gain is attributable to RL rather than prompt design.

    python -m src.two_comparisons --study outputs/study_hard --split hard_final --headline M3_s42 --replications M3_s43 \
        --out submission/tables/two_comparisons.json --md submission/tables/two_comparisons.md \
        --expect M2=34/120 M2_typed=75/120 M3_s42=82/120 M3_s43=81/120 M3_s42:reverse_two_hop=4/20 M3_s42:unanswerable=0/20

Comparison A (fixed evaluation prompt):   M2 under `typed_abstain`   vs  GRPO model under `typed_abstain`.
Comparison B (stronger SFT baseline):     M2 under `typed` (the reminder it was trained with)  vs  GRPO model under `typed_abstain`.
Prompt effect (C):                        M2 under `typed_abstain` vs M2 under `typed`  (same weights, only the reminder changes).

Neither A nor B isolates the effect of RL: the GRPO models also trained on `hard_train` under `typed_abstain`, and there is no
SFT-on-`hard_train` control. All comparisons are paired by task id: contingency counts, difference, paired bootstrap interval,
exact two-sided McNemar test (= exact sign test on the discordant tasks). Every number is read from the saved evaluation JSONs.
"""

import argparse
import json
import re
from pathlib import Path

from src.report_categories import paired_boot, sign_p
from src.summarize_study import wilson

SETTING_KEYS = ("split", "decoding", "prompt_variant", "answer_format")


def load(path):
    rep = json.loads(Path(path).read_text())
    return rep, {e["task_id"]: e for e in rep["episodes"]}


def settings(rep):
    return {k: rep.get(k) for k in SETTING_KEYS}


def contrast(ref, mod, cats=None):
    """Paired comparison of two {task_id: episode} dicts (optionally restricted to categories)."""
    ids = sorted(i for i in set(ref) & set(mod) if cats is None or ref[i].get("category") in cats)
    r = [bool(ref[i]["correct"]) for i in ids]
    m = [bool(mod[i]["correct"]) for i in ids]
    both = sum(x and y for x, y in zip(r, m))
    neither = sum((not x) and (not y) for x, y in zip(r, m))
    mod_only = sum(y and not x for x, y in zip(r, m))
    ref_only = sum(x and not y for x, y in zip(r, m))
    n = len(ids)
    lo, hi = paired_boot([float(v) for v in m], [float(v) for v in r]) if n else (0.0, 0.0)
    p = sign_p(mod_only, ref_only)
    return {"n": n, "ref_correct": sum(r), "model_correct": sum(m), "both_correct": both, "both_wrong": neither,
            "model_only_correct": mod_only, "ref_only_correct": ref_only,
            "ref_acc_pct": 100 * sum(r) / n if n else None, "model_acc_pct": 100 * sum(m) / n if n else None,
            "ref_wilson_pct": [100 * v for v in wilson(sum(r), n)] if n else None,
            "model_wilson_pct": [100 * v for v in wilson(sum(m), n)] if n else None,
            "diff_pts": 100 * (sum(m) - sum(r)) / n if n else None, "ci95_pts": [100 * lo, 100 * hi],
            "mcnemar_exact_p": p, "ci_excludes_zero": bool(n and (lo > 0 or hi < 0)),
            "significant_at_0.05": bool(n and p < 0.05 and (lo > 0 or hi < 0)),
            "ids_equal": True}


def analyse(study, split, headline, replications=()):
    S = Path(study)
    p2, p2t = S / f"eval_{split}_M2.json", S / f"eval_{split}_M2_typed.json"
    reps = {"M2": load(p2), "M2_typed": load(p2t)}
    for lab in (headline, *replications):
        reps[lab] = load(S / f"eval_{split}_{lab}.json")
    res = {"split": split, "headline": headline, "replications": list(replications), "settings": {k: settings(v[0]) for k, v in reps.items()},
           "checkpoints": {k: v[0].get("checkpoint") for k, v in reps.items()},
           "task_id_sets_equal": len({frozenset(v[1]) for v in reps.values()}) == 1, "comparisons": {}, "by_category": {}}

    def add(key, ref, mod, role):
        c = contrast(reps[ref][1], reps[mod][1])
        c["ids_equal"] = set(reps[ref][1]) == set(reps[mod][1])
        c.update({"role": role, "reference": ref, "model": mod})
        diff = {k: (reps[ref][0].get(k), reps[mod][0].get(k)) for k in SETTING_KEYS if reps[ref][0].get(k) != reps[mod][0].get(k)}
        c["settings_that_differ"] = diff
        res["comparisons"][key] = c

    for lab in (headline, *replications):
        add(f"A: {lab} vs M2 (both typed_abstain)", "M2", lab, "A")
        add(f"B: {lab} (typed_abstain) vs M2 (typed)", "M2_typed", lab, "B")
    add("C: M2 typed vs M2 typed_abstain (same weights)", "M2", "M2_typed", "C")
    cats = sorted({e.get("category") for e in reps[headline][1].values() if e.get("category")})
    for c in cats:
        res["by_category"][c] = {"A": contrast(reps["M2"][1], reps[headline][1], {c}), "B": contrast(reps["M2_typed"][1], reps[headline][1], {c}),
                                 "C": contrast(reps["M2"][1], reps["M2_typed"][1], {c})}
    res["category_counts"] = {lab: {c: [sum(bool(e["correct"]) for e in eps.values() if e.get("category") == c),
                                        sum(e.get("category") == c for e in eps.values())] for c in cats} for lab, (_, eps) in reps.items()}
    res["overall_counts"] = {lab: [sum(bool(e["correct"]) for e in eps.values()), len(eps)] for lab, (_, eps) in reps.items()}
    return res


def check_expected(res, expect):
    """expect: ['M2=34/120', 'M3_s42:reverse_two_hop=4/20']. Returns rows (item, expected, found, ok)."""
    rows = []
    for item in expect:
        m = re.fullmatch(r"([^=:]+)(?::([^=]+))?=(\d+)/(\d+)", item)
        if not m:
            rows.append((item, item, "unparseable expectation", False)); continue
        lab, cat, k, n = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        src = res["category_counts"].get(lab, {}).get(cat) if cat else res["overall_counts"].get(lab)
        rows.append((f"{lab}" + (f" / {cat}" if cat else ""), f"{k}/{n}", f"{src[0]}/{src[1]}" if src else "missing", bool(src) and src == [k, n]))
    return rows


def pct(x):
    return f"{x:.1f}%"


def render(res, expect_rows=None):
    L = [f"**Paired comparisons on `{res['split']}` (n={res['overall_counts']['M2'][1]}).** Contingency counts are per question id; the interval is a paired bootstrap of the accuracy difference; "
         "the p-value is the exact two-sided McNemar test (exact sign test on the discordant questions). Statistics are reported whether or not they are significant.", ""]
    if not res["task_id_sets_equal"]:
        L += ["> **WARNING:** the evaluations do not cover identical task ids.", ""]
    L += ["| Comparison | reference | model | both correct | both wrong | model-only correct | reference-only correct | difference [95% paired CI] | McNemar p | significant? |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in res["comparisons"].items():
        L.append(f"| {key} | {c['ref_correct']}/{c['n']} ({pct(c['ref_acc_pct'])}) | {c['model_correct']}/{c['n']} ({pct(c['model_acc_pct'])}) | {c['both_correct']} | {c['both_wrong']} | "
                 f"{c['model_only_correct']} | {c['ref_only_correct']} | {c['diff_pts']:+.1f} pts [{c['ci95_pts'][0]:+.1f}, {c['ci95_pts'][1]:+.1f}] | {c['mcnemar_exact_p']:.3g} | "
                 f"{'yes' if c['significant_at_0.05'] else 'no'} |")
    L += ["", "Evaluation settings of each report (the only intended difference is the reminder in comparisons B and C):", "",
          "| model | split | decoding | prompt variant | answer format | checkpoint |", "|---|---|---|---|---|---|"]
    for lab, s in res["settings"].items():
        L.append(f"| {lab} | {s['split']} | {s['decoding']} | {s['prompt_variant']} | {s['answer_format']} | `{res['checkpoints'][lab]}` |")
    bad = [k for k, c in res["comparisons"].items() if (c["role"] == "A" and c["settings_that_differ"]) or not c["ids_equal"]]
    L += ["", ("Comparison A settings are identical apart from the checkpoint; task ids are identical across all evaluations." if not bad and res["task_id_sets_equal"]
               else f"> **WARNING:** settings or task ids differ in: {bad}")]
    L += ["", f"**Per category for the frozen model `{res['headline']}`** (A = vs M2 typed_abstain, B = vs M2 typed). Counts are per-category n.", "",
          "| category | n | M2 (typed_abstain) | M2 (typed) | " + res["headline"] + " | A: model-only / M2-only, diff | B: model-only / M2typed-only, diff |", "|---|---|---|---|---|---|---|"]
    for cat, d in res["by_category"].items():
        a, b = d["A"], d["B"]
        L.append(f"| {cat} | {a['n']} | {a['ref_correct']}/{a['n']} | {b['ref_correct']}/{b['n']} | {a['model_correct']}/{a['n']} | "
                 f"{a['model_only_correct']} / {a['ref_only_correct']}, {a['diff_pts']:+.0f} pts | {b['model_only_correct']} / {b['ref_only_correct']}, {b['diff_pts']:+.0f} pts |")
    if expect_rows is not None:
        L += ["", "**Reported numbers checked against the raw evaluation JSONs:**", "", "| item | reported | found in raw JSON | status |", "|---|---|---|---|"]
        L += [f"| {a} | {b} | {c} | {'VERIFIED' if ok else 'MISMATCH'} |" for a, b, c, ok in expect_rows]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--split", default="hard_final")
    ap.add_argument("--headline", required=True, help="frozen model label, e.g. M3_s42")
    ap.add_argument("--replications", nargs="*", default=[])
    ap.add_argument("--expect", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    res = analyse(args.study, args.split, args.headline, args.replications)
    rows = check_expected(res, args.expect) if args.expect else None
    if rows is not None:
        res["expected_check"] = [{"item": a, "reported": b, "found": c, "verified": ok} for a, b, c, ok in rows]
    text = render(res, rows)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    if args.md:
        Path(args.md).write_text(text + "\n")
    if rows is not None and not all(ok for *_, ok in rows):
        raise SystemExit("reported numbers do not match the raw evaluation JSONs")


if __name__ == "__main__":
    main()
