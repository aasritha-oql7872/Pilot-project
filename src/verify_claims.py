"""Check the headline claims of the headroom study against the saved evaluation JSONs. Nothing is assumed.

    python -m src.verify_claims --study outputs/study_hard --out claims.md

Each claim is labelled VERIFIED / CONTRADICTED / NOT CHECKABLE (file missing) with the numbers actually found.
"""

import argparse
import json
from pathlib import Path

CLAIMS_TWOHOP = {"M2": [14], "M3": [29], "M4": [29], "M4g": [29, 30]}  # k of 30 on twohop_dev as reported by the user (M4g: across two seeds)
SEEDS = (42, 43)


def kn(path):
    eps = json.loads(Path(path).read_text())["episodes"]
    return sum(bool(e["correct"]) for e in eps), len(eps), eps


def cat_acc(eps, cat):
    rows = [e for e in eps if e.get("category") == cat]
    return (sum(bool(e["correct"]) for e in rows), len(rows)) if rows else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    S = Path(args.study)
    lines, result = ["| Claim | Reported | Found in saved JSONs | Status |", "|---|---|---|---|"], {}

    def row(claim, reported, found, status):
        lines.append(f"| {claim} | {reported} | {found} | {status} |")
        result[claim] = {"reported": reported, "found": found, "status": status}

    def path(split, lab):
        p = S / f"eval_{split}_{lab}.json"
        return p if p.exists() else None

    p = path("twohop_dev", "M2")
    if p:
        k, n, _ = kn(p)
        row("M2 two-hop dev accuracy", "14/30", f"{k}/{n}", "VERIFIED" if (k, n) == (14, 30) else "CONTRADICTED")
    else:
        row("M2 two-hop dev accuracy", "14/30", "eval_twohop_dev_M2.json missing", "NOT CHECKABLE")
    for v in ("M3", "M4", "M4g"):
        found = {}
        for sd in SEEDS:
            p = path("twohop_dev", f"{v}_s{sd}")
            if p:
                k, n, _ = kn(p)
                found[sd] = f"{k}/{n}"
        reported = " and ".join(f"{k}/30" for k in CLAIMS_TWOHOP[v])
        if not found:
            row(f"{v} two-hop dev accuracy", reported, "missing", "NOT CHECKABLE")
        else:
            ks = {int(x.split("/")[0]) for x in found.values()}
            row(f"{v} two-hop dev accuracy", reported, "; ".join(f"seed {s}: {x}" for s, x in found.items()),
                "VERIFIED" if ks == set(CLAIMS_TWOHOP[v]) else "CONTRADICTED (reported value(s) not matched by the per-seed results)")
    # qualitative claims, from hard_dev
    def cats(lab):
        p = path("hard_dev", lab)
        return kn(p)[2] if p else None
    m2 = cats("M2")
    if m2 is None:
        row("GRPO improves distractor/stale tasks", "improved", "eval_hard_dev_M2.json missing", "NOT CHECKABLE")
    else:
        for c in ("distractor_name", "distractor_person", "stale_affiliation", "reverse_two_hop", "unanswerable"):
            b = cat_acc(m2, c)
            afters = {f"{v}_s{sd}": cat_acc(e, c) for v in ("M3", "M4", "M4g") for sd in SEEDS if (e := cats(f"{v}_s{sd}")) is not None}
            if not afters:
                row(f"hard_dev {c}", "-", "no GRPO evals", "NOT CHECKABLE"); continue
            txt = f"M2 {b[0]}/{b[1]}; " + ", ".join(f"{k} {a[0]}/{a[1]}" for k, a in afters.items())
            mean_gain = sum(a[0] for a in afters.values()) / len(afters) - b[0]
            claim = {"reverse_two_hop": "remains challenging", "unanswerable": "remains unsolved"}.get(c, "improved vs M2")
            ok = (mean_gain > 0) if claim == "improved vs M2" else (max(a[0] for a in afters.values()) / max(1, b[1]) < 0.8 if c == "reverse_two_hop" else max(a[0] for a in afters.values()) / max(1, b[1]) < 0.5)
            row(f"hard_dev {c}", claim, txt, "VERIFIED" if ok else "CONTRADICTED")
        tot = {f"{v}_s{sd}": sum(bool(x["correct"]) for x in e) for v in ("M3", "M4", "M4g") for sd in SEEDS if (e := cats(f"{v}_s{sd}")) is not None}
        if tot:
            spread = max(tot.values()) - min(tot.values())
            row("GRPO variants differ by very few hard_dev questions", "few", f"correct counts {tot} (max-min = {spread} of 60)",
                "VERIFIED" if spread <= 4 else "CONTRADICTED")
    row("hard sealed evaluation completed", "not yet", "eval_hard_final_M2.json " + ("present" if path("hard_final", "M2") else "absent"),
        "VERIFIED (not run)" if not path("hard_final", "M2") else "NOTE: hard_final artifacts exist")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
