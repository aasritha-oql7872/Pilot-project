"""Finite-horizon MDP reference for the stop-or-search decision (diagnostic, NOT a competitor to Qwen).

States (about the evidence the agent holds): S0 none, S1 partial (first hop only), S2 complete.
Actions: search (cost c, at most H times) or answer. Bellman optimality, finite horizon k = searches left:

    V(s, 0) = p_ans(s)
    V(s, k) = max( p_ans(s),  -c + sum_s' P(s'|s) V(s', k-1) )

ORACLE LABEL: the state is defined with the gold evidence ids, which the real agent cannot see, so the
resulting policy is an upper-bound reference under privileged information. Transition probabilities P are
estimated on a *development* split by a scripted BM25 query policy (no Qwen). p_ans(s) is the answer
accuracy in state s, estimated from a model's own evaluation episodes (so the MDP describes that model).

    python -m src.mdp_baseline --transitions-split twohop_dev --episodes outputs/main_v2/eval_twohop_trained.json \
        --forced outputs/audit/forced_base.json
"""

import argparse
import json
from pathlib import Path

from data.generate import DATA_DIR, read_jsonl
from src.audit import person_from
from src.search_tool import Retriever

STATES = ["S0_none", "S1_partial", "S2_complete"]


def solve(P, p_ans, cost, horizon):
    """P[s][s'] after one search (row-stochastic, 3x3); p_ans[s]; returns V[k][s], policy[k][s]."""
    V = [list(p_ans)]
    policy = [["answer"] * 3]
    for k in range(1, horizon + 1):
        row, act = [], []
        for s in range(3):
            go = -cost + sum(P[s][t] * V[k - 1][t] for t in range(3))
            row.append(max(p_ans[s], go))
            act.append("search" if go > p_ans[s] else "answer")
        V.append(row)
        policy.append(act)
    return V, policy


def estimate_transitions(split):
    """Scripted chain-aware policy: search 1 = 'Project X principal investigator', search 2 = '<person> affiliation'.
    S2 stays S2; S1 -> S2 with the measured second-search success rate; S0 -> S1/S2 from the first search."""
    r = Retriever.for_split(split)
    n = first = both = 0
    for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl"):
        pi_id, aff_id = t["evidence_doc_ids"]
        d1 = r.search(f"Project {t['subject']} principal investigator")
        pi = next((d for d in d1 if d["doc_id"] == pi_id), None)
        person = person_from(pi["text"]) if pi else None
        d2 = r.search(f"{person} affiliation") if person else []
        n += 1
        first += pi is not None
        both += pi is not None and any(d["doc_id"] == aff_id for d in d2)
    p01 = first / n                      # S0 -> S1 on the first search
    p12 = both / first if first else 0.0  # S1 -> S2 on the second search
    return [[1 - p01, p01, 0.0], [0.0, 1 - p12, p12], [0.0, 0.0, 1.0]]


def answer_accuracy_by_state(report_path, forced_path=None):
    """p_ans(s) from a model's own two-hop episodes grouped by the evidence held when it answered."""
    eps = [e for e in json.loads(Path(report_path).read_text())["episodes"] if e["kind"] == "twohop"]
    buckets = {0: [], 1: [], 2: []}
    for e in eps:
        got = {d for ids in e["retrieved_doc_ids"] for d in ids}
        buckets[len(got & set(e["evidence_doc_ids"]))].append(e["correct"])
    out, source = [], []
    for i in range(3):
        if buckets[i]:
            out.append(sum(buckets[i]) / len(buckets[i])); source.append(f"model episodes (n={len(buckets[i])})")
        elif i == 2 and forced_path:
            out.append(json.loads(Path(forced_path).read_text())["accuracy"]); source.append("forced-search probe")
        else:
            raise SystemExit(f"no observations of state {STATES[i]}; pass --forced for the complete-evidence state")
    return out, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions-split", default="twohop_dev")
    parser.add_argument("--episodes", required=True, help="evaluation JSON whose episodes define p_ans")
    parser.add_argument("--forced", default=None)
    parser.add_argument("--cost", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    P = estimate_transitions(args.transitions_split)
    p_ans, source = answer_accuracy_by_state(args.episodes, args.forced)
    V, policy = solve(P, p_ans, args.cost, args.horizon)
    res = {"label": "ORACLE-STATE reference (uses gold evidence ids); not comparable to Qwen's information",
           "states": STATES, "P_one_search": P, "p_answer": p_ans, "p_answer_source": source,
           "search_cost": args.cost, "V": V, "optimal_policy_by_searches_left": policy}
    print(json.dumps(res, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
