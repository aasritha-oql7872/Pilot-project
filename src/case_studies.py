"""Case studies extracted from SAVED evaluation traces (never written by hand, never edited).

    python -m src.case_studies --split hard_final --orig-grpo A_twohop_dev.json --sft B_twohop_dev.json --orig-split twohop_dev \
        --before eval_hard_final_M2.json --after eval_hard_final_M3_s42.json --out case_studies.md

Selection is rule-based and deterministic: the FIRST matching task id in sorted order (no cherry-picking), shown with the rule.
Each case prints the question, checkpoint, actual search queries, retrieved records, the final message, expected answer,
correctness, reward, and a one-line description of the observable behaviour computed from the trace. No chain of thought is
reconstructed; the model's visible output is quoted as saved.
"""

import argparse
import json
from pathlib import Path

from src.trace_facts import classify_twohop, classify_unanswerable, facts, load_tasks, question, records


def load(path):
    rep = json.loads(Path(path).read_text())
    return rep, {e["task_id"]: e for e in rep["episodes"]}


def final_message(ep):
    msgs = [m for m in ep["messages"] if m.get("role") == "assistant"]
    return (msgs[-1].get("content") or "").strip() if msgs else ""


def observed(f, label):
    bits = [f"{f['searches']} search{'es' if f['searches'] != 1 else ''}"]
    if f["queries"]:
        bits.append("queries: " + " -> ".join(repr(q) for q in f["queries"]))
    if f["kind"] != "unanswerable":
        bits.append("first-hop record retrieved: " + ("yes" if f["first_hop_found"] else "no"))
        bits.append("complete evidence chain retrieved: " + ("yes" if f["complete_chain"] else "no"))
    if f["abstained"]:
        bits.append("answered 'insufficient evidence'")
    elif not f["answered"]:
        bits.append("no parseable <answer> tag")
    elif not f["supported"]:
        bits.append("answer text appears in neither the question nor any retrieved record")
    return f"{label}. " + "; ".join(bits) + "."


def render(title, rule, rep, ep, task, label):
    f = facts(ep, task)
    ev = set(ep["evidence_doc_ids"]); dec = set((task or {}).get("decoy_doc_ids", []))
    lines = [f"### {title}", f"*Selection rule:* {rule}", "",
             f"- **Task:** `{ep['task_id']}` (category: {ep.get('category') or ep['kind']})",
             f"- **Question:** {question(ep)}",
             f"- **Model / checkpoint:** `{rep['checkpoint']}` ({rep.get('decoding')}, answer_format `{rep.get('answer_format')}`)",
             f"- **Search calls:** " + (" ; ".join(f"`{q}`" for q in f["queries"]) or "none")]
    for i, (ids, text) in enumerate(zip(ep["retrieved_doc_ids"], records(ep)), 1):
        tag = ", ".join(f"{d}{' [gold evidence]' if d in ev else ' [decoy]' if d in dec else ''}" for d in ids)
        lines.append(f"- **Retrieved (search {i}):** {tag}\n  > " + text.replace("\n", "\n  > "))
    lines += [f"- **Final message (verbatim):** {final_message(ep)!r}",
              f"- **Extracted answer:** {ep['final_answer']!r} | **Expected:** {ep['gold_answer']!r} | **Correct:** {ep['correct']} | **Reward:** {ep['reward']}",
              f"- **Observable behaviour:** {observed(f, label)}", ""]
    return "\n".join(lines)


def first(eps, pred):
    for tid in sorted(eps):
        if pred(eps[tid]):
            return eps[tid]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="split of --before/--after (hard_dev or hard_final)")
    ap.add_argument("--orig-grpo", default=None, help="eval JSON of the original RL-only GRPO model on a two-hop split")
    ap.add_argument("--sft", default=None, help="eval JSON of the SFT-only model on the SAME split")
    ap.add_argument("--orig-split", default="twohop_dev")
    ap.add_argument("--before", required=True, help="M2 eval on --split")
    ap.add_argument("--after", required=True, help="frozen GRPO model eval on --split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks, parts, missing = load_tasks(args.split), ["# Case studies (real saved traces)\n"], []

    if args.orig_grpo and args.sft:
        (ra, ea), (rb, eb) = load(args.orig_grpo), load(args.sft)
        t = first(ea, lambda e: e["kind"] == "twohop" and e["num_searches"] == 1 and not e["correct"] and eb.get(e["task_id"], {}).get("correct")
                  and eb[e["task_id"]]["num_searches"] >= 2)
        if t:
            tid = t["task_id"]; otasks = load_tasks(args.orig_split)
            parts.append(render("Case 1: original GRPO (no SFT) stops after one search",
                                "first two-hop task (sorted id) where the original GRPO model made 1 search and was wrong while the SFT model searched twice and was right",
                                ra, ea[tid], otasks.get(tid), "Original GRPO"))
            parts.append(render("Case 2: SFT learned the second search (same task as case 1)", "same task id as case 1",
                                rb, eb[tid], otasks.get(tid), "SFT only"))
        else:
            missing.append("cases 1-2: no two-hop task where original GRPO stopped after 1 search and SFT answered correctly with 2")
    else:
        missing.append("cases 1-2: --orig-grpo / --sft evaluation files not supplied")

    (rb_, eb_), (ra_, ea_) = load(args.before), load(args.after)
    gained = first(ea_, lambda e: e["kind"] == "twohop" and e["correct"] and not eb_[e["task_id"]]["correct"])
    if gained:
        tid = gained["task_id"]
        parts.append(render("Case 3a: before GRPO (M2, wrong)", "first multi-hop task (sorted id) that M2 got wrong and the GRPO model got right", rb_, eb_[tid], tasks.get(tid), "M2 (before)"))
        parts.append(render("Case 3b: after GRPO (same task)", "same task id as 3a", ra_, ea_[tid], tasks.get(tid), "after GRPO"))
    else:
        missing.append("case 3: no multi-hop task fixed by GRPO on this split")
    dist = first(ea_, lambda e: e.get("category") in ("stale_affiliation", "distractor_name", "distractor_person") and e["correct"] and not eb_[e["task_id"]]["correct"]) \
        or first(ea_, lambda e: e.get("category") in ("stale_affiliation", "distractor_name", "distractor_person") and e["correct"])
    if dist:
        parts.append(render("Case 4: success on a hard distractor / stale-affiliation question",
                            "first distractor or stale-affiliation task (sorted id) that GRPO got right and M2 got wrong (else any it got right)",
                            ra_, dist, tasks.get(dist["task_id"]), "after GRPO"))
    else:
        missing.append("case 4: GRPO model solved no distractor/stale task")
    rev = first(ea_, lambda e: e.get("category") == "reverse_two_hop" and not e["correct"])
    if rev:
        parts.append(render("Case 5: failed reverse two-hop question", "first reverse_two_hop task (sorted id) the GRPO model got wrong",
                            ra_, rev, tasks.get(rev["task_id"]), "after GRPO"))
    else:
        missing.append("case 5: GRPO model got every reverse_two_hop question right on this split")
    un = first(ea_, lambda e: e.get("category") == "unanswerable" and not e["correct"])
    if un:
        parts.append(render("Case 6: unanswerable question answered without support", "first unanswerable task (sorted id) the GRPO model did not abstain on",
                            ra_, un, tasks.get(un["task_id"]), "after GRPO"))
    else:
        missing.append("case 6: GRPO model abstained correctly on every unanswerable question")
    if missing:
        parts.append("## Cases not available\n" + "\n".join(f"- {m}" for m in missing))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(parts) + "\n")
    print(f"wrote {args.out}; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
