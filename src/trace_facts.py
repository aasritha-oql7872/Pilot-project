"""Observable facts about one saved evaluation episode (tool calls, retrieved ids, answer). Nothing here infers a hidden
reasoning process: every field is computed from the saved messages, the retrieved doc ids and the task metadata.
"""

import json
import re

from data.generate import DATA_DIR, read_jsonl
from src.reward import ABSTAIN_ANSWER, normalize

BRIDGE = re.compile(r"Project\s+[A-Z][a-z]+(?:\s+Prime)?")


def load_tasks(split):
    return {t["task_id"]: t for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl")}


def queries(ep):
    out = []
    for m in ep["messages"]:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            args = (c.get("function") or {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"query": args}
            out.append(str((args or {}).get("query", "")))
    return out


def records(ep):
    """Record texts shown to the agent per search (answer-format reminder stripped)."""
    return [m["content"].split("\n\n")[0] for m in ep["messages"] if m.get("role") == "tool"]


def question(ep):
    return next((m["content"] for m in ep["messages"] if m.get("role") == "user"), "")


def facts(ep, task=None):
    ev, ids = ep["evidence_doc_ids"], ep["retrieved_doc_ids"]
    got = {d for s in ids for d in s}
    ans = ep["final_answer"]
    texts = records(ep)
    q = queries(ep)
    hay = normalize(" ".join([question(ep)] + texts))
    abstained = ans is not None and normalize(ans) == normalize(ABSTAIN_ANSWER)
    decoys = set((task or {}).get("decoy_doc_ids", []))
    first_hit = bool(ev) and bool(ids) and ev[0] in ids[0]
    bridge = None
    if len(ev) >= 2:  # entity the first-hop record names; the second search should mention it
        for sid, text in zip(ids, texts):
            if ev[0] in sid:
                line = next((l for l in text.split("\n") if l.startswith(f"[{ev[0]}]")), "")
                m = BRIDGE.search(line)
                bridge = m.group(0) if m else None
                break
    return {"task_id": ep["task_id"], "category": ep.get("category"), "kind": ep["kind"], "correct": bool(ep["correct"]),
            "searches": ep["num_searches"], "queries": q, "answered": ans is not None, "abstained": abstained,
            "supported": ans is not None and bool(normalize(ans)) and normalize(ans) in hay,
            "first_hop_found": first_hit, "evidence_found": sorted(got & set(ev)), "complete_chain": bool(ev) and set(ev) <= got,
            "second_search": ep["num_searches"] >= 2, "bridge": bridge,
            "second_query_uses_bridge": bool(bridge) and len(q) >= 2 and normalize(bridge) in normalize(q[1]),
            "reversed_order": len(ev) >= 2 and len(ids) >= 2 and ev[1] in ids[0] and ev[0] not in ids[0] and ev[0] in ids[1],
            "answer_from_decoy_record": bool(decoys) and ans is not None and any(
                normalize(ans) in normalize(t) and any(d in sid for d in decoys) for sid, t in zip(ids, texts)),
            "duplicate_query": len(q) == 2 and normalize(q[0]) == normalize(q[1]), "invalid": ep["num_invalid"] > 0,
            "reward": ep["reward"], "final_answer": ans, "gold": ep["gold_answer"]}


def classify_twohop(f):
    """Mutually exclusive failure label for an answerable multi-hop episode (first matching rule)."""
    if f["correct"]:
        return "correct"
    if not f["answered"]:
        return "no_parseable_answer"
    if f["searches"] == 0:
        return "answered_without_search"
    if f["searches"] == 1:
        return "premature_stop_after_first_hop" if f["first_hop_found"] else "wrong_first_search_then_stop"
    if not f["first_hop_found"] and not f["evidence_found"]:
        return "first_search_missed_evidence"
    if not f["complete_chain"]:
        return "second_search_missed_second_hop" if not f["second_query_uses_bridge"] else "second_search_with_bridge_still_missed"
    if f["answer_from_decoy_record"]:
        return "evidence_complete_answer_from_decoy"
    return "evidence_complete_wrong_answer"


def classify_unanswerable(f):
    if not f["answered"]:
        return "no_parseable_answer"
    if f["abstained"]:
        return "correct_abstention" if f["searches"] >= 1 else "abstained_without_searching"
    if f["searches"] == 0:
        return "answered_without_search"
    return "answered_with_value_from_retrieved_record" if f["supported"] else "invented_unsupported_answer"
