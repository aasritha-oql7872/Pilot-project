import json

from src.report_categories import episode_metrics, paired_boot, sign_p


def ep(cat, correct, ans, searches=2, kind="twohop", retrieved=(("p",), ("a",)), req=("p", "a"), tool="[a] Maya is affiliated with Orlan Council."):
    msgs = [{"role": "user", "content": "Where does Maya work?"},
            {"role": "tool", "content": tool + "\n\nfmt reminder"}, {"role": "assistant", "content": f"<answer>{ans}</answer>"}]
    return {"category": cat, "kind": kind, "correct": correct, "final_answer": ans, "num_searches": searches, "num_invalid": 0,
            "retrieved_doc_ids": [list(r) for r in retrieved], "evidence_doc_ids": list(req), "messages": msgs}


def test_supported_correct_answer():
    m = episode_metrics(ep("standard", True, "Orlan Council"))
    assert m["correct"] and m["evidence_supported"] and not m["unsupported"] and m["recall"] == 1.0


def test_unsupported_guess_and_partial_recall():
    m = episode_metrics(ep("standard", False, "Zorvel Institute", retrieved=(("p",),), searches=1))
    assert m["unsupported"] and not m["evidence_supported"] and m["recall"] == 0.5


def test_abstention_is_not_unsupported_and_false_abstention_flag():
    right = episode_metrics(ep("unanswerable", True, "insufficient evidence", kind="unanswerable", req=("p",)))
    assert not right["unsupported"] and not right["false_abstain"] and right["evidence_supported"]
    wrong = episode_metrics(ep("standard", False, "insufficient evidence"))
    assert wrong["false_abstain"] and not wrong["unsupported"]


def test_sign_test_and_bootstrap():
    assert sign_p(0, 0) == 1.0 and abs(sign_p(10, 0) - 2 / 1024) < 1e-9 and sign_p(3, 3) == 1.0
    lo, hi = paired_boot([1.0] * 10, [0.0] * 10)
    assert lo == hi == 1.0
