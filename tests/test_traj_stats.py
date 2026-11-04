from src.traj_stats import stats


def row(step, tid, kind, n, reward, orig=None, correct=False, invalid=0):
    ev = ["p", "a"] if kind == "twohop" else ["p"]
    ret = [["p"], ["a"]][:n] if kind == "twohop" else ([["p"]] if n else [])
    return {"step": step, "task_id": tid, "kind": kind, "num_searches": n, "num_invalid": invalid, "reward": reward,
            "original_reward": reward if orig is None else orig, "correct": correct,
            "evidence_doc_ids": ev, "retrieved_doc_ids": ret}


def test_zero_variance_groups_and_second_search_rate():
    rows = [row(1, "t1", "twohop", 1, -0.05) for _ in range(4)]               # all one-search: no signal
    rows += [row(2, "t2", "twohop", 1, -0.05), row(2, "t2", "twohop", 2, 0.9, correct=True)] * 2
    s = stats(rows)["twohop"]
    assert s["groups"] == 2 and s["nonzero_std_groups"] == 0.5
    assert s["second_search"] == 2 / 8 and s["complete_chain"] == 2 / 8


def test_bonus_created_variance_is_detected():
    # original rewards identical (-0.05), the evidence bonus makes one rollout differ
    rows = [row(1, "t", "twohop", 1, -0.05), row(1, "t", "twohop", 1, 0.05, orig=-0.05)]
    s = stats(rows)["twohop"]
    assert s["orig_nonzero_std_groups"] == 0 and s["nonzero_std_groups"] == 1 and s["bonus_created_variance"] == 1


def test_onehop_controls():
    rows = [row(1, "l", "lookup", 1, 0.95, correct=True), row(1, "n", "no_search", 1, -0.05),
            row(2, "n2", "no_search", 0, 1.0, correct=True)]
    c = stats(rows)["controls"]
    assert c["no_search_search_rate"] == 0.5 and c["lookup_search_rate"] == 1.0
