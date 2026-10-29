import pytest

from data.generate import DATA_DIR, read_jsonl
from src.environment import MAX_SEARCHES, SearchEnv, make_env_factory, run_tool_calls, score_episode
from src.trace_schema import REQUIRED_FIELDS, make_trace, validate_trace

TASK = next(t for t in read_jsonl(DATA_DIR / "train_tasks.jsonl") if t["kind"] == "lookup")


def fresh_env():
    env = make_env_factory("train")()
    env.reset(**TASK)
    return env


def test_valid_query_returns_short_text():
    env = fresh_env()
    out = env.search(TASK["question"])
    assert TASK["evidence_doc_ids"][0] in out and len(out) < 700
    assert env.num_searches == 1 and env.retrieved[0][0] == TASK["evidence_doc_ids"][0]


def test_bad_arguments_rejected():
    env = fresh_env()
    with pytest.raises(ValueError):
        env.search("  ")
    msgs = run_tool_calls(env, [{"type": "function", "function": {"name": "search", "arguments": {"q": "x"}}},
                                {"type": "function", "function": {"name": "browse", "arguments": {}}}])
    assert all("error" in m["content"] for m in msgs) and env.num_searches == 0


def test_call_cap():
    env = fresh_env()
    for _ in range(MAX_SEARCHES):
        env.search("Project")
    with pytest.raises(ValueError, match="budget"):
        env.search("Project")
    assert env.num_searches == MAX_SEARCHES


def test_reset_clears_state():
    env = fresh_env()
    env.search("Project")
    env.reset(**TASK)
    assert env.num_searches == 0 and env.retrieved == []


def test_trace_integrity():
    env = fresh_env()
    env.search(TASK["question"])
    completion = [{"role": "assistant", "content": f"<answer>{TASK['gold']}</answer>"}]
    trace = make_trace("t", TASK, "m", 3, 42, [{"role": "user", "content": TASK["question"]}],
                       completion, env, score_episode(env, completion))
    validate_trace(trace)
    assert set(REQUIRED_FIELDS) <= set(trace) and trace["step"] == 3 and trace["correct"]
    with pytest.raises(ValueError):
        validate_trace({k: v for k, v in trace.items() if k != "reward"})


def test_plot_fails_without_metrics(tmp_path):
    from src.plot_metrics import load_metrics
    with pytest.raises(SystemExit):
        load_metrics(tmp_path)


def test_only_search_is_exposed_as_tool():
    """TRL turns every public env method except reset/get_reward into a tool."""
    import inspect
    public = [n for n, _ in inspect.getmembers(SearchEnv, inspect.isfunction) if not n.startswith("_")]
    assert sorted(public) == ["reset", "search"]
