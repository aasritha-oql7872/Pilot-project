import pytest

from src.reward import normalize, parse_final_answer, score


def answer(text):
    return {"role": "assistant", "content": text}


def call(query="Project Orion lead"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": query}}}]}


def tool(text="[proj_orion] Project Orion principal investigator is Maya Rao."):
    return {"role": "tool", "name": "search", "content": text}


def test_normalize():
    assert normalize(" The Velanser Institute. ") == normalize("velanser institute")
    assert normalize("Maya  Rao!") == "maya rao"


def test_correct_without_search():
    assert score([answer("<answer>2024</answer>")], "2024", 0)["reward"] == 1.0


def test_incorrect_is_zero_before_costs():
    assert score([answer("<answer>2023</answer>")], "2024", 0)["reward"] == 0.0


def test_search_cost():
    r = score([call(), tool(), answer("<answer>Maya Rao</answer>")], "Maya Rao", 1)
    assert r["correct"] and r["num_searches"] == 1 and r["reward"] == pytest.approx(0.95)
    r = score([call(), tool(), call(), tool(), answer("<answer>Maya Rao</answer>")], "Maya Rao", 2)
    assert r["reward"] == pytest.approx(0.90)


def test_missing_answer_tag_is_invalid():
    r = score([answer("Maya Rao")], "Maya Rao", 0)
    assert not r["correct"] and r["num_invalid"] == 1 and r["reward"] == pytest.approx(-0.10)


def test_unexecuted_or_failed_calls_are_invalid():
    # Third call never executed (cap reached) and the episode ends without an answer.
    r = score([call(), tool(), call(), tool(), call()], "Maya Rao", 2)
    assert r["num_invalid"] == 2 and r["reward"] == pytest.approx(-0.10 - 0.20)
    # A call the environment rejected (e.g. bad arguments) still costs as invalid.
    r = score([call(""), tool("{'error': 'query must be a non-empty string.'}"),
               answer("<answer>Maya Rao</answer>")], "Maya Rao", 0)
    assert r["num_invalid"] == 1 and r["reward"] == pytest.approx(0.90)


def test_parse_final_answer():
    assert parse_final_answer([answer("I think <answer> Maya Rao </answer>")]) == "Maya Rao"
    assert parse_final_answer([call()]) is None
    assert parse_final_answer([]) is None
