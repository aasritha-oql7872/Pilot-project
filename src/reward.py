"""Deterministic verifier: R = 1.0 * correct - 0.05 * N_search - 0.10 * N_invalid."""

import re
import string

CORRECT_REWARD = 1.0
SEARCH_COST = 0.05
INVALID_COST = 0.10
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(the|project)\b", " ", text)
    return " ".join(text.split())


def parse_final_answer(completion):
    """The answer inside <answer>...</answer> of the last assistant turn, or None if malformed.

    A turn that still requests a tool, or has no answer tag, is not a final answer.
    """
    last = completion[-1] if completion else {}
    if last.get("role") != "assistant" or last.get("tool_calls"):
        return None
    match = ANSWER_RE.search(last.get("content") or "")
    return match.group(1).strip() if match else None


def count_tool_calls(completion):
    return sum(len(m.get("tool_calls") or []) for m in completion if m.get("role") == "assistant")


def score(completion, gold, num_searches):
    """Score one finished episode.

    num_searches is the number of searches the environment actually executed. Any other
    tool call (unknown tool, bad arguments, over the call cap, never executed) is invalid,
    and so is a missing or malformed final answer.
    """
    answer = parse_final_answer(completion)
    correct = answer is not None and normalize(answer) == normalize(gold)
    num_invalid = (count_tool_calls(completion) - num_searches) + (answer is None)
    reward = CORRECT_REWARD * correct - SEARCH_COST * num_searches - INVALID_COST * num_invalid
    return {"final_answer": answer, "correct": correct, "num_searches": num_searches,
            "num_invalid": num_invalid, "reward": round(reward, 4)}
