from src.reward import score
from src.reward_evidence import EvidenceRewardConfig, evidence_flags, potential_bonus, shaped_result

TASK = {"kind": "twohop", "evidence_doc_ids": ["a_pi", "a_affiliation"], "gold": "X Council"}
ONEHOP = {"kind": "lookup", "evidence_doc_ids": ["b_year"], "gold": "1999"}
NOSEARCH = {"kind": "no_search", "evidence_doc_ids": [], "gold": "1999"}
CFG = EvidenceRewardConfig(mode="evidence", alpha=0.1, beta=0.3)


def result(gold, answer, n):
    call = {"type": "function", "function": {"name": "search", "arguments": {"query": "q"}}}
    completion = [{"role": "assistant", "content": "", "tool_calls": [call]} for _ in range(n)]
    completion.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
    return score(completion, gold, n)


def test_arithmetic_complete_chain_correct():
    r = shaped_result(result("X Council", "X Council", 2), TASK, [["a_pi", "z"], ["a_affiliation"]], CFG)
    assert r["original_reward"] == 0.9 and r["evidence_bonus"] == 0.4 and r["reward"] == 1.3


def test_partial_evidence_gets_only_alpha():
    r = shaped_result(result("X Council", "Bob", 1), TASK, [["a_pi", "z"]], CFG)
    assert (r["first_evidence_found"], r["complete_evidence_chain"]) == (True, False)
    assert r["evidence_bonus"] == 0.1 and r["reward"] == 0.05


def test_irrelevant_search_earns_nothing():
    r = shaped_result(result("X Council", "Bob", 1), TASK, [["z", "y"]], CFG)
    assert r["evidence_bonus"] == 0 and r["reward"] == -0.05


def test_repeated_retrieval_cannot_accumulate():
    once = shaped_result(result("X", "Bob", 1), TASK, [["a_pi"]], CFG)["evidence_bonus"]
    twice = shaped_result(result("X", "Bob", 2), TASK, [["a_pi"], ["a_pi"]], CFG)["evidence_bonus"]
    assert once == twice == 0.1


def test_chain_beats_one_search_even_when_both_wrong():
    one = shaped_result(result("X", "Bob", 1), TASK, [["a_pi"]], CFG)["reward"]
    two = shaped_result(result("X", "Bob", 2), TASK, [["a_pi"], ["a_affiliation"]], CFG)["reward"]
    assert two > one


def test_onehop_and_nosearch_keep_original_reward():
    for task, retrieved, n in [(ONEHOP, [["b_year"]], 1), (NOSEARCH, [], 0)]:
        base = result(task["gold"], task["gold"], n)
        r = shaped_result(base, task, retrieved, CFG)
        assert r["reward"] == base["reward"] and r["evidence_bonus"] == 0


def test_onehop_bonus_is_opt_in():
    cfg = EvidenceRewardConfig(mode="evidence", onehop_bonus=True)
    r = shaped_result(result("1999", "1999", 1), ONEHOP, [["b_year"]], cfg)
    assert r["evidence_bonus"] == 0.4


def test_original_mode_is_identity():
    base = result("X Council", "X Council", 2)
    assert shaped_result(base, TASK, [["a_pi"], ["a_affiliation"]], EvidenceRewardConfig())["reward"] == base["reward"]


def test_potential_telescopes_to_final_coverage():
    assert potential_bonus(TASK, [["a_pi"], ["a_affiliation"]], 0.4) == 0.4 or abs(
        potential_bonus(TASK, [["a_pi"], ["a_affiliation"]], 0.4) - 0.4) < 1e-9
    assert abs(potential_bonus(TASK, [["a_pi"], ["a_pi"]], 0.4) - 0.2) < 1e-9


def test_reward_uses_only_executed_search_results():
    # Gold ids matter only inside the reward; evidence_flags never needs the model's text.
    assert evidence_flags(TASK, [])["evidence_fraction"] == 0.0


def test_no_gold_leakage_to_model_visible_text():
    from src.environment import SearchEnv, SYSTEM_PROMPT, make_prompt
    from src.search_tool import Retriever
    task = {"kind": "twohop", "gold": "Kabraol Laboratory", "subject": "Serethvel",
            "evidence_doc_ids": ["serethvel_pi", "serethvel_affiliation"], "question": "Where does the lead researcher of Project Serethvel work?"}
    env = SearchEnv(Retriever.for_split("twohop"))
    env.reset(**task)
    visible = SYSTEM_PROMPT + make_prompt(task["question"])[1]["content"] + env.search("Project Serethvel principal investigator")
    assert "serethvel_affiliation" not in visible.replace("serethvel_pi", "") and "evidence_doc_ids" not in visible
    assert task["gold"] not in visible
