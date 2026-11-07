"""Phase-2 tests: advantage arithmetic, reward-component behaviour, optimizer updates, checkpoint loading."""

import torch

from src.reward import score
from src.reward_evidence import EvidenceRewardConfig, shaped_result
from src.train_grpo import group_stats, trajectory_signature


def trl_advantages(rewards, g):
    """TRL 1.17 GRPOTrainer (scale_rewards='group'): (r - group mean) / (group std + 1e-4), std = unbiased (nanstd)."""
    r = torch.tensor(rewards, dtype=torch.float32).view(-1, g)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)


# ---- why original-reward and evidence-reward GRPO can produce identical updates ---------------------------------
def test_group_constant_bonus_cancels_in_advantages():
    orig = [0.9, -0.05, 0.9, 0.9,   -0.05, -0.05, -0.05, -0.05]
    bonus = [0.4, 0.4, 0.4, 0.4,    0.1, 0.1, 0.1, 0.1]          # same bonus for every rollout of a group
    shaped = [a + b for a, b in zip(orig, bonus)]
    assert torch.allclose(trl_advantages(orig, 4), trl_advantages(shaped, 4), atol=1e-4)
    assert trl_advantages([-0.05] * 4, 4).abs().max() == 0          # identical rewards -> zero advantage -> no gradient


def test_bonus_that_varies_within_group_changes_the_advantages():
    orig = [-0.05, -0.05, -0.05, -0.05]                              # nobody correct: no signal without the bonus
    shaped = [0.05, -0.05, -0.05, -0.05]                             # one rollout found the first document
    assert trl_advantages(orig, 4).abs().max() == 0
    assert trl_advantages(shaped, 4).abs().max() > 1


def test_evidence_bonus_is_constant_when_all_rollouts_retrieve_the_same_documents():
    task = {"kind": "twohop", "evidence_doc_ids": ["p", "a"], "gold": "X", "question": "q"}
    cfg = EvidenceRewardConfig(mode="evidence")

    def r(answer):
        call = {"type": "function", "function": {"name": "search", "arguments": {"query": "q"}}}
        comp = [{"role": "assistant", "content": "", "tool_calls": [call]}] * 2 + [{"role": "assistant", "content": f"<answer>{answer}</answer>"}]
        return shaped_result(score(comp, "X", 2), task, [["p"], ["a"]], cfg, [["x"], ["y"]])
    rs = [r("X"), r("Y"), r("X"), r("Y")]
    assert {x["evidence_bonus"] for x in rs} == {0.4}                # full chain for all: bonus cannot discriminate
    orig = [x["original_reward"] for x in rs]
    assert torch.allclose(trl_advantages(orig, 4), trl_advantages([x["reward"] for x in rs], 4), atol=1e-4)


def test_group_stats_counts():
    gs = group_stats([1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0], [("a",)] * 4 + [("a",), ("b",), ("a",), ("b",)], 4)
    # group 2: shaped rewards differ but the original rewards are all equal -> the bonus created the variance
    assert gs["groups"] == 2 and gs["frac_nonzero_std"] == 0.5 and gs["frac_orig_nonzero_std"] == 0.0
    assert gs["distinct_trajectories_per_group"] == 1.5 and gs["frac_bonus_created_variance"] == 0.5


def test_trajectory_signature():
    comp = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": {"query": "q1"}}}]},
            {"role": "tool", "content": "r"}, {"role": "assistant", "content": "<answer>Z</answer>"}]
    assert trajectory_signature(comp) == ("q1", "<answer>Z</answer>")


# ---- grounded reward ---------------------------------------------------------------------------------------------
def _episode(answer, n_search=1):
    call = {"type": "function", "function": {"name": "search", "arguments": {"query": "q"}}}
    comp = [{"role": "assistant", "content": "", "tool_calls": [call]} for _ in range(n_search)]
    comp.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
    return comp


TASK = {"kind": "twohop", "evidence_doc_ids": ["p", "a"], "gold": "Orlan Council", "question": "Where does Maya Kornel work?"}
G = EvidenceRewardConfig(mode="grounded", unsupported_penalty=0.25)


def test_grounded_penalises_only_unsupported_guesses():
    texts = [["Maya Kornel is affiliated with Orlan Council."]]
    ok = shaped_result(score(_episode("Orlan Council"), "Orlan Council", 1), TASK, [["a"]], G, texts)
    assert ok["answer_supported"] and ok["unsupported_penalty"] == 0 and ok["reward"] == 0.95
    misread = shaped_result(score(_episode("Maya Kornel"), "Orlan Council", 1), TASK, [["a"]], G, texts)
    assert misread["answer_supported"] and misread["reward"] == -0.05          # supported but wrong: no extra penalty
    guess = shaped_result(score(_episode("Zorvel Institute"), "Orlan Council", 1), TASK, [["a"]], G, texts)
    assert guess["answer_supported"] is False and guess["reward"] == -0.30     # hallucination costs extra


def test_abstention_is_never_penalised_as_unsupported_and_scores_when_gold():
    task = {**TASK, "kind": "unanswerable", "gold": "insufficient evidence", "evidence_doc_ids": ["p"]}
    r = shaped_result(score(_episode("insufficient evidence", 2), task["gold"], 2), task, [["p"], ["p"]], G, [["a"], ["b"]])
    assert r["abstained"] and r["correct"] and r["unsupported_penalty"] == 0 and r["reward"] == 0.9
    wrong = shaped_result(score(_episode("insufficient evidence", 2), "Orlan Council", 2), TASK, [["p"], ["p"]], G, [["a"], ["b"]])
    assert wrong["abstained"] and not wrong["correct"] and wrong["reward"] == -0.1  # abstaining on an answerable task earns 0


def test_grounded_uses_no_gold_evidence_ids():
    a = shaped_result(score(_episode("Orlan Council"), "Orlan Council", 1), TASK, [["a"]], G, [["Orlan Council"]])
    b = shaped_result(score(_episode("Orlan Council"), "Orlan Council", 1), {**TASK, "evidence_doc_ids": ["zz", "yy"]}, [["a"]], G, [["Orlan Council"]])
    assert a["reward"] == b["reward"]


# ---- optimizer / checkpoint mechanics on a tiny Qwen3 -------------------------------------------------------------
def _tiny_peft(seed=0):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3Config
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=16)
    return get_peft_model(AutoModelForCausalLM.from_config(cfg),
                          LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))


def _pg_loss(model, ids, adv):
    logp = torch.log_softmax(model(input_ids=ids).logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
    return -(adv[:, None] * logp).mean()


def test_zero_advantage_gives_no_gradient_and_no_update():
    model = _tiny_peft()
    params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    before = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    ids = torch.randint(0, 64, (4, 10))
    _pg_loss(model, ids, torch.zeros(4)).backward()
    assert max(p.grad.abs().max().item() for p in params) == 0
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, params))        # Adam with zero gradient does not move weights


def test_nonzero_advantage_moves_exactly_the_lora_parameters():
    model = _tiny_peft()
    base_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    ids = torch.randint(0, 64, (4, 10))
    _pg_loss(model, ids, torch.tensor([1.0, -1.0, 0.5, -0.5])).backward()
    opt.step()
    assert any(not torch.equal(lora_before[n], p) for n, p in model.named_parameters() if n in lora_before)
    assert all(torch.equal(base_before[n], p) for n, p in model.named_parameters() if n in base_before)  # base frozen


def test_adapter_reload_for_grpo_restores_weights_and_is_trainable(tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = _tiny_peft(seed=1)
    for n, p in model.named_parameters():                                  # make the adapter non-trivial (B starts at 0)
        if "lora_" in n:
            p.data.normal_(0, 0.1)
    model.save_pretrained(tmp_path / "adapter")
    base = AutoModelForCausalLM.from_config(model.base_model.model.config)
    loaded = PeftModel.from_pretrained(base, tmp_path / "adapter", is_trainable=True)   # same call as train_grpo init_adapter
    a = {n: p for n, p in model.named_parameters() if "lora_" in n}
    b = {n: p for n, p in loaded.named_parameters() if "lora_" in n}
    assert a.keys() == b.keys() and all(torch.allclose(a[n], b[n]) for n in a)
    assert sum(p.numel() for p in loaded.parameters() if p.requires_grad) > 0


def test_two_runs_with_same_seed_and_same_advantages_end_with_identical_weights():
    """Same seed + same (group-constant-shifted) advantages -> bitwise-identical adapters: the mechanism behind the
    previously observed identical C and D adapters."""
    def run(shift):
        model = _tiny_peft(seed=3)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
        torch.manual_seed(5)
        ids = torch.randint(0, 64, (4, 10))
        adv = trl_advantages([0.9 + shift, -0.05 + shift, 0.9 + shift, -0.05 + shift], 4)
        for _ in range(3):
            opt.zero_grad(); _pg_loss(model, ids, adv).backward(); opt.step()
        return [p.detach().clone() for p in params]
    a, b = run(0.0), run(0.4)
    assert max((x - y).abs().max().item() for x, y in zip(a, b)) < 1e-6


def test_signal_probe_summary_groups_by_task():
    from src.signal_probe import summarize
    mk = lambda o, e, g, sig: {"reward": {"original": o, "evidence": e, "grounded": g}, "sig": (sig,)}
    # task 1: eight identical rollouts (no signal anywhere); task 2: outcomes differ in the first group only
    rolls = [mk(-0.05, -0.05, -0.05, "a")] * 8 + [mk(0.9, 1.3, 0.9, "a"), mk(-0.05, 0.05, -0.30, "b"), mk(0.9, 1.3, 0.9, "a"), mk(0.9, 1.3, 0.9, "a")] + [mk(0.9, 1.3, 0.9, "a")] * 4
    s = summarize(rolls, 8)["group_size_4"]
    assert s["original"]["frac_nonzero_std"] == 0.25 and s["grounded"]["frac_nonzero_std"] == 0.25
    assert summarize(rolls, 8)["group_size_8"]["original"]["frac_nonzero_std"] == 0.5
