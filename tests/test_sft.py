import pytest
import torch
from transformers import AutoTokenizer

from data.generate import DATA_DIR, read_jsonl
from src.sft_data import build_examples
from src.sft_tokens import IGNORE, encode

KW = {"enable_thinking": False}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def examples():
    return build_examples(seed=42)


def spans(labels):
    out, cur = [], []
    for i, l in enumerate(labels):
        if l != IGNORE:
            cur.append(i)
        elif cur:
            out.append(cur); cur = []
    return out + ([cur] if cur else [])


def test_only_training_splits_and_no_prompt_leakage(examples):
    train_ids = {t["task_id"]: t for s in ("train", "twohop_train") for t in read_jsonl(DATA_DIR / f"{s}_tasks.jsonl")}
    assert len(examples) > 100
    for e in examples:
        task = train_ids[e["task_id"]]  # KeyError if an example came from dev/test/sealed
        user = e["messages"][1]["content"]
        if e["kind"] != "no_search":
            assert task["gold"] not in user and "evidence_doc_ids" not in str(e["messages"][:2])


def test_kinds_present_and_search_counts(examples):
    for kind, n in [("twohop", 2), ("lookup", 1), ("no_search", 0)]:
        sub = [e for e in examples if e["kind"] == kind]
        assert sub
        for e in sub:
            assert sum(bool(m.get("tool_calls")) for m in e["messages"]) == n


def test_observations_are_real_retrieval_output(examples):
    from src.environment import ANSWER_FORMATS
    from src.search_tool import Retriever, format_results
    r = Retriever.for_split("twohop_train")
    for e in [x for x in examples if x["kind"] == "twohop"][:10]:
        for call, tool in zip([m for m in e["messages"] if m.get("tool_calls")], [m for m in e["messages"] if m["role"] == "tool"]):
            q = call["tool_calls"][0]["function"]["arguments"]["query"]
            assert tool["content"] == format_results(r.search(q)) + "\n\n" + ANSWER_FORMATS["typed"]


def test_only_assistant_tokens_are_supervised(tok, examples):
    e = next(x for x in examples if x["kind"] == "twohop")
    ids, labels = encode(tok, e["messages"], KW)
    segs = spans(labels)
    assert len(segs) == 3  # call, call, answer
    texts = [tok.decode([ids[i] for i in s]) for s in segs]
    assert all(t.startswith("<tool_call>") and t.endswith("<|im_end|>") for t in texts[:2])
    assert texts[2].startswith("<answer>") and texts[2].endswith("</answer><|im_end|>")
    supervised = tok.decode([l for l in labels if l != IGNORE])
    assert "tool_response" not in supervised and "[" not in "".join(texts[:2]).replace('"', "")  # no doc ids
    for m in e["messages"]:  # tool / system / user text never supervised
        if m["role"] in ("tool", "system", "user"):
            assert m["content"][:60] not in supervised
    assert all(labels[i] == ids[i] for s in segs for i in s)


def test_direct_answer_example_supervises_only_the_answer(tok, examples):
    e = next(x for x in examples if x["kind"] == "no_search")
    ids, labels = encode(tok, e["messages"], KW)
    assert len(spans(labels)) == 1 and tok.decode([l for l in labels if l != IGNORE]).startswith("<answer>")


def test_gradients_flow_from_labelled_tokens_only(tok, examples):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3Config
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=16)
    model = get_peft_model(AutoModelForCausalLM.from_config(cfg),
                           LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    ids, labels = encode(tok, next(x for x in examples if x["kind"] == "twohop")["messages"], KW)
    x, y = torch.tensor([ids]), torch.tensor([labels])
    out = model(input_ids=x, labels=y)
    logits = out.logits[0, :-1]
    tgt = y[0, 1:]
    manual = torch.nn.functional.cross_entropy(logits[tgt != IGNORE], tgt[tgt != IGNORE])
    assert torch.allclose(out.loss, manual, atol=1e-5)  # loss averages over supervised tokens only
    out.loss.backward()
    grads = [p.grad.abs().sum().item() for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and max(grads) > 0
