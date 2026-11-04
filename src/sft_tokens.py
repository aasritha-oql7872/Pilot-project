"""Tokenize a trajectory with the SAME template GRPO rollouts use, labelling only assistant-generated tokens.

For assistant message i: prefix = render(messages[:i], add_generation_prompt=True), full = render(messages[:i+1]).
The tokens full[len(prefix):first <|im_end|>] are exactly what the policy generates at rollout time (the
generation-prompt header, tool results and system/user text are context only, label -100).
"""

from src.environment import SearchEnv, rollout_chat_template
from src.search_tool import Retriever

IGNORE = -100
_TOOL = None


def search_tool():
    """The tool schema shown to the model (any env instance exposes the same `search` signature)."""
    global _TOOL
    if _TOOL is None:
        _TOOL = SearchEnv(Retriever([{"doc_id": "x", "text": "x"}])).search
    return _TOOL


def encode(tokenizer, messages, chat_template_kwargs):
    kw = {**chat_template_kwargs, "chat_template": rollout_chat_template(tokenizer)}
    render = lambda msgs, gen=False: tokenizer.apply_chat_template(
        msgs, tools=[search_tool()], add_generation_prompt=gen, tokenize=True, return_dict=False, **kw)
    ids = render(messages)
    labels = [IGNORE] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = render(messages[:i], gen=True)
        full = render(messages[: i + 1])
        assert full[: len(prefix)] == prefix, "template is not prefix-preserving at an assistant turn"
        end = len(full)
        if tokenizer.eos_token_id in full[len(prefix):]:
            end = len(prefix) + full[len(prefix):].index(tokenizer.eos_token_id) + 1  # keep <|im_end|>
        assert ids[:end] == full[:end]
        labels[len(prefix):end] = ids[len(prefix):end]
    return ids, labels
