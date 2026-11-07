"""SearchLite episode environment, shared by GRPO training and evaluation.

Training: TRL's GRPOTrainer builds one SearchEnv per rollout (environment_factory),
calls reset(**task_row), exposes `search` as the tool, and masks tool output tokens
out of the policy loss. Evaluation: run_episode() replays the same protocol greedily.
"""

import torch
from trl.chat_template_utils import (add_response_schema, get_training_chat_template,
                                     is_chat_template_prefix_preserving, parse_response)

from src.reward import ABSTAIN_ANSWER, score
from src.search_tool import Retriever, format_results

MAX_SEARCHES = 2
ANSWER_FORMAT = ("The answer must be only the requested value (a name, organization, city or year), "
                 "not a sentence: <answer>value</answer>.")
# Diagnostic variant (not used by default): names the requested *type* of value.
ANSWER_FORMATS = {
    "default": ANSWER_FORMAT,
    "typed": ("The answer must be only the requested value, not a sentence. Give exactly what the question asks "
              "for: if it asks where someone works, answer with the institution, not a person's name: "
              "<answer>value</answer>."),
    # Hard-benchmark reminder (environment v2): same as typed, plus an explicit way to decline.
    "typed_abstain": ("The answer must be only the requested value, not a sentence. Give exactly what the question asks "
                      "for: if it asks where someone works, answer with the institution, not a person's name: "
                      "<answer>value</answer>. If the records do not contain the answer, reply "
                      f"<answer>{ABSTAIN_ANSWER}</answer>."),
}
SYSTEM_PROMPT = (
    "You answer questions about fictional research projects and people. You have no prior "
    "knowledge of them: the only sources of facts are the question itself and a private "
    "knowledge base you can query with the search function. If the question already states the "
    "needed fact, answer directly without searching. Otherwise first call the search function "
    f"using a <tool_call> block as described below (at most {MAX_SEARCHES} calls), then answer "
    "from the returned records. Never guess. Write your final answer as "
    "<answer>your answer</answer>, containing only the answer."
)


# Prompt-only ablation (condition B): same weights, extra instruction, no training.
PROMPT_VARIANTS = {
    "default": "",
    "continue": (" Some questions need two searches: if a record names a person but does not say what the "
                 "question asks about that person (for example where they work), search again using that "
                 "person's name before answering. Do not answer with an intermediate name."),
}


def make_prompt(question, variant="default"):
    return [{"role": "system", "content": SYSTEM_PROMPT + PROMPT_VARIANTS[variant]},
            {"role": "user", "content": question}]


class SearchEnv:
    def __init__(self, retriever, answer_format=ANSWER_FORMAT):
        self.retriever = retriever
        self.answer_format = answer_format
        self.reset()

    def reset(self, **task):
        self.task = task  # includes the private gold label; never shown to the model
        self.num_searches = 0
        self.retrieved_text = []  # one list of record texts per executed search (for support checks; never gold)
        self.retrieved = []  # doc_ids returned by each search, in order
        return None  # the prompt already holds the question

    def search(self, query: str) -> str:
        """Search the knowledge base for records about a project or person.

        Args:
            query: Keywords describing the fact to look up, e.g. "Project Orion principal investigator".

        Returns:
            Up to two matching records, each prefixed with its document id.
        """
        if self.num_searches >= MAX_SEARCHES:
            raise ValueError(f"Search budget exhausted ({MAX_SEARCHES} searches). Give your final answer.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        self.num_searches += 1
        results = self.retriever.search(query)
        self.retrieved.append([r["doc_id"] for r in results])
        self.retrieved_text.append([r["text"] for r in results])
        return format_results(results) + "\n\n" + self.answer_format


def score_episode(env, completion):
    return score(completion, env.task["gold"], env.num_searches)


def make_env_factory(split, answer_format="default"):
    retriever = Retriever.for_split(split)  # one read-only index shared by all rollouts
    return lambda: SearchEnv(retriever, ANSWER_FORMATS[answer_format])


def run_tool_calls(env, tool_calls):
    """Execute one assistant turn's tool calls; errors become tool messages, as in TRL."""
    messages = []
    for call in tool_calls:
        fn = call.get("function", {})
        try:
            if fn.get("name") != "search":
                raise ValueError(f"Tool {fn.get('name')} not found.")
            result = env.search(**fn.get("arguments", {}))
        except Exception as e:
            result = {"error": str(e)}
        messages.append({"role": "tool", "name": fn.get("name"), "content": str(result)})
    return messages


def rollout_chat_template(tokenizer):
    """The template GRPOTrainer renders with: Qwen3's own template is not prefix-preserving
    across tool turns, so TRL swaps in its training-safe variant."""
    if is_chat_template_prefix_preserving(tokenizer):
        return None
    return get_training_chat_template(tokenizer)


def tool_suffix_ids(tokenizer, tool_messages, chat_template_kwargs):
    """Tokens that append tool results and reopen the assistant turn (mirrors GRPOTrainer)."""
    dummy = [{"role": "user", "content": "dummy"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {}}}]}]
    prefix = tokenizer.apply_chat_template(dummy, tokenize=True, return_dict=False, **chat_template_kwargs)
    full = tokenizer.apply_chat_template(dummy + tool_messages, add_generation_prompt=True,
                                         tokenize=True, return_dict=False, **chat_template_kwargs)
    eos = [i for i, t in enumerate(prefix) if t == tokenizer.eos_token_id]
    prefix = prefix[: eos[-1] + 1] if eos else prefix
    assert full[: len(prefix)] == prefix
    return full[len(prefix):]


@torch.no_grad()
def run_episode(model, tokenizer, env, task, max_completion_length, chat_template_kwargs, temperature=None,
                prompt_variant="default"):
    """Greedy rollout (or sampled, if `temperature` is given) with the training protocol: <=MAX_SEARCHES tool rounds and one
    max_completion_length budget shared by generated and tool tokens."""
    if getattr(tokenizer, "response_template", None) is None and getattr(tokenizer, "response_schema", None) is None:
        add_response_schema(tokenizer)
    chat_template_kwargs = {**chat_template_kwargs, "chat_template": rollout_chat_template(tokenizer)}
    env.reset(**task)
    messages = make_prompt(task["question"], prompt_variant)
    prompt_ids = tokenizer.apply_chat_template(messages, tools=[env.search], add_generation_prompt=True,
                                               tokenize=True, return_dict=False, **chat_template_kwargs)
    completion_ids, completion = [], []
    for rounds in range(MAX_SEARCHES + 1):
        context = torch.tensor([prompt_ids + completion_ids], device=model.device)
        out = model.generate(context, max_new_tokens=max_completion_length - len(completion_ids),
                             do_sample=temperature is not None,
                             **({"temperature": temperature, "top_p": 1.0, "top_k": 0} if temperature else {}),
                             pad_token_id=tokenizer.pad_token_id)
        new_ids = out[0, context.shape[1]:].tolist()
        turn = parse_response(tokenizer, new_ids, prefix=prompt_ids + completion_ids, tools=[env.search])
        completion_ids += new_ids
        completion.append(turn)
        if not turn.get("tool_calls") or rounds == MAX_SEARCHES:
            break
        tool_messages = run_tool_calls(env, turn["tool_calls"])
        suffix = tool_suffix_ids(tokenizer, tool_messages, chat_template_kwargs)
        if len(completion_ids) + len(suffix) >= max_completion_length:
            break  # TRL drops tool results that overflow the budget; the episode ends unanswered
        completion += tool_messages
        completion_ids += suffix
    return messages, completion, score_episode(env, completion)
