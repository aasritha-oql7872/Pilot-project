"""Evidence-aware reward shaping (experimental; the original reward in reward.py is untouched).

    R_new = R_orig + alpha * first_evidence_found + beta * complete_evidence_chain

R_orig = correct - 0.05 * N_search - 0.10 * N_invalid. Credit is given only for unique, gold-verified
documents that an executed search actually returned, so repeating a search cannot add reward and
irrelevant searches earn nothing. Gold document ids are read from task metadata inside the reward
function only; they are never shown to the model. By default one-hop tasks keep the original reward
exactly, so shaping cannot change one-hop incentives.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceRewardConfig:
    mode: str = "original"       # original | evidence | potential | explore
    alpha: float = 0.10          # first required document retrieved
    beta: float = 0.30           # whole required chain retrieved
    onehop_bonus: bool = False   # also shape single-document (lookup) tasks
    potential_scale: float = 0.40  # c in Phi(s) = c * (#required docs held / #required)
    gamma: float = 0.10          # explore mode: bonus for a valid, non-duplicate second search from partial evidence

    @classmethod
    def from_dict(cls, d):
        return cls(**(d or {}))


def retrieved_set(retrieved):
    """Unique doc ids over all executed searches (`retrieved` = one id list per search)."""
    return {doc_id for ids in retrieved for doc_id in ids}


def evidence_flags(task, retrieved):
    required = task.get("evidence_doc_ids") or []
    if task.get("kind") == "no_search" or not required:
        return {"first_evidence_found": False, "complete_evidence_chain": False, "evidence_fraction": 0.0}
    got = retrieved_set(retrieved)
    return {"first_evidence_found": required[0] in got,
            "complete_evidence_chain": set(required) <= got,
            "evidence_fraction": len(got & set(required)) / len(required)}


def potential(task, docs_held, scale):
    required = set(task.get("evidence_doc_ids") or [])
    return scale * len(required & docs_held) / len(required) if required else 0.0


def potential_bonus(task, retrieved, scale):
    """Sum of Phi(s_{t+1}) - Phi(s_t) over the searches (Phi = 0 before the first search)."""
    held, total = set(), 0.0
    for ids in retrieved:
        new = held | set(ids)
        total += potential(task, new, scale) - potential(task, held, scale)
        held = new
    return total


def second_search_from_partial(task, retrieved):
    """True iff a second executed search followed a first search that left the evidence chain incomplete,
    and the second search returned a different result list than the first (a repeated query earns nothing).
    This depends on the state before the action, not on whether the second search succeeded."""
    required = set(task.get("evidence_doc_ids") or [])
    if len(required) < 2 or len(retrieved) < 2:
        return False
    return not required <= set(retrieved[0]) and list(retrieved[1]) != list(retrieved[0])


def shaped_result(result, task, retrieved, cfg):
    """Return the verifier result extended with evidence fields and the shaped reward."""
    out = {**result, **evidence_flags(task, retrieved)}
    out["original_reward"] = result["reward"]
    multi = len(task.get("evidence_doc_ids") or []) >= 2
    eligible = task.get("kind") != "no_search" and (multi or cfg.onehop_bonus)
    bonus = 0.0
    if eligible and cfg.mode == "evidence":
        bonus = cfg.alpha * out["first_evidence_found"] + cfg.beta * out["complete_evidence_chain"]
    elif eligible and cfg.mode == "explore":  # evidence bonuses + an explicit incentive to continue searching
        bonus = (cfg.alpha * out["first_evidence_found"] + cfg.beta * out["complete_evidence_chain"]
                 + cfg.gamma * second_search_from_partial(task, retrieved))
    elif eligible and cfg.mode == "potential":
        bonus = potential_bonus(task, retrieved, cfg.potential_scale)
    out["evidence_bonus"] = round(bonus, 4)
    out["second_search_from_partial"] = second_search_from_partial(task, retrieved)
    out["reward"] = round(result["reward"] + bonus, 4)
    return out
