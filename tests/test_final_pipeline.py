"""Final-submission pipeline on SYNTHETIC fixtures (fabricated only inside tests, to exercise the code paths)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from data.generate import read_jsonl, DATA_DIR
from src.freeze_hard import select
from src.trace_facts import classify_twohop, classify_unanswerable, facts, load_tasks

ROOT = Path(__file__).resolve().parents[1]


def corpus(split):
    return {d["doc_id"]: d["text"] for d in read_jsonl(DATA_DIR / f"{split}_corpus.jsonl")}


def make_ep(task, corp, queries, retrieved, answer, correct, reward=0.0):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": task["question"]}]
    for q, ids in zip(queries, retrieved):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": {"query": q}}}]})
        msgs.append({"role": "tool", "name": "search", "content": "\n".join(f"[{i}] {corp[i]}" for i in ids) + "\n\nreminder"})
    msgs.append({"role": "assistant", "content": f"<answer>{answer}</answer>" if answer is not None else "I do not know"})
    return {"run_id": "t", "task_id": task["task_id"], "split": task["split"], "kind": task["kind"], "category": task["category"], "model": "m",
            "step": None, "seed": 1, "messages": msgs, "evidence_doc_ids": task["evidence_doc_ids"], "retrieved_doc_ids": retrieved,
            "gold_answer": task["gold"], "final_answer": answer, "correct": correct, "num_searches": len(queries), "num_invalid": 0, "reward": reward}


def fixture_eval(path, policy):
    tasks = list(read_jsonl(DATA_DIR / "hard_dev_tasks.jsonl")); corp = corpus("hard_dev")
    eps = [policy(t, corp) for t in tasks]
    Path(path).write_text(json.dumps({"checkpoint": "fake/" + Path(path).stem, "split": "hard_dev", "decoding": "greedy",
                                      "answer_format": "typed_abstain", "metrics": {}, "episodes": eps}))
    return eps


def perfect(t, c):
    ev = t["evidence_doc_ids"]
    if t["kind"] == "unanswerable":
        return make_ep(t, c, ["q"], [ev], "insufficient evidence", True, 0.95)
    return make_ep(t, c, ["person", f"Project {t['subject']}"], [[ev[0]], [ev[-1]]], t["gold"], True, 0.9)


def lazy(t, c):  # one search, wrong answer; invents an answer when unanswerable
    ev = t["evidence_doc_ids"]
    return make_ep(t, c, ["q"], [[ev[0]]], "zzz unseen" if t["kind"] == "unanswerable" else "wrongname", False, -0.05)


def test_classifiers():
    tasks = load_tasks("hard_dev"); corp = corpus("hard_dev")
    rev = next(t for t in tasks.values() if t["category"] == "reverse_two_hop")
    f = facts(lazy(rev, corp), rev)
    assert classify_twohop(f) == "premature_stop_after_first_hop"
    assert classify_twohop(facts(perfect(rev, corp), rev)) == "correct"
    un = next(t for t in tasks.values() if t["category"] == "unanswerable")
    assert classify_unanswerable(facts(lazy(un, corp), un)) == "invented_unsupported_answer"
    assert classify_unanswerable(facts(perfect(un, corp), un)) == "correct_abstention"
    miss = make_ep(rev, corp, ["a", "b"], [[rev["evidence_doc_ids"][0]], [rev["evidence_doc_ids"][0]]], "x", False)
    assert classify_twohop(facts(miss, rev)) == "second_search_missed_second_hop"
    nope = make_ep(rev, corp, [], [], None, False)
    assert classify_twohop(facts(nope, rev)) == "no_parseable_answer"


def test_select_rule():
    m2 = {"hard_dev": 50.0, "dev": 95.0, "twohop_dev": 46.7}
    row = lambda h, d=95.0, t=96.7: {"hard_dev": h, "dev": d, "twohop_dev": t}
    # all comparable -> simplest (M3)
    tab = {"M3": {42: row(80.0), 43: row(78.3)}, "M4": {42: row(81.7), 43: row(80.0)}, "M4g": {42: row(83.3), 43: row(81.7)}}
    assert select(tab, m2)[0] == "M3"
    # a clearly better complex variant wins
    tab["M3"] = {42: row(60.0), 43: row(61.7)}
    assert select(tab, m2)[0] == "M4g" or select(tab, m2)[0] == "M4"
    # nobody beats M2 by 5 pts -> M2 stays
    flat = {v: {42: row(51.7), 43: row(50.0)} for v in ("M3", "M4", "M4g")}
    assert select(flat, m2)[0] == "M2"
    # regression on dev disqualifies
    bad = {"M3": {42: row(80.0, d=85.0), 43: row(80.0, d=85.0)}}
    assert select(bad, m2)[0] == "M2"
    # a seed that does not beat M2 disqualifies a variant from the competitive set
    split = {"M3": {42: row(80.0), 43: row(50.0)}, "M4": {42: row(80.0), 43: row(79.0)}}
    assert select(split, m2)[0] == "M4"


@pytest.fixture
def study(tmp_path):
    S = tmp_path / "study"; S.mkdir()
    fixture_eval(S / "eval_hard_dev_M2.json", lazy)
    fixture_eval(S / "eval_hard_dev_M3_s42.json", perfect)
    fixture_eval(S / "eval_hard_dev_M2_typed.json", lazy)
    for sd, name in ((42, "M3_s42"),):
        d = S / name; d.mkdir()
        with open(d / "train_metrics.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "loss", "reward", "reward_std", "frac_reward_zero_std", "grad_norm", "searchlite/searches_per_episode"])
            for i in range(1, 21):
                w.writerow([i, 0.01 * (i % 3), 0.5 + 0.02 * i, 0.1, 0.5, 0.3, 1.0 + 0.03 * i])
    return S


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)


def test_rl_improvement_and_failures(study, tmp_path):
    r = run("src.rl_improvement", "--study", str(study), "--split", "hard_dev", "--after", "M3_s42", "--out", str(tmp_path / "rl.json"))
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "rl.json").read_text())
    c = res["comparisons"]["M3_s42"]
    assert c["after"] == "60/60" and c["before"].startswith("0/") and c["lost"] == 0 and c["diff_pts"] == 100.0
    assert res["behaviour"]["M3_s42"]["second_search_rate"] == 1.0 and res["behaviour"]["M2"]["premature_stop_rate"] > 0.8
    assert "format_control" in res
    r = run("src.failure_analysis", "--split", "hard_dev", "--json", str(tmp_path / "f.json"),
            f"M2={study / 'eval_hard_dev_M2.json'}", f"M3={study / 'eval_hard_dev_M3_s42.json'}")
    assert r.returncode == 0, r.stderr
    f = json.loads((tmp_path / "f.json").read_text())
    assert f["M2"]["reverse_two_hop"]["labels"] == {"premature_stop_after_first_hop": 10}
    assert f["M2"]["unanswerable"]["labels"] == {"invented_unsupported_answer": 10}
    assert f["M3"]["unanswerable"]["labels"] == {"correct_abstention": 10}


def test_case_studies_and_figures(study, tmp_path):
    out = tmp_path / "cases.md"
    r = run("src.case_studies", "--split", "hard_dev", "--before", str(study / "eval_hard_dev_M2.json"),
            "--after", str(study / "eval_hard_dev_M3_s42.json"), "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "Case 3a" in text and "Case 4" in text and "Search calls" in text
    assert "Case 5" not in text.split("## Cases not available")[0]  # perfect model: no reverse failure to show
    figs = tmp_path / "figs"
    r = run("src.make_figures", "--out", str(figs), "--original", str(tmp_path / "none"), "--study", str(study), "--splits", "hard_dev", "--seeds", "42")
    assert r.returncode == 0, r.stderr
    assert (figs / "fig2_m2_grpo_reward_loss.png").exists() and (figs / "fig5_accuracy_hard_dev.png").exists()
    assert not (figs / "fig1_original_grpo_reward_loss.png").exists()  # no input -> skipped, not invented
    assert "not a supervised cross-entropy" in (figs / "figure_captions.md").read_text()


def test_verify_claims_flags_mismatch(study, tmp_path):
    r = run("src.verify_claims", "--study", str(study), "--out", str(tmp_path / "c.md"))
    assert r.returncode == 0, r.stderr
    txt = (tmp_path / "c.md").read_text()
    assert "NOT CHECKABLE" in txt and "CONTRADICTED" in txt  # fixtures do not match the reported 14/30 and are missing M4


def test_freeze_end_to_end_and_immutable(tmp_path):
    S = tmp_path / "hard"; SFT = tmp_path / "sft"; S.mkdir(); (SFT / "B" / "adapter").mkdir(parents=True)
    (SFT / "B" / "adapter" / "adapter_config.json").write_text("{}")
    (SFT / "B" / "config_snapshot.yaml").write_text("seed: 1\n")
    # M2 weak everywhere on hard_dev; all variants identical and strong -> simplest (M3) must win
    for split in ("hard_dev", "dev", "twohop_dev"):
        fixture_eval(S / f"eval_{split}_M2.json", lazy if split == "hard_dev" else perfect)
    for v in ("M3", "M4", "M4g"):
        for sd in (42, 43):
            d = S / f"{v}_s{sd}"; (d / "adapter").mkdir(parents=True)
            (d / "adapter" / "adapter_config.json").write_text("{}"); (d / "adapter" / "adapter_model.safetensors").write_bytes(b"w" + bytes([sd]))
            (d / "config_snapshot.yaml").write_text("seed: %d\n" % sd)
            (d / "train_metrics.csv").write_text("step,loss,reward,grad_norm,frac_reward_zero_std\n1,0.0,0.5,0.2,0.5\n2,0.1,0.6,0.0,1.0\n")
            for split in ("hard_dev", "dev", "twohop_dev"):
                fixture_eval(S / f"eval_{split}_{v}_s{sd}.json", perfect)
    r = run("src.freeze_hard", "--study", str(S), "--sft", str(SFT), "--dry-run")
    assert r.returncode == 0, r.stderr[-800:]
    assert not (S / "FROZEN_HARD.json").exists()
    r = run("src.freeze_hard", "--study", str(S), "--sft", str(SFT))
    assert r.returncode == 0, r.stderr[-800:]
    m = json.loads((S / "FROZEN_HARD.json").read_text())
    assert m["headline_model"] == "M3" and m["training_seed"] == 42 and m["checkpoint_path"].endswith("M3_s42/adapter")
    assert m["adapter_sha256"] and m["config_sha256"] and m["source_sha256_src"] and "hard_final" in m["sealed_dataset_hashes"]["hard"]
    assert m["hard_final_viewed_before_freeze"] is False and m["evaluation"]["answer_format"] == "typed_abstain"
    assert (S / "FROZEN_HARD.json").stat().st_mode & 0o222 == 0  # read-only
    again = run("src.freeze_hard", "--study", str(S), "--sft", str(SFT))
    assert again.returncode != 0 and "immutable" in (again.stderr + again.stdout)


def test_report_headline_and_conclusions(study, tmp_path):
    from src.build_report import conclusions, headline
    sub = tmp_path / "sub"; (sub / "tables").mkdir(parents=True)
    assert "INCOMPLETE" in headline(sub) and "Not yet established" in conclusions(sub)
    run("src.rl_improvement", "--study", str(study), "--split", "hard_dev", "--after", "M3_s42", "--out", str(sub / "tables" / "rl_hard_final.json"))
    r = run("src.two_comparisons", "--study", str(study), "--split", "hard_dev", "--headline", "M3_s42", "--out", str(sub / "tables" / "two_comparisons.json"))
    assert r.returncode == 0, r.stderr
    (sub / "FROZEN_HARD.json").write_text(json.dumps({"headline_model": "M3", "training_seed": 42, "headline_label": "M3 seed 42", "frozen_at_utc": "t"}))
    h, c = headline(sub), conclusions(sub)
    assert "60/60" in h and "Comparison A" in h and "Comparison B" in h and "typed" in h and "not** be attributed to RL alone" in h
    assert "substantial part of the search-behaviour gain is not explained by the reminder" in c   # chain 0% -> 100% under both baselines
    assert "no SFT-on-`hard_train` control" in c and "Comparison B" in c
    (sub / "FROZEN_HARD.json").write_text(json.dumps({"headline_model": "M2", "training_seed": None, "headline_label": "M2", "frozen_at_utc": "t"}))
    assert "negative result" in headline(sub)
