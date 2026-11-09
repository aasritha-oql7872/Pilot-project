"""Final-fix tools on SYNTHETIC fixtures: paired comparisons A/B, log recovery, report wording, packaging hygiene."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.build_report import attribution, headline
from src.two_comparisons import analyse, check_expected, contrast, render

ROOT = Path(__file__).resolve().parents[1]


def write_eval(path, correct_ids, n=120, split="hard_final", answer_format="typed_abstain", cats=6):
    cn = ["standard", "distractor_name", "distractor_person", "stale_affiliation", "reverse_two_hop", "unanswerable"]
    eps = [{"task_id": f"t{i:03d}", "category": cn[i % cats], "kind": "twohop", "correct": i in correct_ids} for i in range(n)]
    Path(path).write_text(json.dumps({"checkpoint": "x/" + Path(path).stem, "split": split, "decoding": "greedy", "prompt_variant": "default",
                                      "answer_format": answer_format, "episodes": eps}))


@pytest.fixture
def notebook_like(tmp_path):
    """Contingency of the executed notebook's M3 seed 42 vs M2: 33 both correct, 49 model-only, 1 reference-only, 37 neither."""
    S = tmp_path / "S"; S.mkdir()
    m2 = set(range(0, 34)); m3 = set(range(1, 83)); m2t = set(range(1, 76)); m3b = set(range(2, 83))
    write_eval(S / "eval_hard_final_M2.json", m2)
    write_eval(S / "eval_hard_final_M2_typed.json", m2t, answer_format="typed")
    write_eval(S / "eval_hard_final_M3_s42.json", m3)
    write_eval(S / "eval_hard_final_M3_s43.json", m3b)
    return S


def test_contingency_matches_the_executed_notebook(notebook_like):
    res = analyse(notebook_like, "hard_final", "M3_s42", ["M3_s43"])
    A = res["comparisons"]["A: M3_s42 vs M2 (both typed_abstain)"]
    assert (A["both_correct"], A["both_wrong"], A["model_only_correct"], A["ref_only_correct"]) == (33, 37, 49, 1)
    assert A["diff_pts"] == pytest.approx(40.0) and A["mcnemar_exact_p"] == pytest.approx(9.06e-14, rel=0.01)
    assert abs(A["ci95_pts"][0] - 30.8) < 1.5 and abs(A["ci95_pts"][1] - 49.2) < 1.5 and A["significant_at_0.05"]
    assert A["settings_that_differ"] == {}
    B = res["comparisons"]["B: M3_s42 (typed_abstain) vs M2 (typed)"]
    assert B["diff_pts"] == pytest.approx(100 * 7 / 120) and set(B["settings_that_differ"]) == {"answer_format"}
    C = res["comparisons"]["C: M2 typed vs M2 typed_abstain (same weights)"]
    assert C["model_correct"] == 75 and C["ref_correct"] == 34 and res["task_id_sets_equal"]
    assert res["category_counts"]["M3_s42"]["unanswerable"][1] == 20


def test_expectations_verified_and_mismatch_detected(notebook_like):
    res = analyse(notebook_like, "hard_final", "M3_s42", [])
    ok = check_expected(res, ["M2=34/120", "M2_typed=75/120", "M3_s42=82/120"])
    assert all(r[3] for r in ok)
    bad = check_expected(res, ["M2=35/120", "M3_s42:unanswerable=4/20", "nonsense"])
    assert not any(r[3] for r in bad)
    assert "MISMATCH" in render(res, bad) and "VERIFIED" in render(res, ok)
    r = subprocess.run([sys.executable, "-m", "src.two_comparisons", "--study", str(notebook_like), "--split", "hard_final", "--headline", "M3_s42", "--expect", "M2=99/120"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0


def test_no_manufactured_significance():
    ref = {f"t{i}": {"correct": i < 10, "category": "c"} for i in range(40)}
    mod = {f"t{i}": {"correct": (i < 10 and i != 0) or i in (10, 11), "category": "c"} for i in range(40)}   # 2 gained, 1 lost
    c = contrast(ref, mod)
    assert (c["model_only_correct"], c["ref_only_correct"]) == (2, 1) and c["mcnemar_exact_p"] == 1.0 and not c["significant_at_0.05"]


def test_report_states_both_comparisons_and_conditions_on_significance(tmp_path):
    S = tmp_path / "S"; S.mkdir()
    write_eval(S / "eval_hard_final_M2.json", set(range(0, 34)))
    write_eval(S / "eval_hard_final_M2_typed.json", set(range(0, 75)), answer_format="typed")
    write_eval(S / "eval_hard_final_M3_s42.json", set(range(0, 78)))     # +3 over M2 typed: 3 vs 0 discordant -> p=0.25, not significant
    sub = tmp_path / "sub"; (sub / "tables").mkdir(parents=True)
    (sub / "tables" / "two_comparisons.json").write_text(json.dumps(analyse(S, "hard_final", "M3_s42", [])))
    (sub / "FROZEN_HARD.json").write_text(json.dumps({"headline_model": "M3", "training_seed": 42, "headline_label": "M3 seed 42", "frozen_at_utc": "t"}))
    h, a = headline(sub), attribution(sub)
    assert "Comparison A" in h and "Comparison B" in h and "+2.5 points" in h
    assert "not statistically distinguishable from zero" in a and "do not claim that GRPO alone explains" in a
    assert "prompt-sensitive" in a and "not additive causal shares" in a


def test_recover_train_logs_never_invents(tmp_path):
    from src.recover_train_logs import describe, parse_segments, recover
    drive = tmp_path / "drive"; (drive / "outputs" / "main_v2").mkdir(parents=True)
    run = tmp_path / "run"
    # nothing anywhere -> unavailable, no file written
    res = recover(run, [drive]); assert res["status"] == "unavailable" and not (run / "train_metrics.csv").exists()
    assert "UNAVAILABLE" in describe(res) and "Nothing was reconstructed" in describe(res)
    # csv on Drive -> copied byte-for-byte, original untouched
    (drive / "outputs" / "main_v2" / "train_metrics.csv").write_text("step,loss,reward\n1,0.0,0.1\n2,0.1,0.2\n")
    res = recover(run, [drive]); assert res["status"] == "copied" and (run / "train_metrics.csv").read_text().splitlines()[2] == "2,0.1,0.2"
    # an existing file is never overwritten
    (run / "train_metrics.csv").write_text("step,loss,reward\n9,9,9\n")
    assert recover(run, [drive])["status"] == "present" and "9,9,9" in (run / "train_metrics.csv").read_text()
    # console log fallback
    log = ("{'loss': '0', 'reward': '0.5', 'grad_norm': '0'}\n{'loss': '0.01', 'reward': '0.7', 'grad_norm': '0.2'}\n{'train_runtime': '9', 'train_loss': '0.005'}\n"
           '{\n "run_id": "main_v2"\n}\n{\'loss\': \'0\', \'reward\': \'0.1\'}\n{\'train_runtime\': \'9\', \'train_loss\': \'0\'}\n{ "run_id": "seed43" }\n')
    segs = parse_segments(log)
    assert [(r, len(rows)) for r, rows in segs] == [("main_v2", 2), ("seed43", 1)] and segs[0][1][1]["reward"] == 0.7
    d2 = tmp_path / "drive2"; d2.mkdir(); (d2 / "run_all.log").write_text(log)
    run2 = tmp_path / "run2"
    res = recover(run2, [d2]); assert res["status"] == "reconstructed" and res["rows"] == 2 and "log-line order" in res["caveat"]
    assert (run2 / "train_metrics.source.txt").exists()


def test_checker_counts_blocks_not_prose_and_template_has_no_marker():
    from src.submission_check import NA_BLOCK, broken_links
    assert len(NA_BLOCK.findall("see below: *NOT AVAILABLE* means x")) == 0
    assert len(NA_BLOCK.findall("> **NOT AVAILABLE:** `x` has not been produced")) == 1
    assert "NOT AVAILABLE" not in (ROOT / "reports" / "FINAL_PILOT_REPORT.template.md").read_text()
    assert broken_links(ROOT, "![a](reports/FINAL_PILOT_REPORT.template.md) [b](nope/missing.png) [c](https://x.y)") == ["nope/missing.png"]


def test_template_has_required_sections():
    t = (ROOT / "reports" / "FINAL_PILOT_REPORT.template.md").read_text()
    assert "How Much Improvement Is Attributable to RL Versus Prompt Design?" in t and "{{attribution}}" in t and "{{failure_discussion}}" in t
    for n in range(1, 19):
        assert f"## {n}. " in t


def test_figures_before_after_and_unavailable_fig1(tmp_path):
    from src.package_submission import figures_block, status_block
    S = tmp_path / "S"; S.mkdir()
    write_eval(S / "eval_hard_final_M2.json", set(range(0, 34)))
    write_eval(S / "eval_hard_final_M2_typed.json", set(range(0, 75)), answer_format="typed")
    write_eval(S / "eval_hard_final_M3_s42.json", set(range(0, 82)))
    figs = tmp_path / "figs"
    r = subprocess.run([sys.executable, "-m", "src.make_figures", "--out", str(figs), "--original", str(tmp_path / "none"), "--study", str(S), "--splits", "hard_final", "--seeds", "42", "--headline", "M3_s42"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (figs / "fig8_before_after_hard_final.png").exists() and (figs / "fig9_category_before_after_hard_final.png").exists()
    assert not (figs / "fig1_original_grpo_reward_loss.png").exists()
    cap = (figs / "figure_captions.md").read_text()
    assert "exactly 0 on any step where every group in the batch has tied rewards" in cap and "UNAVAILABLE" in cap
    sub = tmp_path / "sub"; (sub / "tables").mkdir(parents=True); (sub / "figures").mkdir()
    shutil.copy(figs / "fig8_before_after_hard_final.png", sub / "figures"); shutil.copy(figs / "fig5_accuracy_hard_final.png", sub / "figures")
    (sub / "tables" / "original_grpo_log_recovery.md").write_text("**Original GRPO (no SFT) reward/loss figure: UNAVAILABLE.** because x")
    blk = figures_block(sub / "figures", sub)
    assert "Figure 1 (original GRPO reward and loss): unavailable" in blk and "fig8" not in blk   # fig8/9 are shown in 10.7, not here
    # partial-SFT / RL-only status: a saved summary is reported, an absent directory is "not run"
    d = tmp_path / "study_early"; d.mkdir(); (d / "summary_early_dev.md").write_text("| E1 | 0/30 |")
    assert "| E1 | 0/30 |" in status_block([d], "partial-SFT", "run_early_study.sh", "summary_early*.md")
    assert "Not run" in status_block([tmp_path / "missing"], "partial-SFT", "run_early_study.sh", "summary_early*.md")


def test_package_hygiene_and_frozen_adapter_hash(tmp_path, monkeypatch):
    from src.package_submission import package_code, sha, write_manifest
    monkeypatch.chdir(ROOT)
    ck = tmp_path / "ck"; ck.mkdir(); (ck / "adapter_model.safetensors").write_bytes(b"frozen-adapter"); (ck / "adapter_config.json").write_text("{}")
    nb = tmp_path / "executed.ipynb"; nb.write_text("{}")
    sub = tmp_path / "sub"; sub.mkdir()
    got, ad = package_code(sub, [nb], {"checkpoint_path": str(ck)})
    assert got == ["executed.ipynb"] and sha(ad / "adapter_model.safetensors") == sha(ck / "adapter_model.safetensors")
    assert (sub / "code" / "src" / "train_grpo.py").exists() and (sub / "code" / "configs" / "train_hard_grpo.yaml").exists() and (sub / "code" / "requirements.lock").exists()
    assert (sub / "code" / "data" / "manifest_hard.json").exists()
    assert [p for p in sub.rglob("*.safetensors")] == [ad / "adapter_model.safetensors"]       # only the frozen headline adapter
    assert not list(sub.rglob("__pycache__")) and not list(sub.rglob("*.pyc")) and not list(sub.rglob(".env*"))
    n = write_manifest(sub)
    lines = (sub / "MANIFEST_SHA256.txt").read_text().splitlines()
    assert len(lines) == n and not any("MANIFEST_SHA256" in l for l in lines)
    # existing notebook copy is never overwritten
    nb.write_text("changed"); package_code(sub, [nb], {"checkpoint_path": str(ck)})
    assert (sub / "notebooks" / "executed.ipynb").read_text() == "{}"


def test_verify_freeze_detects_tampering(tmp_path):
    from data.generate import DATA_DIR
    from src.verify_freeze import sha256, verify
    S = tmp_path / "S"; S.mkdir()
    ck = S / "M3_s42" / "adapter"; ck.mkdir(parents=True); (ck / "adapter_model.safetensors").write_bytes(b"frozen")
    hard = json.loads((DATA_DIR / "manifest_hard.json").read_text())["splits"]["hard_final"]
    man = {"headline_model": "M3", "headline_label": "M3 seed 42", "training_seed": 42, "frozen_at_utc": "2026-11-08T02:31:47+00:00", "checkpoint_path": str(ck),
           "adapter_sha256": sha256(ck / "adapter_model.safetensors"), "hard_final_viewed_before_freeze": False, "sealed_view_statement": "not viewed",
           "git_commit": None, "sealed_dataset_hashes": {"hard": {"hard_final": hard}}}
    (S / "FROZEN_HARD.json").write_text(json.dumps(man))
    (S / "HARD_FINAL_FIRST_EVALUATION").write_text("2026-11-08T03:00:00Z\n")
    write_eval(S / "eval_hard_final_M2.json", {1}, n=10); write_eval(S / "eval_hard_final_M2_typed.json", {1, 2}, n=10, answer_format="typed")
    write_eval(S / "eval_hard_final_M3_s42.json", {1, 2, 3}, n=10)
    j = json.loads((S / "eval_hard_final_M3_s42.json").read_text()); j["checkpoint"] = str(ck); (S / "eval_hard_final_M3_s42.json").write_text(json.dumps(j))
    st = {r[0]: r[1] for r in verify(S)}
    assert all(v in ("PASS", "INFO") for v in st.values()), st
    (ck / "adapter_model.safetensors").write_bytes(b"tampered")
    assert {r[0]: r[1] for r in verify(S)}["headline adapter sha256 equals the freeze manifest"] == "FAIL"
    (ck / "adapter_model.safetensors").write_bytes(b"frozen")
    (S / "HARD_FINAL_FIRST_EVALUATION").write_text("2026-11-08T01:00:00Z\n")          # sealed evaluation BEFORE the freeze
    assert {r[0]: r[1] for r in verify(S)}["freeze happened before the first hard_final evaluation"] == "FAIL"
