"""Final assembly and deck number checks on SYNTHETIC packages (never real results)."""
import json
import zipfile
from pathlib import Path

import pytest

from src.assemble_final import assemble, hygiene, verify_manifest, write_manifest


def make_pkg_zip(tmp_path, extra=None):
    root = tmp_path / "colab_pkg"
    files = {"FINAL_PILOT_REPORT.md": "# Report\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", "README_SUBMISSION.md": "r", "FROZEN_HARD.json": "{}", "case_studies.md": "c",
             "tables/two_comparisons.md": "t", "code/src/train_grpo.py": "x = 1\n", "STATUS_CHECKLIST.md": "| full test suite | FAIL | 5 failed, 107 passed |\n| executed notebook(s) | FAIL |  |\n",
             "MANIFEST_SHA256.txt": "old\n"}
    files.update(extra or {})
    z = tmp_path / "colab.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for k, v in files.items():
            zf.writestr(k, v)
        zf.writestr("checkpoint/frozen_headline_adapter/adapter_model.safetensors", b"frozen")
    return z


def test_assembly_adds_files_keeps_originals_and_never_overwrites(tmp_path):
    z = make_pkg_zip(tmp_path)
    nb = tmp_path / "executed.ipynb"; nb.write_text('{"cells": []}')
    deck = tmp_path / "deck.pptx"; deck.write_bytes(b"pptx"); script = tmp_path / "s.md"; script.write_text("script")
    r = assemble(z, deck, script, [nb], "v9_test", tmp_path / "out", build_pdf=False)
    out = Path(r["zip"])
    assert out.exists() and (tmp_path / "out" / (out.name + ".sha256")).read_text().split()[0] == r["zip_sha256"]
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {"presentation/SearchLite_RL_Final_Presentation.pptx", "presentation/SearchLite_RL_Speaking_Script.md", "notebooks/executed.ipynb",
                "MANIFEST_SHA256.txt", "MANIFEST_SHA256.colab.txt", "ASSEMBLY_NOTES.md", "STATUS_CHECKLIST.md"} <= names
        assert zf.read("STATUS_CHECKLIST.md").decode().startswith("| full test suite | FAIL | 5 failed")          # Colab checklist untouched
        assert "5 failed, 107 passed" in zf.read("ASSEMBLY_NOTES.md").decode()                                  # inherited FAIL is reported, not hidden
        assert zf.read("MANIFEST_SHA256.colab.txt") == b"old\n"
    with pytest.raises(SystemExit):                                                                           # same version again: refuse
        assemble(z, deck, script, [nb], "v9_test", tmp_path / "out", build_pdf=False)


def test_secret_notebook_is_not_packaged_and_bad_package_is_refused(tmp_path):
    z = make_pkg_zip(tmp_path)
    nb = tmp_path / "leaky.ipynb"; nb.write_text('{"x": "hf_' + "a" * 34 + '"}')
    r = assemble(z, None, None, [nb], "v9_secret", tmp_path / "out", build_pdf=False)
    assert any("leaky.ipynb" in s and "NOT packaged" in s for s in r["skipped"])
    with zipfile.ZipFile(r["zip"]) as zf:
        assert "notebooks/leaky.ipynb" not in zf.namelist()
    bad = make_pkg_zip(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, {".env": "A=1", "code/__pycache__/m.pyc": "x"})
    with pytest.raises(SystemExit, match="hygiene"):
        assemble(bad, None, None, [], "v9_bad", tmp_path / "out2", build_pdf=False)
    with pytest.raises(SystemExit, match="missing"):
        broken = tmp_path / "broken.zip"
        with zipfile.ZipFile(broken, "w") as zf:
            zf.writestr("FINAL_PILOT_REPORT.md", "x")
        assemble(broken, None, None, [], "v9_missing", tmp_path / "out3", build_pdf=False)


def test_manifest_detects_modification(tmp_path):
    root = tmp_path / "p"; (root / "a").mkdir(parents=True)
    (root / "a" / "f.txt").write_text("one"); (root / "checkpoint" / "frozen_headline_adapter").mkdir(parents=True)
    (root / "checkpoint" / "frozen_headline_adapter" / "adapter_model.safetensors").write_bytes(b"w")
    assert write_manifest(root) == 2 and verify_manifest(root) == []
    (root / "a" / "f.txt").write_text("two")
    assert verify_manifest(root) == ["a/f.txt"]
    (root / "other.safetensors").write_bytes(b"extra")
    assert any("unexpected checkpoints" in b for b in hygiene(root))


def test_deck_numbers_are_verified_against_the_comparison_json(tmp_path):
    pytest.importorskip("pptx")
    from src import build_deck
    res = {"comparisons": {k: {"both_correct": v[0], "both_wrong": v[1], "model_only_correct": v[2], "ref_only_correct": v[3], "diff_pts": v[4],
                               "ci95_pts": [v[5], v[6]], "mcnemar_exact_p": v[7]} for k, v in build_deck.CMP.items()},
           "overall_counts": {"M2": [34, 120], "M2_typed": [75, 120], "M3_s42": [82, 120], "M3_s43": [81, 120]},
           "category_counts": {lab: {c: [build_deck.CAT_COUNTS[key][i], 20] for i, c in enumerate(build_deck.CATS)}
                               for lab, key in (("M2", "M2 typed_abstain"), ("M2_typed", "M2 typed"), ("M3_s42", "M3 seed 42"))}}
    pkg = tmp_path / "pkg"; (pkg / "tables").mkdir(parents=True)
    (pkg / "tables" / "two_comparisons.json").write_text(json.dumps(res))
    assert build_deck.verify_numbers(pkg)
    res["comparisons"]["B: M3_s42 (typed_abstain) vs M2 (typed)"]["model_only_correct"] = 9
    (pkg / "tables" / "two_comparisons.json").write_text(json.dumps(res))
    with pytest.raises(AssertionError):
        build_deck.verify_numbers(pkg)
