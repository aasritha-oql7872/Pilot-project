import json

import torch
from safetensors.torch import save_file

from src.audit_lineage import diff, info, rollout_overlap


def make(tmp, name, delta=0.0, steps=3):
    d = tmp / name / "adapter"; d.mkdir(parents=True)
    save_file({"a.lora_A": torch.ones(2, 2) + delta, "a.lora_B": torch.zeros(2, 2) + delta}, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": 2, "base_model_name_or_path": "m", "lora_alpha": 4, "target_modules": ["q_proj"]}))
    (tmp / name / "trajectories.jsonl").write_text("\n".join(json.dumps(
        {"step": i, "task_id": "t", "messages": [{"role": "assistant", "content": "x"}]}) for i in range(steps)))
    return tmp / name


def test_identical_and_different_adapters(tmp_path):
    a, b, c = make(tmp_path, "a"), make(tmp_path, "b"), make(tmp_path, "c", 1e-3)
    assert info(a)["adapter_sha256"] == info(b)["adapter_sha256"] != info(c)["adapter_sha256"]
    ta = {k: v for k, v in __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(a / "adapter/adapter_model.safetensors")).items()}
    tc = {k: v for k, v in __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(c / "adapter/adapter_model.safetensors")).items()}
    assert diff(ta, ta)["max_abs_diff"] == 0 and abs(diff(ta, tc)["max_abs_diff"] - 1e-3) < 1e-6
    assert rollout_overlap(a, b)["identical_fraction_in_order"] == 1.0
