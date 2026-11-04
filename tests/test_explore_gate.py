import json

from src.explore_gate import decide_early, decide_pilot, decide_temps
from tests.test_traj_stats import row


def probe(path, rate, temp=0.7, answered=1.0):
    path.write_text(json.dumps({"temperature": temp, "rollouts": 240, "second_search_rate": rate, "correct_rate": 0.2,
                                "tasks_with_any_two_search_rollout": int(rate > 0), "answered_rate": answered, "invalid_rate": 0.1}))
    return str(path)


def test_temps_stops_when_no_second_search_ever(tmp_path):
    d = decide_temps([probe(tmp_path / "a.json", 0.0, 0.7), probe(tmp_path / "b.json", 0.0, 1.3)])
    assert d["proceed"] is False and d["max_second_search_rate"] == 0.0


def test_temps_picks_highest_rate_among_nondegenerate(tmp_path):
    d = decide_temps([probe(tmp_path / "a.json", 0.02, 0.7), probe(tmp_path / "b.json", 0.05, 1.0),
                      probe(tmp_path / "c.json", 0.30, 1.5, answered=0.2)])
    assert d["selected_temperature"] == 1.0 and d["proceed"]


def test_early_selects_in_range_and_does_not_force(tmp_path):
    ps = [probe(tmp_path / f"explore_step{n:03d}.json", r) for n, r in [(2, 0.0), (6, 0.45), (30, 0.99)]]
    assert decide_early(ps)["selected"] == 6
    ps = [probe(tmp_path / f"x_step{n:03d}.json", r) for n, r in [(2, 0.0), (30, 0.99)]]
    assert decide_early(ps)["selected"] is None


def test_pilot_requires_a_real_second_search(tmp_path):
    (tmp_path / "trajectories.jsonl").write_text("\n".join(json.dumps(row(1, "t", "twohop", 1, -0.05)) for _ in range(4)))
    assert decide_pilot(tmp_path)["proceed"] is False
    (tmp_path / "trajectories.jsonl").write_text(json.dumps(row(1, "t", "twohop", 2, 0.9)))
    assert decide_pilot(tmp_path)["proceed"] is True
