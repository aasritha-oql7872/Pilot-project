from src.mdp_baseline import solve

P = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]  # deterministic chain


def test_searches_through_partial_when_answers_pay():
    V, pol = solve(P, [0.0, 0.0, 1.0], 0.05, 2)
    assert pol[1][1] == "search" and pol[2][0] == "search" and pol[2][2] == "answer"
    assert abs(V[1][1] - 0.95) < 1e-9 and abs(V[2][0] - 0.90) < 1e-9


def test_answers_immediately_when_search_is_not_worth_it():
    _, pol = solve(P, [0.0, 0.0, 0.04], 0.05, 2)  # gain 0.04 < cost 0.05
    assert pol[1][1] == "answer"


def test_budget_exhausted_forces_answer():
    _, pol = solve(P, [0.0, 0.0, 1.0], 0.05, 2)
    assert pol[0] == ["answer"] * 3
