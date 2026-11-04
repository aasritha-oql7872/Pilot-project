from src.summarize_study import wilson


def test_wilson_not_degenerate_at_ceiling():
    lo, hi = wilson(30, 30)
    assert 0.88 < lo < 0.89 and hi == 1.0
    lo, hi = wilson(100, 100)
    assert 0.96 < lo < 0.97


def test_wilson_contains_point_estimate():
    lo, hi = wilson(57, 60)
    assert lo < 57 / 60 < hi and wilson(0, 30)[0] == 0.0
