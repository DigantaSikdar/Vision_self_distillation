"""Metric definitions (the single source of truth). Runs anywhere."""
from caad.eval.metrics import avg_pass_at_1, maj_at_k, pass_at_k


def test_pass_at_k_all_correct_is_one():
    samples = [{"correct": [True, True, True]}]
    assert pass_at_k(samples, 1) == 1.0


def test_pass_at_k_none_correct_is_zero():
    samples = [{"correct": [False, False, False, False]}]
    assert pass_at_k(samples, 1) == 0.0


def test_pass_at_k_unbiased_estimator():
    # 2 of 4 correct -> pass@1 = 0.5
    samples = [{"correct": [True, True, False, False]}]
    assert abs(pass_at_k(samples, 1) - 0.5) < 1e-9


def test_avg_pass_at_1():
    samples = [{"correct": [True, False]}, {"correct": [True, True]}]
    assert abs(avg_pass_at_1(samples) - 0.75) < 1e-9


def test_maj_at_k_majority_vote():
    samples = [{"pred": ["a", "a", "b"], "gold": "a"}]
    assert maj_at_k(samples, 3) == 1.0
    samples = [{"pred": ["a", "b", "b"], "gold": "a"}]
    assert maj_at_k(samples, 3) == 0.0


def test_empty_is_zero_not_crash():
    assert pass_at_k([], 1) == 0.0
    assert avg_pass_at_1([]) == 0.0
