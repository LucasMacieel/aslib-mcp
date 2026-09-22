"""Unit and statistical tests for PermutationTester."""

from pathlib import Path

import numpy as np
import pytest

from src.tools.permutation_tester import PermutationTester, permutation_test_scenario


def test_noise_filtering_identity_when_zero():
    """With noise=0.0, vectors must remain completely unmodified."""
    tester = PermutationTester(seed=42)
    v1 = np.array([1.0, 5.0, 10.0])
    v2 = np.array([1.2, 4.8, 10.1])

    f1, f2 = tester.filter_noise(v1, v2, noise=0.0)
    np.testing.assert_array_equal(f1, v1)
    np.testing.assert_array_equal(f2, v2)


def test_noise_filtering_equates_within_threshold():
    """Small differences within noise delta must be equated."""
    tester = PermutationTester(seed=42)
    v1 = np.array([5.0])
    v2 = np.array([5.05])

    f1, f2 = tester.filter_noise(v1, v2, noise=0.1)
    assert f1[0] == f2[0]


def test_vector_length_mismatch_raises():
    """Mismatched vector lengths must raise ValueError."""
    tester = PermutationTester(seed=42)
    with pytest.raises(ValueError, match="Vectors must have the same length"):
        tester.run_test([1.0, 2.0], [1.0, 2.0, 3.0])


def test_empty_vectors_raise():
    """Empty performance vectors must raise ValueError."""
    tester = PermutationTester(seed=42)
    with pytest.raises(ValueError, match="Performance vectors cannot be empty"):
        tester.run_test([], [])


def test_identical_vectors_yield_p_one():
    """Identical vectors must produce p-value = 1.0, zero difference, and tie winner."""
    tester = PermutationTester(seed=42)
    v = [2.5, 4.0, 1.2, 9.8, 3.3]

    res = tester.run_test(v, v, permutations=1000)
    assert res["p_value"] == 1.0
    assert res["observed_diff"] == 0.0
    assert res["is_significant"] is False
    assert res["winner"] == "tie"


def test_strictly_separated_vectors_significant():
    """Strongly separated performance vectors must yield statistical significance (p < 0.05)."""
    tester = PermutationTester(seed=42)
    v1 = [1.0] * 30
    v2 = [10.0] * 30

    res = tester.run_test(v1, v2, permutations=1000, name1="algo_fast", name2="algo_slow")
    assert res["p_value"] < 0.01
    assert res["is_significant"] is True
    assert res["winner"] == "algo_fast"
    assert res["observed_diff"] < 0


def test_maximization_mode_favors_higher_scores():
    """In maximization mode, candidate with higher score must win."""
    tester = PermutationTester(seed=42)
    v1 = [100.0] * 20
    v2 = [10.0] * 20

    res = tester.run_test(v1, v2, maximize=True, name1="algo_high", name2="algo_low")
    assert res["p_value"] < 0.01
    assert res["is_significant"] is True
    assert res["winner"] == "algo_high"


def test_permutation_determinism_with_seed():
    """Using identical seeds must produce bit-for-bit identical test results."""
    v1 = [1.2, 3.4, 2.1, 8.9, 4.5, 2.2, 6.7]
    v2 = [2.1, 2.9, 3.0, 7.5, 5.1, 1.9, 7.2]

    res1 = PermutationTester(seed=999).run_test(v1, v2, permutations=2000)
    res2 = PermutationTester(seed=999).run_test(v1, v2, permutations=2000)

    assert res1["p_value"] == res2["p_value"]
    assert res1["observed_diff"] == res2["observed_diff"]


def test_permutation_test_scenario_on_synthetic_scenario(mini_scenario_dir: Path):
    """Running permutation_test_scenario on synthetic scenario produces valid comparison."""
    res = permutation_test_scenario(
        scenario_dir=mini_scenario_dir,
        algo1="algo_a",
        algo2="algo_b",
        permutations=1000,
        seed=1234,
    )

    assert 0.0 <= res["p_value"] <= 1.0
    assert "candidate_1" in res
    assert "candidate_2" in res
    assert res["candidate_1"]["name"] == "algo_a"
    assert res["candidate_2"]["name"] == "algo_b"


def test_permutation_test_scenario_unknown_algorithm_raises(mini_scenario_dir: Path):
    """Comparing an unknown algorithm must raise ValueError."""
    with pytest.raises(ValueError, match="Algorithm 'algo_unknown' not found"):
        permutation_test_scenario(
            scenario_dir=mini_scenario_dir,
            algo1="algo_a",
            algo2="algo_unknown",
        )
