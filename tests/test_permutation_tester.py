"""Unit and statistical tests for PermutationTester."""

from pathlib import Path

import arff
import numpy as np
import pytest
import yaml

from src.tools.aslib_scenario import (
    aslib_permutation_test,
)
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

    res = tester.run_test(
        v1, v2, permutations=1000, name1="algo_fast", name2="algo_slow"
    )
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


def test_batched_permutation_matches_seeded_reference():
    a = np.arange(7, dtype=float)
    b = a + np.array([1, -2, 3, -4, 5, -6, 7])
    permutations = 1501
    observed = -abs(a.mean() - b.mean())
    diffs = a - b if a.mean() < b.mean() else b - a
    signs = np.random.default_rng(42).choice(
        [-1.0, 1.0], size=(permutations - 1, len(a))
    )
    expected = (
        1 + np.count_nonzero(np.mean(signs * diffs, axis=1) <= observed)
    ) / permutations
    assert PermutationTester(42).run_test(a, b, permutations=permutations)[
        "p_value"
    ] == round(expected, 5)


def test_permutation_uses_bounded_batches(monkeypatch):
    class RecordingRng:
        def __init__(self):
            self.rng = np.random.default_rng(42)
            self.sizes = []

        def choice(self, values, size):
            self.sizes.append(size)
            return self.rng.choice(values, size=size)

    tester = PermutationTester(42)
    recorder = RecordingRng()
    monkeypatch.setattr(tester, "rng", recorder)
    tester.run_test(np.ones(2000), np.ones(2000) * 2, permutations=1201)
    assert len(recorder.sizes) > 1
    assert max(rows * columns for rows, columns in recorder.sizes) <= 1_000_000


def test_scenario_permutation_reports_provenance(mini_scenario_dir):
    from src.tools.permutation_tester import permutation_test_scenario

    result = permutation_test_scenario(
        mini_scenario_dir, "algo_a", "algo_b", permutations=20
    )
    assert "algorithm_runs.arff" in result["provenance"]["sha256"]


@pytest.mark.parametrize("factor, mean", [(1, 5.0), (0.5, 4.0), (20, 43.0)])
def test_scenario_permutation_honors_par_factor(mini_scenario_dir, factor, mean):
    result = aslib_permutation_test(
        scenario_dir=str(mini_scenario_dir),
        algo1="algo_a",
        algo2="algo_b",
        par_factor=factor,
        permutations=100,
        seed=7,
    )
    # algo_a: [1,8,4,timeout,2], cutoff=10.
    assert result["candidate_1"]["mean"] == mean
    assert result["candidate_2"]["mean"] == 4.0


def edit_arff(path, mutate):
    with path.open() as stream:
        data = arff.load(stream)
    mutate(data)
    with path.open("w") as stream:
        arff.dump(data, stream)


@pytest.mark.parametrize("status", ["crash", "timeout"])
@pytest.mark.parametrize("factor, mean", [(0.5, 4.8), (20, 82.8)])
def test_scenario_permutation_penalizes_failed_run_below_cutoff(
    mini_scenario_dir, status, factor, mean
):
    def change(data):
        names = [name for name, _ in data["attributes"]]
        for row in data["data"]:
            if row[0] == "inst_1" and row[names.index("algorithm")] == "algo_a":
                row[names.index("runtime")] = 1.0
                row[names.index("runstatus")] = status

    edit_arff(mini_scenario_dir / "algorithm_runs.arff", change)
    result = aslib_permutation_test(
        scenario_dir=str(mini_scenario_dir),
        algo1="algo_a",
        algo2="algo_b",
        par_factor=factor,
        permutations=100,
        seed=7,
    )
    # algo_a: [failed,8,4,timeout,2]; both failures cost cutoff * factor.
    assert result["candidate_1"]["mean"] == pytest.approx(mean)
    assert result["candidate_2"]["mean"] == 4.0


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [[1, 2]]])
def test_invalid_permutation_vectors_rejected(values):
    with pytest.raises(ValueError):
        PermutationTester().run_test(values, values)


def quality_scenario(path):
    """Convert an independent copy of measured runs into a quality benchmark."""
    desc_path = path / "description.txt"
    desc = yaml.safe_load(desc_path.read_text())
    desc.update(
        performance_measures=["accuracy"],
        performance_type=["solution_quality"],
        maximize=[True],
        algorithm_cutoff_time=0,
    )
    desc_path.write_text(yaml.safe_dump(desc))
    runs = path / "algorithm_runs.arff"
    with runs.open() as f:
        data = arff.load(f)
    data["attributes"][3] = ("accuracy", "NUMERIC")
    scores = {"algo_a": 0.9, "algo_b": 0.7, "algo_c": 0.5}
    for row in data["data"]:
        row[3], row[4] = scores[row[2]], "ok"
    with runs.open("w") as f:
        arff.dump(data, f)


def test_quality_scenario_permutation_does_not_apply_runtime_cutoff(mini_scenario_dir):
    quality_scenario(mini_scenario_dir)
    result = permutation_test_scenario(
        mini_scenario_dir, "algo_a", "algo_b", permutations=1000
    )
    assert result["candidate_1"]["mean"] == pytest.approx(0.9)
    assert result["candidate_2"]["mean"] == pytest.approx(0.7)


def test_exact_maximization_tail_includes_observed_assignment():
    # With two equal positive differences, a one-sided exact tail is 1/4.
    result = PermutationTester(seed=12).run_test(
        [2, 2], [1, 1], maximize=True, permutations=50000
    )
    assert result["p_value"] == pytest.approx(0.25, abs=0.01)
