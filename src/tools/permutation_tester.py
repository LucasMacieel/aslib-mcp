"""Permutation significance tester for algorithm selection benchmarks.

Modernized, high-performance paired permutation test adapted from
coseal/aslib-spec (scripts/stat_tests/PermutationTester.py) based on
Aghaeepour & Hoos (2013).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class PermutationTester:
    """Performs paired permutation test on algorithm performance vectors."""

    def __init__(self, seed: int | None = 1234):
        self.rng = np.random.default_rng(seed)

    def filter_noise(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        noise: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Noise filtering based on Van Gelder (Careful Ranking of Multiple Solvers)."""
        v1 = vec1.copy()
        v2 = vec2.copy()
        for i in range(len(v1)):
            avg_i = (v1[i] + v2[i]) / 2.0
            delta = math.sqrt(noise / 2.0) * math.sqrt(max(avg_i, 0.0))
            if (v1[i] < v2[i] and v1[i] >= (avg_i - delta)) or (
                v2[i] <= v1[i] and v2[i] >= (avg_i - delta)
            ):
                v1[i] = v2[i]
        return v1, v2

    def run_test(
        self,
        vec1: list[float] | np.ndarray,
        vec2: list[float] | np.ndarray,
        alpha: float = 0.05,
        permutations: int = 10000,
        name1: str = "candidate_1",
        name2: str = "candidate_2",
        cutoff: float | None = None,
        par_factor: float = 10.0,
        noise: float | None = None,
        maximize: bool = False,
    ) -> dict[str, Any]:
        """Runs the paired permutation test."""
        v1 = np.asarray(vec1, dtype=float).copy()
        v2 = np.asarray(vec2, dtype=float).copy()

        if len(v1) != len(v2):
            raise ValueError(
                f"Vectors must have the same length: {len(v1)} vs {len(v2)}"
            )
        if len(v1) == 0:
            raise ValueError("Performance vectors cannot be empty.")

        n = len(v1)

        # Apply PAR cutoff penalty if specified
        if cutoff is not None and par_factor > 0:
            if not maximize:
                v1 = np.where(v1 >= cutoff, cutoff * par_factor, v1)
                v2 = np.where(v2 >= cutoff, cutoff * par_factor, v2)
            else:
                # If maximizing, cutoff is a lower bound or timeout penalty
                v1 = np.where(v1 <= cutoff, cutoff / par_factor, v1)
                v2 = np.where(v2 <= cutoff, cutoff / par_factor, v2)

        mean1 = float(np.mean(v1))
        mean2 = float(np.mean(v2))

        # Check identical vectors
        if np.array_equal(v1, v2):
            return {
                "is_significant": False,
                "p_value": 1.0,
                "winner": "tie",
                "message": "Performance vectors are identical.",
                "candidate_1": {"name": name1, "mean": mean1},
                "candidate_2": {"name": name2, "mean": mean2},
                "observed_diff": 0.0,
                "permutations": permutations,
                "alpha": alpha,
            }

        # Apply Van Gelder noise filter if requested
        if noise is not None and noise > 0:
            v1, v2 = self.filter_noise(v1, v2, noise)
            mean1 = float(np.mean(v1))
            mean2 = float(np.mean(v2))

        # Determine which candidate has better observed performance
        # For minimization (default): smaller is better
        # For maximization: larger is better
        if not maximize:
            # col1 better than col2 if mean1 < mean2
            orig_diff = mean1 - mean2  # negative means name1 is better
            candidate_1_better = orig_diff < 0
        else:
            orig_diff = mean2 - mean1  # negative means name1 is better
            candidate_1_better = orig_diff < 0

        # Align so better candidate is first (target_diff <= 0)
        if not candidate_1_better:
            target_v1, target_v2 = v2, v1
            winner_candidate = name2
            loser_candidate = name1
            target_diff = -abs(mean1 - mean2)
        else:
            target_v1, target_v2 = v1, v2
            winner_candidate = name1
            loser_candidate = name2
            target_diff = -abs(mean1 - mean2)

        # Pairwise differences
        diffs = target_v1 - target_v2

        # Vectorized random permutations:
        # Each pair (x, y) can either stay (x, y) with diff d, or swap (y, x) with diff -d.
        # Random sign mask: +1 (keep) or -1 (swap) with 50/50 probability.
        signs = self.rng.choice([-1.0, 1.0], size=(permutations - 1, n))
        perm_means = np.mean(signs * diffs, axis=1)

        # Include the original observed difference
        all_diffs = np.concatenate(([target_diff], perm_means))

        # One-sided p-value: proportion of permuted mean differences as extreme or more extreme (<= target_diff)
        p_value = float(np.sum(all_diffs <= target_diff) / permutations)

        is_significant = bool(p_value < alpha)
        winner = winner_candidate if is_significant else "not_significant"

        return {
            "is_significant": is_significant,
            "p_value": round(p_value, 5),
            "winner": winner,
            "candidate_1": {"name": name1, "mean": round(mean1, 4)},
            "candidate_2": {"name": name2, "mean": round(mean2, 4)},
            "observed_diff": round(float(mean1 - mean2), 4),
            "message": (
                f"{winner_candidate} is statistically significantly better than {loser_candidate} (p={p_value:.4f}, alpha={alpha})"
                if is_significant
                else f"No statistically significant difference between {name1} and {name2} (p={p_value:.4f}, alpha={alpha})"
            ),
            "permutations": permutations,
            "alpha": alpha,
        }


def permutation_test_scenario(
    scenario_dir: str | Path,
    algo1: str,
    algo2: str,
    permutations: int = 10000,
    alpha: float = 0.05,
    noise: float | None = None,
    seed: int | None = 1234,
) -> dict[str, Any]:
    """Runs a paired permutation test on two algorithms directly from an ASlib scenario."""
    from aslib_scenario.aslib_scenario import ASlibScenario

    scen = ASlibScenario()
    scen.read_scenario(str(Path(scenario_dir).resolve()))

    if scen.performance_data is None:
        raise ValueError(f"Scenario at {scenario_dir} has no performance data loaded.")

    if algo1 not in scen.performance_data.columns:
        raise ValueError(
            f"Algorithm '{algo1}' not found. Available algorithms: {list(scen.performance_data.columns)}"
        )
    if algo2 not in scen.performance_data.columns:
        raise ValueError(
            f"Algorithm '{algo2}' not found. Available algorithms: {list(scen.performance_data.columns)}"
        )

    vec1 = scen.performance_data[algo1].values
    vec2 = scen.performance_data[algo2].values

    cutoff = (
        float(scen.algorithm_cutoff_time)
        if scen.algorithm_cutoff_time is not None
        else None
    )
    maximize = bool(scen.maximize[0]) if scen.maximize else False

    tester = PermutationTester(seed=seed)
    return tester.run_test(
        vec1=vec1,
        vec2=vec2,
        alpha=alpha,
        permutations=permutations,
        name1=algo1,
        name2=algo2,
        cutoff=cutoff,
        par_factor=10.0,
        noise=noise,
        maximize=maximize,
    )
