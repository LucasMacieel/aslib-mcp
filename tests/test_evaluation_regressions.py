import arff
import pandas as pd
import pytest
import yaml

from src.tools.permutation_tester import PermutationTester, permutation_test_scenario
from src.tools.selector_evaluator import SelectorEvaluator


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


def predictions(tmp_path, values, instances):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(values, index=instances).to_csv(path)
    return path


def test_max_quality_rankings_use_original_scale(mini_scenario_dir, tmp_path):
    quality_scenario(mini_scenario_dir)
    ev = SelectorEvaluator(mini_scenario_dir)
    path = predictions(
        tmp_path, {"best": ["algo_a"] * 5, "worst": ["algo_c"] * 5}, ev.instances
    )
    res = ev.evaluate(path, include_feature_costs=False)
    stats = {s["name"]: s for s in res["rankings"]}
    assert res["top_selector"] == "best"
    assert stats["VBS"]["par_score"] == 0.9
    assert stats["SBS (algo_a)"]["par_score"] == 0.9
    assert stats["best"]["solved_count"] == 5
    assert stats["best"]["rank"] < stats["worst"]["rank"]


def test_numeric_predictions_require_explicit_measured_mode(
    mini_scenario_dir, tmp_path
):
    ev = SelectorEvaluator(mini_scenario_dir)
    path = predictions(tmp_path, {"optimistic": [0.01] * 5}, ev.instances)
    with pytest.raises(ValueError, match="algorithm"):
        ev.evaluate(path)
    result = ev.evaluate(
        path, input_mode="measured_performance", include_feature_costs=False
    )
    assert result["input_mode"] == "measured_performance"
    assert result["rankings"][0]["par_score"] == 0.01


def test_quality_missing_predictions_are_rejected(mini_scenario_dir, tmp_path):
    quality_scenario(mini_scenario_dir)
    ev = SelectorEvaluator(mini_scenario_dir)
    path = predictions(tmp_path, {"partial": ["algo_a"] * 4}, ev.instances[:4])
    with pytest.raises(ValueError, match="quality"):
        ev.evaluate(path)


def test_only_default_feature_steps_are_charged(mini_scenario_dir, tmp_path):
    ev = SelectorEvaluator(mini_scenario_dir)
    ev.scenario.feature_group_dict = {
        "cheap": {"provides": [], "requires": []},
        "expensive": {"provides": [], "requires": ["cheap"]},
    }
    ev.scenario.feature_steps_default = ["cheap"]
    ev.scenario.feature_cost_data = pd.DataFrame(
        {"cheap": [0.2] * 5, "expensive": [100] * 5}, index=ev.instances
    )
    ev.scenario.feature_runstatus_data = pd.DataFrame(
        {"cheap": ["ok"] * 5, "expensive": ["ok"] * 5}, index=ev.instances
    )
    path = predictions(tmp_path, {"selector": ["algo_b"] * 5}, ev.instances)
    stats = {
        s["name"]: s
        for s in ev.evaluate(path, run_significance_tests=False)["rankings"]
    }
    assert stats["selector"]["par_score"] == 4.2


def test_presolved_instance_uses_only_feature_time(mini_scenario_dir, tmp_path):
    ev = SelectorEvaluator(mini_scenario_dir)
    assert ev.scenario.feature_runstatus_data is not None
    ev.scenario.feature_runstatus_data.loc[:, "ALL"] = "presolved"
    path = predictions(tmp_path, {"selector": ["algo_c"] * 5}, ev.instances)
    stats = {
        s["name"]: s
        for s in ev.evaluate(path, run_significance_tests=False)["rankings"]
    }
    assert stats["selector"]["par_score"] == 0.2
    assert stats["selector"]["solved_count"] == 5


def test_custom_par_factor_applies_to_vbs_and_sbs(mini_scenario_dir, tmp_path):
    ev = SelectorEvaluator(mini_scenario_dir, par_factor=0.5)
    path = predictions(tmp_path, {"selector": ["algo_c"] * 5}, ev.instances)
    stats = {
        s["name"]: s
        for s in ev.evaluate(
            path, include_feature_costs=False, run_significance_tests=False
        )["rankings"]
    }
    # Recompute all baselines using PAR0.5, including non-OK outcomes.
    assert (
        next(s for name, s in stats.items() if name.startswith("SBS ("))["par_score"]
        == 4.0
    )
    assert stats["selector"]["par_score"] == 4.8  # (5+7+2+5+5)/5
    assert stats["selector"]["timeouts_count"] == 2


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [[1, 2]]])
def test_invalid_permutation_vectors_rejected(values):
    with pytest.raises(ValueError):
        PermutationTester().run_test(values, values)


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
