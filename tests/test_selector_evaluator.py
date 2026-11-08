"""Unit tests for SelectorEvaluator, exact PAR10 scores, and baselines."""

from pathlib import Path
from typing import Any

import arff
import pandas as pd
import pytest
import yaml

from src.tools.selector_evaluator import SelectorEvaluator


@pytest.fixture
def predictions_csv(synthetic_data: dict[str, Any], tmp_path: Path) -> Path:
    """Creates a predictions CSV with an oracle selector and a baseline selector."""
    instances = synthetic_data["instances"]
    pred_df = pd.DataFrame(
        {
            "perfect_selector": ["algo_a", "algo_b", "algo_c", "algo_b", "algo_a"],
            "static_sbs_selector": ["algo_b", "algo_b", "algo_b", "algo_b", "algo_b"],
            "poor_selector": ["algo_c", "algo_a", "algo_b", "algo_c", "algo_c"],
        },
        index=instances,
    )
    pred_path = tmp_path / "predictions.csv"
    pred_df.to_csv(pred_path)
    return pred_path


def test_selector_evaluator_initialization(mini_scenario_dir: Path):
    """SelectorEvaluator must successfully initialize and extract scenario metadata."""
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)
    assert evaluator.cutoff == 10.0
    assert evaluator.objective == "runtime"
    assert evaluator.maximize is False
    assert len(evaluator.instances) == 5
    assert set(evaluator.algorithms) == {"algo_a", "algo_b", "algo_c"}


def test_vbs_and_sbs_exact_baselines(mini_scenario_dir: Path, predictions_csv: Path):
    """Evaluates exact ground truth values for VBS and SBS."""
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)
    res = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=False,
        run_significance_tests=False,
    )

    stats = {s["name"]: s for s in res["rankings"]}
    # VBS mean across [1.0, 3.0, 2.0, 5.0, 2.0] = 13.0 / 5 = 2.6
    assert stats["VBS"]["par_score"] == pytest.approx(2.6, rel=1e-3)
    assert stats["VBS"]["timeouts_count"] == 0

    # SBS is algo_b: [2.0, 3.0, 6.0, 5.0, 4.0] = 20.0 / 5 = 4.0
    sbs_key = next(k for k in stats if k.startswith("SBS"))
    assert "algo_b" in sbs_key
    assert stats[sbs_key]["par_score"] == pytest.approx(4.0, rel=1e-3)
    assert stats[sbs_key]["timeouts_count"] == 0


def test_selector_evaluation_with_and_without_feature_costs(
    mini_scenario_dir: Path, predictions_csv: Path
):
    """Verifies selector PAR10 scores and the impact of feature extraction costs."""
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)

    # 1. Without feature costs
    res_no_cost = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=False,
        run_significance_tests=False,
    )
    selectors_no_cost = {s["name"]: s for s in res_no_cost["rankings"]}
    # perfect_selector matches VBS exactly (2.6)
    assert selectors_no_cost["perfect_selector"]["par_score"] == pytest.approx(
        2.6, rel=1e-3
    )
    # static_sbs_selector matches SBS exactly (4.0)
    assert selectors_no_cost["static_sbs_selector"]["par_score"] == pytest.approx(
        4.0, rel=1e-3
    )
    # poor_selector has 2 timeouts on inst_4 and inst_5: 5+8+6+100+100 = 219 / 5 = 43.8
    assert selectors_no_cost["poor_selector"]["par_score"] == pytest.approx(
        43.8, rel=1e-3
    )
    assert selectors_no_cost["poor_selector"]["timeouts_count"] == 2

    # 2. With feature costs (feat_1 cost 0.05 + feat_2 cost 0.15 = 0.20 per instance)
    res_cost = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=True,
        run_significance_tests=False,
    )
    selectors_cost = {s["name"]: s for s in res_cost["rankings"]}
    # 2.6 + 0.20 = 2.80
    assert selectors_cost["perfect_selector"]["par_score"] == pytest.approx(
        2.8, rel=1e-3
    )


def test_selector_rankings_and_winner(
    mini_scenario_dir: Path, predictions_csv: Path, tmp_path: Path
):
    """Winning selector must be ranked first and artifacts created on disk."""
    out_dir = tmp_path / "eval_out"
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)
    res = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=False,
        run_significance_tests=True,
        output_dir=out_dir,
    )

    ranking = res["rankings"]
    # VBS is ranked #1 (score 2.6), perfect_selector is #2 (score 2.6) or #1 among selectors
    assert len(ranking) >= 2
    assert res["top_selector"] == "perfect_selector"
    assert len(res["significance_tests"]) >= 1

    # Check artifacts
    assert len(res["artifacts"]) >= 2
    for art in res["artifacts"]:
        assert Path(art["path"]).is_file()


def test_evaluate_missing_predictions_csv_raises(
    mini_scenario_dir: Path, tmp_path: Path
):
    """Missing predictions CSV file must raise FileNotFoundError."""
    evaluator = SelectorEvaluator(mini_scenario_dir)
    with pytest.raises(FileNotFoundError, match="Predictions CSV not found"):
        evaluator.evaluate(tmp_path / "missing_preds.csv")


def test_evaluation_reports_provenance_and_assumptions(mini_scenario_dir, tmp_path):
    from src.tools.selector_evaluator import SelectorEvaluator

    predictions = tmp_path / "choices.csv"
    predictions.write_text(
        "instance,selector\n" + "".join(f"inst_{i},algo_a\n" for i in range(1, 6))
    )
    result = SelectorEvaluator(mini_scenario_dir).evaluate(
        predictions, run_significance_tests=False
    )
    assert "algorithm_runs.arff" in result["provenance"]["sha256"]
    assert "descriptive" in result["assumptions"]["sbs_selection"]
    assert "unadjusted" in result["assumptions"]["significance"]


def edit_arff(path, mutate):
    with path.open() as stream:
        data = arff.load(stream)
    mutate(data)
    with path.open("w") as stream:
        arff.dump(data, stream)


@pytest.mark.parametrize("mixed_repetitions", [False, True])
def test_unknown_used_cost_is_rejected(mini_scenario_dir, tmp_path, mixed_repetitions):
    def change(data):
        data["data"][0][2] = None
        if mixed_repetitions:
            row = data["data"][0].copy()
            row[1], row[2] = 2, 0.2
            data["data"].append(row)

    edit_arff(mini_scenario_dir / "feature_costs.arff", change)
    evaluator = SelectorEvaluator(mini_scenario_dir)
    path = tmp_path / "choices.csv"
    pd.DataFrame({"selector": ["algo_b"] * 5}, index=evaluator.instances).to_csv(path)
    with pytest.raises(ValueError, match="Unknown.*cost"):
        evaluator.evaluate(path, run_significance_tests=False)
    # Deliberately excluding feature costs remains possible and explicit.
    result = evaluator.evaluate(
        path, include_feature_costs=False, run_significance_tests=False
    )
    assert not result["feature_costs_included"]
    stats = {row["name"]: row for row in result["rankings"]}
    assert stats["selector"]["par_score"] == 4.0


@pytest.mark.parametrize("presolved, score", [(False, 4.2), (True, 0.2)])
def test_unknown_cost_is_allowed_for_steps_not_executed(
    mini_scenario_dir, tmp_path, presolved, score
):
    import yaml

    description_path = mini_scenario_dir / "description.txt"
    description = yaml.safe_load(description_path.read_text())
    description["feature_steps"]["unused"] = {"provides": [], "requires": []}
    if presolved:
        description["default_steps"].append("unused")
    description_path.write_text(yaml.safe_dump(description))

    def costs(data):
        data["attributes"].append(("unused", "NUMERIC"))
        for row in data["data"]:
            row.append(None)

    def statuses(data):
        data["attributes"].append(("unused", ["unknown"]))
        for row in data["data"]:
            if presolved:
                row[2] = "presolved"
            row.append("unknown")

    edit_arff(mini_scenario_dir / "feature_costs.arff", costs)
    edit_arff(mini_scenario_dir / "feature_runstatus.arff", statuses)
    evaluator = SelectorEvaluator(mini_scenario_dir)
    path = tmp_path / "choices.csv"
    pd.DataFrame({"selector": ["algo_b"] * 5}, index=evaluator.instances).to_csv(path)
    result = evaluator.evaluate(path, run_significance_tests=False)
    stats = {row["name"]: row for row in result["rankings"]}
    assert stats["selector"]["par_score"] == score


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
