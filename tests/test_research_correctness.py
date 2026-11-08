"""Regression tests for scientific results and usable scenario loading."""

from pathlib import Path

import arff
import pandas as pd
import pytest

from src.tools.aslib_scenario import (
    aslib_permutation_test,
    aslib_read_csv,
    aslib_read_scenario,
    aslib_validate_scenario,
)
from src.tools.selector_evaluator import SelectorEvaluator
from src.tools.spec_checker import lint_scenario_spec


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


def test_read_artifact_preserves_unknown_cost(mini_scenario_dir):
    edit_arff(
        mini_scenario_dir / "feature_costs.arff",
        lambda data: data["data"][0].__setitem__(2, None),
    )
    result = aslib_read_scenario(str(mini_scenario_dir))
    artifact = next(a for a in result["artifacts"] if "cost" in a["description"])
    assert pd.isna(pd.read_csv(artifact["path"], index_col=0).loc["inst_1", "ALL"])


def test_lint_rejects_missing_algorithm_measurement(mini_scenario_dir):
    edit_arff(
        mini_scenario_dir / "algorithm_runs.arff", lambda data: data["data"].pop(0)
    )
    result = lint_scenario_spec(mini_scenario_dir)
    assert not result["is_valid"]
    assert any("coverage" in error.lower() for error in result["errors"])


@pytest.mark.parametrize(
    "consumer", [aslib_read_scenario, aslib_validate_scenario, SelectorEvaluator]
)
def test_incomplete_scenario_raises_recoverable_error(mini_scenario_dir, consumer):
    edit_arff(
        mini_scenario_dir / "algorithm_runs.arff", lambda data: data["data"].pop(0)
    )
    with pytest.raises(ValueError, match="[Ss]cenario|[Pp]erformance"):
        consumer(str(mini_scenario_dir))


@pytest.mark.parametrize("with_cv", [False, True])
def test_csv_preserves_ids_and_supports_small_datasets(tmp_path, with_cv):
    perf = tmp_path / "perf.csv"
    feat = tmp_path / "feat.csv"
    cv = tmp_path / "cv.csv"
    perf.write_text("id,a,b\n001,1,2\n1,3,4\n")
    feat.write_text("id,f\n001,10\n1,20\n")
    cv.write_text("id,fold\n001,1\n1,2\n")
    result = aslib_read_csv(
        str(perf), str(feat), cv_csv_path=str(cv) if with_cv else None
    )
    for artifact in result["artifacts"]:
        table = pd.read_csv(artifact["path"], dtype=str, keep_default_na=False)
        assert table.iloc[:, 0].tolist() == ["001", "1"]
    assert result["num_instances"] == 2


def test_csv_rejects_extra_performance_instances(tmp_path):
    perf, feat, cv = (tmp_path / name for name in ("perf.csv", "feat.csv", "cv.csv"))
    perf.write_text("id,a\ni1,1\ni2,2\n")
    feat.write_text("id,f\ni1,1\n")
    cv.write_text("id,fold\ni1,1\n")
    with pytest.raises(ValueError, match="same instances"):
        aslib_read_csv(str(perf), str(feat), cv_csv_path=str(cv))


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


def test_categorical_features_aggregate_by_mode(mini_scenario_dir):
    def change(data):
        data["attributes"][2] = ("feat_1", ["low", "high"])
        for row in data["data"]:
            row[2] = "low"
        row = data["data"][0].copy()
        row[1], row[2] = 2, "high"
        data["data"].append(row)
        row = row.copy()
        row[1], row[3] = 3, 0.7
        data["data"].append(row)

    edit_arff(mini_scenario_dir / "feature_values.arff", change)
    result = aslib_read_scenario(str(mini_scenario_dir))
    artifact = next(
        a
        for a in result["artifacts"]
        if a["description"] == "Instance feature data matrix"
    )
    table = pd.read_csv(Path(artifact["path"]), index_col=0)
    assert table.loc["inst_1", "feat_1"] == "high"
    assert table.loc["inst_1", "feat_2"] == pytest.approx(7.2333333333)


@pytest.mark.parametrize("first, second", [("low", "high"), ("high", "low")])
def test_categorical_mode_tie_keeps_first_observed(mini_scenario_dir, first, second):
    def change(data):
        data["attributes"][2] = ("feat_1", ["low", "high"])
        for row in data["data"]:
            row[2] = first
        row = data["data"][0].copy()
        row[1], row[2] = 2, second
        data["data"].append(row)

    edit_arff(mini_scenario_dir / "feature_values.arff", change)
    result = aslib_read_scenario(str(mini_scenario_dir))
    artifact = next(
        a for a in result["artifacts"]
        if a["description"] == "Instance feature data matrix"
    )
    table = pd.read_csv(artifact["path"], index_col=0)
    assert table.loc["inst_1", "feat_1"] == first


def test_categorical_all_missing_repetitions_remain_missing(mini_scenario_dir):
    def change(data):
        data["attributes"][2] = ("feat_1", ["low", "high"])
        for row in data["data"]:
            row[2] = "low"
        data["data"][0][2] = None
        row = data["data"][0].copy()
        row[1] = 2
        data["data"].append(row)

    edit_arff(mini_scenario_dir / "feature_values.arff", change)
    result = aslib_read_scenario(str(mini_scenario_dir))
    artifact = next(
        a for a in result["artifacts"]
        if a["description"] == "Instance feature data matrix"
    )
    table = pd.read_csv(artifact["path"], index_col=0)
    assert pd.isna(table.loc["inst_1", "feat_1"])
    assert table.loc["inst_2", "feat_1"] == "low"
    assert table.index.tolist() == [f"inst_{i}" for i in range(1, 6)]


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
