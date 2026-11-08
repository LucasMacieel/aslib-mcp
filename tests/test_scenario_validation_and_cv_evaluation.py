"""Scenario validation, repeated CV splits, and held-out selector evaluation."""

import arff
import pandas as pd
import pytest

from src.tools.aslib_scenario import aslib_get_cv_split
from src.tools.scenario_loader import ASlibScenario
from src.tools.selector_evaluator import SelectorEvaluator
from src.tools.spec_checker import lint_scenario_spec


def edit_arff(path, change):
    with path.open() as stream:
        data = arff.load(stream)
    change(data)
    with path.open("w") as stream:
        arff.dump(data, stream)


def repeated_cv(path):
    def change(data):
        data["data"] = [
            [f"inst_{i}", rep, 1 if i in test else 2]
            for rep, test in [(1, {1, 2}), (2, {3, 4})]
            for i in range(1, 6)
        ]

    edit_arff(path / "cv.arff", change)


def test_negative_raw_runtime_cannot_hide_in_median(mini_scenario_dir):
    def change(data):
        rows = [row.copy() for row in data["data"]]
        for row in rows:
            row[1] = 2
        rows[0][3] = -1
        data["data"].extend(rows)

    edit_arff(mini_scenario_dir / "algorithm_runs.arff", change)
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]
    with pytest.raises(ValueError, match="[Rr]untime"):
        ASlibScenario().read_scenario(str(mini_scenario_dir))


def test_undeclared_feature_rejected_by_linter(mini_scenario_dir):
    def change(data):
        data["attributes"][2] = ("undeclared", "NUMERIC")

    edit_arff(mini_scenario_dir / "feature_values.arff", change)
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]


@pytest.mark.parametrize("fold", [-0.5, 0, None, 1.5])
def test_invalid_arff_cv_fold_rejected(mini_scenario_dir, fold):
    def change(data):
        data["data"][0][2] = fold

    edit_arff(mini_scenario_dir / "cv.arff", change)
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]
    with pytest.raises(ValueError, match="fold"):
        ASlibScenario().read_scenario(str(mini_scenario_dir))


def test_every_cv_repetition_requires_complete_coverage(mini_scenario_dir):
    repeated_cv(mini_scenario_dir)
    edit_arff(mini_scenario_dir / "cv.arff", lambda data: data["data"].pop())
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]
    with pytest.raises(ValueError, match="instances"):
        ASlibScenario().read_scenario(str(mini_scenario_dir))


def test_second_cv_repetition_is_selectable(mini_scenario_dir):
    repeated_cv(mini_scenario_dir)
    result = aslib_get_cv_split(str(mini_scenario_dir), fold_index=1, repetition=2)
    assert result["test_instances"] == ["inst_3", "inst_4"]
    assert result["available_repetitions"] == [1, 2]
    with pytest.raises(ValueError, match="repetition"):
        aslib_get_cv_split(str(mini_scenario_dir), repetition=3)


def test_fold_evaluation_selects_sbs_on_training_only(mini_scenario_dir, tmp_path):
    repeated_cv(mini_scenario_dir)

    def training_winner(data):
        for row in data["data"]:
            if row[0] in {"inst_3", "inst_4", "inst_5"} and row[2] == "algo_a":
                row[3], row[4] = 0.5, "ok"

    edit_arff(mini_scenario_dir / "algorithm_runs.arff", training_winner)
    pred = tmp_path / "choices.csv"
    pd.DataFrame({"selector": ["algo_a", "algo_b"]}, index=["inst_1", "inst_2"]).to_csv(
        pred
    )
    evaluator = SelectorEvaluator(mini_scenario_dir)
    result = evaluator.evaluate(
        pred, fold_index=1, include_feature_costs=False, run_significance_tests=False
    )
    stats = {r["name"]: r for r in result["rankings"]}
    assert result["num_instances"] == 2
    assert stats["selector"]["par_score"] == 2.0
    assert stats["SBS (algo_a)"]["par_score"] == 4.5
    assert result["training_instances"] == ["inst_3", "inst_4", "inst_5"]
    assert evaluator.instances == [f"inst_{i}" for i in range(1, 6)]
    # The second repetition has a different test set and must not retain state.
    pd.DataFrame({"selector": ["algo_c", "algo_b"]}, index=["inst_3", "inst_4"]).to_csv(
        pred
    )
    second = evaluator.evaluate(
        pred,
        fold_index=1,
        repetition=2,
        include_feature_costs=False,
        run_significance_tests=False,
    )
    assert second["evaluation_instances"] == ["inst_3", "inst_4"]
    with pytest.raises(ValueError, match="instances"):
        evaluator.evaluate(pred, fold_index=1, repetition=1)


def test_read_preserves_all_cv_repetitions(mini_scenario_dir):
    from src.tools.aslib_scenario import aslib_read_scenario

    repeated_cv(mini_scenario_dir)
    result = aslib_read_scenario(str(mini_scenario_dir))
    assert result["available_repetitions"] == [1, 2]
    artifact = next(
        a for a in result["artifacts"] if a["path"].endswith("cv_repetitions.csv")
    )
    table = pd.read_csv(artifact["path"])
    assert len(table) == 10
    assert set(
        table.loc[(table.repetition == 2) & (table.fold == 1), "instance_id"]
    ) == {"inst_3", "inst_4"}


def test_quality_fold_uses_training_maximization(make_scenario, tmp_path):
    path = make_scenario(objective="solution_quality", maximize=True)
    repeated_cv(path)

    def change(data):
        for row in data["data"]:
            if row[0] in {"inst_1", "inst_2"}:
                row[3] = {"algo_a": -2, "algo_b": 3, "algo_c": 0}[row[2]]
            else:
                row[3] = {"algo_a": 4, "algo_b": -1, "algo_c": 0}[row[2]]

    edit_arff(path / "algorithm_runs.arff", change)
    pred = tmp_path / "quality.csv"
    pred.write_text("instance,selector\ninst_1,algo_b\ninst_2,algo_b\n")
    result = SelectorEvaluator(path).evaluate(
        pred, fold_index=1, run_significance_tests=False
    )
    stats = {r["name"]: r for r in result["rankings"]}
    assert stats["SBS (algo_a)"]["par_score"] == -2
    assert stats["selector"]["par_score"] == 3
    assert stats["VBS"]["par_score"] == 3


def test_training_unknown_cost_does_not_block_fold(mini_scenario_dir, tmp_path):
    repeated_cv(mini_scenario_dir)

    def change(data):
        for row in data["data"]:
            if row[0] in {"inst_3", "inst_4", "inst_5"}:
                row[2] = None

    edit_arff(mini_scenario_dir / "feature_costs.arff", change)
    pred = tmp_path / "choices.csv"
    pred.write_text("instance,selector\ninst_1,algo_a\ninst_2,algo_b\n")
    result = SelectorEvaluator(mini_scenario_dir).evaluate(
        pred, fold_index=1, run_significance_tests=False
    )
    stats = {r["name"]: r for r in result["rankings"]}
    assert stats["selector"]["par_score"] == 2.2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fold_index": 99},
        {"fold_index": 0},
        {"repetition": 2},
        {"fold_index": 1, "repetition": 3},
    ],
)
def test_invalid_evaluation_partition_rejected(mini_scenario_dir, tmp_path, kwargs):
    repeated_cv(mini_scenario_dir)
    pred = tmp_path / "choices.csv"
    pred.write_text("instance,selector\ninst_1,algo_a\n")
    with pytest.raises(ValueError):
        SelectorEvaluator(mini_scenario_dir).evaluate(pred, **kwargs)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing_repetition", "wrong_schema", "missing_fold"]
)
def test_malformed_repeated_cv_rejected(mini_scenario_dir, mutation):
    repeated_cv(mini_scenario_dir)

    def change(data):
        if mutation == "duplicate":
            data["data"].append(data["data"][0].copy())
        elif mutation == "missing_repetition":
            for row in data["data"]:
                row[1] += 1
        elif mutation == "wrong_schema":
            data["attributes"][2] = ("other", "NUMERIC")
        else:
            data["data"][0][2] = None

    edit_arff(mini_scenario_dir / "cv.arff", change)
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]
    with pytest.raises(ValueError):
        ASlibScenario().read_scenario(str(mini_scenario_dir))


def test_runtime_validation_honors_upstream_case_insensitive_headers(mini_scenario_dir):
    def change(data):
        name, kind = data["attributes"][3]
        data["attributes"][3] = (name.upper(), kind)
        rows = [row.copy() for row in data["data"]]
        for row in rows:
            row[1] = 2
        rows[0][3] = -1
        data["data"].extend(rows)

    edit_arff(mini_scenario_dir / "algorithm_runs.arff", change)
    with pytest.raises(ValueError, match="Runtime"):
        ASlibScenario().read_scenario(str(mini_scenario_dir))
