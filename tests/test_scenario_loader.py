"""Tests for scenario loader."""

from pathlib import Path

import arff
import pandas as pd
import pytest

from src.tools.aslib_scenario import (
    aslib_read_csv,
    aslib_read_scenario,
    aslib_validate_scenario,
)
from src.tools.selector_evaluator import SelectorEvaluator


def edit_arff(path, mutate):
    with path.open() as stream:
        data = arff.load(stream)
    mutate(data)
    with path.open("w") as stream:
        arff.dump(data, stream)


def test_read_artifact_preserves_unknown_cost(mini_scenario_dir):
    edit_arff(
        mini_scenario_dir / "feature_costs.arff",
        lambda data: data["data"][0].__setitem__(2, None),
    )
    result = aslib_read_scenario(str(mini_scenario_dir))
    artifact = next(a for a in result["artifacts"] if "cost" in a["description"])
    assert pd.isna(pd.read_csv(artifact["path"], index_col=0).loc["inst_1", "ALL"])


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
