"""Negative conformance cases, independent of the exporter's own assertions."""

import arff
import pandas as pd
import pytest
import yaml

from src.tools.scenario_exporter import export_aslib_scenario
from src.tools.spec_checker import lint_scenario_spec


def change_arff(path, mutate):
    with path.open() as f:
        data = arff.load(f)
    mutate(data)
    with path.open("w") as f:
        arff.dump(data, f)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "type", "algorithm", "repetition"]
)
def test_invalid_algorithm_runs_rejected(mini_scenario_dir, mutation):
    def mutate(data):
        if mutation == "missing":
            data["data"][0][3] = None
        elif mutation == "duplicate":
            data["data"].append(data["data"][0].copy())
        elif mutation == "type":
            data["attributes"][3] = ("runtime", "STRING")
        elif mutation == "algorithm":
            data["data"][0][2] = "undeclared"
        else:
            data["data"][0][1] = 0

    change_arff(mini_scenario_dir / "algorithm_runs.arff", mutate)
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]


@pytest.mark.parametrize(
    "mutation", ["length", "boolean", "cycle", "null_type", "configuration"]
)
def test_invalid_metadata_rejected(mini_scenario_dir, mutation):
    path = mini_scenario_dir / "description.txt"
    desc = yaml.safe_load(path.read_text())
    if mutation == "length":
        desc["maximize"] = []
    elif mutation == "boolean":
        desc["maximize"] = ["false"]
    elif mutation == "cycle":
        desc["feature_steps"]["ALL"]["requires"] = ["ALL"]
    elif mutation == "null_type":
        desc["performance_type"] = None
    else:
        desc["metainfo_algorithms"]["algo_a"].pop("configuration", None)
    path.write_text(yaml.safe_dump(desc))
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]


def test_partial_instance_overlap_is_not_silently_dropped(synthetic_data, tmp_path):
    features = pd.read_csv(synthetic_data["feat_csv"], index_col=0).iloc[:-1]
    features.to_csv(synthetic_data["feat_csv"])
    with pytest.raises(ValueError, match="same instances"):
        export_aslib_scenario(
            synthetic_data["perf_csv"],
            synthetic_data["feat_csv"],
            "PARTIAL",
            10,
            tmp_path / "out",
        )


def test_quality_scores_do_not_imply_timeouts(synthetic_data, tmp_path):
    out = tmp_path / "quality"
    export_aslib_scenario(
        synthetic_data["perf_csv"],
        synthetic_data["feat_csv"],
        "QUALITY",
        1,
        out,
        objective="solution_quality",
    )
    with (out / "algorithm_runs.arff").open() as f:
        data = arff.load(f)
    assert {row[-1] for row in data["data"]} == {"ok"}


def test_missing_quality_scores_require_domain_imputation(synthetic_data, tmp_path):
    perf = pd.read_csv(synthetic_data["perf_csv"], index_col=0)
    perf.iloc[0, 0] = float("nan")
    perf.to_csv(synthetic_data["perf_csv"])
    with pytest.raises(ValueError, match="quality"):
        export_aslib_scenario(
            synthetic_data["perf_csv"],
            synthetic_data["feat_csv"],
            "MISSING",
            10,
            tmp_path / "out",
            objective="solution_quality",
        )


def test_export_preserves_negative_feature_values(synthetic_data, tmp_path):
    features = pd.read_csv(synthetic_data["feat_csv"], index_col=0)
    features.iloc[0, 0] = -512
    features.to_csv(synthetic_data["feat_csv"])
    out = tmp_path / "out"
    export_aslib_scenario(
        synthetic_data["perf_csv"], synthetic_data["feat_csv"], "NEG", 10, out
    )
    with (out / "feature_values.arff").open() as f:
        assert arff.load(f)["data"][0][2] == -512


def test_duplicate_csv_headers_rejected_before_pandas_mangling(
    synthetic_data, tmp_path
):
    synthetic_data["perf_csv"].write_text("id,algo,algo\ni1,1,2\n")
    synthetic_data["feat_csv"].write_text("id,f\ni1,1\n")
    with pytest.raises(ValueError, match="unique"):
        export_aslib_scenario(
            synthetic_data["perf_csv"],
            synthetic_data["feat_csv"],
            "DUP",
            10,
            tmp_path / "out",
            generate_cv=False,
        )


@pytest.mark.parametrize("field", ["requires", "default_steps"])
def test_unhashable_step_names_return_diagnostics(mini_scenario_dir, field):
    path = mini_scenario_dir / "description.txt"
    desc = yaml.safe_load(path.read_text())
    if field == "requires":
        desc["feature_steps"]["ALL"][field] = [{}]
    else:
        desc[field] = [{}]
    path.write_text(yaml.safe_dump(desc))
    assert not lint_scenario_spec(mini_scenario_dir)["is_valid"]
