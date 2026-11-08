"""Unit tests for ASlib scenario exporter and CV generation."""

from pathlib import Path
from typing import Any

import arff
import numpy as np
import pandas as pd
import pytest
import yaml
from aslib_scenario.aslib_scenario import ASlibScenario

from src.tools.scenario_exporter import export_aslib_scenario
from src.tools.spec_checker import lint_scenario_spec


def test_export_runtime_scenario_full_roundtrip(
    synthetic_data: dict[str, Any], tmp_path: Path
):
    """Exported runtime scenario must create all files, pass spec linting, and load into ASlibScenario."""
    out_dir = tmp_path / "exported_runtime"
    res = export_aslib_scenario(
        perf_csv_path=synthetic_data["perf_csv"],
        feat_csv_path=synthetic_data["feat_csv"],
        scenario_id="EXPORTED-TEST",
        cutoff=10.0,
        output_dir=out_dir,
        objective="runtime",
        maximize=False,
        feature_costs_csv_path=synthetic_data["costs_csv"],
        generate_cv=True,
        n_folds=2,
        seed=42,
    )

    assert res["scenario_id"] == "EXPORTED-TEST"
    assert res["num_instances"] == 5
    assert len(res["algorithms"]) == 3
    assert len(res["features"]) == 2
    assert any(f.endswith("cv.arff") for f in res["created_files"])

    # 1. Verify files exist on disk
    for f in [
        "description.txt",
        "algorithm_runs.arff",
        "feature_values.arff",
        "feature_runstatus.arff",
        "feature_costs.arff",
        "cv.arff",
    ]:
        assert (out_dir / f).is_file()

    # 2. Spec linting must pass with 0 errors
    lint = lint_scenario_spec(out_dir)
    assert lint["is_valid"] is True
    assert len(lint["errors"]) == 0

    # 3. Must load cleanly into official ASlibScenario
    scen = ASlibScenario()
    scen.read_scenario(str(out_dir))
    assert scen.scenario == "EXPORTED-TEST"
    assert scen.instances is not None and len(scen.instances) == 5
    assert scen.algorithms is not None and len(scen.algorithms) == 3


def test_export_without_feature_costs(synthetic_data: dict[str, Any], tmp_path: Path):
    """Exporting without feature costs must omit feature_costs.arff and still remain valid."""
    out_dir = tmp_path / "exported_no_costs"
    res = export_aslib_scenario(
        perf_csv_path=synthetic_data["perf_csv"],
        feat_csv_path=synthetic_data["feat_csv"],
        scenario_id="NO-COSTS-TEST",
        cutoff=10.0,
        output_dir=out_dir,
        feature_costs_csv_path=None,
    )

    assert res["scenario_id"] == "NO-COSTS-TEST"
    assert not (out_dir / "feature_costs.arff").is_file()
    lint = lint_scenario_spec(out_dir)
    assert lint["is_valid"] is True


def test_export_solution_quality_maximization(
    synthetic_data: dict[str, Any], tmp_path: Path
):
    """Exporting solution quality objective with maximization."""
    out_dir = tmp_path / "exported_quality"
    res = export_aslib_scenario(
        perf_csv_path=synthetic_data["perf_csv"],
        feat_csv_path=synthetic_data["feat_csv"],
        scenario_id="QUALITY-MAX",
        cutoff=0.0,
        output_dir=out_dir,
        objective="solution_quality",
        maximize=True,
    )

    lint = lint_scenario_spec(out_dir)
    assert lint["is_valid"] is True
    assert res["objective"] == "solution_quality"


    with open(out_dir / "description.txt", "r", encoding="utf-8") as f:
        desc = yaml.safe_load(f)
    assert desc["maximize"] == [True]


def test_export_cv_partition_integrity(synthetic_data: dict[str, Any], tmp_path: Path):
    """Generated CV splits must partition instances without duplicate test assignments."""
    out_dir = tmp_path / "exported_cv"
    export_aslib_scenario(
        perf_csv_path=synthetic_data["perf_csv"],
        feat_csv_path=synthetic_data["feat_csv"],
        scenario_id="CV-CHECK",
        cutoff=10.0,
        output_dir=out_dir,
        generate_cv=True,
        n_folds=3,
        seed=123,
    )

    with open(out_dir / "cv.arff", "r", encoding="utf-8") as f:
        cv_data = arff.load(f)

    # cv.arff format: instance_id, repetition, fold
    assigned_instances = [row[0] for row in cv_data["data"]]
    folds = {row[2] for row in cv_data["data"]}

    # All instances accounted for
    assert set(assigned_instances) == set(synthetic_data["instances"])
    assert len(assigned_instances) == len(synthetic_data["instances"])  # No duplicates
    assert len(folds) == 3


def test_export_missing_perf_csv_raises(tmp_path: Path):
    """Non-existent performance CSV path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Performance CSV not found"):
        export_aslib_scenario(
            perf_csv_path=tmp_path / "nonexistent.csv",
            feat_csv_path=tmp_path / "feat.csv",
            scenario_id="FAIL",
            cutoff=10.0,
            output_dir=tmp_path / "out",
        )


def test_export_no_common_instances_raises(
    synthetic_data: dict[str, Any], tmp_path: Path
):
    """Completely disjoint instances between performance and feature tables must raise ValueError."""
    disjoint_feat = pd.DataFrame(
        [[1.0, 2.0]], index=["other_inst"], columns=["feat_1", "feat_2"]
    )
    disjoint_csv = tmp_path / "disjoint_feat.csv"
    disjoint_feat.to_csv(disjoint_csv)

    with pytest.raises(ValueError, match="No common instances found"):
        export_aslib_scenario(
            perf_csv_path=synthetic_data["perf_csv"],
            feat_csv_path=disjoint_csv,
            scenario_id="DISJOINT",
            cutoff=10.0,
            output_dir=tmp_path / "out",
        )


def test_export_rejects_existing_contents(synthetic_data, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "cv.arff"
    sentinel.write_text("old folds")
    with pytest.raises(ValueError, match="empty"):
        export_aslib_scenario(
            synthetic_data["perf_csv"], synthetic_data["feat_csv"], "X", 10, destination
        )
    assert sentinel.read_text() == "old folds"


def test_export_preserves_explicit_feature_status(synthetic_data, tmp_path):
    statuses = tmp_path / "statuses.csv"
    statuses.write_text(
        "instance,ALL\n" + "".join(f"{i},crash\n" for i in synthetic_data["instances"])
    )
    destination = tmp_path / "out"
    export_aslib_scenario(
        synthetic_data["perf_csv"],
        synthetic_data["feat_csv"],
        "X",
        10,
        destination,
        feature_runstatus_csv_path=statuses,
    )
    with (destination / "feature_runstatus.arff").open() as stream:
        assert all(row[2] == "crash" for row in arff.load(stream)["data"])


@pytest.mark.parametrize(
    "contents", ["instance,ALL\ninst_1,bogus\n", "instance,wrong\ninst_1,ok\n"]
)
def test_invalid_feature_statuses_do_not_write(synthetic_data, tmp_path, contents):
    statuses = tmp_path / "bad.csv"
    statuses.write_text(contents)
    destination = tmp_path / "out"
    with pytest.raises(ValueError):
        export_aslib_scenario(
            synthetic_data["perf_csv"],
            synthetic_data["feat_csv"],
            "X",
            10,
            destination,
            feature_runstatus_csv_path=statuses,
        )
    assert not destination.exists()


def test_missing_feature_status_is_unknown(synthetic_data, tmp_path):
    features = synthetic_data["feat_df"].copy()
    features.iloc[0, 0] = np.nan
    features.to_csv(synthetic_data["feat_csv"])
    destination = tmp_path / "out"
    result = export_aslib_scenario(
        synthetic_data["perf_csv"], synthetic_data["feat_csv"], "X", 10, destination
    )
    with (destination / "feature_runstatus.arff").open() as stream:
        assert arff.load(stream)["data"][0][2] == "unknown"
    assert any("unknown" in assumption for assumption in result["assumptions"])


def test_invalid_status_value_is_rejected(synthetic_data, tmp_path):
    statuses = tmp_path / "bad.csv"
    statuses.write_text(
        "instance,ALL\n"
        + "".join(f"{i},invalid\n" for i in synthetic_data["instances"])
    )
    with pytest.raises(ValueError, match="Invalid feature runstatus"):
        export_aslib_scenario(
            synthetic_data["perf_csv"],
            synthetic_data["feat_csv"],
            "X",
            10,
            tmp_path / "out",
            feature_runstatus_csv_path=statuses,
        )


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
