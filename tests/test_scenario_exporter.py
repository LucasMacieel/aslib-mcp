"""Unit tests for ASlib scenario exporter and CV generation."""

from pathlib import Path
from typing import Any

import arff
import pandas as pd
import pytest
from aslib_scenario.aslib_scenario import ASlibScenario

from src.tools.scenario_exporter import export_aslib_scenario
from src.tools.spec_checker import lint_scenario_spec


def test_export_runtime_scenario_full_roundtrip(synthetic_data: dict[str, Any], tmp_path: Path):
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

    for f in [
        "description.txt",
        "algorithm_runs.arff",
        "feature_values.arff",
        "feature_runstatus.arff",
        "feature_costs.arff",
        "cv.arff",
    ]:
        assert (out_dir / f).is_file()

    lint = lint_scenario_spec(out_dir)
    assert lint["is_valid"] is True
    assert len(lint["errors"]) == 0

    scen = ASlibScenario()
    scen.read_scenario(str(out_dir))
    assert scen.scenario == "EXPORTED-TEST"
    assert len(scen.instances) == 5
    assert len(scen.algorithms) == 3


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

    assert not (out_dir / "feature_costs.arff").is_file()
    lint = lint_scenario_spec(out_dir)
    assert lint["is_valid"] is True


def test_export_solution_quality_maximization(synthetic_data: dict[str, Any], tmp_path: Path):
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

    import yaml
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

    assigned_instances = [row[0] for row in cv_data["data"]]
    folds = {row[2] for row in cv_data["data"]}

    assert set(assigned_instances) == set(synthetic_data["instances"])
    assert len(assigned_instances) == len(synthetic_data["instances"])
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


def test_export_no_common_instances_raises(synthetic_data: dict[str, Any], tmp_path: Path):
    """Completely disjoint instances between performance and feature tables must raise ValueError."""
    disjoint_feat = pd.DataFrame([[1.0, 2.0]], index=["other_inst"], columns=["feat_1", "feat_2"])
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
