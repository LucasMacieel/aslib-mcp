"""Unit and negative tests for ASlibSpecLinter and spec_checker."""

import shutil
from collections.abc import Callable
from pathlib import Path

import arff
import pytest
import yaml

from src.tools.spec_checker import ASlibSpecLinter, lint_scenario_spec


def test_lint_valid_synthetic_scenario(mini_scenario_dir: Path):
    """A valid synthetic scenario must pass all linter checks with 0 errors."""
    linter = ASlibSpecLinter(mini_scenario_dir)
    res = linter.lint()

    assert res["is_valid"] is True
    assert res["scenario_id"] == "SYNTHETIC-SCENARIO"
    assert len(res["errors"]) == 0
    assert "description.txt" in res["checked_files"]
    assert "algorithm_runs.arff" in res["checked_files"]
    assert "feature_values.arff" in res["checked_files"]
    assert "feature_runstatus.arff" in res["checked_files"]
    assert "feature_costs.arff" in res["checked_files"]
    assert "cv.arff" in res["checked_files"]
    assert res["summary"]["num_errors"] == 0


def test_lint_nonexistent_directory(tmp_path: Path):
    """Linting a non-existent directory must return is_valid=False with descriptive error."""
    missing_dir = tmp_path / "does_not_exist"
    res = lint_scenario_spec(missing_dir)

    assert res["is_valid"] is False
    assert res["scenario_id"] is None
    assert any("Directory not found" in e for e in res["errors"])


def test_lint_missing_required_file(mini_scenario_dir: Path, tmp_path: Path):
    """Deleting algorithm_runs.arff must trigger a missing required specification file error."""
    corrupt_dir = tmp_path / "missing_runs"
    shutil.copytree(mini_scenario_dir, corrupt_dir)

    (corrupt_dir / "algorithm_runs.arff").unlink()

    res = lint_scenario_spec(corrupt_dir)
    assert res["is_valid"] is False
    assert any(
        "Missing required ASlib specification file: algorithm_runs.arff" in e
        for e in res["errors"]
    )


def test_lint_malformed_description_yaml(mini_scenario_dir: Path, tmp_path: Path):
    """Corrupting description.txt syntax must fail YAML parsing."""
    corrupt_dir = tmp_path / "bad_yaml"
    shutil.copytree(mini_scenario_dir, corrupt_dir)

    desc_path = corrupt_dir / "description.txt"
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write("scenario_id: [unclosed list\nkey: value")

    res = lint_scenario_spec(corrupt_dir)
    assert res["is_valid"] is False
    assert any(
        "Failed to parse description.txt as valid YAML" in e for e in res["errors"]
    )


def test_lint_missing_description_required_keys(
    mini_scenario_dir: Path, tmp_path: Path
):
    """A description missing required keys (e.g. scenario_id) must be rejected."""
    corrupt_dir = tmp_path / "missing_keys"
    shutil.copytree(mini_scenario_dir, corrupt_dir)

    desc_path = corrupt_dir / "description.txt"
    with open(desc_path, "r", encoding="utf-8") as f:
        desc = yaml.safe_load(f)

    del desc["scenario_id"]
    del desc["performance_measures"]

    with open(desc_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(desc, f)

    res = lint_scenario_spec(corrupt_dir)
    assert res["is_valid"] is False
    assert any("missing required key: 'scenario_id'" in e for e in res["errors"])
    assert any(
        "missing required key: 'performance_measures'" in e for e in res["errors"]
    )


def test_lint_instance_mismatch_across_files(mini_scenario_dir: Path, tmp_path: Path):
    """Instances present in algorithm_runs but missing from feature_values must trigger cross-validation error."""
    corrupt_dir = tmp_path / "instance_mismatch"
    shutil.copytree(mini_scenario_dir, corrupt_dir)

    runs_path = corrupt_dir / "algorithm_runs.arff"
    with open(runs_path, "r", encoding="utf-8") as f:
        runs_data = arff.load(f)

    # Append run for phantom instance
    runs_data["data"].append(["inst_phantom", 1, "algo_a", 2.0, "ok"])
    with open(runs_path, "w", encoding="utf-8") as f:
        arff.dump(runs_data, f)

    res = lint_scenario_spec(corrupt_dir)
    assert res["is_valid"] is False
    assert any("missing from feature_values.arff" in e for e in res["errors"])


def test_lint_invalid_runstatus_rejected(mini_scenario_dir: Path, tmp_path: Path):
    """Unknown or invalid algorithm runstatus values must be detected and flagged."""
    corrupt_dir = tmp_path / "invalid_runstatus"
    shutil.copytree(mini_scenario_dir, corrupt_dir)

    runs_path = corrupt_dir / "algorithm_runs.arff"
    with open(runs_path, "r", encoding="utf-8") as f:
        runs_data = arff.load(f)

    # Inject unknown status
    runs_data["data"][0][-1] = "exploded"
    with open(runs_path, "w", encoding="utf-8") as f:
        arff.dump(runs_data, f)

    res = lint_scenario_spec(corrupt_dir)
    assert res["is_valid"] is False
    assert any(
        "Failed to parse algorithm_runs.arff" in e or "Invalid runstatus" in e
        for e in res["errors"]
    )


def test_lint_strict_mode_raises(mini_scenario_dir: Path, tmp_path: Path):
    """Running linter with strict=True on an invalid scenario must raise ValueError."""
    corrupt_dir = tmp_path / "strict_fail"
    shutil.copytree(mini_scenario_dir, corrupt_dir)
    (corrupt_dir / "algorithm_runs.arff").unlink()

    with pytest.raises(ValueError) as exc_info:
        lint_scenario_spec(corrupt_dir, strict=True)
    assert "validation failed" in str(exc_info.value)


def test_lint_missing_cv_warning(make_scenario: Callable[..., Path], tmp_path: Path):
    """A scenario without cv.arff is valid but generates a warning recommending CV splits."""
    target_dir = tmp_path / "no_cv_scenario"
    make_scenario(scenario_id="NO-CV", output_dir=target_dir)

    cv_path = target_dir / "cv.arff"
    if cv_path.is_file():
        cv_path.unlink()

    res = lint_scenario_spec(target_dir)
    assert res["is_valid"] is True
    assert any("cv.arff not found" in w for w in res["warnings"])
