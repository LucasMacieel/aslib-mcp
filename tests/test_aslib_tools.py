"""Functional tests for all 14 MCP tool wrappers in aslib_scenario.py."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.tools.aslib_scenario import (
    aslib_change_perf_measure,
    aslib_create_cv_splits,
    aslib_evaluate_selectors,
    aslib_export_scenario,
    aslib_get_cv_split,
    aslib_lint_spec,
    aslib_permutation_test,
    aslib_read_csv,
    aslib_read_scenario,
    aslib_validate_scenario,
)


def test_mcp_tool_aslib_read_scenario(mini_scenario_dir: Path, tmp_path: Path):
    """aslib_read_scenario parses scenario and emits summary artifacts."""
    out_dir = tmp_path / "read_out"
    res = aslib_read_scenario(str(mini_scenario_dir), output_dir=str(out_dir))

    assert res["scenario_id"] == "SYNTHETIC-SCENARIO"
    assert res["num_instances"] == 5
    assert set(res["algorithms"]) == {"algo_a", "algo_b", "algo_c"}
    assert len(res["artifacts"]) >= 2
    for art in res["artifacts"]:
        assert Path(art["path"]).is_file()


def test_mcp_tool_aslib_validate_scenario(mini_scenario_dir: Path):
    """aslib_validate_scenario confirms valid synthetic scenario integrity."""
    res = aslib_validate_scenario(str(mini_scenario_dir))
    assert res["is_valid"] is True
    assert res["num_instances"] == 5
    assert res["num_algorithms"] == 3
    assert res["instances_aligned"] is True


def test_mcp_tool_aslib_lint_spec(mini_scenario_dir: Path):
    """aslib_lint_spec verifies specification conformance."""
    res = aslib_lint_spec(str(mini_scenario_dir))
    assert res["is_valid"] is True
    assert res["scenario_id"] == "SYNTHETIC-SCENARIO"
    assert len(res["errors"]) == 0


def test_mcp_tool_aslib_read_csv(tmp_path: Path):
    """aslib_read_csv ingests raw tabular CSVs and writes scenario files."""
    # 10 instances to satisfy default 10-fold CV partitioning
    instances = [f"inst_{i}" for i in range(10)]
    algos = ["algo_1", "algo_2"]
    feats = ["f1", "f2"]

    perf_csv = tmp_path / "perf_10.csv"
    feat_csv = tmp_path / "feat_10.csv"
    pd.DataFrame([[1.0, 2.0]] * 10, index=instances, columns=algos).to_csv(perf_csv)
    pd.DataFrame([[0.5, 1.5]] * 10, index=instances, columns=feats).to_csv(feat_csv)

    out_dir = tmp_path / "csv_out"
    res = aslib_read_csv(
        perf_csv_path=str(perf_csv),
        feat_csv_path=str(feat_csv),
        runtime_cutoff=10.0,
        output_dir=str(out_dir),
    )

    assert res["num_instances"] == 10
    assert len(res["algorithms"]) == 2
    assert len(res["features"]) == 2
    assert len(res["artifacts"]) >= 1


def test_mcp_tool_aslib_get_cv_split(mini_scenario_dir: Path, tmp_path: Path):
    """aslib_get_cv_split extracts test and train partitions for a fold."""
    out_dir = tmp_path / "cv_split_out"
    res = aslib_get_cv_split(
        scenario_dir=str(mini_scenario_dir),
        fold_index=1,
        output_dir=str(out_dir),
    )

    assert res["fold_index"] == 1
    assert res["train_num_instances"] + res["test_num_instances"] == 5
    assert len(res["artifacts"]) >= 2


def test_mcp_tool_aslib_create_cv_splits(mini_scenario_dir: Path, tmp_path: Path):
    """aslib_create_cv_splits generates new balanced cross-validation folds."""
    out_dir = tmp_path / "new_cv_out"
    res = aslib_create_cv_splits(
        scenario_dir=str(mini_scenario_dir),
        n_folds=2,
        seed=42,
        output_dir=str(out_dir),
    )

    assert res["n_folds"] == 2
    assert res["total_instances"] == 5
    assert sum(res["instances_per_fold"].values()) == 5
    assert len(res["artifacts"]) >= 1


def test_mcp_tool_aslib_change_perf_measure(mini_scenario_dir: Path, tmp_path: Path):
    """aslib_change_perf_measure selects active performance measure."""
    out_dir = tmp_path / "measure_out"
    res = aslib_change_perf_measure(
        scenario_dir=str(mini_scenario_dir),
        measure_name="runtime",
        output_dir=str(out_dir),
    )

    assert res["active_measure_name"] == "runtime"
    assert "mean_performance_per_algorithm" in res
    assert len(res["artifacts"]) >= 1


def test_mcp_tool_aslib_export_scenario(synthetic_data: dict[str, Any], tmp_path: Path):
    """aslib_export_scenario exports tabular CSV into full ASlib 2.0 scenario."""
    out_dir = tmp_path / "exported_mcp_scenario"
    res = aslib_export_scenario(
        perf_csv_path=str(synthetic_data["perf_csv"]),
        feat_csv_path=str(synthetic_data["feat_csv"]),
        scenario_id="MCP-EXPORT",
        cutoff=10.0,
        output_dir=str(out_dir),
        feature_costs_csv_path=str(synthetic_data["costs_csv"]),
        generate_cv=True,
        n_folds=2,
        seed=123,
    )

    assert res["scenario_id"] == "MCP-EXPORT"
    assert res["num_instances"] == 5
    assert len(res["algorithms"]) == 3
    assert (out_dir / "description.txt").is_file()


def test_mcp_tool_aslib_permutation_test_vectors():
    """aslib_permutation_test running in direct vector comparison mode."""
    res = aslib_permutation_test(
        vec1=[1.0, 2.0, 1.5, 3.0],
        vec2=[1.0, 2.0, 1.5, 3.0],
        permutations=1000,
        seed=42,
    )

    assert res["p_value"] == 1.0
    assert res["is_significant"] is False
    assert res["winner"] == "tie"


def test_mcp_tool_aslib_permutation_test_scenario(mini_scenario_dir: Path):
    """aslib_permutation_test running in scenario algorithm comparison mode."""
    res = aslib_permutation_test(
        scenario_dir=str(mini_scenario_dir),
        algo1="algo_a",
        algo2="algo_b",
        permutations=1000,
        seed=42,
    )

    assert 0.0 <= res["p_value"] <= 1.0
    assert res["candidate_1"]["name"] == "algo_a"
    assert res["candidate_2"]["name"] == "algo_b"


def test_mcp_tool_aslib_evaluate_selectors(mini_scenario_dir: Path, tmp_path: Path):
    """aslib_evaluate_selectors evaluates algorithm predictions against scenario."""
    pred_path = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "my_selector": ["algo_a", "algo_b", "algo_c", "algo_b", "algo_a"],
        },
        index=["inst_1", "inst_2", "inst_3", "inst_4", "inst_5"],
    ).to_csv(pred_path)

    out_dir = tmp_path / "eval_tool_out"
    res = aslib_evaluate_selectors(
        scenario_dir=str(mini_scenario_dir),
        predictions_csv_path=str(pred_path),
        par_factor=10.0,
        include_feature_costs=False,
        run_significance_tests=False,
        output_dir=str(out_dir),
    )

    assert res["num_instances"] == 5
    assert res["top_selector"] == "my_selector"
    assert len(res["artifacts"]) >= 2
