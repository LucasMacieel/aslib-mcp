"""Shared pytest fixtures and synthetic ASlib scenario generator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest

from src.tools.scenario_exporter import export_aslib_scenario


@pytest.fixture
def synthetic_data(tmp_path: Path) -> dict[str, Any]:
    """Generates synthetic performance and feature DataFrames and writes CSV files."""
    data_dir = tmp_path / "synthetic_csv"
    data_dir.mkdir(parents=True, exist_ok=True)

    instances = ["inst_1", "inst_2", "inst_3", "inst_4", "inst_5"]
    algorithms = ["algo_a", "algo_b", "algo_c"]
    features = ["feat_1", "feat_2"]

    # Algorithm performance: algo_b is SBS (all solved, low runtime)
    # algo_a has 1 timeout on inst_4, algo_c has 2 timeouts on inst_4 and inst_5
    perf_df = pd.DataFrame(
        [
            [1.0, 2.0, 5.0],
            [8.0, 3.0, 7.0],
            [4.0, 6.0, 2.0],
            [10.0, 5.0, 10.0],
            [2.0, 4.0, 10.0],
        ],
        index=instances,
        columns=algorithms,
    )
    perf_csv = data_dir / "performance.csv"
    perf_df.to_csv(perf_csv)

    # Instance features
    feat_df = pd.DataFrame(
        [
            [0.1, 10.5],
            [0.5, 20.0],
            [0.9, 15.2],
            [0.2, 5.1],
            [0.8, 30.4],
        ],
        index=instances,
        columns=features,
    )
    feat_csv = data_dir / "features.csv"
    feat_df.to_csv(feat_csv)

    # Feature costs
    costs_df = pd.DataFrame(
        [
            [0.05, 0.15],
            [0.04, 0.16],
            [0.06, 0.14],
            [0.05, 0.15],
            [0.05, 0.15],
        ],
        index=instances,
        columns=features,
    )
    costs_csv = data_dir / "feature_costs.csv"
    costs_df.to_csv(costs_csv)

    return {
        "instances": instances,
        "algorithms": algorithms,
        "features": features,
        "perf_df": perf_df,
        "feat_df": feat_df,
        "costs_df": costs_df,
        "perf_csv": perf_csv,
        "feat_csv": feat_csv,
        "costs_csv": costs_csv,
        "cutoff": 10.0,
    }


@pytest.fixture
def make_scenario(
    synthetic_data: dict[str, Any], tmp_path: Path
) -> Callable[..., Path]:
    """Factory fixture to create customized synthetic ASlib scenarios in temporary directories."""

    def _create(
        scenario_id: str = "SYNTHETIC-SCENARIO",
        objective: Literal["runtime", "solution_quality"] = "runtime",
        maximize: bool = False,
        cutoff: float = 10.0,
        with_costs: bool = True,
        n_folds: int = 2,
        output_dir: Path | None = None,
    ) -> Path:
        target_dir = output_dir or (tmp_path / scenario_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        export_aslib_scenario(
            perf_csv_path=synthetic_data["perf_csv"],
            feat_csv_path=synthetic_data["feat_csv"],
            scenario_id=scenario_id,
            cutoff=cutoff,
            output_dir=target_dir,
            objective=objective,
            maximize=maximize,
            feature_costs_csv_path=synthetic_data["costs_csv"] if with_costs else None,
            generate_cv=True,
            n_folds=n_folds,
            seed=1234,
        )
        return target_dir

    return _create


@pytest.fixture
def mini_scenario_dir(make_scenario: Callable[..., Path]) -> Path:
    """Pre-built valid synthetic ASlib runtime scenario."""
    return make_scenario()
