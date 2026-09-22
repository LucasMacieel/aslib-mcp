"""Tools extracted from repo/ASlibScenario/aslib_scenario/aslib_scenario.py."""

import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

_this_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _this_dir:
    sys.path.pop(0)

_pkg_root = str(Path(__file__).resolve().parent.parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import numpy as np
import pandas as pd
from aslib_scenario.aslib_scenario import ASlibScenario
from fastmcp import FastMCP

from src.tools.permutation_tester import PermutationTester, permutation_test_scenario
from src.tools.scenario_exporter import export_aslib_scenario
from src.tools.scenario_repository import (
    fetch_scenario,
    get_scenario_info,
    list_scenarios,
    resolve_scenario_dir,
    sync_repository,
)
from src.tools.selector_evaluator import SelectorEvaluator
from src.tools.spec_checker import lint_scenario_spec

REFERENCE = (
    "https://github.com/mlindauer/ASlibScenario/blob/"
    "c785d4ae524dd0f4624d4092eea62a05ee2830d4/aslib_scenario/aslib_scenario.py"
)
ASLIB_SPEC_REFERENCE = "https://github.com/coseal/aslib-spec/tree/master"
ASLIB_DATA_REFERENCE = "https://github.com/coseal/aslib_data"

aslib_scenario_mcp = FastMCP(name="aslib_scenario")


def _get_output_dir(output_dir: str | None, prefix: str = "aslib") -> Path:
    """Create a unique collision-resistant output directory."""
    unique_id = (
        f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    if output_dir is not None:
        target = Path(output_dir).resolve() / unique_id
    else:
        target = Path(tempfile.gettempdir()).resolve() / unique_id
    target.mkdir(parents=True, exist_ok=True)
    return target


@aslib_scenario_mcp.tool()
def aslib_list_scenarios(
    domain: Annotated[
        str | None,
        "Optional problem domain filter (e.g. 'SAT', 'CSP', 'QBF', 'MAXSAT', 'MIP', 'TSP', 'ASP')",
    ] = None,
    objective: Annotated[
        str | None,
        "Optional optimization objective filter ('runtime', 'solution_quality')",
    ] = None,
    search: Annotated[
        str | None,
        "Optional case-insensitive substring search for scenario ID or domain",
    ] = None,
    limit: Annotated[
        int,
        "Maximum number of scenarios to return (default 50)",
    ] = 50,
) -> dict:
    """Discover and filter benchmark scenarios available in the ASlib scenario repository (coseal/aslib_data)."""
    res = list_scenarios(domain=domain, objective=objective, search=search, limit=limit)
    res["reference"] = ASLIB_DATA_REFERENCE
    return res


@aslib_scenario_mcp.tool()
def aslib_get_scenario_info(
    scenario_id: Annotated[
        str,
        "Canonical Scenario ID (e.g. 'SAT12-ALL', 'ASP-POTASSCO') or local scenario directory path",
    ],
) -> dict:
    """Retrieve detailed specification metadata for an ASlib scenario without loading full ARFF data matrices."""
    res = get_scenario_info(scenario_id)
    res["reference"] = ASLIB_DATA_REFERENCE
    return res


@aslib_scenario_mcp.tool()
def aslib_fetch_scenario(
    scenario_id: Annotated[
        str,
        "Canonical Scenario ID (e.g. 'SAT12-ALL', 'ASP-POTASSCO', 'QBF-2016')",
    ],
    force_update: Annotated[
        bool,
        "Whether to force re-cloning or refreshing the scenario cache from upstream",
    ] = False,
) -> dict:
    """Ensure a scenario from coseal/aslib_data is present and valid in the local cache, returning its directory path."""
    res = fetch_scenario(scenario_id, force_update=force_update)
    res["reference"] = ASLIB_DATA_REFERENCE
    return res


@aslib_scenario_mcp.tool()
def aslib_sync_repository(
    force: Annotated[
        bool,
        "Whether to force re-clone or re-index the scenario cache",
    ] = False,
) -> dict:
    """Synchronize the local ASlib Scenario Cache with the upstream coseal/aslib_data GitHub repository."""
    res = sync_repository(force=force)
    res["reference"] = ASLIB_DATA_REFERENCE
    return res


@aslib_scenario_mcp.tool()
def aslib_read_scenario(
    scenario_dir: Annotated[
        str,
        "Path to the ASlib scenario directory containing description.txt and ARFF files, or canonical Scenario ID (e.g. 'SAT12-ALL', 'ASP-POTASSCO')",
    ],
    output_dir: Annotated[
        str | None,
        "Base directory for output artifacts; a unique subdirectory is created",
    ] = None,
) -> dict:
    """Read and parse an ASlib benchmark scenario directory into tabular data structures."""
    scen_path = resolve_scenario_dir(scenario_dir)

    required_files = [
        "description.txt",
        "algorithm_runs.arff",
        "feature_values.arff",
        "feature_runstatus.arff",
    ]
    missing = [f for f in required_files if not (scen_path / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required ASlib files in {scen_path}: {missing}"
        )

    scenario = ASlibScenario()
    scenario.read_scenario(str(scen_path))

    out_dir = _get_output_dir(output_dir, prefix="read_scenario")
    artifacts = []

    if scenario.performance_data is not None:
        p = out_dir / "performance_data.csv"
        scenario.performance_data.to_csv(p)
        artifacts.append(
            {"description": "Active performance data matrix", "path": str(p.resolve())}
        )

    if scenario.feature_data is not None:
        p = out_dir / "feature_data.csv"
        scenario.feature_data.to_csv(p)
        artifacts.append(
            {"description": "Instance feature data matrix", "path": str(p.resolve())}
        )

    if scenario.runstatus_data is not None:
        p = out_dir / "runstatus_data.csv"
        scenario.runstatus_data.to_csv(p)
        artifacts.append(
            {"description": "Algorithm runstatus matrix", "path": str(p.resolve())}
        )

    if scenario.feature_runstatus_data is not None:
        p = out_dir / "feature_runstatus_data.csv"
        scenario.feature_runstatus_data.to_csv(p)
        artifacts.append(
            {
                "description": "Feature calculation runstatus matrix",
                "path": str(p.resolve()),
            }
        )

    if scenario.cv_data is not None:
        p = out_dir / "cv_data.csv"
        scenario.cv_data.to_csv(p)
        artifacts.append(
            {
                "description": "Cross-validation fold assignments table",
                "path": str(p.resolve()),
            }
        )

    if scenario.feature_cost_data is not None:
        p = out_dir / "feature_cost_data.csv"
        scenario.feature_cost_data.to_csv(p)
        artifacts.append(
            {"description": "Feature computation costs table", "path": str(p.resolve())}
        )

    if scenario.ground_truth_data is not None:
        p = out_dir / "ground_truth_data.csv"
        scenario.ground_truth_data.to_csv(p)
        artifacts.append(
            {"description": "Ground truth data table", "path": str(p.resolve())}
        )

    return {
        "message": f"Successfully loaded ASlib scenario '{scenario.scenario}' with {len(scenario.instances)} instances.",
        "reference": REFERENCE,
        "artifacts": artifacts,
        "scenario_id": scenario.scenario,
        "performance_measures": list(scenario.performance_measure),
        "performance_type": list(scenario.performance_type),
        "maximize": [bool(m) for m in scenario.maximize],
        "algorithm_cutoff_time": float(scenario.algorithm_cutoff_time)
        if scenario.algorithm_cutoff_time is not None
        else None,
        "features_cutoff_time": float(scenario.features_cutoff_time)
        if scenario.features_cutoff_time is not None
        else None,
        "algorithms": list(scenario.algorithms),
        "features": list(scenario.features),
        "num_instances": len(scenario.instances),
        "feature_steps": list(scenario.feature_steps),
    }


@aslib_scenario_mcp.tool()
def aslib_read_csv(
    perf_csv_path: Annotated[
        str,
        "Path to algorithm performance CSV file where rows are instances and columns are algorithms",
    ],
    feat_csv_path: Annotated[
        str,
        "Path to instance feature CSV file where rows are instances and columns are features",
    ],
    objective: Annotated[
        Literal["runtime", "solution_quality"], "Scenario optimization objective"
    ] = "runtime",
    runtime_cutoff: Annotated[
        float, "Runtime cutoff threshold for timing out algorithm runs"
    ] = 1000.0,
    maximize: Annotated[
        bool, "Whether higher performance values indicate better algorithm performance"
    ] = False,
    cv_csv_path: Annotated[
        str | None, "Optional path to cross-validation fold assignments CSV file"
    ] = None,
    output_dir: Annotated[
        str | None,
        "Base directory for output artifacts; a unique subdirectory is created",
    ] = None,
) -> dict:
    """Create an ASlib scenario from tabular performance and feature CSV files."""
    perf_path = Path(perf_csv_path).resolve()
    if not perf_path.is_file():
        raise FileNotFoundError(f"Performance CSV file not found: {perf_path}")

    feat_path = Path(feat_csv_path).resolve()
    if not feat_path.is_file():
        raise FileNotFoundError(f"Feature CSV file not found: {feat_path}")

    cv_path_resolved = None
    if cv_csv_path is not None:
        cv_path_resolved = Path(cv_csv_path).resolve()
        if not cv_path_resolved.is_file():
            raise FileNotFoundError(
                f"Cross-validation CSV file not found: {cv_path_resolved}"
            )

    scenario = ASlibScenario()
    scenario.read_from_csv(
        perf_fn=str(perf_path),
        feat_fn=str(feat_path),
        objective=objective,
        runtime_cutoff=float(runtime_cutoff),
        maximize=bool(maximize),
        cv_fn=str(cv_path_resolved) if cv_path_resolved else None,
    )

    out_dir = _get_output_dir(output_dir, prefix="read_csv")
    artifacts = []

    if scenario.performance_data is not None:
        p = out_dir / "performance_data.csv"
        scenario.performance_data.to_csv(p)
        artifacts.append(
            {"description": "Active performance data matrix", "path": str(p.resolve())}
        )

    if scenario.feature_data is not None:
        p = out_dir / "feature_data.csv"
        scenario.feature_data.to_csv(p)
        artifacts.append(
            {"description": "Instance feature data matrix", "path": str(p.resolve())}
        )

    if scenario.runstatus_data is not None:
        p = out_dir / "runstatus_data.csv"
        scenario.runstatus_data.to_csv(p)
        artifacts.append(
            {"description": "Algorithm runstatus matrix", "path": str(p.resolve())}
        )

    if scenario.feature_runstatus_data is not None:
        p = out_dir / "feature_runstatus_data.csv"
        scenario.feature_runstatus_data.to_csv(p)
        artifacts.append(
            {
                "description": "Feature calculation runstatus matrix",
                "path": str(p.resolve()),
            }
        )

    if scenario.cv_data is not None:
        p = out_dir / "cv_data.csv"
        scenario.cv_data.to_csv(p)
        artifacts.append(
            {
                "description": "Cross-validation fold assignments table",
                "path": str(p.resolve()),
            }
        )

    num_timeouts = (
        int((scenario.runstatus_data == "timeout").sum().sum())
        if scenario.runstatus_data is not None
        else 0
    )

    return {
        "message": f"Successfully loaded CSV scenario with {len(scenario.instances)} instances and {len(scenario.algorithms)} algorithms.",
        "reference": REFERENCE,
        "artifacts": artifacts,
        "scenario_id": scenario.scenario,
        "algorithms": list(scenario.algorithms),
        "features": list(scenario.features),
        "num_instances": len(scenario.instances),
        "objective": objective,
        "runtime_cutoff": float(runtime_cutoff),
        "maximize": bool(maximize),
        "num_timeouts": num_timeouts,
    }


@aslib_scenario_mcp.tool()
def aslib_get_cv_split(
    scenario_dir: Annotated[
        str,
        "Path to the ASlib scenario directory containing description.txt and ARFF files, or canonical Scenario ID (e.g. 'SAT12-ALL', 'ASP-POTASSCO')",
    ],
    fold_index: Annotated[
        int, "One-based index of the cross-validation fold to extract as test set"
    ] = 1,
    output_dir: Annotated[
        str | None,
        "Base directory for output artifacts; a unique subdirectory is created",
    ] = None,
) -> dict:
    """Partition an ASlib scenario into disjoint training and test splits for a specified fold."""
    scen_path = resolve_scenario_dir(scenario_dir)

    scenario = ASlibScenario()
    scenario.read_scenario(str(scen_path))

    if scenario.cv_data is None:
        scenario.create_cv_splits()

    unique_folds = sorted(scenario.cv_data["fold"].unique().tolist())
    if float(fold_index) not in unique_folds:
        raise ValueError(
            f"Fold index {fold_index} not found in scenario CV data. Available folds: {unique_folds}"
        )

    test_scen, train_scen = scenario.get_split(indx=fold_index)

    out_dir = _get_output_dir(output_dir, prefix=f"split_fold{fold_index}")
    artifacts = []

    if test_scen.performance_data is not None:
        p = out_dir / f"test_perf_fold{fold_index}.csv"
        test_scen.performance_data.to_csv(p)
        artifacts.append(
            {
                "description": f"Test split performance matrix (fold {fold_index})",
                "path": str(p.resolve()),
            }
        )

    if train_scen.performance_data is not None:
        p = out_dir / f"train_perf_fold{fold_index}.csv"
        train_scen.performance_data.to_csv(p)
        artifacts.append(
            {
                "description": f"Train split performance matrix (fold {fold_index})",
                "path": str(p.resolve()),
            }
        )

    if test_scen.feature_data is not None:
        p = out_dir / f"test_feat_fold{fold_index}.csv"
        test_scen.feature_data.to_csv(p)
        artifacts.append(
            {
                "description": f"Test split feature matrix (fold {fold_index})",
                "path": str(p.resolve()),
            }
        )

    if train_scen.feature_data is not None:
        p = out_dir / f"train_feat_fold{fold_index}.csv"
        train_scen.feature_data.to_csv(p)
        artifacts.append(
            {
                "description": f"Train split feature matrix (fold {fold_index})",
                "path": str(p.resolve()),
            }
        )

    if test_scen.runstatus_data is not None:
        p = out_dir / f"test_runstatus_fold{fold_index}.csv"
        test_scen.runstatus_data.to_csv(p)
        artifacts.append(
            {
                "description": f"Test split runstatus matrix (fold {fold_index})",
                "path": str(p.resolve()),
            }
        )

    if train_scen.runstatus_data is not None:
        p = out_dir / f"train_runstatus_fold{fold_index}.csv"
        train_scen.runstatus_data.to_csv(p)
        artifacts.append(
            {
                "description": f"Train split runstatus matrix (fold {fold_index})",
                "path": str(p.resolve()),
            }
        )

    return {
        "message": f"Partitioned scenario into fold {fold_index}: {len(test_scen.instances)} test instances and {len(train_scen.instances)} train instances.",
        "reference": REFERENCE,
        "artifacts": artifacts,
        "fold_index": int(fold_index),
        "train_num_instances": len(train_scen.instances),
        "test_num_instances": len(test_scen.instances),
        "train_instances": list(train_scen.instances),
        "test_instances": list(test_scen.instances),
    }


@aslib_scenario_mcp.tool()
def aslib_create_cv_splits(
    scenario_dir: Annotated[
        str,
        "Path to the ASlib scenario directory containing description.txt and ARFF files, or canonical Scenario ID (e.g. 'SAT12-ALL', 'ASP-POTASSCO')",
    ],
    n_folds: Annotated[int, "Number of cross-validation folds to generate"] = 10,
    seed: Annotated[int | None, "Random seed for reproducible fold generation"] = None,
    output_dir: Annotated[
        str | None,
        "Base directory for output artifacts; a unique subdirectory is created",
    ] = None,
) -> dict:
    """Generate balanced cross-validation fold assignments for instances in an ASlib scenario."""
    scen_path = resolve_scenario_dir(scenario_dir)

    scenario = ASlibScenario()
    scenario.read_scenario(str(scen_path))

    if n_folds < 2:
        raise ValueError(f"Number of folds must be at least 2, got {n_folds}")
    if n_folds > len(scenario.instances):
        raise ValueError(
            f"Number of folds ({n_folds}) cannot exceed number of instances ({len(scenario.instances)})"
        )

    if seed is not None:
        np.random.seed(seed)

    scenario.create_cv_splits(n_folds=n_folds)

    out_dir = _get_output_dir(output_dir, prefix="create_cv")
    p = out_dir / "cv_splits.csv"
    scenario.cv_data.to_csv(p)
    artifacts = [
        {
            "description": f"Generated {n_folds}-fold cross-validation assignments table",
            "path": str(p.resolve()),
        }
    ]

    fold_counts = {
        f"fold_{int(k)}": int(v)
        for k, v in scenario.cv_data["fold"].value_counts().sort_index().items()
    }

    return {
        "message": f"Generated balanced {n_folds}-fold CV split across {len(scenario.instances)} instances.",
        "reference": REFERENCE,
        "artifacts": artifacts,
        "n_folds": int(n_folds),
        "total_instances": len(scenario.instances),
        "instances_per_fold": fold_counts,
    }


@aslib_scenario_mcp.tool()
def aslib_change_perf_measure(
    scenario_dir: Annotated[
        str,
        "Path to the ASlib scenario directory containing description.txt and ARFF files, or canonical Scenario ID (e.g. 'SAT12-ALL', 'ASP-POTASSCO')",
    ],
    measure_name: Annotated[str | None, "Target performance measure name"] = None,
    measure_idx: Annotated[
        int | None, "Zero-based index of target performance measure"
    ] = None,
    output_dir: Annotated[
        str | None,
        "Base directory for output artifacts; a unique subdirectory is created",
    ] = None,
) -> dict:
    """Switch the active performance measure in a multi-metric ASlib scenario."""
    scen_path = resolve_scenario_dir(scenario_dir)

    scenario = ASlibScenario()
    scenario.read_scenario(str(scen_path))

    if measure_name is None and measure_idx is None:
        raise ValueError("Either measure_name or measure_idx must be specified.")

    if measure_name is not None:
        if measure_name not in scenario.performance_measure:
            raise ValueError(
                f"Performance measure '{measure_name}' not found. "
                f"Available measures: {scenario.performance_measure}"
            )
        resolved_idx = scenario.performance_measure.index(measure_name)
    else:
        if measure_idx < 0 or measure_idx >= len(scenario.performance_measure):
            raise ValueError(
                f"Performance measure index {measure_idx} out of range [0, {len(scenario.performance_measure) - 1}]. "
                f"Available measures: {scenario.performance_measure}"
            )
        resolved_idx = measure_idx

    scenario.change_perf_measure(measure_idx=measure_idx, measure_name=measure_name)

    if (
        resolved_idx == 0
        or scenario.performance_data is not scenario.performance_data_all[resolved_idx]
    ):
        scenario.performance_data = scenario.performance_data_all[resolved_idx]

    active_name = scenario.performance_measure[resolved_idx]
    means = {
        algo: float(scenario.performance_data[algo].mean())
        for algo in scenario.algorithms
    }

    out_dir = _get_output_dir(output_dir, prefix=f"perf_measure_{active_name}")
    p = out_dir / f"performance_{active_name}.csv"
    scenario.performance_data.to_csv(p)
    artifacts = [
        {
            "description": f"Active performance data matrix for measure '{active_name}'",
            "path": str(p.resolve()),
        }
    ]

    return {
        "message": f"Switched active performance measure to '{active_name}' (index {resolved_idx}).",
        "reference": REFERENCE,
        "artifacts": artifacts,
        "active_measure_name": active_name,
        "active_measure_index": int(resolved_idx),
        "available_measures": list(scenario.performance_measure),
        "mean_performance_per_algorithm": means,
    }


@aslib_scenario_mcp.tool()
def aslib_validate_scenario(
    scenario_dir: Annotated[
        str,
        "Path to the ASlib scenario directory containing description.txt and ARFF files, or canonical Scenario ID (e.g. 'SAT12-ALL', 'ASP-POTASSCO')",
    ],
    output_dir: Annotated[
        str | None,
        "Base directory for output artifacts; a unique subdirectory is created",
    ] = None,
) -> dict:
    """Validate ASlib scenario integrity, align instance indices, and apply PAR10 or sign adjustments."""
    scen_path = resolve_scenario_dir(scenario_dir)

    scenario = ASlibScenario()
    scenario.CHECK_VALID = False
    scenario.read_scenario(str(scen_path))

    validation_messages = []
    for perf_type_i, perf_type in enumerate(scenario.performance_type):
        if pd.isnull(scenario.performance_data_all[perf_type_i]).sum().sum() > 0:
            raise ValueError("Performance data cannot have missing entries")
        if perf_type == "runtime" and scenario.maximize[perf_type_i]:
            raise ValueError("Maximizing runtime is not supported")

    all_data = [
        scenario.feature_data,
        scenario.feature_cost_data,
        scenario.feature_runstatus_data,
        scenario.ground_truth_data,
        scenario.cv_data,
    ] + list(scenario.performance_data_all)

    set_insts = set(scenario.instances)
    for data in all_data:
        if data is not None:
            diff = set_insts.difference(data.index)
            if diff:
                raise ValueError(
                    f"Not all data matrices have the same instances: {diff}"
                )
            if len(list(set(data.index))) != len(data.index):
                raise ValueError(
                    "Some instances are listed more than once in data matrices"
                )

    validation_messages.append("Instance alignment verified across all data matrices.")
    validation_messages.append("No missing entries found in performance data.")

    par10_count = 0
    maximization_inverted = False
    for perf_type_i, perf_type in enumerate(scenario.performance_type):
        if perf_type == "runtime":
            par10_count += int((scenario.runstatus_data != "ok").sum().sum())
        elif perf_type == "solution_quality" and scenario.maximize[perf_type_i]:
            maximization_inverted = True

    scenario.check_data()
    scenario.performance_data = scenario.performance_data_all[0]

    if par10_count > 0:
        validation_messages.append(
            f"Applied PAR10 penalty ({scenario.algorithm_cutoff_time * 10}) to {par10_count} non-OK runs."
        )
    if maximization_inverted:
        validation_messages.append(
            "Inverted maximization solution_quality scores by -1 for minimization."
        )

    out_dir = _get_output_dir(output_dir, prefix="validate_scenario")
    artifacts = []

    p_perf = out_dir / "validated_performance.csv"
    scenario.performance_data.to_csv(p_perf)
    artifacts.append(
        {
            "description": "Validated and adjusted performance matrix",
            "path": str(p_perf.resolve()),
        }
    )

    p_runstatus = out_dir / "validated_runstatus.csv"
    scenario.runstatus_data.to_csv(p_runstatus)
    artifacts.append(
        {
            "description": "Validated algorithm runstatus matrix",
            "path": str(p_runstatus.resolve()),
        }
    )

    return {
        "message": f"Successfully validated scenario '{scenario.scenario}' ({len(scenario.instances)} instances).",
        "reference": REFERENCE,
        "artifacts": artifacts,
        "is_valid": True,
        "validation_messages": validation_messages,
        "instances_aligned": True,
        "num_instances": len(scenario.instances),
        "num_algorithms": len(scenario.algorithms),
        "num_features": len(scenario.features),
        "par10_imputations_count": par10_count,
        "maximization_inverted": maximization_inverted,
    }


@aslib_scenario_mcp.tool()
def aslib_lint_spec(
    scenario_dir: Annotated[
        str,
        "Path to the ASlib scenario directory or canonical Scenario ID to validate against the official ASlib 2.0 specification",
    ],
    strict: Annotated[
        bool,
        "If True, raises a ValueError on any fatal specification errors instead of returning diagnostic report",
    ] = False,
) -> dict:
    """Validate an ASlib benchmark directory against the official format specification."""
    scen_path = resolve_scenario_dir(scenario_dir)
    report = lint_scenario_spec(scenario_dir=str(scen_path), strict=strict)
    report["reference"] = ASLIB_SPEC_REFERENCE
    scen_id = report.get("scenario_id") or "UNKNOWN"
    report["message"] = (
        f"ASlib scenario '{scen_id}' passed specification checks."
        if report["is_valid"]
        else f"ASlib scenario '{scen_id}' failed specification checks with {len(report['errors'])} error(s)."
    )
    return report


@aslib_scenario_mcp.tool()
def aslib_export_scenario(
    perf_csv_path: Annotated[
        str,
        "Path to algorithm performance CSV (rows: instances, cols: algorithms)",
    ],
    feat_csv_path: Annotated[
        str,
        "Path to instance features CSV (rows: instances, cols: features)",
    ],
    scenario_id: Annotated[str, "Identifier name for the exported scenario"],
    cutoff: Annotated[
        float, "Algorithm execution cutoff budget (time or cost threshold)"
    ],
    output_dir: Annotated[
        str | None,
        "Target directory for exported scenario files; creates unique directory if omitted",
    ] = None,
    objective: Annotated[
        Literal["runtime", "solution_quality"], "Scenario optimization objective"
    ] = "runtime",
    maximize: Annotated[
        bool, "Whether higher performance values indicate better algorithm performance"
    ] = False,
    feature_costs_csv_path: Annotated[
        str | None, "Optional CSV file with feature extraction costs per instance"
    ] = None,
    generate_cv: Annotated[
        bool, "Whether to automatically generate balanced cross-validation fold splits"
    ] = True,
    n_folds: Annotated[
        int, "Number of cross-validation folds if generate_cv is True"
    ] = 10,
    seed: Annotated[int | None, "Random seed for reproducible fold generation"] = None,
) -> dict:
    """Export tabular CSV benchmarks into a fully compliant, self-contained ASlib scenario directory."""
    if output_dir is not None:
        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = _get_output_dir(None, prefix=f"export_{scenario_id}")

    res = export_aslib_scenario(
        perf_csv_path=perf_csv_path,
        feat_csv_path=feat_csv_path,
        scenario_id=scenario_id,
        cutoff=cutoff,
        output_dir=target_dir,
        objective=objective,
        maximize=maximize,
        feature_costs_csv_path=feature_costs_csv_path,
        generate_cv=generate_cv,
        n_folds=n_folds,
        seed=seed,
    )
    res["reference"] = ASLIB_SPEC_REFERENCE
    res["artifacts"] = [
        {
            "description": f"Generated ASlib scenario file: {Path(f).name}",
            "path": str(f),
        }
        for f in res.get("created_files", [])
    ]
    return res


@aslib_scenario_mcp.tool()
def aslib_permutation_test(
    vec1: Annotated[
        list[float] | None,
        "Performance vector of first candidate (used when comparing raw score lists)",
    ] = None,
    vec2: Annotated[
        list[float] | None,
        "Performance vector of second candidate (used when comparing raw score lists)",
    ] = None,
    scenario_dir: Annotated[
        str | None,
        "Optional path to ASlib scenario directory or canonical Scenario ID to extract algorithm performance directly",
    ] = None,
    algo1: Annotated[
        str | None,
        "Name of first algorithm to compare (required when scenario_dir is specified)",
    ] = None,
    algo2: Annotated[
        str | None,
        "Name of second algorithm to compare (required when scenario_dir is specified)",
    ] = None,
    permutations: Annotated[
        int, "Number of random permutation swaps (default 10000)"
    ] = 10000,
    alpha: Annotated[float, "Significance level threshold (default 0.05)"] = 0.05,
    cutoff: Annotated[
        float | None, "Optional cutoff threshold for PAR score penalization"
    ] = None,
    par_factor: Annotated[
        float, "PAR factor multiplier for timeouts (default 10.0)"
    ] = 10.0,
    noise: Annotated[
        float | None,
        "Optional Van Gelder noise filtering coefficient to handle ties and noise",
    ] = None,
    maximize: Annotated[bool, "Whether higher performance values are better"] = False,
    seed: Annotated[int | None, "Random seed for reproducible permutations"] = None,
) -> dict:
    """Run a paired permutation significance test between two algorithms or selectors."""
    if scenario_dir is not None:
        if not algo1 or not algo2:
            raise ValueError(
                "When scenario_dir is provided, both algo1 and algo2 must be specified."
            )
        scen_path = resolve_scenario_dir(scenario_dir)
        res = permutation_test_scenario(
            scenario_dir=str(scen_path),
            algo1=algo1,
            algo2=algo2,
            permutations=permutations,
            alpha=alpha,
            noise=noise,
            seed=seed,
        )
    else:
        if vec1 is None or vec2 is None:
            raise ValueError(
                "Either (vec1, vec2) or (scenario_dir, algo1, algo2) must be provided."
            )
        tester = PermutationTester(seed=seed)
        res = tester.run_test(
            vec1=vec1,
            vec2=vec2,
            alpha=alpha,
            permutations=permutations,
            name1=algo1 or "candidate_1",
            name2=algo2 or "candidate_2",
            cutoff=cutoff,
            par_factor=par_factor,
            noise=noise,
            maximize=maximize,
        )
    res["reference"] = ASLIB_SPEC_REFERENCE
    return res


@aslib_scenario_mcp.tool()
def aslib_evaluate_selectors(
    scenario_dir: Annotated[
        str,
        "Path to the ASlib scenario directory or canonical Scenario ID against which selectors are evaluated",
    ],
    predictions_csv_path: Annotated[
        str,
        "Path to CSV with selector algorithm selections or predicted runtimes (rows: instances)",
    ],
    include_feature_costs: Annotated[
        bool,
        "Whether to add feature extraction costs from feature_costs.arff to selector runtimes",
    ] = True,
    par_factor: Annotated[
        float, "Penalty factor for timeouts or unhandled instances (default 10.0)"
    ] = 10.0,
    run_significance_tests: Annotated[
        bool,
        "Whether to run paired permutation tests comparing all models against the top selector",
    ] = True,
    output_dir: Annotated[
        str | None,
        "Base directory for output artifacts; a unique subdirectory is created if omitted",
    ] = None,
) -> dict:
    """Benchmark and rank algorithm selectors against an ASlib scenario and standard baselines."""
    out_dir = _get_output_dir(output_dir, prefix="evaluate_selectors")
    scen_path = resolve_scenario_dir(scenario_dir)
    evaluator = SelectorEvaluator(scenario_dir=str(scen_path), par_factor=par_factor)
    res = evaluator.evaluate(
        predictions_csv_path=predictions_csv_path,
        include_feature_costs=include_feature_costs,
        run_significance_tests=run_significance_tests,
        output_dir=out_dir,
    )
    res["reference"] = ASLIB_SPEC_REFERENCE
    return res


class _ToolManagerCompat:
    """Compatibility shim for legacy tool manager inspection."""

    def __init__(self, mcp: FastMCP):
        self._mcp = mcp

    def list_tools(self):
        return [
            comp
            for k, comp in self._mcp._local_provider._components.items()
            if k.startswith("tool:")
        ]


aslib_scenario_mcp._tool_manager = _ToolManagerCompat(aslib_scenario_mcp)


if __name__ == "__main__":
    aslib_scenario_mcp.run()
