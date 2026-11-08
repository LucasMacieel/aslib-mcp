"""ASlib scenario exporter.

Generates an ASlib scenario directory containing ARFF files,
YAML description, and optional cross-validation splits from tabular CSV benchmarks.
Modernized and expanded from coseal/aslib-spec (scripts/generate_scenario.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import arff
import numpy as np
import pandas as pd
import yaml

from src.tools.csv_tables import read_matrix_csv


def export_aslib_scenario(
    perf_csv_path: str | Path,
    feat_csv_path: str | Path,
    scenario_id: str,
    cutoff: float,
    output_dir: str | Path,
    objective: Literal["runtime", "solution_quality"] = "runtime",
    maximize: bool = False,
    feature_costs_csv_path: str | Path | None = None,
    generate_cv: bool = True,
    n_folds: int = 10,
    seed: int | None = 1234,
    algorithm_metadata: dict[str, dict[str, Any]] | None = None,
    feature_runstatus_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Exports performance and feature CSV tables into a spec-compliant ASlib scenario directory.

    Args:
        perf_csv_path: Path to algorithm performance CSV (rows: instances, cols: algorithms).
        feat_csv_path: Path to instance feature CSV (rows: instances, cols: features).
        scenario_id: Unique scenario identifier name.
        cutoff: Execution budget cutoff threshold.
        output_dir: Destination directory where the ASlib scenario files will be created.
        objective: 'runtime' or 'solution_quality'.
        maximize: Whether higher performance indicates better performance.
        feature_costs_csv_path: Optional CSV of feature extraction costs.
        generate_cv: Whether to automatically generate cv.arff fold assignments.
        n_folds: Number of CV folds to generate.
        seed: Random seed for CV generation.
        feature_runstatus_csv_path: Optional instance-by-ALL matrix of ASlib statuses.

    Returns:
        Summary dict containing generated file paths, instance counts, and metadata.
    """
    perf_file = Path(perf_csv_path).resolve()
    feat_file = Path(feat_csv_path).resolve()
    target_dir = Path(output_dir).resolve()

    if not perf_file.is_file():
        raise FileNotFoundError(f"Performance CSV not found: {perf_file}")
    if not feat_file.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {feat_file}")

    if target_dir.exists() and (not target_dir.is_dir() or any(target_dir.iterdir())):
        raise ValueError("Export destination must be an empty directory")

    # Load data
    perf_df = read_matrix_csv(perf_file)
    feat_df = read_matrix_csv(feat_file)

    # Clean index strings
    perf_df.index = perf_df.index.astype(str).str.strip()
    feat_df.index = feat_df.index.astype(str).str.strip()
    perf_df.columns = perf_df.columns.astype(str).str.strip()
    feat_df.columns = feat_df.columns.astype(str).str.strip()
    for table in (perf_df, feat_df):
        if table.index.has_duplicates or table.columns.has_duplicates:
            raise ValueError("CSV instance IDs and column names must be unique")
        if table.empty or any(not name for name in table.columns):
            raise ValueError("CSV tables must contain instances and named columns")
    if objective not in ("runtime", "solution_quality"):
        raise ValueError("Invalid objective")
    if not np.isfinite(cutoff) or (objective == "runtime" and cutoff <= 0):
        raise ValueError("Runtime cutoff must be finite and positive")
    if objective == "runtime" and maximize:
        raise ValueError("Maximizing runtime is not supported")

    # Align common instances
    common_instances = sorted(set(perf_df.index).intersection(feat_df.index))
    if not common_instances:
        raise ValueError(
            "No common instances found between performance and feature CSV files."
        )
    if set(perf_df.index) != set(feat_df.index):
        raise ValueError("Performance and feature CSVs must contain the same instances")
    if objective == "solution_quality" and perf_df.isna().any().any():
        raise ValueError("Missing solution quality requires domain-specific imputation")
    for table in (perf_df, feat_df):
        values = table.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("CSV values must be finite or missing")
    if objective == "runtime" and (perf_df < 0).any().any():
        raise ValueError("Runtime values must be non-negative")
    if generate_cv and (n_folds < 2 or len(common_instances) < 2):
        raise ValueError("Cross-validation requires at least two instances and folds")

    perf_df = perf_df.loc[common_instances]
    feat_df = feat_df.loc[common_instances]

    algorithms = [str(c).strip() for c in perf_df.columns]
    features = [str(c).strip() for c in feat_df.columns]
    metadata = algorithm_metadata or {
        algo: {"configuration": "", "deterministic": True} for algo in algorithms
    }
    if set(metadata) != set(algorithms) or any(
        not isinstance(cfg, dict)
        or not isinstance(cfg.get("deterministic"), bool)
        or not isinstance(cfg.get("configuration"), str)
        for cfg in metadata.values()
    ):
        raise ValueError(
            "Algorithm metadata must define configuration and deterministic for each algorithm"
        )
    costs_df = None
    if feature_costs_csv_path is not None:
        c_path = Path(feature_costs_csv_path).resolve()
        if not c_path.is_file():
            raise FileNotFoundError(f"Feature costs CSV not found: {c_path}")
        costs_df = read_matrix_csv(c_path)
        costs_df.index = costs_df.index.astype(str).str.strip()
        if costs_df.index.has_duplicates or set(costs_df.index) != set(
            common_instances
        ):
            raise ValueError(
                "Feature costs must contain the same instances exactly once"
            )
        cost_values = costs_df.to_numpy(dtype=float)
        if np.isinf(cost_values).any() or (cost_values < 0).any():
            raise ValueError("Feature costs must be non-negative and finite or missing")
    statuses = None
    allowed_statuses = {
        "ok", "timeout", "memout", "presolved", "crash", "unknown", "other"
    }
    if feature_runstatus_csv_path is not None:
        statuses = read_matrix_csv(feature_runstatus_csv_path, strings=True)
        if (
            set(statuses.index) != set(common_instances)
            or list(statuses.columns) != ["ALL"]
        ):
            raise ValueError("Feature statuses must contain all instances and one ALL column")
        if not statuses["ALL"].isin(allowed_statuses).all():
            raise ValueError("Invalid feature runstatus")
    assumptions = [
        "All features belong to one ALL step",
        "Features are declared deterministic",
    ]
    if statuses is None:
        assumptions.append("Feature status is unknown for missing values, otherwise assumed ok")
    if algorithm_metadata is None:
        assumptions.append("Algorithms are assumed deterministic with empty configurations")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write description.txt (YAML)
    description = {
        "scenario_id": str(scenario_id),
        "performance_measures": [objective],
        "maximize": [bool(maximize)],
        "performance_type": [objective],
        "algorithm_cutoff_time": float(cutoff) if objective == "runtime" else None,
        "algorithm_cutoff_memory": None,
        "features_cutoff_time": float(cutoff) if objective == "runtime" else None,
        "features_cutoff_memory": None,
        "algorithms_deterministic": [
            a for a in algorithms if metadata[a]["deterministic"]
        ],
        "algorithms_stochastic": [
            a for a in algorithms if not metadata[a]["deterministic"]
        ],
        "features_deterministic": features,
        "features_stochastic": [],
        "number_of_feature_steps": 1,
        "feature_steps": {"ALL": {"provides": features, "requires": []}},
        "default_steps": ["ALL"],
        "metainfo_algorithms": metadata,
    }

    desc_path = target_dir / "description.txt"
    with open(desc_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(description, f, sort_keys=False)

    # 2. Write algorithm_runs.arff
    run_attributes = [
        ("instance_id", "STRING"),
        ("repetition", "NUMERIC"),
        ("algorithm", "STRING"),
        (objective, "NUMERIC"),
        ("runstatus", ["ok", "timeout", "memout", "not_applicable", "crash", "other"]),
    ]

    run_data = []
    for inst in common_instances:
        row = perf_df.loc[inst]
        for algo in algorithms:
            val = row[algo]
            if pd.isna(val):
                status = "crash"
                score = float(cutoff)
            else:
                score = float(val)
                if objective == "runtime":
                    status = "ok" if score < cutoff else "timeout"
                    if score >= cutoff:
                        score = float(cutoff)
                else:
                    status = "ok"
            run_data.append([inst, 1, algo, score, status])

    algo_runs_dict = {
        "relation": f"ALGORITHM_RUNS_{scenario_id}",
        "attributes": run_attributes,
        "data": run_data,
    }
    runs_path = target_dir / "algorithm_runs.arff"
    with open(runs_path, "w", encoding="utf-8") as f:
        arff.dump(algo_runs_dict, f)

    # 3. Write feature_values.arff
    feat_attributes = [("instance_id", "STRING"), ("repetition", "NUMERIC")]
    for f in features:
        feat_attributes.append((f, "NUMERIC"))

    feat_data = []
    inst_has_timeout_feature = {}
    for inst in common_instances:
        row = feat_df.loc[inst]
        vals = []
        has_na = False
        for f in features:
            v = row[f]
            if pd.isna(v):
                vals.append(None)
                has_na = True
            else:
                vals.append(float(v))
        inst_has_timeout_feature[inst] = has_na
        feat_data.append([inst, 1] + vals)

    feat_vals_dict = {
        "relation": f"FEATURE_VALUES_{scenario_id}",
        "attributes": feat_attributes,
        "data": feat_data,
    }
    feat_vals_path = target_dir / "feature_values.arff"
    with open(feat_vals_path, "w", encoding="utf-8") as f:
        arff.dump(feat_vals_dict, f)

    # 4. Write feature_runstatus.arff
    fs_attributes = [
        ("instance_id", "STRING"),
        ("repetition", "NUMERIC"),
        ("ALL", ["ok", "timeout", "memout", "presolved", "crash", "unknown", "other"]),
    ]
    fs_data = [
        [
            inst,
            1,
            str(statuses.loc[inst, "ALL"])
            if statuses is not None
            else "unknown" if inst_has_timeout_feature[inst] else "ok",
        ]
        for inst in common_instances
    ]
    fs_dict = {
        "relation": f"FEATURE_RUNSTATUS_{scenario_id}",
        "attributes": fs_attributes,
        "data": fs_data,
    }
    fs_path = target_dir / "feature_runstatus.arff"
    with open(fs_path, "w", encoding="utf-8") as f:
        arff.dump(fs_dict, f)

    # 5. Optional feature_costs.arff
    costs_path = None
    if costs_df is not None:
        costs_attributes = [
            ("instance_id", "STRING"),
            ("repetition", "NUMERIC"),
            ("ALL", "NUMERIC"),
        ]
        costs_data = []
        for inst in common_instances:
            c_val = (
                None
                if costs_df.loc[inst].isna().any()
                else float(costs_df.loc[inst].sum())
            )
            costs_data.append([inst, 1, c_val])
        costs_dict = {
            "relation": f"FEATURE_COSTS_{scenario_id}",
            "attributes": costs_attributes,
            "data": costs_data,
        }
        costs_path = target_dir / "feature_costs.arff"
        with open(costs_path, "w", encoding="utf-8") as f:
            arff.dump(costs_dict, f)

    # 6. Optional cv.arff
    cv_path = None
    if generate_cv:
        rng = np.random.default_rng(seed)
        shuffled = common_instances.copy()
        rng.shuffle(shuffled)
        n_folds = max(2, min(n_folds, len(shuffled)))
        fold_assignments = {}
        for idx, inst in enumerate(shuffled):
            fold_assignments[inst] = (idx % n_folds) + 1

        cv_attributes = [
            ("instance_id", "STRING"),
            ("repetition", "NUMERIC"),
            ("fold", "NUMERIC"),
        ]
        cv_data = [[inst, 1, int(fold_assignments[inst])] for inst in common_instances]
        cv_dict = {
            "relation": f"CV_{scenario_id}",
            "attributes": cv_attributes,
            "data": cv_data,
        }
        cv_path = target_dir / "cv.arff"
        with open(cv_path, "w", encoding="utf-8") as f:
            arff.dump(cv_dict, f)

    created_files = [
        str(desc_path),
        str(runs_path),
        str(feat_vals_path),
        str(fs_path),
    ]
    if costs_path:
        created_files.append(str(costs_path))
    if cv_path:
        created_files.append(str(cv_path))

    return {
        "message": f"Successfully exported ASlib scenario '{scenario_id}' with {len(common_instances)} instances and {len(algorithms)} algorithms.",
        "scenario_id": scenario_id,
        "output_dir": str(target_dir),
        "created_files": created_files,
        "num_instances": len(common_instances),
        "algorithms": algorithms,
        "features": features,
        "cutoff": float(cutoff),
        "objective": objective,
        "has_cv": generate_cv,
        "assumptions": assumptions,
    }
