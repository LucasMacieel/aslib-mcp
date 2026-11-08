"""ASlib loading adapters that preserve identifiers and scientific information."""

from collections import Counter
from pathlib import Path

import arff
import numpy as np
import pandas as pd
from aslib_scenario.aslib_scenario import ASlibScenario as UpstreamScenario
from sklearn.model_selection import KFold

from src.tools.csv_tables import read_matrix_csv
from src.tools.cv import validate_cv_arff


def generate_cv(instances: list, n_folds: int, seed: int | None) -> pd.DataFrame:
    """Generate balanced folds without changing global random state."""
    assignments = pd.DataFrame(
        {"fold": np.zeros(len(instances), dtype=int)}, index=instances
    )
    for fold, (_, test) in enumerate(
        KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(instances), 1
    ):
        assignments.iloc[test, 0] = fold
    return assignments


def read_cv_csv(path: str | Path, instances: list) -> pd.DataFrame:
    """Read complete positive integer CV assignments with string identifiers."""
    assignments = read_matrix_csv(path)
    if set(assignments.index) != set(instances) or list(assignments.columns) != [
        "fold"
    ]:
        raise ValueError(
            "CV CSV must contain all scenario instances exactly once and one fold column"
        )
    folds = assignments["fold"]
    if (
        not np.isfinite(folds).all()
        or (folds < 1).any()
        or (folds != np.floor(folds)).any()
        or folds.nunique() < 2
    ):
        raise ValueError("CV folds must be positive integers with at least two folds")
    return assignments.loc[instances]


class ASlibScenario(UpstreamScenario):
    """Adapt upstream parsing while retaining its numerical scoring conventions."""

    def __init__(self) -> None:
        super().__init__()
        self.cv_data_all: dict[int, pd.DataFrame] = {}

    def read_algorithm_runs(self, fn: str) -> None:
        """Reject invalid raw runtimes before medians can conceal them."""
        with Path(fn).open() as stream:
            data = arff.load(stream)
        for index, kind in enumerate(self.performance_type):
            if kind != "runtime":
                continue
            # Upstream accepts metric headers case-insensitively and requires
            # their declared order. Validate those same positions.
            column = 3 + index
            for row in data["data"]:
                value = row[column]
                if (
                    not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0
                ):
                    raise ValueError("Runtime measurements must be finite and non-negative")
        super().read_algorithm_runs(fn)

    def read_cv(self, fn: str) -> None:
        """Preserve all CV repetitions and activate repetition one by default."""
        with Path(fn).open() as stream:
            data = arff.load(stream)
        self.cv_data_all = validate_cv_arff(data, self.instances or [])
        self.cv_data = self.cv_data_all[1]

    def cv_assignments(
        self,
        repetition: int = 1,
        cv_csv_path: str | Path | None = None,
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, list[int]]:
        """Choose stored or external folds; generate only when no CV exists."""
        if repetition < 1 or int(repetition) != repetition:
            raise ValueError("CV repetition must be a positive integer")
        if cv_csv_path is not None:
            if repetition != 1:
                raise ValueError("CV CSV supports only repetition 1")
            return read_cv_csv(cv_csv_path, self.instances or []), [1]
        if self.cv_data_all:
            if repetition not in self.cv_data_all:
                raise ValueError(f"Unknown CV repetition {repetition}; available: {list(self.cv_data_all)}")
            return self.cv_data_all[repetition], list(self.cv_data_all)
        if repetition != 1:
            raise ValueError("Only CV repetition 1 is available")
        if self.cv_data is not None:
            return self.cv_data, [1]
        instances = self.instances or []
        if len(instances) < 2:
            raise ValueError("Cross-validation requires at least two instances")
        return generate_cv(instances, min(10, len(instances)), seed), [1]

    def read_scenario(self, dn: str) -> None:
        """Load a scenario, converting upstream process exits into tool errors."""
        try:
            super().read_scenario(dn)
        except SystemExit as exc:
            raise ValueError(
                f"Scenario loading failed for {dn} (upstream validation code {exc.code}); "
                "inspect aslib_lint_spec diagnostics"
            ) from exc

    def check_data(self) -> None:
        """Require exact matrix alignment and report upstream errors recoverably."""
        instances = set(self.instances or [])
        for kind, data in zip(self.performance_type, self.performance_data_all):
            if kind == "runtime" and (
                not np.isfinite(data.to_numpy(dtype=float)).all()
                or (data < 0).any().any()
            ):
                raise ValueError("Runtime measurements must be finite and non-negative")
        tables = [
            self.feature_data,
            self.feature_cost_data,
            self.feature_runstatus_data,
            self.runstatus_data,
            self.ground_truth_data,
            self.cv_data,
            *self.performance_data_all,
        ]
        for table in tables:
            if table is not None and (
                set(table.index) != instances or table.index.has_duplicates
            ):
                raise ValueError(
                    "Scenario data matrices must contain the same instances exactly once"
                )
        try:
            super().check_data()
        except SystemExit as exc:
            raise ValueError(
                f"Scenario validation failed (upstream code {exc.code})"
            ) from exc

    def read_feature_costs(self, fn: str) -> None:
        """Aggregate costs by median, preserving any unknown repetition as unknown."""
        with Path(fn).open() as stream:
            data = arff.load(stream)
        names = [name for name, _ in data["attributes"]]
        if names[:2] != ["instance_id", "repetition"] or set(names[2:]) != set(
            self.feature_steps
        ):
            raise ValueError(
                "Feature cost schema must contain instance_id, repetition and all feature steps"
            )
        frame = pd.DataFrame(data["data"], columns=names)
        costs = frame.set_index("instance_id").drop(columns="repetition").astype(float)
        if np.isinf(costs.to_numpy()).any() or (costs < 0).any().any():
            raise ValueError("Feature costs must be non-negative and finite or missing")
        # skipna=False prevents a known repetition from concealing an unknown one.
        self.feature_cost_data = costs.groupby(level=0).agg(
            lambda values: values.median(skipna=False)
        )

    def read_feature_values(self, fn: str) -> None:
        """Aggregate numeric features by mean and categorical features by mode.

        Missing categories are ignored; modal ties use the first observed category,
        matching upstream runstatus tie breaking. All-missing groups stay missing.
        """
        with Path(fn).open() as stream:
            data = arff.load(stream)
        attributes = data["attributes"]
        names = [name for name, _ in attributes]
        if names[:2] != ["instance_id", "repetition"]:
            raise ValueError(
                "Feature schema must begin with instance_id and repetition"
            )
        self.features = names[2:]
        declared = set(self.features_deterministic or []) | set(
            self.features_stochastic or []
        )
        if set(self.features) != declared:
            raise ValueError(
                "Feature columns must match declared deterministic and stochastic features"
            )
        frame = pd.DataFrame(data["data"], columns=names).set_index("instance_id")
        numeric = [
            name
            for name, kind in attributes[2:]
            if kind in ("NUMERIC", "REAL", "INTEGER")
        ]
        categorical = [name for name in self.features if name not in numeric]
        grouped = frame.groupby(level=0, sort=True)
        result = grouped[numeric].mean()
        for name in categorical:
            result[name] = grouped[name].agg(
                lambda values: (
                    Counter(values.dropna()).most_common(1)[0][0]
                    if values.notna().any()
                    else None
                )
            )
        self.feature_data = result[self.features]

    def read_from_csv(
        self,
        perf_fn: str,
        feat_fn: str,
        objective: str,
        runtime_cutoff: float,
        maximize: bool,
        cv_fn: str | None = None,
    ) -> None:
        """Construct a checked scenario from CSV without coercing string IDs."""
        if objective not in ("runtime", "solution_quality"):
            raise ValueError("Invalid objective")
        if not np.isfinite(runtime_cutoff) or (
            objective == "runtime" and runtime_cutoff <= 0
        ):
            raise ValueError("Runtime cutoff must be finite and positive")
        if objective == "runtime" and maximize:
            raise ValueError("Maximizing runtime is not supported")
        performance = read_matrix_csv(perf_fn)
        features = read_matrix_csv(feat_fn)
        if set(performance.index) != set(features.index):
            raise ValueError(
                "Performance and feature CSVs must contain the same instances"
            )
        features = features.loc[performance.index]
        for table in (performance, features):
            if np.isinf(table.to_numpy(dtype=float)).any():
                raise ValueError("CSV values must be finite or missing")
        if objective == "solution_quality" and performance.isna().any().any():
            raise ValueError(
                "Missing solution quality requires domain-specific imputation"
            )
        if objective == "runtime" and (performance < 0).any().any():
            raise ValueError("Runtime values must be non-negative")
        self.scenario = None
        self.performance_measure = ["dummy"]
        self.performance_type = [objective]
        self.maximize = [maximize]
        self.algorithm_cutoff_time = runtime_cutoff
        self.algorithms = list(performance.columns)
        self.instances = list(performance.index)
        self.features = list(features.columns)
        self.features_deterministic = self.features.copy()
        self.features_stochastic = []
        self.feature_group_dict = {"all": {"provides": self.features.copy()}}
        self.feature_steps = ["all"]
        self.feature_steps_default = ["all"]
        self.feature_data = features
        self.runstatus_data = pd.DataFrame(
            "ok", index=performance.index, columns=performance.columns
        )
        if objective == "runtime":
            self.runstatus_data[performance >= runtime_cutoff] = "timeout"
            self.runstatus_data[performance.isna()] = "crash"
            performance = performance.fillna(runtime_cutoff)
        self.performance_data = performance
        self.performance_data_all = [performance]
        self.feature_runstatus_data = pd.DataFrame(
            "ok", index=features.index, columns=["all"]
        )
        self.cv_data = (
            read_cv_csv(cv_fn, self.instances)
            if cv_fn
            else generate_cv(self.instances, min(10, len(self.instances)), None)
            if len(self.instances) >= 2
            else None
        )
        if self.CHECK_VALID:
            self.check_data()
