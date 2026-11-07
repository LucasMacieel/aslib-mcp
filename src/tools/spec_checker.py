"""ASlib specification checker and linter.

Modernized implementation of ASlib 2.0 format verification adapted from
coseal/aslib-spec (data_check_tool_python/src/coseal_reader.py).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import arff
import yaml


class ASlibSpecLinter:
    """Validates an ASlib benchmark scenario against the official specification."""

    def __init__(self, scenario_dir: str | Path):
        self.scenario_dir = Path(scenario_dir).resolve()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.recommendations: list[str] = []
        self.checked_files: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.instance_ids_per_file: dict[str, set[str]] = {}
        self.measurements_per_file: dict[str, set[tuple]] = {}

    def lint(self) -> dict[str, Any]:
        """Runs all specification linting checks."""
        if not self.scenario_dir.is_dir():
            return {
                "is_valid": False,
                "scenario_id": None,
                "errors": [f"Directory not found: {self.scenario_dir}"],
                "warnings": [],
                "recommendations": [],
                "checked_files": [],
            }

        self._check_file_structure()
        self._check_description()
        self._check_algorithm_runs()
        self._check_feature_values()
        self._check_feature_runstatus()
        self._check_feature_costs()
        self._check_cv()
        self._check_ground_truth()
        self._cross_validate_instances()
        for name in (
            "algorithm_feature_values.arff",
            "algorithm_feature_runstatus.arff",
            "algorithm_feature_costs.arff",
        ):
            if (self.scenario_dir / name).exists():
                self.warnings.append(
                    f"{name}: algorithm-feature validation is not implemented"
                )

        is_valid = len(self.errors) == 0
        scenario_id = self.metadata.get("scenario_id")

        return {
            "is_valid": is_valid,
            "scenario_id": scenario_id,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "checked_files": self.checked_files,
            "summary": {
                "num_errors": len(self.errors),
                "num_warnings": len(self.warnings),
                "num_checked_files": len(self.checked_files),
            },
        }

    def _check_file_structure(self) -> None:
        required = [
            "description.txt",
            "algorithm_runs.arff",
            "feature_values.arff",
            "feature_runstatus.arff",
        ]
        optional = [
            "cv.arff",
            "feature_costs.arff",
            "ground_truth.arff",
            "citation.bib",
            "algorithm_feature_values.arff",
            "algorithm_feature_runstatus.arff",
            "readme.txt",
        ]

        for req in required:
            p = self.scenario_dir / req
            if not p.is_file():
                self.errors.append(f"Missing required ASlib specification file: {req}")
            else:
                self.checked_files.append(req)

        for opt in optional:
            p = self.scenario_dir / opt
            if p.is_file():
                self.checked_files.append(opt)
            else:
                if opt == "citation.bib":
                    self.recommendations.append(
                        "Consider adding citation.bib for benchmark attribution."
                    )
                elif opt == "cv.arff":
                    self.warnings.append(
                        "Optional cv.arff not found. Pre-defined cross-validation splits are strongly recommended."
                    )

    def _check_description(self) -> None:
        desc_path = self.scenario_dir / "description.txt"
        if not desc_path.is_file():
            return

        try:
            with open(desc_path, "r", encoding="utf-8") as f:
                desc = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            self.errors.append(f"Failed to parse description.txt as valid YAML: {e}")
            return

        if not isinstance(desc, dict):
            self.errors.append(
                "description.txt must contain a top-level YAML dictionary."
            )
            return

        self.metadata = desc

        # Required fields in description.txt
        req_keys = [
            "scenario_id",
            "performance_measures",
            "performance_type",
            "maximize",
        ]
        for k in req_keys:
            if k not in desc:
                self.errors.append(f"description.txt is missing required key: '{k}'")

        if "scenario_id" in desc and not isinstance(desc["scenario_id"], (str, int)):
            self.errors.append("'scenario_id' must be a string or identifier.")

        if "performance_measures" in desc and (
            not isinstance(desc["performance_measures"], list)
            or not desc["performance_measures"]
        ):
            self.errors.append(
                "'performance_measures' must be a non-empty list of measure names."
            )

        if "performance_type" in desc:
            if not isinstance(desc["performance_type"], list):
                self.errors.append(
                    "'performance_type' must be a list ('runtime' or 'solution_quality')."
                )
            else:
                for pt in desc["performance_type"]:
                    if pt not in ("runtime", "solution_quality"):
                        self.errors.append(
                            f"Invalid performance_type: '{pt}'. Allowed: 'runtime', 'solution_quality'."
                        )

        if "maximize" in desc and (
            not isinstance(desc["maximize"], list)
            or any(not isinstance(value, bool) for value in desc["maximize"])
        ):
            self.errors.append("'maximize' must be a list of boolean values.")
        measures = desc.get("performance_measures")
        if isinstance(measures, list):
            if any(not isinstance(m, str) or not m for m in measures) or len(
                set(map(str, measures))
            ) != len(measures):
                self.errors.append(
                    "Performance measure names must be unique non-empty strings"
                )
            for key in ("maximize", "performance_type"):
                values = desc.get(key)
                if not isinstance(values, list) or len(values) != len(measures):
                    self.errors.append(
                        f"{key} must have the same length as performance_measures"
                    )
        perf_types = desc.get("performance_type")
        perf_types = perf_types if isinstance(perf_types, list) else []
        if "runtime" in perf_types:
            if (
                "algorithm_cutoff_time" not in desc
                or desc["algorithm_cutoff_time"] is None
            ):
                self.errors.append(
                    "Runtime scenario requires 'algorithm_cutoff_time' in description.txt."
                )
            else:
                try:
                    cutoff = float(desc["algorithm_cutoff_time"])
                    if not math.isfinite(cutoff) or cutoff <= 0:
                        self.errors.append(
                            f"algorithm_cutoff_time must be positive, got {cutoff}"
                        )
                except (ValueError, TypeError):
                    self.errors.append(
                        f"Invalid algorithm_cutoff_time: {desc.get('algorithm_cutoff_time')}"
                    )
        algorithms = desc.get("metainfo_algorithms")
        if not isinstance(algorithms, dict) or not algorithms:
            self.errors.append("metainfo_algorithms must define algorithm metadata")
        else:
            for name, cfg in algorithms.items():
                if (
                    not isinstance(cfg, dict)
                    or not isinstance(cfg.get("configuration"), str)
                    or not isinstance(cfg.get("deterministic"), bool)
                ):
                    self.errors.append(
                        f"Algorithm '{name}' requires configuration and deterministic metadata"
                    )

        if "features_cutoff_time" not in desc:
            self.warnings.append(
                "'features_cutoff_time' not specified in description.txt."
            )

        # Check feature steps definition
        feature_steps = desc.get("feature_steps")
        if not feature_steps or not isinstance(feature_steps, dict):
            self.errors.append(
                "description.txt must define 'feature_steps' as a dictionary of step specifications."
            )
        else:
            default_steps = desc.get("default_steps", [])
            if (
                not isinstance(default_steps, list)
                or not default_steps
                or any(not isinstance(step, str) for step in default_steps)
            ):
                self.errors.append(
                    "description.txt must define a non-empty 'default_steps' list."
                )
            else:
                missing_steps = set(default_steps).difference(feature_steps.keys())
                if missing_steps:
                    self.errors.append(
                        f"default_steps references undefined feature steps: {missing_steps}"
                    )

            for step_name, step_cfg in feature_steps.items():
                if not isinstance(step_cfg, dict):
                    self.errors.append(
                        f"Feature step '{step_name}' must be a dictionary"
                    )
                    continue
                if isinstance(step_cfg, dict):
                    requires = step_cfg.get("requires")
                    if requires is not None and (
                        not isinstance(requires, list)
                        or any(not isinstance(step, str) for step in requires)
                    ):
                        self.errors.append(
                            f"Feature step '{step_name}' 'requires' must be a list, got {type(requires).__name__}"
                        )
                    elif requires:
                        unresolved_req = set(requires).difference(feature_steps.keys())
                        if unresolved_req:
                            self.errors.append(
                                f"Feature step '{step_name}' requires undefined steps: {unresolved_req}"
                            )
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(step):
                if step in visiting:
                    self.errors.append(f"Feature dependency cycle detected at '{step}'")
                    return
                if step in visited or step not in feature_steps:
                    return
                visiting.add(step)
                cfg = feature_steps[step]
                requires = cfg.get("requires", []) if isinstance(cfg, dict) else []
                if isinstance(requires, list):
                    for dependency in requires:
                        if isinstance(dependency, str):
                            visit(dependency)
                visiting.remove(step)
                visited.add(step)

            for step in feature_steps:
                visit(step)

    def _load_arff(self, filename: str) -> dict[str, Any] | None:
        p = self.scenario_dir / filename
        if not p.is_file():
            return None
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                data = arff.load(f)
            self._check_table_schema(filename, data)
            return data
        except (arff.ArffException, OSError, ValueError) as e:
            self.errors.append(f"Failed to parse {filename}: {e}")
            return None

    def _check_table_schema(self, filename: str, data: dict[str, Any]) -> None:
        attributes = data["attributes"]
        names = [a[0] for a in attributes]
        if len(names) != len(set(names)):
            self.errors.append(f"{filename}: duplicate attribute names")
        if not attributes:
            self.errors.append(f"{filename}: empty schema")
            return
        if attributes[0][1] != "STRING":
            self.errors.append(f"{filename}: identifier attribute must be STRING")
        has_repetition = filename != "ground_truth.arff"
        numeric_types = ("NUMERIC", "REAL", "INTEGER")
        if (
            has_repetition
            and len(attributes) > 1
            and attributes[1][1] not in numeric_types
        ):
            self.errors.append(f"{filename}: repetition attribute must be numeric")
        if filename == "algorithm_runs.arff":
            if len(attributes) > 2 and attributes[2][1] != "STRING":
                self.errors.append(f"{filename}: algorithm attribute must be STRING")
            if any(kind not in numeric_types for _, kind in attributes[3:-1]):
                self.errors.append(
                    f"{filename}: performance attributes must be numeric"
                )
        keys = set()
        pairs = set()
        for row in data["data"]:
            if len(row) != len(attributes):
                continue
            if row[0] is None:
                self.errors.append(f"{filename}: missing identifier")
            if has_repetition and len(row) > 1:
                rep = row[1]
                if (
                    not isinstance(rep, (int, float))
                    or not math.isfinite(rep)
                    or rep < 1
                    or int(rep) != rep
                ):
                    self.errors.append(
                        f"{filename}: repetition must be a positive integer"
                    )
                key = tuple(row[:3] if filename == "algorithm_runs.arff" else row[:2])
                pairs.add(tuple(row[:2]))
            else:
                key = (row[0],)
            if key in keys:
                self.errors.append(f"{filename}: duplicate measurement key {key}")
            keys.add(key)
        self.measurements_per_file[filename] = pairs

    def _check_algorithm_runs(self) -> None:
        data = self._load_arff("algorithm_runs.arff")
        if data is None:
            return

        attributes = data.get("attributes", [])
        if len(attributes) < 5:
            self.errors.append(
                "algorithm_runs.arff must have at least 5 attributes: instance_id, repetition, algorithm, metric(s), runstatus."
            )
            return

        attr_names = [a[0] for a in attributes]
        if attr_names[0] != "instance_id":
            self.errors.append(
                f"algorithm_runs.arff: first attribute must be 'instance_id', got '{attr_names[0]}'"
            )
        if attr_names[1] != "repetition":
            self.errors.append(
                f"algorithm_runs.arff: second attribute must be 'repetition', got '{attr_names[1]}'"
            )
        if attr_names[2] != "algorithm":
            self.errors.append(
                f"algorithm_runs.arff: third attribute must be 'algorithm', got '{attr_names[2]}'"
            )
        if attr_names[-1] != "runstatus":
            self.errors.append(
                f"algorithm_runs.arff: last attribute must be 'runstatus', got '{attr_names[-1]}'"
            )

        expected_measures = self.metadata.get("performance_measures", [])
        if expected_measures:
            actual_measures = attr_names[3:-1]
            if actual_measures != expected_measures:
                self.errors.append(
                    f"algorithm_runs.arff performance attributes {actual_measures} do not match description.txt measures {expected_measures}"
                )

        instances = set()
        algorithms = set()
        valid_statuses = {"ok", "timeout", "memout", "not_applicable", "crash", "other"}
        has_null = False

        for row in data.get("data", []):
            if len(row) != len(attributes):
                self.errors.append(
                    f"algorithm_runs.arff row has {len(row)} values, expected {len(attributes)}"
                )
                break
            inst_id = str(row[0])
            algo = str(row[2])
            status = str(row[-1]).lower()
            instances.add(inst_id)
            algorithms.add(algo)
            declared = self.metadata.get("metainfo_algorithms")
            if isinstance(declared, dict) and algo not in declared:
                self.errors.append(
                    f"Undeclared algorithm '{algo}' in algorithm_runs.arff"
                )

            if status not in valid_statuses:
                self.errors.append(
                    f"Invalid runstatus '{status}' in algorithm_runs.arff for instance '{inst_id}', algorithm '{algo}'"
                )
                break

            for val in row[3:-1]:
                if val is None or (
                    isinstance(val, (int, float)) and not math.isfinite(val)
                ):
                    has_null = True

        if has_null:
            self.errors.append(
                "algorithm_runs.arff contains forbidden null/missing or non-finite performance values."
            )

        self.instance_ids_per_file["algorithm_runs.arff"] = instances
        self.metadata["_discovered_algorithms"] = algorithms

    def _check_feature_values(self) -> None:
        data = self._load_arff("feature_values.arff")
        if data is None:
            return

        attributes = data.get("attributes", [])
        if len(attributes) < 3:
            self.errors.append(
                "feature_values.arff must have at least 3 attributes (instance_id, repetition, features...)."
            )
            return

        attr_names = [a[0] for a in attributes]
        if attr_names[0] != "instance_id":
            self.errors.append(
                f"feature_values.arff: first attribute must be 'instance_id', got '{attr_names[0]}'"
            )
        if attr_names[1] != "repetition":
            self.errors.append(
                f"feature_values.arff: second attribute must be 'repetition', got '{attr_names[1]}'"
            )

        instances = set()
        for row in data.get("data", []):
            if row:
                instances.add(str(row[0]))

        self.instance_ids_per_file["feature_values.arff"] = instances

    def _check_feature_runstatus(self) -> None:
        data = self._load_arff("feature_runstatus.arff")
        if data is None:
            return

        attributes = data.get("attributes", [])
        if len(attributes) < 3:
            self.errors.append(
                "feature_runstatus.arff must have at least 3 attributes (instance_id, repetition, step_statuses...)."
            )
            return

        attr_names = [a[0] for a in attributes]
        if attr_names[0] != "instance_id":
            self.errors.append(
                f"feature_runstatus.arff: first attribute must be 'instance_id', got '{attr_names[0]}'"
            )
        if attr_names[1] != "repetition":
            self.errors.append(
                f"feature_runstatus.arff: second attribute must be 'repetition', got '{attr_names[1]}'"
            )

        feature_steps = self.metadata.get("feature_steps", {})
        if isinstance(feature_steps, dict) and feature_steps:
            step_attrs = attr_names[2:]
            diff = set(step_attrs).symmetric_difference(feature_steps.keys())
            if diff:
                self.errors.append(
                    f"feature_runstatus.arff attributes contain steps not listed in description.txt: {diff}"
                )

        instances = set()
        for row in data.get("data", []):
            if row:
                instances.add(str(row[0]))
                if any(
                    value
                    not in {
                        "ok",
                        "timeout",
                        "memout",
                        "presolved",
                        "crash",
                        "unknown",
                        "other",
                    }
                    for value in row[2:]
                ):
                    self.errors.append("Invalid feature runstatus")

        self.instance_ids_per_file["feature_runstatus.arff"] = instances

    def _check_feature_costs(self) -> None:
        data = self._load_arff("feature_costs.arff")
        if data is None:
            return

        attributes = data.get("attributes", [])
        attr_names = [a[0] for a in attributes]
        if len(attributes) >= 1 and attr_names[0] != "instance_id":
            self.errors.append(
                f"feature_costs.arff: first attribute must be 'instance_id', got '{attr_names[0]}'"
            )

        instances = set()
        for row in data.get("data", []):
            if row:
                instances.add(str(row[0]))

        self.instance_ids_per_file["feature_costs.arff"] = instances

    def _check_cv(self) -> None:
        data = self._load_arff("cv.arff")
        if data is None:
            return

        attributes = data.get("attributes", [])
        attr_names = [a[0] for a in attributes]
        if len(attributes) < 3:
            self.errors.append(
                "cv.arff must contain at least instance_id, repetition, fold attributes."
            )
            return

        if attr_names[0] != "instance_id":
            self.errors.append(
                f"cv.arff: first attribute must be 'instance_id', got '{attr_names[0]}'"
            )

        instances = set()
        folds = set()
        for row in data.get("data", []):
            if row:
                instances.add(str(row[0]))
                if len(row) >= 3:
                    folds.add(row[2])

        self.instance_ids_per_file["cv.arff"] = instances
        if len(folds) < 2:
            self.warnings.append(
                f"cv.arff defines fewer than 2 folds ({len(folds)}). Cross-validation requires at least 2 folds."
            )

    def _check_ground_truth(self) -> None:
        data = self._load_arff("ground_truth.arff")
        if data is None:
            return

        attributes = data.get("attributes", [])
        attr_names = [a[0] for a in attributes]
        if len(attributes) >= 1 and attr_names[0] != "instance_id":
            self.errors.append(
                f"ground_truth.arff: first attribute must be 'instance_id', got '{attr_names[0]}'"
            )

        instances = set()
        for row in data.get("data", []):
            if row:
                instances.add(str(row[0]))

        self.instance_ids_per_file["ground_truth.arff"] = instances

    def _cross_validate_instances(self) -> None:
        if "algorithm_runs.arff" not in self.instance_ids_per_file:
            return

        base_instances = self.instance_ids_per_file["algorithm_runs.arff"]

        for fname in [
            "feature_values.arff",
            "feature_runstatus.arff",
            "cv.arff",
            "feature_costs.arff",
            "ground_truth.arff",
        ]:
            if fname in self.instance_ids_per_file:
                other = self.instance_ids_per_file[fname]
                missing = base_instances.difference(other)
                extra = other.difference(base_instances)
                if missing:
                    self.errors.append(
                        f"{len(missing)} instances present in algorithm_runs.arff are missing from {fname}."
                    )
                if extra:
                    self.errors.append(
                        f"{len(extra)} instances in {fname} are not present in algorithm_runs.arff."
                    )
        pairs = self.measurements_per_file.get("feature_values.arff")
        for fname in ("feature_runstatus.arff", "feature_costs.arff"):
            other = self.measurements_per_file.get(fname)
            if pairs is not None and other is not None and pairs != other:
                self.errors.append(
                    f"{fname}: instance/repetition pairs differ from feature_values.arff"
                )


def lint_scenario_spec(
    scenario_dir: str | Path, strict: bool = False
) -> dict[str, Any]:
    """Lints an ASlib benchmark directory against the official format specification.

    Args:
        scenario_dir: Path to the ASlib scenario directory.
        strict: If True, raises ValueError when any fatal errors are detected.

    Returns:
        Structured dictionary with is_valid, errors, warnings, recommendations, checked_files.
    """
    linter = ASlibSpecLinter(scenario_dir)
    report = linter.lint()
    if strict and not report["is_valid"]:
        err_msg = "\n".join(report["errors"])
        raise ValueError(
            f"ASlib specification validation failed with {len(report['errors'])} errors:\n{err_msg}"
        )
    return report
