"""ASlib specification checker and linter.

Modernized implementation of ASlib 2.0 format verification adapted from
coseal/aslib-spec (data_check_tool_python/src/coseal_reader.py).
"""

from __future__ import annotations

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

        if "maximize" in desc and not isinstance(desc["maximize"], list):
            self.errors.append("'maximize' must be a list of boolean values.")

        perf_types = desc.get("performance_type", [])
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
                    if cutoff <= 0:
                        self.errors.append(
                            f"algorithm_cutoff_time must be positive, got {cutoff}"
                        )
                except (ValueError, TypeError):
                    self.errors.append(
                        f"Invalid algorithm_cutoff_time: {desc.get('algorithm_cutoff_time')}"
                    )

        if "features_cutoff_time" not in desc:
            self.warnings.append(
                "'features_cutoff_time' not specified in description.txt."
            )

        feature_steps = desc.get("feature_steps")
        if not feature_steps or not isinstance(feature_steps, dict):
            self.errors.append(
                "description.txt must define 'feature_steps' as a dictionary of step specifications."
            )
        else:
            default_steps = desc.get("default_steps", [])
            if not isinstance(default_steps, list) or not default_steps:
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
                if isinstance(step_cfg, dict):
                    requires = step_cfg.get("requires")
                    if requires is not None and not isinstance(requires, list):
                        self.errors.append(
                            f"Feature step '{step_name}' 'requires' must be a list, got {type(requires).__name__}"
                        )
                    elif requires:
                        unresolved_req = set(requires).difference(feature_steps.keys())
                        if unresolved_req:
                            self.errors.append(
                                f"Feature step '{step_name}' requires undefined steps: {unresolved_req}"
                            )

    def _load_arff(self, filename: str) -> dict[str, Any] | None:
        p = self.scenario_dir / filename
        if not p.is_file():
            return None
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return arff.load(f)
        except (arff.ArffException, OSError, ValueError) as e:
            self.errors.append(f"Failed to parse {filename}: {e}")
            return None

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

            if status not in valid_statuses:
                self.errors.append(
                    f"Invalid runstatus '{status}' in algorithm_runs.arff for instance '{inst_id}', algorithm '{algo}'"
                )
                break

            for val in row[3:-1]:
                if val is None:
                    has_null = True

        if has_null:
            self.warnings.append(
                "algorithm_runs.arff contains null/missing performance values (imputation recommended)."
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
        if feature_steps:
            step_attrs = attr_names[2:]
            diff = set(step_attrs).difference(feature_steps.keys())
            if diff:
                self.warnings.append(
                    f"feature_runstatus.arff attributes contain steps not listed in description.txt: {diff}"
                )

        instances = set()
        for row in data.get("data", []):
            if row:
                instances.add(str(row[0]))

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

        for fname in ["feature_values.arff", "feature_runstatus.arff", "cv.arff"]:
            if fname in self.instance_ids_per_file:
                other = self.instance_ids_per_file[fname]
                missing = base_instances.difference(other)
                extra = other.difference(base_instances)
                if missing:
                    self.errors.append(
                        f"{len(missing)} instances present in algorithm_runs.arff are missing from {fname}."
                    )
                if extra:
                    self.warnings.append(
                        f"{len(extra)} instances in {fname} are not present in algorithm_runs.arff."
                    )


def lint_scenario_spec(
    scenario_dir: str | Path, strict: bool = False
) -> dict[str, Any]:
    """Lints an ASlib benchmark directory against the official format specification."""
    linter = ASlibSpecLinter(scenario_dir)
    report = linter.lint()
    if strict and not report["is_valid"]:
        err_msg = "\n".join(report["errors"])
        raise ValueError(
            f"ASlib specification validation failed with {len(report['errors'])} errors:\n{err_msg}"
        )
    return report
