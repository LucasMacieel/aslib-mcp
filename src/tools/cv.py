"""Validation of ASlib repeated cross-validation assignments."""

import math
from typing import Any

import pandas as pd


def validate_cv_arff(
    data: dict[str, Any], instances: list[str] | set[str]
) -> dict[int, pd.DataFrame]:
    """Return complete assignment tables for every positive CV repetition."""
    attributes = data["attributes"]
    if (
        [name for name, _ in attributes] != ["instance_id", "repetition", "fold"]
        or attributes[0][1] != "STRING"
        or any(kind not in ("NUMERIC", "REAL", "INTEGER") for _, kind in attributes[1:])
    ):
        raise ValueError(
            "CV schema must contain STRING instance_id, numeric repetition and fold"
        )
    records: dict[int, dict[str, int]] = {}
    for row in data["data"]:
        if len(row) != 3 or not isinstance(row[0], str) or not row[0]:
            raise ValueError("CV rows require instance_id, repetition and fold")
        instance, repetition, fold = row
        for name, value in (("repetition", repetition), ("fold", fold)):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 1
                or int(value) != value
            ):
                raise ValueError(f"CV {name} must be a positive integer")
        assignments = records.setdefault(int(repetition), {})
        if instance in assignments:
            raise ValueError("CV instances must occur exactly once per repetition")
        assignments[instance] = int(fold)
    if not records or sorted(records) != list(range(1, len(records) + 1)):
        raise ValueError("CV repetition numbers must start at 1 and be consecutive")
    result = {}
    for repetition, assignments in sorted(records.items()):
        if set(assignments) != set(instances):
            raise ValueError(
                f"CV repetition {repetition} must contain all instances exactly once"
            )
        result[repetition] = pd.DataFrame.from_dict(
            assignments, orient="index", columns=["fold"]
        )
    return result
