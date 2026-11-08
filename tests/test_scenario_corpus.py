"""Compatibility contract for the official ASlib corpus, including known defects."""

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from src.tools.scenario_loader import ASlibScenario
from src.tools.scenario_repository import get_cache_dir
from src.tools.spec_checker import lint_scenario_spec

pytestmark = pytest.mark.integration

BASELINE_PATH = Path(__file__).parent / "fixtures" / "aslib_corpus_baseline.json"
BASELINE = json.loads(BASELINE_PATH.read_text())


def diagnostic_counts(messages: list[str]) -> dict[str, int]:
    """Count diagnostics without depending on Python's set display order."""

    def sort_set(match: re.Match[str]) -> str:
        value = ast.literal_eval(match.group())
        return "{" + ", ".join(repr(item) for item in sorted(value)) + "}"

    return dict(
        sorted(
            Counter(
                re.sub(r"\{[^{}]*\}", sort_set, message) for message in messages
            ).items()
        )
    )


@pytest.fixture(scope="module")
def corpus_dir() -> Path:
    """Use an existing corpus only; never clone or modify benchmark inputs."""
    directory = get_cache_dir()
    assert directory.is_dir(), (
        f"Missing ASlib corpus at {directory}. Set ASLIB_DATA_DIR to a local "
        f"coseal/aslib_data checkout at {BASELINE['revision']}; see README.md."
    )
    return directory


def test_corpus_scenario_inventory(corpus_dir: Path) -> None:
    actual = {
        path.name
        for path in corpus_dir.iterdir()
        if path.is_dir() and (path / "description.txt").is_file()
    }
    expected = set(BASELINE["scenarios"])
    assert actual == expected, (
        f"Corpus inventory changed: added={sorted(actual - expected)}, "
        f"missing={sorted(expected - actual)}. Use the documented revision or "
        "review upstream changes before updating the baseline."
    )


@pytest.fixture(scope="module", params=sorted(BASELINE["scenarios"]))
def corpus_scenario(
    request: pytest.FixtureRequest, corpus_dir: Path
) -> tuple[Path, dict[str, Any]]:
    path = corpus_dir / request.param
    expected = BASELINE["scenarios"][request.param]
    paths = sorted(
        entry
        for entry in path.iterdir()
        if entry.is_file()
        and (entry.suffix == ".arff" or entry.name == "description.txt")
    )
    hashes = {}
    for entry in paths:
        with entry.open("rb") as stream:
            hashes[entry.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    assert hashes == expected["sha256"], (
        f"{path.name}: benchmark inputs differ from the reviewed corpus. "
        "Use the documented revision or review data changes before rebaselining."
    )
    return path, expected


def test_corpus_scenario_loading(corpus_scenario: tuple[Path, dict[str, Any]]) -> None:
    path, expected = corpus_scenario
    scenario = ASlibScenario()
    scenario.read_scenario(str(path))
    assert scenario.performance_data is not None
    assert scenario.feature_data is not None
    assert list(scenario.performance_data.shape) == expected["performance_shape"]
    assert list(scenario.feature_data.shape) == expected["feature_shape"]
    assert list(scenario.cv_data_all) == expected["cv_repetitions"]
    # Compare with reviewed metadata, not another loader-owned collection.
    assert scenario.performance_measure == expected["performance_measures"]
    assert len(scenario.performance_data_all) == len(expected["performance_measures"])
    for table in scenario.performance_data_all:
        assert list(table.shape) == expected["performance_shape"]


def test_corpus_spec_diagnostics(corpus_scenario: tuple[Path, dict[str, Any]]) -> None:
    path, expected = corpus_scenario
    report = lint_scenario_spec(path)
    for kind in ("errors", "warnings"):
        actual = diagnostic_counts(report[kind])
        known = expected[kind]
        added = dict((Counter(actual) - Counter(known)).items())
        resolved = dict((Counter(known) - Counter(actual)).items())
        assert not added and not resolved, (
            f"{path.name}: {kind} changed. New or increased diagnostics: {added}. "
            f"Resolved or reduced diagnostics: {resolved}. "
            "Investigate regressions; for an intentional fix, add a focused "
            "regression test and remove the resolved baseline entries."
        )
    assert report["is_valid"] == (not expected["errors"])
