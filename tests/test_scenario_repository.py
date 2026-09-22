"""Tests for ASlib scenario repository, caching, discovery, and Scenario ID resolution."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from src.tools.aslib_scenario import (
    aslib_get_cv_split,
    aslib_lint_spec,
    aslib_read_scenario,
    aslib_validate_scenario,
)
from src.tools.scenario_repository import (
    fetch_scenario,
    get_cache_dir,
    get_scenario_info,
    is_cache_initialized,
    list_scenarios,
    resolve_scenario_dir,
)


def test_cache_initialization_and_path():
    cache = get_cache_dir()
    assert cache.is_dir()
    assert is_cache_initialized(cache)


def test_custom_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ASLIB_DATA_DIR", str(tmp_path))
    assert get_cache_dir() == tmp_path


def test_list_scenarios_filtering():
    all_res = list_scenarios()
    assert all_res["total_scenarios"] >= 40
    assert len(all_res["scenarios"]) <= 50

    asp_res = list_scenarios(domain="ASP")
    assert asp_res["matched_scenarios"] >= 1
    asp_ids = [s["scenario_id"] for s in asp_res["scenarios"]]
    assert "ASP-POTASSCO" in asp_ids

    sat_res = list_scenarios(domain="SAT")
    assert sat_res["matched_scenarios"] >= 10
    for s in sat_res["scenarios"]:
        assert s["domain"] == "SAT" or "SAT" in s["scenario_id"].upper()

    qbf_res = list_scenarios(search="QBF-2016")
    assert qbf_res["matched_scenarios"] >= 1
    assert any(s["scenario_id"] == "QBF-2016" for s in qbf_res["scenarios"])

    rt_res = list_scenarios(objective="runtime")
    assert rt_res["matched_scenarios"] >= 30


def test_get_scenario_info():
    info = get_scenario_info("ASP-POTASSCO")
    assert info["scenario_id"] == "ASP-POTASSCO"
    assert info["domain"] == "ASP"
    assert info["instances_count"] == 1294
    assert len(info["algorithms"]) == 11
    assert "runtime" in info["performance_measures"]
    assert info["algorithm_cutoff_time"] == 600
    assert info["has_cv"] is True
    assert info["has_costs"] is True
    assert "description.txt" in info["files_present"]


def test_fetch_scenario():
    fetch = fetch_scenario("ASP-POTASSCO")
    assert fetch["is_valid"] is True
    assert fetch["scenario_id"] == "ASP-POTASSCO"
    assert Path(fetch["scenario_dir"]).is_dir()
    assert (Path(fetch["scenario_dir"]) / "description.txt").is_file()


def test_resolve_scenario_dir_direct_path():
    cache = get_cache_dir()
    direct = cache / "ASP-POTASSCO"
    resolved = resolve_scenario_dir(str(direct))
    assert resolved == direct.resolve()


def test_resolve_scenario_dir_canonical_and_case_insensitive():
    res1 = resolve_scenario_dir("ASP-POTASSCO")
    res2 = resolve_scenario_dir("asp-potassco")
    assert res1 == res2
    assert res1.name == "ASP-POTASSCO"


def test_resolve_scenario_dir_fuzzy_error():
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_scenario_dir("SAT12-IND")
    err_msg = str(exc_info.value)
    assert "SAT12-IND" in err_msg
    assert "Did you mean" in err_msg
    assert "SAT12-INDU" in err_msg


def test_mcp_tool_aslib_read_scenario_with_id():
    res = aslib_read_scenario("ASP-POTASSCO")
    assert res["scenario_id"] == "ASP-POTASSCO"
    assert res["num_instances"] == 1294
    assert len(res["algorithms"]) == 11
    assert len(res["artifacts"]) >= 2


def test_mcp_tool_aslib_lint_spec_with_id():
    res = aslib_lint_spec("ASP-POTASSCO")
    assert res["is_valid"] is True
    assert res["scenario_id"] == "ASP-POTASSCO"


def test_mcp_tool_aslib_validate_scenario_with_id():
    res = aslib_validate_scenario("ASP-POTASSCO")
    assert res["is_valid"] is True
    assert res["num_instances"] == 1294
    assert res["num_algorithms"] == 11


def test_mcp_tool_aslib_get_cv_split_with_id():
    res = aslib_get_cv_split("ASP-POTASSCO", fold_index=1)
    assert res["fold_index"] == 1
    assert res["train_num_instances"] + res["test_num_instances"] == 1294
