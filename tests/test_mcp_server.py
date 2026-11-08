"""Protocol-level FastMCP integration tests using in-memory Client."""

from pathlib import Path

import pytest
from fastmcp import Client

from src.aslib_mcp import mcp

EXPECTED_TOOLS = {
    "aslib_list_scenarios",
    "aslib_get_scenario_info",
    "aslib_fetch_scenario",
    "aslib_sync_repository",
    "aslib_read_scenario",
    "aslib_read_csv",
    "aslib_get_cv_split",
    "aslib_create_cv_splits",
    "aslib_change_perf_measure",
    "aslib_validate_scenario",
    "aslib_lint_spec",
    "aslib_export_scenario",
    "aslib_permutation_test",
    "aslib_evaluate_selectors",
}


@pytest.mark.anyio
async def test_mcp_server_tool_enumeration():
    """Server must mount and expose all 14 ASlib tools with non-empty descriptions."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}

        assert tool_names == EXPECTED_TOOLS
        for t in tools:
            assert t.description is not None
            assert len(t.description.strip()) > 10
            assert t.input_schema is not None


@pytest.mark.anyio
async def test_mcp_client_call_list_scenarios(mini_scenario_dir, monkeypatch):
    """Client can invoke aslib_list_scenarios over MCP protocol."""
    monkeypatch.setenv("ASLIB_DATA_DIR", str(mini_scenario_dir.parent))
    async with Client(mcp) as client:
        res = await client.call_tool("aslib_list_scenarios", {"limit": 3})
        assert not res.is_error
        data = res.data
        assert "scenarios" in data
        assert "total_scenarios" in data
        assert len(data["scenarios"]) <= 3


@pytest.mark.anyio
async def test_mcp_client_call_lint_spec(mini_scenario_dir: Path):
    """Client can invoke aslib_lint_spec over MCP protocol."""
    async with Client(mcp) as client:
        res = await client.call_tool(
            "aslib_lint_spec", {"scenario_dir": str(mini_scenario_dir)}
        )
        assert not res.is_error
        data = res.data
        assert data["is_valid"] is True
        assert data["scenario_id"] == "SYNTHETIC-SCENARIO"
        assert len(data["errors"]) == 0


@pytest.mark.anyio
async def test_mcp_client_call_permutation_test():
    """Client can invoke aslib_permutation_test over MCP protocol."""
    async with Client(mcp) as client:
        res = await client.call_tool(
            "aslib_permutation_test",
            {
                "vec1": [1.0, 2.0, 3.0],
                "vec2": [1.0, 2.0, 3.0],
                "permutations": 500,
                "seed": 42,
            },
        )
        assert not res.is_error
        data = res.data
        assert data["p_value"] == 1.0
        assert data["winner"] == "tie"


@pytest.mark.anyio
async def test_mcp_generated_cv_is_reusable(mini_scenario_dir: Path):
    async with Client(mcp) as client:
        generated = await client.call_tool(
            "aslib_create_cv_splits",
            {
                "scenario_dir": str(mini_scenario_dir),
                "n_folds": 3,
                "seed": 19,
            },
        )
        result = await client.call_tool(
            "aslib_get_cv_split",
            {
                "scenario_dir": str(mini_scenario_dir),
                "fold_index": 3,
                "cv_csv_path": generated.data["artifacts"][0]["path"],
            },
        )
        assert not result.is_error
        assert (
            result.data["train_num_instances"] + result.data["test_num_instances"] == 5
        )


@pytest.mark.anyio
async def test_malformed_scenario_does_not_end_session(mini_scenario_dir: Path):
    import arff

    path = mini_scenario_dir / "algorithm_runs.arff"
    with path.open() as stream:
        data = arff.load(stream)
    data["data"].pop(0)
    with path.open("w") as stream:
        arff.dump(data, stream)
    async with Client(mcp, timeout=10) as client:
        failed = await client.call_tool(
            "aslib_read_scenario",
            {"scenario_dir": str(mini_scenario_dir)},
            raise_on_error=False,
        )
        assert failed.is_error
        result = await client.call_tool(
            "aslib_permutation_test", {"vec1": [1, 2], "vec2": [1, 2]}
        )
        assert not result.is_error
        assert result.data["p_value"] == 1.0
