"""ASlib MCP Server.

Exposes Algorithm Selection Library (ASlib) scenario operations as MCP tools:
- aslib_list_scenarios
- aslib_get_scenario_info
- aslib_fetch_scenario
- aslib_sync_repository
- aslib_read_scenario
- aslib_read_csv
- aslib_get_cv_split
- aslib_create_cv_splits
- aslib_change_perf_measure
- aslib_validate_scenario
- aslib_lint_spec
- aslib_export_scenario
- aslib_permutation_test
- aslib_evaluate_selectors
"""

import sys
from pathlib import Path

from fastmcp import FastMCP

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.aslib_scenario import aslib_scenario_mcp

mcp = FastMCP(name="ASlib MCP")
mcp.mount(aslib_scenario_mcp)

if __name__ == "__main__":
    mcp.run()
