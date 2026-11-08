# Repository Guidelines

## Project Structure & Module Organization

This Python 3.12+ project exposes ASlib benchmarking tools through FastMCP. `src/aslib_mcp.py` defines the server and MCP tool entry points. `src/tools/` contains scenario loading, repository caching, CSV handling, export, specification checking, selector evaluation, and permutation testing. Keep reusable logic in these modules and protocol handling in the server.

`tests/` contains unit, regression, integration, and MCP protocol tests. Shared synthetic scenarios and fixtures live in `tests/conftest.py`; benchmark data is cached externally. `pyproject.toml` defines dependencies and tool settings, while `uv.lock` pins the environment.

## Build, Test, and Development Commands

- `uv sync`: install runtime and development dependencies into `.venv`.
- `uv run python src/aslib_mcp.py`: start the MCP server over stdio.
- `uv build`: build source and wheel distributions using Hatchling.
- `uv run pytest -m "not integration"`: run deterministic tests without downloading benchmarks.
- `uv run ruff check src tests`: lint source and tests.
- `uv run ty check src tests`: check Python types.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Add type annotations and concise docstrings to public helpers. Prefer `pathlib.Path` for filesystem operations. Preserve the `aslib_` prefix for exposed MCP tools. Ruff is available for linting; no custom formatter configuration is declared.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Reuse `make_scenario`, `synthetic_data`, and `tmp_path` for isolated fixtures. Add focused regression tests for corrected behavior, including malformed inputs and scientific scoring edge cases. No numerical coverage threshold is configured.

Mark tests needing real benchmark data with `@pytest.mark.integration`. To run the full suite:

```bash
git clone --depth 1 https://github.com/coseal/aslib_data.git /tmp/aslib_data
ASLIB_DATA_DIR=/tmp/aslib_data uv run pytest
```

## Commit & Pull Request Guidelines

Recent commits use imperative, descriptive subjects such as “Fix ASlib cache, evaluation, export and validation correctness”; follow that style. Pull requests should explain the problem, resulting behavior, relevant issues, and validation commands with results. Update `README.md` when tool parameters or scientific assumptions change.

## Configuration & Data

Set `ASLIB_DATA_DIR` to override `~/.cache/aslib_data`. Keep downloaded benchmarks and generated artifacts outside version control. Preserve cache contents when synchronization fails.
