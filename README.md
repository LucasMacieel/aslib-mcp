# ASlib MCP Server

An MCP (Model Context Protocol) server exposing the core functionalities of [`aslib_scenario`](https://github.com/mlindauer/ASlibScenario), the official specification checker from [`aslib-spec`](https://github.com/coseal/aslib-spec), and the standard benchmark scenario repository [`aslib_data`](https://github.com/coseal/aslib_data) from the **Algorithm Selection Library (ASlib)**.

---

## 1. Overview & Scientific Purpose

Algorithm Selection (AS) benchmarks evaluate algorithms across diverse problem instances. The ASlib MCP server provides a comprehensive toolkit for ASlib 2.0 benchmarking:
- **ASlib Benchmark Repository & Discovery**: Integrated access to 45+ standardized scenarios across domains (SAT, CSP, QBF, MAXSAT, MIP, TSP, ASP, etc.) from `coseal/aslib_data`, with high-speed indexing, domain/objective filtering, shallow-clone local caching (`~/.cache/aslib_data`), and transparent Scenario ID resolution.
- **ASlib Benchmark Reading**: Parses ASlib scenario folders containing YAML specifications (`description.txt`) and ARFF files (`algorithm_runs.arff`, `feature_values.arff`, `feature_runstatus.arff`, `cv.arff`, etc.), aggregating repeated measurements according to official rules (median for runtimes and costs, mean for features, mode for runstatuses).
- **Tabular CSV Scenario Loading & Export**: Constructs complete ASlib scenario representations directly from separate CSV matrices, or exports CSV benchmarks into fully compliant ASlib directories.
- **Cross-Validation Partitioning**: Extracts disjoint train/test splits for any cross-validation fold index, or automatically generates balanced $k$-fold CV partitions using scikit-learn.
- **Multi-Metric Management**: Switches between active performance metrics (e.g., runtime vs. solution quality vs. PAR10).
- **Scenario Validation & PAR10 Imputation**: Validates instance alignment and missing value constraints, automatically applying PAR10 penalties ($10 \times \text{cutoff}$) to non-OK runs and minimization sign inversions to maximization objectives.
- **Specification Linting**: Validates scenario directory layout and YAML/ARFF schemas against the official ASlib 2.0 specification.
- **Statistical Permutation Testing & Selector Benchmarking**: Evaluates algorithm selectors against SBS and VBS baselines with non-parametric permutation significance tests.

---

## 2. Supported Platforms & Requirements

- **Operating System**: Linux (x86_64, tested on CachyOS).
- **Python Version**: Python 3.12 (tested on CPython 3.12.13).
- **Hardware**: CPU only; no GPU or accelerator hardware required.
- **Core Dependencies**:
  - `aslib-scenario` (pinned to commit `c785d4ae524dd0f4624d4092eea62a05ee2830d4`)
  - `fastmcp==4.0.3`
  - `scikit-learn==1.9.1`
  - `pandas==3.0.6`
  - `numpy==2.5.3`
  - `scipy==1.18.1`
  - `pyyaml==6.0.3`
  - `liac-arff==2.5.0`

---

## 3. Installation & Setup

Extract the deliverable archive and navigate to the directory:

```bash
unzip aslib-mcp.zip
cd aslib_mcp
```

Create a virtual environment and sync runtime and development dependencies using [`uv`](https://docs.astral.sh/uv/):

```bash
# Sync all dependencies (runtime + dev) into .venv
uv sync

# Or sync only production runtime dependencies
uv sync --no-dev
```

---

## 4. Running the Server

### Direct Stdio Execution
To launch the MCP server over standard I/O (stdio):

```bash
.venv/bin/python src/aslib_mcp.py
```

### Client Configuration

#### Claude Desktop
Add the following entry to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aslib": {
      "command": "/ABSOLUTE/PATH/TO/aslib_mcp/.venv/bin/python",
      "args": ["/ABSOLUTE/PATH/TO/aslib_mcp/src/aslib_mcp.py"]
    }
  }
}
```

#### Claude Code
Register using the FastMCP CLI:

```bash
.venv/bin/fastmcp install claude-code src/aslib_mcp.py
```

#### Gemini CLI / Goose / Cursor
```bash
.venv/bin/fastmcp install gemini-cli src/aslib_mcp.py
.venv/bin/fastmcp install goose src/aslib_mcp.py
.venv/bin/fastmcp install cursor src/aslib_mcp.py
```

---

## 5. Tool Reference

The server exposes 14 distinct tools:

### 1. `aslib_list_scenarios`
Discover and filter benchmark scenarios available in the official ASlib scenario repository (`coseal/aslib_data`).

- **Parameters**:
  - `domain` (`str | null`, *optional*, default `null`): Problem domain filter (e.g. `"SAT"`, `"CSP"`, `"QBF"`, `"MAXSAT"`, `"MIP"`, `"TSP"`, `"ASP"`, `"AutoML"`).
  - `objective` (`str | null`, *optional*, default `null`): Optimization objective filter (`"runtime"` vs. `"solution_quality"`).
  - `search` (`str | null`, *optional*, default `null`): Substring keyword search matching scenario ID or problem domain.
  - `limit` (`int`, default `50`): Maximum number of scenarios to return.
- **Returns**: Dictionary containing total catalog count, matched count, and scenario summaries (domain, objective, instance count, algorithm count, cutoff, and verification health `has_cv`, `has_costs`, `is_valid`).

### 2. `aslib_get_scenario_info`
Retrieve detailed specification metadata for an ASlib scenario without loading full ARFF matrices.

- **Parameters**:
  - `scenario_id` (`str`, **required**): Canonical Scenario ID (e.g. `"SAT12-ALL"`, `"ASP-POTASSCO"`) or local filesystem directory path.
- **Returns**: Dictionary with scenario metadata: domain, instance count, algorithms, feature steps, cutoff time and memory, performance measures, citation BibTeX, and readme notes.

### 3. `aslib_fetch_scenario`
Ensure a benchmark scenario from `coseal/aslib_data` is locally available and validated in the Scenario Cache (`~/.cache/aslib_data`).

- **Parameters**:
  - `scenario_id` (`str`, **required**): Canonical Scenario ID (e.g. `"SAT12-ALL"`, `"ASP-POTASSCO"`, `"QBF-2016"`).
  - `force_update` (`bool`, default `false`): Whether to force re-cloning or refreshing from upstream.
- **Returns**: Dictionary with `scenario_id`, local filesystem `scenario_dir`, validation status, and present files.

### 4. `aslib_sync_repository`
Synchronize the local ASlib Scenario Cache with the upstream `coseal/aslib_data` GitHub repository via git clone or pull.

- **Parameters**:
  - `force` (`bool`, default `false`): Whether to force re-cloning or re-indexing.
- **Returns**: Dictionary with sync status, commit output, and total indexed scenarios.

### 5. `aslib_read_scenario`
Read and parse an ASlib benchmark scenario directory into tabular data structures. Supports either a local directory path or a canonical Scenario ID (with automatic resolution and cache fetching).

- **Parameters**:
  - `scenario_dir` (`str`, **required**): Path to the ASlib scenario directory or canonical Scenario ID (e.g. `"SAT12-ALL"`, `"ASP-POTASSCO"`).
  - `output_dir` (`str | null`, *optional*, default `null`): Base output directory. A fresh unique subdirectory is created for output CSV files.
- **Returns**: Dictionary with:
  - `scenario_id`: Name of the scenario.
  - `num_instances`: Number of unique problem instances.
  - `algorithms`: List of algorithm names.
  - `features`: List of feature names.
  - `performance_measures`: Available performance metric names.
  - `algorithm_cutoff_time`: Cutoff time from description.
  - `artifacts`: List of exported CSV files (`aslib_performance_active.csv`, `aslib_features.csv`, `aslib_runstatus.csv`, etc.).

### 6. `aslib_read_csv`
Construct an ASlibScenario instance from tabular algorithm performance and instance feature CSV files.

- **Parameters**:
  - `perf_csv_path` (`str`, **required**): CSV file where rows are instances and columns are algorithm performance values.
  - `feat_csv_path` (`str`, **required**): CSV file where rows are instances and columns are feature values.
  - `objective` (`Literal["runtime", "solution_quality"]`, default `"runtime"`): Optimization objective.
  - `runtime_cutoff` (`float`, default `1000.0`): Cutoff threshold. Runs with runtime $\ge$ cutoff are marked as `"timeout"`.
  - `maximize` (`bool`, default `false`): Whether higher performance values indicate better solutions.
  - `cv_csv_path` (`str | null`, *optional*, default `null`): Optional CSV file containing CV fold assignments.
  - `output_dir` (`str | null`, *optional*, default `null`): Base output directory.
- **Returns**: Dictionary with `num_instances`, `algorithms`, `features`, `num_timeouts`, and serialized table `artifacts`.

### 7. `aslib_get_cv_split`
Partition an ASlib scenario into disjoint training and test splits for cross-validation evaluation. Supports local directory paths or canonical Scenario IDs.

- **Parameters**:
  - `scenario_dir` (`str`, **required**): Path to the ASlib scenario directory or canonical Scenario ID (e.g. `"SAT12-ALL"`, `"ASP-POTASSCO"`).
  - `fold_index` (`int`, default `1`): 1-based index of the cross-validation fold to use as test split.
  - `output_dir` (`str | null`, *optional*, default `null`): Base output directory.
- **Returns**: Dictionary with `fold_index`, `train_num_instances`, `test_num_instances`, `test_instances` list, and `artifacts` for train/test performance, feature, and runstatus matrices.

### 8. `aslib_create_cv_splits`
Generate balanced $k$-fold cross-validation split assignments for all instances in a scenario. Supports local directory paths or canonical Scenario IDs.

- **Parameters**:
  - `scenario_dir` (`str`, **required**): Path to the ASlib scenario directory or canonical Scenario ID.
  - `n_folds` (`int`, default `10`): Number of cross-validation folds ($\ge 2$).
  - `seed` (`int | null`, *optional*, default `null`): Random seed for reproducible fold generation.
  - `output_dir` (`str | null`, *optional*, default `null`): Base output directory.
- **Returns**: Dictionary with `n_folds`, `total_instances`, `instances_per_fold` distribution, and `cv_splits.csv` artifact.

### 9. `aslib_change_perf_measure`
Switch active performance measure in a multi-metric ASlib benchmark scenario. Supports local directory paths or canonical Scenario IDs.

- **Parameters**:
  - `scenario_dir` (`str`, **required**): Path to the multi-metric ASlib scenario directory or canonical Scenario ID.
  - `measure_name` (`str | null`, *optional*, default `null`): Name of target performance measure (e.g. `"solution_quality"`).
  - `measure_idx` (`int | null`, *optional*, default `null`): 0-based index of target performance measure.
  - `output_dir` (`str | null`, *optional*, default `null`): Base output directory.
- **Returns**: Dictionary with `active_measure_name`, `active_measure_index`, `available_measures`, mean performance per algorithm, and serialized active performance CSV artifact.

### 10. `aslib_validate_scenario`
Validate scenario data integrity against ASlib specifications and apply standard PAR10 and sign transformations. Supports local directory paths or canonical Scenario IDs.

- **Parameters**:
  - `scenario_dir` (`str`, **required**): Path to the ASlib scenario directory or canonical Scenario ID.
  - `output_dir` (`str | null`, *optional*, default `null`): Base output directory.
- **Returns**: Dictionary with `is_valid`, `instances_aligned`, `num_instances`, `num_algorithms`, `num_features`, `par10_imputations_count`, and `maximization_inverted`.

### 11. `aslib_lint_spec`
Validate an ASlib benchmark directory against the official ASlib 2.0 specification (`coseal/aslib-spec`). Performs deep static linting of directory layout, YAML schema in `description.txt`, ARFF headers and attribute types, feature step dependency DAGs, and cross-file instance consistency. Supports local directory paths or canonical Scenario IDs.

- **Parameters**:
  - `scenario_dir` (`str`, **required**): Path to the ASlib scenario directory or canonical Scenario ID.
  - `strict` (`bool`, default `false`): If `true`, raises `ValueError` on fatal specification errors instead of returning diagnostic report.
- **Returns**: Structured dictionary with `is_valid`, `scenario_id`, `errors`, `warnings`, `recommendations`, `checked_files`, and `summary`.

### 12. `aslib_export_scenario`
Export tabular CSV benchmarks into a fully compliant, self-contained ASlib scenario directory (`description.txt`, `algorithm_runs.arff`, `feature_values.arff`, `feature_runstatus.arff`, and optional `cv.arff` / `feature_costs.arff`).

- **Parameters**:
  - `perf_csv_path` (`str`, **required**): Path to algorithm performance CSV (rows: instances, cols: algorithms).
  - `feat_csv_path` (`str`, **required**): Path to instance features CSV (rows: instances, cols: features).
  - `scenario_id` (`str`, **required**): Identifier name for the exported scenario.
  - `cutoff` (`float`, **required**): Algorithm execution cutoff budget (time or cost threshold).
  - `output_dir` (`str | null`, default `null`): Target destination directory for exported scenario files.
  - `objective` (`Literal["runtime", "solution_quality"]`, default `"runtime"`): Scenario optimization objective.
  - `maximize` (`bool`, default `false`): Whether higher performance indicates better performance.
  - `feature_costs_csv_path` (`str | null`, default `null`): Optional CSV with feature extraction costs per instance.
  - `generate_cv` (`bool`, default `true`): Whether to automatically generate balanced CV folds.
  - `n_folds` (`int`, default `10`): Number of CV folds to generate if `generate_cv` is true.
  - `seed` (`int | null`, default `null`): Random seed for reproducible fold generation.
- **Returns**: Dictionary with `message`, `scenario_id`, `output_dir`, `created_files`, and file `artifacts`.

### 13. `aslib_permutation_test`
Run a paired permutation statistical significance test between two algorithms or selectors (adapted from `PermutationTester.py`). Computes empirical one-sided p-values with PAR cutoff penalization and optional Van Gelder noise filtering. Supports local directory paths or canonical Scenario IDs.

- **Parameters**:
  - `vec1` (`list[float] | null`, default `null`): Performance vector of first candidate.
  - `vec2` (`list[float] | null`, default `null`): Performance vector of second candidate.
  - `scenario_dir` (`str | null`, default `null`): Optional path to ASlib scenario or canonical Scenario ID to extract vectors directly.
  - `algo1` (`str | null`, default `null`): Name of first algorithm in scenario.
  - `algo2` (`str | null`, default `null`): Name of second algorithm in scenario.
  - `permutations` (`int`, default `10000`): Number of random permutation swaps.
  - `alpha` (`float`, default `0.05`): Significance level threshold.
  - `cutoff` (`float | null`, default `null`): Cutoff threshold for PAR score penalization.
  - `par_factor` (`float`, default `10.0`): Multiplier for timeouts (PAR10).
  - `noise` (`float | null`, default `null`): Van Gelder noise filtering coefficient.
  - `maximize` (`bool`, default `false`): Whether higher performance values are better.
  - `seed` (`int | null`, default `null`): Random seed for reproducible permutations.
- **Returns**: Dictionary with `is_significant`, `p_value`, `winner`, candidate means, `observed_diff`, and diagnostic `message`.

### 14. `aslib_evaluate_selectors`
Benchmark and rank algorithm selectors against an ASlib scenario and standard baselines (Single Best Solver and Virtual Best Solver), accounting for timeouts, unhandled instances, and feature extraction costs, with automatic significance tests against the winning selector. Supports local directory paths or canonical Scenario IDs.

- **Parameters**:
  - `scenario_dir` (`str`, **required**): Path to the ASlib scenario directory or canonical Scenario ID.
  - `predictions_csv_path` (`str`, **required**): Path to CSV with selector choices or predicted runtimes.
  - `include_feature_costs` (`bool`, default `true`): Whether to add feature extraction costs from `feature_costs.arff`.
  - `par_factor` (`float`, default `10.0`): Penalty factor for timeouts.
  - `run_significance_tests` (`bool`, default `true`): Whether to run permutation tests against the top selector.
  - `output_dir` (`str | null`, default `null`): Output directory for evaluation CSV artifacts.
- **Returns**: Dictionary with `rankings` table (SBS, VBS, and evaluated selectors), `top_selector`, `significance_tests`, and exported summary `artifacts`.

---

## 6. Example Workflow: Scenario Inspection & Cross-Validation

```python
import asyncio
from fastmcp import Client
from src.aslib_mcp import mcp


async def main():
    async with Client(mcp) as client:
        # 1. Discover available benchmark scenarios in coseal/aslib_data
        available = await client.call_tool(
            "aslib_list_scenarios", {"domain": "SAT", "limit": 5}
        )
        print(
            "Available SAT scenarios:",
            [s["scenario_id"] for s in available.data["scenarios"]],
        )

        # 2. Inspect metadata without heavy ARFF data loading
        info = await client.call_tool(
            "aslib_get_scenario_info", {"scenario_id": "ASP-POTASSCO"}
        )
        print(
            "Scenario info:",
            info.data["scenario_id"],
            "- Instances:",
            info.data["instances_count"],
        )

        # 3. Read scenario directly via Scenario ID (transparent resolution)
        scen_info = await client.call_tool(
            "aslib_read_scenario", {"scenario_dir": "ASP-POTASSCO"}
        )
        print("Loaded scenario:", scen_info.data["scenario_id"])
        print("Algorithms:", scen_info.data["algorithms"])

        # 4. Validate scenario and check PAR10 imputations using Scenario ID
        val_info = await client.call_tool(
            "aslib_validate_scenario", {"scenario_dir": "ASP-POTASSCO"}
        )
        print("PAR10 imputations:", val_info.data["par10_imputations_count"])

        # 5. Partition scenario into fold 1 test and train splits
        split_info = await client.call_tool(
            "aslib_get_cv_split", {"scenario_dir": "ASP-POTASSCO", "fold_index": 1}
        )
        print("Train instances:", split_info.data["train_num_instances"])
        print("Test instances:", split_info.data["test_num_instances"])


asyncio.run(main())
```

---

## 7. Validation & Limitations

- **Verification Scope**: All 6 tools passed strict verification with 31 pytest tests and 14 MCP runtime acceptance cases covering reference agreement, changed inputs, input validation, and artifact collision resistance.
- **Independent Runtime Testing**: Verified in both development environment and a completely clean Python 3.12 virtual environment over stdio transport.
- **Known Upstream Notes**:
  - Upstream `change_perf_measure` uses truthiness check for index (`if measure_idx:`), which is handled safely by the tool wrapper when switching to index `0`.
  - When calling `aslib_create_cv_splits`, provide a `seed` parameter if deterministic fold assignment is required.
