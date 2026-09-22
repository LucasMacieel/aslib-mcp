"""Algorithm selector benchmark evaluator.

Evaluates algorithm selector predictions against an ASlib scenario, accounts for
unsolved instances, feature extraction costs, computes PAR10 rankings, benchmarks
against SBS (Single Best Solver) and VBS (Virtual Best Solver), and executes
permutation tests against the winning selector.
Modernized and combined from coseal/aslib-spec (scripts/merge_results.py & scripts/zilla_evaluate.py).
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from aslib_scenario.aslib_scenario import ASlibScenario

from src.tools.permutation_tester import PermutationTester


class SelectorEvaluator:
    """Evaluates algorithm selectors on ASlib benchmark scenarios."""

    def __init__(
        self,
        scenario_dir: str | Path,
        par_factor: float = 10.0,
    ):
        self.scenario_dir = Path(scenario_dir).resolve()
        self.par_factor = par_factor

        self.scenario = ASlibScenario()
        self.scenario.read_scenario(str(self.scenario_dir))

        if self.scenario.performance_data is None:
            raise ValueError(
                f"Scenario at {self.scenario_dir} has no performance data."
            )

        self.cutoff = (
            float(self.scenario.algorithm_cutoff_time)
            if self.scenario.algorithm_cutoff_time is not None
            else 1000.0
        )
        self.maximize = (
            bool(self.scenario.maximize[0]) if self.scenario.maximize else False
        )
        self.objective = (
            str(self.scenario.performance_type[0])
            if self.scenario.performance_type
            else "runtime"
        )
        self.instances = list(self.scenario.instances)
        self.algorithms = list(self.scenario.algorithms)

    def _get_feature_cost_per_instance(self) -> dict[str, float]:
        """Calculates total feature extraction cost per instance for default feature steps."""
        costs = {inst: 0.0 for inst in self.instances}
        if self.scenario.feature_cost_data is not None:
            cost_df = self.scenario.feature_cost_data
            # Identify columns corresponding to default feature steps or total
            for inst in self.instances:
                if inst in cost_df.index:
                    row = cost_df.loc[inst]
                    costs[inst] = float(np.sum(np.nan_to_num(row.values, nan=0.0)))
        return costs

    def evaluate(
        self,
        predictions_csv_path: str | Path,
        include_feature_costs: bool = True,
        run_significance_tests: bool = True,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Evaluates selectors from predictions CSV."""
        pred_path = Path(predictions_csv_path).resolve()
        if not pred_path.is_file():
            raise FileNotFoundError(f"Predictions CSV not found: {pred_path}")

        pred_df = pd.read_csv(pred_path, index_col=0)
        pred_df.index = pred_df.index.astype(str).str.strip()

        feat_costs = (
            self._get_feature_cost_per_instance() if include_feature_costs else {}
        )

        perf_matrix = self.scenario.performance_data
        runstatus_matrix = self.scenario.runstatus_data

        # Determine baselines: SBS and VBS
        baselines_perf: dict[str, dict[str, float]] = {}

        # 1. Compute VBS (Virtual Best Solver)
        vbs_perfs = {}
        for inst in self.instances:
            if inst in perf_matrix.index:
                row = perf_matrix.loc[inst]
                vbs_perfs[inst] = float(row.max() if self.maximize else row.min())
            else:
                vbs_perfs[inst] = (
                    self.cutoff * self.par_factor if not self.maximize else 0.0
                )
        baselines_perf["VBS"] = vbs_perfs

        # 2. Compute SBS (Single Best Solver)
        sbs_algo = None
        sbs_best_score = float("inf") if not self.maximize else float("-inf")
        sbs_scores_dict = {}

        for algo in self.algorithms:
            col = perf_matrix[algo].copy()
            if self.objective == "runtime":
                penalized = np.where(
                    col >= self.cutoff, self.cutoff * self.par_factor, col
                )
            else:
                penalized = col.values
            algo_mean = float(np.mean(penalized))
            sbs_scores_dict[algo] = algo_mean

            if not self.maximize:
                if algo_mean < sbs_best_score:
                    sbs_best_score = algo_mean
                    sbs_algo = algo
            else:
                if algo_mean > sbs_best_score:
                    sbs_best_score = algo_mean
                    sbs_algo = algo

        sbs_perfs = {}
        for inst in self.instances:
            sbs_perfs[inst] = (
                float(perf_matrix.loc[inst, sbs_algo])
                if inst in perf_matrix.index
                else (self.cutoff * self.par_factor if not self.maximize else 0.0)
            )
        baselines_perf[f"SBS ({sbs_algo})"] = sbs_perfs

        # 3. Evaluate each selector column in pred_df
        selector_results: dict[str, dict[str, float]] = {}
        selector_perf_vectors: dict[str, list[float]] = {}
        selector_stats: dict[str, dict[str, Any]] = {}

        candidate_cols = [str(c).strip() for c in pred_df.columns]

        # Check if single column without name
        if len(candidate_cols) == 1 and candidate_cols[0].lower() in (
            "algorithm",
            "algo",
            "selected",
        ):
            candidate_cols = [pred_path.stem]
            pred_df.columns = candidate_cols

        for sel in candidate_cols:
            col_data = pred_df[sel]
            inst_perf = {}
            timeouts_count = 0
            solved_count = 0

            for inst in self.instances:
                if inst not in col_data.index:
                    # Missing prediction: penalty
                    final_time = (
                        self.cutoff * self.par_factor if not self.maximize else 0.0
                    )
                    timeouts_count += 1
                else:
                    val = col_data.loc[inst]
                    f_cost = feat_costs.get(inst, 0.0)

                    # Check if val is an algorithm name
                    if isinstance(val, str) and val in self.algorithms:
                        base_time = float(perf_matrix.loc[inst, val])
                        status = (
                            str(runstatus_matrix.loc[inst, val]).lower()
                            if runstatus_matrix is not None
                            and inst in runstatus_matrix.index
                            else "ok"
                        )
                    else:
                        # Assume precomputed numeric performance
                        try:
                            base_time = float(val)
                            status = "ok" if base_time < self.cutoff else "timeout"
                        except (ValueError, TypeError):
                            base_time = self.cutoff
                            status = "crash"

                    if self.objective == "runtime":
                        total_time = base_time + f_cost
                        if status != "ok" or total_time >= self.cutoff:
                            final_time = self.cutoff * self.par_factor
                            timeouts_count += 1
                        else:
                            final_time = total_time
                            solved_count += 1
                    else:
                        final_time = base_time
                        if not self.maximize:
                            if final_time > self.cutoff or status != "ok":
                                timeouts_count += 1
                            else:
                                solved_count += 1
                        else:
                            if final_time < self.cutoff or status != "ok":
                                timeouts_count += 1
                            else:
                                solved_count += 1

                inst_perf[inst] = final_time

            selector_results[sel] = inst_perf
            vec = [inst_perf[inst] for inst in self.instances]
            selector_perf_vectors[sel] = vec

            mean_score = float(np.mean(vec))
            selector_stats[sel] = {
                "name": sel,
                "par_score": round(mean_score, 3),
                "solved_count": solved_count,
                "solved_percentage": round(
                    float(solved_count / len(self.instances) * 100), 2
                ),
                "timeouts_count": timeouts_count,
                "timeouts_percentage": round(
                    float(timeouts_count / len(self.instances) * 100), 2
                ),
            }

        # Add baselines to stats and vectors
        for b_name, b_dict in baselines_perf.items():
            b_vec = []
            b_timeouts = 0
            b_solved = 0
            for inst in self.instances:
                val = b_dict[inst]
                if self.objective == "runtime" and val >= self.cutoff:
                    val = self.cutoff * self.par_factor
                    b_timeouts += 1
                else:
                    b_solved += 1
                b_vec.append(val)

            selector_perf_vectors[b_name] = b_vec
            b_mean = float(np.mean(b_vec))
            selector_stats[b_name] = {
                "name": b_name,
                "par_score": round(b_mean, 3),
                "solved_count": b_solved,
                "solved_percentage": round(
                    float(b_solved / len(self.instances) * 100), 2
                ),
                "timeouts_count": b_timeouts,
                "timeouts_percentage": round(
                    float(b_timeouts / len(self.instances) * 100), 2
                ),
            }

        # Ranking
        sorted_stats = sorted(
            selector_stats.values(),
            key=lambda x: x["par_score"],
            reverse=self.maximize,
        )
        for rank_idx, stat in enumerate(sorted_stats, start=1):
            stat["rank"] = rank_idx

        # Permutation significance test against the top selector (excluding VBS oracle)
        top_selector_stat = None
        for s in sorted_stats:
            if not s["name"].startswith("VBS"):
                top_selector_stat = s
                break
        if top_selector_stat is None:
            top_selector_stat = sorted_stats[0]

        top_name = top_selector_stat["name"]
        significance_results = []
        if run_significance_tests:
            tester = PermutationTester()
            top_vec = selector_perf_vectors[top_name]

            for s in sorted_stats:
                other_name = s["name"]
                if other_name == top_name or other_name.startswith("VBS"):
                    continue
                test_res = tester.run_test(
                    vec1=top_vec,
                    vec2=selector_perf_vectors[other_name],
                    alpha=0.05,
                    permutations=5000,
                    name1=top_name,
                    name2=other_name,
                    maximize=self.maximize,
                )
                significance_results.append(
                    {
                        "comparison": f"{top_name} vs {other_name}",
                        "p_value": test_res["p_value"],
                        "is_significant": test_res["is_significant"],
                        "winner": test_res["winner"],
                        "observed_diff": test_res["observed_diff"],
                        "message": test_res["message"],
                    }
                )

        # Save artifacts
        artifacts = []
        if output_dir is not None:
            out_path = Path(output_dir).resolve()
        else:
            uid = f"eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            out_path = Path(tempfile.gettempdir()).resolve() / uid
        out_path.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(sorted_stats)
        summary_csv = out_path / "selector_evaluation_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        artifacts.append(
            {
                "description": "Selector ranking summary table",
                "path": str(summary_csv.resolve()),
            }
        )

        per_instance_df = pd.DataFrame(selector_perf_vectors, index=self.instances)
        per_inst_csv = out_path / "per_instance_penalized_performance.csv"
        per_instance_df.to_csv(per_inst_csv)
        artifacts.append(
            {
                "description": "Per-instance penalized performance matrix",
                "path": str(per_inst_csv.resolve()),
            }
        )

        return {
            "message": f"Successfully evaluated {len(candidate_cols)} selector(s) against scenario '{self.scenario.scenario}'.",
            "scenario_id": self.scenario.scenario,
            "num_instances": len(self.instances),
            "cutoff": self.cutoff,
            "par_factor": self.par_factor,
            "objective": self.objective,
            "feature_costs_included": include_feature_costs,
            "rankings": sorted_stats,
            "top_selector": top_name,
            "significance_tests": significance_results,
            "artifacts": artifacts,
        }
