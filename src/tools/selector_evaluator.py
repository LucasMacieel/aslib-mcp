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
from typing import Any, Literal

import numpy as np
import pandas as pd
from aslib_scenario.aslib_scenario import ASlibScenario

from src.tools.csv_tables import read_matrix_csv
from src.tools.permutation_tester import PermutationTester


class SelectorEvaluator:
    """Evaluates algorithm selectors on ASlib benchmark scenarios."""

    def __init__(
        self,
        scenario_dir: str | Path,
        par_factor: float = 10.0,
    ):
        self.scenario_dir = Path(scenario_dir).resolve()
        if not np.isfinite(par_factor) or par_factor <= 0:
            raise ValueError("PAR factor must be finite and positive")
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
        self.instances = list(self.scenario.instances or [])
        self.algorithms = list(self.scenario.algorithms or [])

    def _feature_processing(self) -> tuple[dict[str, float], set[str]]:
        """Charge default steps and dependencies, stopping on feature presolving."""
        groups = self.scenario.feature_group_dict
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(step):
            if step in visiting:
                raise ValueError("Feature dependency cycle")
            if step in ordered:
                return
            if step not in groups:
                raise ValueError(f"Undefined feature step: {step}")
            visiting.add(step)
            for dep in groups[step].get("requires", []) or []:
                visit(dep)
            visiting.remove(step)
            ordered.append(step)

        for step in self.scenario.feature_steps_default or self.scenario.feature_steps:
            visit(step)
        costs = dict.fromkeys(self.instances, 0.0)
        presolved = set()
        cost_df = self.scenario.feature_cost_data
        statuses = self.scenario.feature_runstatus_data
        for inst in self.instances:
            for step in ordered:
                if cost_df is not None:
                    if step not in cost_df or inst not in cost_df.index:
                        raise ValueError(f"Missing feature cost for {inst}/{step}")
                    value = float(cost_df.loc[inst, step])
                    if not np.isfinite(value) or value < 0:
                        raise ValueError(
                            f"Unknown or invalid feature cost for {inst}/{step}"
                        )
                    costs[inst] += value
                if (
                    statuses is not None
                    and str(statuses.loc[inst, step]).lower() == "presolved"
                ):
                    if cost_df is None:
                        raise ValueError(
                            "Presolved runtime evaluation requires feature costs"
                        )
                    presolved.add(inst)
                    break
        return costs, presolved

    def evaluate(
        self,
        predictions_csv_path: str | Path,
        include_feature_costs: bool = True,
        run_significance_tests: bool = True,
        output_dir: str | Path | None = None,
        input_mode: Literal[
            "algorithm_selections", "measured_performance"
        ] = "algorithm_selections",
    ) -> dict[str, Any]:
        """Evaluate measured outcomes of selections, or explicitly supplied measurements.

        Numeric predictions are not observed outcomes. In measured_performance mode
        the caller supplies original-scale measurements, including feature costs
        only when include_feature_costs=False.
        """
        if input_mode not in ("algorithm_selections", "measured_performance"):
            raise ValueError("Unknown evaluation input_mode")
        pred_path = Path(predictions_csv_path).resolve()
        if not pred_path.is_file():
            raise FileNotFoundError(f"Predictions CSV not found: {pred_path}")
        pred_df = read_matrix_csv(pred_path, strings=True)
        pred_df.index = pred_df.index.astype(str).str.strip()
        pred_df.columns = pred_df.columns.astype(str).str.strip()
        if pred_df.index.has_duplicates or pred_df.columns.has_duplicates:
            raise ValueError(
                "Prediction instance IDs and selector names must be unique"
            )
        if not len(pred_df.columns) or not self.instances or not self.algorithms:
            raise ValueError("Evaluation needs instances, algorithms, and selectors")
        if set(pred_df.index) - set(self.instances):
            raise ValueError("Predictions contain unknown instances")
        if self.objective == "solution_quality" and (
            set(pred_df.index) != set(self.instances) or (pred_df == "").any().any()
        ):
            raise ValueError(
                "Missing quality predictions require a domain-specific penalty"
            )
        if len(pred_df.columns) == 1 and pred_df.columns[0].lower() in (
            "algorithm",
            "algo",
            "selected",
        ):
            pred_df.columns = [pred_path.stem]
        candidate_cols = list(pred_df.columns)
        if any(
            name in {"VBS"} or name.startswith("SBS (") or not name
            for name in candidate_cols
        ):
            raise ValueError(
                "Selector names must be non-empty and not use baseline names"
            )

        perf_matrix = self.scenario.performance_data
        if perf_matrix is None:
            raise ValueError("Scenario has no performance data loaded")
        # Upstream negates quality maximization. Report and compare original units.
        perf_matrix = -perf_matrix if self.maximize else perf_matrix.copy()
        runstatus = self.scenario.runstatus_data
        if runstatus is None:
            raise ValueError("Scenario has no algorithm runstatuses")
        ok_matrix = runstatus.eq("ok")
        if self.objective == "runtime":
            ok_matrix = ok_matrix & (perf_matrix < self.cutoff)
            perf_matrix = perf_matrix.where(ok_matrix, self.cutoff * self.par_factor)
        if not np.isfinite(perf_matrix.to_numpy(dtype=float)).all():
            raise ValueError("Performance values must be finite")
        feat_costs, presolved = (
            self._feature_processing()
            if include_feature_costs and self.objective == "runtime"
            else ({}, set())
        )
        selector_perf_vectors: dict[str, list[float]] = {}
        solved_vectors: dict[str, list[bool]] = {}
        for sel in candidate_cols:
            values = []
            solved = []
            for inst in self.instances:
                val = pred_df.loc[inst, sel] if inst in pred_df.index else ""
                if val == "":
                    score, ok = self.cutoff * self.par_factor, False
                elif input_mode == "algorithm_selections":
                    if val not in self.algorithms:
                        raise ValueError(
                            f"Unknown algorithm '{val}'; numeric measurements require measured_performance mode"
                        )
                    score = float(perf_matrix.loc[inst, val])
                    ok = bool(ok_matrix.loc[inst, val])
                else:
                    try:
                        score = float(val)
                    except ValueError as exc:
                        raise ValueError(
                            "Measured performance must be numeric"
                        ) from exc
                    if not np.isfinite(score) or (
                        self.objective == "runtime" and score < 0
                    ):
                        raise ValueError(
                            "Measured performance must be finite and runtime non-negative"
                        )
                    ok = self.objective != "runtime" or score < self.cutoff
                if self.objective == "runtime":
                    if inst in presolved:
                        score, ok = feat_costs[inst], True
                    else:
                        score += feat_costs.get(inst, 0.0)
                    ok = ok and score < self.cutoff
                    if not ok:
                        score = self.cutoff * self.par_factor
                values.append(score)
                solved.append(ok)
            selector_perf_vectors[sel] = values
            solved_vectors[sel] = solved

        means = perf_matrix.mean()
        sbs_algo = means.idxmax() if self.maximize else means.idxmin()
        best_algorithms = (
            perf_matrix.idxmax(axis=1) if self.maximize else perf_matrix.idxmin(axis=1)
        )
        for name, choices in (
            ("VBS", best_algorithms),
            (f"SBS ({sbs_algo})", dict.fromkeys(self.instances, sbs_algo)),
        ):
            selector_perf_vectors[name] = [
                float(perf_matrix.loc[i, choices[i]]) for i in self.instances
            ]
            solved_vectors[name] = [
                bool(ok_matrix.loc[i, choices[i]]) for i in self.instances
            ]
        # Keep unrounded means for ranking; rounding must not change the winner.
        names = sorted(
            selector_perf_vectors,
            key=lambda n: float(np.mean(selector_perf_vectors[n])),
            reverse=self.maximize,
        )
        sorted_stats = []
        for rank, name in enumerate(names, 1):
            solved_count = sum(solved_vectors[name])
            sorted_stats.append(
                {
                    "name": name,
                    "rank": rank,
                    "par_score": round(float(np.mean(selector_perf_vectors[name])), 3),
                    "solved_count": solved_count,
                    "solved_percentage": round(
                        100 * solved_count / len(self.instances), 2
                    ),
                    "timeouts_count": len(self.instances) - solved_count,
                    "timeouts_percentage": round(
                        100
                        * (len(self.instances) - solved_count)
                        / len(self.instances),
                        2,
                    ),
                }
            )
        top_name = next(name for name in names if name in candidate_cols)
        significance_results = []
        if run_significance_tests:
            tester = PermutationTester()
            for other in names:
                if other in (top_name, "VBS"):
                    continue
                result = tester.run_test(
                    selector_perf_vectors[top_name],
                    selector_perf_vectors[other],
                    name1=top_name,
                    name2=other,
                    maximize=self.maximize,
                    permutations=5000,
                )
                significance_results.append(
                    {"comparison": f"{top_name} vs {other}", **result}
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
            "feature_costs_included": include_feature_costs
            and self.objective == "runtime",
            "feature_costs_available": self.scenario.feature_cost_data is not None,
            "input_mode": input_mode,
            "score_direction": "maximize" if self.maximize else "minimize",
            "rankings": sorted_stats,
            "top_selector": top_name,
            "significance_tests": significance_results,
            "artifacts": artifacts,
        }
