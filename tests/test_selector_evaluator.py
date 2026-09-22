"""Unit tests for SelectorEvaluator, exact PAR10 scores, and baselines."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.tools.selector_evaluator import SelectorEvaluator


@pytest.fixture
def predictions_csv(synthetic_data: dict[str, Any], tmp_path: Path) -> Path:
    """Creates a predictions CSV with an oracle selector and a baseline selector."""
    instances = synthetic_data["instances"]
    pred_df = pd.DataFrame(
        {
            "perfect_selector": ["algo_a", "algo_b", "algo_c", "algo_b", "algo_a"],
            "static_sbs_selector": ["algo_b", "algo_b", "algo_b", "algo_b", "algo_b"],
            "poor_selector": ["algo_c", "algo_a", "algo_b", "algo_c", "algo_c"],
        },
        index=instances,
    )
    pred_path = tmp_path / "predictions.csv"
    pred_df.to_csv(pred_path)
    return pred_path


def test_selector_evaluator_initialization(mini_scenario_dir: Path):
    """SelectorEvaluator must successfully initialize and extract scenario metadata."""
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)
    assert evaluator.cutoff == 10.0
    assert evaluator.objective == "runtime"
    assert evaluator.maximize is False
    assert len(evaluator.instances) == 5
    assert set(evaluator.algorithms) == {"algo_a", "algo_b", "algo_c"}


def test_vbs_and_sbs_exact_baselines(mini_scenario_dir: Path, predictions_csv: Path):
    """Evaluates exact ground truth values for VBS and SBS."""
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)
    res = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=False,
        run_significance_tests=False,
    )

    stats = {s["name"]: s for s in res["rankings"]}
    # VBS mean across [1.0, 3.0, 2.0, 5.0, 2.0] = 13.0 / 5 = 2.6
    assert stats["VBS"]["par_score"] == pytest.approx(2.6, rel=1e-3)
    assert stats["VBS"]["timeouts_count"] == 0

    # SBS is algo_b: [2.0, 3.0, 6.0, 5.0, 4.0] = 20.0 / 5 = 4.0
    sbs_key = [k for k in stats if k.startswith("SBS")][0]
    assert "algo_b" in sbs_key
    assert stats[sbs_key]["par_score"] == pytest.approx(4.0, rel=1e-3)
    assert stats[sbs_key]["timeouts_count"] == 0


def test_selector_evaluation_with_and_without_feature_costs(
    mini_scenario_dir: Path, predictions_csv: Path
):
    """Verifies selector PAR10 scores and the impact of feature extraction costs."""
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)

    # 1. Without feature costs
    res_no_cost = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=False,
        run_significance_tests=False,
    )
    selectors_no_cost = {s["name"]: s for s in res_no_cost["rankings"]}
    # perfect_selector matches VBS exactly (2.6)
    assert selectors_no_cost["perfect_selector"]["par_score"] == pytest.approx(2.6, rel=1e-3)
    # static_sbs_selector matches SBS exactly (4.0)
    assert selectors_no_cost["static_sbs_selector"]["par_score"] == pytest.approx(4.0, rel=1e-3)
    # poor_selector has 2 timeouts on inst_4 and inst_5: 5+8+6+100+100 = 219 / 5 = 43.8
    assert selectors_no_cost["poor_selector"]["par_score"] == pytest.approx(43.8, rel=1e-3)
    assert selectors_no_cost["poor_selector"]["timeouts_count"] == 2

    # 2. With feature costs (feat_1 cost 0.05 + feat_2 cost 0.15 = 0.20 per instance)
    res_cost = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=True,
        run_significance_tests=False,
    )
    selectors_cost = {s["name"]: s for s in res_cost["rankings"]}
    # 2.6 + 0.20 = 2.80
    assert selectors_cost["perfect_selector"]["par_score"] == pytest.approx(2.8, rel=1e-3)


def test_selector_rankings_and_winner(mini_scenario_dir: Path, predictions_csv: Path, tmp_path: Path):
    """Winning selector must be ranked first and artifacts created on disk."""
    out_dir = tmp_path / "eval_out"
    evaluator = SelectorEvaluator(mini_scenario_dir, par_factor=10.0)
    res = evaluator.evaluate(
        predictions_csv_path=predictions_csv,
        include_feature_costs=False,
        run_significance_tests=True,
        output_dir=out_dir,
    )

    ranking = res["rankings"]
    # VBS is ranked #1 (score 2.6), perfect_selector is #2 (score 2.6) or #1 among selectors
    assert res["top_selector"] == "perfect_selector"
    assert len(res["significance_tests"]) >= 1

    # Check artifacts
    assert len(res["artifacts"]) >= 2
    for art in res["artifacts"]:
        assert Path(art["path"]).is_file()


def test_evaluate_missing_predictions_csv_raises(mini_scenario_dir: Path, tmp_path: Path):
    """Missing predictions CSV file must raise FileNotFoundError."""
    evaluator = SelectorEvaluator(mini_scenario_dir)
    with pytest.raises(FileNotFoundError, match="Predictions CSV not found"):
        evaluator.evaluate(tmp_path / "missing_preds.csv")
