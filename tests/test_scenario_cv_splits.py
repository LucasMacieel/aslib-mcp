import pandas as pd
import pytest

from src.tools.aslib_scenario import aslib_create_cv_splits, aslib_get_cv_split


def test_generated_cv_preserves_numeric_string_ids(mini_scenario_dir):
    import arff

    for path in mini_scenario_dir.glob("*.arff"):
        with path.open() as f:
            data = arff.load(f)
        for row in data["data"]:
            row[0] = f"00{row[0][-1]}"
        with path.open("w") as f:
            arff.dump(data, f)
    generated = aslib_create_cv_splits(str(mini_scenario_dir), n_folds=2, seed=1)
    result = aslib_get_cv_split(
        str(mini_scenario_dir), cv_csv_path=generated["artifacts"][0]["path"]
    )
    assert all(value.startswith("00") for value in result["test_instances"])


def test_generated_cv_artifact_drives_split(mini_scenario_dir, tmp_path):
    generated = aslib_create_cv_splits(str(mini_scenario_dir), n_folds=3, seed=7)
    csv_path = generated["artifacts"][0]["path"]
    assignments = pd.read_csv(csv_path, index_col=0)
    result = aslib_get_cv_split(
        str(mini_scenario_dir), fold_index=3, cv_csv_path=csv_path
    )
    assert set(result["test_instances"]) == set(
        assignments.index[assignments["fold"] == 3]
    )
    assert set(result["test_instances"]).isdisjoint(result["train_instances"])


def test_partial_cv_assignments_rejected(mini_scenario_dir, tmp_path):
    path = tmp_path / "partial.csv"
    pd.DataFrame({"fold": [1, 2]}, index=["inst_1", "inst_2"]).to_csv(path)
    with pytest.raises(ValueError, match="exactly once"):
        aslib_get_cv_split(str(mini_scenario_dir), cv_csv_path=str(path))
