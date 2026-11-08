"""Tests for provenance."""



def test_provenance_hash_changes_with_input(mini_scenario_dir):
    from src.tools.provenance import scenario_provenance

    before = scenario_provenance(mini_scenario_dir)
    description = mini_scenario_dir / "description.txt"
    description.write_text(description.read_text() + "\n# edited\n")
    after = scenario_provenance(mini_scenario_dir)
    assert before["sha256"]["description.txt"] != after["sha256"]["description.txt"]
    assert before["git_revision"] is None
