"""Cache behavior tested against a real local Git upstream, without network."""

import subprocess

import pytest

from src.tools import scenario_repository as repository


@pytest.fixture
def upstream(tmp_path, monkeypatch, mini_scenario_dir):
    source = tmp_path / "upstream"
    import shutil

    shutil.copytree(mini_scenario_dir, source / "SYNTHETIC-SCENARIO")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(repository, "REPO_URL", source.as_uri())
    return source


def test_fresh_explicit_cache_is_cloned(upstream, tmp_path):
    cache = tmp_path / "cache"
    result = repository.fetch_scenario("SYNTHETIC-SCENARIO", cache_dir=cache)
    assert result["is_valid"]
    assert (cache / ".git").is_dir()


def test_failed_force_refresh_preserves_cache(upstream, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    repository.ensure_cache(cache)
    marker = cache / "local.txt"
    marker.write_text("preserve")
    subprocess.run(
        [
            "git",
            "-C",
            str(cache),
            "remote",
            "set-url",
            "origin",
            (tmp_path / "absent").as_uri(),
        ],
        check=True,
    )
    with pytest.raises(RuntimeError, match="clone"):
        repository.ensure_cache(cache, force_clone=True)
    assert marker.read_text() == "preserve"


def test_non_repository_directory_is_never_replaced(tmp_path, upstream):
    cache = tmp_path / "local-data"
    cache.mkdir()
    (cache / "keep.txt").write_text("data")
    with pytest.raises(ValueError, match="non-repository"):
        repository.ensure_cache(cache, force_clone=True)
    assert (cache / "keep.txt").read_text() == "data"


def test_metadata_counts_unique_instances(mini_scenario_dir, tmp_path):
    import arff

    path = mini_scenario_dir / "feature_values.arff"
    data = arff.load(path.open())
    repeated = data["data"][0].copy()
    repeated[1] = 2
    data["data"].append(repeated)
    with path.open("w") as f:
        arff.dump(data, f)
    result = repository.get_scenario_info(str(mini_scenario_dir))
    assert result["instances_count"] == 5
