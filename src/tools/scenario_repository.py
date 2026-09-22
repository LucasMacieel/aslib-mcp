"""ASlib Scenario Repository and Cache Manager.

Integrates the official ASlib benchmark scenario repository (coseal/aslib_data)
into the ASlib MCP server. Provides shallow-clone local caching, high-speed
metadata indexing, scenario discovery, and transparent Scenario ID resolution.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aslib_data"
ENV_CACHE_DIR_VAR = "ASLIB_DATA_DIR"
REPO_URL = "https://github.com/coseal/aslib_data.git"
INDEX_FILE_NAME = "_aslib_index.json"

KNOWN_DOMAINS: dict[str, str] = {
    "SAT": "SAT",
    "GLUHACK": "SAT",
    "CSP": "CSP",
    "PROTEUS": "CSP",
    "QBF": "QBF",
    "MAXSAT": "MAXSAT",
    "MIP": "MIP",
    "TSP": "TSP",
    "TTP": "TTP",
    "ASP": "ASP",
    "BNSL": "BNSL",
    "CPMP": "CPMP",
    "IPC": "IPC",
    "OPENML": "AutoML",
    "GRAPHS": "Graph",
}


def get_cache_dir() -> Path:
    """Return the resolved local cache directory for ASlib scenarios."""
    env_dir = os.environ.get(ENV_CACHE_DIR_VAR)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_CACHE_DIR.expanduser().resolve()


def is_cache_initialized(cache_dir: Path | None = None) -> bool:
    """Check if the scenario cache directory exists and contains at least one scenario."""
    target = cache_dir or get_cache_dir()
    if not target.is_dir():
        return False
    # Check if there is at least one folder with description.txt
    for item in target.iterdir():
        if (
            item.is_dir()
            and not item.name.startswith(".")
            and (item / "description.txt").is_file()
        ):
            return True
    return False


def ensure_cache(cache_dir: Path | None = None, force_clone: bool = False) -> Path:
    """Ensure the ASlib scenario cache is initialized via shallow git clone.

    If cache exists and is valid, returns the cache directory immediately.
    Raises RuntimeError if cloning fails due to network or missing git.
    """
    target = cache_dir or get_cache_dir()
    if not force_clone and is_cache_initialized(target):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if force_clone and target.exists():
        shutil.rmtree(target)

    return target


def sync_repository(
    cache_dir: Path | None = None, force: bool = False
) -> dict[str, Any]:
    """Synchronize the local Scenario Cache with upstream coseal/aslib_data."""
    target = cache_dir or get_cache_dir()
    if not is_cache_initialized(target) or force:
        ensure_cache(target, force_clone=force)
        indexed = scan_and_index_scenarios(target, force_reindex=True)
        return {
            "status": "initialized",
            "cache_dir": str(target),
            "total_scenarios": len(indexed),
            "message": f"Successfully cloned and indexed {len(indexed)} scenarios into {target}",
        }

    git_dir = target / ".git"
    if not git_dir.is_dir():
        # Directory exists but not a git clone; re-index only
        indexed = scan_and_index_scenarios(target, force_reindex=True)
        return {
            "status": "reindexed",
            "cache_dir": str(target),
            "total_scenarios": len(indexed),
            "message": f"Scenario cache at {target} is not a git repository. Reindexed {len(indexed)} scenarios.",
        }

    try:
        cmd = ["git", "-C", str(target), "pull", "--ff-only"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pull_output = res.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return {
            "status": "sync_failed",
            "cache_dir": str(target),
            "error": str(exc),
            "message": f"Git pull failed: {exc}. Operating with existing cached scenarios.",
        }

    indexed = scan_and_index_scenarios(target, force_reindex=True)
    return {
        "status": "synchronized",
        "cache_dir": str(target),
        "pull_output": pull_output,
        "total_scenarios": len(indexed),
        "message": f"Cache synchronized. {len(indexed)} scenarios indexed.",
    }


def _infer_domain(scenario_id: str, desc: dict[str, Any]) -> str:
    """Infer the problem domain category from scenario ID or description."""
    upper_id = scenario_id.upper()
    for prefix, domain in KNOWN_DOMAINS.items():
        if upper_id.startswith(prefix):
            return domain
    return desc.get("problem_type") or "General"


def _count_instances_fast(feature_file: Path) -> int | None:
    """Quickly count instances from feature_values.arff by counting lines after @data."""
    if not feature_file.is_file():
        return None
    try:
        count = 0
        in_data = False
        with open(feature_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                if in_data:
                    count += 1
                elif line.lower() == "@data":
                    in_data = True
        return count if in_data else None
    except OSError:
        return None


def scan_and_index_scenarios(
    cache_dir: Path | None = None, force_reindex: bool = False
) -> dict[str, dict[str, Any]]:
    """Scan and index all scenarios in the cache directory, storing in _aslib_index.json."""
    target = cache_dir or ensure_cache()
    index_path = target / INDEX_FILE_NAME

    if not force_reindex and index_path.is_file():
        with (
            contextlib.suppress(OSError, json.JSONDecodeError),
            open(index_path, "r", encoding="utf-8") as f,
        ):
            data = json.load(f)
            if isinstance(data, dict) and "scenarios" in data:
                return data["scenarios"]

    scenarios: dict[str, dict[str, Any]] = {}
    for item in sorted(target.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        desc_file = item / "description.txt"
        if not desc_file.is_file():
            continue

        try:
            with open(desc_file, "r", encoding="utf-8", errors="replace") as f:
                desc = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            desc = {}

        scenario_id = str(desc.get("scenario_id") or item.name)
        domain = _infer_domain(scenario_id, desc)
        perf_measures = desc.get("performance_measures") or []
        if isinstance(perf_measures, str):
            perf_measures = [perf_measures]
        perf_types = desc.get("performance_type") or []
        if isinstance(perf_types, str):
            perf_types = [perf_types]

        algos = desc.get("metainfo_algorithms") or desc.get("algorithms") or []
        num_algos = len(algos) if isinstance(algos, (list, dict)) else 0

        det_feats = desc.get("features_deterministic") or []
        stoch_feats = desc.get("features_stochastic") or []
        num_feats = (len(det_feats) if isinstance(det_feats, list) else 0) + (
            len(stoch_feats) if isinstance(stoch_feats, list) else 0
        )

        num_steps = desc.get("number_of_feature_steps")
        if (
            num_steps is None
            and "feature_steps" in desc
            and isinstance(desc["feature_steps"], dict)
        ):
            num_steps = len(desc["feature_steps"])

        feat_file = item / "feature_values.arff"
        num_instances = _count_instances_fast(feat_file)

        has_runs = (item / "algorithm_runs.arff").is_file()
        has_features = feat_file.is_file()
        has_runstatus = (item / "feature_runstatus.arff").is_file()
        has_cv = (item / "cv.arff").is_file()
        has_costs = (item / "feature_costs.arff").is_file()
        has_citation = (item / "citation.bib").is_file()
        is_valid = has_runs and has_features and has_runstatus

        scenarios[scenario_id] = {
            "scenario_id": scenario_id,
            "domain": domain,
            "performance_measures": perf_measures,
            "performance_type": perf_types,
            "algorithm_cutoff_time": desc.get("algorithm_cutoff_time"),
            "algorithm_cutoff_memory": desc.get("algorithm_cutoff_memory"),
            "maximize": desc.get("maximize", [False]),
            "num_algorithms": num_algos,
            "num_features": num_feats,
            "num_feature_steps": num_steps,
            "num_instances": num_instances,
            "has_cv": has_cv,
            "has_costs": has_costs,
            "has_citation": has_citation,
            "is_valid": is_valid,
            "scenario_dir": str(item.resolve()),
        }

    with (
        contextlib.suppress(OSError, TypeError),
        open(index_path, "w", encoding="utf-8") as f,
    ):
        json.dump({"version": "1.0", "scenarios": scenarios}, f, indent=2)

    return scenarios


def list_scenarios(
    domain: str | None = None,
    objective: str | None = None,
    search: str | None = None,
    limit: int = 50,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Filter and list benchmark scenarios in the Scenario Cache."""
    target = cache_dir or ensure_cache()
    indexed = scan_and_index_scenarios(target)

    results: list[dict[str, Any]] = []
    for s in indexed.values():
        if domain:
            d_norm = domain.strip().lower()
            if d_norm != s["domain"].lower() and d_norm not in s["scenario_id"].lower():
                continue
        if objective:
            obj_norm = objective.strip().lower()
            measures = [m.lower() for m in s["performance_measures"]]
            types = [t.lower() for t in s["performance_type"]]
            if obj_norm not in measures and obj_norm not in types:
                continue
        if search:
            q = search.strip().lower()
            if q not in s["scenario_id"].lower() and q not in s["domain"].lower():
                continue
        results.append(s)

    return {
        "total_scenarios": len(indexed),
        "matched_scenarios": len(results),
        "returned_count": min(len(results), limit),
        "cache_dir": str(target),
        "scenarios": results[:limit],
    }


def resolve_scenario_dir(
    scenario_dir_or_id: str, cache_dir: Path | None = None
) -> Path:
    """Resolve a directory path or Scenario ID into an absolute Path to a scenario directory.

    1. If scenario_dir_or_id is an existing local directory, returns it directly.
    2. Otherwise, matches against Scenario IDs in the Scenario Cache (exact or case-insensitive).
    3. If not found, raises FileNotFoundError with fuzzy suggestions.
    """
    cleaned = scenario_dir_or_id.strip()
    direct_path = Path(cleaned).expanduser().resolve()
    if direct_path.is_dir():
        return direct_path

    target = cache_dir or ensure_cache()
    indexed = scan_and_index_scenarios(target)

    # 1. Exact match against indexed scenario IDs
    if cleaned in indexed:
        return Path(indexed[cleaned]["scenario_dir"])

    # 2. Case-insensitive match
    cleaned_lower = cleaned.lower()
    for sid, sinfo in indexed.items():
        if sid.lower() == cleaned_lower:
            return Path(sinfo["scenario_dir"])

    # 3. Direct directory check inside cache_dir
    candidate = target / cleaned
    if candidate.is_dir() and (candidate / "description.txt").is_file():
        return candidate.resolve()

    # 4. Fuzzy match suggestion
    all_sids = list(indexed.keys())
    close_matches = difflib.get_close_matches(cleaned, all_sids, n=3, cutoff=0.35)
    if close_matches:
        suggestions = ", ".join(f"'{m}'" for m in close_matches)
        raise FileNotFoundError(
            f"Scenario '{scenario_dir_or_id}' not found locally or in ASlib cache ({target}). "
            f"Did you mean: {suggestions}? Call 'aslib_list_scenarios' to see all available benchmark scenarios."
        )

    raise FileNotFoundError(
        f"Scenario '{scenario_dir_or_id}' not found locally or in ASlib cache ({target}). "
        f"Call 'aslib_list_scenarios' to browse all {len(all_sids)} available benchmark scenarios."
    )


def get_scenario_info(
    scenario_id: str, cache_dir: Path | None = None
) -> dict[str, Any]:
    """Retrieve full specification metadata for a scenario without loading heavy ARFF matrices."""
    scen_path = resolve_scenario_dir(scenario_id, cache_dir)
    desc_file = scen_path / "description.txt"

    desc: dict[str, Any] = {}
    if desc_file.is_file():
        try:
            with open(desc_file, "r", encoding="utf-8", errors="replace") as f:
                desc = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            desc = {"_yaml_error": str(exc)}

    citation_file = scen_path / "citation.bib"
    citation_text = ""
    if citation_file.is_file():
        with contextlib.suppress(OSError):
            citation_text = citation_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()

    readme_file = scen_path / "readme.txt"
    readme_text = ""
    if readme_file.is_file():
        with contextlib.suppress(OSError):
            readme_text = readme_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()

    feat_file = scen_path / "feature_values.arff"
    instances_count = _count_instances_fast(feat_file)

    files_present = [f.name for f in scen_path.iterdir() if f.is_file()]

    return {
        "scenario_id": desc.get("scenario_id") or scen_path.name,
        "scenario_dir": str(scen_path),
        "domain": _infer_domain(scen_path.name, desc),
        "instances_count": instances_count,
        "algorithms": list(desc.get("metainfo_algorithms", {}).keys())
        or desc.get("algorithms")
        or [],
        "performance_measures": desc.get("performance_measures") or [],
        "performance_type": desc.get("performance_type") or [],
        "algorithm_cutoff_time": desc.get("algorithm_cutoff_time"),
        "algorithm_cutoff_memory": desc.get("algorithm_cutoff_memory"),
        "features_cutoff_time": desc.get("features_cutoff_time"),
        "features_cutoff_memory": desc.get("features_cutoff_memory"),
        "maximize": desc.get("maximize", [False]),
        "feature_steps": desc.get("feature_steps") or {},
        "files_present": sorted(files_present),
        "has_cv": "cv.arff" in files_present,
        "has_costs": "feature_costs.arff" in files_present,
        "citation": citation_text,
        "readme": readme_text,
    }


def fetch_scenario(
    scenario_id: str,
    force_update: bool = False,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Ensure a scenario is present and validated in the local cache, returning its path and status."""
    target = cache_dir or ensure_cache(force_clone=force_update)
    scen_path = resolve_scenario_dir(scenario_id, target)

    required_files = [
        "description.txt",
        "algorithm_runs.arff",
        "feature_values.arff",
        "feature_runstatus.arff",
    ]
    missing = [f for f in required_files if not (scen_path / f).is_file()]
    is_valid = len(missing) == 0

    return {
        "scenario_id": scenario_id,
        "scenario_dir": str(scen_path),
        "is_valid": is_valid,
        "missing_required_files": missing,
        "files_present": sorted([f.name for f in scen_path.iterdir() if f.is_file()]),
        "message": (
            f"Scenario '{scenario_id}' is ready at {scen_path}"
            if is_valid
            else f"Scenario '{scenario_id}' is missing required files: {missing}"
        ),
    }
