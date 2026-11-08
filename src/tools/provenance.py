"""Record the exact local scenario inputs used for analysis."""

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def scenario_provenance(scenario_dir: str | Path) -> dict[str, Any]:
    """Return input hashes and Git revision when available, including local changes."""
    directory = Path(scenario_dir).resolve()
    hashes = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and (
            path.suffix == ".arff" or path.name == "description.txt"
        ):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[path.name] = digest.hexdigest()
    revision = None
    modified = None
    try:
        revision = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        modified = bool(
            subprocess.run(
                ["git", "-C", str(directory), "status", "--porcelain", "--", "."],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "scenario_dir": str(directory),
        "git_revision": revision,
        "scenario_modified": modified,
        "sha256": hashes,
    }
