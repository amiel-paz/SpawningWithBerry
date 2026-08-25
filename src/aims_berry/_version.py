"""Package and source-revision provenance."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

__version__ = "0.3.1"
__release_date__ = "2026-08-25"

# Build systems may replace these values or set the matching environment
# variables when producing an archive without its Git metadata.
__build_commit__: str | None = None
__build_date__: str | None = None


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _git(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


@lru_cache(maxsize=1)
def _version_info() -> dict[str, Any]:
    root = _repository_root()
    commit = os.environ.get("AIMS_BERRY_BUILD_COMMIT") or __build_commit__
    commit_date = os.environ.get("AIMS_BERRY_BUILD_DATE") or __build_date__
    dirty: bool | None = None
    source = "package"
    if root is not None:
        git_commit = _git(root, "rev-parse", "HEAD")
        git_date = _git(root, "show", "-s", "--format=%cI", "HEAD")
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        if git_commit is not None:
            commit = git_commit
            commit_date = git_date
            dirty = bool(status) if status is not None else None
            source = "git"

    return {
        "name": "aims-berry",
        "version": __version__,
        "release_date": __release_date__,
        "last_update": commit_date or __release_date__,
        "commit": commit,
        "dirty": dirty,
        "source": source,
        "python": sys.version.split()[0],
        "package_path": str(Path(__file__).resolve().parent),
    }


def version_info() -> dict[str, Any]:
    """Return the installed release and, when available, Git provenance."""

    return dict(_version_info())


def version_string() -> str:
    """Return a compact human-readable version and revision string."""

    info = version_info()
    if info["commit"]:
        revision = f"commit {info['commit'][:12]}"
    else:
        revision = f"release {info['release_date']}"
    if info["dirty"] is True:
        revision += ", dirty"
    return f"{info['name']} {info['version']} ({revision}; updated {info['last_update']})"
