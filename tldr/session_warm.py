"""
P1: Auto Background Warming for session start.

This module provides utilities to:
- Detect if the call graph cache is stale
- Count source files to determine project size
- Generate human-readable freshness messages
- Trigger background warming when appropriate
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _get_subprocess_detach_kwargs():
    """Get platform-specific kwargs for detaching subprocess."""
    if os.name == "nt":  # Windows
        create_new_pg = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": create_new_pg}
    else:  # Unix (Mac/Linux)
        return {"start_new_session": True}


# Default max age for cache in hours
DEFAULT_MAX_AGE_HOURS = 24

# Default max files threshold for auto-warming
DEFAULT_MAX_FILES = 500

# Directories to skip when counting files
SKIP_DIRS = {
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".git",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    "egg-info",
    ".eggs",
}


def get_cache_path(project_path: Path) -> Path:
    """Get the path to the call graph cache file."""
    return project_path / ".tldr" / "cache" / "call_graph.json"


def get_cache_age(project_path: Path) -> Optional[float]:
    """
    Get the age of the cache in hours.

    Args:
        project_path: Path to project root

    Returns:
        Age in hours, or None if cache doesn't exist or is invalid
    """
    cache_file = get_cache_path(project_path)

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text())
        timestamp = data.get("timestamp")
        if timestamp is None:
            return None

        age_seconds = time.time() - timestamp
        return age_seconds / 3600  # Convert to hours

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def is_cache_stale(
    project_path: Path, max_age_hours: float = DEFAULT_MAX_AGE_HOURS
) -> bool:
    """
    Check if the call graph cache is stale.

    A cache is considered stale if:
    - It doesn't exist
    - It's older than max_age_hours
    - It has invalid JSON
    - It's missing the timestamp field

    Args:
        project_path: Path to project root
        max_age_hours: Maximum age in hours before cache is stale (default: 24)

    Returns:
        True if cache is stale, False if fresh
    """
    age = get_cache_age(project_path)

    if age is None:
        return True

    return age > max_age_hours


def count_source_files(
    project_path: Path,
    max_count: int = DEFAULT_MAX_FILES,
    extensions: Optional[set] = None,
) -> int:
    """
    Count source files in a project.

    Stops early when max_count is reached for performance.
    Skips common non-source directories (venv, node_modules, etc.)

    Args:
        project_path: Path to project root
        max_count: Stop counting at this number (default: 500)
        extensions: File extensions to count (default: {".py"})

    Returns:
        Number of source files found (capped at max_count)
    """
    if extensions is None:
        extensions = {".py"}
    ext_set: set[str] = extensions  # Type narrowing for nested function

    count = 0

    def should_skip_dir(dir_name: str) -> bool:
        """Check if directory should be skipped."""
        return dir_name in SKIP_DIRS or dir_name.endswith(".egg-info")

    def walk_dir(path: Path) -> int:
        """Recursively count files, returning early if max reached."""
        nonlocal count

        try:
            for item in path.iterdir():
                if count >= max_count:
                    return count

                if item.is_file() and item.suffix in ext_set:
                    count += 1
                elif item.is_dir() and not should_skip_dir(item.name):
                    walk_dir(item)

        except PermissionError:
            pass  # Skip directories we can't read

        return count

    walk_dir(project_path)
    return count


def get_cache_freshness_message(project_path: Path, warming: bool = False) -> str:
    """
    Generate a human-readable cache freshness message.

    Args:
        project_path: Path to project root
        warming: Whether background warming is in progress

    Returns:
        Human-readable message like "Call graph: fresh (4h ago)" or
        "Call graph: 2 days old (warming...)"
    """
    age = get_cache_age(project_path)

    if age is None:
        if warming:
            return "Call graph: no cache (warming...)"
        return "Call graph: not found"

    # Format age
    if age < 1:
        age_str = f"{int(age * 60)}m ago"
    elif age < 24:
        age_str = f"{int(age)}h ago"
    else:
        days = age / 24
        if days < 2:
            age_str = f"{int(age)}h ago"
        else:
            age_str = f"{int(days)} days old"

    # Determine freshness
    if age < DEFAULT_MAX_AGE_HOURS:
        status = "fresh"
        if warming:
            return f"Call graph: {status} ({age_str}) (warming...)"
        return f"Call graph: {status} ({age_str})"
    else:
        if warming:
            return f"Call graph: {age_str} (warming...)"
        return f"Call graph: {age_str}"


def maybe_warm_background(
    project_path: Path,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    max_files: int = DEFAULT_MAX_FILES,
    language: str = "python",
) -> bool:
    """
    Spawn background warming if cache is stale and project is small enough.

    Args:
        project_path: Path to project root
        max_age_hours: Maximum cache age before warming (default: 24)
        max_files: Maximum project size for auto-warming (default: 500)
        language: Language for call graph (default: "python")

    Returns:
        True if background warming was spawned, False otherwise
    """
    project_path = Path(project_path)

    # Validate path exists
    if not project_path.exists():
        return False

    # Check if cache is fresh
    if not is_cache_stale(project_path, max_age_hours):
        return False

    # Check project size
    file_count = count_source_files(project_path, max_count=max_files + 1)
    if file_count > max_files:
        return False

    # Spawn background process (cross-platform)
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tldr.cli",
            "warm",
            str(project_path.resolve()),
            "--background",
            "--lang",
            language,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_get_subprocess_detach_kwargs(),
    )

    return True
