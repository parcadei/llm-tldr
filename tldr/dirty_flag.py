"""Dirty flag system for lazy cache invalidation (P3).

This module implements a dirty flag mechanism to track when the cache
needs to be rebuilt due to file edits. Instead of rebuilding immediately
on every edit, we mark files as dirty and rebuild lazily on query.

Usage:
    from tldr.dirty_flag import mark_dirty, is_dirty, clear_dirty

    # After editing a file
    mark_dirty(project_path, "src/auth.py")

    # Before running queries that need fresh data
    if is_dirty(project_path):
        rebuild_cache(project_path)  # Your rebuild logic
        clear_dirty(project_path)

    # Check how many files changed (useful for threshold tuning)
    count = get_dirty_count(project_path)

    # Batch marking for efficiency (single disk write)
    mark_dirty_batch(project_path, ["src/a.py", "src/b.py", "src/c.py"])
"""

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Set, Union


# Path to dirty flag file relative to project root
DIRTY_FILE = ".tldr/cache/dirty.json"


def _get_dirty_path(project_path: Union[str, Path]) -> Path:
    """Get the full path to the dirty flag file."""
    return Path(project_path) / DIRTY_FILE


def _normalize_file_path(file_path: str) -> str:
    """Normalize a file path for consistent storage.

    - Converts backslashes to forward slashes
    """
    return file_path.replace("\\", "/")


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mark_dirty_impl(
    project_path: Union[str, Path], files_to_add: Iterable[str]
) -> None:
    """Internal implementation for marking files dirty.

    Uses set for O(1) membership checks and supports batch operations.
    File locking prevents TOCTOU race conditions in concurrent access.

    Args:
        project_path: Root directory of the project
        files_to_add: Iterable of file paths to mark dirty
    """
    dirty_path = _get_dirty_path(project_path)
    normalized_files = {_normalize_file_path(f) for f in files_to_add}
    now = _get_timestamp()

    if not normalized_files:
        return  # Nothing to add

    # Ensure parent directories exist before opening file
    dirty_path.parent.mkdir(parents=True, exist_ok=True)

    # Use file locking for atomicity - prevents concurrent processes from
    # losing dirty markers due to read-modify-write race
    with open(dirty_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Read existing content
            f.seek(0)
            content = f.read()

            # Parse existing data or create new
            if content:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

            if data is None:
                data = {
                    "dirty_files": [],
                    "first_dirty_at": now,
                    "last_dirty_at": now,
                }

            # Convert to set for O(1) membership check, add new files
            existing_files: Set[str] = set(data["dirty_files"])
            new_files = normalized_files - existing_files

            if new_files:
                # Only write if there are actual new files
                data["dirty_files"] = list(existing_files | normalized_files)
                data["last_dirty_at"] = now

                # Write back atomically (truncate and rewrite)
                f.seek(0)
                f.truncate()
                f.write(json.dumps(data, separators=(',', ':')))
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def mark_dirty(project_path: Union[str, Path], edited_file: str) -> None:
    """Mark a file as dirty (needing cache rebuild).

    Creates or updates the dirty flag file with the edited file path.
    Multiple calls append to the list without duplicates.
    Uses file locking to prevent TOCTOU race conditions when multiple
    processes mark files dirty concurrently.

    Args:
        project_path: Root directory of the project
        edited_file: Relative path to the edited file
    """
    _mark_dirty_impl(project_path, [edited_file])


def mark_dirty_batch(
    project_path: Union[str, Path], edited_files: Iterable[str]
) -> None:
    """Mark multiple files as dirty with a single disk write.

    More efficient than calling mark_dirty() repeatedly when multiple
    files change at once (e.g., during a git checkout or bulk refactor).

    Args:
        project_path: Root directory of the project
        edited_files: Iterable of relative paths to edited files

    Example:
        mark_dirty_batch(project, ["src/a.py", "src/b.py", "src/c.py"])
    """
    _mark_dirty_impl(project_path, edited_files)


def is_dirty(project_path: Union[str, Path]) -> bool:
    """Check if the project has dirty files needing rebuild.

    Args:
        project_path: Root directory of the project

    Returns:
        True if dirty flag file exists, False otherwise
    """
    dirty_path = _get_dirty_path(project_path)
    return dirty_path.exists()


def get_dirty_files(project_path: Union[str, Path]) -> List[str]:
    """Get the list of dirty files.

    Args:
        project_path: Root directory of the project

    Returns:
        List of file paths that were edited since last rebuild.
        Empty list if no dirty flag exists.
    """
    dirty_path = _get_dirty_path(project_path)

    if not dirty_path.exists():
        return []

    try:
        data = json.loads(dirty_path.read_text())
        return data.get("dirty_files", [])
    except (json.JSONDecodeError, IOError):
        return []


def get_dirty_count(project_path: Union[str, Path]) -> int:
    """Get the count of dirty files.

    Useful for threshold-based decisions (e.g., full rebuild vs incremental).

    Args:
        project_path: Root directory of the project

    Returns:
        Number of dirty files, or 0 if no dirty flag exists.
    """
    return len(get_dirty_files(project_path))


def clear_dirty(project_path: Union[str, Path]) -> None:
    """Clear the dirty flag (after rebuild).

    Removes the dirty flag file. Safe to call even if file doesn't exist.

    Args:
        project_path: Root directory of the project
    """
    dirty_path = _get_dirty_path(project_path)

    try:
        dirty_path.unlink(missing_ok=True)
    except (OSError, IOError):
        # Handle any filesystem errors gracefully
        pass
