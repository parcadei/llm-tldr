"""
Workspace configuration for monorepo scoping.

Provides:
- WorkspaceConfig dataclass for holding config
- load_workspace_config() to parse .claude/workspace.json
- should_include_path() to check if a path should be indexed
- filter_paths() to filter a list of paths by config
"""

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union


# Default exclude patterns for common non-source directories
DEFAULT_EXCLUDE_PATTERNS = [
    "**/node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/dist/**",
    "**/build/**",
]


@dataclass
class WorkspaceConfig:
    """Configuration for workspace-scoped indexing."""

    active_packages: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)


def load_workspace_config(project_path: Union[str, Path]) -> WorkspaceConfig:
    """
    Load workspace configuration from .claude/workspace.json.

    Args:
        project_path: Root directory of the project

    Returns:
        WorkspaceConfig with activePackages and excludePatterns.
        Returns defaults if file is missing or invalid.
    """
    project_path = Path(project_path)
    config_file = project_path / ".claude" / "workspace.json"

    if not config_file.exists():
        return WorkspaceConfig(
            active_packages=[],
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS.copy()
        )

    try:
        with open(config_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        # Invalid JSON or read error - return defaults
        return WorkspaceConfig(
            active_packages=[],
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS.copy()
        )

    # Extract fields with defaults for missing keys
    active_packages = data.get("activePackages", [])
    exclude_patterns = data.get("excludePatterns")

    # If excludePatterns is not specified, use defaults
    # If explicitly set to [], use empty list
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS.copy()

    return WorkspaceConfig(
        active_packages=active_packages,
        exclude_patterns=exclude_patterns
    )


def _normalize_path(path: str) -> str:
    """
    Normalize a path for consistent matching.

    - Converts backslashes to forward slashes
    - Removes leading ./
    - Removes trailing /
    """
    # Convert Windows-style backslashes to forward slashes
    path = path.replace("\\", "/")

    # Remove leading ./
    if path.startswith("./"):
        path = path[2:]

    # Remove trailing slash
    path = path.rstrip("/")

    return path


def _matches_any_pattern(path: str, patterns: List[str]) -> bool:
    """
    Check if path matches any of the glob patterns.

    Args:
        path: Path to check (should be normalized)
        patterns: List of glob patterns

    Returns:
        True if path matches any pattern
    """
    for pattern in patterns:
        # fnmatch doesn't handle ** properly for directory matching
        # We need to check if any part of the path matches
        if fnmatch.fnmatch(path, pattern):
            return True

        # Handle ** patterns by extracting the directory name to match
        if "**" in pattern:
            # Pattern like **/dirname/** or **/dirname/*
            # Extract the directory name between ** markers
            # e.g., **/node_modules/** -> node_modules
            # e.g., **/generated/** -> generated
            parts = pattern.split("/")
            for part in parts:
                if part and part != "**" and part != "*":
                    # This is a literal directory name to match
                    dir_name = part.rstrip("*")
                    if dir_name:
                        # Check if this directory name appears in the path
                        # Either at start: dirname/...
                        # Or in middle: .../dirname/...
                        if path.startswith(f"{dir_name}/"):
                            return True
                        if f"/{dir_name}/" in path:
                            return True
                        # Also match if path equals the directory
                        if path == dir_name:
                            return True

    return False


def _is_under_active_package(path: str, active_packages: List[str]) -> bool:
    """
    Check if path is under one of the active packages.

    Args:
        path: Normalized path to check
        active_packages: List of package prefixes

    Returns:
        True if path starts with any active package prefix
    """
    for pkg in active_packages:
        pkg_normalized = _normalize_path(pkg)
        # Path should start with the package path
        if path == pkg_normalized or path.startswith(pkg_normalized + "/"):
            return True
    return False


def should_include_path(path: str, config: WorkspaceConfig) -> bool:
    """
    Determine if a path should be included based on workspace config.

    Logic:
    1. If activePackages is non-empty, path must be under one of them
    2. Path must not match any excludePattern

    Args:
        path: Path to check (can be relative or have various formats)
        config: WorkspaceConfig instance

    Returns:
        True if path should be included in indexing
    """
    normalized_path = _normalize_path(path)

    # Check active packages filter (if specified)
    if config.active_packages:
        if not _is_under_active_package(normalized_path, config.active_packages):
            return False

    # Check exclude patterns
    if config.exclude_patterns:
        if _matches_any_pattern(normalized_path, config.exclude_patterns):
            return False

    return True


def filter_paths(paths: List[str], config: WorkspaceConfig) -> List[str]:
    """
    Filter a list of paths based on workspace configuration.

    Args:
        paths: List of paths to filter
        config: WorkspaceConfig instance

    Returns:
        List of paths that should be included
    """
    return [p for p in paths if should_include_path(p, config)]
