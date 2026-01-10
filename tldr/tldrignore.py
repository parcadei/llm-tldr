"""TLDR ignore file handling (.tldrignore).

Provides gitignore-style pattern matching for excluding files from indexing.
Uses pathspec library for gitignore-compatible pattern matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathspec import PathSpec

# Default .tldrignore template
DEFAULT_TEMPLATE = """\
# TLDR ignore patterns (gitignore syntax)
# Auto-generated - review and customize for your project
# Docs: https://git-scm.com/docs/gitignore

# ===================
# Dependencies
# ===================
node_modules/
.venv/
venv/
env/
__pycache__/
.tox/
.nox/
.pytest_cache/
.mypy_cache/
.ruff_cache/
vendor/
Pods/

# ===================
# Build outputs
# ===================
dist/
build/
out/
target/
*.egg-info/
*.whl
*.pyc
*.pyo

# ===================
# Binary/large files
# ===================
*.so
*.dylib
*.dll
*.exe
*.bin
*.o
*.a
*.lib

# ===================
# IDE/editors
# ===================
.idea/
.vscode/
*.swp
*.swo
*~

# ===================
# Security (always exclude)
# ===================
.env
.env.*
*.pem
*.key
*.p12
*.pfx
credentials.*
secrets.*

# ===================
# Version control
# ===================
.git/
.hg/
.svn/

# ===================
# OS files
# ===================
.DS_Store
Thumbs.db

# ===================
# Project-specific
# Add your custom patterns below
# ===================
# large_test_fixtures/
# data/
"""


def load_ignore_patterns(project_dir: str | Path) -> "PathSpec":
    """Load ignore patterns from .tldrignore file.

    Args:
        project_dir: Root directory of the project

    Returns:
        PathSpec matcher for checking if files should be ignored
    """
    import pathspec

    project_path = Path(project_dir)
    tldrignore_path = project_path / ".tldrignore"

    patterns: list[str] = []

    if tldrignore_path.exists():
        content = tldrignore_path.read_text()
        patterns = content.splitlines()
    else:
        # Use defaults if no .tldrignore exists
        patterns = DEFAULT_TEMPLATE.splitlines()

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def ensure_tldrignore(project_dir: str | Path) -> tuple[bool, str]:
    """Ensure .tldrignore exists, creating with defaults if needed.

    Args:
        project_dir: Root directory of the project

    Returns:
        Tuple of (created: bool, message: str)
    """
    project_path = Path(project_dir)
    tldrignore_path = project_path / ".tldrignore"

    if tldrignore_path.exists():
        return False, f".tldrignore already exists at {tldrignore_path}"

    # Create with default template
    tldrignore_path.write_text(DEFAULT_TEMPLATE)

    return True, f"""Created .tldrignore with sensible defaults:
  - node_modules/, .venv/, __pycache__/
  - dist/, build/, *.egg-info/
  - Binary files (*.so, *.dll, *.whl)
  - Security files (.env, *.pem, *.key)

Review .tldrignore before indexing large codebases.
Edit to exclude vendor code, test fixtures, etc."""


def should_ignore(
    file_path: str | Path,
    project_dir: str | Path,
    spec: "PathSpec | None" = None,
) -> bool:
    """Check if a file should be ignored.

    Args:
        file_path: Path to check (absolute or relative)
        project_dir: Root directory of the project
        spec: Optional pre-loaded PathSpec (for efficiency in loops)

    Returns:
        True if file should be ignored, False otherwise
    """
    if spec is None:
        spec = load_ignore_patterns(project_dir)

    project_path = Path(project_dir)
    file_path = Path(file_path)

    # Make path relative to project for matching
    try:
        rel_path = file_path.relative_to(project_path)
    except ValueError:
        # File is not under project_dir, use as-is
        rel_path = file_path

    return spec.match_file(str(rel_path))


def filter_files(
    files: list[Path],
    project_dir: str | Path,
    respect_ignore: bool = True,
) -> list[Path]:
    """Filter a list of files, removing those matching .tldrignore patterns.

    Args:
        files: List of file paths to filter
        project_dir: Root directory of the project
        respect_ignore: If False, skip filtering (--no-ignore mode)

    Returns:
        Filtered list of files
    """
    if not respect_ignore:
        return files

    spec = load_ignore_patterns(project_dir)
    return [f for f in files if not should_ignore(f, project_dir, spec)]
