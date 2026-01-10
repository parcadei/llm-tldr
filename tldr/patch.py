"""True Incremental Updates (P4) - File-level graph patching.

This module provides O(1) per-file updates to the call graph, instead of
rebuilding the entire graph on every file edit.

Key functions:
- compute_file_hash(file_path) - SHA-1 hash for content-based deduplication
- extract_edges_from_file(file_path, lang) - Extract edges from single file
- patch_call_graph(graph, edited_file, project_root, lang) - Patch graph incrementally
- has_file_changed(file_path, cached_hash) - Check if file content changed

Usage:
    from tldr.patch import patch_call_graph, compute_file_hash, has_file_changed

    # Check if file changed
    if has_file_changed(file_path, cached_hash):
        # Patch the graph for just this file
        graph = patch_call_graph(graph, file_path, project_root, lang="python")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tldr.cross_file_calls import (
    ProjectCallGraph,
    _extract_file_calls,
    _extract_ts_file_calls,
    _extract_go_file_calls,
    _extract_rust_file_calls,
)


@dataclass(frozen=True)
class Edge:
    """Represents a call edge from one function to another.

    Attributes:
        from_file: Source file path (relative to project root)
        from_func: Source function name
        to_file: Target file path (relative to project root)
        to_func: Target function name
    """
    from_file: str
    from_func: str
    to_file: str
    to_func: str

    def to_tuple(self) -> tuple[str, str, str, str]:
        """Convert to the tuple format used by ProjectCallGraph."""
        return (self.from_file, self.from_func, self.to_file, self.to_func)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-1 hash of file content.

    Args:
        file_path: Absolute path to the file

    Returns:
        40-character hex string representing the SHA-1 hash

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)
    content = path.read_bytes()
    return hashlib.sha1(content).hexdigest()


def has_file_changed(file_path: str, cached_hash: str) -> bool:
    """Check if file content has changed from cached hash.

    Args:
        file_path: Absolute path to the file
        cached_hash: Previously computed SHA-1 hash

    Returns:
        True if file has changed (or doesn't exist), False otherwise
    """
    try:
        current_hash = compute_file_hash(file_path)
        return current_hash != cached_hash
    except (FileNotFoundError, IOError):
        # Missing or unreadable file is considered "changed"
        return True


def _can_parse_file(file_path: Path, lang: str) -> bool:
    """Check if a file can be parsed without syntax errors.

    Args:
        file_path: Path to the source file
        lang: Language - "python", "typescript", "go", or "rust"

    Returns:
        True if file parses successfully, False on syntax error
    """
    import ast

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IOError):
        return False

    if lang == "python":
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False
    elif lang in ("typescript", "go", "rust"):
        # For non-Python languages, we rely on the extractor's behavior
        # These use tree-sitter which is more tolerant of partial parses
        # Return True to attempt extraction (extractor will handle errors)
        return True
    else:
        return False


def extract_edges_from_file(
    file_path: str,
    lang: str = "python",
    project_root: Optional[str] = None
) -> Optional[List[Edge]]:
    """Extract call edges from a single source file.

    Args:
        file_path: Absolute path to the source file
        lang: Language - "python", "typescript", "go", or "rust"
        project_root: Optional project root for computing relative paths

    Returns:
        List of Edge objects representing intra-file calls, or None if
        extraction failed (e.g., syntax error). Empty list means file
        has no calls (valid state).
    """
    path = Path(file_path)

    # Pre-check: verify file can be parsed before extraction
    # This catches syntax errors that extractors silently swallow
    if not _can_parse_file(path, lang):
        return None

    if project_root:
        root = Path(project_root)
        try:
            rel_path = path.relative_to(root)
            file_name = str(rel_path)
        except ValueError:
            file_name = path.name
    else:
        file_name = path.name

    # Get the appropriate extractor based on language
    if lang == "python":
        extractor = _extract_file_calls
    elif lang == "typescript":
        extractor = _extract_ts_file_calls
    elif lang == "go":
        extractor = _extract_go_file_calls
    elif lang == "rust":
        extractor = _extract_rust_file_calls
    else:
        raise ValueError(f"Unsupported language: {lang}")

    try:
        # Use project root or file's parent as root
        root_path = Path(project_root) if project_root else path.parent
        calls_by_func = extractor(path, root_path)
    except Exception:
        # Return None on extraction failure (syntax error, etc.)
        # This signals to callers that they should preserve existing edges
        return None

    edges = []
    for caller_func, calls in calls_by_func.items():
        for call_type, call_target in calls:
            # Only include intra-file calls for now
            # Cross-file resolution requires the full function index
            if call_type == 'intra':
                edges.append(Edge(
                    from_file=file_name,
                    from_func=caller_func,
                    to_file=file_name,
                    to_func=call_target
                ))
            elif call_type == 'ref':
                # Function references (e.g., higher-order)
                edges.append(Edge(
                    from_file=file_name,
                    from_func=caller_func,
                    to_file=file_name,
                    to_func=call_target
                ))

    return edges


def patch_call_graph(
    graph: ProjectCallGraph,
    edited_file: str,
    project_root: str,
    lang: str = "python"
) -> ProjectCallGraph:
    """Incrementally update call graph for an edited file.

    This is the core incremental update algorithm:
    1. Extract new edges from the edited file FIRST
    2. Only if extraction succeeds, remove old edges from the graph
    3. Add new edges to the graph
    4. Return the updated graph

    If extraction fails (syntax error, etc.), the graph is left unchanged
    to preserve existing edges rather than losing all call information.

    Args:
        graph: Existing ProjectCallGraph to patch
        edited_file: Absolute path to the edited file
        project_root: Project root directory
        lang: Language - "python", "typescript", "go", or "rust"

    Returns:
        Updated ProjectCallGraph (modifies in place and returns same object)
    """
    edited_path = Path(edited_file)
    root_path = Path(project_root)

    # Compute relative path for matching
    try:
        rel_path = str(edited_path.relative_to(root_path))
    except ValueError:
        rel_path = edited_path.name

    # Step 1: Extract new edges FIRST (before removing old)
    # This ensures we don't lose edges if extraction fails
    new_edges = extract_edges_from_file(
        str(edited_file),
        lang=lang,
        project_root=project_root
    )

    # Step 2: Only remove old edges if extraction succeeded
    # None means extraction failed (syntax error, etc.) - preserve old edges
    if new_edges is not None:
        edges_to_remove = {e for e in graph.edges if e[0] == rel_path}
        graph._edges -= edges_to_remove

        # Step 3: Add new edges to the graph
        for edge in new_edges:
            graph.add_edge(edge.from_file, edge.from_func, edge.to_file, edge.to_func)

    return graph


def get_file_hash_cache(project_root: str) -> dict[str, str]:
    """Load cached file hashes from project cache directory.

    Args:
        project_root: Project root directory

    Returns:
        Dict mapping relative file paths to their SHA-1 hashes
    """
    import json
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_hashes.json"

    if not cache_path.exists():
        return {}

    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def save_file_hash_cache(project_root: str, cache: dict[str, str]) -> None:
    """Save file hash cache to project cache directory.

    Args:
        project_root: Project root directory
        cache: Dict mapping relative file paths to their SHA-1 hashes
    """
    import json
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_hashes.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def patch_dirty_files(
    graph: ProjectCallGraph,
    project_root: str,
    dirty_files: list[str],
    lang: str = "python"
) -> ProjectCallGraph:
    """Patch the graph for all dirty files.

    This is the main entry point for incremental updates when the dirty
    flag system reports changed files.

    Args:
        graph: Existing ProjectCallGraph to patch
        project_root: Project root directory
        dirty_files: List of relative file paths that changed
        lang: Language for all files

    Returns:
        Updated ProjectCallGraph
    """
    root_path = Path(project_root)

    for rel_file in dirty_files:
        abs_file = root_path / rel_file
        if abs_file.exists():
            graph = patch_call_graph(graph, str(abs_file), project_root, lang=lang)

    return graph
