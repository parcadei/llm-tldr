"""True Incremental Updates (P4) - File-level graph patching.

This module provides O(1) per-file updates to the call graph, instead of
rebuilding the entire graph on every file edit.

Key functions:
- compute_file_hash(file_path) - SHA-1 hash for content-based deduplication
- extract_edges_from_file(file_path, lang) - Extract edges from single file
- patch_call_graph(graph, edited_file, project_root, lang) - Patch graph incrementally
- has_file_changed(file_path, cached_hash) - Check if file content changed

Performance-optimized functions (recommended):
- FileInfo - Dataclass storing hash + mtime for fast change detection
- get_file_info(path, prev_mtime, prev_hash) - Get hash with mtime fast path
- has_file_changed_with_mtime(path, cached_info) - ~99.8% faster for unchanged files
- get_file_info_cache/save_file_info_cache - Cache with mtime support

Usage:
    from tldr.patch import (
        patch_call_graph, FileInfo,
        get_file_info_cache, save_file_info_cache,
        has_file_changed_with_mtime
    )

    # Load cache with mtime info
    cache = get_file_info_cache(project_root)

    # Check if file changed (uses mtime fast path)
    changed, new_info = has_file_changed_with_mtime(file_path, cache.get(rel_path))
    if changed:
        # Patch the graph for just this file
        graph = patch_call_graph(graph, file_path, project_root, lang="python")
        if new_info:
            cache[rel_path] = new_info

    # Save updated cache
    save_file_info_cache(project_root, cache)
"""

from __future__ import annotations

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

# Performance-optimized JSON with orjson fallback
try:
    import orjson

    def _json_dumps(obj: object) -> bytes:
        return orjson.dumps(obj)

    def _json_loads(data: bytes | str) -> object:
        return orjson.loads(data)
except ImportError:
    import json

    def _json_dumps(obj: object) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode()

    def _json_loads(data: bytes | str) -> object:
        return json.loads(data if isinstance(data, str) else data.decode())


# Performance-optimized hashing with blake3 fallback
# HASH_ALGORITHM is stored in cache to detect when algorithm changes
# (e.g., blake3 installed/uninstalled), triggering cache invalidation
try:
    import blake3

    HASH_ALGORITHM = "blake3"

    def _hash_bytes(data: bytes) -> str:
        return blake3.blake3(data).hexdigest()
except ImportError:
    import hashlib

    HASH_ALGORITHM = "sha1"

    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha1(data).hexdigest()


# Languages supported by the call graph extraction system
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"python", "typescript", "go", "rust"})


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


@dataclass
class FileInfo:
    """File metadata for change detection with mtime optimization.

    Attributes:
        hash: SHA-1 hash of file content (40-char hex string)
        mtime_ns: File modification time in nanoseconds (from stat.st_mtime_ns)
    """

    hash: str
    mtime_ns: int


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
    return _hash_bytes(content)


def get_file_info(
    file_path: str, prev_mtime_ns: Optional[int] = None, prev_hash: Optional[str] = None
) -> Optional[FileInfo]:
    """Get file hash with mtime-based fast path.

    Performance optimization: If mtime hasn't changed since the previous check,
    the file content cannot have changed, so we skip the expensive hash computation.
    This provides ~99.8% savings for unchanged files in typical workflows.

    Args:
        file_path: Absolute path to the file
        prev_mtime_ns: Previous modification time in nanoseconds (from prior FileInfo)
        prev_hash: Previous hash (required if prev_mtime_ns is provided)

    Returns:
        FileInfo with hash and mtime, or None if file doesn't exist/unreadable
    """
    path = Path(file_path)
    try:
        stat_result = path.stat()
        current_mtime_ns = stat_result.st_mtime_ns

        # Fast path: mtime unchanged means content unchanged
        if prev_mtime_ns is not None and prev_hash is not None:
            if current_mtime_ns == prev_mtime_ns:
                return FileInfo(hash=prev_hash, mtime_ns=current_mtime_ns)

        # Slow path: compute hash
        content = path.read_bytes()
        content_hash = _hash_bytes(content)
        return FileInfo(hash=content_hash, mtime_ns=current_mtime_ns)

    except (FileNotFoundError, IOError, OSError):
        return None


def has_file_changed(file_path: str, cached_hash: str) -> bool:
    """Check if file content has changed from cached hash.

    Note: For better performance, use has_file_changed_with_mtime() which
    avoids hash computation when mtime is unchanged.

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


def has_file_changed_with_mtime(
    file_path: str, cached_info: Optional[FileInfo]
) -> tuple[bool, Optional[FileInfo]]:
    """Check if file changed using mtime optimization.

    Fast path: If mtime matches cached value, file is unchanged (no hash needed).
    This provides ~99.8% savings for unchanged files.

    Args:
        file_path: Absolute path to the file
        cached_info: Previously stored FileInfo (hash + mtime), or None

    Returns:
        Tuple of (changed: bool, new_info: FileInfo or None)
        - (True, new_info) if file changed or was newly added
        - (False, cached_info) if file unchanged (mtime match)
        - (True, None) if file was deleted or unreadable
    """
    if cached_info is None:
        # No prior info - file is new, compute hash
        new_info = get_file_info(file_path)
        return (True, new_info) if new_info else (True, None)

    new_info = get_file_info(
        file_path, prev_mtime_ns=cached_info.mtime_ns, prev_hash=cached_info.hash
    )

    if new_info is None:
        # File deleted or unreadable
        return (True, None)

    # Compare hashes (note: if mtime matched, new_info.hash == cached_info.hash)
    changed = new_info.hash != cached_info.hash
    return (changed, new_info)


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
    file_path: str, lang: str = "python", project_root: Optional[str] = None
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

    Raises:
        ValueError: If lang is not a supported language
    """
    # Validate language early to fail fast on misconfiguration
    # This prevents silent failures where unsupported languages return None
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {lang!r}. "
            f"Supported languages: {sorted(SUPPORTED_LANGUAGES)}"
        )

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
    # Note: lang is already validated against SUPPORTED_LANGUAGES above,
    # but we keep this structure explicit for maintainability
    if lang == "python":
        extractor = _extract_file_calls
    elif lang == "typescript":
        extractor = _extract_ts_file_calls
    elif lang == "go":
        extractor = _extract_go_file_calls
    elif lang == "rust":
        extractor = _extract_rust_file_calls
    else:
        # This branch is only reachable if SUPPORTED_LANGUAGES is extended
        # without adding a corresponding extractor - a programming error
        raise ValueError(
            f"Language {lang!r} is in SUPPORTED_LANGUAGES but has no extractor. "
            f"This is a bug - please add an extractor for {lang!r}."
        )

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
            if call_type == "intra":
                edges.append(
                    Edge(
                        from_file=file_name,
                        from_func=caller_func,
                        to_file=file_name,
                        to_func=call_target,
                    )
                )
            elif call_type == "ref":
                # Function references (e.g., higher-order)
                edges.append(
                    Edge(
                        from_file=file_name,
                        from_func=caller_func,
                        to_file=file_name,
                        to_func=call_target,
                    )
                )

    return edges


def patch_call_graph(
    graph: ProjectCallGraph, edited_file: str, project_root: str, lang: str = "python"
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
        str(edited_file), lang=lang, project_root=project_root
    )

    # Step 2: Only remove old edges if extraction succeeded
    # None means extraction failed (syntax error, etc.) - preserve old edges
    if new_edges is not None:
        # Use O(1) indexed removal instead of O(E) scan
        # The remove_edges_for_file method uses a secondary index by source file
        graph.remove_edges_for_file(rel_path)

        # Step 3: Add new edges to the graph
        for edge in new_edges:
            graph.add_edge(edge.from_file, edge.from_func, edge.to_file, edge.to_func)

    return graph


def get_file_hash_cache(project_root: str) -> dict[str, str]:
    """Load cached file hashes from project cache directory.

    DEPRECATED: Use get_file_info_cache() for mtime optimization.

    Args:
        project_root: Project root directory

    Returns:
        Dict mapping relative file paths to their SHA-1 hashes
    """
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_hashes.json"

    if not cache_path.exists():
        return {}

    try:
        data = _json_loads(cache_path.read_bytes())
        # Validate JSON structure matches expected dict[str, str]
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except (ValueError, IOError):
        return {}


def save_file_hash_cache(project_root: str, cache: dict[str, str]) -> None:
    """Save file hash cache to project cache directory.

    DEPRECATED: Use save_file_info_cache() for mtime optimization.

    Args:
        project_root: Project root directory
        cache: Dict mapping relative file paths to their SHA-1 hashes
    """
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_hashes.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(_json_dumps(cache))


def get_file_info_cache(project_root: str) -> dict[str, FileInfo]:
    """Load cached file info (hash + mtime) from project cache directory.

    This is the recommended cache function as it enables mtime-based
    fast path for change detection (~99.8% savings for unchanged files).

    If the cache was created with a different hash algorithm (e.g., sha1 vs blake3),
    the cache is invalidated and an empty dict is returned to force re-hashing.

    Args:
        project_root: Project root directory

    Returns:
        Dict mapping relative file paths to FileInfo objects
    """
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_info.json"

    if not cache_path.exists():
        return {}

    try:
        raw_data = _json_loads(cache_path.read_bytes())
        # Validate JSON structure is a dict before calling .items()
        if not isinstance(raw_data, dict):
            return {}

        # Check hash algorithm compatibility - invalidate cache if algorithm changed
        # This prevents stale cache hits when switching between blake3 and sha1
        cached_algorithm = raw_data.get("_algorithm")
        if cached_algorithm != HASH_ALGORITHM:
            return {}  # Force re-hash with current algorithm

        # Convert raw dict entries to FileInfo objects
        result: dict[str, FileInfo] = {}
        for path, info in raw_data.items():
            # Skip metadata keys (start with underscore)
            if path.startswith("_"):
                continue
            if isinstance(path, str) and isinstance(info, dict):
                hash_val = info.get("hash")
                mtime_val = info.get("mtime_ns")
                if isinstance(hash_val, str) and isinstance(mtime_val, int):
                    result[path] = FileInfo(hash=hash_val, mtime_ns=mtime_val)
        return result
    except (ValueError, IOError, KeyError, TypeError):
        return {}


def save_file_info_cache(project_root: str, cache: dict[str, FileInfo]) -> None:
    """Save file info cache (hash + mtime) to project cache directory.

    Includes the hash algorithm identifier so cache can be invalidated
    when the algorithm changes (e.g., blake3 installed/uninstalled).

    Args:
        project_root: Project root directory
        cache: Dict mapping relative file paths to FileInfo objects
    """
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_info.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert FileInfo objects to dicts for JSON serialization
    # Include algorithm metadata for cache compatibility check
    raw_data: dict[str, object] = {
        "_algorithm": HASH_ALGORITHM,  # Metadata for cache compatibility check
    }
    raw_data.update(
        {
            path: {"hash": info.hash, "mtime_ns": info.mtime_ns}
            for path, info in cache.items()
        }
    )
    cache_path.write_bytes(_json_dumps(raw_data))


def patch_dirty_files(
    graph: ProjectCallGraph,
    project_root: str,
    dirty_files: list[str],
    lang: str = "python",
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
