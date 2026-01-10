"""
TLDR Unified API - Token-efficient code context for LLMs.

Usage:
    from tldr.api import get_relevant_context

    context = get_relevant_context(
        project="/path/to/project",
        entry_point="ClassName.method_name",  # or "function_name"
        depth=2,
        language="python"
    )

    # Returns LLM-ready string with call graph, signatures, complexity
"""

import functools
import os
import re
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .ast_extractor import (
    CallGraphInfo,  # Re-exported for API consumers
    ClassInfo,  # Re-exported for API consumers
    FunctionInfo,
    ImportInfo,  # Re-exported for API consumers
    extract_file as _extract_file_impl,
)

# Re-export for public API
__all__ = [
    # Dataclasses from ast_extractor
    "CallGraphInfo",
    "ClassInfo",
    "FunctionInfo",
    "ImportInfo",
    # Main API functions
    "get_relevant_context",
    "get_imports",
    "get_intra_file_calls",
    "extract_file",
    "get_dfg_context",
    "get_pdg_context",
    "get_slice",
    "query",
    # Cross-file functions
    "build_project_call_graph",
    "scan_project_files",
    "build_function_index",
    # Project navigation functions
    "get_file_tree",
    "search",
    "Selection",
    "get_code_structure",
    # P5 #21: Content-hash deduplication
    "ContentHashedIndex",
]
from .cfg_extractor import (
    CFGBlock,  # Re-exported for type hints
    CFGEdge,  # Re-exported for type hints
    CFGInfo,
    extract_c_cfg,
    extract_cpp_cfg,
    extract_csharp_cfg,
    extract_elixir_cfg,
    extract_go_cfg,
    extract_java_cfg,
    extract_kotlin_cfg,
    extract_lua_cfg,
    extract_php_cfg,
    extract_python_cfg,
    extract_python_cfgs_batch,
    extract_ruby_cfg,
    extract_rust_cfg,
    extract_scala_cfg,
    extract_swift_cfg,
    extract_typescript_cfg,
)
from .dedup import ContentHashedIndex  # P5 #21: Content-hash deduplication
from .cross_file_calls import (
    build_project_call_graph,
    ProjectCallGraph,
)
from .cross_file_calls import (
    build_function_index as _build_function_index,
)
from .cross_file_calls import (
    parse_go_imports as _parse_go_imports,
)
from .cross_file_calls import (
    parse_imports as _parse_imports,
)
from .cross_file_calls import (
    parse_rust_imports as _parse_rust_imports,
)
from .cross_file_calls import (
    parse_ts_imports as _parse_ts_imports,
)
from .cross_file_calls import (
    parse_java_imports as _parse_java_imports,
)
from .cross_file_calls import (
    parse_c_imports as _parse_c_imports,
)
from .cross_file_calls import (
    parse_cpp_imports as _parse_cpp_imports,
)
from .cross_file_calls import (
    parse_ruby_imports as _parse_ruby_imports,
)
from .cross_file_calls import (
    parse_kotlin_imports as _parse_kotlin_imports,
)
from .cross_file_calls import (
    parse_scala_imports as _parse_scala_imports,
)
from .cross_file_calls import (
    parse_php_imports as _parse_php_imports,
)
from .cross_file_calls import (
    parse_swift_imports as _parse_swift_imports,
)
from .cross_file_calls import (
    parse_csharp_imports as _parse_csharp_imports,
)
from .cross_file_calls import (
    parse_lua_imports as _parse_lua_imports,
)
from .cross_file_calls import (
    parse_elixir_imports as _parse_elixir_imports,
)
from .cross_file_calls import (
    scan_project as _scan_project,
)
from .dfg_extractor import (
    DFGInfo,
    extract_c_dfg,
    extract_cpp_dfg,
    extract_csharp_dfg,
    extract_elixir_dfg,
    extract_go_dfg,
    extract_java_dfg,
    extract_kotlin_dfg,
    extract_lua_dfg,
    extract_php_dfg,
    extract_python_dfg,
    extract_ruby_dfg,
    extract_rust_dfg,
    extract_scala_dfg,
    extract_swift_dfg,
    extract_typescript_dfg,
)
from .hybrid_extractor import (
    HybridExtractor,  # Re-exported for API
)
from .pdg_extractor import (
    extract_pdg,
)

# Explicit exports for public API
__all__ = [
    # Layer 3: CFG types and functions
    "CFGBlock",
    "CFGEdge",
    "get_cfg_context",
    "get_cfg_blocks",
    "get_cfg_edges",
    # Layer 4: DFG functions
    "get_dfg_context",
    # Layer 5: PDG functions
    "get_pdg_context",
    "get_slice",
    # Main API
    "get_relevant_context",
    "query",
    "FunctionContext",
    "RelevantContext",
    # Cross-file functions
    "build_project_call_graph",
    "scan_project_files",
    "get_imports",
    "build_function_index",
    # Security exceptions
    "PathTraversalError",
]


# =============================================================================
# Security: Path Containment Validation
# =============================================================================


class PathTraversalError(ValueError):
    """Raised when a path attempts to escape its container via directory traversal.

    This is a security error indicating an attempted path traversal attack
    (e.g., using ../../../etc/passwd to escape the project directory).
    """

    pass


def _validate_path_containment(file_path: str, base_path: str | None = None) -> Path:
    """Validate that file_path doesn't escape base_path via traversal.

    Detects directory traversal attacks (../..) and symlink escapes.

    Args:
        file_path: The path to validate
        base_path: Optional container directory. If None, detects traversal
                   patterns that escape the apparent starting directory.

    Returns:
        Resolved Path object

    Raises:
        PathTraversalError: If path contains traversal or escapes base
        ValueError: If path is empty or whitespace-only
    """
    # Reject empty or whitespace-only paths
    if not file_path or not file_path.strip():
        raise ValueError("Path cannot be empty or whitespace-only")

    # Check for null bytes (path truncation attack)
    if "\x00" in file_path:
        raise ValueError("Path contains null byte")

    # Resolve the path (follows symlinks, normalizes ..)
    try:
        resolved = Path(file_path).resolve()
    except OSError as e:
        # Handle paths that are too long or have invalid characters
        raise ValueError(f"Invalid path: {e}")

    # Check for traversal patterns in original path
    if ".." in file_path:
        if base_path:
            # Explicit base path provided - enforce containment
            base = Path(base_path).resolve()
            try:
                if not resolved.is_relative_to(base):
                    raise PathTraversalError(
                        f"Path '{file_path}' escapes base directory '{base_path}' via traversal"
                    )
            except ValueError:
                raise PathTraversalError(
                    f"Path '{file_path}' escapes base directory '{base_path}'"
                )
        else:
            # No explicit base path - detect suspicious traversal patterns
            # A path like "/tmp/project/../outside/file.py" is suspicious because
            # the ".." effectively escapes from "project" into a sibling directory
            #
            # Strategy: Find directory components that are "entered" then immediately
            # "exited" via .. - this indicates intentional escape
            path_obj = Path(file_path)
            parts = list(path_obj.parts)

            # Look for pattern: <dir>/.. which indicates entering then leaving a directory
            # This is almost always indicative of traversal attack
            i = 0
            while i < len(parts) - 1:
                current = parts[i]
                next_part = parts[i + 1]

                # Skip root components like "/" or "C:\"
                if current in ("/", "\\") or (len(current) == 2 and current[1] == ":"):
                    i += 1
                    continue

                # If we have a real directory name followed by "..", that's traversal
                if current not in (".", "..") and next_part == "..":
                    raise PathTraversalError(
                        f"Path '{file_path}' contains directory traversal pattern '{current}/..'"
                    )
                i += 1

    # Check symlink targets if the path exists
    # Wrap filesystem operations in try/except to handle mocked/broken stat
    try:
        path_exists = resolved.exists()
    except (OSError, TypeError):
        # stat might be mocked or broken - skip symlink checks
        path_exists = False

    if path_exists:
        # Check if the resolved path is a symlink (readlink on the original)
        original_path = Path(file_path)
        try:
            is_symlink = original_path.is_symlink()
        except (OSError, TypeError):
            # lstat might be mocked or broken
            is_symlink = False

        if is_symlink:
            try:
                target = original_path.readlink()
                # Resolve the target relative to the symlink's parent
                abs_target = (original_path.parent / target).resolve()

                if base_path:
                    base = Path(base_path).resolve()
                    if not abs_target.is_relative_to(base):
                        raise PathTraversalError(
                            f"Symlink '{file_path}' points outside base directory '{base_path}'"
                        )
                else:
                    # No base path - check if symlink target escapes the symlink's directory
                    symlink_parent = original_path.parent.resolve()
                    if not abs_target.is_relative_to(symlink_parent):
                        raise PathTraversalError(
                            f"Symlink '{file_path}' points outside its containing directory"
                        )
            except OSError:
                # Can't read symlink - might be broken, let normal file ops handle it
                pass

    return resolved


def _resolve_source(source_or_path: str) -> tuple[str, str | None]:
    """
    Resolve source code from either source string or file path.

    Auto-detects whether the input is:
    1. A file path (exists on disk) -> reads and returns contents
    2. Source code string -> returns as-is

    Args:
        source_or_path: Either source code string or path to a file

    Returns:
        Tuple of (source_code, file_path_or_none)
        - If path: (file_contents, path)
        - If source: (source, None)

    Raises:
        PathTraversalError: If path contains directory traversal patterns
        ValueError: If path is empty or contains invalid characters
    """
    # Check if it looks like a file path and exists
    if len(source_or_path) < 500:  # Paths are typically short
        try:
            # Security: Validate path before accessing
            # PathTraversalError must propagate - don't catch it
            _validate_path_containment(source_or_path)

            path = Path(source_or_path)
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8"), str(path)
        except PathTraversalError:
            # Security error - must propagate
            raise
        except (OSError, ValueError):
            # Long strings can cause OSError on path.exists()
            # ValueError from empty/whitespace paths - treat as source code
            pass

    # Treat as source code
    return source_or_path, None


# =============================================================================
# Call Graph Caching (Performance Optimization)
# =============================================================================

# Module-level cache for call graphs: {(project_path, language): (mtime, graph)}
_call_graph_cache: dict[tuple[str, str], tuple[float, ProjectCallGraph]] = {}


def _get_project_mtime(project: Path, extensions: set[str]) -> float:
    """Get latest mtime of source files in project for cache invalidation.

    Only checks files with relevant extensions to avoid scanning irrelevant files.

    Args:
        project: Project root path
        extensions: Set of file extensions to check (e.g., {".py"})

    Returns:
        Latest mtime as float, or 0.0 if no files found or errors occur.
    """
    latest = 0.0
    try:
        for f in project.rglob("*"):
            if f.is_file() and f.suffix in extensions:
                # Skip hidden directories
                try:
                    rel = f.relative_to(project)
                    if any(p.startswith(".") for p in rel.parts):
                        continue
                except ValueError:
                    continue
                try:
                    mtime = f.stat().st_mtime
                    if mtime > latest:
                        latest = mtime
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return latest


def _get_cached_call_graph(
    project: Path,
    language: str,
    extensions: set[str],
) -> ProjectCallGraph:
    """Get call graph with mtime-based caching.

    Caches call graph per (project, language) and invalidates when any
    source file has been modified since the cache was created.

    Args:
        project: Project root path
        language: Programming language
        extensions: Set of file extensions for this language

    Returns:
        ProjectCallGraph from cache or freshly built.
    """
    cache_key = (str(project), language)
    project_mtime = _get_project_mtime(project, extensions)

    if cache_key in _call_graph_cache:
        cached_mtime, cached_graph = _call_graph_cache[cache_key]
        if cached_mtime >= project_mtime:
            return cached_graph

    # Cache miss or stale - rebuild
    call_graph = build_project_call_graph(str(project), language=language)
    _call_graph_cache[cache_key] = (project_mtime, call_graph)
    return call_graph


@dataclass
class FunctionContext:
    """Context for a single function."""

    name: str
    file: str
    line: int
    signature: str
    docstring: str | None = None
    calls: list[str] = field(default_factory=list)
    blocks: int | None = None  # CFG blocks count
    cyclomatic: int | None = None  # Cyclomatic complexity


@dataclass
class RelevantContext:
    """The full context returned by get_relevant_context."""

    entry_point: str
    depth: int
    functions: list[FunctionContext] = field(default_factory=list)

    def to_llm_string(self) -> str:
        """Format for LLM injection."""
        lines = [f"## Code Context: {self.entry_point} (depth={self.depth})", ""]

        for i, func in enumerate(self.functions):
            # Indentation based on call depth
            indent = "  " * min(i, self.depth)

            # Function header
            short_file = Path(func.file).name if func.file else "?"
            lines.append(f"{indent}📍 {func.name} ({short_file}:{func.line})")
            lines.append(f"{indent}   {func.signature}")

            # Docstring (truncated)
            if func.docstring:
                doc = func.docstring.split("\n")[0][:80]
                lines.append(f"{indent}   # {doc}")

            # Complexity
            if func.blocks is not None:
                complexity_marker = (
                    "🔥" if func.cyclomatic and func.cyclomatic > 10 else ""
                )
                lines.append(
                    f"{indent}   ⚡ complexity: {func.cyclomatic or '?'} ({func.blocks} blocks) {complexity_marker}"
                )

            # Calls
            if func.calls:
                calls_str = ", ".join(func.calls[:5])
                if len(func.calls) > 5:
                    calls_str += f" (+{len(func.calls) - 5} more)"
                lines.append(f"{indent}   → calls: {calls_str}")

            lines.append("")

        # Footer with stats
        result = "\n".join(lines)
        token_estimate = len(result) // 4
        return (
            result
            + f"\n---\n📊 {len(self.functions)} functions | ~{token_estimate} tokens"
        )


def _get_module_exports(
    project: Path,
    module_path: str,
    language: str = "python",
    include_docstrings: bool = True,
) -> "RelevantContext":
    """Get all exports from a module path.

    Args:
        project: Project root path
        module_path: Module path like "providers/anthropic" or "multimodal/video/processor"
        language: Language for extension mapping
        include_docstrings: Whether to include docstrings

    Returns:
        RelevantContext with all functions/classes from the module
    """
    ext_map = {"python": ".py", "typescript": ".ts", "go": ".go", "rust": ".rs"}
    ext = ext_map.get(language, ".py")

    # Try to find the module file
    # module_path "providers/anthropic" -> providers/anthropic.py
    module_file = project / f"{module_path}{ext}"

    if not module_file.exists():
        # Try as directory with __init__.py (Python package)
        init_file = project / module_path / "__init__.py"
        if init_file.exists():
            module_file = init_file
        else:
            raise ValueError(
                f"Module not found: {module_path} (tried {module_file} and {init_file})"
            )

    # Extract all functions and classes from the module
    extractor = HybridExtractor()
    try:
        module_info = extractor.extract(str(module_file))
    except Exception as e:
        raise ValueError(f"Failed to parse module {module_path}: {e}")

    functions: list[FunctionContext] = []

    # Add all functions
    for func in module_info.functions:
        ctx = FunctionContext(
            name=func.name,
            signature=f"def {func.name}({', '.join(func.params)}) -> {func.return_type or 'None'}",
            file=str(module_file),
            line=func.line_number,
            docstring=func.docstring if include_docstrings else None,
            calls=[],
        )
        functions.append(ctx)

    # Add all classes (as constructors/callables)
    for cls in module_info.classes:
        ctx = FunctionContext(
            name=cls.name,
            signature=f"class {cls.name}",
            file=str(module_file),
            line=cls.line_number,
            docstring=cls.docstring if include_docstrings else None,
            calls=[m.name for m in cls.methods],
        )
        functions.append(ctx)

        # Also add class methods
        for method in cls.methods:
            method_ctx = FunctionContext(
                name=f"{cls.name}.{method.name}",
                signature=f"def {method.name}({', '.join(method.params)}) -> {method.return_type or 'None'}",
                file=str(module_file),
                line=method.line_number,
                docstring=method.docstring if include_docstrings else None,
                calls=[],
            )
            functions.append(method_ctx)

    return RelevantContext(entry_point=module_path, depth=0, functions=functions)


# =============================================================================
# Cached Project File Scanner
# =============================================================================


@functools.lru_cache(maxsize=32)
def _cached_scan_project_files(project: str, language: str) -> tuple[str, ...]:
    """Cached project file scan to avoid redundant rglob traversals.

    Returns all source files for a given project and language, cached by
    (project, language) pair. Uses tuple return for hashability.

    Args:
        project: Absolute path to project root (string for hashability)
        language: Language identifier ("python", "typescript", "go", "rust")

    Returns:
        Tuple of absolute file paths matching the language's extensions
    """
    extensions = {
        "python": {".py"},
        "typescript": {".ts", ".tsx"},
        "go": {".go"},
        "rust": {".rs"},
    }.get(language, {".py"})

    project_path = Path(project)
    result = []

    for f in project_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in extensions:
            continue
        # Skip hidden directories
        try:
            rel = f.relative_to(project_path)
            if any(part.startswith(".") for part in rel.parts):
                continue
        except ValueError:
            continue
        result.append(str(f))

    return tuple(result)


def get_relevant_context(
    project: str | Path,
    entry_point: str,
    depth: int = 2,
    language: str = "python",
    include_docstrings: bool = True,
) -> RelevantContext:
    """
    Get token-efficient context for an LLM starting from an entry point.

    Args:
        project: Path to project root
        entry_point: Function/method name (e.g., "Client.stream") or module path (e.g., "providers/anthropic")
        depth: How deep to traverse the call graph
        language: python, typescript, go, or rust
        include_docstrings: Whether to include function docstrings

    Returns:
        RelevantContext with functions reachable from entry_point
    """
    project = Path(project)

    # Module query mode: path with / and no . (e.g., "providers/anthropic")
    if "/" in entry_point and "." not in entry_point:
        return _get_module_exports(project, entry_point, language, include_docstrings)

    # Check if entry_point is a module name (no / or .)
    # e.g., "unified_gate" -> check for unified_gate.py
    ext_for_lang = {
        "python": ".py",
        "typescript": ".ts",
        "go": ".go",
        "rust": ".rs",
    }.get(language, ".py")

    # Search for module file matching entry_point (using cached scan)
    if "." not in entry_point and "/" not in entry_point:
        module_file = None
        cached_files = _cached_scan_project_files(str(project), language)
        target_name = f"{entry_point}{ext_for_lang}"
        for f_str in cached_files:
            f = Path(f_str)
            if f.name == target_name:
                module_file = f
                break

        if module_file:
            # Found a module file - return its exports as module query
            rel_path = module_file.relative_to(project)
            module_path = str(rel_path.with_suffix(""))  # Remove extension
            return _get_module_exports(
                project, module_path, language, include_docstrings
            )

    # Extension map for language (needed for caching and file scanning)
    ext_map = {
        "python": {".py"},
        "typescript": {".ts", ".tsx"},
        "go": {".go"},
        "rust": {".rs"},
    }
    extensions = ext_map.get(language, {".py"})

    # Build cross-file call graph (cached with mtime invalidation)
    call_graph = _get_cached_call_graph(project, language, extensions)

    # Index all signatures (using cached file scan)
    extractor = HybridExtractor()
    signatures: dict[str, tuple[str, FunctionInfo]] = {}  # func_name -> (file, info)

    # Also cache file sources for CFG extraction
    file_sources: dict[str, str] = {}

    # Use cached scan (already filters by extension and excludes hidden dirs)
    cached_files = _cached_scan_project_files(str(project), language)
    for file_path_str in cached_files:
        file_path = Path(file_path_str)
        try:
            source = file_path.read_text()
            file_sources[file_path_str] = source

            info = extractor.extract(file_path_str)
            for func in info.functions:
                # Primary key: module.function (e.g., "claude_spawn.spawn_agent")
                module_name = file_path.stem  # "claude_spawn" from "claude_spawn.py"
                qualified_key = f"{module_name}.{func.name}"
                signatures[qualified_key] = (file_path_str, func)

                # Also store unqualified for backward compat (first wins)
                if func.name not in signatures:
                    signatures[func.name] = (file_path_str, func)
            for cls in info.classes:
                # Index class itself as callable (dataclasses, constructors)
                # Create a pseudo-FunctionInfo for the class
                class_as_func = FunctionInfo(
                    name=cls.name,
                    params=[],  # Could extract __init__ params if needed
                    return_type=cls.name,
                    docstring=cls.docstring,
                    line_number=cls.line_number,
                )
                signatures[cls.name] = (file_path_str, class_as_func)

                for method in cls.methods:
                    # Store as ClassName.method
                    key = f"{cls.name}.{method.name}"
                    signatures[key] = (file_path_str, method)
                    # Also store just method name (for call graph join)
                    # Only if not already taken by a standalone function
                    if method.name not in signatures:
                        signatures[method.name] = (file_path_str, method)
        except Exception:
            pass  # Skip files that fail to parse

    # Build suffix index for O(1) unqualified name lookup (avoids O(n) scan per lookup)
    suffix_index: dict[str, list[str]] = defaultdict(list)
    for qualified_name in signatures:
        if "." in qualified_name:
            _, short_name = qualified_name.rsplit(".", 1)
            suffix_index[short_name].append(qualified_name)
        else:
            suffix_index[qualified_name].append(qualified_name)

    # CFG extractor based on language
    cfg_extractors = {
        "python": extract_python_cfg,
        "typescript": extract_typescript_cfg,
        "go": extract_go_cfg,
        "rust": extract_rust_cfg,
        "java": extract_java_cfg,
        "c": extract_c_cfg,
        "php": extract_php_cfg,
        "kotlin": extract_kotlin_cfg,
        "swift": extract_swift_cfg,
        "csharp": extract_csharp_cfg,
        "scala": extract_scala_cfg,
        "lua": extract_lua_cfg,
        "elixir": extract_elixir_cfg,
    }
    cfg_extractor_fn = cfg_extractors.get(language, extract_python_cfg)

    # Lazy CFG cache: file_path -> dict[func_name, CFGInfo]
    # Batch extracts all CFGs per file on first access (reduces O(N) parses to O(1))
    cfg_cache: dict[str, dict[str, CFGInfo]] = {}

    def get_cfg_cached(file_path: str, func_name: str) -> CFGInfo | None:
        """Get CFG for a function, using batch extraction with per-file caching.

        For Python, extracts all function CFGs in the file on first access.
        For other languages, falls back to individual extraction (no batch support yet).
        """
        if file_path not in cfg_cache:
            source = file_sources.get(file_path)
            if not source:
                cfg_cache[file_path] = {}
            elif language == "python":
                # Batch extract all CFGs for this file in one parse
                cfg_cache[file_path] = extract_python_cfgs_batch(source)
            else:
                # Other languages: extract lazily per function
                cfg_cache[file_path] = {}

        # Return from cache (may be None if function not found)
        cached_file_cfgs = cfg_cache[file_path]
        if func_name in cached_file_cfgs:
            return cached_file_cfgs[func_name]

        # For non-Python: try individual extraction and cache result
        if language != "python" and file_path in file_sources:
            try:
                cfg = cfg_extractor_fn(file_sources[file_path], func_name)
                cached_file_cfgs[func_name] = cfg
                return cfg
            except Exception:
                pass

        return None

    # Build adjacency list from call graph edges
    # Edge format: (caller_file, caller_func, callee_file, callee_func)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in call_graph.edges:
        caller_file, caller_func, callee_file, callee_func = edge
        adjacency[caller_func].append(callee_func)

    # BFS from entry point up to depth
    visited = set()
    queue = deque([(entry_point, 0)])  # O(1) popleft vs O(n) list.pop(0)
    result_functions = []

    # Helper to resolve function name to signature (handles qualified/unqualified)
    def resolve_func_name(name: str) -> list[tuple[str, tuple[str, FunctionInfo]]]:
        """Resolve function name, returning all matches for ambiguous names."""
        # If qualified (has dot), do direct lookup - O(1)
        if "." in name:
            if name in signatures:
                return [(name, signatures[name])]
            return []

        # Unqualified name - O(1) lookup via suffix index (was O(n) full scan)
        qualified_names = suffix_index.get(name, [])
        if qualified_names:
            return [(qn, signatures[qn]) for qn in qualified_names]
        elif name in signatures:
            # Fall back to direct unqualified lookup
            return [(name, signatures[name])]

        return []

    while queue:
        func_name, current_depth = queue.popleft()

        if func_name in visited or current_depth > depth:
            continue
        visited.add(func_name)

        # Get signature info if available (may return multiple matches)
        resolved_list = resolve_func_name(func_name)
        if resolved_list:
            for resolved_name, (file_path, func_info) in resolved_list:
                # Skip if we already processed this qualified name
                if resolved_name in visited and resolved_name != func_name:
                    continue
                visited.add(resolved_name)

                # Try to get CFG complexity (uses batch extraction with caching)
                blocks = None
                cyclomatic = None
                cfg = get_cfg_cached(file_path, func_info.name)
                if cfg and cfg.blocks:
                    blocks = len(cfg.blocks)
                    cyclomatic = cfg.cyclomatic_complexity

                ctx = FunctionContext(
                    name=resolved_name,  # Use qualified name for clarity
                    file=file_path,
                    line=func_info.line_number,
                    signature=func_info.signature(),
                    docstring=func_info.docstring if include_docstrings else None,
                    calls=adjacency.get(
                        func_info.name, []
                    ),  # Use unqualified for adjacency lookup
                    blocks=blocks,
                    cyclomatic=cyclomatic,
                )
                result_functions.append(ctx)

                # Queue callees from this function
                for callee in adjacency.get(func_info.name, []):
                    if callee not in visited and current_depth < depth:
                        queue.append((callee, current_depth + 1))
        else:
            # Function not found in signatures, still include it
            ctx = FunctionContext(
                name=func_name,
                file="?",
                line=0,
                signature=f"def {func_name}(...)",
                calls=adjacency.get(func_name, []),
            )
            result_functions.append(ctx)

            # Queue callees
            for callee in adjacency.get(func_name, []):
                if callee not in visited and current_depth < depth:
                    queue.append((callee, current_depth + 1))

    return RelevantContext(
        entry_point=entry_point, depth=depth, functions=result_functions
    )


def get_dfg_context(
    source_or_path: str, function_name: str, language: str = "python"
) -> dict:
    """
    Get data flow analysis for a function.

    Extracts variable references (definitions, updates, uses) and
    def-use chains (dataflow edges) for the specified function.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        Dict with:
          - function: function name
          - refs: list of variable references (name, type, line, column)
          - edges: list of def-use edges (var, def_line, use_line, def, use)
          - variables: list of variable names found
    """
    source_code, _ = _resolve_source(source_or_path)

    # Select extractor based on language
    dfg_extractors = {
        "python": extract_python_dfg,
        "typescript": extract_typescript_dfg,
        "javascript": extract_typescript_dfg,  # JS uses TS extractor
        "go": extract_go_dfg,
        "rust": extract_rust_dfg,
        "java": extract_java_dfg,
        "c": extract_c_dfg,
        "cpp": extract_cpp_dfg,
        "ruby": extract_ruby_dfg,
        "php": extract_php_dfg,
        "kotlin": extract_kotlin_dfg,
        "swift": extract_swift_dfg,
        "csharp": extract_csharp_dfg,
        "scala": extract_scala_dfg,
        "lua": extract_lua_dfg,
        "elixir": extract_elixir_dfg,
    }

    # Default to Python for unknown languages
    extractor_fn = dfg_extractors.get(language, extract_python_dfg)

    try:
        dfg_info: DFGInfo = extractor_fn(source_code, function_name)
        return dfg_info.to_dict()
    except Exception:
        # Return empty DFG on extraction failure
        return {"function": function_name, "refs": [], "edges": [], "variables": []}


# =============================================================================
# CFG API Functions (Layer 3)
# =============================================================================


def get_cfg_context(
    source_or_path: str, function_name: str, language: str = "python"
) -> dict:
    """
    Get control flow graph context for a function.

    Extracts basic blocks, control flow edges, and complexity metrics
    for the specified function.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        Dict with:
          - function: function name
          - blocks: list of basic block dicts (id, type, lines, calls)
          - edges: list of edge dicts (from, to, type, condition)
          - entry_block: entry block ID
          - exit_blocks: list of exit block IDs
          - cyclomatic_complexity: cyclomatic complexity metric
          - nested_functions: dict of nested function CFGs (if any)
    """
    source_code, _ = _resolve_source(source_or_path)

    cfg_extractors = {
        "python": extract_python_cfg,
        "typescript": extract_typescript_cfg,
        "javascript": extract_typescript_cfg,
        "go": extract_go_cfg,
        "rust": extract_rust_cfg,
        "java": extract_java_cfg,
        "c": extract_c_cfg,
        "cpp": extract_cpp_cfg,
        "ruby": extract_ruby_cfg,
        "php": extract_php_cfg,
        "swift": extract_swift_cfg,
        "csharp": extract_csharp_cfg,
        "lua": extract_lua_cfg,
        "elixir": extract_elixir_cfg,
    }

    extractor_fn = cfg_extractors.get(language, extract_python_cfg)

    try:
        cfg_info: CFGInfo = extractor_fn(source_code, function_name)
        if cfg_info is None:
            return {
                "function": function_name,
                "blocks": [],
                "edges": [],
                "entry_block": 0,
                "exit_blocks": [],
                "cyclomatic_complexity": 0,
            }
        return cfg_info.to_dict()
    except Exception:
        # Return empty CFG on extraction failure
        return {
            "function": function_name,
            "blocks": [],
            "edges": [],
            "entry_block": 0,
            "exit_blocks": [],
            "cyclomatic_complexity": 0,
        }


def get_cfg_blocks(
    source_or_path: str, function_name: str, language: str = "python"
) -> list[dict]:
    """
    Get CFG basic blocks for a function.

    Basic blocks are sequences of statements with no internal branches.
    Control enters only at the first statement and leaves only at the last.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        List of block dicts, each containing:
          - id: block identifier
          - type: block type (entry, branch, loop_header, return, exit, body)
          - lines: [start_line, end_line]
          - calls: list of function calls in this block (if any)

        Returns empty list if function not found.
    """
    cfg = get_cfg_context(source_or_path, function_name, language)
    return cfg.get("blocks", [])


def get_cfg_edges(
    source_or_path: str, function_name: str, language: str = "python"
) -> list[dict]:
    """
    Get CFG control flow edges for a function.

    Edges represent possible control flow transitions between basic blocks.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        List of edge dicts, each containing:
          - from: source block ID
          - to: target block ID
          - type: edge type (true, false, unconditional, back_edge, break, continue)
          - condition: human-readable condition (for conditional edges)

        Returns empty list if function not found.
    """
    cfg = get_cfg_context(source_or_path, function_name, language)
    return cfg.get("edges", [])


def query(
    project: str | Path, query: str, depth: int = 2, language: str = "python"
) -> str:
    """
    Convenience function that returns LLM-ready string directly.

    Args:
        project: Path to project root
        query: Function or method name to start from
        depth: Call graph traversal depth
        language: Programming language

    Returns:
        Formatted string ready for LLM context injection
    """
    ctx = get_relevant_context(project, query, depth, language)
    return ctx.to_llm_string()


# =============================================================================
# PDG API Functions (Layer 5)
# =============================================================================


def get_pdg_context(
    source_or_path: str, function_name: str, language: str = "python"
) -> dict | None:
    """
    Get program dependence graph context for a function.

    Provides control and data dependencies unified in a single graph,
    useful for understanding code impact and program slicing.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of the function to analyze
        language: One of "python", "typescript", "javascript", "go", "rust", "java", "c"

    Returns:
        Dict with PDG summary including:
        - function: Function name
        - nodes: Number of PDG nodes
        - edges: List of edge dicts with type and label
        - control_edges: Count of control dependency edges
        - data_edges: Count of data dependency edges
        - complexity: Cyclomatic complexity from CFG
        - variables: List of tracked variable names

        Returns None if function not found or extraction fails.

    Raises:
        ValueError: If language is not supported

    Example:
        >>> code = "def add(a, b):\\n    c = a + b\\n    return c"
        >>> ctx = get_pdg_context(code, "add")
        >>> ctx["function"]
        'add'
        >>> ctx["complexity"]
        1
    """
    source_code, _ = _resolve_source(source_or_path)
    pdg = extract_pdg(source_code, function_name, language)
    if pdg is None:
        return None

    return pdg.to_compact_dict()


def get_slice(
    source_or_path: str,
    function_name: str,
    line: int,
    direction: str = "backward",
    variable: str | None = None,
    language: str = "python",
) -> set[int]:
    """
    Get program slice - lines affecting or affected by a given line.

    Program slicing identifies which parts of code are relevant to
    a specific computation, useful for debugging and understanding
    code dependencies.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of the function to analyze
        line: Line number to slice from
        direction: "backward" (what affects this line) or
                   "forward" (what this line affects)
        variable: Optional specific variable to trace (traces all if None)
        language: One of "python", "typescript", "javascript", "go", "rust", "java", "c"

    Returns:
        Set of line numbers in the slice. Empty set if function not found
        or line is invalid.

    Raises:
        ValueError: If direction is not "backward" or "forward"
        ValueError: If language is not supported

    Example:
        >>> code = '''
        ... def compute(x):
        ...     a = x + 1
        ...     b = a * 2
        ...     return b
        ... '''
        >>> get_slice(code, "compute", line=5, direction="backward")
        {3, 4, 5}  # Lines that affect the return
    """
    if direction not in ("backward", "forward"):
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'backward' or 'forward'."
        )

    source_code, _ = _resolve_source(source_or_path)
    pdg = extract_pdg(source_code, function_name, language)
    if pdg is None:
        return set()

    if direction == "backward":
        return pdg.backward_slice(line, variable)
    else:
        return pdg.forward_slice(line, variable)


# ==============================================================================
# Layer 2: Cross-File Call Graph Functions
# ==============================================================================


def scan_project_files(
    root: str,
    language: str = "python",
    respect_ignore: bool = True,
) -> list[str]:
    """
    Find all source files in project for given language.

    Args:
        root: Project root directory path
        language: "python", "typescript", "go", or "rust"
        respect_ignore: If True, respect .tldrignore patterns (default True)

    Returns:
        List of absolute paths to source files

    Example:
        >>> files = scan_project_files("/path/to/project", "python")
        >>> print(files)
        ['/path/to/project/main.py', '/path/to/project/utils/helper.py']
    """
    return _scan_project(root, language, respect_ignore=respect_ignore)


def get_imports(file_path: str, language: str = "python") -> list[dict]:
    """
    Parse imports from a source file.

    Args:
        file_path: Path to source file
        language: "python", "typescript", "go", or "rust"

    Returns:
        List of import info dicts. Structure varies by language:
        - Python: {module, names, is_from, alias/aliases}
        - TypeScript: {module, names, is_default, aliases}
        - Go: {module, alias}
        - Rust: {module, names, is_mod}

    Example:
        >>> imports = get_imports("/path/to/file.py", "python")
        >>> print(imports)
        [{'module': 'os', 'names': [], 'is_from': False, 'alias': None},
         {'module': 'pathlib', 'names': ['Path'], 'is_from': True, 'aliases': {}}]
    """
    if language == "python":
        return _parse_imports(file_path)
    elif language == "typescript" or language == "javascript":
        return _parse_ts_imports(file_path)
    elif language == "go":
        return _parse_go_imports(file_path)
    elif language == "rust":
        return _parse_rust_imports(file_path)
    elif language == "java":
        return _parse_java_imports(file_path)
    elif language == "c":
        return _parse_c_imports(file_path)
    elif language == "cpp":
        return _parse_cpp_imports(file_path)
    elif language == "ruby":
        return _parse_ruby_imports(file_path)
    elif language == "php":
        return _parse_php_imports(file_path)
    elif language == "kotlin":
        return _parse_kotlin_imports(file_path)
    elif language == "swift":
        return _parse_swift_imports(file_path)
    elif language == "csharp":
        return _parse_csharp_imports(file_path)
    elif language == "scala":
        return _parse_scala_imports(file_path)
    elif language == "lua":
        return _parse_lua_imports(file_path)
    elif language == "elixir":
        return _parse_elixir_imports(file_path)
    else:
        raise ValueError(f"Unsupported language: {language}")


def build_function_index(root: str, language: str = "python") -> dict:
    """
    Build index mapping (module, func) -> file_path for all functions.

    Args:
        root: Project root directory path
        language: "python", "typescript", "go", or "rust"

    Returns:
        Dict mapping (module_name, func_name) tuples and "module.func" strings
        to relative file paths

    Example:
        >>> index = build_function_index("/path/to/project", "python")
        >>> print(index[("utils", "helper")])
        'utils.py'
        >>> print(index["utils.helper"])
        'utils.py'
    """
    return _build_function_index(root, language)


# =============================================================================
# Layer 1 (AST) API Functions
# =============================================================================


def get_intra_file_calls(file_path: str) -> dict:
    """
    Get call graph within a single file.

    Extracts function call relationships showing which functions
    call which other functions within the same file.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Dict with two keys:
        - calls: dict mapping caller -> list of callees
        - called_by: dict mapping callee -> list of callers

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be parsed

    Example:
        >>> cg = get_intra_file_calls("/path/to/file.py")
        >>> cg["calls"]["main"]  # Functions called by main
        ['helper', 'process']
        >>> cg["called_by"]["helper"]  # Functions that call helper
        ['main']
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    module_info = _extract_file_impl(file_path)
    return {
        "calls": dict(module_info.call_graph.calls),
        "called_by": dict(module_info.call_graph.called_by),
    }


def extract_file(file_path: str, base_path: str | None = None) -> dict:
    """
    Extract code structure from any supported file.

    Generic file extractor that returns complete module information
    including imports, functions, classes, and call graph.

    Args:
        file_path: Path to the file to analyze
        base_path: Optional base directory for path containment validation.
                   If provided, file_path must resolve within base_path.

    Returns:
        Dict containing:
        - file_path: Path to the analyzed file
        - language: Detected language (e.g., "python")
        - docstring: Module-level docstring if present
        - imports: List of import dicts
        - functions: List of function dicts with signatures
        - classes: List of class dicts with methods
        - call_graph: Dict with calls and called_by relationships

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file type is not supported or path is invalid
        PathTraversalError: If path escapes base_path via traversal

    Example:
        >>> info = extract_file("/path/to/module.py")
        >>> print(info["functions"][0]["signature"])
        'def my_function(x: int) -> str'
    """
    # Security: Validate path containment
    _validate_path_containment(file_path, base_path)

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    module_info = _extract_file_impl(file_path)
    return module_info.to_dict()


# =============================================================================
# Project Navigation Functions
# =============================================================================


def get_file_tree(
    root: str | Path,
    extensions: set[str] | None = None,
    exclude_hidden: bool = True,
) -> dict:
    """
    Get file tree structure for a project.

    Args:
        root: Root directory to scan
        extensions: Optional set of extensions to include (e.g., {".py", ".ts"})
        exclude_hidden: If True, exclude hidden files/directories (default True)

    Returns:
        Dict with tree structure:
        {
            "name": "project",
            "type": "dir",
            "children": [
                {"name": "src", "type": "dir", "children": [...]},
                {"name": "main.py", "type": "file", "path": "src/main.py"}
            ]
        }

    Raises:
        PathTraversalError: If root path contains directory traversal patterns
    """
    # Security: Validate path containment
    _validate_path_containment(str(root))

    root = Path(root)

    def scan_dir(path: Path) -> dict:
        result = {"name": path.name, "type": "dir", "children": []}

        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return result

        for item in items:
            # Skip hidden files/dirs
            if exclude_hidden and item.name.startswith("."):
                continue

            if item.is_dir():
                child = scan_dir(item)
                # Only include non-empty directories
                if child["children"] or extensions is None:
                    result["children"].append(child)
            elif item.is_file():
                if extensions is None or item.suffix in extensions:
                    result["children"].append(
                        {
                            "name": item.name,
                            "type": "file",
                            "path": str(item.relative_to(root)),
                        }
                    )

        return result

    return scan_dir(root)


def search(
    pattern: str,
    root: str | Path,
    extensions: set[str] | None = None,
    context_lines: int = 0,
    max_results: int = 100,
    max_files: int = 10000,
) -> list[dict]:
    """
    Search files for a regex pattern.

    Args:
        pattern: Regex pattern to search for
        root: Root directory to search in
        extensions: Optional set of extensions to filter (e.g., {".py"})
        context_lines: Number of context lines to include (default 0)
        max_results: Maximum matches to return (default 100, 0 = unlimited)
        max_files: Maximum files to scan (default 10000, 0 = unlimited)

    Returns:
        List of matches:
        [
            {"file": "src/main.py", "line": 10, "content": "def hello():"},
            ...
        ]

    Raises:
        PathTraversalError: If root path contains directory traversal patterns
    """
    # Security: Validate path containment
    _validate_path_containment(str(root))

    # Directories to skip (common junk)
    SKIP_DIRS = {
        "node_modules",
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "coverage",
        ".tox",
        "venv",
        ".venv",
        "env",
        ".env",
        "vendor",
        ".cache",
    }

    results = []
    root = Path(root)
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")
    files_scanned = 0

    for file_path in root.rglob("*"):
        # Check file limit
        if max_files > 0 and files_scanned >= max_files:
            break

        if not file_path.is_file():
            continue

        # Skip hidden files and junk directories
        try:
            rel_path = file_path.relative_to(root)
            parts = rel_path.parts
            if any(part.startswith(".") for part in parts):
                continue
            if any(part in SKIP_DIRS for part in parts):
                continue
        except ValueError:
            continue

        # Filter by extension
        if extensions and file_path.suffix not in extensions:
            continue

        files_scanned += 1

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            for i, line in enumerate(lines, 1):
                if compiled.search(line):
                    match = {
                        "file": str(file_path.relative_to(root)),
                        "line": i,
                        "content": line.strip(),
                    }

                    # Add context if requested
                    if context_lines > 0:
                        start = max(0, i - 1 - context_lines)
                        end = min(len(lines), i + context_lines)
                        match["context"] = lines[start:end]

                    results.append(match)

                    # Check result limit
                    if max_results > 0 and len(results) >= max_results:
                        return results
        except (OSError, UnicodeDecodeError):
            pass

    return results


class Selection:
    """
    Manage file selection state for batch operations.

    Usage:
        sel = Selection()
        sel.add("src/main.py", "src/utils.py")
        sel.remove("src/utils.py")

        for f in sel.files:
            info = extract_file(f)
    """

    def __init__(self):
        self._selected: set[str] = set()

    def add(self, *paths: str) -> "Selection":
        """Add paths to selection."""
        self._selected.update(paths)
        return self

    def remove(self, *paths: str) -> "Selection":
        """Remove paths from selection."""
        self._selected -= set(paths)
        return self

    def clear(self) -> "Selection":
        """Clear all selection."""
        self._selected.clear()
        return self

    def set(self, *paths: str) -> "Selection":
        """Replace entire selection with new paths."""
        self._selected = set(paths)
        return self

    @property
    def files(self) -> list[str]:
        """Return selected files as sorted list."""
        return sorted(self._selected)

    def __contains__(self, path: str) -> bool:
        """Check if path is selected."""
        return path in self._selected

    def __len__(self) -> int:
        """Return number of selected files."""
        return len(self._selected)


# Parallelization threshold for get_code_structure
MIN_FILES_FOR_PARALLEL = 15


def _extract_file_for_structure(file_path: str) -> tuple[str, dict | None]:
    """
    Extract file info for parallel processing in get_code_structure.

    Args:
        file_path: Absolute path to file

    Returns:
        Tuple of (file_path, info_dict) or (file_path, None) on error
    """
    try:
        info = _extract_file_impl(file_path)
        return (file_path, info.to_dict())
    except Exception:
        return (file_path, None)


def get_code_structure(
    root: str | Path,
    language: str = "python",
    max_results: int = 100,
) -> dict:
    """
    Get code structure (codemaps) for all files in a project.

    Args:
        root: Root directory to analyze
        language: Language to analyze ("python", "typescript", "go", "rust")
        max_results: Maximum number of files to analyze (default 100)

    Returns:
        Dict with codemap structure:
        {
            "root": "/path/to/project",
            "files": [
                {
                    "path": "src/main.py",
                    "functions": ["main", "helper"],
                    "classes": ["MyClass"],
                    "imports": ["os", "sys"]
                },
                ...
            ]
        }
    """
    # Security: Validate path containment to prevent directory traversal
    _validate_path_containment(str(root))

    root = Path(root)

    # Get extension map for language
    ext_map = {
        "python": {".py"},
        "typescript": {".ts", ".tsx"},
        "javascript": {".js", ".jsx"},
        "go": {".go"},
        "rust": {".rs"},
        "java": {".java"},
        "c": {".c", ".h"},
        "cpp": {".cpp", ".cc", ".cxx", ".hpp"},
        "swift": {".swift"},
        "kotlin": {".kt", ".kts"},
        "scala": {".scala"},
        "ruby": {".rb"},
        "php": {".php"},
        "csharp": {".cs"},
        "elixir": {".ex", ".exs"},
        "lua": {".lua"},
    }

    extensions = ext_map.get(language, {".py"})

    # Collect files to process (apply filtering before extraction)
    files_to_process: list[str] = []
    for file_path in root.rglob("*"):
        if len(files_to_process) >= max_results:
            break

        if not file_path.is_file():
            continue

        if file_path.suffix not in extensions:
            continue

        # Skip hidden files (only check relative path, not parent directories)
        try:
            rel_path = file_path.relative_to(root)
            if any(part.startswith(".") for part in rel_path.parts):
                continue
        except ValueError:
            continue

        files_to_process.append(str(file_path))

    # Extract file info - parallel for large sets, sequential for small
    if len(files_to_process) > MIN_FILES_FOR_PARALLEL:
        max_workers = min(os.cpu_count() or 4, 8)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            extraction_results = list(
                executor.map(_extract_file_for_structure, files_to_process)
            )
    else:
        extraction_results = [_extract_file_for_structure(f) for f in files_to_process]

    # Build result from extraction results
    result = {"root": str(root), "language": language, "files": []}
    for file_path, info_dict in extraction_results:
        if info_dict is None:
            continue

        try:
            rel_path = Path(file_path).relative_to(root)
            file_entry = {
                "path": str(rel_path),
                "functions": [f["name"] for f in info_dict.get("functions", [])],
                "classes": [c["name"] for c in info_dict.get("classes", [])],
                "imports": info_dict.get("imports", []),
            }
            result["files"].append(file_entry)
        except Exception:
            # Skip files with path resolution issues
            pass

    return result


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m tldr.api <project_path> <entry_point> [depth] [language]"
        )
        print(
            "Example: python -m tldr.api /path/to/project build_project_call_graph 2 python"
        )
        sys.exit(1)

    project_path = sys.argv[1]
    entry = sys.argv[2]
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    lang = sys.argv[4] if len(sys.argv) > 4 else "python"

    print(query(project_path, entry, depth, lang))
