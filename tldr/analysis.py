"""Codebase analysis tools built on TLDR's call graph.

Provides:
- Impact analysis: Find all callers of a function (reverse call graph)
- Dead code detection: Find unreachable functions
- Architecture extraction: Detect layers from call patterns

These operate on the call graph from cross_file_calls.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cross_file_calls import CallGraph


@dataclass
class FunctionRef:
    """A function reference in the codebase."""

    file: str
    name: str

    def __hash__(self):
        return hash((self.file, self.name))

    def __eq__(self, other):
        if not isinstance(other, FunctionRef):
            return False
        return self.file == other.file and self.name == other.name

    def __repr__(self):
        return f"{self.file}:{self.name}"


def build_reverse_graph(
    edges: list[tuple[str, str, str, str]],
) -> dict[FunctionRef, list[FunctionRef]]:
    """Build reverse call graph: who calls each function?

    Args:
        edges: List of (from_file, from_func, to_file, to_func) tuples

    Returns:
        Dict mapping callee -> list of callers
    """
    reverse = defaultdict(list)
    for from_file, from_func, to_file, to_func in edges:
        callee = FunctionRef(file=to_file, name=to_func)
        caller = FunctionRef(file=from_file, name=from_func)
        reverse[callee].append(caller)
    return reverse


def build_forward_graph(
    edges: list[tuple[str, str, str, str]],
) -> dict[FunctionRef, list[FunctionRef]]:
    """Build forward call graph: what does each function call?

    Args:
        edges: List of (from_file, from_func, to_file, to_func) tuples

    Returns:
        Dict mapping caller -> list of callees
    """
    forward = defaultdict(list)
    for from_file, from_func, to_file, to_func in edges:
        caller = FunctionRef(file=from_file, name=from_func)
        callee = FunctionRef(file=to_file, name=to_func)
        forward[caller].append(callee)
    return forward


def impact_analysis(
    call_graph: "CallGraph",
    target_func: str,
    max_depth: int = 3,
    target_file: str | None = None,
) -> dict:
    """Find all callers of a function, up to max_depth levels.

    This is the reverse call graph - useful for understanding
    what code would be affected by changing a function.

    Args:
        call_graph: CallGraph from cross_file_calls
        target_func: Function name to find callers of
        max_depth: How deep to traverse callers
        target_file: Optional file filter

    Returns:
        Dict with 'targets' (tree of callers) and 'total_targets' count
    """
    edges = call_graph.edges
    reverse = build_reverse_graph(edges)

    # Find target function(s)
    targets = []

    # Check all callees
    all_callees = set()
    for from_file, from_func, to_file, to_func in edges:
        callee = FunctionRef(file=to_file, name=to_func)
        if callee.name == target_func:
            if target_file is None or target_file in callee.file:
                all_callees.add(callee)

    targets = list(all_callees)

    if not targets:
        return {"error": f"Function '{target_func}' not found in call graph"}

    results = {}
    for target in targets:
        tree = _build_caller_tree(target, reverse, max_depth, set())
        results[str(target)] = tree

    return {"targets": results, "total_targets": len(targets)}


def _build_caller_tree(
    func: FunctionRef,
    reverse: dict[FunctionRef, list[FunctionRef]],
    depth: int,
    visited: set,
) -> dict:
    """Recursively build caller tree."""
    callers = reverse.get(func, [])

    # Base case: truncate at depth 0 or if we've seen this node
    if depth <= 0 or func in visited:
        return {
            "function": func.name,
            "file": func.file,
            "caller_count": len(callers),
            "callers": [],
            "truncated": True,
        }

    visited.add(func)

    tree = {
        "function": func.name,
        "file": func.file,
        "caller_count": len(callers),
        "callers": [],
        "truncated": False,
    }

    for caller in callers:
        subtree = _build_caller_tree(caller, reverse, depth - 1, visited.copy())
        tree["callers"].append(subtree)

    return tree


def dead_code_analysis(
    call_graph: "CallGraph",
    all_functions: list[dict],
    entry_points: list[str] | None = None,
) -> dict:
    """Find functions that are never called (excluding entry points).

    Args:
        call_graph: CallGraph from cross_file_calls
        all_functions: List of {file, name} dicts from structure analysis
        entry_points: Additional entry point patterns to exclude

    Returns:
        Dict with dead_functions, by_file, totals, and percentage
    """
    edges = call_graph.edges
    entry_points = entry_points or []

    # Build set of all called functions
    called = set()
    for _, _, to_file, to_func in edges:
        called.add(FunctionRef(file=to_file, name=to_func))

    # Build set of all callers (these are "alive" by definition)
    callers = set()
    for from_file, from_func, _, _ in edges:
        callers.add(FunctionRef(file=from_file, name=from_func))

    # Common entry point patterns
    entry_patterns = [
        "main",
        "__main__",
        "cli",
        "app",
        "run",
        "start",
        "test_",
        "pytest_",
        "setup",
        "teardown",
    ] + entry_points

    # Find dead functions
    dead = []
    for func_info in all_functions:
        func = FunctionRef(file=func_info["file"], name=func_info["name"])

        # Skip if it's called
        if func in called:
            continue

        # Skip if it's an entry point pattern
        is_entry = any(
            pattern in func.name or pattern in func.file for pattern in entry_patterns
        )
        if is_entry:
            continue

        # Skip dunder methods
        if func.name.startswith("__") and func.name.endswith("__"):
            continue

        # Skip if it calls something (it's a root/entry)
        if func in callers:
            continue

        dead.append(func)

    # Group by file
    by_file = defaultdict(list)
    for func in dead:
        by_file[func.file].append(func.name)

    total_funcs = len(all_functions)
    return {
        "dead_functions": [{"file": f.file, "function": f.name} for f in dead],
        "by_file": dict(by_file),
        "total_dead": len(dead),
        "total_functions": total_funcs,
        "dead_percentage": round(len(dead) / max(total_funcs, 1) * 100, 1),
    }


def architecture_analysis(call_graph: "CallGraph") -> dict:
    """Detect architectural layers from call patterns.

    Heuristics:
    - Functions that call but are not called = entry layer
    - Functions that are called but don't call = leaf layer
    - Analyze directory structure for layer hints
    - Detect circular dependencies

    Args:
        call_graph: CallGraph from cross_file_calls

    Returns:
        Dict with layer info, directory analysis, and circular deps
    """
    edges = call_graph.edges
    forward = build_forward_graph(edges)
    reverse = build_reverse_graph(edges)

    # Categorize functions
    entry_layer = []  # Call others but not called
    leaf_layer = []  # Called but don't call others
    middle_layer = []  # Both call and are called

    all_in_graph = set(forward.keys()) | set(reverse.keys())

    for func in all_in_graph:
        calls_others = func in forward and len(forward[func]) > 0
        is_called = func in reverse and len(reverse[func]) > 0

        if calls_others and not is_called:
            entry_layer.append(func)
        elif is_called and not calls_others:
            leaf_layer.append(func)
        elif calls_others and is_called:
            middle_layer.append(func)

    # Analyze directory patterns
    dir_stats = defaultdict(lambda: {"calls_out": 0, "calls_in": 0, "functions": []})

    for func in all_in_graph:
        dir_name = str(Path(func.file).parent) if "/" in func.file else "."
        dir_stats[dir_name]["functions"].append(func.name)

    for from_file, _, to_file, _ in edges:
        from_dir = str(Path(from_file).parent) if "/" in from_file else "."
        to_dir = str(Path(to_file).parent) if "/" in to_file else "."

        if from_dir != to_dir:
            dir_stats[from_dir]["calls_out"] += 1
            dir_stats[to_dir]["calls_in"] += 1

    # Detect circular dependencies
    circular = []
    seen_pairs = set()
    for from_file, _, to_file, _ in edges:
        pair = (from_file, to_file)
        reverse_pair = (to_file, from_file)
        if reverse_pair in seen_pairs and pair not in seen_pairs:
            circular.append({"a": from_file, "b": to_file})
        seen_pairs.add(pair)

    # Infer layers from directory call ratios
    layer_inference = []
    for dir_name, stats in sorted(dir_stats.items()):
        ratio = stats["calls_out"] / max(stats["calls_in"], 1)
        if ratio > 2:
            layer = "HIGH (entry/controller)"
        elif ratio < 0.5:
            layer = "LOW (utility/data)"
        else:
            layer = "MIDDLE (service)"

        layer_inference.append(
            {
                "directory": dir_name,
                "calls_out": stats["calls_out"],
                "calls_in": stats["calls_in"],
                "inferred_layer": layer,
                "function_count": len(stats["functions"]),
            }
        )

    return {
        "entry_layer": [
            {"file": f.file, "function": f.name} for f in entry_layer[:20]
        ],
        "leaf_layer": [{"file": f.file, "function": f.name} for f in leaf_layer[:20]],
        "middle_layer_count": len(middle_layer),
        "directory_layers": layer_inference,
        "circular_dependencies": circular,
        "summary": {
            "entry_count": len(entry_layer),
            "leaf_count": len(leaf_layer),
            "middle_count": len(middle_layer),
            "circular_count": len(circular),
        },
    }


# Convenience functions that take path instead of CallGraph
def analyze_impact(
    path: str,
    target_func: str,
    max_depth: int = 3,
    target_file: str | None = None,
    language: str = "python",
) -> dict:
    """Convenience wrapper that builds call graph from path.

    Args:
        path: Project path to analyze
        target_func: Function name to find callers of
        max_depth: How deep to traverse callers
        target_file: Optional file filter
        language: Source language

    Returns:
        Impact analysis results
    """
    from .api import build_project_call_graph

    call_graph = build_project_call_graph(path, language=language)
    return impact_analysis(call_graph, target_func, max_depth, target_file)


def analyze_dead_code(
    path: str,
    entry_points: list[str] | None = None,
    language: str = "python",
) -> dict:
    """Convenience wrapper that builds call graph from path.

    Args:
        path: Project path to analyze
        entry_points: Additional entry point patterns
        language: Source language

    Returns:
        Dead code analysis results
    """
    from .api import build_project_call_graph, get_code_structure

    call_graph = build_project_call_graph(path, language=language)
    structure = get_code_structure(path, language=language, max_results=1000)

    # Build function list from structure
    all_functions = []
    for file_info in structure.get("files", []):
        file_path = file_info.get("path", "")
        for func_name in file_info.get("functions", []):
            all_functions.append({"file": file_path, "name": func_name})

    return dead_code_analysis(call_graph, all_functions, entry_points)


def analyze_architecture(path: str, language: str = "python") -> dict:
    """Convenience wrapper that builds call graph from path.

    Args:
        path: Project path to analyze
        language: Source language

    Returns:
        Architecture analysis results
    """
    from .api import build_project_call_graph

    call_graph = build_project_call_graph(path, language=language)
    return architecture_analysis(call_graph)
