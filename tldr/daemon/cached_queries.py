"""
Salsa-memoized query functions for the TLDR daemon.

These functions wrap the TLDR API with automatic caching via SalsaDB.
Results are memoized and automatically invalidated when source files change.
"""

from pathlib import Path

from tldr.salsa import SalsaDB, salsa_query


@salsa_query
def cached_search(db: SalsaDB, project: str, pattern: str, max_results: int) -> dict:
    """Cached search query - memoized by SalsaDB."""
    from tldr import api
    from tldr.tldrignore import IgnoreSpec
    ignore_spec = IgnoreSpec(project, use_gitignore=True)
    results = api.search(pattern=pattern, root=Path(project), max_results=max_results, ignore_spec=ignore_spec)
    return {"status": "ok", "results": results}


@salsa_query
def cached_extract(db: SalsaDB, file_path: str) -> dict:
    """Cached file extraction - memoized by SalsaDB."""
    from tldr import api
    result = api.extract_file(file_path)
    return {"status": "ok", "result": result}


@salsa_query
def cached_dead_code(db: SalsaDB, project: str, entry_points: tuple, language: str) -> dict:
    """Cached dead code analysis - memoized by SalsaDB."""
    from tldr.analysis import analyze_dead_code
    # Convert tuple back to list for the API
    entry_list = list(entry_points) if entry_points else None
    result = analyze_dead_code(project, entry_points=entry_list, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_architecture(db: SalsaDB, project: str, language: str) -> dict:
    """Cached architecture analysis - memoized by SalsaDB."""
    from tldr.analysis import analyze_architecture
    result = analyze_architecture(project, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_cfg(db: SalsaDB, file_path: str, function: str, language: str) -> dict:
    """Cached CFG extraction - memoized by SalsaDB."""
    from tldr.api import get_cfg_context
    result = get_cfg_context(file_path, function, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_dfg(db: SalsaDB, file_path: str, function: str, language: str) -> dict:
    """Cached DFG extraction - memoized by SalsaDB."""
    from tldr.api import get_dfg_context
    result = get_dfg_context(file_path, function, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_slice(db: SalsaDB, file_path: str, function: str, line: int, direction: str, variable: str) -> dict:
    """Cached program slice - memoized by SalsaDB."""
    from tldr.api import get_slice
    var = variable if variable else None
    lines = get_slice(file_path, function, line, direction=direction, variable=var)
    return {"status": "ok", "lines": sorted(lines), "count": len(lines)}


@salsa_query
def cached_tree(db: SalsaDB, project: str, extensions: tuple, exclude_hidden: bool) -> dict:
    """Cached file tree - memoized by SalsaDB."""
    from tldr.api import get_file_tree
    from tldr.tldrignore import IgnoreSpec
    ext_set = set(extensions) if extensions else None
    ignore_spec = IgnoreSpec(project, use_gitignore=True)
    result = get_file_tree(project, extensions=ext_set, exclude_hidden=exclude_hidden, ignore_spec=ignore_spec)
    return {"status": "ok", "result": result}


@salsa_query
def cached_structure(db: SalsaDB, project: str, language: str, max_results: int) -> dict:
    """Cached code structure - memoized by SalsaDB."""
    from tldr.api import get_code_structure
    from tldr.tldrignore import IgnoreSpec
    ignore_spec = IgnoreSpec(project, use_gitignore=True)
    result = get_code_structure(project, language=language, max_results=max_results, ignore_spec=ignore_spec)
    return {"status": "ok", "result": result}


@salsa_query
def cached_context(db: SalsaDB, project: str, entry: str, language: str, depth: int) -> dict:
    """Cached relevant context - memoized by SalsaDB."""
    from tldr.api import get_relevant_context
    from dataclasses import asdict
    result = get_relevant_context(project, entry, language=language, depth=depth)
    # Convert RelevantContext dataclass to dict for JSON serialization
    return {"status": "ok", "result": asdict(result)}


@salsa_query
def cached_imports(db: SalsaDB, file_path: str, language: str) -> dict:
    """Cached imports extraction - memoized by SalsaDB."""
    from tldr.api import get_imports
    result = get_imports(file_path, language=language)
    return {"status": "ok", "imports": result}


@salsa_query
def cached_importers(db: SalsaDB, project: str, module: str, language: str) -> dict:
    """Cached reverse import lookup - memoized by SalsaDB."""
    from tldr.api import get_imports, scan_project_files

    files = scan_project_files(project, language=language)
    importers = []
    project_path = Path(project)

    for file_path in files:
        try:
            imports = get_imports(file_path, language=language)
            for imp in imports:
                mod = imp.get("module", "")
                names = imp.get("names", [])
                if module in mod or module in names:
                    importers.append({
                        "file": str(Path(file_path).relative_to(project_path)),
                        "import": imp,
                    })
        except Exception:
            pass

    return {"status": "ok", "module": module, "importers": importers}
