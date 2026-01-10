"""
Change Impact Analysis for TLDR.

Determines which tests to run based on changed files.
Uses session-based tracking (dirty_flag) or explicit file list.
"""

import subprocess
from pathlib import Path

from .analysis import analyze_impact
from .api import extract_file, get_imports, scan_project_files
from .dirty_flag import get_dirty_files


def get_changed_functions(
    file_path: str,
    language: str = "python",
) -> list[dict]:
    """
    Extract function names from a file.

    Returns list of {"name": str, "file": str}
    """
    try:
        # extract_file auto-detects language, doesn't take language arg
        result = extract_file(file_path)
        functions = []
        for func in result.get("functions", []):
            name = func.get("name", "")
            if name:
                functions.append(
                    {
                        "name": name,
                        "file": file_path,
                    }
                )
        return functions
    except Exception:
        return []


def is_test_file(file_path: str) -> bool:
    """
    Check if a file is a test file based on naming conventions.

    Uses fast string methods instead of regex for ~18x speedup.
    """
    path = Path(file_path)
    name = path.name.lower()

    # Python tests
    if name.endswith(".py"):
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if name in ("test.py", "tests.py", "conftest.py"):
            return True

    # JavaScript/TypeScript tests
    if name.endswith((".js", ".jsx", ".ts", ".tsx")):
        if ".test." in name or ".spec." in name:
            return True
        if name.startswith("test_") or name.endswith(
            ("_test.js", "_test.jsx", "_test.ts", "_test.tsx")
        ):
            return True

    # Go tests
    if name.endswith("_test.go"):
        return True

    # Rust tests (common patterns)
    if name.endswith(".rs") and (name.startswith("test_") or name == "tests.rs"):
        return True

    # Check if in test directory (case-insensitive)
    parts_lower = [p.lower() for p in path.parts]
    return "tests" in parts_lower or "test" in parts_lower or "__tests__" in parts_lower


def get_module_name(file_path: str, project_path: str) -> str | None:
    """
    Convert file path to Python module name.

    E.g., "src/foo/bar.py" -> "src.foo.bar" or "foo.bar"
    """
    try:
        path = Path(file_path)
        project = Path(project_path).resolve()

        # Get relative path
        if path.is_absolute():
            rel = path.relative_to(project)
        else:
            rel = path

        # Remove .py extension and convert to module
        parts = list(rel.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # Skip __init__
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        return ".".join(parts) if parts else None
    except Exception:
        return None


def find_tests_importing_module(
    project_path: str,
    module_name: str,
    language: str = "python",
) -> list[str]:
    """
    Find test files that import a given module.
    """
    if not module_name:
        return []

    project = Path(project_path).resolve()
    importing_tests = []

    try:
        all_files = scan_project_files(str(project), language=language)
        test_files = [f for f in all_files if is_test_file(f)]

        for test_file in test_files:
            try:
                imports = get_imports(test_file, language=language)
                for imp in imports:
                    imported_module = imp.get("module", "")
                    # Check if the import matches our module (exact or prefix)
                    if imported_module == module_name or imported_module.startswith(
                        f"{module_name}."
                    ):
                        try:
                            rel_path = Path(test_file).relative_to(project)
                            importing_tests.append(str(rel_path))
                        except ValueError:
                            importing_tests.append(test_file)
                        break
            except Exception:
                continue
    except Exception:
        pass

    return importing_tests


def find_affected_tests(
    project_path: str,
    changed_files: list[str],
    language: str = "python",
    max_depth: int = 5,
) -> dict:
    """
    Find test files affected by changes to the given files.

    Args:
        project_path: Root directory of the project
        changed_files: List of file paths that were modified
        language: Programming language
        max_depth: Max depth for call graph traversal

    Returns:
        Dict with affected_tests, changed_functions, and metadata
    """
    project = Path(project_path).resolve()
    affected_tests = set()
    all_changed_functions = []

    # Get all functions from changed files
    for file_path in changed_files:
        abs_path = (
            (project / file_path).resolve()
            if not Path(file_path).is_absolute()
            else Path(file_path)
        )
        if not abs_path.exists():
            continue

        functions = get_changed_functions(str(abs_path), language=language)
        all_changed_functions.extend(functions)

        # If the changed file IS a test file, include it directly
        if is_test_file(str(abs_path)):
            try:
                rel_path = abs_path.relative_to(project)
                affected_tests.add(str(rel_path))
            except ValueError:
                affected_tests.add(str(abs_path))

    # For each changed function, find callers and filter to test files
    for func_info in all_changed_functions:
        func_name = func_info["name"]
        if not func_name:
            continue

        try:
            impact = analyze_impact(
                str(project),
                func_name,
                max_depth=max_depth,
                language=language,
            )

            # Walk the caller tree and collect test files
            def collect_test_files(node: dict):
                if not node:
                    return
                file_path = node.get("file", "")
                if file_path and is_test_file(file_path):
                    try:
                        rel_path = Path(file_path).relative_to(project)
                        affected_tests.add(str(rel_path))
                    except ValueError:
                        affected_tests.add(file_path)

                for caller in node.get("callers", []):
                    collect_test_files(caller)

            collect_test_files(impact.get("callers", {}))

        except Exception:
            # If impact analysis fails for a function, continue
            pass

    # Also find tests that import from changed modules (backup method)
    for file_path in changed_files:
        abs_path = (
            (project / file_path).resolve()
            if not Path(file_path).is_absolute()
            else Path(file_path)
        )
        module_name = get_module_name(str(abs_path), str(project))
        if module_name:
            importing_tests = find_tests_importing_module(
                str(project), module_name, language
            )
            affected_tests.update(importing_tests)

    # Count total test files for skip calculation
    all_test_files = []
    try:
        all_files = scan_project_files(str(project), language=language)
        all_test_files = [f for f in all_files if is_test_file(f)]
    except Exception:
        pass

    affected_list = sorted(affected_tests)
    skipped_count = len(all_test_files) - len(affected_list)

    # Build test command (as list to avoid shell injection)
    if language == "python":
        if affected_list:
            test_cmd = ["pytest"] + affected_list
        else:
            test_cmd = ["pytest"]
    elif language in ("typescript", "javascript"):
        if affected_list:
            test_cmd = ["npm", "test", "--"] + affected_list
        else:
            test_cmd = ["npm", "test"]
    else:
        test_cmd = None

    return {
        "changed_files": changed_files,
        "changed_functions": [f["name"] for f in all_changed_functions],
        "affected_tests": affected_list,
        "affected_count": len(affected_list),
        "skipped_count": max(0, skipped_count),
        "total_tests": len(all_test_files),
        "test_command": test_cmd,
    }


def get_git_changed_files(project_path: str, base: str = "HEAD~1") -> list[str]:
    """
    Get list of changed files from git diff.

    Args:
        project_path: Project root
        base: Git ref to diff against (default: HEAD~1)

    Returns:
        List of changed file paths (relative to project)
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base],
            capture_output=True,
            text=True,
            cwd=project_path,
            timeout=10,
        )
        if result.returncode == 0:
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return files
    except Exception:
        pass
    return []


def analyze_change_impact(
    project_path: str,
    files: list[str] | None = None,
    use_session: bool = False,
    use_git: bool = False,
    git_base: str = "HEAD~1",
    language: str = "python",
    max_depth: int = 5,
) -> dict:
    """
    Main entry point for change impact analysis.

    Args:
        project_path: Root directory of the project
        files: Explicit list of changed files (optional)
        use_session: Use dirty_flag to get session-modified files
        use_git: Use git diff to get changed files
        git_base: Git ref to diff against (default: HEAD~1)
        language: Programming language
        max_depth: Max depth for call graph traversal

    Returns:
        Dict with affected tests and metadata
    """
    project = Path(project_path).resolve()

    # Determine changed files
    changed_files = []
    source = "explicit"

    if files:
        changed_files = files
        source = "explicit"
    elif use_session:
        changed_files = get_dirty_files(project)
        source = "session"
    elif use_git:
        changed_files = get_git_changed_files(str(project), git_base)
        source = f"git:{git_base}"
    else:
        # Default: try session first, then git
        changed_files = get_dirty_files(project)
        if changed_files:
            source = "session"
        else:
            changed_files = get_git_changed_files(str(project))
            source = "git:HEAD~1" if changed_files else "none"

    if not changed_files:
        return {
            "changed_files": [],
            "changed_functions": [],
            "affected_tests": [],
            "affected_count": 0,
            "skipped_count": 0,
            "total_tests": 0,
            "test_command": None,
            "source": source,
            "message": "No changed files detected",
        }

    # Filter to source files only (not tests, configs, etc.)
    source_extensions = {
        "python": {".py"},
        "typescript": {".ts", ".tsx"},
        "javascript": {".js", ".jsx"},
        "go": {".go"},
        "rust": {".rs"},
    }
    valid_exts = source_extensions.get(language, {".py"})
    source_files = [
        f for f in changed_files if Path(f).suffix in valid_exts and not is_test_file(f)
    ]

    # Also include test files that changed directly
    test_files = [f for f in changed_files if is_test_file(f)]

    result = find_affected_tests(
        str(project),
        source_files + test_files,
        language=language,
        max_depth=max_depth,
    )
    result["source"] = source

    return result
