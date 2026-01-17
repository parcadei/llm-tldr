#!/usr/bin/env python3
"""
TLDR-Code CLI - Token-efficient code analysis for LLMs.

Usage:
    tldr tree [path]                    Show file tree
    tldr structure [path]               Show code structure (codemaps)
    tldr search <pattern> [path]        Search files for pattern
    tldr extract <file>                 Extract full file info
    tldr context <entry> [--project]    Get relevant context for LLM
    tldr cfg <file> <function>          Control flow graph
    tldr dfg <file> <function>          Data flow graph
    tldr slice <file> <func> <line>     Program slice
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Fix for Windows: Explicitly import tree-sitter bindings early to prevent
# silent DLL loading failures when running as a console script entry point.
if os.name == 'nt':
    try:
        import tree_sitter
        import tree_sitter_python
        import tree_sitter_javascript
        import tree_sitter_typescript
    except ImportError:
        pass

from . import __version__


def _get_subprocess_detach_kwargs():
    """Get platform-specific kwargs for detaching subprocess."""
    import subprocess
    if os.name == 'nt':  # Windows
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    else:  # Unix (Mac/Linux)
        return {'start_new_session': True}


# Extension to language mapping for auto-detection
EXTENSION_TO_LANGUAGE = {
    '.java': 'java',
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.hpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hh': 'cpp',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.cs': 'csharp',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.scala': 'scala',
    '.sc': 'scala',
    '.lua': 'lua',
    '.luau': 'luau',
    '.ex': 'elixir',
    '.exs': 'elixir',
}


def detect_language_from_extension(file_path: str) -> str:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the source file

    Returns:
        Language name (defaults to 'python' if unknown)
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, 'python')


def _show_first_run_tip():
    """Show a one-time tip about Swift support on first run."""
    marker = Path.home() / ".tldr_first_run"
    if marker.exists():
        return

    # Check if Swift is already installed
    try:
        import tree_sitter_swift
        # Swift already works, no tip needed
        marker.touch()
        return
    except ImportError:
        pass

    # Show tip
    import sys
    print("Tip: For Swift support, run: python -m tldr.install_swift", file=sys.stderr)
    print("     (This message appears once)", file=sys.stderr)
    print(file=sys.stderr)

    marker.touch()


def main():
    _show_first_run_tip()
    parser = argparse.ArgumentParser(
        prog="tldr",
        description="Token-efficient code analysis for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Version: %(prog)s """ + __version__ + """

Examples:
    tldr tree src/                      # File tree for src/
    tldr structure . --lang python      # Code structure for Python files
    tldr search "def process" .         # Search for pattern
    tldr extract src/main.py            # Full file analysis
    tldr context main --project .       # LLM context starting from main()
    tldr cfg src/main.py process        # Control flow for process()
    tldr slice src/main.py func 42      # Lines affecting line 42

Ignore Patterns:
    TLDR respects .tldrignore files (gitignore syntax).
    First run creates .tldrignore with sensible defaults.
    Use --ignore PATTERN to add patterns from CLI (repeatable).
    Use --no-ignore to bypass all ignore patterns.

Daemon:
    TLDR runs a per-project daemon for fast repeated queries.
    - Socket: /tmp/tldr-{hash}.sock (hash from project path)
    - Auto-shutdown: 30 minutes idle
    - Memory: ~50-100MB base, +500MB-1GB with semantic search

    Start explicitly:  tldr daemon start
    Check status:      tldr daemon status
    Stop:              tldr daemon stop

Semantic Search:
    First run downloads embedding model (1.3GB default).
    Use --model all-MiniLM-L6-v2 for smaller 80MB model.
    Set TLDR_AUTO_DOWNLOAD=1 to skip download prompts.
        """,
    )

    # Global flags
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--no-ignore",
        action="store_true",
        help="Ignore .tldrignore patterns (include all files)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        metavar="PATTERN",
        help="Additional ignore patterns (gitignore syntax, can be repeated)",
    )

    # Shell completion support
    try:
        import shtab
        shtab.add_argument_to(parser, ["--print-completion", "-s"])
    except ImportError:
        pass  # shtab is optional

    subparsers = parser.add_subparsers(dest="command", required=True)

    # tldr tree [path]
    tree_p = subparsers.add_parser("tree", help="Show file tree")
    tree_p.add_argument("path", nargs="?", default=".", help="Directory to scan")
    tree_p.add_argument(
        "--ext", nargs="+", help="Filter by extensions (e.g., --ext .py .ts)"
    )
    tree_p.add_argument(
        "--show-hidden", action="store_true", help="Include hidden files"
    )

    # tldr structure [path]
    struct_p = subparsers.add_parser("structure", help="Show code structure (codemaps)")
    struct_p.add_argument("path", nargs="?", default=".", help="Directory to analyze")
    struct_p.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "all", "python", "typescript", "javascript", "go", "rust", "java", "c",
                 "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir"],
        help="Language to analyze (auto=use cached, all=detect all)",
    )
    struct_p.add_argument(
        "--max", type=int, default=50, help="Max files to analyze (default: 50)"
    )

    # tldr search <pattern> [path]
    search_p = subparsers.add_parser("search", help="Search files for pattern")
    search_p.add_argument("pattern", help="Regex pattern to search")
    search_p.add_argument("path", nargs="?", default=".", help="Directory to search")
    search_p.add_argument("--ext", nargs="+", help="Filter by extensions")
    search_p.add_argument(
        "-C", "--context", type=int, default=0, help="Context lines around match"
    )
    search_p.add_argument(
        "--max", type=int, default=100, help="Max results (default: 100, 0=unlimited)"
    )
    search_p.add_argument(
        "--max-files", type=int, default=10000, help="Max files to scan (default: 10000)"
    )

    # tldr extract <file> [--class X] [--function Y] [--method Class.method]
    extract_p = subparsers.add_parser("extract", help="Extract full file info")
    extract_p.add_argument("file", help="File to analyze")
    extract_p.add_argument("--class", dest="filter_class", help="Filter to specific class")
    extract_p.add_argument("--function", dest="filter_function", help="Filter to specific function")
    extract_p.add_argument("--method", dest="filter_method", help="Filter to specific method (Class.method)")
    extract_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr context <entry>
    ctx_p = subparsers.add_parser("context", help="Get relevant context for LLM")
    ctx_p.add_argument("entry", help="Entry point (function_name or Class.method)")
    ctx_p.add_argument("--project", default=".", help="Project root directory")
    ctx_p.add_argument("--depth", type=int, default=2, help="Call depth (default: 2)")
    ctx_p.add_argument(
        "--lang",
        default="python",
        choices=["python", "typescript", "javascript", "go", "rust", "java", "c",
                 "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir"],
        help="Language",
    )

    # tldr cfg <file> <function>
    cfg_p = subparsers.add_parser("cfg", help="Control flow graph")
    cfg_p.add_argument("file", help="Source file")
    cfg_p.add_argument("function", help="Function name")
    cfg_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr dfg <file> <function>
    dfg_p = subparsers.add_parser("dfg", help="Data flow graph")
    dfg_p.add_argument("file", help="Source file")
    dfg_p.add_argument("function", help="Function name")
    dfg_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr slice <file> <function> <line>
    slice_p = subparsers.add_parser("slice", help="Program slice")
    slice_p.add_argument("file", help="Source file")
    slice_p.add_argument("function", help="Function name")
    slice_p.add_argument("line", type=int, help="Line number to slice from")
    slice_p.add_argument(
        "--direction",
        default="backward",
        choices=["backward", "forward"],
        help="Slice direction",
    )
    slice_p.add_argument("--var", help="Variable to track (optional)")
    slice_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr calls <path>
    calls_p = subparsers.add_parser("calls", help="Build cross-file call graph")
    calls_p.add_argument("path", nargs="?", default=".", help="Project root")
    calls_p.add_argument("--lang", default="auto", help="Language (auto=cached, all=detect)")

    # tldr impact <func> [path]
    impact_p = subparsers.add_parser(
        "impact", help="Find all callers of a function (reverse call graph)"
    )
    impact_p.add_argument("func", help="Function name to find callers of")
    impact_p.add_argument("path", nargs="?", default=None, help="Project root")
    impact_p.add_argument("--project", dest="project_path", default=".", help="Project root (alternative to positional path)")
    impact_p.add_argument("--depth", type=int, default=3, help="Max depth (default: 3)")
    impact_p.add_argument("--file", help="Filter by file containing this string")
    impact_p.add_argument("--lang", default="auto", help="Language (auto=cached, all=detect)")

    # tldr dead [path]
    dead_p = subparsers.add_parser("dead", help="Find unreachable (dead) code")
    dead_p.add_argument("path", nargs="?", default=".", help="Project root")
    dead_p.add_argument(
        "--entry", nargs="*", default=[], help="Additional entry point patterns"
    )
    dead_p.add_argument("--lang", default="auto", help="Language (auto=cached, all=detect)")

    # tldr arch [path]
    arch_p = subparsers.add_parser(
        "arch", help="Detect architectural layers from call patterns"
    )
    arch_p.add_argument("path", nargs="?", default=".", help="Project root")
    arch_p.add_argument("--lang", default="auto", help="Language (auto=cached, all=detect)")

    # tldr imports <file>
    imports_p = subparsers.add_parser(
        "imports", help="Parse imports from a source file"
    )
    imports_p.add_argument("file", help="Source file to analyze")
    imports_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr importers <module> [path]
    importers_p = subparsers.add_parser(
        "importers", help="Find all files that import a module (reverse import lookup)"
    )
    importers_p.add_argument("module", help="Module name to search for importers")
    importers_p.add_argument("path", nargs="?", default=".", help="Project root")
    importers_p.add_argument("--lang", default="python", help="Language")

    # tldr change-impact [files...]
    impact_p = subparsers.add_parser(
        "change-impact", help="Find tests affected by changed files"
    )
    impact_p.add_argument(
        "files", nargs="*", help="Files to analyze (default: auto-detect from session/git)"
    )
    impact_p.add_argument(
        "--session", action="store_true", help="Use session-modified files (dirty_flag)"
    )
    impact_p.add_argument(
        "--git", action="store_true", help="Use git diff to find changed files"
    )
    impact_p.add_argument(
        "--git-base", default="HEAD~1", help="Git ref to diff against (default: HEAD~1)"
    )
    impact_p.add_argument("--lang", default="python", help="Language")
    impact_p.add_argument(
        "--depth", type=int, default=5, help="Max call graph depth (default: 5)"
    )
    impact_p.add_argument(
        "--run", action="store_true", help="Actually run the affected tests"
    )

    # tldr diagnostics <file|path>
    diag_p = subparsers.add_parser(
        "diagnostics", help="Get type and lint diagnostics"
    )
    diag_p.add_argument("target", help="File or project directory to check")
    diag_p.add_argument(
        "--project", action="store_true", help="Check entire project (default: single file)"
    )
    diag_p.add_argument(
        "--no-lint", action="store_true", help="Skip linter, only run type checker"
    )
    diag_p.add_argument(
        "--format", choices=["json", "text"], default="json", help="Output format"
    )
    diag_p.add_argument("--lang", default=None, help="Override language detection")

    # tldr warm <path>
    warm_p = subparsers.add_parser(
        "warm", help="Pre-build call graph cache for faster queries"
    )
    warm_p.add_argument("path", help="Project root directory")
    warm_p.add_argument(
        "--background", action="store_true", help="Build in background process"
    )
    warm_p.add_argument(
        "--lang",
        default="all",
        choices=["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir", "all"],
        help="Language (default: auto-detect all)",
    )

    # tldr semantic index <path> / tldr semantic search <query>
    semantic_p = subparsers.add_parser(
        "semantic", help="Semantic code search using embeddings"
    )
    semantic_sub = semantic_p.add_subparsers(dest="action", required=True)

    # tldr semantic index [path]
    index_p = semantic_sub.add_parser("index", help="Build semantic index for project")
    index_p.add_argument("path", nargs="?", default=".", help="Project root")
    index_p.add_argument(
        "--lang",
        default="python",
        choices=["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir", "all"],
        help="Language (use 'all' for multi-language)",
    )
    index_p.add_argument(
        "--model",
        default=None,
        help="Embedding model: bge-large-en-v1.5 (1.3GB, default) or all-MiniLM-L6-v2 (80MB)",
    )

    # tldr semantic search <query>
    search_p = semantic_sub.add_parser("search", help="Search semantically")
    search_p.add_argument("query", help="Natural language query")
    search_p.add_argument("--path", default=".", help="Project root")
    search_p.add_argument("--k", type=int, default=5, help="Number of results")
    search_p.add_argument("--expand", action="store_true", help="Include call graph expansion")
    search_p.add_argument("--lang", default="python", help="Language")
    search_p.add_argument(
        "--model",
        default=None,
        help="Embedding model (uses index model if not specified)",
    )

    # tldr daemon start/stop/status/query
    daemon_p = subparsers.add_parser(
        "daemon", help="Daemon management subcommands"
    )
    daemon_sub = daemon_p.add_subparsers(dest="action", required=True)

    # tldr daemon start [--project PATH]
    daemon_start_p = daemon_sub.add_parser("start", help="Start daemon for project (background)")
    daemon_start_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr daemon stop [--project PATH]
    daemon_stop_p = daemon_sub.add_parser("stop", help="Stop daemon gracefully")
    daemon_stop_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr daemon status [--project PATH]
    daemon_status_p = daemon_sub.add_parser("status", help="Check if daemon running")
    daemon_status_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr daemon query CMD [--project PATH]
    # or: tldr daemon query --json '{"cmd":"semantic","action":"search",...}'
    daemon_query_p = daemon_sub.add_parser("query", help="Send command to daemon")
    daemon_query_p.add_argument(
        "cmd",
        nargs="?",
        help="Simple command name (e.g., ping, status). Ignored if --json is provided.",
    )
    daemon_query_p.add_argument(
        "--json",
        help="Raw JSON command to send (e.g., '{\"cmd\":\"semantic\",\"action\":\"search\",\"query\":\"...\"}')",
    )
    daemon_query_p.add_argument(
        "--project", "-p", default=".", help="Project path (default: current directory)"
    )

    # tldr daemon notify FILE [--project PATH]
    daemon_notify_p = daemon_sub.add_parser("notify", help="Notify daemon of file change (triggers reindex at threshold)")
    daemon_notify_p.add_argument("file", help="Path to changed file")
    daemon_notify_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr doctor [--install LANG]
    doctor_p = subparsers.add_parser(
        "doctor", help="Check and install diagnostic tools (type checkers, linters)"
    )
    doctor_p.add_argument(
        "--install", metavar="LANG", help="Install missing tools for language (e.g., python, go)"
    )
    doctor_p.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from .api import (
        build_project_call_graph,
        extract_file,
        get_cfg_context,
        get_code_structure,
        get_dfg_context,
        get_file_tree,
        get_imports,
        get_relevant_context,
        get_slice,
        scan_project_files,
        search as api_search,
    )
    from .analysis import (
        analyze_architecture,
        analyze_dead_code,
        analyze_impact,
    )
    from .dirty_flag import is_dirty, get_dirty_files, clear_dirty
    from .patch import patch_call_graph
    from .cross_file_calls import ProjectCallGraph

    def _get_or_build_graph(project_path, lang, build_fn):
        """Get cached graph with incremental patches, or build fresh.

        This implements P4 incremental updates:
        1. If no cache exists, do full build
        2. If cache exists but no dirty files, load cache
        3. If cache exists with dirty files, patch incrementally
        """
        import time
        project = Path(project_path).resolve()
        cache_dir = project / ".tldr" / "cache"
        cache_file = cache_dir / "call_graph.json"

        # Check if we have a cached graph
        if cache_file.exists():
            try:
                cache_data = json.loads(cache_file.read_text())
                
                # Validate cache language compatibility
                cache_langs = cache_data.get("languages", [])
                if cache_langs and lang not in cache_langs and lang != "all":
                    # Cache was built with different languages; rebuild
                    raise ValueError("Cache language mismatch")
                
                # Reconstruct graph from cache
                graph = ProjectCallGraph()
                for e in cache_data.get("edges", []):
                    graph.add_edge(e["from_file"], e["from_func"], e["to_file"], e["to_func"])

                # Check for dirty files
                if is_dirty(project):
                    dirty_files = get_dirty_files(project)
                    # Patch incrementally for each dirty file
                    for rel_file in dirty_files:
                        abs_file = project / rel_file
                        if abs_file.exists():
                            graph = patch_call_graph(graph, str(abs_file), str(project), lang=lang)

                    # Update cache with patched graph
                    cache_data = {
                        "edges": [
                            {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                            for e in graph.edges
                        ],
                        "languages": cache_langs if cache_langs else [lang],
                        "timestamp": time.time(),
                    }
                    cache_file.write_text(json.dumps(cache_data, indent=2))

                    # Clear dirty flag
                    clear_dirty(project)

                return graph
            except (json.JSONDecodeError, KeyError, ValueError):
                # Invalid cache or language mismatch, fall through to fresh build
                pass

        # No cache or invalid cache - do fresh build
        graph = build_fn(project_path, language=lang)

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "edges": [
                {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                for e in graph.edges
            ],
            "languages": [lang],
            "timestamp": time.time(),
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))

        # Clear any dirty flag since we just rebuilt
        clear_dirty(project)

        return graph

    # Helper to load ignore patterns from .tldrignore + CLI --ignore flags + .gitignore
    def get_ignore_spec(project_path: str | Path):
        """Load ignore patterns, combining .tldrignore, .gitignore, and CLI --ignore flags."""
        if getattr(args, 'no_ignore', False):
            return None

        from .tldrignore import IgnoreSpec

        cli_patterns = getattr(args, 'ignore', None) or []
        return IgnoreSpec(
            project_dir=project_path,
            use_gitignore=True,
            cli_patterns=cli_patterns if cli_patterns else None,
        )

    def get_cached_languages(project_path: str | Path) -> list[str] | None:
        """Read cached languages from .tldr/languages.json if available."""
        lang_cache = Path(project_path) / ".tldr" / "languages.json"
        if lang_cache.exists():
            try:
                data = json.loads(lang_cache.read_text())
                return data.get("languages")
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def resolve_language(lang_arg: str, project_path: str | Path) -> str:
        """Resolve 'auto'/'all' to actual language. Returns first language for single-lang commands."""
        project_path = Path(project_path).resolve()
        if lang_arg == "auto":
            # Try cache first, then detect if no cache
            cached = get_cached_languages(project_path)
            if cached:
                return cached[0]
            # No cache - detect languages
            from .semantic import _detect_project_languages
            respect_ignore = not getattr(args, 'no_ignore', False)
            langs = _detect_project_languages(project_path, respect_ignore=respect_ignore)
            return langs[0] if langs else "python"
        elif lang_arg == "all":
            from .semantic import _detect_project_languages
            respect_ignore = not getattr(args, 'no_ignore', False)
            langs = _detect_project_languages(project_path, respect_ignore=respect_ignore)
            return langs[0] if langs else "python"
        return lang_arg

    try:
        if args.command == "tree":
            ext = set(args.ext) if args.ext else None
            ignore_spec = get_ignore_spec(args.path)
            result = get_file_tree(
                args.path, extensions=ext, exclude_hidden=not args.show_hidden,
                ignore_spec=ignore_spec
            )
            print(json.dumps(result, indent=2))

        elif args.command == "structure":
            ignore_spec = get_ignore_spec(args.path)
            project_path = Path(args.path).resolve()

            # Determine language(s) to analyze
            if args.lang == "auto":
                # Use cached languages, or detect if no cache
                cached = get_cached_languages(project_path)
                if cached:
                    languages = cached
                else:
                    from .semantic import _detect_project_languages
                    respect_ignore = not getattr(args, 'no_ignore', False)
                    languages = _detect_project_languages(project_path, respect_ignore=respect_ignore)
                    if not languages:
                        languages = ["python"]
            elif args.lang == "all":
                # Detect all languages in project
                from .semantic import _detect_project_languages
                respect_ignore = not getattr(args, 'no_ignore', False)
                languages = _detect_project_languages(project_path, respect_ignore=respect_ignore)
                if not languages:
                    languages = ["python"]
            else:
                languages = [args.lang]

            # Collect results for all languages
            all_files = []
            for lang in languages:
                result = get_code_structure(
                    args.path, language=lang, max_results=args.max,
                    ignore_spec=ignore_spec
                )
                all_files.extend(result.get("files", []))

            combined_result = {
                "root": str(project_path),
                "languages": languages,
                "files": all_files[:args.max],  # Respect max across all languages
            }
            print(json.dumps(combined_result, indent=2))

        elif args.command == "search":
            ext = set(args.ext) if args.ext else None
            ignore_spec = get_ignore_spec(args.path)
            result = api_search(
                args.pattern, args.path,
                extensions=ext,
                context_lines=args.context,
                max_results=args.max,
                max_files=args.max_files,
                ignore_spec=ignore_spec,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "extract":
            result = extract_file(args.file)

            # Apply filters if specified
            filter_class = getattr(args, "filter_class", None)
            filter_function = getattr(args, "filter_function", None)
            filter_method = getattr(args, "filter_method", None)

            if filter_class or filter_function or filter_method:
                # Filter classes
                if filter_class:
                    result["classes"] = [
                        c for c in result.get("classes", [])
                        if c.get("name") == filter_class
                    ]
                elif filter_method:
                    # Parse Class.method syntax
                    parts = filter_method.split(".", 1)
                    if len(parts) == 2:
                        class_name, method_name = parts
                        filtered_classes = []
                        for c in result.get("classes", []):
                            if c.get("name") == class_name:
                                # Filter to only the requested method
                                c_copy = dict(c)
                                c_copy["methods"] = [
                                    m for m in c.get("methods", [])
                                    if m.get("name") == method_name
                                ]
                                filtered_classes.append(c_copy)
                        result["classes"] = filtered_classes
                else:
                    # No class filter, clear classes
                    result["classes"] = []

                # Filter functions
                if filter_function:
                    result["functions"] = [
                        f for f in result.get("functions", [])
                        if f.get("name") == filter_function
                    ]
                elif not filter_method:
                    # No function filter (and not method filter), clear functions if class filter active
                    if filter_class:
                        result["functions"] = []

            print(json.dumps(result, indent=2))

        elif args.command == "context":
            ctx = get_relevant_context(
                args.project, args.entry, depth=args.depth, language=args.lang
            )
            # Output LLM-ready string directly
            print(ctx.to_llm_string())

        elif args.command == "cfg":
            lang = args.lang or detect_language_from_extension(args.file)
            result = get_cfg_context(args.file, args.function, language=lang)
            print(json.dumps(result, indent=2))

        elif args.command == "dfg":
            lang = args.lang or detect_language_from_extension(args.file)
            result = get_dfg_context(args.file, args.function, language=lang)
            print(json.dumps(result, indent=2))

        elif args.command == "slice":
            lang = args.lang or detect_language_from_extension(args.file)
            lines = get_slice(
                args.file,
                args.function,
                args.line,
                direction=args.direction,
                variable=args.var,
                language=lang,
            )
            result = {"lines": sorted(lines), "count": len(lines)}
            print(json.dumps(result, indent=2))

        elif args.command == "calls":
            # Check for cached graph and dirty files for incremental update
            lang = resolve_language(args.lang, args.path)
            graph = _get_or_build_graph(args.path, lang, build_project_call_graph)
            result = {
                "edges": [
                    {
                        "from_file": e[0],
                        "from_func": e[1],
                        "to_file": e[2],
                        "to_func": e[3],
                    }
                    for e in graph.edges
                ],
                "count": len(graph.edges),
            }
            print(json.dumps(result, indent=2))

        elif args.command == "impact":
            # Support both positional path and --project flag
            project_root = args.path if args.path else args.project_path
            lang = resolve_language(args.lang, project_root)
            result = analyze_impact(
                project_root,
                args.func,
                max_depth=args.depth,
                target_file=args.file,
                language=lang,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "dead":
            lang = resolve_language(args.lang, args.path)
            result = analyze_dead_code(
                args.path,
                entry_points=args.entry if args.entry else None,
                language=lang,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "arch":
            lang = resolve_language(args.lang, args.path)
            result = analyze_architecture(args.path, language=lang)
            print(json.dumps(result, indent=2))

        elif args.command == "imports":
            file_path = Path(args.file).resolve()
            if not file_path.exists():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            lang = args.lang or detect_language_from_extension(args.file)
            result = get_imports(str(file_path), language=lang)
            print(json.dumps(result, indent=2))

        elif args.command == "importers":
            # Find all files that import the given module
            project = Path(args.path).resolve()
            if not project.exists():
                print(f"Error: Path not found: {args.path}", file=sys.stderr)
                sys.exit(1)

            # Scan all source files and check their imports
            respect_ignore = not getattr(args, 'no_ignore', False)
            files = scan_project_files(str(project), language=args.lang, respect_ignore=respect_ignore)
            importers = []
            for file_path in files:
                try:
                    imports = get_imports(file_path, language=args.lang)
                    for imp in imports:
                        module = imp.get("module", "")
                        names = imp.get("names", [])
                        # Check if module matches or if any imported name matches
                        if args.module in module or args.module in names:
                            importers.append({
                                "file": str(Path(file_path).relative_to(project)),
                                "import": imp,
                            })
                except Exception:
                    # Skip files that can't be parsed
                    pass

            print(json.dumps({"module": args.module, "importers": importers}, indent=2))

        elif args.command == "change-impact":
            from .change_impact import analyze_change_impact

            result = analyze_change_impact(
                project_path=".",
                files=args.files if args.files else None,
                use_session=args.session,
                use_git=args.git,
                git_base=args.git_base,
                language=args.lang,
                max_depth=args.depth,
            )

            if args.run and result.get("test_command"):
                # Actually run the tests (test_command is a list to avoid shell injection)
                import shlex
                import subprocess as sp
                cmd = result["test_command"]
                print(f"Running: {shlex.join(cmd)}", file=sys.stderr)
                sp.run(cmd)  # No shell=True - safe from injection
            else:
                print(json.dumps(result, indent=2))

        elif args.command == "diagnostics":
            from .diagnostics import (
                get_diagnostics,
                get_project_diagnostics,
                format_diagnostics_for_llm,
            )

            target = Path(args.target).resolve()
            if not target.exists():
                print(f"Error: Target not found: {args.target}", file=sys.stderr)
                sys.exit(1)

            if args.project or target.is_dir():
                result = get_project_diagnostics(
                    str(target),
                    language=args.lang or "python",
                    include_lint=not args.no_lint,
                )
            else:
                result = get_diagnostics(
                    str(target),
                    language=args.lang,
                    include_lint=not args.no_lint,
                )

            if args.format == "text":
                print(format_diagnostics_for_llm(result))
            else:
                print(json.dumps(result, indent=2))

        elif args.command == "warm":
            import os
            import subprocess
            import time

            project_path = Path(args.path).resolve()

            # Validate path exists
            if not project_path.exists():
                print(f"Error: Path not found: {args.path}", file=sys.stderr)
                sys.exit(1)

            if args.background:
                # Spawn background process (cross-platform)
                subprocess.Popen(
                    [sys.executable, "-m", "tldr.cli", "warm", str(project_path), "--lang", args.lang],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **_get_subprocess_detach_kwargs(),
                )
                print(f"Background indexing spawned for {project_path}")
            else:
                # Build call graph
                from .cross_file_calls import scan_project, ProjectCallGraph
                from .tldrignore import ensure_tldrignore

                # Ensure .tldrignore exists (create with defaults if not)
                created, msg = ensure_tldrignore(project_path)
                if created:
                    print(msg)

                respect_ignore = not getattr(args, 'no_ignore', False)
                
                # Determine languages to process
                if args.lang == "all":
                    try:
                        from .semantic import _detect_project_languages
                        target_languages = _detect_project_languages(project_path, respect_ignore=respect_ignore)
                        print(f"Detected languages: {', '.join(target_languages)}")
                    except ImportError:
                        # Fallback if semantic module issue
                        target_languages = ["python", "typescript", "javascript", "go", "rust"]
                else:
                    target_languages = [args.lang]

                all_files = set()
                combined_edges = []
                processed_languages = []
                
                for lang in target_languages:
                    try:
                        # Scan files
                        files = scan_project(project_path, language=lang, respect_ignore=respect_ignore)
                        all_files.update(files)
                        
                        # Build graph
                        graph = build_project_call_graph(project_path, language=lang)
                        combined_edges.extend([
                            {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                            for e in graph.edges
                        ])
                        print(f"Processed {lang}: {len(files)} files, {len(graph.edges)} edges")
                        processed_languages.append(lang)
                    except ValueError as e:
                        # Expected for unsupported languages
                        print(f"Warning: {lang}: {e}", file=sys.stderr)
                    except Exception as e:
                        # Unexpected error - show traceback if debug enabled
                        print(f"Warning: Failed to process {lang}: {e}", file=sys.stderr)
                        if os.environ.get("TLDR_DEBUG"):
                            import traceback
                            traceback.print_exc()

                # Create cache directory
                cache_dir = project_path / ".tldr" / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Save cache file
                cache_file = cache_dir / "call_graph.json"
                # Deduplicate edges
                unique_edges = list({(e["from_file"], e["from_func"], e["to_file"], e["to_func"]): e for e in combined_edges}.values())
                
                cache_data = {
                    "edges": unique_edges,
                    "languages": processed_languages if processed_languages else target_languages,
                    "timestamp": time.time(),
                }
                cache_file.write_text(json.dumps(cache_data, indent=2))

                # Also save quick-access language cache for structure/search auto-detect
                lang_cache_file = project_path / ".tldr" / "languages.json"
                lang_cache_file.write_text(json.dumps({
                    "languages": processed_languages if processed_languages else target_languages,
                    "timestamp": time.time(),
                }, indent=2))

                # Print stats
                print(f"Total: Indexed {len(all_files)} files, found {len(unique_edges)} edges")

        elif args.command == "semantic":
            from .semantic import build_semantic_index, semantic_search

            if args.action == "index":
                respect_ignore = not getattr(args, 'no_ignore', False)
                count = build_semantic_index(args.path, lang=args.lang, model=args.model, respect_ignore=respect_ignore)
                print(f"Indexed {count} code units")

            elif args.action == "search":
                results = semantic_search(
                    args.path,
                    args.query,
                    k=args.k,
                    expand_graph=args.expand,
                    model=args.model,
                )
                print(json.dumps(results, indent=2))

        elif args.command == "doctor":
            import shutil
            import subprocess

            # Tool definitions: language -> (type_checker, linter, install_commands)
            TOOL_INFO = {
                "python": {
                    "type_checker": ("pyright", "pip install pyright  OR  npm install -g pyright"),
                    "linter": ("ruff", "pip install ruff"),
                },
                "typescript": {
                    "type_checker": ("tsc", "npm install -g typescript"),
                    "linter": None,
                },
                "javascript": {
                    "type_checker": None,
                    "linter": ("eslint", "npm install -g eslint"),
                },
                "go": {
                    "type_checker": ("go", "https://go.dev/dl/"),
                    "linter": ("golangci-lint", "brew install golangci-lint  OR  go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"),
                },
                "rust": {
                    "type_checker": ("cargo", "https://rustup.rs/"),
                    "linter": ("cargo-clippy", "rustup component add clippy"),
                },
                "java": {
                    "type_checker": ("javac", "Install JDK: https://adoptium.net/"),
                    "linter": ("checkstyle", "brew install checkstyle  OR  download from checkstyle.org"),
                },
                "c": {
                    "type_checker": ("gcc", "xcode-select --install  OR  apt install gcc"),
                    "linter": ("cppcheck", "brew install cppcheck  OR  apt install cppcheck"),
                },
                "cpp": {
                    "type_checker": ("g++", "xcode-select --install  OR  apt install g++"),
                    "linter": ("cppcheck", "brew install cppcheck  OR  apt install cppcheck"),
                },
                "ruby": {
                    "type_checker": None,
                    "linter": ("rubocop", "gem install rubocop"),
                },
                "php": {
                    "type_checker": None,
                    "linter": ("phpstan", "composer global require phpstan/phpstan"),
                },
                "kotlin": {
                    "type_checker": ("kotlinc", "brew install kotlin  OR  sdk install kotlin"),
                    "linter": ("ktlint", "brew install ktlint"),
                },
                "swift": {
                    "type_checker": ("swiftc", "xcode-select --install"),
                    "linter": ("swiftlint", "brew install swiftlint"),
                },
                "csharp": {
                    "type_checker": ("dotnet", "https://dotnet.microsoft.com/download"),
                    "linter": None,
                },
                "scala": {
                    "type_checker": ("scalac", "brew install scala  OR  sdk install scala"),
                    "linter": None,
                },
                "elixir": {
                    "type_checker": ("elixir", "brew install elixir  OR  asdf install elixir"),
                    "linter": ("mix", "Included with Elixir"),
                },
                "lua": {
                    "type_checker": None,
                    "linter": ("luacheck", "luarocks install luacheck"),
                },
            }

            # Install commands for --install flag
            INSTALL_COMMANDS = {
                "python": ["pip", "install", "pyright", "ruff"],
                "go": ["go", "install", "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"],
                "rust": ["rustup", "component", "add", "clippy"],
                "ruby": ["gem", "install", "rubocop"],
                "kotlin": ["brew", "install", "kotlin", "ktlint"],
                "swift": ["brew", "install", "swiftlint"],
                "lua": ["luarocks", "install", "luacheck"],
            }

            if args.install:
                lang = args.install.lower()
                if lang not in INSTALL_COMMANDS:
                    print(f"Error: No auto-install available for '{lang}'", file=sys.stderr)
                    print(f"Available: {', '.join(sorted(INSTALL_COMMANDS.keys()))}", file=sys.stderr)
                    sys.exit(1)

                cmd = INSTALL_COMMANDS[lang]
                print(f"Installing tools for {lang}: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True)
                    print(f"✓ Installed {lang} tools")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Install failed: {e}", file=sys.stderr)
                    sys.exit(1)
                except FileNotFoundError:
                    print(f"✗ Command not found: {cmd[0]}", file=sys.stderr)
                    sys.exit(1)
            else:
                # Check all tools
                results = {}
                for lang, tools in TOOL_INFO.items():
                    lang_result = {"type_checker": None, "linter": None}

                    if tools["type_checker"]:
                        tool_name, install_cmd = tools["type_checker"]
                        path = shutil.which(tool_name)
                        lang_result["type_checker"] = {
                            "name": tool_name,
                            "installed": path is not None,
                            "path": path,
                            "install": install_cmd if not path else None,
                        }

                    if tools["linter"]:
                        tool_name, install_cmd = tools["linter"]
                        path = shutil.which(tool_name)
                        lang_result["linter"] = {
                            "name": tool_name,
                            "installed": path is not None,
                            "path": path,
                            "install": install_cmd if not path else None,
                        }

                    results[lang] = lang_result

                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print("TLDR Diagnostics Check")
                    print("=" * 50)
                    print()

                    missing_count = 0
                    for lang, checks in sorted(results.items()):
                        has_issues = False
                        lines = []

                        tc = checks["type_checker"]
                        if tc:
                            if tc["installed"]:
                                lines.append(f"  ✓ {tc['name']} - {tc['path']}")
                            else:
                                lines.append(f"  ✗ {tc['name']} - not found")
                                lines.append(f"    → {tc['install']}")
                                has_issues = True
                                missing_count += 1

                        linter = checks["linter"]
                        if linter:
                            if linter["installed"]:
                                lines.append(f"  ✓ {linter['name']} - {linter['path']}")
                            else:
                                lines.append(f"  ✗ {linter['name']} - not found")
                                lines.append(f"    → {linter['install']}")
                                has_issues = True
                                missing_count += 1

                        if lines:
                            print(f"{lang.capitalize()}:")
                            for line in lines:
                                print(line)
                            print()

                    if missing_count > 0:
                        print(f"Missing {missing_count} tool(s). Run: tldr doctor --install <lang>")
                    else:
                        print("All diagnostic tools installed!")

        elif args.command == "daemon":
            from .daemon import start_daemon, stop_daemon, query_daemon

            project_path = Path(args.project).resolve()

            if args.action == "start":
                # Ensure .tldr directory exists
                tldr_dir = project_path / ".tldr"
                tldr_dir.mkdir(parents=True, exist_ok=True)
                # Start daemon (will fork to background on Unix)
                start_daemon(project_path, foreground=False)

            elif args.action == "stop":
                if stop_daemon(project_path):
                    print("Daemon stopped")
                else:
                    print("Daemon not running")

            elif args.action == "status":
                try:
                    result = query_daemon(project_path, {"cmd": "status"})
                    print(f"Status: {result.get('status', 'unknown')}")
                    if 'uptime' in result:
                        uptime = int(result['uptime'])
                        mins, secs = divmod(uptime, 60)
                        hours, mins = divmod(mins, 60)
                        print(f"Uptime: {hours}h {mins}m {secs}s")
                except (ConnectionRefusedError, FileNotFoundError):
                    print("Daemon not running")

            elif args.action == "query":
                try:
                    # If --json is provided, send raw JSON payload to daemon.
                    # Otherwise, fall back to simple {"cmd": CMD} behavior.
                    if getattr(args, "json", None):
                        try:
                            command = json.loads(args.json)
                        except json.JSONDecodeError as e:
                            print(f"Error: invalid JSON for --json: {e}", file=sys.stderr)
                            sys.exit(1)
                    else:
                        if not args.cmd:
                            print("Error: either CMD or --json must be provided", file=sys.stderr)
                            sys.exit(1)
                        command = {"cmd": args.cmd}

                    result = query_daemon(project_path, command)
                    print(json.dumps(result, indent=2))
                except (ConnectionRefusedError, FileNotFoundError):
                    print("Error: Daemon not running", file=sys.stderr)
                    sys.exit(1)

            elif args.action == "notify":
                try:
                    file_path = Path(args.file).resolve()
                    result = query_daemon(project_path, {
                        "cmd": "notify",
                        "file": str(file_path)
                    })
                    if result.get("status") == "ok":
                        dirty = result.get("dirty_count", 0)
                        threshold = result.get("threshold", 20)
                        if result.get("reindex_triggered"):
                            print(f"Reindex triggered ({dirty}/{threshold} files)")
                        else:
                            print(f"Tracked: {dirty}/{threshold} files")
                    else:
                        print(f"Error: {result.get('message', 'Unknown error')}", file=sys.stderr)
                        sys.exit(1)
                except (ConnectionRefusedError, FileNotFoundError):
                    # Daemon not running - silently ignore, file edits shouldn't fail
                    pass

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
