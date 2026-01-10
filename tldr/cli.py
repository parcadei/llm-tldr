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
try:
    import tree_sitter
    try:
        import tree_sitter_typescript
        import tree_sitter_javascript
        import tree_sitter_python
    except ImportError:
        pass
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
}


def _detect_language(path):
    """Detect language from file path."""
    if isinstance(path, str):
        path = Path(path)
    return EXTENSION_TO_LANGUAGE.get(path.suffix.lower())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="tldr",
        description="Token-efficient code analysis for LLMs."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # === tree ===
    tree_parser = subparsers.add_parser("tree", help="Show file tree")
    tree_parser.add_argument("path", nargs="?", default=".", help="Project root (default: .)")
    tree_parser.add_argument("--ext", nargs="+", help="Filter by extensions (e.g. .py .ts)")
    tree_parser.add_argument("-a", "--all", action="store_true", dest="show_hidden", help="Show hidden files")
    tree_parser.add_argument("--max", type=int, default=100, help="Maximum number of files")

    # === structure ===
    struct_parser = subparsers.add_parser("structure", help="Show code structure (codemaps)")
    struct_parser.add_argument("path", nargs="?", default=".", help="Project root (default: .)")
    struct_parser.add_argument("--lang", help="Language filter (e.g. python, typescript)")
    struct_parser.add_argument("--max", type=int, default=100, help="Maximum files to analyze")
    struct_parser.add_argument("--json", action="store_true", help="Output JSON")

    # === search ===
    search_parser = subparsers.add_parser("search", help="Search files for pattern")
    search_parser.add_argument("pattern", help="Regex pattern")
    search_parser.add_argument("path", nargs="?", default=".", help="Project root (default: .)")
    search_parser.add_argument("--ext", nargs="+", help="Filter by extensions")
    search_parser.add_argument("-C", "--context", type=int, default=0, help="Context lines")
    search_parser.add_argument("--max", type=int, default=100, help="Maximum matches")

    # === extract ===
    extract_parser = subparsers.add_parser("extract", help="Extract structure from a file")
    extract_parser.add_argument("file", help="Path to file")
    extract_parser.add_argument("--compact", action="store_true", help="Output compact format")

    # === context ===
    context_parser = subparsers.add_parser("context", help="Get LLM context for entry point")
    context_parser.add_argument("entry", help="Entry point (function/method name)")
    context_parser.add_argument("--project", default=".", help="Project root")
    context_parser.add_argument("--depth", type=int, default=2, help="Call graph depth (default: 2)")
    context_parser.add_argument("--lang", help="Language (default: auto)")

    # === cfg ===
    cfg_parser = subparsers.add_parser("cfg", help="Get control flow graph")
    cfg_parser.add_argument("file", help="Path to file")
    cfg_parser.add_argument("function", help="Function name")

    # === dfg ===
    dfg_parser = subparsers.add_parser("dfg", help="Get data flow graph")
    dfg_parser.add_argument("file", help="Path to file")
    dfg_parser.add_argument("function", help="Function name")

    # === slice ===
    slice_parser = subparsers.add_parser("slice", help="Get program slice")
    slice_parser.add_argument("file", help="Path to file")
    slice_parser.add_argument("function", help="Function name")
    slice_parser.add_argument("line", type=int, help="Line number")
    slice_parser.add_argument("--var", help="Variable to trace")
    slice_parser.add_argument("--direction", choices=["forward", "backward"], default="backward", help="Slice direction")

    # === semantic ===
    semantic_parser = subparsers.add_parser("semantic", help="Semantic code search")
    semantic_subparsers = semantic_parser.add_subparsers(dest="semantic_cmd", help="Semantic command")

    # semantic index
    sem_index_parser = semantic_subparsers.add_parser("index", help="Build semantic index")
    sem_index_parser.add_argument("path", nargs="?", default=".", help="Project root")
    sem_index_parser.add_argument("--lang", help="Language")
    sem_index_parser.add_argument("--model", help="Embedding model: bge-large-en-v1.5 (1.3GB, default) or all-MiniLM-L6-v2 (80MB)")

    # semantic search
    sem_search_parser = semantic_subparsers.add_parser("search", help="Search the index")
    sem_search_parser.add_argument("query", help="Query string")
    sem_search_parser.add_argument("path", nargs="?", default=".", help="Project root")
    sem_search_parser.add_argument("-k", type=int, default=5, help="Number of results")
    sem_search_parser.add_argument("--expand", action="store_true", help="Expand context using call graph")

    # === calls ===
    calls_parser = subparsers.add_parser("calls", help="Build call graph")
    calls_parser.add_argument("path", nargs="?", default=".", help="Project root")
    calls_parser.add_argument("--lang", help="Language")
    calls_parser.add_argument("--json", action="store_true", help="Output JSON")

    # === doctor ===
    doctor_parser = subparsers.add_parser("doctor", help="Check and install dependencies")
    doctor_parser.add_argument("--install", metavar="LANG", help="Install missing tools for language (e.g., python, go)")
    doctor_parser.add_argument("--json", action="store_true", help="JSON output")

    # === dead ===
    dead_parser = subparsers.add_parser("dead", help="Find dead code")
    dead_parser.add_argument("path", nargs="?", default=".", help="Project root")
    dead_parser.add_argument("--entries", nargs="+", default=["main", "test_", "cli", "app"], help="Entry point patterns")

    # === impact ===
    impact_parser = subparsers.add_parser("impact", help="Analyze change impact")
    impact_parser.add_argument("path", nargs="?", default=".", help="Project root")
    impact_parser.add_argument("--files", nargs="+", help="Changed files (default: git staged)")

    # === daemon ===
    daemon_parser = subparsers.add_parser("daemon", help="Manage background daemon")
    daemon_subparsers = daemon_parser.add_subparsers(dest="action", help="Daemon action")
    
    # daemon start
    d_start = daemon_subparsers.add_parser("start", help="Start daemon")
    d_start.add_argument("project", nargs="?", default=".", help="Project root")
    
    # daemon stop
    d_stop = daemon_subparsers.add_parser("stop", help="Stop daemon")
    d_stop.add_argument("project", nargs="?", default=".", help="Project root")
    
    # daemon status
    d_status = daemon_subparsers.add_parser("status", help="Daemon status")
    d_status.add_argument("project", nargs="?", default=".", help="Project root")
    
    # daemon query (internal)
    d_query = daemon_subparsers.add_parser("query", help="Query daemon")
    d_query.add_argument("project", nargs="?", default=".", help="Project root")
    d_query.add_argument("cmd", help="Command")
    
    # daemon notify (internal)
    d_notify = daemon_subparsers.add_parser("notify", help="Notify daemon of file change")
    d_notify.add_argument("project", nargs="?", default=".", help="Project root")
    d_notify.add_argument("file", help="Changed file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "tree":
            from .tree import print_tree
            ext = set(args.ext) if args.ext else None
            try:
                print_tree(
                    args.path, 
                    extensions=ext, 
                    exclude_hidden=not args.show_hidden,
                    max_files=args.max
                )
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "structure":
            from .api import get_code_structure
            lang = args.lang or "python"
            try:
                result = get_code_structure(args.path, language=lang, max_results=args.max)
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Structure for {lang} in {args.path}:")
                    print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "search":
            from .api import search
            ext = set(args.ext) if args.ext else None
            try:
                results = search(
                    args.pattern, 
                    args.path, 
                    extensions=ext, 
                    context_lines=args.context,
                    max_results=args.max
                )
                print(json.dumps(results, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "extract":
            from .ast_extractor import extract_file
            try:
                info = extract_file(args.file)
                if args.compact:
                    print(json.dumps(info.to_compact(), indent=2))
                else:
                    print(json.dumps(info.to_dict(), indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "context":
            from .context import get_context
            try:
                print(get_context(
                    project_root=args.project,
                    entry_point=args.entry,
                    depth=args.depth,
                    language=args.lang
                ))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "cfg":
            from .cfg_extractor import extract_python_cfg
            # TODO: dispatch based on extension
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    source = f.read()
                cfg = extract_python_cfg(source, args.function)
                print(json.dumps(cfg.to_dict(), indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "dfg":
            from .dfg_extractor import extract_python_dfg
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    source = f.read()
                dfg = extract_python_dfg(source, args.function)
                print(json.dumps(dfg.to_dict(), indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "slice":
            from .pdg_extractor import extract_python_pdg
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    source = f.read()
                pdg = extract_python_pdg(source, args.function)
                
                # Simple slice
                result = pdg.slice(args.line, direction=args.direction, var_name=args.var)
                print(json.dumps({
                    "lines": sorted(list(result)),
                    "count": len(result),
                    "direction": args.direction,
                    "criteria": {"line": args.line, "var": args.var}
                }, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "semantic":
            if args.semantic_cmd == "index":
                from .semantic import build_semantic_index
                try:
                    count = build_semantic_index(args.path, lang=args.lang, model=args.model)
                    # build_semantic_index handles printing
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
            
            elif args.semantic_cmd == "search":
                from .semantic import semantic_search
                try:
                    results = semantic_search(
                        args.path, 
                        args.query, 
                        k=args.k, 
                        expand_graph=args.expand
                    )
                    print(json.dumps(results, indent=2))
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)

        elif args.command == "calls":
            from .cross_file_calls import ProjectCallGraph
            try:
                graph = ProjectCallGraph(args.path, language=args.lang)
                if args.json:
                    output = {
                        "nodes": list(graph.graph.nodes),
                        "edges": list(graph.graph.edges)
                    }
                    print(json.dumps(output, indent=2))
                else:
                    print(f"Call graph for {args.path}:")
                    print(f"Nodes: {len(graph.graph.nodes)}")
                    print(f"Edges: {len(graph.graph.edges)}")
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
                
        elif args.command == "dead":
            from .dead_code import find_dead_code
            try:
                dead_funcs = find_dead_code(args.path, args.entries)
                print(json.dumps(dead_funcs, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
                
        elif args.command == "impact":
            from .change_impact import find_impact
            try:
                impact = find_impact(args.path, args.files)
                print(json.dumps(impact, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "doctor":
            from .analysis import TOOL_INFO
            import shutil
            
            if args.install:
                import subprocess
                lang = args.install.lower()
                if lang in TOOL_INFO:
                    tools = TOOL_INFO[lang]
                    to_install = []
                    if tools["type_checker"] and not shutil.which(tools["type_checker"][0]):
                        to_install.append(tools["type_checker"])
                    if tools["linter"] and not shutil.which(tools["linter"][0]):
                        to_install.append(tools["linter"])
                    
                    if not to_install:
                        print(f"All tools for {lang} are already installed.")
                    else:
                        print(f"Installing tools for {lang}...")
                        for tool_name, cmd in to_install:
                            print(f"Running: {cmd}")
                            try:
                                subprocess.check_call(cmd, shell=True)
                                print(f"✓ Installed {tool_name}")
                            except subprocess.CalledProcessError:
                                print(f"✗ Failed to install {tool_name}")
                else:
                    print(f"No tool info for language: {lang}")
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
                    result = query_daemon(project_path, {"cmd": args.cmd})
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
