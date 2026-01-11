"""
TLDR MCP Server - Model Context Protocol interface for TLDR.

Provides 1:1 mapping with TLDR daemon commands, enabling AI tools
(OpenCode, Claude Desktop, Claude Code) to use TLDR's code analysis.

Usage:
    tldr-mcp --project /path/to/project
"""

import hashlib
import json
import socket
import subprocess
import sys
import time
import os
import platform
import tempfile
from pathlib import Path

# Conditional imports for file locking
if os.name == "nt":
    import msvcrt
else:
    import fcntl

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tldr-code")


def _get_socket_path(project: str) -> Path:
    """Compute socket path matching daemon.py logic."""
    hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / f"tldr-{hash_val}.sock"


def _get_lock_path(project: str) -> Path:
    """Get lock file path for daemon startup synchronization."""
    hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / f"tldr-{hash_val}.lock"


def _ping_daemon(project: str) -> bool:
    """Check if daemon is alive and responding."""
    socket_path = _get_socket_path(project)
    if not socket_path.exists():
        return False
    try:
        result = _send_raw(project, {"cmd": "ping"})
        return result.get("status") == "ok"
    except Exception:
        return False


def _ensure_daemon(project: str, timeout: float = 10.0) -> None:
    """Ensure daemon is running, starting it if needed.

    Uses file locking to prevent race conditions when multiple agents
    try to start the daemon simultaneously.
    """
    # Fast path: daemon already running (no lock needed)
    if _ping_daemon(project):
        return

    socket_path = _get_socket_path(project)
    lock_path = _get_lock_path(project)

    # Acquire exclusive lock for startup coordination
    lock_path.touch(exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            if os.name == "nt":
                # Windows locking logic
                # Lock first 10 bytes (arbitrary small region)
                # LK_RLCK blocks for 10s if locked, raising OSError if it fails
                # We loop to simulate cleaner blocking behavior if needed, 
                # but standard msvcrt.locking with LK_RLCK is the closest to LOCK_EX
                # However, Python's msvcrt.locking doesn't wait indefinitely nicely.
                # A simple busy wait loop with LK_NBLCK is often more robust.
                while True:
                    try:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 10)
                        break
                    except OSError:
                        time.sleep(0.1)
            else:
                # Unix locking
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            # Re-check after acquiring lock (another process may have started daemon)
            if _ping_daemon(project):
                return

            # Clean up stale socket if daemon is dead
            if socket_path.exists():
                socket_path.unlink(missing_ok=True)

            # Start daemon
            subprocess.Popen(
                [sys.executable, "-m", "tldr.cli", "daemon", "start", "--project", project],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for daemon to be ready
            start = time.time()
            while time.time() - start < timeout:
                if _ping_daemon(project):
                    return
                time.sleep(0.1)

            raise RuntimeError(f"Failed to start TLDR daemon for {project}")
        finally:
        finally:
            if os.name == "nt":
                 try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 10)
                 except OSError:
                    pass
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _send_raw(project: str, command: dict) -> dict:
    """Send command to daemon socket."""
    socket_path = _get_socket_path(project)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(socket_path))
        sock.sendall(json.dumps(command).encode() + b"\n")

        # Read response
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            # Check if we have complete JSON
            try:
                return json.loads(b"".join(chunks))
            except json.JSONDecodeError:
                continue

        return json.loads(b"".join(chunks))
    finally:
        sock.close()


def _send_command(project: str, command: dict) -> dict:
    """Send command to daemon, auto-starting if needed."""
    _ensure_daemon(project)
    return _send_raw(project, command)


# === NAVIGATION TOOLS ===


@mcp.tool()
def tree(project: str = ".", extensions: list[str] | None = None) -> dict:
    """Get file tree structure for a project.

    Args:
        project: Project root directory
        extensions: Optional list of extensions to filter (e.g., [".py", ".ts"])
    """
    return _send_command(
        project,
        {
            "cmd": "tree",
            "extensions": tuple(extensions) if extensions else None,
            "exclude_hidden": True,
        },
    )


@mcp.tool()
def structure(
    project: str = ".", language: str = "python", max_results: int = 100
) -> dict:
    """Get code structure (codemaps) - functions, classes, imports per file.

    Args:
        project: Project root directory
        language: Programming language (python, typescript, go, rust, etc.)
        max_results: Maximum files to analyze
    """
    return _send_command(
        project,
        {"cmd": "structure", "language": language, "max_results": max_results},
    )


@mcp.tool()
def search(project: str, pattern: str, max_results: int = 100) -> dict:
    """Search files for a regex pattern.

    Args:
        project: Project root directory
        pattern: Regex pattern to search for
        max_results: Maximum matches to return
    """
    return _send_command(
        project, {"cmd": "search", "pattern": pattern, "max_results": max_results}
    )


@mcp.tool()
def extract(file: str) -> dict:
    """Extract full code structure from a file.

    Returns imports, functions, classes, and intra-file call graph.

    Args:
        file: Path to source file
    """
    project = str(Path(file).parent)
    return _send_command(project, {"cmd": "extract", "file": file})


# === CONTEXT TOOLS (Key differentiator - 95% token savings) ===


@mcp.tool()
def context(
    project: str, entry: str, depth: int = 2, language: str = "python"
) -> str:
    """Get token-efficient LLM context starting from an entry point.

    Follows call graph to specified depth, returning signatures and complexity
    metrics. This is TLDR's key value - 95% token savings vs reading raw files.

    Args:
        project: Project root directory
        entry: Entry point (function_name or Class.method)
        depth: How deep to follow calls (default 2)
        language: Programming language

    Returns:
        LLM-ready formatted context string
    """
    result = _send_command(
        project,
        {"cmd": "context", "entry": entry, "depth": depth, "language": language},
    )
    # Return formatted string for LLM consumption
    if result.get("status") == "ok":
        ctx = result.get("result", {})
        if hasattr(ctx, "to_llm_string"):
            return ctx.to_llm_string()
        return str(ctx)
    return str(result)


# === FLOW ANALYSIS TOOLS ===


@mcp.tool()
def cfg(file: str, function: str, language: str = "python") -> dict:
    """Get control flow graph for a function.

    Returns basic blocks, control flow edges, and cyclomatic complexity.

    Args:
        file: Path to source file
        function: Function name to analyze
        language: Programming language
    """
    project = str(Path(file).parent)
    return _send_command(
        project,
        {"cmd": "cfg", "file": file, "function": function, "language": language},
    )


@mcp.tool()
def dfg(file: str, function: str, language: str = "python") -> dict:
    """Get data flow graph for a function.

    Returns variable references and def-use chains.

    Args:
        file: Path to source file
        function: Function name to analyze
        language: Programming language
    """
    project = str(Path(file).parent)
    return _send_command(
        project,
        {"cmd": "dfg", "file": file, "function": function, "language": language},
    )


@mcp.tool()
def slice(
    file: str,
    function: str,
    line: int,
    direction: str = "backward",
    variable: str | None = None,
    language: str = "python",
) -> dict:
    """Get program slice - lines affecting or affected by a given line.

    Args:
        file: Path to source file
        function: Function name
        line: Line number to slice from
        direction: "backward" (what affects this line) or "forward" (what this line affects)
        variable: Optional specific variable to trace
        language: Programming language

    Returns:
        Dict with lines in the slice and count
    """
    project = str(Path(file).parent)
    return _send_command(
        project,
        {
            "cmd": "slice",
            "file": file,
            "function": function,
            "line": line,
            "direction": direction,
            "variable": variable or "",
            "language": language,
        },
    )


# === CODEBASE ANALYSIS TOOLS ===


@mcp.tool()
def impact(project: str, function: str) -> dict:
    """Find all callers of a function (reverse call graph).

    Useful before refactoring to understand what would break.

    Args:
        project: Project root directory
        function: Function name to find callers of
    """
    return _send_command(project, {"cmd": "impact", "func": function})


@mcp.tool()
def dead(
    project: str,
    entry_points: list[str] | None = None,
    language: str = "python",
) -> dict:
    """Find unreachable (dead) code not called from entry points.

    Args:
        project: Project root directory
        entry_points: List of entry point patterns (default: main, test_, cli)
        language: Programming language
    """
    return _send_command(
        project,
        {"cmd": "dead", "entry_points": entry_points, "language": language},
    )


@mcp.tool()
def arch(project: str, language: str = "python") -> dict:
    """Detect architectural layers from call patterns.

    Identifies entry layer (controllers), middle layer (services),
    and leaf layer (utilities). Also detects circular dependencies.

    Args:
        project: Project root directory
        language: Programming language
    """
    return _send_command(project, {"cmd": "arch", "language": language})


@mcp.tool()
def calls(project: str, language: str = "python") -> dict:
    """Build cross-file call graph for the project.

    Args:
        project: Project root directory
        language: Programming language
    """
    return _send_command(project, {"cmd": "calls", "language": language})


# === IMPORT ANALYSIS ===


@mcp.tool()
def imports(file: str, language: str = "python") -> dict:
    """Parse imports from a source file.

    Args:
        file: Path to source file
        language: Programming language
    """
    project = str(Path(file).parent)
    return _send_command(
        project, {"cmd": "imports", "file": file, "language": language}
    )


@mcp.tool()
def importers(project: str, module: str, language: str = "python") -> dict:
    """Find all files that import a given module.

    Args:
        project: Project root directory
        module: Module name to search for
        language: Programming language
    """
    return _send_command(
        project, {"cmd": "importers", "module": module, "language": language}
    )


# === SEMANTIC SEARCH ===


@mcp.tool()
def semantic(project: str, query: str, k: int = 10) -> dict:
    """Semantic code search using embeddings.

    Searches over function/class summaries using vector similarity.
    Auto-downloads embedding model and builds index on first use.

    Args:
        project: Project root directory
        query: Natural language query
        k: Number of results to return
    """
    return _send_command(
        project, {"cmd": "semantic", "action": "search", "query": query, "k": k}
    )


# === QUALITY TOOLS ===


@mcp.tool()
def diagnostics(path: str, language: str = "python") -> dict:
    """Get type and lint diagnostics.

    For Python: runs pyright (types) + ruff (lint).

    Args:
        path: File or directory path
        language: Programming language
    """
    project = str(Path(path).parent) if Path(path).is_file() else path
    return _send_command(
        project, {"cmd": "diagnostics", "file": path, "language": language}
    )


@mcp.tool()
def change_impact(project: str, files: list[str] | None = None) -> dict:
    """Find tests affected by changed files.

    Uses call graph + import analysis to identify which tests to run.

    Args:
        project: Project root directory
        files: List of changed files (auto-detects from git if None)
    """
    return _send_command(project, {"cmd": "change_impact", "files": files})


# === DAEMON MANAGEMENT ===


@mcp.tool()
def status(project: str = ".") -> dict:
    """Get daemon status including uptime and cache statistics.

    Args:
        project: Project root directory
    """
    return _send_command(project, {"cmd": "status"})


def main():
    """Entry point for tldr-mcp command."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="TLDR MCP Server")
    parser.add_argument("--project", default=".", help="Project root directory")
    args = parser.parse_args()

    # Set default project for tools that need it
    os.environ["TLDR_PROJECT"] = str(Path(args.project).resolve())

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
