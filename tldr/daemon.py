"""
Socket-based daemon that holds indexes in memory.

Features:
- Loads call_graph.json, semantic embeddings on start
- Builds symbol_index from call_graph for O(1) lookups
- Handles: search, impact, extract, ping, status commands
- Auto-shutdown after 30min idle
- Persists indexes to .tldr/ for fast restart

P5 Features (Incremental Performance):
- ContentHashedIndex: Skip unchanged files via content hashing
- SalsaDB: Memoize query results with automatic invalidation
- File change notifications trigger cache invalidation
"""

import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from tldr.dedup import ContentHashedIndex
from tldr.salsa import SalsaDB, salsa_query


# Fast hashing: blake3 (5-10x faster than md5) with hashlib fallback
try:
    import blake3

    def _hash_string(s: str, length: int = 8) -> str:
        """Hash a string using blake3 for speed.

        Args:
            s: String to hash
            length: Number of hex characters to return (default 8)

        Returns:
            Truncated hexadecimal hash string
        """
        return blake3.blake3(s.encode()).hexdigest()[:length]

except ImportError:
    import hashlib

    def _hash_string(s: str, length: int = 8) -> str:
        """Hash a string using md5 (fallback when blake3 unavailable).

        Args:
            s: String to hash
            length: Number of hex characters to return (default 8)

        Returns:
            Truncated hexadecimal hash string
        """
        return hashlib.md5(s.encode()).hexdigest()[:length]


# Fast JSON serialization for IPC hot paths
# orjson is 3-10x faster than stdlib json for encoding/decoding
try:
    import orjson

    def _json_dumps(obj: Any) -> bytes:
        """Serialize object to JSON bytes using orjson (fast path)."""
        return orjson.dumps(obj)

    def _json_loads(data: bytes | str) -> Any:
        """Deserialize JSON bytes/str to object using orjson (fast path)."""
        return orjson.loads(data)

except ImportError:
    # Fallback to stdlib json if orjson not installed
    def _json_dumps(obj: Any) -> bytes:
        """Serialize object to JSON bytes using stdlib json (slow path)."""
        return json.dumps(obj).encode("utf-8")

    def _json_loads(data: bytes | str) -> Any:
        """Deserialize JSON bytes/str to object using stdlib json (slow path)."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)


# Idle timeout: default 30 minutes, configurable via TLDR_IDLE_TIMEOUT env var
# Set to 0 to disable idle timeout (daemon runs forever)
IDLE_TIMEOUT = int(os.environ.get("TLDR_IDLE_TIMEOUT", 30 * 60))

# Maximum request size: 10MB - prevents OOM from malicious clients sending
# infinite data without newline terminator
MAX_REQUEST_SIZE = 10 * 1024 * 1024

# Client socket timeout: 30 seconds - prevents indefinite blocking when
# daemon hangs or becomes unresponsive
CLIENT_SOCKET_TIMEOUT = 30.0

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Salsa Query Functions (P5)
# -------------------------------------------------------------------------


@salsa_query
def cached_search(db: SalsaDB, project: str, pattern: str, max_results: int) -> dict:
    """Cached search query - memoized by SalsaDB."""
    from tldr import api

    results = api.search(pattern=pattern, root=Path(project), max_results=max_results)
    return {"status": "ok", "results": results}


@salsa_query
def cached_extract(db: SalsaDB, file_path: str) -> dict:
    """Cached file extraction - memoized by SalsaDB."""
    from tldr import api

    result = api.extract_file(file_path)
    return {"status": "ok", "result": result}


@salsa_query
def cached_dead_code(
    db: SalsaDB, project: str, entry_points: tuple, language: str
) -> dict:
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
def cached_slice(
    db: SalsaDB, file_path: str, function: str, line: int, direction: str, variable: str
) -> dict:
    """Cached program slice - memoized by SalsaDB."""
    from tldr.api import get_slice

    var = variable if variable else None
    lines = get_slice(file_path, function, line, direction=direction, variable=var)
    return {"status": "ok", "lines": sorted(lines), "count": len(lines)}


@salsa_query
def cached_tree(
    db: SalsaDB, project: str, extensions: tuple, exclude_hidden: bool
) -> dict:
    """Cached file tree - memoized by SalsaDB."""
    from tldr.api import get_file_tree

    ext_set = set(extensions) if extensions else None
    result = get_file_tree(project, extensions=ext_set, exclude_hidden=exclude_hidden)
    return {"status": "ok", "result": result}


@salsa_query
def cached_structure(
    db: SalsaDB, project: str, language: str, max_results: int
) -> dict:
    """Cached code structure - memoized by SalsaDB."""
    from tldr.api import get_code_structure

    result = get_code_structure(project, language=language, max_results=max_results)
    return {"status": "ok", "result": result}


@salsa_query
def cached_context(
    db: SalsaDB, project: str, entry: str, language: str, depth: int
) -> dict:
    """Cached relevant context - memoized by SalsaDB."""
    from dataclasses import asdict
    from tldr.api import get_relevant_context

    result = get_relevant_context(
        project=project, entry_point=entry, language=language, depth=depth
    )
    # Convert dataclass to dict for JSON serialization
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
    from pathlib import Path

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
                    importers.append(
                        {
                            "file": str(Path(file_path).relative_to(project_path)),
                            "import": imp,
                        }
                    )
        except Exception:
            pass

    return {"status": "ok", "module": module, "importers": importers}


class TLDRDaemon:
    """
    TLDR daemon server holding indexes in memory.

    Listens on a Unix socket for commands and responds with JSON.
    Automatically shuts down after IDLE_TIMEOUT seconds of inactivity.
    """

    def __init__(self, project_path: Path):
        """
        Initialize the daemon for a project.

        Args:
            project_path: Root path of the project to index
        """
        self.project = project_path
        self.tldr_dir = project_path / ".tldr"
        self.socket_path = self._compute_socket_path()
        self.last_query = time.time()
        self.indexes: dict[str, Any] = {}

        # Internal state
        self._status = "initializing"
        self._start_time = time.time()
        self._shutdown_requested = False
        self._socket: Optional[socket.socket] = None

        # P5 Features: Content-hash deduplication and query memoization
        self.dedup_index: Optional[ContentHashedIndex] = None
        self.salsa_db: SalsaDB = SalsaDB()

        # P6 Features: Dirty-count triggered semantic re-indexing
        # Lock protects _dirty_count, _dirty_files, _reindex_in_progress
        # These are accessed from main thread (notify handler) and background reindex thread
        self._dirty_lock = threading.Lock()
        self._dirty_count: int = 0
        self._dirty_files: set[str] = set()
        self._reindex_in_progress: bool = False
        self._semantic_config = self._load_semantic_config()

    def _compute_socket_path(self) -> Path:
        """Compute deterministic socket path from project path."""
        hash_val = _hash_string(str(self.project))
        return Path(f"/tmp/tldr-{hash_val}.sock")

    def _load_semantic_config(self) -> dict:
        """Load semantic search configuration.

        Checks for config in:
        1. .claude/settings.json (Claude Code settings)
        2. .tldr/config.json (TLDR-specific settings)

        Returns default config if no file found.
        """
        default_config = {
            "enabled": True,
            "auto_reindex_threshold": 20,  # Files changed before auto re-index
            "model": "bge-large-en-v1.5",
        }

        # Try Claude settings first
        claude_settings = self.project / ".claude" / "settings.json"
        if claude_settings.exists():
            try:
                settings = json.loads(claude_settings.read_text())
                if "semantic_search" in settings:
                    return {**default_config, **settings["semantic_search"]}
            except Exception as e:
                logger.warning(f"Failed to load Claude settings: {e}")

        # Try TLDR config
        tldr_config = self.tldr_dir / "config.json"
        if tldr_config.exists():
            try:
                config = json.loads(tldr_config.read_text())
                if "semantic" in config:
                    return {**default_config, **config["semantic"]}
            except Exception as e:
                logger.warning(f"Failed to load TLDR config: {e}")

        return default_config

    def _get_connection_info(self) -> tuple[str, int | None]:
        """Return (address, port) - port is None for Unix sockets.

        On Windows, uses TCP on localhost with a deterministic port.
        On Unix (Linux/macOS), uses Unix domain sockets.
        """
        if sys.platform == "win32":
            # TCP on localhost with deterministic port from hash
            hash_val = _hash_string(str(self.project))
            port = 49152 + (int(hash_val, 16) % 10000)
            return ("127.0.0.1", port)
        else:
            # Unix socket path
            return (str(self.socket_path), None)

    def is_idle(self) -> bool:
        """Check if daemon has been idle longer than IDLE_TIMEOUT.

        Returns False if IDLE_TIMEOUT is 0 (timeout disabled).
        """
        if IDLE_TIMEOUT <= 0:
            return False  # Timeout disabled
        return (time.time() - self.last_query) > IDLE_TIMEOUT

    def handle_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """
        Route and handle a command.

        Args:
            command: Dict with 'cmd' key and optional parameters

        Returns:
            Response dict with 'status' and command-specific fields
        """
        # Update last query time for any command
        self.last_query = time.time()

        cmd = command.get("cmd", "")

        handlers = {
            "ping": self._handle_ping,
            "status": self._handle_status,
            "shutdown": self._handle_shutdown,
            "search": self._handle_search,
            "extract": self._handle_extract,
            "impact": self._handle_impact,
            "dead": self._handle_dead,
            "arch": self._handle_arch,
            "cfg": self._handle_cfg,
            "dfg": self._handle_dfg,
            "slice": self._handle_slice,
            "calls": self._handle_calls,
            "warm": self._handle_warm,
            "semantic": self._handle_semantic,
            "tree": self._handle_tree,
            "structure": self._handle_structure,
            "context": self._handle_context,
            "imports": self._handle_imports,
            "importers": self._handle_importers,
            "notify": self._handle_notify,
            "diagnostics": self._handle_diagnostics,
            "change_impact": self._handle_change_impact,
        }

        handler = handlers.get(cmd)
        if handler:
            return handler(command)
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}

    def _handle_ping(self, command: dict) -> dict:
        """Handle ping command."""
        return {"status": "ok"}

    def _handle_status(self, command: dict) -> dict:
        """Handle status command with P5 cache statistics."""
        uptime = time.time() - self._start_time

        # Get SalsaDB stats
        salsa_stats = self.salsa_db.get_stats()

        # Get dedup stats if loaded
        dedup_stats = {}
        if self.dedup_index:
            dedup_stats = self.dedup_index.stats()

        return {
            "status": self._status,
            "uptime": uptime,
            "files": len(self.indexes.get("files", [])),
            "project": str(self.project),
            "salsa_stats": salsa_stats,
            "dedup_stats": dedup_stats,
        }

    def _handle_shutdown(self, command: dict) -> dict:
        """Handle shutdown command."""
        self._shutdown_requested = True
        return {"status": "shutting_down"}

    def _handle_search(self, command: dict) -> dict:
        """Handle search command with SalsaDB caching."""
        pattern = command.get("pattern")
        if not pattern:
            return {"status": "error", "message": "Missing required parameter: pattern"}

        try:
            max_results = command.get("max_results", 100)
            # Use SalsaDB for cached search
            return self.salsa_db.query(
                cached_search,
                self.salsa_db,
                str(self.project),
                pattern,
                max_results,
            )
        except Exception as e:
            logger.exception("Search failed")
            return {"status": "error", "message": str(e)}

    def _handle_extract(self, command: dict) -> dict:
        """Handle extract command with SalsaDB caching."""
        file_path = command.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing required parameter: file"}

        try:
            # Use SalsaDB for cached extraction
            return self.salsa_db.query(cached_extract, self.salsa_db, file_path)
        except Exception as e:
            logger.exception("Extract failed")
            return {"status": "error", "message": str(e)}

    def _handle_impact(self, command: dict) -> dict:
        """Handle impact command - find callers of a function.

        Uses pre-built reverse index for O(1) lookup instead of O(n) edge scan.
        For 100K edges, this is ~5000x faster.
        """
        func_name = command.get("func")
        if not func_name:
            return {"status": "error", "message": "Missing required parameter: func"}

        try:
            self._ensure_call_graph_loaded()

            # O(1) lookup using reverse index instead of O(n) edge scan
            reverse_index = self.indexes.get("reverse_call_graph", {})
            callers = reverse_index.get(func_name, [])

            return {"status": "ok", "callers": callers}
        except Exception as e:
            logger.exception("Impact analysis failed")
            return {"status": "error", "message": str(e)}

    def _build_reverse_index(self, call_graph: dict) -> dict[str, list[dict]]:
        """Build reverse index from call graph for O(1) caller lookups.

        The call graph edges may use different key formats:
        - Legacy format: {"caller", "callee", "file", "line"}
        - New format: {"from_func", "to_func", "from_file", "to_file"}

        Args:
            call_graph: Call graph dict with "edges" list

        Returns:
            Reverse index mapping callee -> list of {caller, file, line}
        """
        reverse_index: dict[str, list[dict]] = {}
        for edge in call_graph.get("edges", []):
            # Handle both legacy and new edge formats
            caller = edge.get("caller") or edge.get("from_func")
            callee = edge.get("callee") or edge.get("to_func")
            caller_file = edge.get("file") or edge.get("from_file")
            line = edge.get("line")

            if callee:
                if callee not in reverse_index:
                    reverse_index[callee] = []
                reverse_index[callee].append(
                    {
                        "caller": caller,
                        "file": caller_file,
                        "line": line,
                    }
                )
        return reverse_index

    def _ensure_call_graph_loaded(self):
        """Load call graph and build reverse index for O(1) caller lookups."""
        if "call_graph" in self.indexes:
            return

        call_graph_path = self.tldr_dir / "call_graph.json"
        if call_graph_path.exists():
            try:
                call_graph = json.loads(call_graph_path.read_text())
                self.indexes["call_graph"] = call_graph
                self.indexes["reverse_call_graph"] = self._build_reverse_index(
                    call_graph
                )
                logger.info(
                    f"Loaded call graph from {call_graph_path}: "
                    f"{len(call_graph.get('edges', []))} edges, "
                    f"{len(self.indexes['reverse_call_graph'])} unique callees indexed"
                )
            except Exception as e:
                logger.error(f"Failed to load call graph: {e}")
                self.indexes["call_graph"] = {"edges": [], "nodes": {}}
                self.indexes["reverse_call_graph"] = {}
        else:
            logger.warning(f"No call graph found at {call_graph_path}")
            self.indexes["call_graph"] = {"edges": [], "nodes": {}}
            self.indexes["reverse_call_graph"] = {}

    # -------------------------------------------------------------------------
    # New Command Handlers: dead, arch, cfg, dfg, slice, calls, warm, semantic
    # -------------------------------------------------------------------------

    def _handle_dead(self, command: dict) -> dict:
        """Handle dead code analysis command."""
        try:
            language = command.get("language", "python")
            entry_points = command.get("entry_points")
            # Convert to tuple for hashability (SalsaDB cache key)
            entry_tuple = tuple(entry_points) if entry_points else ()
            return self.salsa_db.query(
                cached_dead_code,
                self.salsa_db,
                str(self.project),
                entry_tuple,
                language,
            )
        except Exception as e:
            logger.exception("Dead code analysis failed")
            return {"status": "error", "message": str(e)}

    def _handle_arch(self, command: dict) -> dict:
        """Handle architecture analysis command."""
        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_architecture,
                self.salsa_db,
                str(self.project),
                language,
            )
        except Exception as e:
            logger.exception("Architecture analysis failed")
            return {"status": "error", "message": str(e)}

    def _handle_cfg(self, command: dict) -> dict:
        """Handle CFG extraction command."""
        file_path = command.get("file")
        function = command.get("function")
        if not file_path or not function:
            return {
                "status": "error",
                "message": "Missing required parameters: file, function",
            }

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_cfg,
                self.salsa_db,
                file_path,
                function,
                language,
            )
        except Exception as e:
            logger.exception("CFG extraction failed")
            return {"status": "error", "message": str(e)}

    def _handle_dfg(self, command: dict) -> dict:
        """Handle DFG extraction command."""
        file_path = command.get("file")
        function = command.get("function")
        if not file_path or not function:
            return {
                "status": "error",
                "message": "Missing required parameters: file, function",
            }

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_dfg,
                self.salsa_db,
                file_path,
                function,
                language,
            )
        except Exception as e:
            logger.exception("DFG extraction failed")
            return {"status": "error", "message": str(e)}

    def _handle_slice(self, command: dict) -> dict:
        """Handle program slice command."""
        file_path = command.get("file")
        function = command.get("function")
        line = command.get("line")
        if not file_path or not function or line is None:
            return {
                "status": "error",
                "message": "Missing required parameters: file, function, line",
            }

        try:
            direction = command.get("direction", "backward")
            variable = command.get("variable", "")
            return self.salsa_db.query(
                cached_slice,
                self.salsa_db,
                file_path,
                function,
                int(line),
                direction,
                variable,
            )
        except Exception as e:
            logger.exception("Program slice failed")
            return {"status": "error", "message": str(e)}

    def _handle_calls(self, command: dict) -> dict:
        """Handle call graph building command."""
        try:
            language = command.get("language", "python")
            from tldr.cross_file_calls import build_project_call_graph

            graph = build_project_call_graph(self.project, language=language)
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
            return {"status": "ok", "result": result}
        except Exception as e:
            logger.exception("Call graph building failed")
            return {"status": "error", "message": str(e)}

    def _handle_warm(self, command: dict) -> dict:
        """Handle cache warming command (builds call graph cache)."""
        try:
            language = command.get("language", "python")
            from tldr.cross_file_calls import scan_project, build_project_call_graph

            files = scan_project(self.project, language=language)
            graph = build_project_call_graph(self.project, language=language)

            # Create cache directory and save
            cache_dir = self.tldr_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "call_graph.json"
            cache_data = {
                "edges": [
                    {
                        "from_file": e[0],
                        "from_func": e[1],
                        "to_file": e[2],
                        "to_func": e[3],
                    }
                    for e in graph.edges
                ],
                "timestamp": time.time(),
            }
            cache_file.write_text(json.dumps(cache_data, indent=2))

            # Update in-memory index and rebuild reverse index for O(1) lookups
            self.indexes["call_graph"] = cache_data
            self.indexes["reverse_call_graph"] = self._build_reverse_index(cache_data)

            return {"status": "ok", "files": len(files), "edges": len(graph.edges)}
        except Exception as e:
            logger.exception("Cache warming failed")
            return {"status": "error", "message": str(e)}

    def _handle_semantic(self, command: dict) -> dict:
        """Handle semantic search/index command."""
        action = command.get("action", "search")

        try:
            from tldr.semantic import build_semantic_index, semantic_search

            if action == "index":
                language = command.get("language", "python")
                count = build_semantic_index(str(self.project), lang=language)
                return {"status": "ok", "indexed": count}

            elif action == "search":
                query = command.get("query")
                if not query:
                    return {
                        "status": "error",
                        "message": "Missing required parameter: query",
                    }
                k = command.get("k", 10)
                results = semantic_search(str(self.project), query, k=k)
                return {"status": "ok", "results": results}

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception("Semantic operation failed")
            return {"status": "error", "message": str(e)}

    # -------------------------------------------------------------------------
    # New handlers: tree, structure, context, imports, importers
    # -------------------------------------------------------------------------

    def _handle_tree(self, command: dict) -> dict:
        """Handle file tree command."""
        try:
            extensions = command.get("extensions")
            ext_tuple = tuple(extensions) if extensions else ()
            exclude_hidden = command.get("exclude_hidden", True)
            return self.salsa_db.query(
                cached_tree,
                self.salsa_db,
                str(self.project),
                ext_tuple,
                exclude_hidden,
            )
        except Exception as e:
            logger.exception("File tree failed")
            return {"status": "error", "message": str(e)}

    def _handle_structure(self, command: dict) -> dict:
        """Handle code structure command."""
        try:
            language = command.get("language", "python")
            max_results = command.get("max_results", 100)
            return self.salsa_db.query(
                cached_structure,
                self.salsa_db,
                str(self.project),
                language,
                max_results,
            )
        except Exception as e:
            logger.exception("Code structure failed")
            return {"status": "error", "message": str(e)}

    def _handle_context(self, command: dict) -> dict:
        """Handle relevant context command."""
        entry = command.get("entry")
        if not entry:
            return {"status": "error", "message": "Missing required parameter: entry"}

        try:
            language = command.get("language", "python")
            depth = command.get("depth", 2)
            return self.salsa_db.query(
                cached_context,
                self.salsa_db,
                str(self.project),
                entry,
                language,
                depth,
            )
        except Exception as e:
            logger.exception("Relevant context failed")
            return {"status": "error", "message": str(e)}

    def _handle_imports(self, command: dict) -> dict:
        """Handle imports extraction command."""
        file_path = command.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing required parameter: file"}

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_imports,
                self.salsa_db,
                file_path,
                language,
            )
        except Exception as e:
            logger.exception("Imports extraction failed")
            return {"status": "error", "message": str(e)}

    def _handle_importers(self, command: dict) -> dict:
        """Handle reverse import lookup command."""
        module = command.get("module")
        if not module:
            return {"status": "error", "message": "Missing required parameter: module"}

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_importers,
                self.salsa_db,
                str(self.project),
                module,
                language,
            )
        except Exception as e:
            logger.exception("Importers lookup failed")
            return {"status": "error", "message": str(e)}

    # -------------------------------------------------------------------------
    # P5 Features: Content-Hash Deduplication
    # -------------------------------------------------------------------------

    def _ensure_dedup_index_loaded(self):
        """Load or create ContentHashedIndex for file deduplication."""
        if self.dedup_index is not None:
            return

        self.dedup_index = ContentHashedIndex(str(self.project))

        # Try to load persisted index
        if self.dedup_index.load():
            logger.info("Loaded content-hash index from disk")
        else:
            logger.info("Created new content-hash index")

        # Index all Python files in project
        for py_file in self.project.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            try:
                self.dedup_index.get_or_create_edges(str(py_file), lang="python")
            except Exception as e:
                logger.debug(f"Could not index {py_file}: {e}")

    def _save_dedup_index(self):
        """Persist ContentHashedIndex to disk."""
        if self.dedup_index:
            try:
                self.dedup_index.save()
                logger.info("Saved content-hash index to disk")
            except Exception as e:
                logger.error(f"Failed to save dedup index: {e}")

    # -------------------------------------------------------------------------
    # P5 Features: File Change Notifications
    # -------------------------------------------------------------------------

    def _handle_notify(self, command: dict) -> dict:
        """Handle file change notification from hooks.

        Tracks dirty files and triggers background semantic re-indexing
        when threshold is reached.

        Args:
            command: Dict with 'file' (path to changed file)

        Returns:
            Response with dirty count and reindex status
        """
        file_path = command.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing required parameter: file"}

        # Check if semantic search is enabled
        if not self._semantic_config.get("enabled", True):
            # Still notify for Salsa cache invalidation
            self.notify_file_changed(file_path)
            return {"status": "ok", "semantic_enabled": False}

        # Track dirty file and check reindex threshold under lock
        # Lock protects _dirty_files, _dirty_count, _reindex_in_progress from
        # concurrent access by main thread and background reindex thread
        threshold = self._semantic_config.get("auto_reindex_threshold", 20)
        with self._dirty_lock:
            if file_path not in self._dirty_files:
                self._dirty_files.add(file_path)
                self._dirty_count += 1
                logger.info(
                    f"Dirty file tracked: {file_path} (count: {self._dirty_count})"
                )

            # Check if we should trigger background re-indexing
            should_reindex = (
                self._dirty_count >= threshold and not self._reindex_in_progress
            )
            # Capture current count for response while holding lock
            current_dirty_count = self._dirty_count

        # Notify Salsa for cache invalidation (outside lock - no shared state)
        self.notify_file_changed(file_path)

        if should_reindex:
            self._trigger_background_reindex()

        return {
            "status": "ok",
            "dirty_count": current_dirty_count,
            "threshold": threshold,
            "reindex_triggered": should_reindex,
        }

    def _trigger_background_reindex(self):
        """Trigger background semantic re-indexing.

        Spawns a subprocess to rebuild the semantic index,
        allowing the daemon to continue serving requests.

        Thread-safe: Uses _dirty_lock to prevent TOCTOU race on _reindex_in_progress.
        """
        import subprocess

        # Atomic check-and-set under lock prevents TOCTOU race where two threads
        # both pass the check before either sets the flag
        with self._dirty_lock:
            if self._reindex_in_progress:
                logger.info("Re-index already in progress, skipping")
                return
            self._reindex_in_progress = True
            dirty_files = list(self._dirty_files)

        logger.info(
            f"Triggering background semantic re-index for {len(dirty_files)} files"
        )

        def do_reindex():
            try:
                # Run semantic index command
                cmd = [
                    sys.executable,
                    "-m",
                    "tldr.cli",
                    "semantic",
                    "index",
                    str(self.project),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 min max
                )

                if result.returncode == 0:
                    logger.info("Background semantic re-index completed successfully")
                else:
                    logger.error(
                        f"Background semantic re-index failed: {result.stderr}"
                    )

            except Exception as e:
                logger.exception(f"Background semantic re-index error: {e}")
            finally:
                # Reset dirty tracking under lock
                with self._dirty_lock:
                    self._dirty_files.clear()
                    self._dirty_count = 0
                    self._reindex_in_progress = False

        # Run in thread to not block daemon
        thread = threading.Thread(target=do_reindex, daemon=True)
        thread.start()

    def _handle_diagnostics(self, command: dict) -> dict:
        """Handle diagnostics command - type check + lint.

        Runs pyright for type checking and ruff for linting.
        Returns structured errors for pre-test validation.

        Args:
            command: Dict with optional:
                - file: Single file to check
                - project: If True, check whole project
                - no_lint: If True, skip ruff (type check only)

        Returns:
            Response with errors list and summary
        """
        import subprocess

        file_path = command.get("file")
        check_project = command.get("project", False)
        no_lint = command.get("no_lint", False)

        target = str(self.project) if check_project else file_path
        if not target:
            return {
                "status": "error",
                "message": "Missing required parameter: file or project",
            }

        errors = []

        # Run pyright for type checking
        try:
            pyright_cmd = ["pyright", "--outputjson", target]
            result = subprocess.run(
                pyright_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project),
            )
            if result.stdout:
                try:
                    import json

                    pyright_output = json.loads(result.stdout)
                    for diag in pyright_output.get("generalDiagnostics", []):
                        errors.append(
                            {
                                "type": "type",
                                "severity": diag.get("severity", "error"),
                                "file": diag.get("file", ""),
                                "line": diag.get("range", {})
                                .get("start", {})
                                .get("line", 0),
                                "message": diag.get("message", ""),
                                "rule": diag.get("rule", "pyright"),
                            }
                        )
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            logger.debug("pyright not found, skipping type check")
        except subprocess.TimeoutExpired:
            logger.warning("pyright timed out")
        except Exception as e:
            logger.debug(f"pyright error: {e}")

        # Run ruff for linting (unless disabled)
        if not no_lint:
            try:
                ruff_cmd = ["ruff", "check", "--output-format=json", target]
                result = subprocess.run(
                    ruff_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(self.project),
                )
                if result.stdout:
                    try:
                        import json

                        ruff_output = json.loads(result.stdout)
                        for diag in ruff_output:
                            errors.append(
                                {
                                    "type": "lint",
                                    "severity": "warning"
                                    if diag.get("fix")
                                    else "error",
                                    "file": diag.get("filename", ""),
                                    "line": diag.get("location", {}).get("row", 0),
                                    "message": diag.get("message", ""),
                                    "rule": diag.get("code", "ruff"),
                                }
                            )
                    except json.JSONDecodeError:
                        pass
            except FileNotFoundError:
                logger.debug("ruff not found, skipping lint")
            except subprocess.TimeoutExpired:
                logger.warning("ruff timed out")
            except Exception as e:
                logger.debug(f"ruff error: {e}")

        type_errors = len([e for e in errors if e["type"] == "type"])
        lint_errors = len([e for e in errors if e["type"] == "lint"])

        return {
            "status": "ok",
            "errors": errors,
            "summary": {
                "total": len(errors),
                "type_errors": type_errors,
                "lint_errors": lint_errors,
            },
        }

    def _handle_change_impact(self, command: dict) -> dict:
        """Handle change-impact command - find affected tests.

        Uses call graph to find what tests are affected by changed files.
        Two-method discovery:
        1. Call graph traversal: tests that call changed functions
        2. Import analysis: tests that import changed modules

        Args:
            command: Dict with optional:
                - files: List of changed file paths
                - session: If True, use session's dirty files
                - git: If True, use git diff to find changed files

        Returns:
            Response with affected tests list
        """
        import subprocess

        files = command.get("files", [])
        use_session = command.get("session", False)
        use_git = command.get("git", False)

        # Get changed files from various sources
        # Protect read of _dirty_files from concurrent modification by reindex thread
        if use_session:
            with self._dirty_lock:
                if self._dirty_files:
                    files = list(self._dirty_files)
        if not files and use_git:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(self.project),
                )
                if result.returncode == 0:
                    files = [
                        f.strip()
                        for f in result.stdout.strip().split("\n")
                        if f.strip()
                    ]
            except Exception as e:
                logger.debug(f"git diff failed: {e}")

        if not files:
            return {"status": "ok", "affected_tests": [], "message": "No changed files"}

        affected_tests = set()
        changed_functions = set()

        # Extract functions from changed files
        for file_path in files:
            if not file_path.endswith(".py"):
                continue
            # Security: validate path is within project (prevent path traversal)
            try:
                if Path(file_path).is_absolute():
                    full_path = Path(file_path)
                else:
                    full_path = (self.project / file_path).resolve()
                # Check path is within project - raises ValueError if not
                full_path.relative_to(self.project)
            except ValueError:
                logger.warning(f"Path traversal attempt blocked: {file_path}")
                continue
            if not full_path.exists():
                continue

            try:
                from tldr.ast_extractor import extract_file

                info = extract_file(str(full_path))
                for func in info.functions:
                    changed_functions.add(func.name)
            except Exception as e:
                logger.debug(f"Could not extract {file_path}: {e}")

        # Method 1: Call graph traversal - find tests that call changed functions
        # Uses O(1) reverse index lookup instead of O(n) edge scan
        self._ensure_call_graph_loaded()
        reverse_index = self.indexes.get("reverse_call_graph", {})
        if changed_functions and reverse_index:
            for func_name in changed_functions:
                # O(1) lookup of callers using reverse index
                callers = reverse_index.get(func_name, [])
                for caller_info in callers:
                    caller_file = caller_info.get("file", "")
                    if caller_file and "test" in caller_file.lower():
                        affected_tests.add(caller_file)

        # Method 2: Import analysis - find test files that import changed modules
        # Optimization: Scan project once, read each test file once, check all modules
        module_names = {Path(fp).stem for fp in files if fp.endswith(".py")}

        if module_names:
            try:
                from tldr.cross_file_calls import scan_project

                # Scan project ONCE (was called once per file - O(f) down to O(1))
                test_files = [
                    f for f in scan_project(self.project) if "test" in f.lower()
                ]

                # Read each test file ONCE and check all module names
                for test_file in test_files:
                    try:
                        with open(self.project / test_file) as f:
                            content = f.read()
                            # Check all changed modules against this test file
                            for module_name in module_names:
                                if (
                                    f"import {module_name}" in content
                                    or f"from {module_name}" in content
                                ):
                                    affected_tests.add(test_file)
                                    break  # No need to check more modules for this file
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Import analysis failed: {e}")

        return {
            "status": "ok",
            "affected_tests": sorted(list(affected_tests)),
            "changed_files": files,
            "changed_functions": sorted(list(changed_functions)),
            "summary": {
                "files_changed": len(files),
                "functions_changed": len(changed_functions),
                "tests_affected": len(affected_tests),
            },
        }

    def notify_file_changed(self, file_path: str):
        """Notify daemon that a file has changed.

        This invalidates cached queries that depend on this file.

        Args:
            file_path: Absolute path to the changed file
        """
        logger.debug(f"File change notification: {file_path}")

        # Invalidate SalsaDB cache entries for this file
        self.salsa_db.set_file(file_path, "changed")  # Triggers invalidation

        # Update dedup index if loaded
        if self.dedup_index:
            # Re-extract edges for the changed file
            try:
                # Detect language from extension
                lang = "python"
                if file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
                    lang = "typescript"
                elif file_path.endswith(".go"):
                    lang = "go"
                elif file_path.endswith(".rs"):
                    lang = "rust"

                self.dedup_index.get_or_create_edges(file_path, lang=lang)
            except Exception as e:
                logger.debug(f"Could not re-index {file_path}: {e}")

    def write_pid_file(self):
        """Write daemon PID to .tldr/daemon.pid."""
        try:
            self.tldr_dir.mkdir(parents=True, exist_ok=True)
            pid_file = self.tldr_dir / "daemon.pid"
            pid_file.write_text(str(os.getpid()))
            logger.info(f"Wrote PID {os.getpid()} to {pid_file}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to write PID file: {e}")

    def remove_pid_file(self):
        """Remove the PID file."""
        try:
            pid_file = self.tldr_dir / "daemon.pid"
            pid_file.unlink(missing_ok=True)
            logger.info(f"Removed PID file {pid_file}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to remove PID file: {e}")

    def write_status(self, status: str):
        """Write status to .tldr/status file.

        Note: This is called in finally blocks, so it must never raise.
        """
        try:
            self.tldr_dir.mkdir(parents=True, exist_ok=True)
            status_file = self.tldr_dir / "status"
            status_file.write_text(status)
            self._status = status
            logger.info(f"Status: {status}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to write status file: {e}")
            self._status = status  # Still update in-memory status

    def read_status(self) -> str:
        """Read status from .tldr/status file."""
        status_file = self.tldr_dir / "status"
        try:
            if status_file.exists() and status_file.is_file():
                return status_file.read_text().strip()
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to read status file: {e}")
        return "unknown"

    def _create_socket(self):
        """Create and bind the socket (legacy method, calls _create_server_socket)."""
        self._socket = self._create_server_socket()

    def _create_server_socket(self) -> socket.socket:
        """Create appropriate socket for platform.

        On Windows, creates a TCP socket bound to localhost.
        On Unix, creates a Unix domain socket.

        Returns:
            Configured and bound socket ready for listening.

        Raises:
            RuntimeError: If socket binding fails.
        """
        if sys.platform == "win32":
            # TCP on localhost for Windows
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            addr, port = self._get_connection_info()
            try:
                sock.bind((addr, port))
            except (PermissionError, OSError) as e:
                sock.close()
                raise RuntimeError(f"Failed to bind socket at {addr}:{port}: {e}")
            sock.listen(5)
            sock.settimeout(1.0)
            logger.info(f"Listening on {addr}:{port}")
        else:
            # Unix socket for Linux/macOS
            # Clean up any existing socket
            if self.socket_path.exists():
                self.socket_path.unlink()

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(str(self.socket_path))
            except (PermissionError, OSError) as e:
                sock.close()
                raise RuntimeError(f"Failed to bind socket at {self.socket_path}: {e}")
            sock.listen(5)
            sock.settimeout(1.0)
            logger.info(f"Listening on {self.socket_path}")

        return sock

    def _cleanup_socket(self):
        """Clean up the socket."""
        try:
            if self._socket:
                self._socket.close()
                self._socket = None
        except OSError as e:
            logger.warning(f"Error closing socket: {e}")
            self._socket = None

        try:
            self.socket_path.unlink(missing_ok=True)
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to remove socket file: {e}")

        logger.info("Socket cleaned up")

    def _handle_one_connection(self):
        """Handle a single client connection."""
        if not self._socket:
            return

        try:
            conn, _ = self._socket.accept()
        except socket.timeout:
            return
        except OSError:
            return

        try:
            conn.settimeout(5.0)
            data = b""
            size_exceeded = False
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Security: Prevent OOM from malicious clients sending infinite data
                if len(data) > MAX_REQUEST_SIZE:
                    size_exceeded = True
                    logger.warning(
                        f"Request size exceeded {MAX_REQUEST_SIZE} bytes, rejecting connection"
                    )
                    break
                if b"\n" in data:
                    break

            if size_exceeded:
                response = {
                    "status": "error",
                    "message": f"Request too large: exceeds {MAX_REQUEST_SIZE} byte limit",
                }
                conn.sendall(_json_dumps(response) + b"\n")
            elif data:
                try:
                    command = _json_loads(data.strip())
                    response = self.handle_command(command)
                except (json.JSONDecodeError, ValueError) as e:
                    response = {"status": "error", "message": f"Invalid JSON: {e}"}

                conn.sendall(_json_dumps(response) + b"\n")
        except BrokenPipeError:
            # Client disconnected before receiving response - normal occurrence
            logger.debug("Client disconnected before receiving response")
        except Exception:
            logger.exception("Error handling connection")
        finally:
            conn.close()

    def run(self):
        """Run the daemon main loop."""
        self.write_pid_file()
        self.write_status("indexing")

        try:
            self._create_socket()
            self.write_status("ready")

            logger.info(f"TLDR daemon started for {self.project}")

            while not self._shutdown_requested:
                self._handle_one_connection()

                # Check for idle timeout
                if self.is_idle():
                    logger.info("Idle timeout reached, shutting down")
                    break

        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down")
        except Exception:
            logger.exception("Daemon error")
        finally:
            self._cleanup_socket()
            self.remove_pid_file()
            self.write_status("stopped")
            logger.info("Daemon stopped")


def start_daemon(project_path: str | Path, foreground: bool = False):
    """
    Start the TLDR daemon for a project.

    Args:
        project_path: Path to the project root
        foreground: If True, run in foreground; otherwise daemonize
    """
    from .tldrignore import ensure_tldrignore

    project = Path(project_path).resolve()

    # Check if daemon is already running
    try:
        result = query_daemon(project, {"cmd": "ping"})
        if result.get("status") == "ok":
            print("Daemon already running")
            return
    except (ConnectionRefusedError, FileNotFoundError, OSError, json.JSONDecodeError):
        pass  # Daemon not running, proceed with start

    # Ensure .tldrignore exists (create with defaults if not)
    created, message = ensure_tldrignore(project)
    if created:
        print(f"\n\033[33m{message}\033[0m\n")  # Yellow warning

    daemon = TLDRDaemon(project)

    if foreground:
        daemon.run()
    else:
        if sys.platform == "win32":
            # Windows: Use subprocess to run in background
            import subprocess

            # Get the connection info for display
            addr, port = daemon._get_connection_info()

            # Start detached process on Windows
            # These Windows-specific APIs are guarded by platform check above
            startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
            startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]

            # Windows-specific creation flags (0 on other platforms for type safety)
            detached = getattr(subprocess, "DETACHED_PROCESS", 0)
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            proc = subprocess.Popen(
                [sys.executable, "-m", "tldr.daemon", str(project), "--foreground"],
                startupinfo=startupinfo,
                creationflags=detached | no_window,
            )
            print(f"Daemon started with PID {proc.pid}")
            print(f"Listening on {addr}:{port}")
        else:
            # Unix: Fork and run in background
            pid = os.fork()
            if pid == 0:
                # Child process
                os.setsid()
                daemon.run()
            else:
                # Parent process
                print(f"Daemon started with PID {pid}")
                print(f"Socket: {daemon.socket_path}")


def _create_client_socket(daemon: TLDRDaemon) -> socket.socket:
    """Create appropriate client socket for platform.

    Args:
        daemon: TLDRDaemon instance to get connection info from

    Returns:
        Connected socket ready for communication

    Note:
        Socket has CLIENT_SOCKET_TIMEOUT set to prevent indefinite blocking
        if daemon becomes unresponsive.
    """
    addr, port = daemon._get_connection_info()

    if port is not None:
        # TCP socket for Windows
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(CLIENT_SOCKET_TIMEOUT)
        client.connect((addr, port))
    else:
        # Unix socket for Linux/macOS
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(CLIENT_SOCKET_TIMEOUT)
        client.connect(addr)

    return client


def stop_daemon(project_path: str | Path) -> bool:
    """
    Stop the TLDR daemon for a project.

    Args:
        project_path: Path to the project root

    Returns:
        True if daemon was stopped, False if not running
    """
    project = Path(project_path).resolve()
    daemon = TLDRDaemon(project)

    try:
        client = _create_client_socket(daemon)
        client.sendall(_json_dumps({"cmd": "shutdown"}) + b"\n")
        client.recv(4096)  # Wait for response before closing
        client.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


def query_daemon(project_path: str | Path, command: dict) -> dict:
    """
    Send a command to the daemon and get the response.

    Args:
        project_path: Path to the project root
        command: Command dict to send

    Returns:
        Response dict from daemon

    Note:
        Uses chunked reading to handle large responses that exceed
        the initial recv buffer size.
    """
    project = Path(project_path).resolve()
    daemon = TLDRDaemon(project)

    client = _create_client_socket(daemon)
    try:
        client.sendall(_json_dumps(command) + b"\n")
        # Read until newline (complete JSON response) using chunked reads
        response = b""
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            response += chunk
            if b"\n" in response:
                break
        return _json_loads(response)
    finally:
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TLDR Daemon")
    parser.add_argument("project", help="Project path")
    parser.add_argument(
        "--foreground", "-f", action="store_true", help="Run in foreground"
    )
    parser.add_argument("--stop", action="store_true", help="Stop the daemon")
    parser.add_argument("--status", action="store_true", help="Get daemon status")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.stop:
        if stop_daemon(args.project):
            print("Daemon stopped")
        else:
            print("Daemon not running")
    elif args.status:
        try:
            result = query_daemon(args.project, {"cmd": "status"})
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Daemon not running: {e}")
    else:
        start_daemon(args.project, foreground=args.foreground)
