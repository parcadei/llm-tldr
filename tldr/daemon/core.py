"""
TLDR Daemon core - the main TLDRDaemon server class.

Holds indexes in memory and handles commands via Unix/TCP socket.
"""

import atexit
import hashlib
import json
import logging
import os
import signal
import socket
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Optional

from tldr.dedup import ContentHashedIndex
from tldr.salsa import SalsaDB
from tldr.stats import (
    HookStats,
    HookStatsStore,
    SessionStats,
    StatsStore,
    count_tokens,
    get_default_store,
)

from .cached_queries import (
    cached_architecture,
    cached_cfg,
    cached_context,
    cached_dead_code,
    cached_dfg,
    cached_extract,
    cached_importers,
    cached_imports,
    cached_search,
    cached_slice,
    cached_structure,
    cached_tree,
)

# Idle timeout: 30 minutes
IDLE_TIMEOUT = 30 * 60

logger = logging.getLogger(__name__)


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
        self._dirty_count: int = 0
        self._dirty_files: set[str] = set()
        self._reindex_in_progress: bool = False
        self._semantic_config = self._load_semantic_config()

        # P7 Features: Per-session token stats tracking
        self._session_stats: dict[str, SessionStats] = {}
        self._stats_store: StatsStore = get_default_store()

        # P8 Features: Per-hook activity stats tracking with persistence
        self._hook_stats_store: HookStatsStore = HookStatsStore(project_path)
        self._hook_stats: dict[str, HookStats] = self._hook_stats_store.load()
        self._hook_stats_baseline: dict[str, HookStats] = self._snapshot_hook_stats()
        self._hook_invocation_count: int = 0
        self._hook_flush_threshold: int = 5  # Flush every N invocations

        # Cross-platform graceful shutdown: register atexit handler
        # This ensures stats persist even if daemon is killed (works on all platforms)
        self._stats_persisted = False  # Guard against double-persist
        atexit.register(self._persist_all_stats)

    def _compute_socket_path(self) -> Path:
        """Compute deterministic socket path from project path."""
        hash_val = hashlib.md5(str(self.project).encode()).hexdigest()[:8]
        temp_dir = Path(tempfile.gettempdir())
        return temp_dir / f"tldr-{hash_val}.sock"

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
            hash_val = hashlib.md5(str(self.project).encode()).hexdigest()[:8]
            port = 49152 + (int(hash_val, 16) % 10000)
            return ("127.0.0.1", port)
        else:
            # Unix socket path
            return (str(self.socket_path), None)

    def is_idle(self) -> bool:
        """Check if daemon has been idle longer than IDLE_TIMEOUT."""
        return (time.time() - self.last_query) > IDLE_TIMEOUT

    @property
    def call_graph(self) -> dict:
        """Get the call graph, loading if necessary."""
        self._ensure_call_graph_loaded()
        return self.indexes.get("call_graph", {"edges": [], "nodes": {}})

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
            "track": self._handle_track,
        }

        handler = handlers.get(cmd)
        if handler:
            return handler(command)
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}

    def _handle_ping(self, command: dict) -> dict:
        """Handle ping command."""
        return {"status": "ok"}

    def _get_session_stats(self, session_id: str) -> SessionStats:
        """Get or create session stats for a session ID."""
        if session_id not in self._session_stats:
            self._session_stats[session_id] = SessionStats(session_id=session_id)
        return self._session_stats[session_id]

    def _get_hook_stats(self, hook_name: str) -> HookStats:
        """Get or create hook stats for a hook name."""
        if hook_name not in self._hook_stats:
            self._hook_stats[hook_name] = HookStats(hook_name=hook_name)
        return self._hook_stats[hook_name]

    def _snapshot_hook_stats(self) -> dict[str, HookStats]:
        """Create a deep copy of current hook stats for delta tracking."""
        from copy import deepcopy
        return {name: deepcopy(stats) for name, stats in self._hook_stats.items()}

    def _handle_track(self, command: dict) -> dict:
        """Handle track command for hook activity reporting.

        Command format:
            {
                "cmd": "track",
                "hook": "hook-name",
                "success": true/false (default: true),
                "metrics": {"key": value, ...} (optional)
            }

        Flushes to disk every N invocations (default: 5) for durability
        while avoiding excessive I/O.
        """
        hook_name = command.get("hook")
        if not hook_name:
            return {"status": "error", "message": "Missing 'hook' field"}

        success = command.get("success", True)
        metrics = command.get("metrics", {})

        # Record the invocation
        hook_stats = self._get_hook_stats(hook_name)
        hook_stats.record_invocation(success=success, metrics=metrics)

        # Increment global invocation counter and flush periodically
        self._hook_invocation_count += 1
        flushed = False
        if self._hook_invocation_count >= self._hook_flush_threshold:
            self._flush_hook_stats()
            flushed = True

        return {
            "status": "ok",
            "hook": hook_name,
            "total_invocations": hook_stats.invocations,
            "flushed": flushed,
        }

    def _flush_hook_stats(self) -> None:
        """Flush hook stats delta to disk and reset counter."""
        try:
            self._hook_stats_store.flush_delta(self._hook_stats, self._hook_stats_baseline)
            self._hook_stats_baseline = self._snapshot_hook_stats()
            self._hook_invocation_count = 0
            logger.debug("Flushed hook stats to disk")
        except Exception as e:
            logger.error(f"Failed to flush hook stats: {e}")

    def _handle_status(self, command: dict) -> dict:
        """Handle status command with P5 cache statistics."""
        uptime = time.time() - self._start_time

        # Get SalsaDB stats
        salsa_stats = self.salsa_db.get_stats()

        # Get dedup stats if loaded
        dedup_stats = {}
        if self.dedup_index:
            dedup_stats = self.dedup_index.stats()

        # Get session stats if session ID provided
        session_id = command.get("session")
        session_stats = None
        if session_id:
            stats = self._session_stats.get(session_id)
            if stats:
                session_stats = stats.to_dict()

        # Get all sessions summary
        all_sessions_stats = {
            "active_sessions": len(self._session_stats),
            "total_raw_tokens": sum(s.raw_tokens for s in self._session_stats.values()),
            "total_tldr_tokens": sum(s.tldr_tokens for s in self._session_stats.values()),
            "total_requests": sum(s.requests for s in self._session_stats.values()),
        }

        # Get all hook stats (P8)
        hook_stats_dict = {
            name: stats.to_dict() for name, stats in self._hook_stats.items()
        }

        return {
            "status": self._status,
            "uptime": uptime,
            "files": len(self.indexes.get("files", [])),
            "project": str(self.project),
            "salsa_stats": salsa_stats,
            "dedup_stats": dedup_stats,
            "session_stats": session_stats,
            "all_sessions": all_sessions_stats,
            "hook_stats": hook_stats_dict,
        }

    def _handle_shutdown(self, command: dict) -> dict:
        """Handle shutdown command with stats persistence."""
        # Persist all session stats before shutdown
        self._persist_all_stats()
        self._shutdown_requested = True
        return {"status": "shutting_down"}

    def _persist_all_stats(self) -> None:
        """Persist all session and hook stats to JSONL stores.

        Thread-safe: uses a flag to prevent double-persist when both
        atexit and finally block trigger this method.
        """
        # Guard against double-persist (atexit + finally can both trigger)
        if self._stats_persisted:
            return
        self._stats_persisted = True

        # Persist session stats
        for session_id, stats in self._session_stats.items():
            if stats.requests > 0:  # Only persist if there were actual requests
                try:
                    self._stats_store.append(stats)
                    logger.info(
                        f"Persisted stats for session {session_id}: "
                        f"{stats.requests} requests, {stats.savings_percent:.1f}% savings"
                    )
                except Exception as e:
                    logger.error(f"Failed to persist stats for session {session_id}: {e}")

        # Persist hook stats (final flush)
        if self._hook_invocation_count > 0:
            self._flush_hook_stats()
            logger.info(f"Persisted hook stats for {len(self._hook_stats)} hooks")

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
        """Handle extract command with SalsaDB caching and token tracking."""
        file_path = command.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing required parameter: file"}

        try:
            # Track tokens if session ID provided
            session_id = command.get("session")
            raw_tokens = 0

            if session_id:
                # Count raw file tokens (what vanilla Claude would use)
                try:
                    raw_content = Path(file_path).read_text()
                    raw_tokens = count_tokens(raw_content)
                except Exception:
                    pass  # File might not exist or be binary

            # Use SalsaDB for cached extraction
            result = self.salsa_db.query(cached_extract, self.salsa_db, file_path)

            # Track token savings if session ID provided
            if session_id and raw_tokens > 0:
                tldr_tokens = count_tokens(json.dumps(result))
                stats = self._get_session_stats(session_id)
                stats.record_request(raw_tokens=raw_tokens, tldr_tokens=tldr_tokens)

                # Incremental persistence: save every 10 requests
                if stats.requests % 10 == 0:
                    try:
                        self._stats_store.append(stats)
                        logger.debug(f"Persisted stats for session {session_id}: {stats.requests} requests")
                    except Exception as e:
                        logger.warning(f"Failed to persist stats: {e}")

            return result
        except Exception as e:
            logger.exception("Extract failed")
            return {"status": "error", "message": str(e)}

    def _handle_impact(self, command: dict) -> dict:
        """Handle impact command - find callers of a function."""
        func_name = command.get("func")
        if not func_name:
            return {"status": "error", "message": "Missing required parameter: func"}

        try:
            self._ensure_call_graph_loaded()
            call_graph = self.indexes.get("call_graph", {})

            # Find all callers of the function
            callers = []
            edges = call_graph.get("edges", [])
            for edge in edges:
                if edge.get("callee") == func_name:
                    callers.append({
                        "caller": edge.get("caller"),
                        "file": edge.get("file"),
                        "line": edge.get("line"),
                    })

            return {"status": "ok", "callers": callers}
        except Exception as e:
            logger.exception("Impact analysis failed")
            return {"status": "error", "message": str(e)}

    def _ensure_call_graph_loaded(self):
        """Load call graph if not already loaded."""
        if "call_graph" in self.indexes:
            return

        call_graph_path = self.tldr_dir / "call_graph.json"
        if call_graph_path.exists():
            try:
                self.indexes["call_graph"] = json.loads(call_graph_path.read_text())
                logger.info(f"Loaded call graph from {call_graph_path}")
            except Exception as e:
                logger.error(f"Failed to load call graph: {e}")
                self.indexes["call_graph"] = {"edges": [], "nodes": {}}
        else:
            logger.warning(f"No call graph found at {call_graph_path}")
            self.indexes["call_graph"] = {"edges": [], "nodes": {}}

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
            return {"status": "error", "message": "Missing required parameters: file, function"}

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
            return {"status": "error", "message": "Missing required parameters: file, function"}

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
            return {"status": "error", "message": "Missing required parameters: file, function, line"}

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
                    {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
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
                    {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                    for e in graph.edges
                ],
                "languages": [language],
                "timestamp": time.time(),
            }
            cache_file.write_text(json.dumps(cache_data, indent=2))

            # Also update in-memory index
            self.indexes["call_graph"] = cache_data

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
                    return {"status": "error", "message": "Missing required parameter: query"}
                k = command.get("k", 10)
                results = semantic_search(str(self.project), query, k=k)
                return {"status": "ok", "results": results}

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception("Semantic operation failed")
            return {"status": "error", "message": str(e)}

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

        # Track dirty file
        if file_path not in self._dirty_files:
            self._dirty_files.add(file_path)
            self._dirty_count += 1
            logger.info(f"Dirty file tracked: {file_path} (count: {self._dirty_count})")

        # Notify Salsa for cache invalidation
        self.notify_file_changed(file_path)

        # Check if we should trigger background re-indexing
        threshold = self._semantic_config.get("auto_reindex_threshold", 20)
        should_reindex = (
            self._dirty_count >= threshold
            and not self._reindex_in_progress
        )

        if should_reindex:
            self._trigger_background_reindex()

        return {
            "status": "ok",
            "dirty_count": self._dirty_count,
            "threshold": threshold,
            "reindex_triggered": should_reindex,
        }

    def _trigger_background_reindex(self):
        """Trigger background semantic re-indexing.

        Spawns a subprocess to rebuild the semantic index,
        allowing the daemon to continue serving requests.
        """
        if self._reindex_in_progress:
            logger.info("Re-index already in progress, skipping")
            return

        self._reindex_in_progress = True
        dirty_files = list(self._dirty_files)
        logger.info(f"Triggering background semantic re-index for {len(dirty_files)} files")

        def do_reindex():
            try:
                import subprocess

                # Run semantic index command
                cmd = [
                    sys.executable, "-m", "tldr.cli",
                    "semantic", "index", str(self.project)
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
                    logger.error(f"Background semantic re-index failed: {result.stderr}")

            except Exception as e:
                logger.exception(f"Background semantic re-index error: {e}")
            finally:
                # Reset dirty tracking
                self._dirty_files.clear()
                self._dirty_count = 0
                self._reindex_in_progress = False

        # Run in thread to not block daemon
        import threading
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
            return {"status": "error", "message": "Missing required parameter: file or project"}

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
                    pyright_output = json.loads(result.stdout)
                    for diag in pyright_output.get("generalDiagnostics", []):
                        errors.append({
                            "type": "type",
                            "severity": diag.get("severity", "error"),
                            "file": diag.get("file", ""),
                            "line": diag.get("range", {}).get("start", {}).get("line", 0),
                            "message": diag.get("message", ""),
                            "rule": diag.get("rule", "pyright"),
                        })
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
                        ruff_output = json.loads(result.stdout)
                        for diag in ruff_output:
                            errors.append({
                                "type": "lint",
                                "severity": "warning" if diag.get("fix") else "error",
                                "file": diag.get("filename", ""),
                                "line": diag.get("location", {}).get("row", 0),
                                "message": diag.get("message", ""),
                                "rule": diag.get("code", "ruff"),
                            })
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
        if use_session and self._dirty_files:
            files = list(self._dirty_files)
        elif use_git:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(self.project),
                )
                if result.returncode == 0:
                    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
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
            full_path = self.project / file_path if not Path(file_path).is_absolute() else Path(file_path)
            if not full_path.exists():
                continue

            try:
                from tldr.ast_extractor import extract_file
                info = extract_file(str(full_path))
                for func in info.get("functions", []):
                    changed_functions.add(func.get("name", ""))
            except Exception as e:
                logger.debug(f"Could not extract {file_path}: {e}")

        # Method 1: Call graph traversal - find tests that call changed functions
        if changed_functions and self.call_graph:
            for func_name in changed_functions:
                # Find callers of this function
                for edge in self.call_graph.get("edges", []):
                    if edge.get("to_func") == func_name:
                        caller_file = edge.get("from_file", "")
                        if "test" in caller_file.lower():
                            affected_tests.add(caller_file)

        # Method 2: Import analysis - find test files that import changed modules
        for file_path in files:
            if not file_path.endswith(".py"):
                continue
            module_name = Path(file_path).stem

            # Search for imports of this module in test files
            try:
                from tldr.cross_file_calls import scan_project
                test_files = [f for f in scan_project(self.project) if "test" in f.lower()]

                for test_file in test_files:
                    try:
                        with open(self.project / test_file) as f:
                            content = f.read()
                            if f"import {module_name}" in content or f"from {module_name}" in content:
                                affected_tests.add(test_file)
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
        self.tldr_dir.mkdir(parents=True, exist_ok=True)
        pid_file = self.tldr_dir / "daemon.pid"
        pid_file.write_text(str(os.getpid()))
        logger.info(f"Wrote PID {os.getpid()} to {pid_file}")

    def remove_pid_file(self):
        """Remove the PID file."""
        pid_file = self.tldr_dir / "daemon.pid"
        if pid_file.exists():
            pid_file.unlink()
            logger.info(f"Removed PID file {pid_file}")

    def write_status(self, status: str):
        """Write status to .tldr/status file."""
        self.tldr_dir.mkdir(parents=True, exist_ok=True)
        status_file = self.tldr_dir / "status"
        status_file.write_text(status)
        self._status = status
        logger.info(f"Status: {status}")

    def read_status(self) -> str:
        """Read status from .tldr/status file."""
        status_file = self.tldr_dir / "status"
        if status_file.exists():
            return status_file.read_text().strip()
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
        """
        import errno

        if sys.platform == "win32":
            # TCP on localhost for Windows
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            addr, port = self._get_connection_info()
            sock.bind((addr, port))
            sock.listen(5)
            sock.settimeout(1.0)
            logger.info(f"Listening on {addr}:{port}")
        else:
            # Unix socket for Linux/macOS
            # Try to bind without deleting existing socket - if bind fails,
            # another daemon is running. This prevents race conditions.
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # Don't use SO_REUSEADDR for Unix sockets - it allows multiple binds
            try:
                sock.bind(str(self.socket_path))
            except OSError as e:
                # Socket exists and is in use - clean up and retry once
                # EADDRINUSE is 48 on macOS, 98 on Linux
                if e.errno == errno.EADDRINUSE or "Address already in use" in str(e):
                    # Check if existing daemon is responsive
                    if self.socket_path.exists():
                        try:
                            test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            test_sock.connect(str(self.socket_path))
                            test_sock.close()
                            # Another daemon is running - exit
                            sock.close()
                            raise RuntimeError("Another daemon is already running")
                        except ConnectionRefusedError:
                            # Stale socket - remove and retry
                            self.socket_path.unlink()
                            sock.bind(str(self.socket_path))
                        except FileNotFoundError:
                            # Socket was removed between check and connect
                            sock.bind(str(self.socket_path))
                else:
                    raise
            sock.listen(5)
            sock.settimeout(1.0)
            logger.info(f"Listening on {self.socket_path}")

        return sock

    def _cleanup_socket(self):
        """Clean up the socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self.socket_path.exists():
            self.socket_path.unlink()
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
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if data:
                try:
                    command = json.loads(data.decode().strip())
                    response = self.handle_command(command)
                except json.JSONDecodeError as e:
                    response = {"status": "error", "message": f"Invalid JSON: {e}"}

                conn.sendall(json.dumps(response).encode() + b"\n")
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

        # Cross-platform signal handling for graceful shutdown
        # Signal handlers just set the flag - actual cleanup happens in finally block
        def _signal_handler(signum: int, frame: Any) -> None:
            signame = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
            logger.info(f"Received {signame}, initiating graceful shutdown")
            self._shutdown_requested = True

        # SIGINT works on all platforms (Ctrl+C)
        signal.signal(signal.SIGINT, _signal_handler)

        # SIGTERM only on Unix/Mac (Windows ignores it but doesn't raise)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, _signal_handler)

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
            # Persist stats before cleanup (graceful shutdown)
            try:
                self._persist_all_stats()
                logger.info("Stats persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist stats on shutdown: {e}")

            self._cleanup_socket()
            self.remove_pid_file()
            self.write_status("stopped")
            logger.info("Daemon stopped")
