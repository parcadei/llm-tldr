"""Salsa-style query memoization for TLDR (P5).

Salsa is rust-analyzer's incremental computation framework. Key concepts:

1. **Queries as Functions**: Everything is a query with automatic memoization
2. **Automatic Dependency Tracking**: Queries record which other queries they call
3. **Minimal Re-computation**: Only affected queries re-run on change

Example usage:
    from tldr.salsa import SalsaDB, salsa_query

    @salsa_query
    def read_file(db: SalsaDB, path: str) -> str:
        return db.get_file(path)

    @salsa_query
    def parse_file(db: SalsaDB, path: str) -> dict:
        content = db.query(read_file, db, path)
        return parse(content)

    db = SalsaDB()
    db.set_file("auth.py", "def login(): pass")
    result = db.query(parse_file, db, "auth.py")

    # When file changes, dependent queries auto-invalidate
    db.set_file("auth.py", "def login(): pass\\ndef logout(): pass")
    result = db.query(parse_file, db, "auth.py")  # Recomputes
"""
from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

# Type variables for generic query handling
T = TypeVar("T")
QueryKey = Tuple[Callable, Tuple[Any, ...]]


# Marker for salsa queries
_SALSA_QUERY_MARKER = "_is_salsa_query"


def salsa_query(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to mark a function as a Salsa query.

    Salsa queries:
    - Are automatically memoized when called through SalsaDB.query()
    - Track their dependencies on other queries
    - Can be invalidated, cascading to dependents

    Example:
        @salsa_query
        def get_functions(db: SalsaDB, path: str) -> List[str]:
            content = db.query(read_file, db, path)
            return extract_functions(content)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # When called directly (not through db.query), just execute
        return func(*args, **kwargs)

    # Mark as salsa query
    setattr(wrapper, _SALSA_QUERY_MARKER, True)
    wrapper._original_func = func

    return wrapper


def is_salsa_query(func: Callable) -> bool:
    """Check if a function is decorated with @salsa_query."""
    return getattr(func, _SALSA_QUERY_MARKER, False)


@dataclass
class CacheEntry:
    """Cache entry for a query result."""

    result: Any
    dependencies: Set[QueryKey] = field(default_factory=set)
    file_dependencies: Dict[str, int] = field(default_factory=dict)  # path -> revision


@dataclass
class QueryStats:
    """Statistics for query execution."""

    cache_hits: int = 0
    cache_misses: int = 0
    invalidations: int = 0
    recomputations: int = 0


class SalsaDB:
    """Database for Salsa-style query memoization.

    Tracks:
    - File contents and revisions
    - Query results and their dependencies
    - Reverse dependency graph for invalidation cascading

    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # File storage
        self._file_contents: Dict[str, str] = {}
        self._file_revisions: Dict[str, int] = {}

        # Query cache: (func, args) -> CacheEntry
        self._query_cache: Dict[QueryKey, CacheEntry] = {}

        # Reverse dependencies: query_key -> set of dependent query_keys
        self._reverse_deps: Dict[QueryKey, Set[QueryKey]] = {}

        # File to query dependencies: file_path -> set of query_keys
        self._file_to_queries: Dict[str, Set[QueryKey]] = {}

        # Stats
        self._stats = QueryStats()

        # Active query stack for dependency tracking
        self._query_stack: List[QueryKey] = []

        # Pending dependencies for queries currently being computed
        self._pending_deps: Dict[QueryKey, Set[QueryKey]] = {}

    # -------------------------------------------------------------------------
    # File Management
    # -------------------------------------------------------------------------

    def set_file(self, path: str, content: str) -> None:
        """Set or update file content.

        This increments the file's revision and invalidates any queries
        that depend on this file.

        Args:
            path: File path (used as key)
            content: File content
        """
        with self._lock:
            old_revision = self._file_revisions.get(path, 0)
            self._file_contents[path] = content
            self._file_revisions[path] = old_revision + 1

            # Invalidate queries that depend on this file
            self._invalidate_file_dependents(path)

    def get_file(self, path: str) -> Optional[str]:
        """Get file content.

        If called during a query, registers the file as a dependency.

        Args:
            path: File path

        Returns:
            File content or None if not found
        """
        with self._lock:
            # Track file dependency if in a query context
            if self._query_stack:
                current_query = self._query_stack[-1]
                if path not in self._file_to_queries:
                    self._file_to_queries[path] = set()
                self._file_to_queries[path].add(current_query)

                # Also track in the cache entry if it exists
                if current_query in self._query_cache:
                    entry = self._query_cache[current_query]
                    entry.file_dependencies[path] = self._file_revisions.get(path, 0)

            return self._file_contents.get(path)

    def get_revision(self, path: str) -> int:
        """Get current revision number for a file.

        Args:
            path: File path

        Returns:
            Revision number (0 if file never set)
        """
        with self._lock:
            return self._file_revisions.get(path, 0)

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def query(self, func: Callable[..., T], *args) -> T:
        """Execute a query with memoization and dependency tracking.

        If the query result is cached and valid, returns cached result.
        Otherwise, computes the result and caches it.

        Args:
            func: The query function (should be decorated with @salsa_query)
            *args: Arguments to pass to the function

        Returns:
            Query result
        """
        key = self._make_key(func, args)

        with self._lock:
            # Check cache first
            if key in self._query_cache:
                if self._is_entry_valid(key):
                    self._stats.cache_hits += 1
                    # Still register dependency to parent even on cache hit
                    self._register_dependency_to_parent(key)
                    return self._query_cache[key].result

            self._stats.cache_misses += 1

            # Create a placeholder for collecting dependencies during execution
            self._pending_deps[key] = set()

            # Push onto query stack for dependency tracking
            self._query_stack.append(key)

            try:
                # Execute the query
                if is_salsa_query(func):
                    # Use original function to avoid wrapper overhead
                    original = getattr(func, "_original_func", func)
                    result = original(*args)
                else:
                    result = func(*args)

                self._stats.recomputations += 1

                # Create cache entry with collected dependencies
                entry = CacheEntry(
                    result=result,
                    dependencies=self._pending_deps.get(key, set()).copy(),
                )

                # Capture file dependencies
                for path in list(self._file_to_queries.keys()):
                    if key in self._file_to_queries.get(path, set()):
                        entry.file_dependencies[path] = self._file_revisions.get(
                            path, 0
                        )

                self._query_cache[key] = entry

                # Register dependency to parent only on successful completion
                # Must be inside try block to avoid registering failed queries
                # as dependencies (which corrupts parent's validity check)
                self._register_dependency_to_parent(key)

                return result

            finally:
                self._query_stack.pop()

                # Clean up pending deps for this key
                if key in self._pending_deps:
                    del self._pending_deps[key]

    def _register_dependency_to_parent(self, child_key: QueryKey) -> None:
        """Register a child query as a dependency of the current parent query."""
        if not self._query_stack:
            return

        parent_key = self._query_stack[-1]

        # Track in pending deps (for queries still being computed)
        if hasattr(self, "_pending_deps") and parent_key in self._pending_deps:
            self._pending_deps[parent_key].add(child_key)

        # Track in cached entry (for queries already computed)
        if parent_key in self._query_cache:
            self._query_cache[parent_key].dependencies.add(child_key)

        # Track reverse dependency
        if child_key not in self._reverse_deps:
            self._reverse_deps[child_key] = set()
        self._reverse_deps[child_key].add(parent_key)

    def _make_key(self, func: Callable, args: Tuple[Any, ...]) -> QueryKey:
        """Create a cache key from function and arguments.

        Handles SalsaDB instances by using id() for hashing.
        """
        # Convert args to hashable form
        hashable_args = []
        for arg in args:
            if isinstance(arg, SalsaDB):
                # Use id for SalsaDB instances (they're not hashable otherwise)
                hashable_args.append(("__salsa_db__", id(arg)))
            elif isinstance(arg, (list, dict, set)):
                # Convert mutable types
                hashable_args.append(self._to_hashable(arg))
            else:
                hashable_args.append(arg)

        return (func, tuple(hashable_args))

    def _to_hashable(self, obj: Any) -> Any:
        """Convert an object to a hashable form."""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._to_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._to_hashable(x) for x in obj)
        elif isinstance(obj, set):
            return frozenset(self._to_hashable(x) for x in obj)
        return obj

    def _is_entry_valid(
        self, key: QueryKey, visited: Optional[Set[QueryKey]] = None
    ) -> bool:
        """Check if a cache entry is still valid.

        An entry is valid if:
        - All file dependencies have the same revision
        - All query dependencies are still valid

        Args:
            key: The query key to validate
            visited: Set of already-visited keys for cycle detection

        Returns:
            True if the entry is valid, False otherwise
        """
        # Initialize visited set on first call to detect cycles
        if visited is None:
            visited = set()

        # Cycle detection: if we've already visited this key, consider it valid
        # to break the recursion (we haven't invalidated it yet in this traversal)
        if key in visited:
            return True

        if key not in self._query_cache:
            return False

        visited.add(key)
        entry = self._query_cache[key]

        # Check file dependencies
        for path, revision in entry.file_dependencies.items():
            current_revision = self._file_revisions.get(path, 0)
            if current_revision != revision:
                return False

        # Check query dependencies (recursively with cycle detection)
        for dep_key in entry.dependencies:
            if not self._is_entry_valid(dep_key, visited):
                return False

        return True

    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------

    def get_dependencies(self, func: Callable, *args) -> Set[QueryKey]:
        """Get the dependencies of a query.

        Args:
            func: Query function
            *args: Query arguments

        Returns:
            Set of (func, args) tuples this query depends on
        """
        key = self._make_key(func, args)
        with self._lock:
            if key in self._query_cache:
                return self._query_cache[key].dependencies.copy()
            return set()

    # -------------------------------------------------------------------------
    # Invalidation
    # -------------------------------------------------------------------------

    def invalidate(self, func: Callable, *args) -> None:
        """Invalidate a specific query and its dependents.

        Args:
            func: Query function
            *args: Query arguments (if empty, invalidates all instances)
        """
        with self._lock:
            self._stats.invalidations += 1

            if args:
                key = self._make_key(func, args)
                self._invalidate_key(key)
            else:
                # Invalidate all instances of this function
                keys_to_invalidate = [
                    k for k in self._query_cache.keys() if k[0] == func
                ]
                for key in keys_to_invalidate:
                    self._invalidate_key(key)

    def _invalidate_key(self, key: QueryKey) -> None:
        """Invalidate a specific query key and cascade to dependents."""
        if key not in self._query_cache:
            return

        # Remove from cache
        del self._query_cache[key]

        # Cascade to dependents (reverse deps)
        if key in self._reverse_deps:
            dependents = list(self._reverse_deps[key])
            del self._reverse_deps[key]
            for dep_key in dependents:
                self._invalidate_key(dep_key)

    def _invalidate_file_dependents(self, path: str) -> None:
        """Invalidate all queries that depend on a file."""
        if path not in self._file_to_queries:
            return

        queries = list(self._file_to_queries[path])
        del self._file_to_queries[path]

        for key in queries:
            self._invalidate_key(key)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Get query execution statistics.

        Returns:
            Dict with keys: cache_hits, cache_misses, invalidations, recomputations
        """
        with self._lock:
            return {
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "invalidations": self._stats.invalidations,
                "recomputations": self._stats.recomputations,
            }

    def clear(self) -> None:
        """Clear all cached queries and file data."""
        with self._lock:
            self._query_cache.clear()
            self._reverse_deps.clear()
            self._file_to_queries.clear()
            # Keep file contents and revisions
            # Reset stats
            self._stats = QueryStats()
