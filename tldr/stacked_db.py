"""Stacked Immutable DBs (P5 #19) - Meta Glean Pattern.

This module implements the stacked immutable database pattern:

```
Base Index (immutable)
    |
Stack 1: Session start snapshot (immutable)
    |
Stack 2: Morning edits (immutable)
    |
Stack 3: Current working changes (mutable)
```

**Query:** Search all stacks top-to-bottom, first match wins.

Benefits:
1. Non-destructive: Never lose data, can always roll back
2. Concurrent readers: Readers see consistent snapshots
3. Cheap branching: Fork a stack for speculative work
4. Time travel: Query "what did the codebase look like at 2pm?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4


@dataclass
class Edge:
    """A call graph edge with unique ID for tracking across stacks.

    Format: (src_file, src_func, dst_file, dst_func)
    """
    id: str
    src_file: str
    src_func: str
    dst_file: str
    dst_func: str

    def to_tuple(self) -> tuple[str, str, str, str]:
        """Convert to standard tuple format."""
        return (self.src_file, self.src_func, self.dst_file, self.dst_func)

    @classmethod
    def from_tuple(
        cls,
        src_file: str,
        src_func: str,
        dst_file: str,
        dst_func: str,
        edge_id: Optional[str] = None
    ) -> Edge:
        """Create Edge from standard tuple format."""
        return cls(
            id=edge_id or str(uuid4()),
            src_file=src_file,
            src_func=src_func,
            dst_file=dst_file,
            dst_func=dst_func
        )

    def to_dict(self) -> dict:
        """Serialize to dict for persistence."""
        return {
            "id": self.id,
            "src_file": self.src_file,
            "src_func": self.src_func,
            "dst_file": self.dst_file,
            "dst_func": self.dst_func
        }

    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            src_file=data["src_file"],
            src_func=data["src_func"],
            dst_file=data["dst_file"],
            dst_func=data["dst_func"]
        )


@dataclass
class ImmutableStack:
    """An immutable stack layer containing edges and deletions.

    Each stack can:
    - Add new edges (stored in `edges`)
    - Mark edges as deleted (stored in `deletions`)
    - Reference a parent stack for hierarchical queries

    Query semantics: Search current stack first, then parent, then grandparent, etc.
    If an edge ID is in `deletions`, it's treated as non-existent even if in parent.
    """
    id: str
    parent: Optional[ImmutableStack] = None
    created_at: datetime = field(default_factory=datetime.now)
    edges: list[Edge] = field(default_factory=list)
    deletions: set[str] = field(default_factory=set)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to this stack."""
        self.edges.append(edge)

    def mark_deleted(self, edge_id: str) -> None:
        """Mark an edge as deleted in this stack."""
        self.deletions.add(edge_id)

    def query_edge(self, edge_id: str) -> Optional[Edge]:
        """Query for an edge by ID, respecting deletions.

        Returns None if:
        - Edge is marked deleted in this stack
        - Edge not found in this stack or any parent
        """
        # Check if deleted in this stack
        if edge_id in self.deletions:
            return None

        # Check local edges
        for edge in self.edges:
            if edge.id == edge_id:
                return edge

        # Recurse to parent
        if self.parent:
            return self.parent.query_edge(edge_id)

        return None

    def get_all_edges(self, seen_deletions: Optional[set[str]] = None) -> list[Edge]:
        """Get all edges from this stack and parents, respecting deletions.

        Args:
            seen_deletions: Set of edge IDs already marked as deleted
                           (accumulated from child stacks)

        Returns:
            List of edges not marked as deleted
        """
        if seen_deletions is None:
            seen_deletions = set()

        # Accumulate this stack's deletions
        all_deletions = seen_deletions | self.deletions

        result = []

        # Add local edges not deleted
        for edge in self.edges:
            if edge.id not in all_deletions:
                result.append(edge)

        # Recurse to parent
        if self.parent:
            parent_edges = self.parent.get_all_edges(all_deletions)
            result.extend(parent_edges)

        return result

    def depth(self) -> int:
        """Return the depth of this stack chain."""
        if self.parent is None:
            return 1
        return 1 + self.parent.depth()

    def to_dict(self) -> dict:
        """Serialize this stack (and parent chain) to dict."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "edges": [e.to_dict() for e in self.edges],
            "deletions": list(self.deletions),
            "parent": self.parent.to_dict() if self.parent else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImmutableStack:
        """Deserialize stack chain from dict."""
        parent = None
        if data.get("parent"):
            parent = cls.from_dict(data["parent"])

        return cls(
            id=data["id"],
            parent=parent,
            created_at=datetime.fromisoformat(data["created_at"]),
            edges=[Edge.from_dict(e) for e in data.get("edges", [])],
            deletions=set(data.get("deletions", []))
        )


class StackedDB:
    """A database with stacked immutable layers.

    Provides:
    - Non-destructive updates via stack layering
    - Forking for speculative work
    - Rollback to previous states
    - Time travel queries
    - Compaction for performance
    """

    def __init__(self, base: Optional[ImmutableStack] = None):
        """Initialize with optional base stack."""
        if base is None:
            base = ImmutableStack(id=str(uuid4()))
        self.current = base

    def add_edge(
        self,
        src_file: str,
        src_func: str,
        dst_file: str,
        dst_func: str
    ) -> Edge:
        """Add an edge to the current stack."""
        edge = Edge.from_tuple(src_file, src_func, dst_file, dst_func)
        self.current.add_edge(edge)
        return edge

    def remove_edge(self, edge_id: str) -> None:
        """Mark an edge as deleted in the current stack."""
        self.current.mark_deleted(edge_id)

    def get_all_edges(self) -> list[tuple[str, str, str, str]]:
        """Get all edges as tuples, merged from all stacks."""
        edges = self.current.get_all_edges()
        return [e.to_tuple() for e in edges]

    def get_edges_for_file(self, file_path: str) -> list[tuple[str, str, str, str]]:
        """Get all edges originating from a specific file."""
        all_edges = self.get_all_edges()
        return [e for e in all_edges if e[0] == file_path]

    def fork(self) -> StackedDB:
        """Create a new StackedDB with current as parent.

        The forked DB shares history with the original but can diverge.
        """
        new_stack = ImmutableStack(
            id=str(uuid4()),
            parent=self.current,
            created_at=datetime.now(),
            edges=[],
            deletions=set()
        )
        return StackedDB(base=new_stack)

    def rollback(self) -> StackedDB:
        """Return a new StackedDB at the parent stack.

        If at root, returns a new empty StackedDB.
        """
        if self.current.parent is None:
            # At root - return empty
            return StackedDB()
        return StackedDB(base=self.current.parent)

    def depth(self) -> int:
        """Return the depth of the stack chain."""
        return self.current.depth()

    def compact(self) -> StackedDB:
        """Merge all stacks into a single stack.

        Creates a new StackedDB with one stack containing all non-deleted edges.
        """
        all_edges = self.current.get_all_edges()

        new_stack = ImmutableStack(
            id=str(uuid4()),
            parent=None,
            created_at=datetime.now(),
            edges=all_edges,
            deletions=set()
        )
        return StackedDB(base=new_stack)

    def query_at_stack(self, stack_id: str) -> list[tuple[str, str, str, str]]:
        """Query edges as they existed at a specific stack.

        Finds the stack with the given ID and returns all edges visible from it.
        """
        stack = self._find_stack_by_id(stack_id)
        if stack is None:
            return []

        edges = stack.get_all_edges()
        return [e.to_tuple() for e in edges]

    def query_at_time(self, target_time: datetime) -> list[tuple[str, str, str, str]]:
        """Query edges as they existed at approximately the given time.

        Finds the most recent stack created before or at target_time.
        """
        stack = self._find_stack_by_time(target_time)
        if stack is None:
            return []

        edges = stack.get_all_edges()
        return [e.to_tuple() for e in edges]

    def _find_stack_by_id(self, stack_id: str) -> Optional[ImmutableStack]:
        """Find a stack in the chain by ID."""
        current = self.current
        while current is not None:
            if current.id == stack_id:
                return current
            current = current.parent
        return None

    def _find_stack_by_time(self, target_time: datetime) -> Optional[ImmutableStack]:
        """Find the most recent stack created at or before target_time."""
        current = self.current
        result = None

        while current is not None:
            if current.created_at <= target_time:
                # This is a candidate - but keep looking for more recent
                if result is None or current.created_at > result.created_at:
                    result = current
            current = current.parent

        return result

    def save(self, path: str) -> None:
        """Save the stack chain to a JSON file."""
        data = self.current.to_dict()
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> StackedDB:
        """Load a stack chain from a JSON file."""
        data = json.loads(Path(path).read_text())
        stack = ImmutableStack.from_dict(data)
        return cls(base=stack)
