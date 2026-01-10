"""Durability Partitioning for Call Graph Indexes (P5).

Classifies files by how often they change:
- DURABLE: Never changes (node_modules, .venv, vendor, site-packages)
- VOLATILE: Changes frequently (your source code)

Benefit: Never re-index durable portions. Load durable on startup,
volatile on demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tldr.cross_file_calls import ProjectCallGraph


# Patterns that indicate a file is durable (rarely/never changes)
DURABLE_PATTERNS = [
    "node_modules/",
    ".venv/",
    "venv/",
    "vendor/",
    "__pycache__/",
    "site-packages/",
    ".tox/",
    "dist-packages/",
]


def is_durable(file_path: str) -> bool:
    """Classify a file path as durable or volatile.

    Durable files are dependencies that rarely change:
    - node_modules/ (npm packages)
    - .venv/, venv/ (Python virtual environments)
    - site-packages/ (installed Python packages)
    - vendor/ (vendored dependencies)
    - __pycache__/ (compiled Python bytecode)

    Args:
        file_path: Path to classify (can be relative or absolute)

    Returns:
        True if durable (rarely changes), False if volatile (user code)
    """
    # Normalize path separators
    normalized = file_path.replace("\\", "/")

    for pattern in DURABLE_PATTERNS:
        if pattern in normalized:
            return True

    return False


@dataclass
class DurablePartition:
    """A partition for durable (rarely changing) call graph edges.

    Stores edges from a specific package/dependency. The package_key
    identifies the package (e.g., "lodash@4.17.21", "numpy").
    """

    package_key: str = ""
    _edges: Set[Tuple[str, str, str, str]] = field(default_factory=set)
    _edges_by_file: Dict[str, Set[Tuple[str, str, str, str]]] = field(
        default_factory=dict
    )

    def add_edge(
        self, src_file: str, src_func: str, dst_file: str, dst_func: str
    ) -> None:
        """Add a call edge to this partition."""
        edge = (src_file, src_func, dst_file, dst_func)
        self._edges.add(edge)

        # Index by source file for fast lookup
        if src_file not in self._edges_by_file:
            self._edges_by_file[src_file] = set()
        self._edges_by_file[src_file].add(edge)

    @property
    def edges(self) -> Set[Tuple[str, str, str, str]]:
        """Return all edges in this partition."""
        return self._edges

    def get_edges_for_file(self, file_path: str) -> List[Tuple[str, str, str, str]]:
        """Return all edges originating from a specific file."""
        return list(self._edges_by_file.get(file_path, set()))

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "package_key": self.package_key,
            "edges": list(self._edges),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DurablePartition":
        """Deserialize from dictionary."""
        partition = cls(package_key=data.get("package_key", ""))
        for edge in data.get("edges", []):
            partition.add_edge(*edge)
        return partition


@dataclass
class VolatilePartition:
    """A partition for volatile (frequently changing) call graph edges.

    Wraps a ProjectCallGraph for user source code that changes often.
    """

    graph: ProjectCallGraph = field(default_factory=ProjectCallGraph)

    @property
    def edges(self) -> Set[Tuple[str, str, str, str]]:
        """Return all edges in this partition."""
        return self.graph.edges

    def add_edge(
        self, src_file: str, src_func: str, dst_file: str, dst_func: str
    ) -> None:
        """Add a call edge to this partition."""
        self.graph.add_edge(src_file, src_func, dst_file, dst_func)

    def remove_edges_from_file(self, file_path: str) -> None:
        """Remove all edges originating from a specific file.

        This is used during incremental updates when a file is re-indexed.
        """
        # Get current edges and filter out those from the target file
        current_edges = set(self.graph.edges)
        new_graph = ProjectCallGraph()

        for edge in current_edges:
            if edge[0] != file_path:
                new_graph.add_edge(*edge)

        self.graph = new_graph

    def get_edges_for_file(self, file_path: str) -> List[Tuple[str, str, str, str]]:
        """Return all edges originating from a specific file."""
        return [e for e in self.edges if e[0] == file_path]

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "edges": list(self.edges),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VolatilePartition":
        """Deserialize from dictionary."""
        partition = cls()
        for edge in data.get("edges", []):
            partition.add_edge(*edge)
        return partition


@dataclass
class PartitionedIndex:
    """A call graph index partitioned by durability.

    Maintains separate partitions for durable (dependencies) and
    volatile (user code) files. This enables:

    1. Never re-indexing durable files (they don't change)
    2. Fast startup by loading only durable (cached) indexes
    3. On-demand loading of volatile indexes
    4. Merged queries across both partitions
    """

    durable: Dict[str, DurablePartition] = field(default_factory=dict)
    volatile: VolatilePartition = field(default_factory=VolatilePartition)

    def add_edge(
        self, src_file: str, src_func: str, dst_file: str, dst_func: str
    ) -> None:
        """Add an edge, routing to appropriate partition based on durability."""
        if is_durable(src_file):
            package_key = self._extract_package(src_file)
            if package_key not in self.durable:
                self.durable[package_key] = DurablePartition(package_key=package_key)
            self.durable[package_key].add_edge(src_file, src_func, dst_file, dst_func)
        else:
            self.volatile.add_edge(src_file, src_func, dst_file, dst_func)

    def _extract_package(self, file_path: str) -> str:
        """Extract package key from a durable file path.

        Examples:
            node_modules/lodash/chunk.js -> lodash
            node_modules/@types/react/index.d.ts -> @types/react
            site-packages/numpy/core.py -> numpy
            .venv/lib/python3.12/site-packages/requests/api.py -> requests
            vendor/github.com/pkg/errors/errors.go -> github.com/pkg/errors
        """
        normalized = file_path.replace("\\", "/")

        # Handle node_modules (including scoped packages)
        if "node_modules/" in normalized:
            # Find the part after node_modules/
            parts = normalized.split("node_modules/")
            if len(parts) > 1:
                remainder = parts[-1]
                path_parts = remainder.split("/")

                # Scoped package: @scope/package
                if path_parts[0].startswith("@") and len(path_parts) > 1:
                    return f"{path_parts[0]}/{path_parts[1]}"
                else:
                    return path_parts[0]

        # Handle site-packages
        if "site-packages/" in normalized:
            parts = normalized.split("site-packages/")
            if len(parts) > 1:
                remainder = parts[-1]
                return remainder.split("/")[0]

        # Handle vendor (Go style: vendor/github.com/pkg/errors)
        if "vendor/" in normalized:
            parts = normalized.split("vendor/")
            if len(parts) > 1:
                remainder = parts[-1]
                # For Go, take first 3 parts (github.com/user/repo)
                path_parts = remainder.split("/")
                if len(path_parts) >= 3 and "." in path_parts[0]:
                    return "/".join(path_parts[:3])
                else:
                    return path_parts[0]

        # Fallback: use first directory component
        return normalized.split("/")[0]

    def get_all_edges(self) -> List[Tuple[str, str, str, str]]:
        """Get all edges from both partitions (merged)."""
        all_edges = list(self.volatile.edges)
        for partition in self.durable.values():
            all_edges.extend(partition.edges)
        return all_edges

    def get_all_durable_edges(self) -> List[Tuple[str, str, str, str]]:
        """Get all edges from durable partitions only."""
        all_edges = []
        for partition in self.durable.values():
            all_edges.extend(partition.edges)
        return all_edges

    def get_edges_for_file(self, file_path: str) -> List[Tuple[str, str, str, str]]:
        """Get edges from the appropriate partition for a file."""
        if is_durable(file_path):
            package_key = self._extract_package(file_path)
            if package_key in self.durable:
                return self.durable[package_key].get_edges_for_file(file_path)
            return []
        else:
            return self.volatile.get_edges_for_file(file_path)

    def filter_reindexable(self, dirty_files: List[str]) -> List[str]:
        """Filter dirty files to only those that should be re-indexed.

        Durable files are never re-indexed; only volatile files are.
        """
        return [f for f in dirty_files if not is_durable(f)]

    def save_durable(self, path: str) -> None:
        """Save all durable partitions to a directory.

        Each package gets its own JSON file for efficient loading.
        """
        durable_dir = Path(path)
        durable_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest listing all packages
        manifest = {"packages": list(self.durable.keys())}
        manifest_path = durable_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Save each package
        for package_key, partition in self.durable.items():
            # Sanitize package key for filename
            safe_key = package_key.replace("/", "__").replace("@", "_at_")
            pkg_path = durable_dir / f"{safe_key}.json"
            with open(pkg_path, "w") as f:
                json.dump(partition.to_dict(), f)

    def load_durable(self, path: str) -> None:
        """Load all durable partitions from a directory."""
        durable_dir = Path(path)
        if not durable_dir.exists():
            return

        manifest_path = durable_dir / "manifest.json"
        if not manifest_path.exists():
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        for package_key in manifest.get("packages", []):
            safe_key = package_key.replace("/", "__").replace("@", "_at_")
            pkg_path = durable_dir / f"{safe_key}.json"
            if pkg_path.exists():
                with open(pkg_path) as f:
                    data = json.load(f)
                self.durable[package_key] = DurablePartition.from_dict(data)

    def save_volatile(self, path: str) -> None:
        """Save volatile partition to a JSON file."""
        volatile_path = Path(path)
        volatile_path.parent.mkdir(parents=True, exist_ok=True)

        with open(volatile_path, "w") as f:
            json.dump(self.volatile.to_dict(), f)

    def load_volatile(self, path: str) -> None:
        """Load volatile partition from a JSON file."""
        volatile_path = Path(path)
        if not volatile_path.exists():
            return

        with open(volatile_path) as f:
            data = json.load(f)
        self.volatile = VolatilePartition.from_dict(data)
