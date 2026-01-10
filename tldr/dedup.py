"""Content-Hash Deduplication (P5 #21).

Files with identical content share the same index entry.
Storage: {content_hash: edges} with lookup {file_path: content_hash}.

Benefits:
- Duplicate files indexed once (copy-pasted utils)
- Generated files (protobuf, graphql) share index
- 10-20% storage savings typical
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from tldr.patch import compute_file_hash, extract_edges_from_file, Edge


@dataclass
class ContentHashedIndex:
    """Content-hash based index for call graph edges.

    Instead of storing {file_path: edges}, stores {content_hash: edges}
    with a lookup table {file_path: content_hash}.

    This enables deduplication: files with identical content share indexes.
    """

    project_root: str

    # {content_hash: list of Edge tuples}
    _by_hash: Dict[str, List[tuple]] = field(default_factory=dict)

    # {absolute_file_path: content_hash}
    _path_to_hash: Dict[str, str] = field(default_factory=dict)

    # Stats tracking
    _extractions: int = field(default=0)
    _cache_hits: int = field(default=0)

    def get_or_create_edges(
        self,
        file_path: str,
        lang: str = "python"
    ) -> List[Edge]:
        """Get edges for file, creating if needed. Uses content-hash dedup.

        Args:
            file_path: Absolute path to source file
            lang: Language - "python", "typescript", "go", or "rust"

        Returns:
            List of Edge objects for this file
        """
        # Compute current content hash
        try:
            content_hash = compute_file_hash(file_path)
        except (FileNotFoundError, IOError):
            return []

        # Check if we've seen this file before with different content
        old_hash = self._path_to_hash.get(file_path)
        if old_hash and old_hash != content_hash:
            # Content changed - clean up old hash if no other files use it
            other_users = [p for p, h in self._path_to_hash.items()
                          if h == old_hash and p != file_path]
            if not other_users and old_hash in self._by_hash:
                del self._by_hash[old_hash]
        elif content_hash in self._by_hash:
            # Content-hash cache hit - reuse existing edges with remapped paths
            self._cache_hits += 1
            self._path_to_hash[file_path] = content_hash
            # Remap edge paths to current file (cached edges have original file's path)
            cached_tuples = self._by_hash[content_hash]
            remapped = [(file_path, func, file_path, target)
                       for (_, func, _, target) in cached_tuples]
            return self._edges_from_tuples(remapped)

        # Extract edges (new content or changed content)
        self._extractions += 1
        edges = extract_edges_from_file(file_path, lang=lang, project_root=self.project_root)

        # Store by content hash
        edge_tuples = [e.to_tuple() for e in edges]
        self._by_hash[content_hash] = edge_tuples
        self._path_to_hash[file_path] = content_hash

        return edges

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get the content hash for a file path.

        Args:
            file_path: Absolute path to file

        Returns:
            Content hash if file is indexed, None otherwise
        """
        # If not in lookup, compute it
        if file_path not in self._path_to_hash:
            try:
                self._path_to_hash[file_path] = compute_file_hash(file_path)
            except (FileNotFoundError, IOError):
                return None
        return self._path_to_hash.get(file_path)

    def stats(self) -> Dict[str, int]:
        """Get deduplication statistics.

        Returns:
            Dict with:
            - unique_hashes: Number of unique content hashes
            - total_files: Number of files tracked
            - dedup_savings: Number of extractions avoided
        """
        return {
            "unique_hashes": len(self._by_hash),
            "total_files": len(self._path_to_hash),
            "dedup_savings": self._cache_hits,
        }

    def save(self) -> None:
        """Persist index to disk."""
        cache_dir = Path(self.project_root) / ".tldr" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        index_file = cache_dir / "content_index.json"

        # Convert to relative paths for portability
        root = Path(self.project_root)
        rel_path_to_hash = {}
        for abs_path, hash_val in self._path_to_hash.items():
            try:
                rel_path = str(Path(abs_path).relative_to(root))
            except ValueError:
                rel_path = abs_path
            rel_path_to_hash[rel_path] = hash_val

        data = {
            "by_hash": self._by_hash,
            "path_to_hash": rel_path_to_hash,
            "stats": {
                "extractions": self._extractions,
                "cache_hits": self._cache_hits,
            }
        }

        index_file.write_text(json.dumps(data, indent=2))

    def load(self) -> bool:
        """Load index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        cache_dir = Path(self.project_root) / ".tldr" / "cache"
        index_file = cache_dir / "content_index.json"

        if not index_file.exists():
            return False

        try:
            data = json.loads(index_file.read_text())
        except (json.JSONDecodeError, IOError):
            return False

        self._by_hash = data.get("by_hash", {})

        # Convert relative paths back to absolute
        root = Path(self.project_root)
        rel_path_to_hash = data.get("path_to_hash", {})
        self._path_to_hash = {}
        for rel_path, hash_val in rel_path_to_hash.items():
            abs_path = str(root / rel_path)
            self._path_to_hash[abs_path] = hash_val

        stats = data.get("stats", {})
        self._extractions = stats.get("extractions", 0)
        self._cache_hits = stats.get("cache_hits", 0)

        return True

    def _edges_from_tuples(self, tuples: List[tuple]) -> List[Edge]:
        """Convert edge tuples back to Edge objects."""
        return [
            Edge(from_file=t[0], from_func=t[1], to_file=t[2], to_func=t[3])
            for t in tuples
        ]
