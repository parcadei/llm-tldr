"""Tree-sitter Incremental Parsing (P5).

This module provides incremental parsing support for tree-sitter, enabling
10-100x speedup by only re-parsing changed portions of files.

Key components:
- EditRange: Describes a text edit in byte offsets and row/column points
- TreeCache: Stores parsed trees for reuse
- IncrementalParser: Main interface for incremental parsing
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tree-sitter availability checks
TREE_SITTER_AVAILABLE = False
TREE_SITTER_PYTHON_AVAILABLE = False
TREE_SITTER_GO_AVAILABLE = False
TREE_SITTER_RUST_AVAILABLE = False

try:
    from tree_sitter import Language, Parser, Tree
    import tree_sitter_typescript
    import tree_sitter_javascript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    pass

try:
    import tree_sitter_python
    TREE_SITTER_PYTHON_AVAILABLE = True
except ImportError:
    pass

try:
    import tree_sitter_go
    TREE_SITTER_GO_AVAILABLE = True
except ImportError:
    pass

try:
    import tree_sitter_rust
    TREE_SITTER_RUST_AVAILABLE = True
except ImportError:
    pass


@dataclass
class EditRange:
    """Describes a text edit for incremental parsing.

    Tree-sitter's edit() method requires both byte offsets and row/column points.
    """

    start_byte: int
    old_end_byte: int
    new_end_byte: int
    start_point: tuple[int, int]  # (row, column)
    old_end_point: tuple[int, int]
    new_end_point: tuple[int, int]


def _byte_offset_to_point(content: bytes, byte_offset: int) -> tuple[int, int]:
    """Convert byte offset to (row, column) point.

    Args:
        content: The source content as bytes
        byte_offset: Byte offset into content

    Returns:
        (row, column) tuple (0-indexed)
    """
    # Ensure we don't go past the end
    byte_offset = min(byte_offset, len(content))

    # Get content up to offset
    prefix = content[:byte_offset]

    # Count newlines for row
    row = prefix.count(b"\n")

    # Find last newline position for column
    last_newline = prefix.rfind(b"\n")
    if last_newline == -1:
        column = byte_offset
    else:
        column = byte_offset - last_newline - 1

    return (row, column)


def calculate_edit_range(old_content: bytes, new_content: bytes) -> Optional[EditRange]:
    """Calculate the edit range between old and new content.

    Uses a simple diff algorithm to find the changed region.

    Args:
        old_content: Original file content
        new_content: New file content

    Returns:
        EditRange describing the change, or None if content is identical
    """
    if old_content == new_content:
        return None

    # Find first differing byte (from start)
    start_byte = 0
    min_len = min(len(old_content), len(new_content))
    while start_byte < min_len and old_content[start_byte] == new_content[start_byte]:
        start_byte += 1

    # Find first differing byte (from end)
    old_end_offset = 0
    new_end_offset = 0
    while (
        old_end_offset < len(old_content) - start_byte
        and new_end_offset < len(new_content) - start_byte
        and old_content[-(old_end_offset + 1)] == new_content[-(new_end_offset + 1)]
    ):
        old_end_offset += 1
        new_end_offset += 1

    old_end_byte = len(old_content) - old_end_offset
    new_end_byte = len(new_content) - new_end_offset

    # Ensure end >= start
    old_end_byte = max(old_end_byte, start_byte)
    new_end_byte = max(new_end_byte, start_byte)

    # Calculate points
    start_point = _byte_offset_to_point(old_content, start_byte)
    old_end_point = _byte_offset_to_point(old_content, old_end_byte)
    new_end_point = _byte_offset_to_point(new_content, new_end_byte)

    return EditRange(
        start_byte=start_byte,
        old_end_byte=old_end_byte,
        new_end_byte=new_end_byte,
        start_point=start_point,
        old_end_point=old_end_point,
        new_end_point=new_end_point,
    )


@dataclass
class CacheEntry:
    """Entry in the tree cache."""

    source: bytes
    source_hash: str
    language: str
    # Note: Tree objects can't be directly pickled, so we store source
    # and re-parse on cache load


class TreeCache:
    """Cache for parsed syntax trees.

    Stores source content and metadata for each file. Trees are re-parsed
    on cache hit since tree-sitter Tree objects aren't directly serializable.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the tree cache.

        Args:
            cache_dir: Directory to store cache files. If None, uses in-memory only.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: dict[str, tuple[Any, bytes]] = {}  # path -> (tree, source)

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._index_path = self._cache_dir / "index.json"
            self._load_index()
        else:
            self._index: dict[str, CacheEntry] = {}

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_path and self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    data = json.load(f)
                self._index = {
                    k: CacheEntry(**v) for k, v in data.items()
                }
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        if self._cache_dir and self._index_path:
            data = {
                k: {
                    "source": v.source.decode("utf-8", errors="replace"),
                    "source_hash": v.source_hash,
                    "language": v.language,
                }
                for k, v in self._index.items()
            }
            with open(self._index_path, "w") as f:
                json.dump(data, f)

    def _get_cache_path(self, file_path: str) -> Path:
        """Get the cache file path for a given source file."""
        # Use hash of path for cache filename
        path_hash = hashlib.md5(file_path.encode()).hexdigest()
        return self._cache_dir / f"{path_hash}.cache"

    def store(self, file_path: str, tree: Any, source: bytes) -> None:
        """Store a parsed tree in the cache.

        Args:
            file_path: Path to the source file
            tree: Parsed tree-sitter Tree
            source: Source content as bytes
        """
        # Store in memory
        self._memory_cache[file_path] = (tree, source)

        # Store on disk (source only, trees can't be pickled)
        if self._cache_dir:
            source_hash = hashlib.sha1(source).hexdigest()
            # Detect language from tree or infer from extension
            language = self._detect_language(file_path)

            self._index[file_path] = CacheEntry(
                source=source,
                source_hash=source_hash,
                language=language,
            )

            # Save source to disk
            cache_path = self._get_cache_path(file_path)
            with open(cache_path, "wb") as f:
                f.write(source)

            self._save_index()

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        suffix = Path(file_path).suffix.lower()
        lang_map = {
            ".ts": "typescript",
            ".tsx": "tsx",
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".py": "python",
            ".go": "go",
            ".rs": "rust",
        }
        return lang_map.get(suffix, "unknown")

    def get(self, file_path: str) -> Optional[tuple[Any, bytes]]:
        """Retrieve a cached tree and source.

        Args:
            file_path: Path to the source file

        Returns:
            (tree, source) tuple if cached, None otherwise
        """
        # Check memory cache first
        if file_path in self._memory_cache:
            return self._memory_cache[file_path]

        # Check disk cache
        if file_path in self._index and self._cache_dir:
            entry = self._index[file_path]
            cache_path = self._get_cache_path(file_path)

            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        source = f.read()

                    # Re-parse tree from source
                    parser = _get_parser(entry.language)
                    if parser:
                        tree = parser.parse(source)
                        self._memory_cache[file_path] = (tree, source)
                        return (tree, source)
                except Exception as e:
                    logger.warning(f"Failed to load cached tree for {file_path}: {e}")

        return None

    def invalidate(self, file_path: str) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to invalidate
        """
        self._memory_cache.pop(file_path, None)

        if file_path in self._index:
            del self._index[file_path]

            if self._cache_dir:
                cache_path = self._get_cache_path(file_path)
                if cache_path.exists():
                    cache_path.unlink()
                self._save_index()

    def clear(self) -> None:
        """Clear all cached trees."""
        self._memory_cache.clear()
        self._index.clear()

        if self._cache_dir:
            # Remove all cache files
            for cache_file in self._cache_dir.glob("*.cache"):
                cache_file.unlink()
            if self._index_path.exists():
                self._index_path.unlink()


def _get_parser(language: str) -> Optional[Any]:
    """Get a tree-sitter parser for the given language.

    Args:
        language: Language identifier (typescript, javascript, python, go, rust)

    Returns:
        Configured Parser or None if language not available
    """
    if not TREE_SITTER_AVAILABLE:
        return None

    parser = Parser()

    if language in ("typescript", "tsx"):
        if language == "tsx":
            parser.language = Language(tree_sitter_typescript.language_tsx())
        else:
            parser.language = Language(tree_sitter_typescript.language_typescript())
    elif language == "javascript":
        parser.language = Language(tree_sitter_javascript.language())
    elif language == "python":
        if TREE_SITTER_PYTHON_AVAILABLE:
            parser.language = Language(tree_sitter_python.language())
        else:
            return None
    elif language == "go":
        if TREE_SITTER_GO_AVAILABLE:
            parser.language = Language(tree_sitter_go.language())
        else:
            return None
    elif language == "rust":
        if TREE_SITTER_RUST_AVAILABLE:
            parser.language = Language(tree_sitter_rust.language())
        else:
            return None
    else:
        return None

    return parser


class IncrementalParser:
    """Main interface for incremental parsing.

    Manages tree cache and provides incremental parsing when possible.
    """

    SUPPORTED_LANGUAGES = {"typescript", "tsx", "javascript", "python", "go", "rust"}

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the incremental parser.

        Args:
            cache_dir: Directory for tree cache. If None, uses in-memory only.
        """
        self._cache = TreeCache(cache_dir=cache_dir)
        self._parsers: dict[str, Any] = {}

        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "full_parses": 0,
            "incremental_parses": 0,
        }

    def _get_parser(self, language: str) -> Any:
        """Get or create a parser for the language."""
        if language not in self._parsers:
            parser = _get_parser(language)
            if parser is None:
                raise ValueError(f"Unsupported language: {language}")
            self._parsers[language] = parser
        return self._parsers[language]

    def parse(self, file_path: str, language: str) -> Any:
        """Parse a file, using incremental parsing if possible.

        Args:
            file_path: Path to file to parse
            language: Language of the file

        Returns:
            Parsed tree-sitter Tree

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If language is not supported
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read current content
        with open(path, "rb") as f:
            new_content = f.read()

        # Check cache
        cached = self._cache.get(file_path)

        if cached is not None:
            old_tree, old_content = cached

            # Check if content unchanged (cache hit)
            if old_content == new_content:
                self._stats["cache_hits"] += 1
                return old_tree

            # Content changed - try incremental parse
            self._stats["cache_misses"] += 1

            edit_range = calculate_edit_range(old_content, new_content)

            if edit_range is not None:
                # Apply edit to old tree
                old_tree.edit(
                    start_byte=edit_range.start_byte,
                    old_end_byte=edit_range.old_end_byte,
                    new_end_byte=edit_range.new_end_byte,
                    start_point=edit_range.start_point,
                    old_end_point=edit_range.old_end_point,
                    new_end_point=edit_range.new_end_point,
                )

                # Parse with old tree for incremental parsing
                parser = self._get_parser(language)
                new_tree = parser.parse(new_content, old_tree)
                self._stats["incremental_parses"] += 1

                # Update cache
                self._cache.store(file_path, new_tree, new_content)

                return new_tree

        # No cache or can't do incremental - full parse
        self._stats["cache_misses"] += 1
        self._stats["full_parses"] += 1

        parser = self._get_parser(language)
        tree = parser.parse(new_content)

        # Store in cache
        self._cache.store(file_path, tree, new_content)

        return tree

    def get_stats(self) -> dict[str, int]:
        """Get parsing statistics.

        Returns:
            Dictionary with cache_hits, cache_misses, full_parses, incremental_parses
        """
        return self._stats.copy()

    def clear_cache(self) -> None:
        """Clear the tree cache."""
        self._cache.clear()

    def invalidate(self, file_path: str) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to invalidate
        """
        self._cache.invalidate(file_path)


def parse_incremental(
    file_path: str,
    language: str,
    old_tree: Optional[Any] = None,
    edit_range: Optional[EditRange] = None,
) -> Any:
    """Convenience function for incremental parsing.

    Args:
        file_path: Path to file to parse
        language: Language identifier
        old_tree: Previous tree (optional, for incremental parsing)
        edit_range: Edit range (required if old_tree provided)

    Returns:
        Parsed tree-sitter Tree
    """
    parser = _get_parser(language)
    if parser is None:
        raise ValueError(f"Unsupported language: {language}")

    with open(file_path, "rb") as f:
        content = f.read()

    if old_tree is not None and edit_range is not None:
        # Apply edit to old tree
        old_tree.edit(
            start_byte=edit_range.start_byte,
            old_end_byte=edit_range.old_end_byte,
            new_end_byte=edit_range.new_end_byte,
            start_point=edit_range.start_point,
            old_end_point=edit_range.old_end_point,
            new_end_point=edit_range.new_end_point,
        )
        return parser.parse(content, old_tree)

    return parser.parse(content)
