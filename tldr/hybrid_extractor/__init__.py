"""
Hybrid Extractor.

This package provides backward compatibility while the codebase transitions
from the monolithic hybrid_extractor.py to the package structure.
"""

from tldr.hybrid_extractor_legacy import *

from tldr.hybrid_extractor_legacy import (
    FileTooLargeError,
    ParseError,
    HybridExtractor,
    extract_directory,
)

__all__ = [
    "FileTooLargeError",
    "ParseError",
    "HybridExtractor",
    "extract_directory",
]
