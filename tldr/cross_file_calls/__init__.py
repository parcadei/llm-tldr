"""
Cross-file call graph resolution.

This package provides backward compatibility while the codebase transitions
from the monolithic cross_file_calls.py to the package structure.

For now, we re-export everything from the original module.

Note:
    Internal functions starting with `_` (e.g., `_extract_file_calls`) are 
    deprecated and exposed only for backward compatibility. They will be 
    removed in a future version. Please use the public API functions instead.
"""

from tldr.cross_file_calls_legacy import *

from tldr.cross_file_calls_legacy import (
    scan_project,
    build_project_call_graph,
    build_function_index,
    parse_imports,
    parse_ts_imports,
    parse_go_imports,
    parse_rust_imports,
    parse_java_imports,
    parse_c_imports,
    parse_cpp_imports,
    parse_ruby_imports,
    parse_kotlin_imports,
    parse_scala_imports,
    parse_php_imports,
    parse_swift_imports,
    parse_csharp_imports,
    parse_lua_imports,
    parse_luau_imports,
    parse_elixir_imports,
    ProjectCallGraph,
    TREE_SITTER_AVAILABLE,
    TREE_SITTER_GO_AVAILABLE,
    TREE_SITTER_RUST_AVAILABLE,
    TREE_SITTER_JAVA_AVAILABLE,
    TREE_SITTER_C_AVAILABLE,
    TREE_SITTER_RUBY_AVAILABLE,
    TREE_SITTER_PHP_AVAILABLE,
    TREE_SITTER_CPP_AVAILABLE,
    TREE_SITTER_KOTLIN_AVAILABLE,
    TREE_SITTER_SWIFT_AVAILABLE,
    TREE_SITTER_CSHARP_AVAILABLE,
    TREE_SITTER_SCALA_AVAILABLE,
    TREE_SITTER_LUA_AVAILABLE,
    TREE_SITTER_LUAU_AVAILABLE,
    TREE_SITTER_ELIXIR_AVAILABLE,
    _extract_file_calls,
    _extract_ts_file_calls,
    _extract_go_file_calls,
    _extract_rust_file_calls,
    _extract_java_file_calls,
    _extract_c_file_calls,
    _extract_php_file_calls,
)

__all__ = [
    # Main API functions
    "scan_project",
    "build_project_call_graph",
    "build_function_index",
    "parse_imports",
    # Language-specific import parsers
    "parse_ts_imports",
    "parse_go_imports",
    "parse_rust_imports",
    "parse_java_imports",
    "parse_c_imports",
    "parse_cpp_imports",
    "parse_ruby_imports",
    "parse_kotlin_imports",
    "parse_scala_imports",
    "parse_php_imports",
    "parse_swift_imports",
    "parse_csharp_imports",
    "parse_lua_imports",
    "parse_luau_imports",
    "parse_elixir_imports",
    # Core types
    "ProjectCallGraph",
    # Tree-sitter availability flags
    "TREE_SITTER_AVAILABLE",
    "TREE_SITTER_GO_AVAILABLE",
    "TREE_SITTER_RUST_AVAILABLE",
    "TREE_SITTER_JAVA_AVAILABLE",
    "TREE_SITTER_C_AVAILABLE",
    "TREE_SITTER_RUBY_AVAILABLE",
    "TREE_SITTER_PHP_AVAILABLE",
    "TREE_SITTER_CPP_AVAILABLE",
    "TREE_SITTER_KOTLIN_AVAILABLE",
    "TREE_SITTER_SWIFT_AVAILABLE",
    "TREE_SITTER_CSHARP_AVAILABLE",
    "TREE_SITTER_SCALA_AVAILABLE",
    "TREE_SITTER_LUA_AVAILABLE",
    "TREE_SITTER_LUAU_AVAILABLE",
    "TREE_SITTER_ELIXIR_AVAILABLE",
    # Internal extraction functions (for backward compatibility)
    "_extract_file_calls",
    "_extract_ts_file_calls",
    "_extract_go_file_calls",
    "_extract_rust_file_calls",
    "_extract_java_file_calls",
    "_extract_c_file_calls",
    "_extract_php_file_calls",
]
