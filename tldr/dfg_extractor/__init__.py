"""
Data Flow Graph (DFG) Extractor.

This package provides backward compatibility while the codebase transitions
from the monolithic dfg_extractor.py to the package structure.
"""

from tldr.dfg_extractor_legacy import *

from tldr.dfg_extractor_legacy import (
    VarRef,
    DataflowEdge,
    DFGInfo,
    PythonDefUseVisitor,
    PythonReachingDefsAnalyzer,
    CFGReachingDefsAnalyzer,
    TreeSitterDefUseVisitor,
    extract_python_dfg,
    extract_python_dfg_with_cfg,
    extract_typescript_dfg,
    extract_go_dfg,
    extract_rust_dfg,
    extract_java_dfg,
    extract_c_dfg,
    extract_cpp_dfg,
    extract_ruby_dfg,
    extract_php_dfg,
    extract_swift_dfg,
    extract_csharp_dfg,
    extract_kotlin_dfg,
    extract_scala_dfg,
    extract_lua_dfg,
    extract_luau_dfg,
    extract_elixir_dfg,
)

__all__ = [
    # Core types
    "VarRef",
    "DataflowEdge",
    "DFGInfo",
    # Visitors and analyzers
    "PythonDefUseVisitor",
    "PythonReachingDefsAnalyzer",
    "CFGReachingDefsAnalyzer",
    "TreeSitterDefUseVisitor",
    # Python extractors
    "extract_python_dfg",
    "extract_python_dfg_with_cfg",
    # Language-specific extractors
    "extract_typescript_dfg",
    "extract_go_dfg",
    "extract_rust_dfg",
    "extract_java_dfg",
    "extract_c_dfg",
    "extract_cpp_dfg",
    "extract_ruby_dfg",
    "extract_php_dfg",
    "extract_swift_dfg",
    "extract_csharp_dfg",
    "extract_kotlin_dfg",
    "extract_scala_dfg",
    "extract_lua_dfg",
    "extract_luau_dfg",
    "extract_elixir_dfg",
]
