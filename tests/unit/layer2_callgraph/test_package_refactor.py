"""
Tests for the refactored package structure.

Verifies that the package structure maintains backward compatibility.
"""

import pytest


class TestCrossFileCallsPackage:
    def test_imports_main_functions(self):
        from tldr.cross_file_calls import (
            scan_project,
            build_project_call_graph,
            parse_imports,
            ProjectCallGraph,
        )
        assert callable(scan_project)
        assert callable(build_project_call_graph)
        assert callable(parse_imports)
        assert ProjectCallGraph is not None

    def test_imports_language_parsers(self):
        from tldr.cross_file_calls import (
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
        )
        assert callable(parse_ts_imports)
        assert callable(parse_go_imports)
        assert callable(parse_rust_imports)
        assert callable(parse_java_imports)
        assert callable(parse_c_imports)
        assert callable(parse_cpp_imports)
        assert callable(parse_ruby_imports)
        assert callable(parse_kotlin_imports)
        assert callable(parse_scala_imports)
        assert callable(parse_php_imports)
        assert callable(parse_swift_imports)
        assert callable(parse_csharp_imports)
        assert callable(parse_lua_imports)
        assert callable(parse_luau_imports)
        assert callable(parse_elixir_imports)

    def test_imports_constants(self):
        from tldr.cross_file_calls import (
            TREE_SITTER_AVAILABLE,
            TREE_SITTER_GO_AVAILABLE,
            TREE_SITTER_RUST_AVAILABLE,
        )
        assert isinstance(TREE_SITTER_AVAILABLE, bool)
        assert isinstance(TREE_SITTER_GO_AVAILABLE, bool)
        assert isinstance(TREE_SITTER_RUST_AVAILABLE, bool)


class TestDFGExtractorPackage:
    def test_imports_main_types(self):
        from tldr.dfg_extractor import (
            VarRef,
            DataflowEdge,
            DFGInfo,
        )
        assert VarRef is not None
        assert DataflowEdge is not None
        assert DFGInfo is not None

    def test_imports_extractors(self):
        from tldr.dfg_extractor import (
            extract_python_dfg,
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
        assert callable(extract_python_dfg)
        assert callable(extract_typescript_dfg)
        assert callable(extract_go_dfg)
        assert callable(extract_rust_dfg)
        assert callable(extract_java_dfg)
        assert callable(extract_c_dfg)
        assert callable(extract_cpp_dfg)
        assert callable(extract_ruby_dfg)
        assert callable(extract_php_dfg)
        assert callable(extract_swift_dfg)
        assert callable(extract_csharp_dfg)
        assert callable(extract_kotlin_dfg)
        assert callable(extract_scala_dfg)
        assert callable(extract_lua_dfg)
        assert callable(extract_luau_dfg)
        assert callable(extract_elixir_dfg)

    def test_python_dfg_extraction(self):
        from tldr.dfg_extractor import extract_python_dfg
        
        code = '''
def example(x):
    y = x + 1
    return y
'''
        result = extract_python_dfg(code, 'example')
        assert result is not None
        assert hasattr(result, 'var_refs')
        assert hasattr(result, 'dataflow_edges')
        assert len(result.var_refs) > 0
        assert len(result.dataflow_edges) > 0


class TestHybridExtractorPackage:
    def test_imports_main_classes(self):
        from tldr.hybrid_extractor import (
            HybridExtractor,
            FileTooLargeError,
            ParseError,
        )
        assert HybridExtractor is not None
        assert FileTooLargeError is not None
        assert ParseError is not None

    def test_imports_functions(self):
        from tldr.hybrid_extractor import extract_directory
        assert callable(extract_directory)

    def test_hybrid_extractor_instantiation(self):
        from tldr.hybrid_extractor import HybridExtractor
        
        extractor = HybridExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'extract')


class TestBackwardCompatibility:
    def test_cross_file_calls_import_style(self):
        import tldr.cross_file_calls as cfc
        assert hasattr(cfc, 'scan_project')
        assert hasattr(cfc, 'build_project_call_graph')
        assert hasattr(cfc, 'ProjectCallGraph')

    def test_dfg_extractor_import_style(self):
        import tldr.dfg_extractor as dfg
        assert hasattr(dfg, 'DFGInfo')
        assert hasattr(dfg, 'extract_python_dfg')
        assert hasattr(dfg, 'VarRef')

    def test_hybrid_extractor_import_style(self):
        import tldr.hybrid_extractor as hybrid
        assert hasattr(hybrid, 'HybridExtractor')
        assert hasattr(hybrid, 'extract_directory')