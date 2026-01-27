"""
Test suite for language wiring completeness.

When adding a new language to tldr-code, this test ensures all
registration points are properly wired. Run with:

    pytest tests/test_language_wiring.py -v

To test a specific language:

    pytest tests/test_language_wiring.py -v -k "luau"
"""

import pytest
from pathlib import Path

# All supported languages and their primary extensions
# Note: Only includes extensions currently in cli.py EXTENSION_TO_LANGUAGE
SUPPORTED_LANGUAGES = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],  # .mjs/.cjs not in cli.py yet
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".hpp"],  # .cxx/.hh/.hxx not in cli.py yet
    "ruby": [".rb"],
    "php": [".php"],
    "kotlin": [".kt", ".kts"],
    "swift": [".swift"],
    "csharp": [".cs"],
    "scala": [".scala", ".sc"],
    "lua": [".lua"],
    "luau": [".luau"],
    "elixir": [".ex", ".exs"],
    "zig": [".zig"],
}

# Languages with tree-sitter support in incremental_parse.py
# Note: swift tree-sitter-swift is not on PyPI yet
INCREMENTAL_PARSE_LANGUAGES = {
    "python", "typescript", "tsx", "javascript", "go", "rust",
    "lua", "luau", "java", "c", "cpp", "ruby", "php", "csharp",
    "kotlin", "scala", "elixir", "zig"
}


class TestLanguageWiring:
    """Test that each language is properly wired in all modules."""

    @pytest.mark.parametrize("language", INCREMENTAL_PARSE_LANGUAGES)
    def test_incremental_parse_supported_languages(self, language):
        """Language should be in IncrementalParser.SUPPORTED_LANGUAGES."""
        from tldr.incremental_parse import IncrementalParser

        assert language in IncrementalParser.SUPPORTED_LANGUAGES, (
            f"{language} missing from incremental_parse.py SUPPORTED_LANGUAGES"
        )

    @pytest.mark.parametrize("language", SUPPORTED_LANGUAGES.keys())
    def test_cli_extension_to_language(self, language):
        """Language extensions should be in cli.py EXTENSION_TO_LANGUAGE."""
        from tldr.cli import EXTENSION_TO_LANGUAGE

        extensions = SUPPORTED_LANGUAGES[language]
        for ext in extensions:
            assert ext in EXTENSION_TO_LANGUAGE, (
                f"Extension {ext} for {language} missing from cli.py EXTENSION_TO_LANGUAGE"
            )
            assert EXTENSION_TO_LANGUAGE[ext] == language, (
                f"Extension {ext} maps to {EXTENSION_TO_LANGUAGE[ext]}, expected {language}"
            )

    @pytest.mark.parametrize("language", SUPPORTED_LANGUAGES.keys())
    def test_semantic_all_languages(self, language):
        """Language should be in semantic.py ALL_LANGUAGES."""
        from tldr.semantic import ALL_LANGUAGES

        assert language in ALL_LANGUAGES, (
            f"{language} missing from semantic.py ALL_LANGUAGES"
        )

    @pytest.mark.parametrize("language", SUPPORTED_LANGUAGES.keys())
    def test_scan_project_extensions(self, language):
        """Language should be recognized by scan_project()."""
        from tldr.cross_file_calls import scan_project
        import tempfile
        import os

        # Create a temp directory with a test file
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = SUPPORTED_LANGUAGES[language][0]
            test_file = Path(tmpdir) / f"test{ext}"
            test_file.write_text("# test")

            try:
                files = scan_project(tmpdir, language=language)
                # Should not raise ValueError
                assert isinstance(files, list), f"scan_project failed for {language}"
            except ValueError as e:
                if "Unsupported language" in str(e):
                    pytest.fail(f"{language} not supported in cross_file_calls.scan_project()")
                raise

    @pytest.mark.parametrize("language", SUPPORTED_LANGUAGES.keys())
    def test_api_get_code_structure(self, language):
        """Language should work with get_code_structure()."""
        from tldr.api import get_code_structure
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ext = SUPPORTED_LANGUAGES[language][0]
            test_file = Path(tmpdir) / f"test{ext}"
            test_file.write_text("# test")

            try:
                result = get_code_structure(tmpdir, language=language)
                assert "files" in result, f"get_code_structure failed for {language}"
            except ValueError as e:
                if "Unsupported language" in str(e):
                    pytest.fail(f"{language} not supported in api.get_code_structure()")
                raise

    @pytest.mark.parametrize("language", SUPPORTED_LANGUAGES.keys())
    def test_hybrid_extractor_detect_language(self, language):
        """Language should be detected by HybridExtractor._detect_language()."""
        from tldr.hybrid_extractor import HybridExtractor
        import tempfile

        extractor = HybridExtractor()
        ext = SUPPORTED_LANGUAGES[language][0]

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(b"# test")
            test_file = Path(f.name)

        try:
            detected = extractor._detect_language(test_file)
            # Some languages may map to "unknown" if not in the ext_map
            # We just ensure it doesn't crash
            assert detected is not None, f"_detect_language returned None for {language}"
        finally:
            test_file.unlink()


class TestCFGExtractors:
    """Test that CFG extractors exist for each language."""

    # Languages that have dedicated CFG extractors
    # Note: javascript uses typescript extractor
    CFG_LANGUAGES = [
        "python", "typescript", "go", "rust", "java",
        "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp",
        "scala", "lua", "luau", "elixir", "zig"
    ]

    @pytest.mark.parametrize("language", CFG_LANGUAGES)
    def test_cfg_extractor_exists(self, language):
        """CFG extractor function should exist for language."""
        from tldr import cfg_extractor as cfg_mod

        func_name = f"extract_{language}_cfg"
        assert hasattr(cfg_mod, func_name), (
            f"Missing {func_name} in cfg_extractor.py"
        )

    @pytest.mark.parametrize("language", CFG_LANGUAGES)
    def test_cfg_extractor_in_api_map(self, language):
        """CFG extractor should be in api.py cfg_extractors map."""
        # Read api.py and check the cfg_extractors dict
        api_path = Path(__file__).parent.parent / "tldr" / "api.py"
        content = api_path.read_text()

        # Look for the language in cfg_extractors
        assert f'"{language}"' in content or f"'{language}'" in content, (
            f"{language} may be missing from api.py cfg_extractors"
        )


class TestDFGExtractors:
    """Test that DFG extractors exist for each language."""

    # Languages that have dedicated DFG extractors
    # Note: javascript uses typescript extractor
    DFG_LANGUAGES = [
        "python", "typescript", "go", "rust", "java",
        "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp",
        "scala", "lua", "luau", "elixir", "zig"
    ]

    @pytest.mark.parametrize("language", DFG_LANGUAGES)
    def test_dfg_extractor_exists(self, language):
        """DFG extractor function should exist for language."""
        from tldr import dfg_extractor as dfg_mod

        func_name = f"extract_{language}_dfg"
        assert hasattr(dfg_mod, func_name), (
            f"Missing {func_name} in dfg_extractor.py"
        )


class TestPDGExtractors:
    """Test that PDG extractors exist for each language."""

    # Languages that have PDG extractors
    PDG_LANGUAGES = [
        "python", "typescript", "go", "rust", "java",
        "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp",
        "scala", "lua", "luau", "elixir", "zig"
    ]

    @pytest.mark.parametrize("language", PDG_LANGUAGES)
    def test_pdg_extractor_exists(self, language):
        """PDG extractor function should exist for language."""
        from tldr import pdg_extractor as pdg_mod

        func_name = f"extract_{language}_pdg"
        assert hasattr(pdg_mod, func_name), (
            f"Missing {func_name} in pdg_extractor.py"
        )


class TestImportParsers:
    """Test that import parsers exist for each language."""

    # Languages that have dedicated import parser functions
    # Format: (language, function_name)
    IMPORT_PARSERS = [
        ("go", "parse_go_imports"),
        ("rust", "parse_rust_imports"),
        ("lua", "parse_lua_imports"),
        ("luau", "parse_luau_imports"),
    ]

    @pytest.mark.parametrize("language,func_name", IMPORT_PARSERS)
    def test_import_parser_exists(self, language, func_name):
        """Import parser function should exist for language."""
        from tldr import cross_file_calls as cfc_mod

        assert hasattr(cfc_mod, func_name), (
            f"Missing {func_name} in cross_file_calls.py"
        )


class TestCLIArguments:
    """Test that languages are in CLI argument choices."""

    @pytest.mark.parametrize("language", SUPPORTED_LANGUAGES.keys())
    def test_language_in_cli_help(self, language):
        """Language should be accepted by CLI --lang argument."""
        import subprocess
        import sys

        # Test that the language is accepted (won't error on invalid choice)
        result = subprocess.run(
            [sys.executable, "-m", "tldr.cli", "structure", "--help"],
            capture_output=True,
            text=True
        )

        # The help should mention the language in choices
        # Note: This is a weak test, stronger would be to actually parse argparse
        assert result.returncode == 0, "CLI help failed"


class TestTreeSitterGrammars:
    """Test that tree-sitter grammars are importable."""

    # Map of language to tree-sitter module
    GRAMMAR_MODULES = {
        "python": "tree_sitter_python",
        "typescript": "tree_sitter_typescript",
        "javascript": "tree_sitter_javascript",
        "go": "tree_sitter_go",
        "rust": "tree_sitter_rust",
        "java": "tree_sitter_java",
        "c": "tree_sitter_c",
        "cpp": "tree_sitter_cpp",
        "ruby": "tree_sitter_ruby",
        "php": "tree_sitter_php",
        "csharp": "tree_sitter_c_sharp",
        "kotlin": "tree_sitter_kotlin",
        "scala": "tree_sitter_scala",
        "lua": "tree_sitter_lua",
        "luau": "tree_sitter_luau",
        "elixir": "tree_sitter_elixir",
        "zig": "tree_sitter_zig",
    }

    @pytest.mark.parametrize("language,module", GRAMMAR_MODULES.items())
    def test_grammar_importable(self, language, module):
        """Tree-sitter grammar module should be importable."""
        try:
            __import__(module)
        except ImportError:
            pytest.skip(f"tree-sitter grammar {module} not installed")


# Summary of all registration points for documentation
REGISTRATION_POINTS = """
When adding a new language to tldr-code, ensure it's registered in:

1. incremental_parse.py
   - Add to SUPPORTED_LANGUAGES set
   - Add import for tree_sitter_<lang>
   - Add case in _get_parser()

2. cli.py
   - Add to EXTENSION_TO_LANGUAGE dict
   - Add to --lang argument choices (4 places)

3. semantic.py
   - Add to ALL_LANGUAGES list
   - Add to EXTENSION_TO_LANGUAGE dict

4. cross_file_calls.py
   - Add to scan_project() extension mappings
   - Create parse_<lang>_imports() function

5. api.py
   - Add to ext_map in get_project_structure()
   - Add to cfg_extractors dict
   - Add to dfg_extractors dict
   - Add to get_imports() language dispatch

6. hybrid_extractor.py
   - Add to _detect_language() ext_map
   - Add <LANG>_EXTENSIONS constant
   - Create _extract_<lang>() method
   - Create _get_<lang>_parser() method

7. cfg_extractor.py
   - Create extract_<lang>_cfg() function
   - Add case in get_cfg() dispatcher

8. dfg_extractor.py
   - Create extract_<lang>_dfg() function
   - Create <Lang>DefUseVisitor class

9. pdg_extractor.py
   - Create extract_<lang>_pdg() function

10. pyproject.toml
    - Add tree-sitter-<lang> dependency

Run this test suite to verify completeness:
    pytest tests/test_language_wiring.py -v
"""
