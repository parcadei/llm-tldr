"""
Tests for PHP semantic indexing with CFG/DFG summaries.

Verifies that semantic embeddings include all 5 layers for PHP,
not just Python and TypeScript.

Run with:
    pytest tests/test_php_semantic.py -v
"""

import pytest
import json
from pathlib import Path


# Sample PHP code with clear control flow and data flow
PHP_WITH_BRANCHES = """
<?php

function classify($x) {
    if ($x > 10) {
        return "big";
    } else if ($x > 5) {
        return "medium";
    } else {
        return "small";
    }
}

function processData($input) {
    $trimmed = trim($input);
    $upper = strtoupper($trimmed);
    $result = $upper . "!";
    return $result;
}

class UserService {
    public function getUser($id) {
        if ($id > 0) {
            return "User-{$id}";
        }
        return null;
    }
}
"""


@pytest.fixture
def temp_php_project(tmp_path):
    """Create a temporary PHP project."""
    filepath = tmp_path / "functions.php"
    filepath.write_text(PHP_WITH_BRANCHES)
    return str(tmp_path)


class TestSemanticCFGSummary:
    """Test that CFG summaries are populated for PHP."""

    def test_cfg_summary_populated(self, temp_php_project):
        """_get_cfg_summary should return complexity and blocks for PHP."""
        from tldr.semantic import _get_cfg_summary

        file_path = Path(temp_php_project) / "functions.php"
        summary = _get_cfg_summary(file_path, "classify", "php")

        assert summary != "", f"CFG summary should not be empty, got: '{summary}'"
        assert "complexity:" in summary, f"Should include complexity, got: {summary}"
        assert "blocks:" in summary, f"Should include blocks, got: {summary}"

        # classify has 2 if branches, so complexity should be >= 3
        parts = summary.split(",")
        complexity_part = next((p for p in parts if "complexity:" in p), None)
        assert complexity_part is not None, f"No complexity in summary: {summary}"
        complexity = int(complexity_part.split(":")[1].strip())
        assert complexity >= 3, f"classify() should have complexity >= 3, got {complexity}"

    def test_cfg_summary_simple_function(self, temp_php_project):
        """CFG summary for simple function should have low complexity."""
        from tldr.semantic import _get_cfg_summary

        file_path = Path(temp_php_project) / "functions.php"
        summary = _get_cfg_summary(file_path, "processData", "php")

        assert summary != "", "CFG summary should not be empty"
        assert "complexity:" in summary


class TestSemanticDFGSummary:
    """Test that DFG summaries are populated for PHP."""

    def test_dfg_summary_populated(self, temp_php_project):
        """_get_dfg_summary should return vars and def-use chains for PHP."""
        from tldr.semantic import _get_dfg_summary

        file_path = Path(temp_php_project) / "functions.php"
        summary = _get_dfg_summary(file_path, "processData", "php")

        assert summary != "", f"DFG summary should not be empty, got: '{summary}'"
        assert "vars:" in summary, f"Should include vars count, got: {summary}"
        assert "def-use chains:" in summary, f"Should include def-use chains, got: {summary}"

        # processData has variables: input, trimmed, upper, result
        parts = summary.split(",")
        vars_part = next((p for p in parts if "vars:" in p), None)
        assert vars_part is not None, f"No vars in summary: {summary}"
        var_count = int(vars_part.split(":")[1].strip())
        assert var_count >= 3, f"processData() should have >= 3 vars, got {var_count}"


class TestSemanticPHPExtraction:
    """Test that PHP function extraction works correctly."""

    def test_php_extract_units_for_extraction(self, temp_php_project):
        """_process_file_for_extraction should extract PHP functions with metadata."""
        from tldr.semantic import _process_file_for_extraction
        from tldr.cross_file_calls import build_project_call_graph

        file_path = Path(temp_php_project) / "functions.php"

        # Build call graph for PHP
        call_graph = build_project_call_graph(temp_php_project, language="php")

        # Build calls maps
        calls_map = {}
        called_by_map = {}
        for edge in call_graph.edges:
            src_file, src_func, dst_file, dst_func = edge
            if src_func not in calls_map:
                calls_map[src_func] = []
            calls_map[src_func].append(dst_func)
            if dst_func not in called_by_map:
                called_by_map[dst_func] = []
            called_by_map[dst_func].append(src_func)

        # Process file
        file_info = {
            "path": "functions.php",
            "functions": ["classify", "processData"],
            "classes": [
                {
                    "name": "UserService",
                    "methods": ["getUser"]
                }
            ]
        }

        units = _process_file_for_extraction(
            file_info, temp_php_project, "php", calls_map, called_by_map
        )

        # Should extract functions
        function_units = [u for u in units if u.unit_type == "function"]
        assert len(function_units) >= 2, f"Should extract at least 2 functions, got {len(function_units)}"

        # Check classify function
        classify = next((u for u in function_units if u.name == "classify"), None)
        assert classify is not None, "classify function should be extracted"

        # Check CFG and DFG summaries are populated
        assert classify.cfg_summary != "", f"classify should have cfg_summary, got: {classify.cfg_summary}"
        assert "complexity:" in classify.cfg_summary
        assert classify.dfg_summary != "", f"classify should have dfg_summary, got: {classify.dfg_summary}"
        assert "vars:" in classify.dfg_summary

        # Check method extraction
        method_units = [u for u in units if u.unit_type == "method"]
        assert len(method_units) >= 1, f"Should extract at least 1 method, got {len(method_units)}"

        getUser = next((u for u in method_units if u.name == "getUser"), None)
        assert getUser is not None, "getUser method should be extracted"
        # Check CFG summary for method
        assert getUser.cfg_summary != "", f"getUser should have cfg_summary"


class TestSemanticLayerParity:
    """Test that PHP has parity with Python for semantic features."""

    def test_php_cfg_summary_works(self, tmp_path):
        """PHP CFG summary should work like Python."""
        from tldr.semantic import _get_cfg_summary

        php_file = tmp_path / "test.php"
        php_file.write_text("""
<?php
function example($x) {
    if ($x > 10) {
        return "big";
    }
    return "small";
}
""")
        summary = _get_cfg_summary(php_file, "example", "php")

        assert summary != "", "PHP CFG summary should work"
        assert "complexity:" in summary

    def test_php_matches_python_format(self, temp_php_project, tmp_path):
        """PHP CFG/DFG summaries should use same format as Python."""
        from tldr.semantic import _get_cfg_summary, _get_dfg_summary

        # Get Python format
        py_file = tmp_path / "test.py"
        py_file.write_text("""
def example(x):
    if x > 10:
        return "big"
    return "small"
""")
        py_cfg = _get_cfg_summary(py_file, "example", "python")

        # Get PHP format
        php_file = Path(temp_php_project) / "functions.php"
        php_cfg = _get_cfg_summary(php_file, "classify", "php")

        # Both should have same format: "complexity:N, blocks:M"
        assert py_cfg.startswith("complexity:"), f"Python format: {py_cfg}"
        assert php_cfg.startswith("complexity:"), f"PHP format: {php_cfg}"
        assert ", blocks:" in py_cfg
        assert ", blocks:" in php_cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
