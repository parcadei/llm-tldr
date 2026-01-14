"""
Tests for TypeScript semantic indexing with CFG/DFG summaries.

Verifies that semantic embeddings include all 5 layers for TypeScript,
not just Python.

Run with:
    pytest tests/test_typescript_semantic.py -v
"""

import pytest
import json
from pathlib import Path


# Sample TypeScript code with clear control flow and data flow
TYPESCRIPT_WITH_BRANCHES = """
export function classify(x: number): string {
    if (x > 10) {
        return "big";
    } else if (x > 5) {
        return "medium";
    } else {
        return "small";
    }
}

export function processData(input: string): string {
    const trimmed = input.trim();
    const upper = trimmed.toUpperCase();
    const result = upper + "!";
    return result;
}
"""


@pytest.fixture
def temp_ts_project(tmp_path):
    """Create a temporary TypeScript project."""
    filepath = tmp_path / "functions.ts"
    filepath.write_text(TYPESCRIPT_WITH_BRANCHES)
    return str(tmp_path)


class TestSemanticCFGSummary:
    """Test that CFG summaries are populated for TypeScript."""

    def test_cfg_summary_populated(self, temp_ts_project):
        """_get_cfg_summary should return complexity and blocks for TypeScript."""
        from tldr.semantic import _get_cfg_summary

        file_path = Path(temp_ts_project) / "functions.ts"
        summary = _get_cfg_summary(file_path, "classify", "typescript")

        assert summary != "", f"CFG summary should not be empty, got: '{summary}'"
        assert "complexity:" in summary, f"Should include complexity, got: {summary}"
        assert "blocks:" in summary, f"Should include blocks, got: {summary}"

        # classify has 2 if branches, so complexity should be >= 3
        # Parse the complexity value
        parts = summary.split(",")
        complexity_part = next((p for p in parts if "complexity:" in p), None)
        assert complexity_part is not None, f"No complexity in summary: {summary}"
        complexity = int(complexity_part.split(":")[1].strip())
        assert complexity >= 3, f"classify() should have complexity >= 3, got {complexity}"

    def test_cfg_summary_simple_function(self, temp_ts_project):
        """CFG summary for simple function should have low complexity."""
        from tldr.semantic import _get_cfg_summary

        file_path = Path(temp_ts_project) / "functions.ts"
        summary = _get_cfg_summary(file_path, "processData", "typescript")

        assert summary != "", "CFG summary should not be empty"
        assert "complexity:" in summary

    def test_cfg_summary_javascript(self, temp_ts_project):
        """CFG summary should also work for JavaScript."""
        from tldr.semantic import _get_cfg_summary

        # Create a JS file
        js_file = Path(temp_ts_project) / "functions.js"
        js_file.write_text("""
function greet(name) {
    if (name) {
        return "Hello, " + name;
    }
    return "Hello, stranger";
}
""")
        summary = _get_cfg_summary(js_file, "greet", "javascript")
        assert summary != "", "CFG summary should work for JavaScript"
        assert "complexity:" in summary


class TestSemanticDFGSummary:
    """Test that DFG summaries are populated for TypeScript."""

    def test_dfg_summary_populated(self, temp_ts_project):
        """_get_dfg_summary should return vars and def-use chains for TypeScript."""
        from tldr.semantic import _get_dfg_summary

        file_path = Path(temp_ts_project) / "functions.ts"
        summary = _get_dfg_summary(file_path, "processData", "typescript")

        assert summary != "", f"DFG summary should not be empty, got: '{summary}'"
        assert "vars:" in summary, f"Should include vars count, got: {summary}"
        assert "def-use chains:" in summary, f"Should include def-use chains, got: {summary}"

        # processData has variables: input, trimmed, upper, result
        parts = summary.split(",")
        vars_part = next((p for p in parts if "vars:" in p), None)
        assert vars_part is not None, f"No vars in summary: {summary}"
        var_count = int(vars_part.split(":")[1].strip())
        assert var_count >= 3, f"processData() should have >= 3 vars, got {var_count}"

    def test_dfg_summary_javascript(self, temp_ts_project):
        """DFG summary should also work for JavaScript."""
        from tldr.semantic import _get_dfg_summary

        js_file = Path(temp_ts_project) / "functions.js"
        js_file.write_text("""
function transform(data) {
    const parsed = JSON.parse(data);
    const filtered = parsed.filter(x => x.active);
    return filtered;
}
""")
        summary = _get_dfg_summary(js_file, "transform", "javascript")
        assert summary != "", "DFG summary should work for JavaScript"
        assert "vars:" in summary


class TestSemanticIndexIntegration:
    """Test that semantic index includes CFG/DFG for TypeScript functions."""

    def test_semantic_index_has_cfg_dfg(self, temp_ts_project):
        """Semantic index metadata should include cfg_summary and dfg_summary."""
        from tldr.semantic import build_semantic_index

        # Build the semantic index
        build_semantic_index(temp_ts_project, lang="typescript", show_progress=False)

        # Check the metadata
        metadata_path = Path(temp_ts_project) / ".tldr" / "cache" / "semantic" / "metadata.json"
        assert metadata_path.exists(), "Semantic index should create metadata.json"

        with open(metadata_path) as f:
            metadata = json.load(f)

        units = metadata.get("units", [])
        assert len(units) > 0, "Should have indexed some functions"

        # Find the classify function
        classify_unit = None
        process_unit = None
        for unit in units:
            if unit.get("name") == "classify":
                classify_unit = unit
            if unit.get("name") == "processData":
                process_unit = unit

        # Check classify has CFG summary (it has branches)
        assert classify_unit is not None, "classify function should be indexed"
        cfg = classify_unit.get("cfg_summary", "")
        assert cfg != "", f"classify should have cfg_summary, got: {classify_unit}"
        assert "complexity:" in cfg, "cfg_summary should include complexity"

        # Check processData has DFG summary (it has data flow)
        assert process_unit is not None, "processData function should be indexed"
        dfg = process_unit.get("dfg_summary", "")
        assert dfg != "", f"processData should have dfg_summary, got: {process_unit}"
        assert "vars:" in dfg, "dfg_summary should include vars"


class TestSemanticLayerParity:
    """Test that TypeScript has parity with Python for semantic features."""

    def test_python_cfg_summary_works(self, tmp_path):
        """Sanity check: Python CFG summary should work."""
        from tldr.semantic import _get_cfg_summary

        py_file = tmp_path / "test.py"
        py_file.write_text("""
def example(x):
    if x > 10:
        return "big"
    return "small"
""")
        summary = _get_cfg_summary(py_file, "example", "python")

        assert summary != "", "Python CFG summary should work"
        assert "complexity:" in summary

    def test_typescript_matches_python_format(self, temp_ts_project, tmp_path):
        """TypeScript CFG/DFG summaries should use same format as Python."""
        from tldr.semantic import _get_cfg_summary

        # Get Python format
        py_file = tmp_path / "test.py"
        py_file.write_text("""
def example(x):
    if x > 10:
        return "big"
    return "small"
""")
        py_cfg = _get_cfg_summary(py_file, "example", "python")

        # Get TypeScript format
        ts_file = Path(temp_ts_project) / "functions.ts"
        ts_cfg = _get_cfg_summary(ts_file, "classify", "typescript")

        # Both should have same format: "complexity:N, blocks:M"
        assert py_cfg.startswith("complexity:"), f"Python format: {py_cfg}"
        assert ts_cfg.startswith("complexity:"), f"TypeScript format: {ts_cfg}"
        assert ", blocks:" in py_cfg
        assert ", blocks:" in ts_cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
